use std::io::{self, BufRead, Write};
use crate::{EsmConfig, EsmWeights, RunState, active_simd_backend, generate_sequence};

pub fn escape_json_str(s: &str) -> String {
    let bs = 92u8 as char;
    let quote = 34u8 as char;
    let mut out = String::with_capacity(s.len() + 32);
    for c in s.chars() {
        if c == quote || c == bs {
            out.push(bs);
        }
        out.push(c);
    }
    out
}

pub fn extract_json_str<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let mut pattern = String::with_capacity(key.len() + 2);
    pattern.push('"');
    pattern.push_str(key);
    pattern.push('"');

    let pos = json.find(&pattern)?;
    let after_key = &json[pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();

    if !after_colon.starts_with('"') {
        return None;
    }
    let content = &after_colon[1..];
    let end_quote = content.find('"')?;
    Some(&content[..end_quote])
}

pub fn extract_json_raw_value<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let mut pattern = String::with_capacity(key.len() + 2);
    pattern.push('"');
    pattern.push_str(key);
    pattern.push('"');

    let pos = json.find(&pattern)?;
    let after_key = &json[pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();

    if after_colon.starts_with('"') {
        let end_quote = after_colon[1..].find('"')?;
        return Some(&after_colon[..=end_quote + 1]);
    }

    let end = after_colon
        .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        .unwrap_or(after_colon.len());
    if end == 0 {
        None
    } else {
        Some(&after_colon[..end])
    }
}

pub fn wrap_jsonrpc_result(id_raw: &str, result_json: &str) -> String {
    let mut out = String::from(r#"{"jsonrpc":"2.0","id":"#);
    out.push_str(id_raw);
    out.push_str(r#","result":"#);
    out.push_str(result_json);
    out.push('}');
    out
}

pub fn wrap_jsonrpc_error(id_raw: &str, code: i32, message: &str) -> String {
    let mut out = String::from(r#"{"jsonrpc":"2.0","id":"#);
    out.push_str(id_raw);
    out.push_str(r#","error":{"code":"#);
    out.push_str(&code.to_string());
    out.push_str(r#","message":""#);
    out.push_str(&escape_json_str(message));
    out.push_str(r#""}}"#);
    out
}

pub fn run_stdio_server(model_path: &str) -> io::Result<()> {
    let config = EsmConfig::esm2_8m();
    let mut weights_opt: Option<EsmWeights> = None;
    let mut run_state = RunState::new(&config);

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line_res in stdin.lock().lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let id_raw = extract_json_raw_value(trimmed, "id").unwrap_or("null");
        let method = match extract_json_str(trimmed, "method") {
            Some(m) => m,
            None => {
                let err = wrap_jsonrpc_error(id_raw, -32700, "Parse error: missing method");
                writeln!(stdout, "{}", err)?;
                stdout.flush()?;
                continue;
            }
        };

        if method == "notifications/initialized" {
            continue;
        }

        let response = match method {
            "initialize" => {
                let mut res = String::from(
                    r#"{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"codon-infer","version":""#,
                );
                res.push_str(env!("CARGO_PKG_VERSION"));
                res.push_str(r#""}}"#);
                wrap_jsonrpc_result(id_raw, &res)
            }
            "ping" => wrap_jsonrpc_result(id_raw, "{}"),
            "tools/list" => {
                let tools_schema = r#"{"tools":[{"name":"generate_protein","description":"De novo protein sequence generation via Pure Rust ESM-2 engine","inputSchema":{"type":"object","properties":{"prompt":{"type":"string","description":"Seed amino acid sequence (default: M)"},"temperature":{"type":"number","description":"Sampling temperature (default: 0.7)"},"max_tokens":{"type":"integer","description":"Maximum new residues to generate"},"seed":{"type":"integer","description":"Optional RNG seed"}}}}]}"#;
                wrap_jsonrpc_result(id_raw, tools_schema)
            }
            "tools/call" => {
                if weights_opt.is_none() {
                    match EsmWeights::from_file(model_path, &config) {
                        Ok(w) => weights_opt = Some(w),
                        Err(e) => {
                            let err = wrap_jsonrpc_error(id_raw, -32603, &format!("Model load failed: {}", e));
                            writeln!(stdout, "{}", err)?;
                            stdout.flush()?;
                            continue;
                        }
                    }
                }
                let weights = weights_opt.as_ref().unwrap();

                let prompt = extract_json_str(trimmed, "prompt")
                    .or_else(|| extract_json_str(trimmed, "sequence"))
                    .unwrap_or("M");
                let temperature = extract_json_raw_value(trimmed, "temperature")
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.7);
                let max_tokens = extract_json_raw_value(trimmed, "max_tokens")
                    .and_then(|v| v.parse::<usize>().ok());
                let seed = extract_json_raw_value(trimmed, "seed")
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(42);

                let metrics = generate_sequence(
                    prompt,
                    temperature,
                    max_tokens,
                    seed,
                    &config,
                    weights,
                    &mut run_state,
                    None,
                );

                let mut inner_json = String::from(r#"{"sequence":""#);
                inner_json.push_str(&metrics.full_sequence);
                inner_json.push_str(r#"","generated":""#);
                inner_json.push_str(&metrics.generated_suffix);
                inner_json.push_str(r#"","tokens_generated":"#);
                inner_json.push_str(&metrics.tokens_generated.to_string());
                inner_json.push_str(r#","prefill_ms":"#);
                inner_json.push_str(&format!("{:.2}", metrics.prefill_ms));
                inner_json.push_str(r#","generation_ms":"#);
                inner_json.push_str(&format!("{:.2}", metrics.generation_ms));
                inner_json.push_str(r#","throughput":"#);
                inner_json.push_str(&format!("{:.2}", metrics.throughput_tps));
                inner_json.push_str(r#","simd_backend":""#);
                inner_json.push_str(active_simd_backend());
                inner_json.push_str(r#""}"#);

                let mut payload = String::from(r#"{"content":[{"type":"text","text":""#);
                payload.push_str(&escape_json_str(&inner_json));
                payload.push_str(r#""}],"isError":false}"#);

                wrap_jsonrpc_result(id_raw, &payload)
            }
            _ => wrap_jsonrpc_error(id_raw, -32601, "Method not found"),
        };

        writeln!(stdout, "{}", response)?;
        stdout.flush()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_json_parser_and_formatter() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"generate_protein","arguments":{"prompt":"MCMC","temperature":0.5,"max_tokens":25}}}"#;
        assert_eq!(extract_json_raw_value(req, "id"), Some("1"));
        assert_eq!(extract_json_str(req, "method"), Some("tools/call"));
        assert_eq!(extract_json_str(req, "prompt"), Some("MCMC"));

        let wrapped = wrap_jsonrpc_result("1", r#"{"ok":true}"#);
        assert!(wrapped.contains(r#"result":{"ok":true}"#));
    }
}

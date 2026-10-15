use std::io::{self, BufRead, Write};
use crate::{
    active_simd_backend, compute_pipeline_contacts, compute_pipeline_inpaint,
    contacts_to_json_string, dms_to_json_string, generate_sequence, inpaint_to_json_string,
    parse_fasta_file, parse_raw_sequence, run_pipeline_dms, EsmConfig, EsmWeights, FastaRecord,
    RunState, TerminalTheme,
};

/// Escapes special characters for JSON string values (including quotes, backslashes, and control characters).
pub fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

/// Extracts a string value from a JSON snippet given a key.
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

/// Extracts a raw token (number, boolean, null, or string) from a JSON snippet given a key.
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

/// Extracts integer parameter from JSON (handling both raw numbers and string-encoded numbers).
pub fn extract_json_int(json: &str, key: &str) -> Option<usize> {
    if let Some(v) = extract_json_str(json, key) {
        if let Ok(n) = v.parse::<usize>() {
            return Some(n);
        }
    }
    if let Some(v) = extract_json_raw_value(json, key) {
        let trimmed = v.trim().trim_matches('"');
        if let Ok(n) = trimmed.parse::<usize>() {
            return Some(n);
        }
    }
    None
}

/// Extracts floating point parameter from JSON (handling both raw numbers and string-encoded numbers).
pub fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    if let Some(v) = extract_json_str(json, key) {
        if let Ok(f) = v.parse::<f32>() {
            return Some(f);
        }
    }
    if let Some(v) = extract_json_raw_value(json, key) {
        let trimmed = v.trim().trim_matches('"');
        if let Ok(f) = trimmed.parse::<f32>() {
            return Some(f);
        }
    }
    None
}

/// Extracts tool name from a tools/call request.
pub fn extract_tool_name(json: &str) -> Option<&str> {
    if let Some(params_pos) = json.find("\"params\"") {
        extract_json_str(&json[params_pos..], "name")
    } else {
        extract_json_str(json, "name")
    }
}

/// Resolves an input protein record from JSON arguments (either via 'fasta_path' or 'sequence'/'prompt').
pub fn resolve_mcp_record(json: &str, default_id: &str) -> io::Result<FastaRecord> {
    if let Some(fasta_path) = extract_json_str(json, "fasta_path").or_else(|| extract_json_str(json, "fasta")) {
        let records = parse_fasta_file(fasta_path)?;
        if let Some(first) = records.into_iter().next() {
            return Ok(first);
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("FASTA file '{}' contains no sequences", fasta_path),
            ));
        }
    }
    if let Some(seq) = extract_json_str(json, "sequence").or_else(|| extract_json_str(json, "prompt")) {
        let rec = parse_raw_sequence(seq, default_id);
        if rec.sequence.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Provided sequence contains no valid amino acid characters",
            ));
        }
        return Ok(rec);
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "Missing required parameter: provide either 'sequence' or 'fasta_path'",
    ))
}

/// Wraps a JSON result into a JSON-RPC 2.0 response.
pub fn wrap_jsonrpc_result(id_raw: &str, result_json: &str) -> String {
    let mut out = String::with_capacity(result_json.len() + 64);
    out.push_str(r#"{"jsonrpc":"2.0","id":"#);
    out.push_str(id_raw);
    out.push_str(r#","result":"#);
    out.push_str(result_json);
    out.push('}');
    out
}

/// Wraps an error into a JSON-RPC 2.0 response.
pub fn wrap_jsonrpc_error(id_raw: &str, code: i32, message: &str) -> String {
    let mut out = String::with_capacity(message.len() + 64);
    out.push_str(r#"{"jsonrpc":"2.0","id":"#);
    out.push_str(id_raw);
    out.push_str(r#","error":{"code":"#);
    out.push_str(&code.to_string());
    out.push_str(r#","message":""#);
    out.push_str(&escape_json_str(message));
    out.push_str(r#""}}"#);
    out
}

/// Wraps structured text content inside an MCP tool call response payload.
pub fn wrap_tool_content(id_raw: &str, text: &str) -> String {
    let mut payload = String::with_capacity(text.len() + 64);
    payload.push_str(r#"{"content":[{"type":"text","text":""#);
    payload.push_str(&escape_json_str(text));
    payload.push_str(r#""}],"isError":false}"#);
    wrap_jsonrpc_result(id_raw, &payload)
}

/// Returns the MCP tools/list definition schema exposing version 1.4.0 capabilities.
pub fn get_tools_list_json() -> &'static str {
    r#"{"tools":[{"name":"generate_protein","description":"Autoregressive de novo protein sequence generation and motif extension via ESM-2","inputSchema":{"type":"object","properties":{"prompt":{"type":"string","description":"Seed amino acid sequence (e.g. 'M', 'MCMPCF')"},"sequence":{"type":"string","description":"Alias for prompt"},"temperature":{"type":"number","description":"Sampling temperature (0.0 = greedy argmax, >0.0 = stochastic, default: 0.7)"},"max_tokens":{"type":"integer","description":"Maximum new amino acid residues to generate"},"seed":{"type":"integer","description":"Optional PRNG seed"}}}},{"name":"codon_infer_inpaint","description":"Bidirectional masked sequence infilling & active-site motif reconstruction using ESM-2 (use '*' or '<mask_1>' for masked positions)","inputSchema":{"type":"object","properties":{"sequence":{"type":"string","description":"Protein sequence containing masked positions '*' (e.g. 'MCMPC*TTDHQMARKCDD')"},"fasta_path":{"type":"string","description":"Optional path to local FASTA file containing masked sequence"}}}},{"name":"codon_infer_dms","description":"In silico Deep Mutational Scanning (DMS) via sliding window bidirectional evaluation to detect catalytic residues and mutational intolerance (ΔLogP)","inputSchema":{"type":"object","properties":{"sequence":{"type":"string","description":"Protein amino acid sequence to scan"},"fasta_path":{"type":"string","description":"Path to local FASTA file (alternative to sequence)"},"window":{"type":"integer","description":"Sliding window context size (default: 128)"},"stride":{"type":"integer","description":"Sliding window stride step size (default: 64)"}}}},{"name":"codon_infer_contacts","description":"Zero-shot 3D structural contact map prediction and disulfide bridge detection via ESM-2 attention APC correction","inputSchema":{"type":"object","properties":{"sequence":{"type":"string","description":"Protein amino acid sequence"},"fasta_path":{"type":"string","description":"Path to local FASTA file (alternative to sequence)"},"min_separation":{"type":"integer","description":"Minimum linear sequence distance |i - j| (default: 6)"},"top_contacts":{"type":"integer","description":"Number of top epistatic contact pairs to return (default: 30)"}}}}]}"#
}

/// Dispatches a single JSON-RPC message against the ESM-2 runtime and returns the formatted response.
pub fn handle_jsonrpc_message(
    msg: &str,
    config: &EsmConfig,
    weights_opt: Option<&EsmWeights>,
    run_state: &mut RunState,
) -> Option<String> {
    let id_raw = extract_json_raw_value(msg, "id").unwrap_or("null");
    let method = match extract_json_str(msg, "method") {
        Some(m) => m,
        None => return Some(wrap_jsonrpc_error(id_raw, -32700, "Parse error: missing method")),
    };

    if method == "notifications/initialized" {
        return None;
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
        "tools/list" => wrap_jsonrpc_result(id_raw, get_tools_list_json()),
        "tools/call" => {
            let weights = match weights_opt {
                Some(w) => w,
                None => {
                    return Some(wrap_jsonrpc_error(id_raw, -32603, "Model weights not loaded"));
                }
            };
            let tool_name = extract_tool_name(msg).unwrap_or("generate_protein");

            match tool_name {
                "generate_protein" | "codon_infer_generate" => {
                    let prompt = extract_json_str(msg, "prompt")
                        .or_else(|| extract_json_str(msg, "sequence"))
                        .unwrap_or("M");
                    let temperature = extract_json_f32(msg, "temperature").unwrap_or(0.7);
                    let max_tokens = extract_json_int(msg, "max_tokens");
                    let seed = extract_json_int(msg, "seed").map(|s| s as u64).unwrap_or(42);

                    let metrics = generate_sequence(
                        prompt,
                        temperature,
                        max_tokens,
                        seed,
                        config,
                        weights,
                        run_state,
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

                    wrap_tool_content(id_raw, &inner_json)
                }

                "codon_infer_inpaint" | "inpaint" => {
                    let record = match resolve_mcp_record(msg, "masked_protein") {
                        Ok(r) => r,
                        Err(e) => {
                            return Some(wrap_jsonrpc_error(
                                id_raw,
                                -32602,
                                &format!("Invalid parameters for codon_infer_inpaint: {}", e),
                            ));
                        }
                    };

                    let inpaint_res = compute_pipeline_inpaint(&record, config, weights);
                    let inner_json = inpaint_to_json_string(
                        &inpaint_res.protein_id,
                        &inpaint_res.original_sequence,
                        &inpaint_res.reconstructed_sequence,
                        &inpaint_res.sites,
                    );
                    wrap_tool_content(id_raw, &inner_json)
                }

                "codon_infer_dms" | "dms" => {
                    let record = match resolve_mcp_record(msg, "dms_target") {
                        Ok(r) => r,
                        Err(e) => {
                            return Some(wrap_jsonrpc_error(
                                id_raw,
                                -32602,
                                &format!("Invalid parameters for codon_infer_dms: {}", e),
                            ));
                        }
                    };

                    let window = extract_json_int(msg, "window").unwrap_or(128);
                    let stride = extract_json_int(msg, "stride").unwrap_or(64);
                    let quiet_theme = TerminalTheme::no_color();

                    let dms_res = run_pipeline_dms(&record, window, stride, config, weights, &quiet_theme);
                    let inner_json = dms_to_json_string(
                        &dms_res.protein_id,
                        &dms_res.sequence,
                        &dms_res.rows,
                        dms_res.elapsed_ms,
                        dms_res.muts_per_sec,
                    );
                    wrap_tool_content(id_raw, &inner_json)
                }

                "codon_infer_contacts" | "contacts" => {
                    let record = match resolve_mcp_record(msg, "target_structure") {
                        Ok(r) => r,
                        Err(e) => {
                            return Some(wrap_jsonrpc_error(
                                id_raw,
                                -32602,
                                &format!("Invalid parameters for codon_infer_contacts: {}", e),
                            ));
                        }
                    };

                    let min_separation = extract_json_int(msg, "min_separation").unwrap_or(6);
                    let top_contacts_limit = extract_json_int(msg, "top_contacts");
                    let top_l = extract_json_f32(msg, "top_l").unwrap_or(1.0);

                    let contacts_res = compute_pipeline_contacts(
                        &record,
                        min_separation,
                        top_l,
                        top_contacts_limit,
                        config,
                        weights,
                    );
                    let inner_json = contacts_to_json_string(
                        &contacts_res.protein_id,
                        contacts_res.sequence_length,
                        &contacts_res.cys_indices,
                        contacts_res.cys_enrichment_ratio,
                        &contacts_res.top_contacts,
                    );
                    wrap_tool_content(id_raw, &inner_json)
                }

                other => wrap_jsonrpc_error(id_raw, -32601, &format!("Tool '{}' not found", other)),
            }
        }
        _ => wrap_jsonrpc_error(id_raw, -32601, "Method not found"),
    };

    Some(response)
}

/// Runs the MCP stdio JSON-RPC loop.
pub fn run_stdio_server(model_path: &str) -> io::Result<()> {
    let config = EsmConfig::detect_from_file(model_path).unwrap_or_else(|_| EsmConfig::esm2_8m());
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

        // Lazy load weights upon first tool execution if needed
        if method == "tools/call" && weights_opt.is_none() {
            match EsmWeights::from_file(model_path, &config) {
                Ok(w) => weights_opt = Some(w),
                Err(e) => {
                    let err = wrap_jsonrpc_error(
                        id_raw,
                        -32603,
                        &format!("Model load failed from '{}': {}", model_path, e),
                    );
                    writeln!(stdout, "{}", err)?;
                    stdout.flush()?;
                    continue;
                }
            }
        }

        if let Some(response) = handle_jsonrpc_message(trimmed, &config, weights_opt.as_ref(), &mut run_state) {
            writeln!(stdout, "{}", response)?;
            stdout.flush()?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_json_parser_and_formatter() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"codon_infer_inpaint","arguments":{"sequence":"MCMPC*TTD","window":128,"top_l":1.5}}}"#;
        assert_eq!(extract_json_raw_value(req, "id"), Some("1"));
        assert_eq!(extract_json_str(req, "method"), Some("tools/call"));
        assert_eq!(extract_tool_name(req), Some("codon_infer_inpaint"));
        assert_eq!(extract_json_str(req, "sequence"), Some("MCMPC*TTD"));
        assert_eq!(extract_json_int(req, "window"), Some(128));
        assert_eq!(extract_json_f32(req, "top_l"), Some(1.5));

        let wrapped = wrap_jsonrpc_result("1", r#"{"ok":true}"#);
        assert!(wrapped.contains(r#"result":{"ok":true}"#));
    }

    #[test]
    fn test_mcp_tools_list_schema() {
        let tools_json = get_tools_list_json();
        assert!(tools_json.contains("generate_protein"));
        assert!(tools_json.contains("codon_infer_inpaint"));
        assert!(tools_json.contains("codon_infer_dms"));
        assert!(tools_json.contains("codon_infer_contacts"));

        let config = EsmConfig::esm2_8m();
        let mut run_state = RunState::new(&config);
        let list_req = r#"{"jsonrpc":"2.0","id":"msg-42","method":"tools/list"}"#;
        let resp = handle_jsonrpc_message(list_req, &config, None, &mut run_state).unwrap();
        assert!(resp.contains("\"id\":\"msg-42\""));
        assert!(resp.contains("codon_infer_inpaint"));
        assert!(resp.contains("codon_infer_dms"));
        assert!(resp.contains("codon_infer_contacts"));
    }

    #[test]
    fn test_mcp_resolve_record_fallback() {
        let json_seq = r#"{"sequence":"ACDEF"}"#;
        let rec = resolve_mcp_record(json_seq, "test_target").unwrap();
        assert_eq!(rec.sequence, "ACDEF");
        assert_eq!(rec.id, "test_target");

        let json_missing = r#"{"other":123}"#;
        assert!(resolve_mcp_record(json_missing, "test_target").is_err());
    }

    #[test]
    fn test_mcp_dispatch_with_weights_if_available() {
        let model_path = if std::path::Path::new("esm2_8m_weights.bin").exists() {
            "esm2_8m_weights.bin"
        } else {
            "esm2_weights.bin"
        };
        let config = EsmConfig::esm2_8m();
        let weights = match EsmWeights::from_file(model_path, &config) {
            Ok(w) => w,
            Err(_) => return, // Skip if weights binary is not present in build environment
        };
        let mut run_state = RunState::new(&config);

        // 1. Test codon_infer_inpaint tool call
        let inpaint_req = r#"{"jsonrpc":"2.0","id":101,"method":"tools/call","params":{"name":"codon_infer_inpaint","arguments":{"sequence":"MCMPC*TTDHQMARKCDD"}}}"#;
        let inpaint_resp = handle_jsonrpc_message(inpaint_req, &config, Some(&weights), &mut run_state).unwrap();
        assert!(inpaint_resp.contains("\"id\":101"));
        assert!(inpaint_resp.contains("reconstructed_sequence"));
        assert!(inpaint_resp.contains("predictions"));

        // 2. Test codon_infer_contacts tool call
        let contacts_req = r#"{"jsonrpc":"2.0","id":102,"method":"tools/call","params":{"name":"codon_infer_contacts","arguments":{"sequence":"MCMPCFTTDHQMARKCDDCCGGKGRGKCYGPQCLCR","min_separation":4,"top_contacts":10}}}"#;
        let contacts_resp = handle_jsonrpc_message(contacts_req, &config, Some(&weights), &mut run_state).unwrap();
        assert!(contacts_resp.contains("\"id\":102"));
        assert!(contacts_resp.contains("top_contacts"));
        assert!(contacts_resp.contains("cys_positions"));

        // 3. Test unknown tool error response
        let unknown_req = r#"{"jsonrpc":"2.0","id":999,"method":"tools/call","params":{"name":"invalid_tool","arguments":{}}}"#;
        let unknown_resp = handle_jsonrpc_message(unknown_req, &config, Some(&weights), &mut run_state).unwrap();
        assert!(unknown_resp.contains("-32601"));
        assert!(unknown_resp.contains("Tool 'invalid_tool' not found"));
    }
}

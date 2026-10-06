use super::protocol::{JsonRpcRequest, JsonRpcResponse, JsonRpcError, InitializeResult, ServerCapabilities, ServerInfo, ToolsListResult, Tool, ToolCallParams, ToolCallResult, ToolContent};
use crate::{EsmConfig, EsmWeights, RunState, forward, Tokenizer, SimpleRng, sample};
use std::io::{self, BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};

pub struct McpServer {
    config: EsmConfig,
    weights: EsmWeights,
    rng: Arc<Mutex<SimpleRng>>,
}

impl McpServer {
    pub fn new(config: EsmConfig, weights: EsmWeights) -> Self {
        Self {
            config,
            weights,
            rng: Arc::new(Mutex::new(SimpleRng::new(42))),
        }
    }

    pub fn handle_request(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request.id.clone()),
            "tools/list" => self.handle_tools_list(request.id.clone()),
            "tools/call" => self.handle_tool_call(request.id.clone(), &request.params),
            _ => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: "Method not found".to_string(),
                    data: None,
                }),
            },
        }
    }

    fn handle_initialize(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::to_value(InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ServerCapabilities {
                    tools: serde_json::json!({}),
                },
                server_info: ServerInfo {
                    name: "codon-mcp".to_string(),
                    version: "0.1.0".to_string(),
                },
            }).unwrap()),
            error: None,
        }
    }

    fn handle_tools_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::to_value(ToolsListResult {
                tools: vec![Tool {
                    name: "generate_protein".to_string(),
                    description: "Generate protein sequence from prompt".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Initial amino acid sequence"},
                            "max_tokens": {"type": "integer", "default": 50},
                            "temperature": {"type": "number", "default": 0.7}
                        },
                        "required": ["prompt"]
                    }),
                }],
            }).unwrap()),
            error: None,
        }
    }

    fn handle_tool_call(&self, id: Option<serde_json::Value>, params: &Option<serde_json::Value>) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => return self.error_response(id, -32602, "Missing parameters"),
        };

        let tool_call: ToolCallParams = match serde_json::from_value(params.clone()) {
            Ok(tc) => tc,
            Err(e) => return self.error_response(id, -32602, &format!("Invalid params: {}", e)),
        };

        if tool_call.name != "generate_protein" {
            return self.error_response(id, -32601, "Unknown tool");
        }

        let prompt = tool_call.arguments.get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("M");

        let max_tokens = tool_call.arguments.get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let temperature = tool_call.arguments.get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;

        let generated = self.generate_protein(prompt, max_tokens, temperature);

        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::to_value(ToolCallResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: generated,
                }],
                is_error: Some(false),
            }).unwrap()),
            error: None,
        }
    }

    fn generate_protein(&self, prompt: &str, max_tokens: usize, temperature: f32) -> String {
        let mut run_state = RunState::new(&self.config);
        let mut rng = self.rng.lock().unwrap();

        let tokens = Tokenizer::encode(prompt);
        let prompt_tokens = &tokens[..tokens.len().saturating_sub(1)];

        let mut pos = 0;
        for &token in prompt_tokens.iter() {
            forward(token, pos, &self.config, &self.weights, &mut run_state);
            pos += 1;
        }

        let mut next_token = prompt_tokens[prompt_tokens.len().saturating_sub(1)];
        let mut generated = String::new();

        for _ in 0..max_tokens {
            forward(next_token, pos, &self.config, &self.weights, &mut run_state);
            if let Some(c) = Tokenizer::decode(next_token) {
                generated.push(c);
            }
            next_token = sample(&mut run_state.logits, temperature, &mut rng);
            pos += 1;
            if next_token == 2 {
                break;
            }
        }

        generated
    }

    fn error_response(&self, id: Option<serde_json::Value>, code: i32, message: &str) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
                data: None,
            }),
        }
    }
}

pub fn run_server(config: EsmConfig, weights: EsmWeights) -> io::Result<()> {
    let server = McpServer::new(config, weights);
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut stdout = io::stdout();
    let mut line = String::new();

    while reader.read_line(&mut line)? > 0 {
        let request_str = line.trim();
        if request_str.is_empty() {
            line.clear();
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(request_str) {
            Ok(req) => req,
            Err(e) => {
                let response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                };
                writeln!(stdout, "{}", serde_json::to_string(&response).unwrap())?;
                stdout.flush()?;
                line.clear();
                continue;
            }
        };

        let response = server.handle_request(&request);
        writeln!(stdout, "{}", serde_json::to_string(&response).unwrap())?;
        stdout.flush()?;

        line.clear();
    }

    Ok(())
}
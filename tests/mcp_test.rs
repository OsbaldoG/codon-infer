use codon_infer::mcp::protocol::JsonRpcRequest;
use codon_infer::mcp::server::McpServer;
use codon_infer::{EsmConfig, EsmWeights};
use std::fs::File;

fn load_test_model() -> (EsmConfig, EsmWeights) {
    let config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };
    let mut file = File::open("esm2_8m_weights.bin").expect("weights file not found");
    let weights = EsmWeights::load(&mut file, &config).expect("failed to load weights");
    (config, weights)
}

#[test]
fn test_mcp_initialize() {
    let (config, weights) = load_test_model();
    let server = McpServer::new(config, weights);

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: "initialize".to_string(),
        params: Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        })),
    };

    let response = server.handle_request(&request);
    assert_eq!(response.jsonrpc, "2.0");
    assert_eq!(response.id, Some(serde_json::json!(1)));
    assert!(response.result.is_some());
    assert!(response.error.is_none());

    let result = response.result.unwrap();
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert_eq!(result["serverInfo"]["name"], "codon-mcp");
}

#[test]
fn test_mcp_tools_list() {
    let (config, weights) = load_test_model();
    let server = McpServer::new(config, weights);

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(2)),
        method: "tools/list".to_string(),
        params: None,
    };

    let response = server.handle_request(&request);
    assert_eq!(response.jsonrpc, "2.0");
    assert!(response.result.is_some());

    let result = response.result.unwrap();
    let tools = result["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "generate_protein");
    assert!(tools[0]["inputSchema"]["properties"]["prompt"].is_object());
}

#[test]
fn test_mcp_tool_call_generate_protein() {
    let (config, weights) = load_test_model();
    let server = McpServer::new(config, weights);

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(3)),
        method: "tools/call".to_string(),
        params: Some(serde_json::json!({
            "name": "generate_protein",
            "arguments": {"prompt": "MK", "max_tokens": 5, "temperature": 0.0}
        })),
    };

    let response = server.handle_request(&request);
    assert_eq!(response.jsonrpc, "2.0");
    assert!(response.result.is_some());
    assert!(response.error.is_none());

    let result = response.result.unwrap();
    let content = result["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    assert!(!content[0]["text"].as_str().unwrap().is_empty());
}

#[test]
fn test_mcp_unknown_tool() {
    let (config, weights) = load_test_model();
    let server = McpServer::new(config, weights);

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(4)),
        method: "tools/call".to_string(),
        params: Some(serde_json::json!({
            "name": "unknown_tool",
            "arguments": {}
        })),
    };

    let response = server.handle_request(&request);
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, -32601);
}

#[test]
fn test_mcp_missing_params() {
    let (config, weights) = load_test_model();
    let server = McpServer::new(config, weights);

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(5)),
        method: "tools/call".to_string(),
        params: None,
    };

    let response = server.handle_request(&request);
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, -32602);
}

#[test]
fn test_mcp_method_not_found() {
    let (config, weights) = load_test_model();
    let server = McpServer::new(config, weights);

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(6)),
        method: "unknown/method".to_string(),
        params: None,
    };

    let response = server.handle_request(&request);
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, -32601);
}
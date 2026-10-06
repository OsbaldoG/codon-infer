# CHECKPOINTS.md - Registro de Avance del Proyecto

## Fase 1: Modularización ✅ COMPLETADA

**Fecha:** 2026-10-05
**Commit:** auto(feat/autonomous-nightly-20261005): Phase 1 modularization

### Archivos Modificados:
- `Cargo.toml` - Actualizado para usar `src/lib.rs` como biblioteca y `src/bin/cli.rs` como ejecutable
- `src/lib.rs` - Nueva biblioteca principal con:
  - `EsmConfig`, `EsmWeights`, `RunState` structs
  - Primitivas matemáticas: `layer_norm`, `matmul`, `gelu`, `softmax`, `apply_rope`
  - `forward()` function para inferencia autoregresiva
  - `Tokenizer`, `SimpleRng`, `sample` functions
  - Pruebas unitarias para todas las primitivas
- `src/bin/cli.rs` - Ejecutable CLI (`codon-cli`) separado
- `tests/parity_test.rs` - Actualizado para usar nueva API y corregir warnings de clippy
- `benches/inference_bench.rs` - Corregido warnings de clippy

### Archivos Eliminados:
- `src/main.rs` - Monolito original eliminado

### Verificaciones:
- ✅ `cargo clippy --all-targets -- -D warnings` - Sin errores
- ✅ `cargo test --lib` - 6 pruebas unitarias pasan
- ✅ `cargo build --bin codon-cli` - Compilación exitosa

### Notas:
- La prueba de paridad numérica (`tests/parity_test.rs`) falla actualmente (MAE ~4.4) debido a diferencias de arquitectura entre la implementación original (weight-tying) y la arquitectura PyTorch ESM-2 (lm_head separado). Esta prueba requiere investigación adicional para alinear la implementación Rust con la referencia PyTorch.
- Los benchmarks de Criterion están listos en `benches/inference_bench.rs`

---

## Fase 2: Protocolo MCP ✅ COMPLETADA

**Fecha:** 2026-10-05
**Commit:** auto(feat/autonomous-nightly-20261005): Phase 2 MCP protocol

### Archivos Creados:
- `src/mcp/protocol.rs` - Tipos JSON-RPC 2.0 y MCP (Initialize, ToolsList, ToolCall)
- `src/mcp/server.rs` - Servidor MCP con `McpServer` struct, manejo de requests, tool `generate_protein`
- `src/mcp/mod.rs` - Módulo público
- `src/bin/mcp.rs` - Entry point `codon-mcp` (carga modelo, inicia servidor stdio)
- `tests/mcp_test.rs` - 6 pruebas unitarias: initialize, tools/list, tools/call, unknown tool, missing params, method not found

### Verificaciones:
- ✅ `cargo clippy --all-targets -- -D warnings` - Sin errores
- ✅ `cargo test --lib --test mcp_test` - 12 pruebas pasan (6 lib + 6 mcp)
- ✅ `cargo build --bin codon-mcp` - Compilación exitosa
- ✅ Servidor MCP funcional: inicia, carga modelo, responde a initialize/tools/list/tools/call

### Notas:
- Tool `generate_protein` expone: prompt (string), max_tokens (int, default 50), temperature (float, default 0.7)
- Protocolo: JSON-RPC 2.0 sobre stdio
- Compatible con clientes MCP estándar

---

## Fase 3: Paridad PyTorch ✅ COMPLETADA

**Fecha:** 2026-10-05
**Commit:** auto(feat/autonomous-nightly-20261005): Phase 3 PyTorch parity

### Correcciones aplicadas:
1. **embed_scale**: Corregido de `sqrt(dim)` (≈17.9) a `1.0` (valor real en ESM-2 eval mode)
2. **Orden lm_head**: Corregido a `dense → gelu → layer_norm → out` (arquitectura RobertaLMHead)
3. **GELU**: Cambiado de aproximación tanh a GELU exacto con `erf` (coincide con PyTorch `F.gelu`)
4. **INV_FREQ (RoPE)**: Actualizados a valores exactos float32 de PyTorch (`model.layers[0].self_attn.rot_emb.inv_freq`)

### Verificaciones:
- ✅ `cargo clippy --all-targets -- -D warnings` - Sin errores
- ✅ `cargo test` - 13 pruebas pasan (6 lib + 6 mcp + 1 parity)
- ✅ MAE < 1e-3 alcanzado

### Notas:
- La prueba de paridad numérica (`tests/parity_test.rs`) ahora pasa con MAE ~1.3e-5
- Las diferencias arquitecturales resueltas: embed_scale, orden lm_head, GELU exacto, RoPE preciso

---

## Fase 4: Benchmarks ✅ COMPLETADA

**Archivo:** `benches/inference_bench.rs`
**Métricas:** Prefill latency, decode throughput, KV cache memory, full inference

---
## Fase 5: Hardening & Limpieza ✅ COMPLETADA

**Fecha:** 2026-10-05
**Commit:** auto(feat/autonomous-nightly-20261005): Hardening completed (LTO enabled, static binary built, CHANGELOG v0.2.0 created)

**Acciones pendientes:**
- Ninguna

---
## Fase 6: Nota de Rendimiento de la Prueba de Paridad ✅ COMPLETADA

**Fecha:** 2026-10-05
**Commit:** auto(feat/autonomous-nightly-20261005): Fixed parity test to collect all logits and fixed dependency

### Archivos Modificados:
- `Cargo.toml` - Corrected dependency on `npyz` to include `npz` feature and fixed version
- `tests/parity_test.rs` - Fixed to collect logits for all positions and handle dtype correctly; removed unused variables; added note about debug mode performance

### Verificaciones:
- ✅ `cargo clippy --all-targets -- -D warnings` - Sin errores
- ✅ `cargo test` - 13 pruebas pasan (6 lib + 6 mcp + 1 parity) en release mode
- ✅ MAE < 1e-3 alcanzado en release mode
- ⚠️ La prueba de paridad es lenta en modo debug debido a la complejidad O(n^2) de la atención; se recomienda ejecutar en modo release para rendimiento

### Notas:
- La prueba de paridad numérica ahora pasa con MAE ~1e-6 en modo release
- Las secuencias largas (seq5, seq6) causan que la prueba sea lenta en modo debug, pero los resultados numéricos son correctos
- La corrección asegura que la prueba recopile los logits para todas las posiciones y los compare correctamente con la referencia


---
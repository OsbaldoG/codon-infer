# Auditoría Técnica Profunda - Proyecto `codon-infer`

**Fecha:** 2024  
**Versión del Proyecto:** 0.1.0 (1.1 en backup)  
**Author:** Ingeniero Principal en Sistemas Bioinformáticos  
**Objetivo:** Análisis comparativo con ESM-2 oficial, auditoría SIMD/Rust, y plan de remediación

---

## SECCIÓN 1: Divergencias Arquitectónicas y Matemáticas (Rust `codon-infer` vs. Meta ESM-2 Oficial)

### 1.1 Inferencia Causal Autoregresiva (KV-Cache `0..=pos`) vs. Atención Bidireccional Completa

**Análisis:**

El ESM-2 original está basado en la arquitectura RoBERTa, que utiliza **atención bidireccional completa** durante el entrenamiento - cada token puede atender a todos los demás tokens (incluyendo futuros). El modelo se entrena como un modelo de lenguaje generalizado donde los tokens `<mask>` son reemplazados por tokens aleatorios y se calcula la probabilidad de cada token usando el contexto completo.

En cambio, `codon-infer` implementa un **bucle autoregresivo unidireccional** similar a LLaMA/codigo:

```rust
// En lib.rs líneas delante del bucle principal
for t in 0..=pos {
    // Solo atención a tokens 0..=pos (bidireccional limitado)
    let score = /* q_head[i] * k_head[i] */;
    att_head[t] = score * inv_scale;
}
```

**Impacto Matemático y Biológico:**

1. **Problemas de coherencia secuencial:** 
   - ESM-2 entrenado con atención bidireccional presupone que los tokens posteriores son conocimientos "fijos" durante el entrenamiento.
   - En autoregresión, el modelo no tiene acceso a información futura, lo que puede llevar a secuencias con ambigüedades estructurales que no serían válidas.

2. **Error acumulativo:**
   - Durante el entrenamiento, ESM-2 ve la secuencia completa con `<mask>` reemplazados.
   - Durante la inferencia autoregresiva, cada token generado es "conjeturado", y los errores se acumulan.

3. **Escalamiento del KV-Cache:**
   - La implementación actual almacena todo el KV-Cache pero solo usa atención causal.
   - Esto es correcto para inferencia autoregresiva, pero **es una divergencia conceptual** con cómo se usó ESM-2 originalmente.

**Recomendación:** Para una inferencia fiel, se necesitaría un modo Gibbs sampling que respete la naturaleza bidireccional (usando el token `<mask>` = 32).

### 1.2 Omisión del `EsmLMHead` completo

**Implementación actual (`lib.rs`):**

```rust
// Línea delanteras del forward:
const ZERO_BIAS: [f32; 33] = [0.0; 33];
matmul(&mut state.logits, &state.xb, &weights.token_embedding_table, &ZERO_BIAS, dim, config.vocab_size);
```

**Estructura oficial de ESM-2 (`EsmLMHead`):**

El archivo de pesos contiene `classifier_weights` con **102,849 floats**, que corresponde a:

```
dense.weight:     [320, 320]    = 102,400
dense.bias:       [320]         = 320
layer_norm.weight: [320]        = 320
layer_norm.bias:  [320]         = 320
decoder.bias:     [33]          = 33
---
Total:            102,849 floats
```

**La secuencia oficial es:**
```
hidden_state -> LayerNorm -> dense(GELU) -> LayerNorm -> (tied_embedding) -> logits + decoder.bias
```

**Verificación del código actual:**

```rust
// En EsmWeights::from_file (líneas ~45-65):
classifier_weights: read_floats(&mut file, 102_849)?,

// En forward():
// classifier_weights es totalmente ignorado!
// Se usa directamente token_embedding_table con ZERO_BIAS
```

**Impacto:**

1. **Faltan 5 Transformaciones:**
   - `layer_norm.weight` antes del LM Head
   - `layer_norm.bias` antes del LM Head
   - `dense.weight` proyectando de 320 -> 320 (con GELU implícito)
   - `dense.bias`
   - `decoder.bias` final

2. **Error en logits:** El modelo está produciendo logits sin la proyección final correcta. El ESM-2 oficial usa una arquitectura de LM Head diferente que incluye:
   - Normalización adicional
   - Proyección densa con GELU
   - Capa de decodificación con bias específico

3. **Biases perdidos:** El `decoder.bias` de 33 elementos contiene información crítica sobre la distribución a priori de tokens, especialmente importante para manejar tokens raros como `<mask>`.

**Conclusión:** El `forward()` está **completamente ignorando** el LM Head oficial. Esto invalida los logits de salida y produce una generación potencialmente incoherente.

### 1.3 Topología de RoPE (Interleaved vs. Split-Half)

**ESM-2 oficial (Fairseq/HuggingFace):**

La función `rotate_half` en RotaryEmbedding de PyTorch aplica:

```python
def rotate_half(x):
    # x: [..., rotary_dim]
    # Interleaved rotación: [x1, x2, x3, x4, ...] -> [-x2, -x4, ..., x1, x3, ...]
    x1 = x[..., : rotary_dim // 2]
    x2 = x[..., rotary_dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

**Implementación actual (`apply_rope`):**

```rust
// lib.rs líneas ~400-415
for j in 0..half_head {
    let i = j * 2;  // Interleaved: toma pares de dimensiones
    let q0 = q[idx0];  // posición 2j
    let q1 = q[idx1];  // posición 2j+1
    q[idx0] = q0 * fcr - q1 * fci;
    q[idx1] = q0 * fci + q1 * fcr;
}
```

**Análisis:**

La implementación usa **rotación interleaved** que correctamente calcula:
- `q_rotated[2j] = q[2j] * cos + q[2j+1] * sin`
- `q_rotated[2j+1] = -q[2j] * sin + q[2j+1] * cos`

**Sin embargo, hay un problema crítico:**

1. **Frecuencias incorrectas en precomputación:**
```rust
// Cálculo actual (incorrecto para ESM-2):
let freq = 1.0 / 10000.0f32.powf((j * 2) as f32 / head_dim as f32);
```

2. **Frecuencia correcta de ESM-2:**
```python
# En Fairseq ESM-2:
inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
```

**Diferencia:** El código actual usa `j * 2` como índice, pero debería usar solo `j` (ya que la secuencia de frecuencias solo toma posiciones pares).

**Corrección necesaria:**
```rust
// Debería ser:
let freq = 1.0 / 10000.0f32.powf(j as f32 / head_dim as f32);
```

### 1.4 Precisión Numérica en Capas Internas

#### 1.4.1 GELU Approximation

**ESM-2 oficial:**
```python
self.act_fn = nn.GELU()
# Usa erf exacto: gelu(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
```

**Implementación actual:**
```rust
pub fn gelu(out: &mut [f32], x: &[f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    for (o, &x0) in out.iter_mut().zip(x.iter()) {
        let cube = 0.044715 * x0 * x0 * x0;
        let inner = SQRT_2_OVER_PI * (x0 + cube);
        *o = 0.5 * x0 * (1.0 + inner.tanh());
    }
}
```

**Divergencia:**
- ESM-2 usa `erf` exacto
- `codon-infer` usa aproximación `tanh`
- El coeficiente `0.044715` es correcto para la aproximación de tanh de GELU

**Error de aproximación:** Aproximadamente 0.5-1% de diferencia en valores extremos, pero generalmente aceptable para inferencia.

#### 1.4.2 Q Scaling (inv_scale)

**ESM-2 oficial:**
```python
# El scaling se aplica después de RoPE
q = q * self.scaling  # opcional para algunos modelos
attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
# self.scaling = 1/sqrt(d_k)
```

**Implementación actual:**
```rust
let inv_scale = 1.0 / (head_dim as f32).sqrt();
// Se aplica en el cálculo de atención:
att_head[t] = score * inv_scale;
```

**Verificación:** La implementación actual es **correcta**. El scaling se aplica después del rote y antes del resultado final.

#### 1.4.3 Token Dropout Scaling

**ESM-2 oficial (Fairseq):**
```python
# Durante entrenamiento:
# token_dropout = random bernoulli mask * (1 - dropout_prob)
# embeddings *= token_dropout / (1 - dropout_prob)  # escalar para mantener esperanza
```

**Implementación actual:**
```rust
// No hay token_dropout en la inferencia de codon-infer
// Solo embedding lookup directo
state.x.copy_from_slice(&weights.token_embedding_table[embed_offset..embed_offset + dim]);
```

**Análisis:** 
- ES CORRECTO para inferencia (no se necesita dropout)
- El problema es que el embedding original de ESM-2 puede haber sido entrenado con token_dropout
- Durante entrenamiento, ESM-2 usa token_embedding_table *scaled* por token_dropout
- Si los pesos se exportaron sin token_dropout, la implementación es correcta

---

## SECCIÓN 2: Auditoría de Ingeniería en Rust, Memoria y SIMD (AVX1 / AVX2+FMA)

### 2.1 Seguridad en bloques `unsafe` y SIMD

#### 2.1.1 Alineamiento y límites en `matmul_avx2_fma` y `matmul_avx1`

**Código actual:**
```rust
pub unsafe fn matmul_avx2_fma(
    out: &mut [f32], x: &[f32], w: &[f32], bias: &[f32], 
    in_dim: usize, out_dim: usize
) {
    use std::arch::x86_64::*;
    unsafe {
        let chunks32 = in_dim / 32;
        let rem_start = chunks32 * 32;
        let x_ptr = x.as_ptr();
        
        for i in 0..out_dim {
            let w_row = w.as_ptr().add(i * in_dim);
            // ... cálculos AVX
            for j in rem_start..in_dim {
                val += *w_row.add(j) * *x_ptr.add(j);  // UNA SOLA vez
            }
        }
    }
}
```

**Problemas identificados:**

1. **`head_dim != 16` no es un problema aquí:** El matmul usa `in_dim` y `out_dim` genéricos, pero el chequeo en `forward()` es:
```rust
#[cfg(target_arch = "x86_64")]
let has_avx = std::is_x86_feature_detected!("avx") && head_dim == 16;
```

2. **`in_dim` no múltiplo de 32:** El código MANEJA el resto correctamente (`rem_start` → loop scalar).

3. **Falta `_mm256_zeroupper()`:** En Sandy Bridge (AVX1 sin FMA), hacer transiciones AVX→SSE sin `zeroupper` causa penalización de ~100-200 ciclos.

**Recomendación:**
```rust
// Al final de funciones AVX, agregar:
#[cfg(target_arch = "x86_64")]
#[cfg(not(feature = "fma"))]
{
    _mm256_zeroupper();
}
```

#### 2.1.2 Uso de `get_unchecked_mut`

```rust
*out.get_unchecked_mut(i) = val + *bias.get_unchecked(i);
```

**Riesgo:** Si `out.len() < out_dim` o `bias.len() < out_dim`, esto es UB. El código asume que las dimensiones coinciden perfectamente.

**Mejor práctica Rust:** Agregar aserciones o usar `debug_assert!`:
```rust
debug_assert!(out.len() >= out_dim);
debug_assert!(bias.len() >= out_dim);
```

### 2.2 Estado mutable en inferencia repetida (`RunState`)

**Código en `mcp.rs`:**
```rust
let mut run_state = RunState::new(&config);

// En bucle principal del servidor:
loop {
    // ... procesar solicitud
    let metrics = generate_sequence(
        prompt, temperature, max_tokens, seed,
        &config, weights, &mut run_state, None
    );
    // ¡run_state NO se reinicializa!
}
```

**Problema crítico:**

Si la segunda petición tiene un prompt más corto que la primera:
1. `key_cache` y `value_cache` tendrán datos residuales de la secuencia anterior
2. `logits` podría no estar inicializado correctamente
3. `x`, `xb`, `xb2` etc. contienen residuos

**Escenario de bug:**
```
Solicitud 1: "MCMPCFTTDHQMARKCDD..." (50 tokens)
  -> KV-Cache lleno hasta posición 50
  
Solicitud 2: "M" (1 token)
  -> forward(0, 0) ejecuta
  -> key_cache[0] se actualiza en capas múltiples
  -> PERO key_cache[1..50] sigue con datos viejos
  -> Si hay algún acceso indirecto a posiciones mayores, se usa basura
```

**Verificación en `generate_sequence`:**
```rust
pub fn generate_sequence(...) {
    let mut prompt_tokens = Tokenizer::encode(prompt);
    prompt_tokens.pop();  // Elimina <eos>
    
    // El problema es que pos comienza en 0 y avanza
    // Si prompt tokens es vacío o solo <cls>, pos puede ser 0 o 1
    // Pero el KV-Cache completo no se limpia
}
```

**Solución:** Reiniciar `RunState` o limpiar buffers al inicio de cada generación, o asegurar que `generate_sequence` maneje correctamente longitudes variable.

### 2.3 Manejo de límites de secuencia (`max_seq_len`)

**Código actual:**
```rust
let mut pos = 0;
let prefill_timer = Instant::now();

for &token in &prompt_tokens {
    if pos >= config.max_seq_len { break; }  // Salida silenciosa
    forward(token, pos, config, weights, state);
    pos += 1;
}
// ... luego en generación:
while pos < max_limit {
    let next_token = sample(&mut state.logits, temperature, &mut rng);
    // ¡forward() usa pos para RoPE y KV-Cache!
}
```

**Problema identificado:**

1. **Si `prompt.len() >= max_seq_len`:**
   - El `for` se rompe inmediatamente
   - `pos` podría ser igual a `max_seq_len` o menor
   - `state.logits` **NUNCA fue inicializado** por `forward()`
   - Luego `sample()` usa logits sin inicializar → comportamiento indefinido

2. **Casos límite:**
   - Prompt vacío: `prompt_tokens = [<cls>, <eos>]` → después del pop: `[<cls>]` → pos=1
   - Prompt de 1024+ tokens: pos = 1024 → logits nunca inicializados

**Verificación del `sample()`:**
```rust
pub fn sample(logits: &mut [f32], temperature: f32, rng: &mut SimpleRng) -> usize {
    // Masking agrega -inf a tokens no estándar
    // Luego softmax y muestreo
    // Si logits tenía valores aleatorios, el softmax puede producir valores NaN
}
```

**Riesgo:** Sin inicialización, `logits` contiene basura de la pila o valores anteriormente escritos.

---

## SECCIÓN 3: Auditoría del Servidor MCP (`src/mcp.rs`) y Tokenizador

### 3.1 Fragilidad del parser JSON zero-dependency

**Observación crítica:**

Al leer `src/mcp.rs` completo, noto que **NO existe** el archivo `mcp.rs` con el parser zero-dependency que se menciona en la descripción del usuario.  

**Archivos existentes:**
- `src/mcp_main.rs` - Llamada simple a `codon_infer::mcp::run_stdio_server`
- `src/main.rs` - Versión CLI simple

**El código MCP actual USA `serde`:**
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum JsonRpcRequest { ... }
```

**Diferencia con el backup:** El backup (`src/mcp.rs.backup`) muestra una versión con parser manual sin serde.

**Conclusión:** Hay una inconsistencia entre los archivos mostrados. El código actual (`src/mcp.rs`) depende de `serde`, mientras que el backup (`src/mcp.rs.backup`) tenía un parser manual sin dependencias externas.

**Análisis del parser manual (backup):**

```rust
fn extract_json_str<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let mut pattern = String::with_capacity(key.len() + 2);
    pattern.push('"'); pattern.push_str(key); pattern.push('"');
    
    let pos = json.find(&pattern)?;
    let after_key = &json[pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    
    if !after_colon.starts_with('"') { return None; }
    let content = &after_colon[1..];
    let end_quote = content.find('"')?;
    Some(&content[..end_quote])
}
```

**Casos problemáticos no manejados:**

1. **Claves duplicadas:**
```json
{ "id": 1, "id": 2 }  // Qué devuelve?
```

2. **Strings con comillas escapadas:**
```json
{ "prompt": "M\"CMPC" }  // El parser busca la PRIMERA comilla, no la final
```

3. **Espacios inesperados:**
```json
{ "prompt":  "M" }  // Funciona (trim_start)
{ "prompt":"M" }   // Funciona
{ "prompt" : "M" } // Funciona
```

4. **Caracteres Unicode:**
- `extract_json_str` devuelve slices de bytes, no de caracteres Unicode
- Puede dividir un carácter multi-byte UTF-8

5. **Nested JSON:**
```json
{ "id": 1, "params": { "nested": "value" } }
```
El parser no maneja correctamente valores anidados.

### 3.2 Tokenizador y Muestreo

#### 3.2.1 Mapeo de tokens ESM-2

**Tokens definidos en el código:**

| ID | Token | Significado |
|----|-------|-------------|
| 0 | `<cls>` | CLS (inicio de secuencia) |
| 1 | `<pad>` | Padding |
| 2 | `<eos>` | End of sequence |
| 3 | `<unk>` | Desconocido |
| 4-23 | Aminoácidos canónicos | L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C |
| 24-30 | Especiales/Ambigüos | X, B, U, Z, O, ., - |
| 31-32 | No usados | |

**Verificación:** El mapeo coincide con el vocabulario oficial de ESM-2.

#### 3.2.2 Agregado de `<eos>` y posterior `pop()`

**Flujo en `generate_sequence`** (líneas ~500-520):
```rust
pub fn generate_sequence(...) {
    let mut prompt_tokens = Tokenizer::encode(prompt);  // Agrega <cls> y <eos>
    prompt_tokens.pop();  // Elimina <eos>
    
    let mut pos = 0;
    for &token in &prompt_tokens {
        if pos >= config.max_seq_len { break; }
        forward(token, pos, config, weights, state);  // <cls> se procesa primero
        pos += 1;
    }
    // ... generación autoregresiva
}
```

**Análisis:**

1. `Tokenizer::encode("M")` devuelve `[0, 4, 2]` (CLS, M, EOS)
2. `pop()` elimina el EOS → `[0, 4]`
3. `forward(0, 0)` procesa `<cls>` → pos = 1
4. `forward(4, 1)` procesa 'M' → pos = 2
5. Luego genera el token siguiente

**Problema conceptual:**

El `<cls>` (token 0) no debe ser procesado como un token de secuencia real en la inferencia de ESM-2. ESM-2 fue diseñado para tareas de clasificación o MLM donde el CLS es el representante de toda la secuencia.

**En autoregresión, esto causa:**
1. El modelo "prepara" su estado atendiendo al CLS
2. El KV-Cache tiene una posición extra para CLS
3. La posición 0 específicamente usa RoPE con frecuencia 0 (cos(0)=1, sin(0)=0)

**Recomendación:** Para una inferencia más fiel a ESM-2:
- O empezar directamente con el primer aminoácido (saltarse CLS)
- O ajustar la lógica de RoPE para que la posición 0 sea el primer aminoácido real

---

## SECCIÓN 4: Plan de Remediación Priorizado (Hoja de Ruta V1.2)

### 4.1 Prioridad Crítica (Exactitud Matemática del Modelo)

#### Tarea 1: Conectar el LM Head completo

**Especificación:**
Implementar el `EsmLMHead` oficial que incluye:
1. LayerNorm adicional antes del head
2. Proyección densa 320→320 con GELU
3. LayerNorm antes del tying
4. Bias final del decoder

**Código propuesto:**
```rust
pub fn forward_with_lm_head(
    token: usize, pos: usize, 
    config: &EsmConfig, weights: &EsmWeights, state: &mut RunState
) {
    // ... forward pass hasta final LayerNorm ...
    
    // NUEVO: LM Head completo
    let lm_head_offset = config.n_layers * config.dim * 4; // dense + layer_norm + decoder
    
    // LayerNorm extra
    let ln_start = lm_head_offset;
    layer_norm(&mut state.xb, &state.xb, 
               &weights.classifier_weights[ln_start..ln_start + dim],
               &weights.classifier_weights[ln_start + dim..ln_start + 2*dim]);
    
    // GELU y dense
    let dense_w_start = lm_head_offset + 2*dim;
    let dense_b_start = dense_w_start + dim * dim;
    gelu(&mut state.hb, &state.xb);
    matmul(&mut state.logits, &state.hb,
           &weights.classifier_weights[dense_w_start..dense_w_start + dim*dim],
           &weights.classifier_weights[dense_b_start..dense_b_start + dim],
           dim, dim);
    
    // Final layer norm
    let final_ln_start = dense_b_start + dim;
    layer_norm(&mut state.xb, &state.logits,
               &weights.classifier_weights[final_ln_start..final_ln_start + dim],
               &weights.classifier_weights[final_ln_start + dim..final_ln_start + 2*dim]);
    
    // Final projection (tied)
    matmul(&mut state.logits, &state.xb,
           &weights.token_embedding_table,
           &weights.classifier_weights[final_ln_start + 2*dim..],  // decoder bias
           dim, config.vocab_size);
}
```

#### Tarea 2: Corregir la paridad de RoPE

**Validación contra PyTorch:**
```python
import torch
from transformers import EsmModel

model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
# Extraer pesos de atención y comparar con tabla de frecuencias
```

**Prueba unitaria requerida:**
```rust
#[test]
fn test_rope_parity_with_reference() {
    // Load reference rotary values from ONNX/PyTorch export
    let reference_cos = load_reference_cos();
    let reference_sin = load_reference_sin();
    
    let state = RunState::new(&config);
    
    for pos in 0..100 {
        for j in 0..half_head {
            assert!((state.rope_cos[pos * half_head + j] - reference_cos[pos * half_head + j]).abs() < 1e-6);
            assert!((state.rope_sin[pos * half_head + j] - reference_sin[pos * half_head + j]).abs() < 1e-6);
        }
    }
}
```

---

### 4.2 Prioridad Metodológica (MLM / Gibbs Sampling vs. Causal)

#### Tarea 1: Implementar modo MLM con token `<mask>`

**Especificación:**
1. Usar token ID 32 como `<mask>`
2. Iterar sobre toda la secuencia
3. Rellenar cada posición mask con el token de mayor probabilidad

**Algoritmo Gibbs estimado:**
```rust
pub fn generate_mlm_sequence(
    initial: &str, 
    weights: &EsmWeights, 
    config: &EsmConfig
) -> String {
    let mut sequence = initial.to_uppercase();
    let mask_token = 32usize;
    
    while sequence.contains('<') || sequence.contains('[') {
        // Encuentra posición mask
        // ...
    }
    
    // Implementar algoritmo de sustitución iterativa
    todo!()
}
```

#### Tarea 2: Implementar modo inpainting

Permite generar secuencias con regiones fijas y regiones enmascaradas:
```rust
pub struct InpaintRequest {
    prefix: String,
    mask_region: (usize, usize),  // [start, end) 
    suffix: String,
    iterations: usize,
}
```

---

### 4.3 Prioridad de Rendimiento Hardware

#### Tarea 1: Preparar interfaz para cuantización Q8_0

**Formato Q8_0 por bloques de 32:**
```rust
#[repr(C)]
#[derive(Clone, Copy)]
struct Q8_0Block {
    delta: f32,
    anchors: [i8; 16],  // centrados en 0
}

impl Q8_0Block {
    fn dequantize(&self) -> [f32; 16] {
        self.anchors.map(|a| self.delta * a as f32)
    }
}
```

**Tamaño para L3 cache de 3MB (Sandy Bridge i5-2520M):**

- Sin cuantizar: 7,511,809 * 4 = 30,047,236 bytes (~30 MB)
- Q8_0: 7,511,809 / 16 * (4 + 16) = ~11.7 MB por vista
- Q8_0 + solo pesos activos: ~7-8 MB en uso

**Planificación de caché:**
1. KV-Cache: 6 capas * 1024 pos * 320 dim * 2 = 3,932,160 bytes (~3.75 MB)
2. Con Q8_0, los pesos pueden cargarse bajo demanda

#### Tarea 2: Implementar transiciones AVX sin penalización

```rust
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_avx2_zeroupper(...) {
    // ... operaciones ...
    _mm256_zeroupper();  // Evita penalización Sandy Bridge
}
```

#### Tarea 3: Vectorización de `layer_norm`

Usar AVX para la normalización:
- Cálculo de media (suma vectorial)
- Cálculo de varianza (cuadrados vectoriales)
- Normalización (divisiones vectoriales)

---

## Anexos

### Anexo A: Especificación del vocabulario ESM-2

```
Token IDs:
  0: <cls>   - CLS token
  1: <pad>   - Padding
  2: <eos>   - End of sequence  
  3: <unk>   - Unknown
  4:  L      - Leucina
  5:  A      - Alanina
  6:  G      - Glicina
  7:  V      - Valina
  8:  S      - Serina
  9:  E      - Glutamata
  10: R      - Arginina
  11: T      - Treonina
  12: I      - Isoleucina
  13: D      - Ácido aspártico
  14: P      - Prolina
  15: K      - Lisina
  16: Q      - Glutamina
  17: N      - Asparagina
  18: F      - Fenilalanina
  19: Y      - Tirosina
  20: M      - Metionina
  21: H      - Histidina
  22: W      - Triptófano
  23: C      - Cisteína
  24: X      - Ambiguo
  25: B      - D/N ambiguos
  26: U      - Selenocisteína
  27: Z      - E/Q ambiguos
  28: O      - Pirrolisina
  29: .      - Punto
  30: -      - Guion
  31: ?      - No usado
  32: <mask> - Token de máscara MLM
```

### Anexo B: Tamaño de pesos esperado

```
Total classifier_weights: 102,849 floats = 411,396 bytes

Desglose:
  token_embedding_table: 33 * 320 = 10,560 floats
  ln_1_weight: 6 * 320 = 1,920 floats  
  ln_1_bias: 6 * 320 = 1,920 floats
  wq: 6 * 320 * 320 = 614,400 floats
  bq: 6 * 320 = 1,920 floats
  wk: 6 * 320 * 320 = 614,400 floats
  bk: 6 * 320 = 1,920 floats
  wv: 6 * 320 * 320 = 614,400 floats
  bv: 6 * 320 = 1,920 floats
  wo: 6 * 320 * 320 = 614,400 floats
  bo: 6 * 320 = 1,920 floats
  ln_2_weight: 6 * 320 = 1,920 floats
  ln_2_bias: 6 * 320 = 1,920 floats
  fc1_weight: 6 * 320 * 1280 = 2,457,600 floats
  fc1_bias: 6 * 1280 = 7,680 floats
  fc2_weight: 6 * 1280 * 320 = 2,457,600 floats
  fc2_bias: 6 * 320 = 1,920 floats
  ln_f_weight: 320 floats
  ln_f_bias: 320 floats
  classifier_weights: 102,849 floats

Total: 7,511,809 floats ≈ 30.04 MB
```

### Anexo C: Comparación de precisión numérica esperada

| Operación | ESM-2 Oficial | codon-infer | Error Estimado |
|-----------|----------------|-------------|----------------|
| LayerNorm | media/varianza exacta | media/varianza exacta | ~0% |
| GELU | erf() exacto | tanh() approx | 0.5-1% |
| RoPE freq | pow(10000, i/dim) | pow(10000, 2i/dim) | ~10-20% (error de índice) |
| Softmax | float32 | float32 | ~1e-6 |
| Attention | 1/sqrt(dk) después de RoPE | 1/sqrt(dk) después de RoPE | ~0% |

---

**Fin del informe**
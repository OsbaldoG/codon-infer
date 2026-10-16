# ESPECIFICACIÓN TÉCNICA MOTOR Q4_0: CODON-INFER (v1.6.0)

**Estado:** CONGELADA / INMUTABLE  
**Autor:** Orquestador y Arquitecto Principal (Rust de sistemas, pLM, SIMD x86)  
**Versión:** 1.6.0  
**Fecha:** 2026-10-15  

---

## 1. Visión General y Objetivos
Esta especificación define el diseño formal, los contratos binarios, las estructuras en memoria y las convenciones SIMD para el motor de cuantización y ejecución **Q4_0** en `codon-infer`.

### Restricciones Arquitecturales
1. **Zero External Dependencies**: Únicamente `std` y `std::arch` de Rust.
2. **Zero Allocations en el Hot Path**: Los kernels SIMD y el forward pass de Q4_0 no deben realizar allocations dinámicas en el bucle de inferencia.
3. **Compatibilidad Estricta**: FP32 y Q8_0 (incluyendo el archivo legacy sin cabecera `esm2_650m_q8_0.bin`) deben mantenerse 100% operativos sin regresiones numéricas ni de rendimiento.
4. **Determinismo y Fidelidad**: Coincidencia matemática exacta entre el kernel de referencia escalar y los backends vectorizados SIMD (AVX1, AVX2+FMA), con tolerancia relativa ≤ 10⁻⁵.

---

## 2. Definición del Bloque: `BlockQ4_0`

### 2.1 Representación en Memoria
El bloque fundamental almacena 32 pesos continuos cuantizados a 4 bits (nibbles) con una escala compartida `d` de 32 bits (IEEE-754 float):

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub d: f32,          // Delta / factor de escala (4 bytes, little-endian)
    pub qs: [u8; 16],    // 16 bytes = 32 valores empaquetados en nibbles de 4 bits
}

const _: () = assert!(core::mem::size_of::<BlockQ4_0>() == 20);
const _: () = assert!(core::mem::align_of::<BlockQ4_0>() == 4);
```

- **Tamaño por bloque**: 20 bytes exactos.
- **Tasa de compresión**: 32 floats (128 bytes) → 20 bytes = **6.4×** de compresión teórica sobre matrices cuantizadas (0.625 bytes por parámetro).
- **Alineación**: `align_of == 4`. Dado que la cabecera `CODQ` mide 64 bytes (múltiplo de 4 y de 64), los bloques en archivo residen naturalmente alineados para `from_raw_parts`.

### 2.2 Convención de Cuantización (llama.cpp standard)
Para un vector de 32 floats `X = [x_0, x_1, ..., x_31]`:
1. Se localiza el elemento `x_max` con mayor valor absoluto conservando su signo:
   ```text
   x_max = argmax_{x ∈ X} |x|   (conservando el signo de x)
   ```
2. Se calcula la escala `d`:
   ```text
   d = x_max / -8.0
   id = (d == 0.0) ? 0.0 : (1.0 / d)
   ```
3. Para cada elemento `x_i` (`i ∈ [0, 31]`):
   ```text
   q_i = clamp(round(x_i · id) + 8, 0, 15) ∈ [0, 15]
   ```

### 2.3 Empaquetado y Desempaquetado de Nibbles
Los 32 valores cuantizados `q_0 ... q_31` se empaquetan en `qs: [u8; 16]`:
- Para `j ∈ [0, 15]`:
  - **Nibble bajo** (bits `0..=3`): `q_j`
  - **Nibble alto** (bits `4..=7`): `q_{j + 16}`
  - Formato del byte:
    ```text
    qs[j] = (q_j & 0x0F) | ((q_{j+16} & 0x0F) << 4)
    ```

### 2.4 Decuantización
Para recuperar los valores continuos aproximados `x̂_i`:
- Para `j ∈ [0, 15]`:
  ```text
  x̂_j      = ((qs[j] & 0x0F) - 8) · d
  x̂_{j+16} = (((qs[j] >> 4) & 0x0F) - 8) · d
  ```

### 2.5 Cota de Error de Cuantización
Dado que el paso de cuantización es `|d|` y el redondeo es al entero más próximo:
```text
|x_i - x̂_i| ≤ |d| / 2.0 + 10⁻⁵
```
Para un bloque de ceros (`x_i = 0`), `d = 0`, `q_i = 8`, `qs[j] = 0x88`, `x̂_i = 0.0`.

### 2.6 Matrices Cuantizadas vs Matrices en FP32
Siguiendo la arquitectura probada y optimizada para fidelidad:
- **Cuantizadas a `BlockQ4_0`**:
  - `wq`, `wk`, `wv`, `wo` (atención en todas las capas)
  - `fc1_weight`, `fc2_weight` (feed-forward en todas las capas)
- **Conservadas en `f32` (FP32)**:
  - `token_embedding_table` (preservación de precisión en la tabla de entrada de tokens)
  - LayerNorm weights y biases (`ln_1`, `ln_2`, `ln_f`)
  - Biases de proyección (`bq`, `bk`, `bv`, `bo`, `fc1_bias`, `fc2_bias`)
  - `classifier_weights` (LM head)

---

## 3. Cabecera Universal de Archivo Cuantizado: `CODQ` (64 Bytes)

Para eliminar cualquier ambigüedad de detección por tamaño entre escalas y precisiones (ej. 650M-Q4 ~410MB colisiona en orden de magnitud con 150M-FP32 ~592MB o 150M-Q8 ~166MB; 3B-Q4 colisiona con 650M-Q8):

### 3.1 Estructura Binaria (Little-Endian)
```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodqHeader {
    pub magic: [u8; 4],       // Offset  0..4:  b"CODQ" (0x43, 0x4F, 0x44, 0x51)
    pub version: u32,         // Offset  4..8:  1 (versión del formato de cabecera)
    pub dtype: u32,           // Offset  8..12: 0=FP32, 1=Q8_0, 2=Q4_0
    pub arch: u32,            // Offset 12..16: 0=8M, 1=35M, 2=150M, 3=650M, 4=3B
    pub n_layers: u32,        // Offset 16..20: Número de capas Transformer
    pub dim: u32,             // Offset 20..24: Dimensión del modelo (d_model)
    pub hidden_dim: u32,      // Offset 24..28: Dimensión intermedia FFN
    pub vocab_size: u32,      // Offset 28..32: Tamaño de vocabulario (33)
    pub reserved: [u8; 32],   // Offset 32..64: Padding relleno con 0x00
}

const _: () = assert!(core::mem::size_of::<CodqHeader>() == 64);
const _: () = assert!(core::mem::align_of::<CodqHeader>() == 4);
```

### 3.2 Constantes de Tipo y Arquitectura
- `CODQ_MAGIC: [u8; 4] = *b"CODQ";`
- `CODQ_VERSION: u32 = 1;`
- `DTYPE_FP32: u32 = 0;`
- `DTYPE_Q8_0: u32 = 1;`
- `DTYPE_Q4_0: u32 = 2;`
- `ARCH_8M: u32 = 0;`
- `ARCH_35M: u32 = 1;`
- `ARCH_150M: u32 = 2;`
- `ARCH_650M: u32 = 3;`
- `ARCH_3B: u32 = 4;`

### 3.3 Protocolo de Detección y Carga
1. **Inspección de Cabecera**:
   Al abrir el archivo, se leen los primeros 4 bytes.
   - Si `&magic == b"CODQ"`, se leen los 64 bytes de `CodqHeader`.
   - Se valida `version == 1`.
   - Se valida la coherencia entre `arch`, `n_layers`, `dim`, `hidden_dim` y `vocab_size` contra las configuraciones canónicas de `EsmConfig`.
   - Se procesa el archivo con el `dtype` especificado.
2. **Fallback Legacy**:
   - Si los primeros 4 bytes NO son `b"CODQ"`, se activa `EsmConfig::detect_from_file_len(file_len)` que soporta exclusivamente binarios FP32 canónicos y el binario legacy `esm2_650m_q8_0.bin`.

---

## 4. Especificación de Kernels SIMD y Concurrencia

### 4.1 Firmas Canónicas
```rust
pub fn matmul_q4(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn matmul_q4_avx1(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_q4_avx2_fma(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
);

pub fn matmul_q4_simd(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
);

pub fn matmul_q4_parallel(
    out: &mut [f32],
    x: &[f32],
    w: &[BlockQ4_0],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
    num_threads: usize,
);
```

### 4.2 Algoritmo AVX2 + FMA
Por cada bloque de 32 elementos (`w` ∈ `BlockQ4_0`):
1. Cargar 16 bytes: `raw16 = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i)`.
2. Aislar nibbles:
   - `mask_low = _mm_set1_epi8(0x0F)`
   - `c8 = _mm_set1_epi8(8)`
   - `lo_i8 = _mm_sub_epi8(_mm_and_si128(raw16, mask_low), c8)`
   - `hi_i8 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw16, 4), mask_low), c8)`
3. Expandir a 4 vectores de 8 flotantes (`__m256`):
   - Elementos 0..7: `lo_i8` → `_mm256_cvtepi8_epi32` → `_mm256_cvtepi32_ps`
   - Elementos 8..15: `_mm_srli_si128(lo_i8, 8)` → `_mm256_cvtepi8_epi32` → `_mm256_cvtepi32_ps`
   - Elementos 16..23: `hi_i8` → `_mm256_cvtepi8_epi32` → `_mm256_cvtepi32_ps`
   - Elementos 24..31: `_mm_srli_si128(hi_i8, 8)` → `_mm256_cvtepi8_epi32` → `_mm256_cvtepi32_ps`
4. Acumular mediante FMA:
   - `acc = _mm256_fmadd_ps(vq, vx, acc)`
5. Reducción horizontal:
   - `hsum256_ps_avx(acc) * block.d`
   - Sumar bias final: `out[i] = row_sum + bias[i]`

### 4.3 Algoritmo AVX1 (Fallback x86_64)
- AVX1 carece de instrucciones vectoriales de enteros de 256 bits (`_mm256_cvtepi8_epi32`).
- Se desempaqueta con SSE (`__m128i` de 4 enteros → `_mm_cvtepi8_epi32` → `_mm_cvtepi32_ps`), se combina con `_mm256_set_m128(ps_hi, ps_lo)`, y se ejecuta `_mm256_mul_ps` + `_mm256_add_ps`.

### 4.4 Paralelismo por Filas
- Si `num_threads > 1` y `out_dim >= 32`, se utiliza `get_global_pool().run_parallel` particionando uniformemente las filas de salida `out_dim` entre los hilos asignados, garantizando nula contención y cero sincronización mutex en el hot path.

---

## 5. Mapeo de Regiones y Asignación a Subagentes

Para garantizar la independencia de trabajo y evitar colisiones de git:

| Componente | Líneas aproximadas en `src/lib.rs` / Archivos | Subagente Responsable | Entregables |
|---|---|---|---|
| **Estructura BlockQ4_0, CodqHeader, Serialización y Loader** | `src/lib.rs` líneas ~320-560 (bloque Q4 data/loader) | **Subagente A** (Format & Loader Engineer) | Implementación de `BlockQ4_0`, `CodqHeader`, `quantize_slice_to_q4`, `EsmWeightsQ4`, `EsmWeights::quantize_q4`, `save_to_file`, `from_file`, tests unitarios de formato |
| **Kernels SIMD (Escalar, AVX1, AVX2+FMA, SIMD Dispatch, Paralelo)** | `src/lib.rs` líneas ~1150-1360 (bloque Q4 kernels) | **Subagente B** (SIMD Kernel Engineer) | `matmul_q4`, `matmul_q4_avx1`, `matmul_q4_avx2_fma`, `matmul_q4_simd`, `matmul_q4_parallel`, microtests de paridad con tolerancia ≤ 10⁻⁵ |
| **Integración de Inferencia Q4 (Forward & LM Head)** | `src/lib.rs` líneas ~1750-2250, `src/pipeline.rs` | **Orquestador Principal** (Fase 2) | Conexión de `forward_q4`, `forward_bidirectional_q4`, `project_lm_head_q4`, `ModelWeightsRef::Q4` |
| **CLI `quantize q4_0`, Generación de Pesos y Validación Multi-Escala** | `src/main.rs`, `src/benchmark.rs` | **Subagente C** (CLI & Validation Engineer) | Soporte CLI `quantize q4_0`, generación binarios 8M/35M/150M/650M, métricas de compresión, coseno y latencia |
| **Auditoría de Seguridad, Unsafe y Zero-Allocation** | Todo el repositorio (solo lectura) | **Subagente D** (Reviewer / Auditor) | Informe de seguridad `unsafe`, límites de punteros, balance de memoria, clippy y regresiones |

---
*Fin de especificación congelada.*

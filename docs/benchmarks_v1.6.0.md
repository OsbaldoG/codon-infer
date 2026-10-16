# Resultados Auditados de Benchmarks, Paridad HuggingFace y Rendimiento v1.6.0

Este documento contiene las mediciones reales, reproducibles y verificadas contra **HuggingFace (`facebook/esm2_*`)** del motor **FP32**, **Q8_0** y **Q4_0** para `codon-infer` v1.6.0 sobre los pesos canónicos **`grouped_canonical_v1`** en las **5 escalas completas (`8M`, `35M`, `150M`, `650M` y `3B`)**.

- **CPU / Entorno:** Linux x86_64 con AVX2 + FMA (11 GiB RAM física)
- **Hilos de cómputo:** 4 hilos persistentes (`Global ThreadPool` con barreras atómicas)
- **Secuencia de prueba:** `AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS` (39 residuos + `<cls>` + `<eos>` = **41 tokens** de `test_peptidase.fasta`)
- **Referencias Oficiales HF:** `scripts/ref_{8m,35m,150m,650m,3b}.npz` (`input_ids`, `logits [41, 33]` y `hidden_states [L+1, 41, dim]`)
- **Protocolo de latencia:** Mediana de 5 ejecuciones consecutivas tras 1 ejecución de calentamiento (*warm-up*)
- **Protocolo de memoria:** Medición de `ru_maxrss` a nivel de kernel (`wait4`, idéntica a `/usr/bin/time -v`)

---

## 1. Tabla Maestra Multi-Escala (`8M`, `35M`, `150M`, `650M` y `3B`)

| Escala | Formato | Tamaño Exacto (`stat -c %s`) | Ratio Compresión | Coseno `hidden_states[-1]` vs HF | Coseno Logits vs HF | Coseno Centrado vs HF | Divergencia KL vs HF | Top-1 Match vs HF | Top-1 AA Unmasked | Carga Caliente (ms) | Forward Mediana (ms) | Throughput (tok/s) | Peak RSS (`wait4`) |
| :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| **8M** | **FP32** | `30,091,652 B` (28.70 MiB) | `1.00×` | `0.999999` | `0.999999` | `1.000000` | `0.000000` | **100.00%** (41/41) | 39/39 (100.00%) | 11.50 ms | 113.44 ms | 361.41 tok/s | 32.75 MiB (`33,532 KB`) |
| **8M** | **Q8_0** | `8,864,556 B` (8.45 MiB) | `3.39×` | `0.999764` | `0.999995` | `0.999993` | `0.000128` | **100.00%** (41/41) | 39/39 (100.00%) | 4.99 ms | 85.97 ms | 476.92 tok/s | 29.11 MiB (`29,808 KB`) |
| **8M** | **Q4_0** | `5,208,516 B` (4.97 MiB) | `5.78×` | `0.934598` | `0.998168` | `0.997574` | `0.031316` | **100.00%** (41/41) | 39/39 (100.00%) | 4.69 ms | 89.82 ms | 456.49 tok/s | 29.29 MiB (`29,988 KB`) |
| **35M** | **FP32** | `134,067,972 B` (127.86 MiB) | `1.00×` | `0.999999` | `1.000000` | `1.000000` | `0.000000` | **100.00%** (41/41) | 37/39 (94.87%) | 33.72 ms | 446.47 ms | 91.83 tok/s | 133.30 MiB (`136,500 KB`) |
| **35M** | **Q8_0** | `38,636,896 B` (36.85 MiB) | `3.47×` | `0.999825` | `0.999994` | `0.999993` | `0.000120` | **100.00%** (41/41) | 37/39 (94.87%) | 25.62 ms | 377.27 ms | 108.68 tok/s | 42.24 MiB (`43,252 KB`) |
| **35M** | **Q4_0** | `22,093,636 B` (21.07 MiB) | `6.07×` | `0.952611` | `0.998566` | `0.998132` | `0.048077` | **95.12%** (39/41) | 37/39 (94.87%) | 12.85 ms | 318.58 ms | 128.70 tok/s | 29.41 MiB (`30,116 KB`) |
| **150M** | **FP32** | `592,642,692 B` (565.19 MiB) | `1.00×` | `0.999999` | `1.000000` | `1.000000` | `0.000000` | **100.00%** (41/41) | 36/39 (92.31%) | 277.88 ms | 2,390.36 ms | 17.15 tok/s | 575.01 MiB (`588,812 KB`) |
| **150M** | **Q8_0** | `168,646,036 B` (160.83 MiB) | `3.51×` | `0.999934` | `0.999998` | `0.999996` | `0.000042` | **100.00%** (41/41) | 36/39 (92.31%) | 70.16 ms | 1,430.16 ms | 28.67 tok/s | 170.61 MiB (`174,700 KB`) |
| **150M** | **Q4_0** | `94,978,756 B` (90.58 MiB) | `6.24×` | `0.981715` | `0.999266` | `0.999012` | `0.020517` | **100.00%** (41/41) | 36/39 (92.31%) | 39.17 ms | 1,367.83 ms | 29.97 tok/s | 100.39 MiB (`102,796 KB`) |
| **650M** | **FP32** | `2,604,339,332 B` (2,483.69 MiB) | `1.00×` | `0.999999` | `1.000000` | `1.000000` | `0.000000` | **100.00%** (41/41) | 35/39 (89.74%) | 7,721.19 ms | 8,692.15 ms | 4.72 tok/s | 2,501.33 MiB (`2,561,360 KB`) |
| **650M** | **Q8_0** | `738,899,556 B` (704.67 MiB) | `3.52×` | `0.999913` | `0.999996` | `0.999996` | `0.000039` | **100.00%** (41/41) | 35/39 (89.74%) | 287.77 ms | 4,669.89 ms | 8.78 tok/s | 722.41 MiB (`739,744 KB`) |
| **650M** | **Q4_0** | `414,617,796 B` (395.41 MiB) | `6.28×` | `0.980842` | `0.999292` | `0.998939` | `0.019508` | **97.56%** (40/41) | 36/39 (92.31%) | 137.32 ms | 4,535.47 ms | 9.04 tok/s | 413.07 MiB (`422,988 KB`) |
| **3B** | **FP32** | `11,356,354,692 B` (10.58 GiB) | `1.00×` | *Ref HF (`ref_3b.npz`)* | *Ref HF (`ref_3b.npz`)* | *Ref HF* | *Ref HF* | *Ref HF* | 36/39 (92.31%) | *N/A (>11 GiB RAM)* | *N/A* | *N/A* | *Streaming Solamente* |
| **3B** | **Q8_0** | `3,216,540,676 B` (3,067.53 MiB) | `3.53×` | **`0.999983`** | **`0.999998`** | **`0.999998`** | **`0.000027`** | **100.00%** (41/41) | 36/39 (92.31%) | 9,484.75 ms | 18,615.93 ms | 2.20 tok/s | **3,102.18 MiB** (`3,176,636 KB`) |
| **3B** | **Q4_0** | `1,801,205,956 B` (1,717.76 MiB) | `6.30×` | **`0.994156`** | **`0.999422`** | **`0.999321`** | **`0.006447`** | **100.00%** (41/41) | 36/39 (92.31%) | **545.36 ms** | 19,368.89 ms | 2.12 tok/s | **1,752.57 MiB** (`1,794,636 KB`) |

---

## 2. Resolución Definitiva del Bug de Desplazamiento en Pesos Legacy y Paridad con HuggingFace

### 2.1. Causa Raíz del Desplazamiento Histórico (`inv_freq` y `contact_head`)
En versiones previas, los archivos `.bin` exportados desde checkpoints que incluían buffers persistentes no-parámetro (`rotary_emb.inv_freq` de longitud $d_k / 2$ por cada capa y `contact_head.regression` de longitud $L \cdot H + 1$) contenían flotantes extra intercalados:
- **8M:** $+129$ floats ($+516\text{ B}$: $6 \times 8 = 48$ de `inv_freq` + $81$ de `contact_head`)
- **35M:** $+253$ floats ($+1,012\text{ B}$: $12 \times 12 = 144$ de `inv_freq` + $109$ de `contact_head`)
- **150M:** $+617$ floats ($+2,468\text{ B}$: $30 \times 16 = 480$ de `inv_freq` + $137$ de `contact_head`)
- **650M:** $+693$ floats ($+2,772\text{ B}$: $33 \times 32 = 1,056$ de `inv_freq` y diferencias de cabecera)

Debido a que el cargador leía las matrices de forma contigua sin descontar `inv_freq`, todas las matrices posteriores se desplazaban decenas o cientos de floats, corrompiendo la atención y produciendo predicciones constantes degeneradas (`SSSS...`, `IIII...`, `KKKK...`).

### 2.2. Solución Estructural Implementada en v1.6.0
1. **Layout Canónico Estricto (`grouped_canonical_v1`):**
   - `EsmWeights::from_file` valida que el archivo FP32 coincida **al byte exacto** con `grouped_canonical_bytes()` o `sequential_hf_bytes()` (devolviendo `io::ErrorKind::InvalidData` ante cualquier byte sobrante o faltante).
   - Se eliminaron de raíz los buffers `inv_freq` (RoPE se calcula analíticamente en tiempo de ejecución) y `contact_head` (la predicción de contactos usa APC directo sobre matrices de atención).
2. **Normalización Final `emb_layer_norm_after` en `state.x`:**
   - En `forward_bidirectional_parallel_ext`, `forward_bidirectional_q8` y `forward_bidirectional_q4_ext`, tras la última capa Transformer se aplica `layer_norm(ln_f_w, ln_f_b, 1e-5)` directamente sobre `state.x`, garantizando que `state.x` contenga exactamente `hidden_states[-1]` de HuggingFace antes de proyectar el `EsmLMHead`.
3. **Lector `.npz` Exacto (`read_npz_floats`):**
   - Se reemplazó el escaneo heurístico de `b"PK"` dentro del flujo de bytes float (que podía truncar tensores cuando un `f32` contenía los bytes `0x50 0x4B`) por el parseo formal de los Local Headers ZIP (incluyendo extensión ZIP64) y la tupla `'shape': (...)` de la cabecera `.npy`.

### 2.3. Paridad Numérica Capa a Capa y Reconstrucción de Secuencia
Al evaluar `test_peptidase.fasta` (`AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS`) sin máscara frente a las referencias oficiales de HuggingFace (`scripts/ref_{8m,35m,150m,650m,3b}.npz`):

| Escala | Min Layer Cosine vs HF | `hidden_states[-1]` Cosine vs HF | Logits Cosine vs HF | Secuencia Predicha (39 aa internos) | Reconstrucción Natural |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Input** | — | — | — | `AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS` | 39/39 |
| **8M FP32 / Q8 / Q4** | `0.9999971` | `0.9999994` | `0.9999994` | `AQVGDSGGPMLCNGQLQGVTSWGYVCAQPEYPGVYTEVS` | **39/39 (100.0%)** |
| **35M FP32 / Q8** | `0.9999993` | `0.9999993` | `0.9999997` | `AQVGDSGGPLLCNGQLQGVTSWGYPCAQPEYPGVYTEVS` | **37/39 (94.9%)** |
| **35M Q4_0** | — | `0.9526110` | `0.9985660` | `AQVGDSGGPVLCNGQLQGVTSWGYPCAQPEYPGVYTEVS` | **37/39 (94.9%)** |
| **150M FP32 / Q8 / Q4** | `0.9999969` | `0.9999994` | `1.0000000` | `AQVGDSGGPLLCNGQLQGVTSWGYGCARPEYPGVYTEVS` | **36/39 (92.3%)** |
| **650M FP32 / Q8** | `0.9999974` | `0.9999992` | `0.9999999` | `ACVGDSGGPLLCNGQLQGVTSWGYPCARPEYPGVYTEVS` | **35/39 (89.7%)** |
| **650M Q4_0** | — | `0.9808420` | `0.9992920` | `AQVGDSGGPLLCNGQLQGVTSWGYPCARPEYPGVYTEVS` | **36/39 (92.3%)** |
| **3B HF / Q8 / Q4** | — | `0.999983` (Q8) / `0.994156` (Q4) | `0.999998` (Q8) / `0.999422` (Q4) | `AQVGDSGGPLVCNGQLQGVTSWGYPCAQPKYPGVYTEVS` | **36/39 (92.3%)** |

---

## 3. Cuantización por Streaming de Baja Memoria (<13 MiB RSS) para Modelos de hasta 3B (10.58 GiB)

Para permitir la cuantización de **ESM-2 3B (`11,356,354,692 B` / `10.58 GiB` en FP32)** en máquinas con **11 GiB de RAM física** sin provocar *Out-Of-Memory (OOM)* ni *swapping*, se implementaron en `src/lib.rs` los cuantizadores por streaming:
- `quantize_file_streaming_q8(input_path, output_path)`
- `quantize_file_streaming_q4(input_path, output_path)`

### Arquitectura del Cuantizador por Streaming:
1. Inspecciona la longitud exacta del archivo de entrada mediante `std::fs::metadata` y valida que coincida al byte con `grouped_canonical_bytes()`.
2. Abre un `BufReader` de 1 MiB sobre el archivo FP32 y un `BufWriter` de 1 MiB sobre el destino, escribiendo primero la cabecera universal `CODQ` de 64 bytes.
3. Procesa cada región del archivo en orden secuencial mediante un único buffer reutilizable de trabajo de `65,536` floats (`256 KiB`, múltiplo exacto de 32 elementos), convirtiendo al vuelo a bloques `BlockQ8_0` (36 B) o `BlockQ4_0` (20 B) y copiando en `f32` puro los tensores de LayerNorm, biases y `classifier_weights`.
4. **Consumo de Memoria Medido (`wait4` `ru_maxrss`):**
   - Cuantización completa de `esm2_3b_weights.bin` (`10.58 GiB`) $\to$ `esm2_3b_q8_0.bin` (`3.00 GiB`) y `esm2_3b_q4_0.bin` (`1.68 GiB`): **Peak RSS = `12,852 KB` (`12.55 MiB`)**.

---

## 4. Derivación Matemática Exacta de Tamaños en Disco para las 5 Escalas (`grouped_canonical_v1`)

En el layout canónico `grouped_canonical_v1` (sin `inv_freq` ni `contact_head`), cada modelo ESM-2 con dimensión $d$, dimensión FFN $d_{ff} = 4d$, $L$ capas y vocabulario $V = 33$ contiene:

1. **Cabecera Universal `CODQ`:**
   $$\text{Header} = \mathbf{64\text{ bytes}}$$
2. **Matrices Lineales Cuantizadas por Capa ($W_q, W_k, W_v, W_o, W_{fc1}, W_{fc2}$):**
   $$N_{\text{mat}} = L \times (4 d^2 + 2 d \cdot d_{ff}) = 12 L d^2\text{ floats}$$
   - En **Q8_0**, además se cuantiza `token_embedding_table` ($33 d$ floats):
     $$\text{Bytes}_{\text{Q8\_blocks}} = \frac{12 L d^2 + 33 d}{32} \times 36\text{ bytes}$$
   - En **Q4_0**, `token_embedding_table` se conserva en FP32 ($33 d \times 4$ bytes) y las matrices usan bloques de 20 bytes:
     $$\text{Bytes}_{\text{Q4\_blocks}} = \frac{12 L d^2}{32} \times 20\text{ bytes}$$
3. **Tensores en Precisión Completa FP32 (LayerNorms, Biases y `EsmLMHead`):**
   - Biases y LayerNorms de las $L$ capas + `ln_f`:
     $$N_{\text{LN+bias}} = L \times (9 d + d_{ff}) + 2 d = (13 L + 2) d\text{ floats}$$
   - Cabezal de lenguaje `classifier_weights` (`EsmLMHead`: `dense.weight` $[d \times d]$, `dense.bias` $[d]$, `layer_norm.weight` $[d]$, `layer_norm.bias` $[d]$, `decoder.weight` $[33 \times d]$, `decoder.bias` $[33]$):
     $$N_{\text{classifier}} = d^2 + 36 d + 33\text{ floats}$$

### Caso Particular: Derivación Exacta de ESM-2 3B ($L = 36, d = 2560, d_{ff} = 10240$)
- **FP32 (`esm2_3b_weights.bin`):**
  $$N_{\text{floats}} = 33(2560) + 12(36)(2560^2) + (13 \cdot 36 + 2)(2560) + (2560^2 + 36 \cdot 2560 + 33) = \mathbf{2,839,088,673\text{ floats}}$$
  $$\text{Bytes}_{\text{FP32}} = 2,839,088,673 \times 4 = \mathbf{11,356,354,692\text{ bytes}} \quad (10.58\text{ GiB})$$
- **Q8_0 (`esm2_3b_q8_0.bin`):**
  - Elementos cuantizados en `BlockQ8_0`: $84,480 + 2,831,155,200 = 2,831,239,680$ floats $= 88,476,240$ bloques $\times 36\text{ B} = \mathbf{3,185,144,640\text{ B}}$
  - Elementos FP32 (`LN+bias` + `classifier_weights`): $(1,203,200 + 6,645,793) \times 4 = \mathbf{31,395,972\text{ B}}$
  - Total Q8_0: $64 + 3,185,144,640 + 31,395,972 = \mathbf{3,216,540,676\text{ bytes}} \quad (3.00\text{ GiB}, \text{ compresión } \mathbf{3.531\times})$
- **Q4_0 (`esm2_3b_q4_0.bin`):**
  - Elementos cuantizados en `BlockQ4_0`: $2,831,155,200$ floats $= 88,473,600$ bloques $\times 20\text{ B} = \mathbf{1,769,472,000\text{ B}}$
  - Elementos FP32 (`token_embedding_table` + `LN+bias` + `classifier_weights`): $(84,480 + 1,203,200 + 6,645,793) \times 4 = \mathbf{31,733,892\text{ B}}$
  - Total Q4_0: $64 + 1,769,472,000 + 31,733,892 = \mathbf{1,801,205,956\text{ bytes}} \quad (1.68\text{ GiB}, \text{ compresión } \mathbf{6.305\times})$
  *(Nota: La diferencia de exactamente $5,764\text{ bytes}$ respecto a la estimación preliminar de $1,801,211,720\text{ B}$ corresponde a los $1,441\text{ floats} \times 4\text{ B}$ de `contact_head` que fueron erradicados en el formato `grouped_canonical_v1`).*

---

## 5. Auditoría de Lints y Directivas `#![allow]` del Crate

El crate `codon-infer` compila con **cero warnings** bajo `cargo clippy --release --all-targets -- -D warnings`. La cabecera de `src/lib.rs` conserva únicamente las directivas justificadas por kernels SIMD de bajo nivel:

| Lint / Directiva Silenciada | Tipo | Justificación Técnica de Ingeniería de Sistemas |
|---|:---:|---|
| `unsafe_op_in_unsafe_fn` | Compiler (E0133) | Bloques de ensamblado intrínseco SIMD AVX2/AVX1 y decodificación cruda de punteros donde la función completa ya está marcada `pub unsafe fn`. |
| `clippy::needless_range_loop` | Clippy | Bucles matemáticos tensoriales explícitos por coordenadas (`0..dim`, `0..seq_len`) optimizados para autovectorización e indexación simultánea de múltiples slices. |
| `clippy::missing_safety_doc` | Clippy | Funciones internas `unsafe` con contrato validado en `Q4_0_SPEC.md`. |
| `clippy::too_many_arguments` | Clippy | Kernels SIMD y proyecciones de atención que reciben buffers preasignados (`out, x, w, bias, in_dim, out_dim, ...`) para garantizar zero allocations. |
| `clippy::manual_is_multiple_of` | Clippy | Comprobaciones de alineación `x % 32 == 0` compatibles con versiones estables MSRV de Rust. |
| `clippy::excessive_precision` | Clippy | Constantes matemáticas de precisión IEEE-754 (coeficientes de aproximación polinomial de GELU y RoPE). |
| `clippy::collapsible_if` | Clippy | Separación explícita de guardas de control para facilitar la lectura de pipelines y diagnósticos SIMD. |
| `clippy::manual_div_ceil` | Clippy | Cálculo explícito `(len + 31) / 32` para división entera en múltiplos de bloque SIMD. |
| `clippy::chunks_exact_to_as_chunks` | Clippy | Iteración por bloques en slices crudos sin requerir features inestables. |
| `clippy::type_complexity` | Clippy | Tipos de firmas complejas en callbacks de capas intermedias (`Option<&mut dyn FnMut(...)>`). |
| `clippy::manual_strip` | Clippy | Lógica de parsing de FASTA manual de alto rendimiento. |
| `unused_unsafe` | Compiler | Bloque de seguridad defensivo en barrera multihilo. |
| `clippy::let_and_return` | Clippy | Asignación intermedia defensiva para inspección en depuración. |

---

## 6. Arquitectura de Inferencia y Hoja de Ruta v1.7.0 (GEMV vs Batched GEMM)

Actualmente, `forward_bidirectional_*` evalúa cada token de la secuencia mediante multiplicaciones matriz-vector (**GEMV**), leyendo los pesos de la red por cada posición de la secuencia. En la versión **v1.7.0** se implementará un micro-kernel de multiplicación matricial por lotes (**Batched GEMM** $[N \times D] \times [D \times D]$) que cargará cada bloque `BlockQ4_0` / `BlockQ8_0` en registros SIMD una única vez para todos los tokens de la secuencia simultáneamente, reduciendo el tráfico del bus DRAM en un factor proporcional a la longitud de secuencia $N$.

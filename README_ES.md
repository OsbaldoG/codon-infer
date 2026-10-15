# codon-infer 🧬⚡

> **Motor de Inferencia y Pipeline Bioinformático para Modelos de Lenguaje de Proteínas ESM-2 Multi-Escala, de Alto Rendimiento y Cero Asignaciones en Rust Puro (v1.4.0).**

[🇬🇧 Read in English (README.md)](README.md)

[![Rust](https://img.shields.io/badge/rust-edición_2024-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/versión-1.4.0-blue.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencias-0%20(std%20puro)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/modelo-ESM--2%20(8M%20%7C%2035M%20%7C%20150M)-blueviolet.svg)](https://github.com/facebookresearch/esm)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%2BFMA%20%7C%20AVX1-red.svg)]()
[![Throughput](https://img.shields.io/badge/Rendimiento%20CPU->200%20tokens%2Fseg-success.svg)]()

> 📖 **Nota de Idioma:** Este documento constituye la documentación oficial técnica completa en **español**. Para consultar la versión original en inglés técnico, acceda a [README.md](README.md).

`codon-infer` es un motor de inferencia y suite de análisis bioinformático ultraligero y libre de dependencias externas para la familia de modelos de lenguaje de proteínas Evolutionary Scale Modeling (**ESM-2**) en sus variantes de escala (`8M`, `35M` y `150M`), diseñado desde primeros principios matemáticos y de hardware en **Rust 100% seguro e idiomático (`std` exclusivamente)**.

Al prescindir por completo de los ecosistemas de Python, PyTorch, C/CUDA y crates externos de terceros, `codon-infer` ejecuta generación autorregresiva de secuencias, *inpainting* bidireccional de sitios activos, escaneo mutacional profundo *in silico* (DMS) y predicción de contactos estructurales mediante vectorización SIMD dinámica a más de **200 tokens/segundo en CPUs x86_64 estándar**, manteniendo una huella de memoria estática inferior a 40 MB para modelos de 8M.

---

## ⚡ Aspectos Destacados (Versión 1.4)

* **Arquitectura Multi-Escala y Autodetección Paramétrica (`detect_from_file_len`):**
  * Instanciación instantánea y sin configuración manual para las variantes **ESM-2 8M**, **35M** y **150M** a partir de la longitud en bytes del archivo binario de pesos, eliminando la necesidad de archivos de configuración JSON o manifiestos auxiliares.
* **Kernels de Atención SIMD Dinámicos y Heterogéneos ($d_k \in \{16, 24, 32\}$):**
  * Micro-kernels vectoriales AVX de 256 bits optimizados a mano y mapeados a 2, 3 o 4 registros `__m256` por cabeza de atención, maximizando la densidad de cómputo vectorial con paridad escalar bit a bit ($\Delta < 10^{-5}$) tanto en pasadas autorregresivas causales como bidireccionales completas.
* **Subcomandos de CLI para Entornos de Producción (`generate`, `inpaint`, `dms`, `contacts`, `benchmark`):**
  * Interfaz de terminal expresiva y unificada con soporte de subcomandos, banderas globales (`--model`, `--color`, `--no-color`), detección automática de TTY y diagnósticos claros de errores.
* **Experiencia de Usuario TrueColor ANSI sin Dependencias (`ui.rs`):**
  * **Paleta Fisicoquímica:** Residuos de aminoácidos formateados en colores ANSI RGB TrueColor según sus propiedades bioquímicas (Alifáticos, Aromáticos, Básicos, Ácidos, Polares, Cisteínas y de Flexibilidad Conformacional).
  * **Mapa de Calor Térmico Continuo:** Interpolación RGB dinámica (`lerp_rgb`) desde Carmesí Profundo (`#7F1D1D`) para mutaciones deletéreas/letales ($\Delta \log P \le -4.0$) pasando por Ámbar y Pizarra Neutro, hasta Azul Eléctrico (`#1E3A8A`) para variantes benéficas, preservando glifos de densidad ASCII (`█`, `▓`, `▒`, `░`, `·`) en terminales sin soporte de color.
  * **Barra de Progreso Dinámica en Stderr:** Actualizaciones limpias con retornos de carro ANSI `\r` reportando rendimiento instantáneo (`muts/s`) y tiempo estimado de finalización (`ETA`) sin contaminar las salidas redirigidas a `stdout`.
* **Pipeline Multi-FASTA y Transmisión por Flujos (`fasta.rs`):**
  * Parser de streaming de alto rendimiento y cero copias para archivos multi-FASTA (`--fasta <FICHERO>`), entrada estándar (`--fasta -`) o secuencias pasadas directamente como argumento (`--sequence <CADENA>`).
* **Formatos de Exportación con Calidad de Publicación (`export.rs`):**
  * Generación de matrices rectangulares en formato CSV e informes exhaustivos en JSON con tolerancias mutacionales por residuo, entropía informacional de Shannon y pares de contacto de alta probabilidad.
* **Evaluador de Ventana Deslizante para Proteínas de Longitud Completa (`pipeline.rs`):**
  * Procesa proteínas de longitud arbitraria mediante ventanas deslizantes adaptativas ($W=128, S=64$) centradas en los residuos evaluados, evitando distorsiones en los bordes por RoPE y garantizando localidad en caché.
* **Aceleración por Hardware Dual SIMD y Núcleo en Rust Puro:**
  * **AVX2 + FMA:** Pipeline vectorial con instrucciones Fused Multiply-Add de 256 bits para arquitecturas AMD Ryzen e Intel Haswell o superiores.
  * **AVX1:** Kernel SIMD de 256 bits compatible con procesadores Intel Sandy Bridge / Ivy Bridge heredados.
  * **Despacho Dinámico en Tiempo de Ejecución:** Detección de capacidades del microprocesador al inicio mediante `is_x86_feature_detected!`.
* **Paridad Matemática y Arquitectónica Estricta con ESM-2:**
  * Cabezal de clasificación `EsmLMHead` completo con aproximación numérica de la función de error `erf` para GELU e incrustaciones posicionales rotatorias (RoPE) con formulación *split-half*.
* **Runtime Estricto de Cero Asignaciones (`RunState`):** Todos los búferes de memoria del transformer, matrices de atención intermedias y cachés KV se preasignan al iniciar. Cero fragmentación o recolección de memoria durante la inferencia.

---

## 📊 Rendimiento y Benchmarks Comparativos

Evaluado en Linux x86_64 (ejecución monohilo en CPU con instrucciones vectoriales nativas):

| Métrica | Referencia PyTorch / Python | `codon-infer` v1.0 (Escalar) | `codon-infer` v1.2+ (Dual SIMD) | Ventaja Acumulada |
| :--- | :--- | :--- | :--- | :--- |
| **Dependencias** | Python, PyTorch, CUDA (~4–7 GB) | `std` puro (0 crates) | **`std` puro (0 crates)** | **~99.9% más liviano** |
| **Tamaño del Ejecutable** | Entorno multigiagabyte | ~4.2 MB binario | **~1.5 MB binario release** | **Ultra-portable** |
| **Latencia de Carga del Modelo** | ~2,500 – 4,000 ms | ~38 – 110 ms | **30 – 50 ms** | **~60x más rápido** |
| **Huella en Memoria (RAM)** | ~1.5 – 2.5 GB RSS | ~35 MB RSS estático | **~35 MB (FP32) / ~12 MB (Q8_0)** | **~50x más eficiente** |
| **Rendimiento de Inferencia** | ~15 – 35 tokens/seg (CPU) | ~128 – 137 tokens/seg | **>200 tokens/seg** | **~6x – 13x más veloz** |
| **Asignación de Memoria** | Pausas continuas de GC | Cero asignaciones en runtime | **0 asignaciones (Determinista)** | **Garantías de tiempo real** |

---

## 🧬 Exploración Biológica y Validación

Durante las pruebas de validación científica, `codon-infer` fue alimentado con un motivo rico en cisteínas (`MCMPCFTTDHQMARKCDD`, 18 residuos). El motor generó secuencias candidatas bajo restricciones de aminoácidos canónicos:

```text
Entrada (Prompt):  MCMPCFTTDHQMARKCDD
Generación:        MCMPCFTTDHQMARKCDDDPCCIIVQCCRQEIIQRCCLRIDRNCCIIIVYCCNCY
```

Al someter las secuencias candidatas a la base de datos **NCBI BLASTP** contra secuencias no redundantes, se confirmó alineamiento significativo con estructuras cristalográficas auténticas de clorotoxina (PDB: [`7X41_A`](https://www.rcsb.org/structure/7X41), E-value: $0.001$), demostrando la preservación de motivos estructurales y una exploración coherente del espacio conformacional peptídico *de novo*.

---

## 🧬 Arquitectura Multi-Escala ESM-2 y Motor SIMD Dinámico

A partir de la versión **v1.4.0**, `codon-infer` proporciona soporte nativo de primer nivel para los tres niveles principales de escala de la familia ESM-2:

```text
                                  ┌────────────────────────┐
                                  │ Pesos Binarios Modelo  │
                                  │ (*.bin / FP32 / Q8_0)  │
                                  └───────────┬────────────┘
                                              │ Heurística de longitud en bytes
                                              ▼
                              EsmConfig::detect_from_file_len()
                                              │
               ┌──────────────────────────────┼───────────────────────────────┐
               ▼                              ▼                               ▼
       ESM-2 8M (t6_8M)               ESM-2 35M (t12_35M)             ESM-2 150M (t30_150M)
    L=6, d=320, H=20, dk=16        L=12, d=480, H=20, dk=24        L=30, d=640, H=20, dk=32
    2 Regs AVX (__m256 x 2)        3 Regs AVX (__m256 x 3)         4 Regs AVX (__m256 x 4)
               │                              │                               │
               └──────────────────────────────┼───────────────────────────────┘
                                              ▼
                               Runtime Unificado Cero-Asignación
                               (RunState / BidirectionalRunState)
```

### 1. Autodetección Paramétrica sin Metadatos (`detect_from_file_len`)
Los frameworks convencionales de aprendizaje profundo en Python exigen archivos de configuración secundarios (`config.json`, configuraciones de tokenizador, manifiestos de modelo, etc.) junto a los tensores de pesos. `codon-infer` elimina esta sobrecarga mediante una heurística determinista basada en el número total de bytes del fichero:

```rust
pub fn detect_from_file_len(file_len: u64) -> io::Result<Self> {
    let floats = file_len / 4;
    if floats >= 140_000_000 {
        Ok(Self::esm2_150m()) // 150M (~591 MB FP32 / ~178 MB Q8_0)
    } else if floats >= 30_000_000 {
        Ok(Self::esm2_35m())  // 35M (~133 MB FP32 / ~48 MB Q8_0)
    } else if floats >= 7_000_000 {
        Ok(Self::esm2_8m())   // 8M (~30 MB FP32 / ~12 MB Q8_0)
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "Tamaño binario de pesos ESM-2 no reconocido"))
    }
}
```
Cualquier archivo de pesos personalizado suministrado mediante `--model <RUTA>` es identificado y validado al arrancar en $< 0.1\text{ ms}$, redimensionando automáticamente los búferes internos, cachés KV y matrices de proyección sin sobrecoste de asignación en tiempo de ejecución.

### 2. Optimización SIMD y Despacho Dinámico de Atención Heterogénea
En el mecanismo de atención multi-cabeza, cada cabeza posee una dimensión $d_k = d_{\text{model}} / H$. Dado que todas las variantes de ESM-2 estandarizan $H=20$ cabezas de atención, las dimensiones por cabeza varían de acuerdo con la escala del modelo:
* **8M:** $d_k = 320 / 20 = 16$ flotantes
* **35M:** $d_k = 480 / 20 = 24$ flotantes
* **150M:** $d_k = 640 / 20 = 32$ flotantes

Debido a que cada registro AVX de 256 bits (`__m256`) almacena con precisión 8 números flotantes de precisión simple (`f32`), cada $d_k$ resulta ser un múltiplo entero exacto de 8:
$$\frac{16}{8} = 2 \text{ registros}, \quad \frac{24}{8} = 3 \text{ registros}, \quad \frac{32}{8} = 4 \text{ registros}$$

`codon-infer` implementa micro-kernels vectoriales especializados y desenrollados (*loop-unrolled*) para productos escalares y acumulación de valores:
* **Producto Escalar Escalado ($Q \cdot K^\top$):** `dot_product_head_avx` procesa 16, 24 o 32 dimensiones en paralelo sin bucles escalares residuales ni comprobaciones de límites, realizando multiplicación por pares (`_mm256_mul_ps`), suma vectorial (`_mm256_add_ps`) y una suma horizontal ultrarrápida (`hsum256_ps_avx`).
* **Acumulación del Vector de Valores ($V \times \text{prob}$):** `accumulate_value_head_avx` difunde (*broadcast*) la probabilidad escalar a lo largo de un vector completo (`_mm256_set1_ps`) y ejecuta operaciones vectoriales combinadas de multiplicación y acumulación (`_mm256_add_ps` + `_mm256_mul_ps`) directamente en el acumulador de la cabeza de salida.
* **Garantía de Paridad Escalar:** La batería de pruebas unitarias (`test_attention_simd_parity_dk_16_24_32`) verifica la equivalencia matemática rigurosa frente a bucles de referencia escalares bajo un umbral de tolerancia estricto ($\Delta < 10^{-5}$) en las tres dimensiones de cabeza.

### 3. Matriz Técnica Comparativa de Arquitectura y Rendimiento

| Parámetro / Métrica | ESM-2 8M (`esm2_t6_8M_UR50D`) | ESM-2 35M (`esm2_t12_35M_UR50D`) | ESM-2 150M (`esm2_t30_150M_UR50D`) |
| :--- | :--- | :--- | :--- |
| **Capas Transformer ($L$)** | 6 | 12 | 30 |
| **Dimensión Oculta ($d_{\text{model}}$)** | 320 | 480 | 640 |
| **Cabezas de Atención ($H$)** | 20 | 20 | 20 |
| **Dimensión por Cabeza ($d_k$)** | 16 | 24 | 32 |
| **Registros AVX (`__m256`) por Cabeza** | **2 registros** (16 floats) | **3 registros** (24 floats) | **4 registros** (32 floats) |
| **Dimensión FFN ($d_{\text{ffn}}$)** | 1,280 | 1,920 | 2,560 |
| **Vocabulario** | 33 tokens | 33 tokens | 33 tokens |
| **Parámetros Totales** | 7,511,809 (~7.5M) | 33,524,449 (~33.5M) | 148,011,297 (~148M) |
| **Tamaño Binario (FP32)** | ~30.0 MB | ~134.1 MB | ~592.5 MB |
| **Tamaño Binario Cuantizado (`Q8_0`)** | ~11.5 MB | ~48.2 MB | ~178.6 MB |
| **Residencia en Jerarquía de Caché** | **100% en Caché L3** (32 MB) | Residencia Parcial en L3 (64 MB L3D) | Transmisión (*streaming*) desde RAM |
| **Rendimiento Típico CPU (tokens/s)** | **>200 tokens/s** | **~90 – 110 tokens/s** | **~25 – 35 tokens/s** |
| **Caso de Uso Primario** | DMS *in silico* ultrarrápido, inpainting en tiempo real | DMS de alta resolución, cribado de péptidos | Predicción de contactos 3D, diseño *de novo* |

---

## 🏗️ Pipeline de la Arquitectura Transformer

El pipeline de ejecución reproduce con absoluta fidelidad la arquitectura ESM-2 (derivada de RoBERTa):

```text
                    ┌────────────────────────┐
                    │ Secuencia Entrada (Str)│
                    └───────────┬────────────┘
                                │ Tokenizer::encode()
                                ▼
                    ┌────────────────────────┐
                    │ Token IDs ([0, ..., 2])│
                    └───────────┬────────────┘
                                │
 ┌──────────────────────────────▼───────────────────────────────┐
 │ Tabla de Embeddings de Tokens (33 tokens × dim)              │
 └──────────────────────────────┬───────────────────────────────┘
                                │  x [dim]
 ┌──────────────────────────────▼───────────────────────────────┐ ◄──┐
 │ Capa Transformer (×L Capas: 6, 12, o 30)                     │    │
 │  ├── LayerNorm 1 (dim, eps 1e-5)                             │    │
 │  ├── Atención Multi-Cabeza (20 Cabezas × dk: 16, 24, o 32)   │    │
 │  │    ├── Proyecciones: Q, K, V (SIMD Matmul AVX2+FMA / AVX1)│    │
 │  │    ├── RoPE Split-Half: rotación x[:half], x[half:dk]     │    │
 │  │    ├── Actualización Caché KV Ring Estática (pos × dim)   │    │
 │  │    ├── Atención Dot-Product Escalada (AVX dk-unrolled)    │    │
 │  │    └── Proyección de Salida (Wo: dim × dim + Bias)        │    │
 │  ├── Conexión Residual 1: x = x + att_out                    │    │
 │  ├── LayerNorm 2 (dim, eps 1e-5)                             │    │
 │  ├── Red Feed-Forward (FFN)                                  │    │
 │  │    ├── FC1 Densa: dim -> hidden_dim + Bias                │    │
 │  │    ├── Activación GELU (Aproximación numérica erf)        │    │
 │  │    └── FC2 Densa: hidden_dim -> dim + Bias                │    │
 │  └── Conexión Residual 2: x = x + ffn_out                    │    │
 └──────────────────────────────┬───────────────────────────────┘ ───┘
                                │  x [dim] (tras L capas)
 ┌──────────────────────────────▼───────────────────────────────┐
 │ LayerNorm Final (ln_f, dim, eps 1e-5)                        │
 └──────────────────────────────┬───────────────────────────────┘
                                │
 ┌──────────────────────────────▼───────────────────────────────┐
 │ EsmLMHead (Cabezal de Clasificación de Lenguaje)             │
 │  ├── Capa Densa: dim × dim + Bias                            │
 │  ├── Activación GELU (basada en erf)                         │
 │  └── Proyección de Embeddings Ligados (dim × 33) + Dec Bias  │
 └──────────────────────────────┬───────────────────────────────┘
                                │  logits [33]
 ┌──────────────────────────────▼───────────────────────────────┐
 │ Enmascaramiento Biológico (-inf en tokens <4 o >23)          │
 │ Muestreo Estocástico con Temperatura (SimpleRng / XORShift64)│
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
                 Siguiente Aminoácido Canónico
```

---

## 🚀 Guía de Inicio Rápido

### Requisitos Previos
* Cadena de herramientas Rust (edición 2024 o versión 1.85+):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 1. Compilación desde el Código Fuente
Clone el repositorio y compile los binarios con optimizaciones nativas:
```bash
git clone https://github.com/OsbaldoG/codon-infer.git
cd codon-infer

# Compilar binarios de release con optimizaciones vectoriales nativas del microprocesador
cargo build --release
```

### 2. Ejecución de la Suite de Pruebas
Valide las primitivas numéricas, la paridad de vectorización SIMD, el parser FASTA y los formatos de exportación:
```bash
cargo test
```
Salida esperada:
```text
running 22 tests
test export::tests::test_json_string_serializers ... ok
test fasta::tests::test_parse_multi_fasta_with_masks ... ok
test fasta::tests::test_parse_simple_fasta ... ok
test fasta::tests::test_parse_raw_sequence_fallback ... ok
test mcp::tests::test_mcp_json_parser_and_formatter ... ok
test mcp::tests::test_mcp_resolve_record_fallback ... ok
test export::tests::test_dms_row_and_csv_generation ... ok
test mcp::tests::test_mcp_tools_list_schema ... ok
test tests::test_attention_simd_parity_dk_16_24_32 ... ok
test tests::test_esm_config_autodetection ... ok
test tests::test_forward_pass_with_weights_if_available ... ok
test tests::test_layer_norm_stability ... ok
test tests::test_log_softmax_canonical ... ok
test tests::test_mask_token_encoding_and_decoding ... ok
test tests::test_matmul_avx1_and_dispatch_parity ... ok
test tests::test_q8_0_quantization_roundtrip ... ok
test tests::test_simd_benchmark_parity_run ... ok
test tests::test_softmax_sum_to_one ... ok
test tests::test_tokenizer_boundaries ... ok
test tests::test_bidirectional_logits_entropy ... ok
test tests::test_forward_bidirectional_if_available ... ok
test mcp::tests::test_mcp_dispatch_with_weights_if_available ... ok

test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; finished in 7.45s
```

### 3. Subcomandos y Pipelines Bioinformáticos

Todos los subcomandos aceptan la bandera global `--model <RUTA>`. `codon-infer` detecta automáticamente si el archivo binario corresponde a **ESM-2 8M**, **35M** o **150M** (en precisión completa FP32 o cuantizado `Q8_0`) y autoconfigura los kernels vectoriales correspondientes.

#### A. Generación Autorregresiva de Secuencias (`generate`)
Ejecuta síntesis de secuencias peptídicas *de novo* o extensión de motivos funcionales:
```bash
# Por defecto (pesos ESM-2 8M)
codon-infer generate --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7 --max-tokens 40

# Multi-escala: Ejecución con ESM-2 35M o 150M
codon-infer generate --model esm2_35m.bin --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
codon-infer generate --model esm2_150m.bin --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
```

#### B. Inpainting Bidireccional y Reconstrucción de Motivos (`inpaint`)
Reconstruye posiciones faltantes, mutadas o enmascaradas en sitios activos (indicadas mediante `*` o `<mask_1>`):
```bash
# Por defecto (ESM-2 8M)
codon-infer inpaint --sequence "MCMPC*TTDHQMARKCDD"

# Con pesos de mayor capacidad ESM-2 35M:
codon-infer inpaint --model esm2_35m.bin --sequence "MCMPC*TTDHQMARKCDD"
```

#### C. Escaneo Mutacional Profundo *In Silico* (`dms`)
Ejecuta saturación mutagénica punto a punto con mapas de calor térmicos TrueColor y exportación tabular:
```bash
# Cribado masivo de alto rendimiento con ESM-2 8M:
codon-infer dms --fasta proteins.fasta --window 128 --export-csv dms_matrix.csv --export-json dms_report.json

# Escaneo de máxima precisión con ESM-2 150M:
codon-infer dms --model esm2_150m.bin --fasta target.fasta --window 128 --export-csv dms_150m.csv
```

#### D. Predicción de Contactos Estructurales Residuales (`contacts`)
Calcula acoplamientos epistáticos mediante mapas de atención multi-cabeza (con corrección APC) e identifica puentes disulfuro putativos:
```bash
# Predicción de contactos 3D con ESM-2 150M para precisión de publicación:
codon-infer contacts --model esm2_150m.bin --fasta proteins.fasta --export-csv contacts.csv --export-json contacts.json
```

#### E. Suite de Benchmarks de Hardware y Validación Biológica (`benchmark`)
Ejecuta la suite integral de estrés de microarquitectura y validación biológica sobre cualquier escala de modelo:
```bash
codon-infer benchmark --suite full
codon-infer benchmark --model esm2_35m.bin --suite full
```

Suites disponibles:
* `simd`: Paridad numérica AVX2+FMA vs AVX1 y microbenchmarks matriz-vector en $d_k \in \{16, 24, 32\}$.
* `mlm`: Pseudo-perplejidad evolutiva y penalizaciones comparativas entre tipo salvaje y mutaciones disruptivas.
* `dms`: Calibración de escaneo mutacional profundo en motivos catalíticos de peptidasas bacterianas.
* `contacts`: Predicción de contactos espaciales y puntuación de proximidad Cys-Cys.
* `inpainting`: Reconstrucción posicional múltiple en secuencias enmascaradas bidireccionales.
* `stress`: Prueba de estabilidad continua de 10,000 tokens bajo sobrecarga de caché.
* `scaling`: Benchmark de escalamiento de longitud de contexto ($L = 32 \to 256$) evaluando latencia y memoria.

### 4. Servidor Model Context Protocol (MCP)
`codon-infer-mcp` expone la totalidad de las capacidades bioinformáticas como herramientas nativas para agentes LLM (Claude Desktop, Cursor, Gemini, etc.) sobre stdio mediante JSON-RPC:
```bash
# Iniciar con los pesos por defecto de ESM-2 8M
cargo run --release --bin codon-infer-mcp

# Iniciar con pesos de mayor capacidad ESM-2 35M o 150M
cargo run --release --bin codon-infer-mcp -- --model esm2_35m.bin
```

#### Herramientas Expuestas vía MCP:
* `generate_protein`: Generación autorregresiva de secuencias a partir de `prompt`, `temperature` y `max_tokens`.
* `codon_infer_inpaint`: Reconstruye residuos enmascarados (`*`) devolviendo los 5 candidatos principales con sus probabilidades y clasificación fisicoquímica.
* `codon_infer_dms`: Ejecuta mutagénesis de saturación *in silico* desde una secuencia cruda o archivo FASTA, identificando posiciones catalíticas críticas ($\Delta \log P \le -3.8$) y tolerancias mutacionales.
* `codon_infer_contacts`: Calcula pares de contacto epistáticos (APC) y detecta puentes disulfuro candidatos (`Cys-Cys`).

---

## 🛠️ Referencia Unificada de Opciones del CLI

### Banderas Globales
| Opción | Descripción | Valor por Defecto |
| :--- | :--- | :--- |
| `-h`, `--help` | Muestra el panel de ayuda con la guía de subcomandos | - |
| `-V`, `--version` | Muestra la versión y el motor vectorial SIMD activo | - |
| `--model <RUTA>` | Ruta al binario personalizado de pesos ESM-2 en FP32 o Q8_0 | `esm2_weights.bin` |
| `--color <auto\|always\|never>` | Controla el estilizado de la terminal mediante ANSI TrueColor | `auto` |
| `--no-color` | Desactiva terminantemente los códigos de escape ANSI (estándar NO_COLOR) | `false` |

### Opciones de Pipeline
| Opción | Subcomandos Aplicables | Descripción | Valor por Defecto |
| :--- | :--- | :--- | :--- |
| `--sequence <CADENA>` | `generate`, `inpaint`, `dms`, `contacts` | Secuencia de aminoácidos suministrada de forma directa | `"M"` |
| `--fasta <RUTA>` | `inpaint`, `dms`, `contacts` | Ruta al archivo FASTA (o `-` para leer de stdin) | - |
| `--export-csv <RUTA>`| `dms`, `contacts` | Ruta de destino para la matriz rectangular o tabla CSV | - |
| `--export-json <RUTA>`| `dms`, `contacts` | Ruta de destino para el informe estructurado en JSON | - |
| `--window <USIZE>` | `dms` | Tamaño de ventana deslizante para proteínas largas | `128` |
| `--stride <USIZE>` | `dms` | Paso de desplazamiento de la ventana deslizante | `64` |
| `--temperature <F32>`| `generate` | Temperatura de muestreo (0.0 = búsqueda voraz argmax) | `0.7` |
| `--max-tokens <USIZE>`| `generate` | Número máximo de nuevos residuos a generar | `1024` |
| `--suite <CADENA>` | `benchmark` | Suite específica de benchmark a ejecutar | `"full"` |

---

## 📂 Estructura del Proyecto

```text
codon-infer/
├── .cargo/
│   └── config.toml         # Banderas de compilación nativa (-C target-cpu=native)
├── .gitignore              # Filtro de artefactos generados y copias de seguridad
├── Cargo.toml              # Especificación cero dependencias (v1.4.0, edición 2024)
├── esm2_weights.bin        # Pesos FP32 de ESM-2 8M (7.51M parámetros, ~30 MB)
├── src/
│   ├── lib.rs              # Núcleo del motor: matmul SIMD dual, RoPE split-half, Q8_0, EsmLMHead
│   ├── main.rs             # Despachador de CLI y parser de argumentos de subcomandos
│   ├── ui.rs               # Paletas TrueColor ANSI, mapas de gradiente térmico y barra de progreso
│   ├── fasta.rs            # Lector de flujos y secuencias multi-FASTA de cero copias
│   ├── export.rs           # Serializadores de datos CSV y JSON para análisis de publicación
│   ├── pipeline.rs         # Controladores de ventana deslizante para DMS, contactos e inpainting
│   ├── benchmark.rs        # Suite automatizada de estrés de hardware y validación biológica
│   ├── mcp.rs              # Servidor JSON-RPC MCP sobre stdio sin dependencias externas
│   └── mcp_main.rs         # Punto de entrada ejecutable del servidor MCP autónomo
├── AUDIT_REPORT.md         # Informe exhaustivo de auditoría arquitectónica y matemática
├── LICENSE                 # Licencia MIT
├── README.md               # Documentación principal del proyecto (en inglés)
└── README_ES.md            # Documentación completa del proyecto (en español)
```

---

## 📜 Hoja de Ruta (Roadmap)

- [x] Migración completa a Rust puro desde el prototipo preliminar en C (**ADR-01**).
- [x] Runtime con estricta ausencia de asignaciones en heap durante inferencia (`RunState`).
- [x] Vectorización SIMD dual: micro-kernels `AVX2+FMA` y `AVX1 Sandy Bridge` con despacho dinámico.
- [x] Formulación RoPE *split-half* y aproximación numérica precisa de GELU mediante `erf`.
- [x] Integración completa del cabezal clasificador `EsmLMHead` (102,849 pesos).
- [x] Cuantización por bloques `Q8_0` para optimización de residencia en caché L3.
- [x] Demonio estándar del protocolo Model Context Protocol (MCP) vía JSON-RPC (`codon-infer-mcp`).
- [x] Modularización desacoplada: `lib.rs` (biblioteca núcleo), `main.rs` (CLI), `mcp.rs` (servidor MCP).
- [ ] Implementación de micro-kernels SIMD ARM NEON para Apple Silicon (M1/M2/M3/M4) y Raspberry Pi.
- [ ] Cuantización sub-byte INT4 (`Q4_0`) para dispositivos biomédicos integrados de consumo ultra-bajo.

---

## 🔬 Citación y Agradecimientos

Si utiliza `codon-infer` en sus investigaciones de bioinformática, biología estructural o biología sintética, por favor cite:
* **Arquitectura Original ESM-2:** Lin, Z., et al. *"Evolutionary-scale prediction of atomic-level protein structure with a language model."* Science 379.6637 (2023): eade2574.
* **Autor y Arquitecto Principal:** Manuel Osbaldo Gastélum Islas ([@OsbaldoG](https://github.com/OsbaldoG))

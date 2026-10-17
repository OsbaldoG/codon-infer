# codon-infer 🧬⚡

> **Motor de Inferencia y Pipeline Bioinformático para Modelos de Lenguaje de Proteínas ESM-2 Multi-Escala, de Alto Rendimiento y Cero Asignaciones en Rust Puro (v1.8.0).**

[🇬🇧 Read in English (README.md)](README.md)

[![Rust](https://img.shields.io/badge/rust-edición_2024-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/versión-1.8.0-blue.svg)](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.8.0)
[![Tests](https://img.shields.io/badge/tests-57%20ejecutados%3A%2056%20pasados%2C%200%20fallidos%2C%201%20ignorado-brightgreen.svg)](#3-ejecución-de-la-suite-de-pruebas)
[![Dependencies](https://img.shields.io/badge/dependencias-0%20(std%20puro)-brightgreen.svg)](Cargo.toml)
[![License](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/modelo-ESM--2%20(8M%20a%203B)-blueviolet.svg)](https://github.com/facebookresearch/esm)
[![SIMD](https://img.shields.io/badge/SIMD-AVX512%20%7C%20AVX2%2BFMA%20%7C%20AVX1-red.svg)](#2-optimización-simd-y-despacho-dinámico-de-atención-heterogénea)
[![Precision](https://img.shields.io/badge/precisión-FP32%20%7C%20Q8__0%20%7C%20Q4__0-green.svg)](#e-cuantización-nativa-de-modelos-a-q4_0-y-q8_0-quantize)
[![Throughput](https://img.shields.io/badge/Rendimiento%20CPU->200%20tokens%2Fseg-success.svg)](#-rendimiento-y-benchmarks-comparativos)

> 📖 **Nota de Idioma:** Este documento constituye la documentación oficial técnica completa en **español**. Para consultar la versión original en inglés técnico, acceda a [README.md](README.md).

`codon-infer` es un motor de inferencia y suite de análisis bioinformático ultraligero y libre de dependencias externas para la familia de modelos de lenguaje de proteínas Evolutionary Scale Modeling (**ESM-2**) en todas sus variantes de escala (`8M`, `35M`, `150M`, `650M` y `3B`), diseñado desde primeros principios matemáticos y de hardware en **Rust 100% seguro e idiomático (`std` exclusivamente)**.

Al prescindir por completo de los ecosistemas de Python, PyTorch, C/CUDA y crates externos de terceros, `codon-infer` ejecuta generación autorregresiva de secuencias, *inpainting* bidireccional de sitios activos, escaneo mutacional profundo *in silico* (DMS), predicción de contactos y reconstrucción 3D *ab initio* mediante vectorización SIMD dinámica a más de **200 tokens/segundo en CPUs x86_64 estándar**, manteniendo una huella de memoria estática inferior a 40 MB para modelos de 8M.

---

## ⚡ Aspectos Destacados (Versión 1.8: Reconstrucción 3D Zero-Trunk, Motor Tiled GEMM y Suite Canónica de Contactos)

> 💡 **Consolidación de Versión:** La versión 1.8.0 unifica las actualizaciones algorítmicas fundacionales de v1.7.0 (Motor Tiled GEMM Multi-Token `CI-1`, DMS Canónico `BIO-1` y Cabezal de Contacto Multicapa Oficial `STRUCT-1`) junto con el pionero pipeline de Reconstrucción Geométrica 3D Zero-Trunk (`STRUCT-2`).

* **Reconstrucción Geométrica 3D Zero-Trunk (`STRUCT-2`):**
  * Resuelve plegamientos proteicos 3D *ab initio* y genera coordenadas atómicas $C_\alpha$ directamente a partir de las representaciones de atención de ESM-2 en **15–35 ms** sobre procesadores x86_64 estándar, prescindiendo por completo de los pesados troncos de plegamiento de 3 GB de ESMFold o AlphaFold.
  * **Optimización Híbrida de Escalamiento Multidimensional (MDS) + Gradiente Biofísico:**
    1. *Fase A (MDS Métrico Espectral):* Clausura geodésica de Floyd-Warshall sobre el grafo de distancias cota superior, seguida por la descomposición espectral de la matriz de Gram doblemente centrada ($B = -\frac{1}{2} J D^{(2)} J$) y el método de las potencias con deflación para los 3 autovectores principales, reescalados al enlace peptídico canónico de $3.80\text{ \AA}$.
    2. *Fase B (Descenso de Gradiente Multi-Inicio con Momento):* 450 pasos de optimización acelerada por momento sobre 4 cuencas conformacionales (MDS espectral + 3 cintas helicoidales torsionales) minimizando un potencial físico canónico:
       * **Enlace Peptídico Exacto:** $d(C_{\alpha, i}, C_{\alpha, i+1}) = 3.80\text{ \AA}$ ($w = 18 \to 43$).
       * **Ángulos de Flexión Local:** $d(C_{\alpha, i}, C_{\alpha, i+2}) \in [5.30\text{ \AA}, 6.90\text{ \AA}]$ con guiado dinámico según señal helicoidal local.
       * **Restricciones de Contacto Terciario:** Potencial suave de Huber sobre los mejores $1.5 \cdot N$ pares con ponderación cuadrática por confianza ($w \propto P_{\text{rel}}^2$).
       * **Exclusión Estérica de Van der Waals:** Repulsión dura para $|i - j| \ge 3$ a distancias $d < 4.10\text{ \AA}$ y repulsión epistática de no-contacto.
       * **Compactación Globular:** Regularizador de radio de giro teórico de Flory $R_g = 2.25 \cdot N^{0.38}\text{ \AA}$.
  * **Paridad de Coordenadas de Nivel Publicación:** Reconstruye estructuras cristalográficas con RMSD $< 3.0\text{ \AA}$ y TM-score $> 0.50$ (p. ej. `2L2D` Trp-cage: **RMSD 2.26 Å**, TM 0.525; `1O6X`: **RMSD 2.95 Å**, TM 0.602; oráculo de cristal: **RMSD 2.43 Å**, TM 0.652, 0 choques estéricos).
  * Generación directa de archivos estándar `.pdb` (`codon-infer fold --sequence ... --export-pdb structure.pdb` y `codon-infer contacts ... --export-pdb contacts.pdb`).

* **Motor Tiled GEMM Multi-Token Bloqueado en Caché (CI-1):**
  * Multiplicación matricial por bloques de tokens (`matmul_tiled_fp32_parallel`, `matmul_tiled_q8_parallel`, `matmul_tiled_q4_parallel`).
  * Micro-kernels AVX2+FMA y AVX1 con $M_r = 4$ tokens simultáneos (`dot_row_batch_m4_avx2_fma`, `dot_row_batch_m4_avx1`) y desempaquetado por filas en caché L1d (`unpack_q8_row_avx1`, `unpack_q4_row_avx1`).
  * Elimina la saturación del bus DRAM (pasando de régimen limitado por memoria a limitado por cómputo), ofreciendo hasta **3.76× de aceleración en FP32** y **2.41×–3.44× en regímenes cuantizados** con paridad matemática bit a bit exacta ($\cos = 1.00000000$, RMSE $\sim 10^{-6}$, Top-1 Match 100.0%).

* **Escaneo Mutacional Profundo Canónico (BIO-1):**
  * Cálculo estricto del ratio de log-verosimilitud sin heurísticas arbitrarias ni reglas catalíticas codificadas: $\Delta \log P = \log P(x_i = m \mid x_{\setminus i}) - \log P(x_i = wt \mid x_{\setminus i})$.
  * Modo rápido en 1 pasada `WildtypeMarginal` (`--fast`, `--wt-marginal`, `--mode wt`) que acelera la mutagénesis de saturación hasta **42×** (superando **16,170 mutaciones/seg**) identificando de forma fiable motivos catalíticos estrictamente conservados (p. ej., GDSGGP).

* **Cabezal de Contacto Multicapa Oficial con Streaming O(N²) (STRUCT-1):**
  * Integra los pesos oficiales de regresión logística multicapa de Meta AI en las 5 escalas (8M, 35M, 150M, 650M, 3B) en apenas 12.2 KiB de memoria estática, preservando 100% de compatibilidad binaria (`.bin`, `_q8_0.bin`, `_q4_0.bin`).
  * Acumulación en streaming $O(N^2)$ (`accumulate_head_apc`), reduciendo el consumo de memoria en **1,440×** (sin necesidad de materializar pesados tensores $L \times H \times N \times N$) y alcanzando hasta **96.63% de Precision@L/5** en modelos cuantizados de 650M con solo 413 MiB de RAM.

* **Herramienta 3D para Model Context Protocol (MCP) (`codon_infer_fold`):**
  * Expone el plegamiento 3D y la generación atómica PDB mediante el demonio stdio JSON-RPC, entregando coordenadas y estructuras `.pdb` válidas directamente a entornos de agentes.

---

## ⚡ Aspectos Destacados (Versión 1.6: Motor Multi-Escala Q4_0, Paridad Canónica HF y Cuantizador Streaming 3B)

* **Motor de Cuantización Nativa por Bloques `Q4_0` a 4 Bits (`BlockQ4_0`):**
  * Representación sub-byte que empaqueta 32 nibbles de 4 bits en 16 bytes junto a un factor de escala flotante `d` de 32 bits (`#[repr(C)]`, exactamente **20 bytes** por bloque de 32 parámetros).
  * Convención de cuantización simétrica estándar compatible con llama.cpp: `d = max / -8.0`, `id = 1/d`, `q = clamp(round(x · id) + 8, 0, 15)`. El nibble bajo almacena el índice `j ∈ [0, 15]` y el nibble alto el índice `j + 16`.
  * **Compresión Teórica de Matrices de 6.40×** (0.625 B/parámetro frente a 4.0 B en FP32) y hasta **6.30× de Reducción Total del Archivo** (**ESM-2 3B** reducido de 11.36 GB a **1.80 GB**, y **ESM-2 650M** reducido de 2.60 GB a **414.62 MB**).
  * **Reducción de ~83.5% en Memoria Pico (RSS):** La huella de memoria en RAM de ESM-2 650M pasa de 2,501.33 MiB en FP32 a **413.07 MiB en Q4_0**, mientras que **ESM-2 3B** se ejecuta en apenas **1,752.57 MiB RSS en Q4_0** (`3,102.18 MiB` en Q8_0) sobre una máquina de 11 GiB de RAM.
  * **Aceleración de 1.92× en Inferencia para ESM-2 650M:** Alivia el cuello de botella de ancho de banda DRAM en ejecuciones con 4 hilos (**4,535.47 ms** en Q4_0 y **4,669.89 ms** en Q8_0 frente a **8,692.15 ms** en FP32).
* **Paridad Matemática Exacta con HuggingFace (`grouped_canonical_v1`):**
  * Validación estricta al byte exacto en las 5 escalas (`8M`, `35M`, `150M`, `650M`, `3B`) con **100.0% de coincidencia Top-1** y similitud de coseno ≥ 0.999999 frente a las referencias oficiales de HuggingFace `facebook/esm2_*` (`hidden_states` y `logits`).
  * Cuantizadores por streaming bloque a bloque (`quantize_file_streaming_q8` / `quantize_file_streaming_q4`) mediante `BufReader` / `BufWriter` que convierten los 10.58 GiB de `esm2_3b_weights.bin` con un **Peak RSS de solo 12.55 MiB**.
* **Arquitectura de Cabecera Universal `CODQ` de 64 Bytes:**
  * Cabecera fija little-endian de 64 bytes (`b"CODQ"`, versión 1, tipo de datos, arquitectura, profundidad de capas, dimensión oculta, dimensión FFN, vocabulario).
  * Garantiza alineación estricta de 64 bytes (línea de caché y SIMD) para el primer bloque de pesos, permitiendo carga zero-copy inmediata.
  * Resuelve la ambigüedad paramétrica entre escalas cuantizadas con retrocompatibilidad completa mediante `detect_from_file_len`.
* **Micro-Kernels Vectoriales SIMD con Manipulación de Bits:**
  * Kernels AVX2+FMA que desempaquetan 16 bytes con `_mm_srli_epi16` y `_mm_and_si128`, convierten con signo a enteros de 32 bits y flotantes con `_mm256_cvtepi8_epi32` y `_mm256_cvtepi32_ps`, y acumulan con FMA (`_mm256_fmadd_ps`).
  * Kernel de reserva AVX1 con desempaquetado entero de 128 bits y acumulación flotante de 256 bits.
  * Cero asignaciones en el heap durante el ciclo de inferencia (*hot path*).
* **CLI Multi-Formato Unificada (`quantize`, `generate`, `inpaint`, `dms`, `contacts`, `benchmark`):**
  * Autodetección dinámica del formato (FP32, Q8_0, Q4_0) mediante cabecera `CODQ` o fallback heurístico legacy.
  * Nuevas opciones de cuantización por streaming en `codon-infer quantize --model <input> --output <output> [--format <q4_0|q8_0>]` con autodetección por extensión.

---

## ⚡ Aspectos Destacados (Versión 1.5)

* **Motor de Cuantización Nativa por Bloques `Q8_0` y Serialización Binaria Directa (`BlockQ8_0`):**
  * Formato de pesos cuantizados por bloques (32 valores `i8` + factor de escala `f32` por bloque) que reduce el ancho de banda y la huella en RAM en un factor de **3.56x** (de ~2.48 GB a **698.5 MB** en ESM-2 650M) con **99.980% de fidelidad de logits** (Similitud de Coseno `0.99980`, RMSE `0.00021`).
  * Carga zero-copy directa desde binarios cuantizados nativos (`*_q8_0.bin`) en solo **219.98 ms** (frente a ~751 ms en FP32).
  * Micro-kernels vectoriales GEMV de descuantización *on-the-fly* (`AVX1` / `AVX2` / `AVX-512`), eliminando asignaciones en el heap y rotación de tensores durante el producto matriz-vector.
  * Subcomando de CLI `quantize` dedicado para convertir pesos FP32 directamente en binarios optimizados `Q8_0`.
* **Motor de Concurrencia Persistente `ThreadPool` Cero-Spawn (`std` puro):**
  * Paralelización sin coste de llamadas al sistema operativo a través de las proyecciones densas FFN y cabezas de atención mediante hilos persistentes, erradicando ~230 transiciones de kernel `pthread_create` / join por cada token generado.
  * Barrera híbrida spin/futex (`ThreadBarrier`) con rendimiento spin lock-free adaptativo y estacionamiento eficiente para sincronización sub-microsegundo.
  * Detección automática de la topología del procesador (`detect_optimal_threads` vía `/proc/cpuinfo`) que prioriza núcleos físicos de cómputo sobre hilos lógicos SMT/Hyper-Threading, saturando los puertos vectoriales AVX de 256 bits sin contención.
* **Arquitectura Multi-Escala y Autodetección Paramétrica (`detect_from_file_len`):**
  * Instanciación instantánea y sin configuración manual para las variantes **ESM-2 8M**, **35M**, **150M**, **650M** y **3B** a partir de la longitud en bytes del archivo binario de pesos, eliminando la necesidad de archivos de configuración JSON o manifiestos auxiliares.
* **Kernels de Atención SIMD Dinámicos y Heterogéneos (`d_k` ∈ {16, 24, 32, 64}):**
  * Micro-kernels vectoriales AVX de 256 bits y AVX-512 de 512 bits optimizados a mano y mapeados a 2, 3, 4 u 8 registros `__m256` (o 4 registros `__m512` ZMM) por cabeza de atención, maximizando la densidad de cómputo vectorial con paridad escalar bit a bit (Δ < 10⁻⁵) tanto en pasadas autorregresivas causales como bidireccionales completas.
* **Subcomandos de CLI para Entornos de Producción (`generate`, `inpaint`, `dms`, `contacts`, `benchmark`, `quantize`):**
  * Interfaz de terminal expresiva y unificada con soporte de subcomandos, banderas globales (`--model`, `--threads`, `--quantize`, `--color`, `--no-color`), detección automática de TTY y diagnósticos claros de errores.
* **Experiencia de Usuario TrueColor ANSI sin Dependencias (`ui.rs`):**
  * **Paleta Fisicoquímica:** Residuos de aminoácidos formateados en colores ANSI RGB TrueColor según sus propiedades bioquímicas (Alifáticos, Aromáticos, Básicos, Ácidos, Polares, Cisteínas y de Flexibilidad Conformacional).
  * **Mapa de Calor Térmico Continuo:** Interpolación RGB dinámica (`lerp_rgb`) desde Carmesí Profundo (`#7F1D1D`) para mutaciones deletéreas/letales (Δ log P ≤ -4.0) pasando por Ámbar y Pizarra Neutro, hasta Azul Eléctrico (`#1E3A8A`) para variantes benéficas, preservando glifos de densidad ASCII (`█`, `▓`, `▒`, `░`, `·`) en terminales sin soporte de color.
  * **Barra de Progreso Dinámica en Stderr:** Actualizaciones limpias con retornos de carro ANSI `\r` reportando rendimiento instantáneo (`muts/s`) y tiempo estimado de finalización (`ETA`) sin contaminar las salidas redirigidas a `stdout`.
* **Pipeline Multi-FASTA y Transmisión por Flujos (`fasta.rs`):**
  * Parser de streaming de alto rendimiento y cero copias para archivos multi-FASTA (`--fasta <FICHERO>`), entrada estándar (`--fasta -`) o secuencias pasadas directamente como argumento (`--sequence <CADENA>`).
* **Formatos de Exportación con Calidad de Publicación (`export.rs`):**
  * Generación de matrices rectangulares en formato CSV e informes exhaustivos en JSON con tolerancias mutacionales por residuo, entropía informacional de Shannon y pares de contacto de alta probabilidad.
* **Evaluador de Ventana Deslizante para Proteínas de Longitud Completa (`pipeline.rs`):**
  * Procesa proteínas de longitud arbitraria mediante ventanas deslizantes adaptativas (W = 128, S = 64) centradas en los residuos evaluados, evitando distorsiones en los bordes por RoPE y garantizando localidad en caché.
* **Aceleración por Hardware Multi-SIMD y Núcleo en Rust Puro:**
  * **AVX-512:** Pipeline vectorial de 512 bits para procesamiento ultraveloz de atención con `d_k = 64` en arquitecturas Intel Xeon y AMD Zen 4+.
  * **AVX2 + FMA:** Pipeline vectorial con instrucciones Fused Multiply-Add de 256 bits para arquitecturas AMD Ryzen e Intel Haswell o superiores.
  * **AVX1:** Kernel SIMD de 256 bits compatible con procesadores Intel Sandy Bridge / Ivy Bridge heredados.
  * **Despacho Dinámico en Tiempo de Ejecución:** Detección de capacidades del microprocesador al inicio mediante `is_x86_feature_detected!`.
* **Paridad Matemática y Arquitectónica Estricta con ESM-2:**
  * Cabezal de clasificación `EsmLMHead` completo con aproximación numérica de la función de error `erf` para GELU e incrustaciones posicionales rotatorias (RoPE) con formulación *split-half*.
* **Runtime Estricto de Cero Asignaciones (`RunState`):** Todos los búferes de memoria del transformer, matrices de atención intermedias y cachés KV se preasignan al iniciar. Cero fragmentación o recolección de memoria durante la inferencia.

---

## 📊 Rendimiento y Benchmarks Comparativos

Evaluado en Linux x86_64 (ejecución monohilo en CPU con instrucciones vectoriales nativas AVX/AVX2+FMA):

### 1. Contraste frente a la Línea Base de Python / PyTorch
La migración arquitectónica a Rust puro con garantías deterministas de cero asignaciones en memoria supera drásticamente al ecosistema tradicional de Python/PyTorch en todas las métricas operativas:

| Métrica | Referencia PyTorch / Python (ESM-2 8M) | `codon-infer` v1.8 (Rust Puro + SIMD) | Ventaja de Ingeniería |
| :--- | :--- | :--- | :--- |
| **Dependencias** | Python, PyTorch, LibTorch, CUDA (~4–7 GB) | **`std` puro (0 crates, 0 dependencias dinámicas)** | **~99.9% más liviano** |
| **Tamaño del Ejecutable** | Entorno multigigabyte | **~1.1 MB binario release compilado** | **Binario único ultra-portable** |
| **Latencia de Carga** | ~2,500 – 4,000 ms (inicialización PyTorch) | **4.69 ms (8M Q4) / 12.85 ms (35M Q4) / 137.3 ms (650M Q4)** | **~20x – 850x arranque más rápido** |
| **Huella en Memoria (RAM)** | ~1.5 – 2.5 GB RSS base | **29.3 MB RSS (8M Q4) / 413.1 MB (650M Q4) / 1.75 GB (3B Q4)** | **~4x – 85x más eficiente en memoria** |
| **Rendimiento de Inferencia** | ~15 – 35 tokens/seg (CPU monohilo) | **456.5 tok/s (8M Q4) / 9.04 tok/s (650M Q4) / 2.12 tok/s (3B Q4)** | **Máxima densidad de cómputo, 0 asignaciones** |
| **Asignación de Memoria** | Pausas continuas de GC y churn de tensores | **0 asignaciones de heap en el hot-path** | **Garantías deterministas en tiempo real** |

### 2. Matriz de Rendimiento Multi-Escala Real (ESM-2 8M a 3B)
Métricas empíricas auditadas en Linux x86_64 (AVX2+FMA, 4 hilos, 41 tokens, kernel `wait4 ru_maxrss` de `docs/benchmarks_v1.6.0.md`):

| Escala del Modelo | Capas (L) | Dimensión Oculta (d) | Cabezas Atención (H × d_k) | Tamaño Binario Q4_0 | Latencia Carga Q4_0 | Bidireccional (41 tok, 4T) | Pico RAM RSS (Q4_0) | Jerarquía de Caché y Memoria |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| **ESM-2 8M** | 6 | 320 | 20 × 16 (2 YMM) | **4.97 MiB** (`5,208,516 B`) | **4.69 ms** | **456.5 tok/s** (89.8 ms) | **29.29 MiB** | **100% en Caché L3** (≤ 32 MB) · Limitado por Cómputo |
| **ESM-2 35M** | 12 | 480 | 20 × 24 (3 YMM) | **21.07 MiB** (`22,093,636 B`) | **12.85 ms** | **128.7 tok/s** (318.6 ms) | **29.41 MiB** | **Residencia Parcial L3 / DDR** · Balanceado Caché/DRAM |
| **ESM-2 150M** | 30 | 640 | 20 × 32 (4 YMM) | **90.58 MiB** (`94,978,756 B`) | **39.17 ms** | **30.0 tok/s** (1,367.8 ms) | **100.39 MiB** | **Streaming desde RAM** (> 500 MB FP32) · Limitado por Bus |
| **ESM-2 650M** | 33 | 1280 | 20 × 64 (8 YMM) | **395.41 MiB** (`414,617,796 B`) | **137.32 ms** | **9.04 tok/s** (4,535.5 ms) | **413.07 MiB** | **Optimizado para Streaming DRAM** · Núcleos Saturados |
| **ESM-2 3B** | 36 | 2560 | 40 × 64 (8 YMM) | **1,717.76 MiB** (`1,801,205,956 B`) | **545.36 ms** | **2.12 tok/s** (19,368.9 ms) | **1,752.57 MiB** | **Streaming Masivo DRAM** (Opera en < 2 GB RAM) |

### 3. Escalamiento Multihilo y Cuantización en ESM-2 650M (Núcleos Físicos vs SMT)

Evaluado en procesador de doble núcleo Intel Sandy Bridge (Core i5-2520M @ 2.50 GHz, 2 núcleos físicos, 4 hilos lógicos SMT) con micro-kernels vectoriales AVX1:

| Configuración de Ejecución | Precisión de Pesos | Huella en Memoria (RAM) | Latencia de Carga en Frío | Throughput | Latency por Token | Aceleración Relativa |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Hilo (Monocore)** | FP32 | 2,485 MB | 751.2 ms | 4.57 tok/s | 218.8 ms/tok | 1.00x (Base) |
| **1 Hilo (Monocore)** | Q8_0 | 698.5 MB | 219.98 ms | 5.15 tok/s | 194.2 ms/tok | 1.13x |
| **2 Núcleos Físicos (`-t 2`)** | Q8_0 | 698.5 MB | 219.98 ms | 7.70 tok/s | 129.9 ms/tok | 1.68x |
| **4 Hilos SMT (`-t 4`)** | Q8_0 | 698.5 MB | 219.98 ms | 7.48 tok/s | 133.7 ms/tok | 1.64x (Contención AVX) |
| **4 Hilos (`-t 4`)** | **Q4_0** | **413.1 MB** | **137.32 ms** | **9.04 tok/s** | **110.6 ms/tok** | **1.98x (Aceleración Máxima y Menor RAM)** |

> [!NOTE]
> **Saturación de Núcleos Físicos vs Hyper-Threading:** En arquitecturas donde los hilos lógicos SMT comparten el puerto de ejecución vectorial AVX de 256 bits por núcleo físico, asignar 2 hilos a los 2 núcleos reales (`-t 2`) otorga el rendimiento máximo (7.70 tok/s) y la mínima latencia de sincronización. `codon-infer` detecta automáticamente los núcleos físicos vía `detect_optimal_threads` si se omite `--threads`.

> [!TIP]
> **Aceleración con Motor Tiled GEMM Multi-Token (CI-1):**
> En los pipelines forward de secuencias completas (`dms`, `contacts` y `fold`), el motor Tiled GEMM agrupa la evaluación de tokens en micro-kernels de $M_r = 4$ tokens bloqueados en caché (`dot_row_batch_m4_avx2_fma`), eliminando la relectura secuencial de matrices de pesos desde DRAM. Esto acelera las pasadas forward de secuencia completa hasta **3.76× en FP32** y **2.41×–3.44× en regímenes cuantizados (Q8_0/Q4_0)**, reduciendo el forward de ESM-2 650M a solo **2.3 s – 4.4 s** manteniendo paridad numérica bit a bit exacta ($\cos = 1.00000000$).

---

## 🧬 Exploración Biológica y Validación

Durante las pruebas de validación científica, `codon-infer` fue alimentado con un motivo rico en cisteínas (`MCMPCFTTDHQMARKCDD`, 18 residuos). El motor generó secuencias candidatas bajo restricciones de aminoácidos canónicos:

```text
Entrada (Prompt):  MCMPCFTTDHQMARKCDD
Generación:        MCMPCFTTDHQMARKCDDDPCCIIVQCCRQEIIQRCCLRIDRNCCIIIVYCCNCY
```

Al someter las secuencias candidatas a la base de datos **NCBI BLASTP** contra secuencias no redundantes, se confirmó alineamiento significativo con estructuras cristalográficas auténticas de clorotoxina (PDB: [`7X41_A`](https://www.rcsb.org/structure/7X41), E-value: 0.001), demostrando la preservación de motivos estructurales y una exploración coherente del espacio conformacional peptídico *de novo*.

---

## 🧬 Arquitectura Multi-Escala ESM-2 y Motor SIMD Dinámico

A partir de la versión **v1.4.0**, `codon-infer` proporciona soporte nativo de primer nivel para la totalidad del espectro multi-escala de la familia ESM-2 (8M, 35M, 150M, 650M y 3B):

```text
                                  ┌────────────────────────┐
                                  │ Pesos Binarios Modelo  │
                                  │ (*.bin / FP32 / Q8_0)  │
                                  └───────────┬────────────┘
                                              │ Heurística de longitud en bytes
                                              ▼
                              EsmConfig::detect_from_file_len()
                                              │
  ┌───────────────────┬───────────────────────┼───────────────────────┬───────────────────┐
  ▼                   ▼                       ▼                       ▼                   ▼
8M (t6_8M)          35M (t12_35M)           150M (t30_150M)         650M (t33_650M)     3B (t36_3B)
L=6, d=320          L=12, d=480             L=30, d=640             L=33, d=1280        L=36, d=2560
H=20, dk=16         H=20, dk=24             H=20, dk=32             H=20, dk=64         H=40, dk=64
2 Regs AVX          3 Regs AVX              4 Regs AVX              8 AVX / 4 AVX-512   8 AVX / 4 AVX-512
  │                   │                       │                       │                   │
  └───────────────────┴───────────────────────┼───────────────────────┴───────────────────┘
                                              ▼
                               Runtime Unificado Cero-Asignación
                               (RunState / BidirectionalRunState)
```

### 1. Autodetección Paramétrica sin Metadatos (`detect_from_file_len`)
Los frameworks convencionales de aprendizaje profundo en Python exigen archivos de configuración secundarios (`config.json`, configuraciones de tokenizador, manifiestos de modelo, etc.) junto a los tensores de pesos. `codon-infer` elimina esta sobrecarga mediante una heurística determinista basada en el número total de bytes del fichero:

```rust
pub fn detect_from_file_len(file_len: u64) -> io::Result<Self> {
    match file_len {
        30_091_652 | 30_092_168 => return Ok(Self::esm2_8m()),
        134_067_972 | 134_068_984 => return Ok(Self::esm2_35m()),
        592_642_692 | 592_645_160 => return Ok(Self::esm2_150m()),
        2_604_339_332 | 2_604_342_104 => return Ok(Self::esm2_650m()),
        11_356_354_692 | 11_356_360_584 | 11_357_091_972 => return Ok(Self::esm2_3b()),
        _ => {}
    }
    if file_len >= 3_000_000_000 {
        // ESM-2 3B: ~11.36 GB (FP32) o ~3.2 GB - 3.4 GB (Q8_0)
        Ok(Self::esm2_3b())
    } else if file_len >= 2_000_000_000 {
        // ESM-2 650M: ~2.60 GB (FP32)
        Ok(Self::esm2_650m())
    } else if file_len >= 600_000_000 {
        // ESM-2 650M: ~698.5 MB - 780 MB (Q8_0)
        Ok(Self::esm2_650m())
    } else if file_len >= 450_000_000 {
        // ESM-2 150M: ~592.6 MB (FP32)
        Ok(Self::esm2_150m())
    } else if file_len >= 140_000_000 {
        // ESM-2 150M: ~166 MB - 178 MB (Q8_0)
        Ok(Self::esm2_150m())
    } else if file_len >= 80_000_000 {
        // ESM-2 35M: ~134.1 MB (FP32)
        Ok(Self::esm2_35m())
    } else if file_len >= 30_500_000 {
        // ESM-2 35M: ~38 MB - 48 MB (Q8_0)
        Ok(Self::esm2_35m())
    } else if file_len >= 20_000_000 {
        // ESM-2 8M: ~30.1 MB (FP32)
        Ok(Self::esm2_8m())
    } else if file_len >= 6_000_000 {
        // ESM-2 8M: ~8.5 MB - 12 MB (Q8_0)
        Ok(Self::esm2_8m())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "Tamaño binario de pesos ESM-2 no reconocido"))
    }
}
```
Cualquier archivo de pesos personalizado suministrado mediante `--model <RUTA>` es identificado y validado al arrancar en < 0.1 ms, redimensionando automáticamente los búferes internos, cachés KV y matrices de proyección sin sobrecoste de asignación en tiempo de ejecución.

### 2. Optimización SIMD y Despacho Dinámico de Atención Heterogénea
En el mecanismo de atención multi-cabeza, cada cabeza posee una dimensión d_k = d_model / H. Mientras que 8M, 35M, 150M y 650M estandarizan H = 20 cabezas de atención, 3B escala a H = 40 cabezas:
* **8M:** d_k = 320 / 20 = 16 flotantes
* **35M:** d_k = 480 / 20 = 24 flotantes
* **150M:** d_k = 640 / 20 = 32 flotantes
* **650M:** d_k = 1280 / 20 = 64 flotantes
* **3B:** d_k = 2560 / 40 = 64 flotantes

Debido a que cada registro AVX de 256 bits (`__m256`) almacena con precisión 8 números flotantes de precisión simple (`f32`), y cada registro AVX-512 de 512 bits (`__m512`) almacena 16 flotantes, cada `d_k` resulta ser un múltiplo entero exacto:

```text
16 / 8 = 2 YMM,   24 / 8 = 3 YMM,   32 / 8 = 4 YMM,   64 / 8 = 8 YMM  (o 4 ZMM en AVX-512)
```

`codon-infer` implementa micro-kernels vectoriales especializados y desenrollados (*loop-unrolled*) para productos escalares y acumulación de valores:
* **Producto Escalar Escalado (Q · Kᵀ):** `dot_product_head_avx` procesa 16, 24, 32 o 64 dimensiones en paralelo sin bucles escalares residuales ni comprobaciones de límites (`_mm256_mul_ps`, `_mm256_add_ps`, `hsum256_ps_avx`). En hardware con AVX-512, d_k = 64 se despacha a `dot_product_head_avx512_d64` aprovechando 4 registros ZMM y reducción horizontal de ciclo único (`_mm512_reduce_add_ps`).
* **Acumulación del Vector de Valores (V × prob):** `accumulate_value_head_avx` difunde (*broadcast*) la probabilidad escalar a lo largo de un vector completo (`_mm256_set1_ps`) y ejecuta operaciones vectoriales combinadas de multiplicación y acumulación. Para d_k = 64, AVX-512 despacha `accumulate_value_head_avx512_d64` (`_mm512_fmadd_ps`).
* **Garantía de Paridad Escalar:** Las baterías de pruebas unitarias (`test_attention_simd_parity_dk_16_24_32`, `test_attention_simd_parity_dk_64`) verifican la equivalencia matemática rigurosa frente a bucles de referencia escalares bajo un umbral de tolerancia estricto (Δ < 10⁻⁵) en todas las 4 dimensiones de cabeza.

### 3. Matriz Técnica Comparativa de Arquitectura y Rendimiento

| Parámetro / Métrica | ESM-2 8M (`esm2_t6_8M_UR50D`) | ESM-2 35M (`esm2_t12_35M_UR50D`) | ESM-2 150M (`esm2_t30_150M_UR50D`) | ESM-2 650M (`esm2_t33_650M_UR50D`) | ESM-2 3B (`esm2_t36_3B_UR50D`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Capas Transformer (L)** | 6 | 12 | 30 | 33 | 36 |
| **Dimensión Oculta (d_model)** | 320 | 480 | 640 | 1,280 | 2,560 |
| **Cabezas de Atención (H)** | 20 | 20 | 20 | 20 | 40 |
| **Dimensión por Cabeza (d_k)** | 16 | 24 | 32 | 64 | 64 |
| **Registros SIMD por Cabeza** | **2 YMM** (16 floats) | **3 YMM** (24 floats) | **4 YMM** (32 floats) | **8 YMM / 4 ZMM** (64 floats) | **8 YMM / 4 ZMM** (64 floats) |
| **Dimensión FFN (d_ffn)** | 1,280 | 1,920 | 2,560 | 5,120 | 10,240 |
| **Vocabulario** | 33 tokens | 33 tokens | 33 tokens | 33 tokens | 33 tokens |
| **Flotantes Totales (`grouped_canonical_v1`)** | 7,522,913 (~7.5M) | 33,516,993 (~33.5M) | 148,160,673 (~148M) | 651,084,833 (~651M) | 2,839,088,673 (~2.84B) |
| **Tamaño Binario (FP32)** | `30,091,652 B` (28.70 MiB) | `134,067,972 B` (127.86 MiB) | `592,642,692 B` (565.19 MiB) | **`2,604,339,332 B` (2,483.69 MiB)** | **`11,356,354,692 B` (10.58 GiB)** |
| **Tamaño Binario Cuantizado (`Q8_0`)** | `8,864,556 B` (8.45 MiB) | `38,636,896 B` (36.85 MiB) | `168,646,036 B` (160.83 MiB) | **`738,899,556 B` (704.67 MiB)** | **`3,216,540,676 B` (3.00 GiB)** |
| **Tamaño Binario Cuantizado (`Q4_0`)** | **`5,208,516 B` (4.97 MiB)** | **`22,093,636 B` (21.07 MiB)** | **`94,978,756 B` (90.58 MiB)** | **`414,617,796 B` (395.41 MiB)** | **`1,801,205,956 B` (1.68 GiB)** |
| **Latencia Carga Caliente (Q4_0)** | **4.69 ms** | **12.85 ms** | **39.17 ms** | **137.32 ms** | **545.36 ms** |
| **Throughput Bidireccional (L = 41, 4T)** | **476.9 tok/s (Q8) / 456.5 tok/s (Q4)** | **128.7 tok/s (Q4)** | **30.0 tok/s (Q4)** | **9.04 tok/s (Q4)** | **2.20 tok/s (Q8) / 2.12 tok/s (Q4)** |
| **Memoria de Estado de Trabajo (L = 128)** | **~911 KB** | **~1,321 KB** | **~1,732 KB** | **~3,450 KB** | **~7,200 KB** |
| **Residencia en Jerarquía de Caché** | **100% en Caché L3** (≤ 32 MB) | Residencia Parcial en L3 / DDR | Transmisión (*streaming*) desde RAM (> 500 MB) | Streaming DRAM (~2.48 GB FP32 / ~395 MB Q4) | Streaming DRAM (~1.68 GiB Q4) |
| **Demanda de Ancho de Banda de Memoria** | **Limitado por Cómputo (AVX/SIMD)** | **Balanceado Caché/DRAM** | **Limitado por Bus de Memoria** | **Limitado por Bus (Optimizado Q4/Q8)** | **Limitado por DRAM (Habilitado por Q4/Q8)** |
| **Caso de Uso Primario** | DMS *in silico* ultrarrápido, inpainting en tiempo real | DMS de alta resolución, cribado de péptidos | Predicción de contactos 3D, diseño *de novo* | Mapas de contacto de alta fidelidad, DMS fino | Modelos de lenguaje de proteínas de frontera en CPU |

### 4. Validación Empírica Multi-Escala y Rendimiento frente a HuggingFace (`8M` a `3B`)

Evaluación empírica en `test_peptidase.fasta` (41 tokens) frente a los tensores oficiales de referencia de HuggingFace `facebook/esm2_*` (`scripts/ref_{8m,35m,150m,650m,3b}.npz`, generados localmente mediante `scripts/generate_ref_8m.py` o `scripts/colab_export_esm2.ipynb`) bajo ejecución vectorial SIMD nativa (4 hilos, `wait4` kernel `ru_maxrss`):

| Escala del Modelo | Formato | Tamaño Exacto (`stat -c %s`) | Compresión Total | Coseno `hidden_states[-1]` vs HF | Coseno Logits vs HF | Divergencia KL vs HF | Top-1 Match vs HF | Forward Mediana (4 hilos) | Throughput | Memoria Pico (RSS) |
| :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| **ESM-2 8M** | FP32 | `30,091,652 B` (28.70 MiB) | 1.00× | 0.999999 | 0.999999 | 0.000000 | **100.0% (41/41)** | 113.44 ms | 361.4 tok/s | 32.75 MiB |
| | Q8_0 | `8,864,556 B` (8.45 MiB) | 3.39× | 0.999764 | 0.999995 | 0.000128 | **100.0% (41/41)** | 85.97 ms | 476.9 tok/s | 29.11 MiB |
| | **Q4_0** | **`5,208,516 B` (4.97 MiB)** | **5.78×** | **0.934598** | **0.998168** | **0.031316** | **100.0% (41/41)** | **89.82 ms** | **456.5 tok/s** | **29.29 MiB** |
| **ESM-2 35M** | FP32 | `134,067,972 B` (127.86 MiB) | 1.00× | 0.999999 | 1.000000 | 0.000000 | **100.0% (41/41)** | 446.47 ms | 91.8 tok/s | 133.30 MiB |
| | Q8_0 | `38,636,896 B` (36.85 MiB) | 3.47× | 0.999825 | 0.999994 | 0.000120 | **100.0% (41/41)** | 377.27 ms | 108.7 tok/s | 42.24 MiB |
| | **Q4_0** | **`22,093,636 B` (21.07 MiB)** | **6.07×** | **0.952611** | **0.998566** | **0.048077** | **95.1% (39/41)** | **318.58 ms** | **128.7 tok/s** | **29.41 MiB** |
| **ESM-2 150M** | FP32 | `592,642,692 B` (565.19 MiB) | 1.00× | 0.999999 | 1.000000 | 0.000000 | **100.0% (41/41)** | 2,390.36 ms | 17.2 tok/s | 575.01 MiB |
| | Q8_0 | `168,646,036 B` (160.83 MiB) | 3.51× | 0.999934 | 0.999998 | 0.000042 | **100.0% (41/41)** | 1,430.16 ms | 28.7 tok/s | 170.61 MiB |
| | **Q4_0** | **`94,978,756 B` (90.58 MiB)** | **6.24×** | **0.981715** | **0.999266** | **0.020517** | **100.0% (41/41)** | **1,367.83 ms** | **30.0 tok/s** | **100.39 MiB** |
| **ESM-2 650M** | FP32 | `2,604,339,332 B` (2,483.69 MiB) | 1.00× | 0.999999 | 1.000000 | 0.000000 | **100.0% (41/41)** | 8,692.15 ms | 4.72 tok/s | 2,501.33 MiB |
| | Q8_0 | `738,899,556 B` (704.67 MiB) | 3.52× | 0.999913 | 0.999996 | 0.000039 | **100.0% (41/41)** | 4,669.89 ms | 8.78 tok/s | 722.41 MiB |
| | **Q4_0** | **`414,617,796 B` (395.41 MiB)** | **6.28×** | **0.980842** | **0.999292** | **0.019508** | **97.6% (40/41)** | **4,535.47 ms** | **9.04 tok/s** | **413.07 MiB** |
| **ESM-2 3B** | FP32 | `11,356,354,692 B` (10.58 GiB) | 1.00× | *Ref HF* | *Ref HF* | *Ref HF* | *Ref HF* | *N/A (>11 GiB RAM)* | *N/A* | *Streaming Solamente* |
| | **Q8_0** | **`3,216,540,676 B` (3.00 GiB)** | **3.53×** | **0.999983** | **0.999998** | **0.000027** | **100.0% (41/41)** | **18,615.93 ms** | **2.20 tok/s** | **3,102.18 MiB** |
| | **Q4_0** | **`1,801,205,956 B` (1.68 GiB)** | **6.30×** | **0.994156** | **0.999422** | **0.006447** | **100.0% (41/41)** | **19,368.89 ms** | **2.12 tok/s** | **1,752.57 MiB** |

> [!NOTE]
> **Notas de Fidelidad, Tiled GEMM y Generación de Referencias (`docs/benchmarks_v1.6.0.md` y `docs/Q4_0_SPEC.md`):**
> 1. **Paridad Exacta con HuggingFace:** En todos los modelos FP32 (`8M` a `650M`), `codon-infer` reproduce los `hidden_states` capa a capa y los `logits` finales de HuggingFace `facebook/esm2_*` con similitud de coseno ≥ 0.999999 y **100.0% de coincidencia Top-1**. Los archivos de referencia `.npz` (`scripts/ref_*.npz`) pueden generarse localmente con `python3 scripts/generate_ref_8m.py` o `scripts/colab_export_esm2.ipynb` (las pruebas de regresión omiten automáticamente los archivos `.npz` opcionales si no están presentes en un clon limpio).
> 2. **ESM-2 3B en RAM Convencional:** El cuantizador por streaming (`12.55 MiB` Peak RSS) comprime los 10.58 GiB de `esm2_3b_weights.bin` a un binario `Q4_0` de `1.68 GiB` (`1,801,205,956 B`) que carga en **545 ms** y se ejecuta en **1,752.57 MiB Peak RSS** con **0.999422 de coseno de logits** y **100.0% de coincidencia Top-1** frente a HuggingFace `esm2_t36_3B_UR50D`.
> 3. **Línea Base GEMV Monotoken vs. Motor Tiled GEMM Multi-Token (`CI-1`):** Las latencias de la tabla anterior corresponden a la línea base GEMV monotoken previa al tiling (`docs/benchmarks_v1.6.0.md`). Con el motor **Multi-Token Tiled GEMM (`CI-1`)** de v1.7.0/v1.8.0 (`M_r = 4`), las pasadas bidireccionales alcanzan una aceleración adicional de **3.76× en FP32** y **2.41×–3.44× en Q8_0/Q4_0** manteniendo paridad matemática exacta ($\cos = 1.00000000$).

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
 │ Capa Transformer (×L Capas: 6, 12, 30, 33, o 36)            │    │
 │  ├── LayerNorm 1 (dim, eps 1e-5)                             │    │
 │  ├── Atención Multi-Cabeza (20/40 Cabezas × dk: 16, 24, 32, 64)│    │
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
 │  ├── Head LayerNorm (dim, eps 1e-5)                          │
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

### 2. Pesos Preentrenados y GitHub Releases

> [!NOTE]
> **Compatibilidad Binaria al 100% con los Pesos Canónicos de `v1.6.0`:** `codon-infer v1.8.0` utiliza exactamente los mismos archivos binarios canónicos de pesos (`.bin`, `_q8_0.bin`, `_q4_0.bin`) publicados en [Release `v1.6.0`](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0). Dado que los pesos de regresión logística de contactos oficiales de Meta AI (`STRUCT-1`) están incrustados directamente en 12.2 KiB de memoria estática de Rust (`src/contact_heads.rs`) y el solver de plegamiento 3D (`STRUCT-2`) es *zero-trunk*, no se requieren cambios en el formato de archivos ni redescargas multigigabyte.

`codon-infer` estandariza los archivos binarios de pesos bajo una nomenclatura limpia y explícita:
* **ESM-2 8M (`esm2_8m_weights.bin`):** Pre-empaquetado en la raíz del repositorio (~28.7 MB FP32 canónico) para inferencia instantánea sin configuraciones adicionales, desarrollo local y pruebas unitarias rápidas sin descargas externas.
* **ESM-2 35M (`esm2_35m_weights.bin`):** Modelo de precisión completa (~127.9 MB FP32), hospedado en [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0).
* **ESM-2 150M (`esm2_150m_weights.bin`):** Modelo canónico de alta fidelidad para predicción de contactos y bioinformática (~565.2 MB FP32), hospedado en [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0).
* **ESM-2 650M Q8_0 (`esm2_650m_q8_0.bin`):** Binario cuantizado nativo con cabecera CODQ (704.7 MB Q8_0), hospedado en [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0).
* **ESM-2 650M Q4_0 (`esm2_650m_q4_0.bin`):** Binario cuantizado sub-byte a 4 bits (395.41 MiB / 414.62 MB Q4_0), hospedado en [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0).
* **ESM-2 3B Q4_0 (`esm2_3b_q4_0.bin`):** Modelo masivo de 3B cuantizado a 4 bits (1.68 GiB / 1,801.21 MB Q4_0), hospedado en [GitHub Releases v1.6.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.6.0) para ejecución con solo 1.75 GiB de RAM.
* **ESM-2 650M FP32 (`esm2_650m_weights.bin`) y ESM-2 3B (`esm2_3b_weights.bin`):** Debido al límite de 2.0 GiB por archivo en GitHub Releases, los modelos FP32 (>2 GiB: ~2.60 GB para 650M y ~11.36 GB para 3B) pueden generarse con el notebook de Google Colab (`scripts/colab_export_esm2.ipynb`) o exportarse localmente.

#### Descarga Directa de Pesos vía `curl`
Descargue directamente los pesos binarios deseados en la raíz del repositorio:
```bash
# Descargar pesos de ESM-2 35M (~128 MB FP32, hospedado en v1.6.0)
curl -L -o esm2_35m_weights.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_35m_weights.bin

# Descargar pesos de ESM-2 150M (~565 MB FP32, hospedado en v1.6.0)
curl -L -o esm2_150m_weights.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_150m_weights.bin

# Descargar pesos de ESM-2 650M Q8_0 (705 MB Q8_0, hospedado en v1.6.0)
curl -L -o esm2_650m_q8_0.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_650m_q8_0.bin

# Descargar pesos de ESM-2 650M Q4_0 (395 MiB Q4_0, hospedado en v1.6.0)
curl -L -o esm2_650m_q4_0.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_650m_q4_0.bin

# Descargar pesos de ESM-2 3B Q4_0 (1.68 GiB Q4_0, hospedado en v1.6.0)
curl -L -o esm2_3b_q4_0.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.6.0/esm2_3b_q4_0.bin

# O utilizar el script automatizado:
./scripts/download_weights.sh 35m
./scripts/download_weights.sh 150m
./scripts/download_weights.sh 650m-q8
./scripts/download_weights.sh 650m-q4
./scripts/download_weights.sh 3b-q4
```

> [!TIP]
> Si prefiere exportar pesos personalizados o reentrenados directamente desde Hugging Face Transformers o Meta `fair-esm`, puede emplear el script en Python incluido:
> ```bash
> python3 scripts/export_weights.py --model 35M
> python3 scripts/export_weights.py --model 650M
> ```

### 3. Ejecución de la Suite de Pruebas
Valide las primitivas numéricas, la paridad de vectorización SIMD, el parser FASTA y los formatos de exportación:
```bash
cargo test --release
```
Salida esperada (**57 tests ejecutados: 56 pasados, 0 fallidos, 1 ignorado**):
```text
running 57 tests
test benchmark::tests::test_benchmark_report_to_json_validity ... ok
test benchmark::tests::test_discover_available_models ... ok
test export::tests::test_dms_row_and_csv_generation ... ok
test export::tests::test_json_string_serializers ... ok
test export::tests::test_pdb_and_tsv_export ... ok
test fasta::tests::test_parse_multi_fasta_with_masks ... ok
test fasta::tests::test_parse_raw_sequence_fallback ... ok
test fasta::tests::test_parse_simple_fasta ... ok
test mcp::tests::test_mcp_dispatch_with_weights_if_available ... ok
test mcp::tests::test_mcp_json_parser_and_formatter ... ok
test mcp::tests::test_mcp_resolve_record_fallback ... ok
test mcp::tests::test_mcp_tools_list_schema ... ok
test tests::test_attention_simd_parity_dk_16_24_32 ... ok
test tests::test_attention_simd_parity_dk_64 ... ok
test tests::test_bidirectional_logits_entropy ... ok
test tests::test_esm_config_autodetection ... ok
test tests::test_evaluation_metrics_unit ... ok
test tests::test_forward_650m_if_available ... ok
test tests::test_forward_650m_q8_and_multithread_if_available ... ok
test tests::test_forward_bidirectional_if_available ... ok
test tests::test_forward_pass_with_weights_if_available ... ok
test tests::test_forward_q4_cosine_similarity_8m ... ok
test tests::test_forward_q8_cosine_similarity_8m ... ok
test tests::test_layer_norm_stability ... ok
test tests::test_log_softmax_canonical ... ok
test tests::test_mask_token_encoding_and_decoding ... ok
test tests::test_matmul_avx1_and_dispatch_parity ... ok
test tests::test_matmul_fp32_parallel_parity ... ok
test tests::test_matmul_q8_simd_parity ... ok
test tests::test_q8_0_quantization_roundtrip ... ok
test tests::test_q8_weights_save_load_roundtrip ... ok
test tests::test_simd_benchmark_parity_run ... ok
test tests::test_softmax_sum_to_one ... ok
test tests::test_tokenizer_boundaries ... ok
test tests_q4_format::test_codq_header_valid_corrupt_incoherent ... ok
test tests_q4_format::test_first_block_alignment_at_offset_64 ... ok
test tests_q4_format::test_no_ambiguity_650m_q4_vs_150m ... ok
test tests_q4_format::test_q4_0_exact_nibble_packing ... ok
test tests_q4_format::test_q4_0_roundtrip_quantize_dequantize_bound ... ok
test tests_q4_format::test_q4_0_zero_block ... ok
test tests_q4_format::test_q4_weights_save_load_roundtrip_8m ... ok
test tests_q4_integration::test_forward_q4_integration_8m ... ok
test tests_q4_integration::test_forward_q4_integration_35m ... ok
test tests_q4_integration::test_forward_q4_integration_150m ... ok
test tests_q4_integration::test_forward_q4_integration_650m ... ok
test tests_q4_integration::test_forward_q8_codq_header_integration ... ok
test tests_q4_integration::test_generate_validation_report ... ignored
test tests_q4_simd::test_q4_edge_case_extreme_values ... ok
test tests_q4_simd::test_q4_edge_case_small_out_dim ... ok
test tests_q4_simd::test_q4_edge_case_zero_blocks ... ok
test tests_q4_simd::test_q4_parallel_vs_sequential_identity_threads ... ok
test tests_q4_simd::test_q4_simd_parity_esm_dimensions ... ok
test tests_regression::test_regression_esm2_fp32_all_scales_against_ref_npz ... ok
test tests_regression::test_regression_quantized_q8_q4_against_ref_npz_including_3b ... ok
test tests_regression::test_strict_fp32_loader_rejects_corrupted_size ... ok
test tests_regression::test_tiled_gemm_and_contact_head_v1_7_0 ... ok
test benchmark::tests::test_multi_scale_simd_parity ... ok

test result: ok. 56 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
```

### 4. Subcomandos y Pipelines Bioinformáticos

Todos los subcomandos aceptan las banderas globales `--model <RUTA>`, `--threads <N>`, `--quantize <q4_0|q8_0|fp32>`, `--q4` y `--q8`. `codon-infer` detecta automáticamente si el archivo binario corresponde a **ESM-2 8M**, **35M**, **150M**, **650M** o **3B** (en precisión completa FP32 o cuantizado `Q8_0` / `Q4_0`) mediante la cabecera `CODQ` de 64 bytes o la heurística de tamaño, y autoconfigura los kernels vectoriales correspondientes.

#### A. Generación Autorregresiva de Secuencias (`generate`)
Ejecuta síntesis de secuencias peptídicas *de novo* o extensión de motivos funcionales:
```bash
# Por defecto (pesos ESM-2 8M)
codon-infer generate --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7 --max-tokens 40

# Multi-escala: Ejecución con ESM-2 35M, 150M o 650M (con 2 hilos físicos)
codon-infer generate --model esm2_35m_weights.bin --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
codon-infer generate --model esm2_150m_weights.bin --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
codon-infer generate --model esm2_650m_q8_0.bin --threads 2 --sequence "MCMPCFTTDHQMARKCDD" --temperature 0.7
```

#### B. Inpainting Bidireccional y Reconstrucción de Motivos (`inpaint`)
Reconstruye posiciones faltantes, mutadas o enmascaradas en sitios activos (indicadas mediante `*` o `<mask_1>`):
```bash
# Por defecto (ESM-2 8M)
codon-infer inpaint --sequence "MCMPC*TTDHQMARKCDD"

# Con pesos de gran capacidad ESM-2 650M Q8_0 y paralelismo multinúcleo:
codon-infer inpaint --model esm2_650m_q8_0.bin --threads 2 --sequence "MCMPC*TTDHQMARKCDD"
```

#### C. Escaneo Mutacional Profundo *In Silico* (`dms`)
Ejecuta saturación mutagénica punto a punto con mapas de calor térmicos TrueColor y exportación tabular:
```bash
# Cribado masivo de alto rendimiento con ESM-2 8M:
codon-infer dms --fasta proteins.fasta --window 128 --export-csv dms_matrix.csv --export-json dms_report.json

# Modo ultra-rápido en 1 pasada Wildtype Marginal (>16,170 mutaciones/seg, aceleración de 42×):
codon-infer dms --fasta proteins.fasta --fast --export-csv dms_fast.csv

# Escaneo de alta precisión con ESM-2 150M o 650M:
codon-infer dms --model esm2_150m_weights.bin --fasta target.fasta --window 128 --export-csv dms_150m.csv
codon-infer dms --model esm2_650m_q8_0.bin --threads 2 --fasta target.fasta --window 128 --export-csv dms_650m.csv
```

#### D. Predicción de Contactos Estructurales Residuales (`contacts`)
Calcula acoplamientos epistáticos mediante mapas de atención multi-cabeza (con corrección APC) e identifica puentes disulfuro putativos:
```bash
# Predicción de contactos 3D con ESM-2 150M o 650M para precisión de publicación:
codon-infer contacts --model esm2_150m_weights.bin --fasta proteins.fasta --export-csv contacts.csv --export-json contacts.json
codon-infer contacts --model esm2_650m_q8_0.bin --threads 2 --fasta proteins.fasta --export-csv contacts.csv

# Predicción de contactos y exportación simultánea del esqueleto 3D C_alpha ab initio (.pdb):
codon-infer contacts --model esm2_650m_q8_0.bin --threads 4 --sequence "MCMPCFTTDHQMARKCDD" --export-pdb contacts_backbone.pdb
```

#### E. Cuantización Nativa de Modelos a Q4_0 y Q8_0 (`quantize`)
Convierte los pesos de puntos de control FP32 directamente en binarios zero-copy optimizados `Q4_0` o `Q8_0` en disco con cabeceras universales `CODQ`:
```bash
# Cuantizar pesos FP32 de ESM-2 650M (~2.60 GB) a binario directo Q4_0 (395.41 MiB / 414.62 MB, 6.28x menor):
codon-infer quantize --model esm2_650m_weights.bin --output esm2_650m_q4_0.bin --format q4_0

# O utilizar autodetección automática del formato según el nombre del archivo de salida:
codon-infer quantize --model esm2_650m_weights.bin --output esm2_650m_q4_0.bin
codon-infer quantize --model esm2_650m_weights.bin --output esm2_650m_q8_0.bin

# Inferencia instantánea zero-copy utilizando el binario cuantizado resultante (autodetectado por cabecera CODQ):
codon-infer generate --model esm2_650m_q4_0.bin --threads 4 --sequence "MCMPCFTTDHQMARKCDD"
codon-infer dms --model esm2_650m_q4_0.bin --threads 4 --fasta target.fasta --window 128
```

#### F. Suite de Benchmarks de Hardware y Validación Biológica (`benchmark`)
Ejecuta la suite integral de estrés de microarquitectura y validación biológica sobre cualquier escala de modelo:
```bash
# Ejecutar benchmark comparativo entre todas las variantes de pesos descubiertas (8M, 35M, 150M, 650M, 3B):
codon-infer benchmark --suite full

# Evaluar una variante de modelo específica con configuración personalizada de hilos:
codon-infer benchmark --model esm2_35m_weights.bin --suite full
codon-infer benchmark --model esm2_650m_weights.bin --threads 2 --suite full
codon-infer benchmark --model esm2_650m_q8_0.bin --threads 2 --suite full

# Exportar reporte estructurado de rendimiento para análisis e integración continua (CI/CD):
codon-infer benchmark --suite full --export-json benchmark_report.json
```

Suites disponibles:
* `full`: Evaluación comparativa multi-modelo (latencia de carga, throughput en tokens/s, memoria de trabajo, perfil de caché) más pruebas algorítmicas completas.
* `simd`: Microbenchmarks multi-escala matriz-vector y paridad numérica AVX2+FMA vs AVX1 a través de dimensiones de proyección (320 → 1280, 480 → 1920, 640 → 2560, 1280 → 5120, 2560 → 10240) y dimensiones de cabeza d_k ∈ {16, 24, 32, 64}.
* `mlm`: Pseudo-perplejidad evolutiva y penalizaciones comparativas entre tipo salvaje y mutaciones disruptivas.
* `dms`: Calibración de escaneo mutacional profundo en motivos catalíticos de peptidasas bacterianas.
* `contacts`: Predicción de contactos espaciales y puntuación de proximidad Cys-Cys.
* `inpainting`: Reconstrucción posicional múltiple en secuencias enmascaradas bidireccionales.
* `stress`: Prueba de estabilidad continua bajo sobrecarga de caché con cuantización dinámica de rebanadas Q8.
* `scaling`: Benchmark de escalamiento de longitud de contexto (L = 32 → 256) adaptativo a la profundidad de capas del modelo.

#### G. Reconstrucción y Plegamiento Proteico 3D Zero-Trunk (`fold`)
Reconstruye coordenadas 3D $C_\alpha$ *ab initio* en formato estándar `.pdb` o `.tsv` en **15–35 ms** mediante MDS Métrico y optimización biofísica por gradiente:
```bash
# Reconstrucción 3D rápida del esqueleto desde secuencia con ESM-2 8M:
codon-infer fold --sequence "MCMPCFTTDHQMARKCDD" --export-pdb structure_8m.pdb

# Plegamiento de alta precisión con ESM-2 650M Q4_0 y motor multihilo Tiled GEMM:
codon-infer fold --model esm2_650m_q4_0.bin --threads 4 --sequence "NLYIQWLKDGGPSSGRPPPS" --export-pdb trpcage.pdb --export-tsv trpcage.tsv

# Plegar secuencia de proteína diana desde archivo multi-FASTA:
codon-infer fold --model esm2_650m_q4_0.bin --threads 4 --fasta target.fasta --export-pdb target_folded.pdb
```

### 5. Servidor Model Context Protocol (MCP)
`codon-infer-mcp` expone la totalidad de las capacidades bioinformáticas como herramientas nativas para agentes LLM (Claude Desktop, Cursor, Gemini, etc.) sobre stdio mediante JSON-RPC:
```bash
# Iniciar con los pesos por defecto de ESM-2 8M
cargo run --release --bin codon-infer-mcp

# Iniciar con pesos de mayor capacidad ESM-2 35M, 150M, 650M o 3B
cargo run --release --bin codon-infer-mcp -- --model esm2_35m_weights.bin
cargo run --release --bin codon-infer-mcp -- --model esm2_650m_q8_0.bin
```

#### Herramientas Expuestas vía MCP:
* `generate_protein`: Generación autorregresiva de secuencias a partir de `prompt`, `temperature` y `max_tokens`.
* `codon_infer_inpaint`: Reconstruye residuos enmascarados (`*`) devolviendo los 5 candidatos principales con sus probabilidades y clasificación fisicoquímica.
* `codon_infer_dms`: Ejecuta mutagénesis de saturación *in silico* desde una secuencia cruda o archivo FASTA, identificando posiciones catalíticas críticas (Δ log P ≤ -3.8) y tolerancias mutacionales.
* `codon_infer_contacts`: Calcula pares de contacto epistáticos (APC) y detecta puentes disulfuro candidatos (`Cys-Cys`).
* `codon_infer_fold`: Reconstruye el plegamiento proteico 3D de nivel publicación (coordenadas $C_\alpha$ en formato `.pdb`) mediante MDS Zero-Trunk y optimización por gradiente biofísico.

---

## 🛠️ Referencia Unificada de Opciones del CLI

### Banderas Globales
| Opción | Descripción | Valor por Defecto |
| :--- | :--- | :--- |
| `-h`, `--help` | Muestra el panel de ayuda con la guía de subcomandos | - |
| `-V`, `--version` | Muestra la versión y el motor vectorial SIMD activo | - |
| `-m`, `--model <RUTA>` | Ruta al binario personalizado de pesos ESM-2 en FP32, Q8_0 o Q4_0 (o `all` para benchmark comparativo) | `esm2_8m_weights.bin` |
| `-t`, `--threads <N>` | Hilos de trabajo para la ejecución paralela matriz-vector | Núcleos físicos detectados |
| `--quantize <q4_0|q8_0|fp32>`, `--q4`, `--q8` | Modo de cuantización en ejecución (`auto`, `fp32`, `q8_0`, `q4_0`). `--q4` / `--q8` son atajos | `auto` |
| `--color <auto|always|never>` | Controla el estilizado de la terminal mediante ANSI TrueColor | `auto` |
| `--no-color` | Desactiva terminantemente los códigos de escape ANSI (estándar NO_COLOR) | `false` |

### Opciones de Pipeline
| Opción | Subcomandos Aplicables | Descripción | Valor por Defecto |
| :--- | :--- | :--- | :--- |
| `-s`, `--sequence <CADENA>` | `generate`, `inpaint`, `dms`, `contacts`, `fold` | Secuencia de aminoácidos suministrada de forma directa | `"M"` (solo en `generate`; obligatorio si se omite `--fasta` en los demás subcomandos) |
| `-f`, `--fasta <RUTA>` | `generate`, `inpaint`, `dms`, `contacts`, `fold` | Ruta al archivo FASTA (o `-` para leer de stdin) | - |
| `--export-pdb <RUTA>` | `contacts`, `fold` | Ruta de archivo de destino para coordenadas atómicas en formato `.pdb` | - |
| `--export-tsv <RUTA>` | `fold` | Ruta de archivo de destino para la tabla de coordenadas 3D separadas por tabuladores | - |
| `--output <RUTA>`, `-o` | `quantize` | Ruta de salida para el archivo binario cuantizado (`*_q4_0.bin` o `*_q8_0.bin`) | - |
| `--format <q4_0|q8_0>`, `-f` | `quantize` | Formato objetivo de cuantización (opcional si es inferible por el nombre del archivo) | `q8_0` |
| `--export-csv <RUTA>`| `dms`, `contacts` | Ruta de destino para la matriz rectangular o tabla CSV | - |
| `--export-json <RUTA>`| `dms`, `contacts`, `inpaint`, `fold`, `benchmark` | Ruta de destino para el informe estructurado en JSON | - |
| `--fast`, `--wt-marginal` | `dms` | Modo de escaneo rápido en 1 pasada Wildtype Marginal (acelera hasta 42×) | `false` |
| `--window <USIZE>` | `dms` | Tamaño de ventana deslizante para proteínas largas | `128` |
| `--stride <USIZE>` | `dms` | Paso de desplazamiento de la ventana deslizante | `64` |
| `--min-separation <USIZE>` | `contacts` | Separación mínima de secuencia para contactos epistáticos | `6` |
| `--top-l <F32>` | `contacts` | Umbral de factor de contactos Top L (ej. 1.0 = L contactos) | `1.0` |
| `--temperature <F32>`| `generate` | Temperatura de muestreo (0.0 = búsqueda voraz argmax) | `0.7` |
| `--max-tokens <USIZE>`| `generate` | Número máximo de nuevos residuos a generar | `1024` |
| `--seed <U64>` | `generate` | Semilla numérica personalizada para generador pseudoaleatorio | - |
| `--suite <CADENA>` | `benchmark` | Suite específica de benchmark (full, simd, mlm, dms, contacts, inpainting, stress, scaling) | `"full"` |

---

## 📂 Estructura del Proyecto

```text
codon-infer/
├── .cargo/
│   └── config.toml         # Banderas de compilación nativa (-C target-cpu=native)
├── .gitignore              # Filtro de artefactos generados y copias de seguridad
├── Cargo.toml              # Especificación cero dependencias (v1.8.0, edición 2024)
├── esm2_8m_weights.bin     # Pesos FP32 de ESM-2 8M (7.52M parámetros, ~30 MB)
├── scripts/
│   ├── download_weights.sh # Descargador veloz de pesos desde GitHub Releases
│   ├── export_weights.py   # Exportador de pesos desde HuggingFace y fair-esm
│   ├── generate_ref_8m.py  # Generador local de referencias .npz de HuggingFace
│   └── colab_export_esm2.ipynb # Exportador Google Colab de pesos canónicos y referencias HF
├── src/
│   ├── lib.rs              # Núcleo del motor: ThreadPool persistente, Tiled GEMM, Q8/Q4, atención SIMD
│   ├── main.rs             # Despachador de CLI y parser de argumentos de subcomandos
│   ├── ui.rs               # Paletas TrueColor ANSI, mapas de gradiente térmico y barra de progreso
│   ├── fasta.rs            # Lector de flujos y secuencias multi-FASTA de cero copias
│   ├── export.rs           # Serializadores de datos CSV, TSV, JSON y PDB para análisis de publicación
│   ├── pipeline.rs         # Controladores de DMS, contactos, inpainting y plegamiento 3D STRUCT-2
│   ├── contact_heads.rs    # Pesos y sesgos de regresión logística del cabezal oficial de Meta AI
│   ├── benchmark.rs        # Suite automatizada de estrés de microarquitectura y validación biológica
│   ├── mcp.rs              # Servidor JSON-RPC MCP sobre stdio sin dependencias externas
│   └── mcp_main.rs         # Punto de entrada ejecutable del servidor MCP autónomo
├── LICENSE                 # Licencia MIT
├── README.md               # Documentación técnica principal del proyecto (inglés)
└── README_ES.md            # Documentación técnica completa del proyecto (español)
```

---

## 📜 Hoja de Ruta (Roadmap)

- [x] Migración completa a Rust puro desde el prototipo preliminar en C (**ADR-01**).
- [x] Runtime con estricta ausencia de asignaciones en heap durante inferencia (`RunState`).
- [x] Vectorización SIMD dual: micro-kernels `AVX2+FMA` y `AVX1 Sandy Bridge` con despacho dinámico.
- [x] Escalamiento multi-escala de la arquitectura (8M, 35M, 150M, 650M, 3B) con atención heterogénea (d_k ∈ {16, 24, 32, 64}).
- [x] Micro-kernels vectoriales SIMD AVX-512 para el cómputo de atención con d_k = 64.
- [x] Formulación RoPE *split-half* y aproximación numérica precisa de GELU mediante `erf`.
- [x] Integración completa del cabezal clasificador `EsmLMHead` (denso canónico + GELU + LayerNorm + sesgo de decodificador).
- [x] Cuantización por bloques `Q8_0` para optimización de residencia en caché L3 y ancho de banda DRAM.
- [x] Motor de concurrencia persistente `ThreadPool` en `std` puro con barrera híbrida spin/futex.
- [x] Soporte nativo y validado para ESM-2 650M y serialización binaria zero-copy `Q8_0`.
- [x] Demonio estándar del protocolo Model Context Protocol (MCP) vía JSON-RPC (`codon-infer-mcp`).
- [x] Modularización desacoplada: `lib.rs` (biblioteca núcleo), `main.rs` (CLI), `mcp.rs` (servidor MCP).
- [x] Cuantización sub-byte INT4 (`Q4_0`) y cabecera universal `CODQ` de 64 bytes en las 5 escalas (8M a 3B).
- [x] Batched GEMM `[N × D] × [D × D]` en forward secuencial para eliminar la transmisión de pesos de DRAM por token (Tiled GEMM 3.76× speedup).
- [x] Escaneo Mutacional Profundo Canónico (BIO-1) sin heurísticas artificiales + modo rápido Wildtype Marginal (16,170 mut/s).
- [x] Cabezal de Contacto Multicapa Oficial (L × H APC + Regresión Logística) con streaming $O(N^2)$ en RAM.
- [x] Reconstrucción Geométrica 3D Zero-Trunk (STRUCT-2) con MDS Métrico + Descenso de Gradiente Biofísico (exportación PDB).
- [ ] Implementación de micro-kernels SIMD ARM NEON para Apple Silicon (M1/M2/M3/M4) y Raspberry Pi.

---

## 🔬 Citación y Agradecimientos

Si utiliza `codon-infer` en sus investigaciones de bioinformática, biología estructural o biología sintética, por favor cite:
* **Arquitectura Original ESM-2:** Lin, Z., et al. *"Evolutionary-scale prediction of atomic-level protein structure with a language model."* Science 379.6637 (2023): eade2574.
* **Autor y Arquitecto Principal:** Manuel Osbaldo Gastélum Islas ([@OsbaldoG](https://github.com/OsbaldoG))

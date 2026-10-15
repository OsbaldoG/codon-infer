# codon-infer 🧬⚡

> **Motor de Inferencia y Pipeline Bioinformático para Modelos de Lenguaje de Proteínas ESM-2 Multi-Escala, de Alto Rendimiento y Cero Asignaciones en Rust Puro (v1.6.0).**

[🇬🇧 Read in English (README.md)](README.md)

[![Rust](https://img.shields.io/badge/rust-edición_2024-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/versión-1.6.0-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-50%20pasados%2C%200%20fallidos-brightgreen.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencias-0%20(std%20puro)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/modelo-ESM--2%20(8M%20a%203B)-blueviolet.svg)](https://github.com/facebookresearch/esm)
[![SIMD](https://img.shields.io/badge/SIMD-AVX512%20%7C%20AVX2%2BFMA%20%7C%20AVX1-red.svg)]()
[![Precision](https://img.shields.io/badge/precisión-FP32%20%7C%20Q8__0%20%7C%20Q4__0-green.svg)]()
[![Throughput](https://img.shields.io/badge/Rendimiento%20CPU->200%20tokens%2Fseg-success.svg)]()

> 📖 **Nota de Idioma:** Este documento constituye la documentación oficial técnica completa en **español**. Para consultar la versión original en inglés técnico, acceda a [README.md](README.md).

`codon-infer` es un motor de inferencia y suite de análisis bioinformático ultraligero y libre de dependencias externas para la familia de modelos de lenguaje de proteínas Evolutionary Scale Modeling (**ESM-2**) en todas sus variantes de escala (`8M`, `35M`, `150M`, `650M` y `3B`), diseñado desde primeros principios matemáticos y de hardware en **Rust 100% seguro e idiomático (`std` exclusivamente)**.

Al prescindir por completo de los ecosistemas de Python, PyTorch, C/CUDA y crates externos de terceros, `codon-infer` ejecuta generación autorregresiva de secuencias, *inpainting* bidireccional de sitios activos, escaneo mutacional profundo *in silico* (DMS) y predicción de contactos estructurales mediante vectorización SIMD dinámica a más de **200 tokens/segundo en CPUs x86_64 estándar**, manteniendo una huella de memoria estática inferior a 40 MB para modelos de 8M.

---

## ⚡ Aspectos Destacados (Versión 1.6: Motor Multi-Escala Q4_0 y Cabecera Universal CODQ)

* **Motor de Cuantización Nativa por Bloques `Q4_0` a 4 Bits (`BlockQ4_0`):**
  * Representación sub-byte que empaqueta 32 nibbles de 4 bits en 16 bytes junto a un factor de escala flotante `d` de 32 bits (`#[repr(C)]`, exactamente **20 bytes** por bloque de 32 parámetros).
  * Convención de cuantización simétrica estándar compatible con llama.cpp: $d = \max / -8.0$, $id = 1/d$, $q = \text{clamp}(\text{round}(x \cdot id) + 8, 0, 15)$. El nibble bajo almacena el índice $j \in [0, 15]$ y el nibble alto el índice $j + 16$.
  * **Compresión Teórica de Matrices de 6.40×** (0.625 B/parámetro frente a 4.0 B en FP32) y hasta **6.28× de Reducción Total del Archivo** (ESM-2 650M reducido de 2.48 GB a **395.28 MB**).
  * **Reducción de ~80% en Memoria Pico (RSS):** La huella de memoria en RAM de ESM-2 650M pasa de 2,624 MB en FP32 a **527 MB en Q4_0**.
  * **Aceleración de 1.52× en Inferencia para ESM-2 650M:** Alivia el cuello de botella de ancho de banda DRAM en ejecuciones multihilo (5.53 s vs 8.41 s con 4 hilos).
* **Arquitectura de Cabecera Universal `CODQ` de 64 Bytes:**
  * Cabecera fija little-endian de 64 bytes (`b"CODQ"`, versión 1, tipo de datos, arquitectura, profundidad de capas, dimensión oculta, dimensión FFN, vocabulario).
  * Garantiza alineación estricta de 64 bytes (línea de caché y SIMD) para el primer bloque de pesos, permitiendo carga zero-copy inmediata.
  * Resuelve la ambigüedad paramétrica entre escalas cuantizadas (p. ej., 650M-Q4 ~395 MB colisionaba con 150M-Q8, 150M-Q4 ~95 MB colisionaba con 35M-FP32).
  * Retrocompatibilidad absoluta y transparente mediante fallback a `detect_from_file_len` para modelos heredados sin cabecera (p. ej., el archivo legacy `esm2_650m_q8_0.bin`).
* **Micro-Kernels Vectoriales SIMD con Manipulación de Bits:**
  * Kernels AVX2+FMA que desempaquetan 16 bytes con `_mm_srli_epi16` y `_mm_and_si128`, convierten con signo a enteros de 32 bits y flotantes con `_mm256_cvtepi8_epi32` y `_mm256_cvtepi32_ps`, y acumulan con FMA (`_mm256_fmadd_ps`).
  * Kernel de reserva AVX1 con desempaquetado entero de 128 bits y acumulación flotante de 256 bits.
  * Cero asignaciones en el heap durante el ciclo de inferencia (*hot path*).
* **CLI Multi-Formato Unificada (`quantize`, `generate`, `inpaint`, `dms`, `contacts`, `benchmark`):**
  * Autodetección dinámica del formato (FP32, Q8_0, Q4_0) mediante cabecera `CODQ` o fallback heurístico legacy.
  * Nuevas opciones en `codon-infer quantize --model <input> --output <output> [--format <q4_0|q8_0>]` con autodetección por extensión.

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
* **Kernels de Atención SIMD Dinámicos y Heterogéneos ($d_k \in \{16, 24, 32, 64\}$):**
  * Micro-kernels vectoriales AVX de 256 bits y AVX-512 de 512 bits optimizados a mano y mapeados a 2, 3, 4 u 8 registros `__m256` (o 4 registros `__m512` ZMM) por cabeza de atención, maximizando la densidad de cómputo vectorial con paridad escalar bit a bit ($\Delta < 10^{-5}$) tanto en pasadas autorregresivas causales como bidireccionales completas.
* **Subcomandos de CLI para Entornos de Producción (`generate`, `inpaint`, `dms`, `contacts`, `benchmark`, `quantize`):**
  * Interfaz de terminal expresiva y unificada con soporte de subcomandos, banderas globales (`--model`, `--threads`, `--quantize`, `--color`, `--no-color`), detección automática de TTY y diagnósticos claros de errores.
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
* **Aceleración por Hardware Multi-SIMD y Núcleo en Rust Puro:**
  * **AVX-512:** Pipeline vectorial de 512 bits para procesamiento ultraveloz de atención con $d_k=64$ en arquitecturas Intel Xeon y AMD Zen 4+.
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

| Métrica | Referencia PyTorch / Python (ESM-2 8M) | `codon-infer` v1.5 (Rust Puro + SIMD) | Ventaja de Ingeniería |
| :--- | :--- | :--- | :--- |
| **Dependencias** | Python, PyTorch, LibTorch, CUDA (~4–7 GB) | **`std` puro (0 crates, 0 dependencias dinámicas)** | **~99.9% más liviano** |
| **Tamaño del Ejecutable** | Entorno multigiagabyte | **~1.5 MB binario release compilado** | **Binario único ultra-portable** |
| **Latencia de Carga** | ~2,500 – 4,000 ms (inicialización PyTorch) | **34.8 ms (8M) / 117.4 ms (35M) / 219.98 ms (650M Q8)** | **~12x – 100x arranque más rápido** |
| **Huella en Memoria (RAM)** | ~1.5 – 2.5 GB RSS base | **~35 MB RSS estático (8M) / ~698.5 MB (650M Q8)** | **~4x – 50x más eficiente en memoria** |
| **Rendimiento de Inferencia** | ~15 – 35 tokens/seg (CPU monohilo) | **~333.3 tok/s (8M) / 7.70 tok/s (650M Q8_0 -t 2)** | **Máxima densidad de cómputo, 0 asignaciones** |
| **Asignación de Memoria** | Pausas continuas de GC y churn de tensores | **0 asignaciones de heap en el hot-path** | **Garantías deterministas en tiempo real** |

### 2. Matriz de Rendimiento Multi-Escala Real (ESM-2 8M a 650M)
Medido empíricamente con `codon-infer benchmark --suite full` bajo ejecución en CPU con despacho SIMD nativo:

| Escala del Modelo | Capas ($L$) | Dimensión Oculta ($d$) | Cabezas Atención ($H \times d_k$) | Latencia Carga Pesos | Throughput Autorregresivo | Throughput Bidireccional ($L=32$) | Buffer Estado Trabajo ($L=128$) | Jerarquía de Caché y Memoria |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ESM-2 8M** | 6 | 320 | $20 \times 16$ (2 reg AVX) | **34.8 ms** | **~333.3 tokens/s** (3.0 ms/tok) | **~418.8 tokens/s** (76.4 ms) | **911.0 KB** | **100% en Caché L3** ($\le 32$ MB) · Limitado por Cómputo |
| **ESM-2 35M** | 12 | 480 | $20 \times 24$ (3 reg AVX) | **117.4 ms** | **~84.7 tokens/s** (11.8 ms/tok) | **~89.9 tokens/s** (356.1 ms) | **1,321.2 KB** | **Residencia Parcial L3 / DDR** · Balanceado Caché/DRAM |
| **ESM-2 150M** | 30 | 640 | $20 \times 32$ (4 reg AVX) | **466.8 ms** | **~18.9 tokens/s** (53.0 ms/tok) | **~20.2 tokens/s** (1,583.7 ms) | **1,731.5 KB** | **Streaming desde RAM** ($>500$ MB) · Limitado por Bus de Memoria |
| **ESM-2 650M (FP32)** | 33 | 1280 | $20 \times 64$ (8 reg AVX) | **751.2 ms** | **4.57 tokens/s** (218.8 ms/tok) | **~4.8 tokens/s** (~6,600 ms) | **3,450.0 KB** | **Streaming desde RAM** (~2.48 GB RSS) · Limitado por Bus de Memoria |
| **ESM-2 650M (Q8_0, -t 2)** | 33 | 1280 | $20 \times 64$ (8 reg AVX) | **219.98 ms** | **7.70 tokens/s** (129.9 ms/tok) | **~8.2 tokens/s** (~3,900 ms) | **3,450.0 KB** | **Optimizado para Streaming DRAM** (~698.5 MB RSS) · Núcleos Saturados |

### 3. Escalamiento Multihilo y Cuantización en ESM-2 650M (Núcleos Físicos vs SMT)

Evaluado en procesador de doble núcleo Intel Sandy Bridge (Core i5-2520M @ 2.50 GHz, 2 núcleos físicos, 4 hilos lógicos SMT) con micro-kernels vectoriales AVX1:

| Configuración de Ejecución | Precisión de Pesos | Huella en Memoria (RAM) | Latencia de Carga en Frío | Throughput | Latencia por Token | Aceleración Relativa |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Hilo (Monocore)** | FP32 | 2,485 MB | 751.2 ms | 4.57 tok/s | 218.8 ms/tok | 1.00x (Base) |
| **1 Hilo (Monocore)** | Q8_0 | 698.5 MB | 219.98 ms | 5.15 tok/s | 194.2 ms/tok | 1.13x |
| **2 Núcleos Físicos (`-t 2`)** | **Q8_0** | **698.5 MB** | **219.98 ms** | **7.70 tok/s** | **129.9 ms/tok** | **1.68x (Óptimo)** |
| **4 Hilos SMT (`-t 4`)** | Q8_0 | 698.5 MB | 219.98 ms | 7.48 tok/s | 133.7 ms/tok | 1.64x (Contención AVX) |

> [!NOTE]
> **Saturación de Núcleos Físicos vs Hyper-Threading:** En arquitecturas donde los hilos lógicos SMT comparten el puerto de ejecución vectorial AVX de 256 bits por núcleo físico, asignar 2 hilos a los 2 núcleos reales (`-t 2`) otorga el rendimiento máximo (7.70 tok/s) y la mínima latencia de sincronización. `codon-infer` detecta automáticamente los núcleos físicos vía `detect_optimal_threads` si se omite `--threads`.

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
    if file_len >= 3_000_000_000 {
        // ESM-2 3B: ~11.4 GB (FP32) o ~3.2 GB - 3.4 GB (Q8_0)
        Ok(Self::esm2_3b())
    } else if file_len >= 2_000_000_000 {
        // ESM-2 650M: ~2.60 GB (FP32: 2,604,342,104 bytes)
        Ok(Self::esm2_650m())
    } else if file_len >= 600_000_000 {
        // ESM-2 650M: ~738.9 MB (Q8_0: 738,902,264 bytes)
        Ok(Self::esm2_650m())
    } else if file_len >= 450_000_000 {
        // ESM-2 150M: ~592.5 MB (FP32)
        Ok(Self::esm2_150m())
    } else if file_len >= 155_000_000 {
        // ESM-2 150M: ~166 MB - 178 MB (Q8_0)
        Ok(Self::esm2_150m())
    } else if file_len >= 95_000_000 {
        // ESM-2 35M: ~134 MB (FP32)
        Ok(Self::esm2_35m())
    } else if file_len >= 34_000_000 {
        // ESM-2 35M: ~38 MB - 48 MB (Q8_0)
        Ok(Self::esm2_35m())
    } else if file_len >= 22_000_000 {
        // ESM-2 8M: ~30 MB (FP32)
        Ok(Self::esm2_8m())
    } else if file_len >= 6_000_000 {
        // ESM-2 8M: ~8.5 MB - 12 MB (Q8_0)
        Ok(Self::esm2_8m())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "Tamaño binario de pesos ESM-2 no reconocido"))
    }
}
```
Cualquier archivo de pesos personalizado suministrado mediante `--model <RUTA>` es identificado y validado al arrancar en $< 0.1\text{ ms}$, redimensionando automáticamente los búferes internos, cachés KV y matrices de proyección sin sobrecoste de asignación en tiempo de ejecución.

### 2. Optimización SIMD y Despacho Dinámico de Atención Heterogénea
En el mecanismo de atención multi-cabeza, cada cabeza posee una dimensión $d_k = d_{\text{model}} / H$. Mientras que 8M, 35M, 150M y 650M estandarizan $H=20$ cabezas de atención, 3B escala a $H=40$ cabezas:
* **8M:** $d_k = 320 / 20 = 16$ flotantes
* **35M:** $d_k = 480 / 20 = 24$ flotantes
* **150M:** $d_k = 640 / 20 = 32$ flotantes
* **650M:** $d_k = 1280 / 20 = 64$ flotantes
* **3B:** $d_k = 2560 / 40 = 64$ flotantes

Debido a que cada registro AVX de 256 bits (`__m256`) almacena con precisión 8 números flotantes de precisión simple (`f32`), y cada registro AVX-512 de 512 bits (`__m512`) almacena 16 flotantes, cada $d_k$ resulta ser un múltiplo entero exacto:
$$\frac{16}{8} = 2 \text{ YMM}, \quad \frac{24}{8} = 3 \text{ YMM}, \quad \frac{32}{8} = 4 \text{ YMM}, \quad \frac{64}{8} = 8 \text{ YMM} \quad (\text{o } 4 \text{ ZMM en AVX-512})$$

`codon-infer` implementa micro-kernels vectoriales especializados y desenrollados (*loop-unrolled*) para productos escalares y acumulación de valores:
* **Producto Escalar Escalado ($Q \cdot K^\top$):** `dot_product_head_avx` procesa 16, 24, 32 o 64 dimensiones en paralelo sin bucles escalares residuales ni comprobaciones de límites (`_mm256_mul_ps`, `_mm256_add_ps`, `hsum256_ps_avx`). En hardware con AVX-512, $d_k=64$ se despacha a `dot_product_head_avx512_d64` aprovechando 4 registros ZMM y reducción horizontal de ciclo único (`_mm512_reduce_add_ps`).
* **Acumulación del Vector de Valores ($V \times \text{prob}$):** `accumulate_value_head_avx` difunde (*broadcast*) la probabilidad escalar a lo largo de un vector completo (`_mm256_set1_ps`) y ejecuta operaciones vectoriales combinadas de multiplicación y acumulación. Para $d_k=64$, AVX-512 despacha `accumulate_value_head_avx512_d64` (`_mm512_fmadd_ps`).
* **Garantía de Paridad Escalar:** Las baterías de pruebas unitarias (`test_attention_simd_parity_dk_16_24_32`, `test_attention_simd_parity_dk_64`) verifican la equivalencia matemática rigurosa frente a bucles de referencia escalares bajo un umbral de tolerancia estricto ($\Delta < 10^{-5}$) en todas las 4 dimensiones de cabeza.

### 3. Matriz Técnica Comparativa de Arquitectura y Rendimiento

| Parámetro / Métrica | ESM-2 8M (`esm2_t6_8M_UR50D`) | ESM-2 35M (`esm2_t12_35M_UR50D`) | ESM-2 150M (`esm2_t30_150M_UR50D`) | ESM-2 650M (`esm2_t33_650M_UR50D`) | ESM-2 3B (`esm2_t36_3B_UR50D`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Capas Transformer ($L$)** | 6 | 12 | 30 | 33 | 36 |
| **Dimensión Oculta ($d_{\text{model}}$)** | 320 | 480 | 640 | 1,280 | 2,560 |
| **Cabezas de Atención ($H$)** | 20 | 20 | 20 | 20 | 40 |
| **Dimensión por Cabeza ($d_k$)** | 16 | 24 | 32 | 64 | 64 |
| **Registros SIMD por Cabeza** | **2 YMM** (16 floats) | **3 YMM** (24 floats) | **4 YMM** (32 floats) | **8 YMM / 4 ZMM** (64 floats) | **8 YMM / 4 ZMM** (64 floats) |
| **Dimensión FFN ($d_{\text{ffn}}$)** | 1,280 | 1,920 | 2,560 | 5,120 | 10,240 |
| **Vocabulario** | 33 tokens | 33 tokens | 33 tokens | 33 tokens | 33 tokens |
| **Parámetros Totales** | 7,523,042 (~7.5M) | 33,517,246 (~33.5M) | 148,161,290 (~148M) | 651,085,526 (~651M) | ~2.84B (~3B) |
| **Tamaño Binario (FP32)** | 30,092,168 B (~30.1 MB) | 134,068,984 B (~134.1 MB) | 592,645,160 B (~592.6 MB) | **2,604,342,104 B (~2.60 GB)** | ~11.36 GB |
| **Tamaño Binario Cuantizado (`Q8_0`)** | ~8.5 MB | ~37.7 MB | ~166.5 MB | **738,902,264 B (~738.9 MB)** | ~3.19 GB |
| **Tamaño Binario Cuantizado (`Q4_0`)** | **5,172,344 B (~5.17 MB)** | **22,042,296 B (~22.04 MB)** | **94,909,976 B (~94.91 MB)** | **395,279,848 B (~395.28 MB)** | **~1.71 GB** |
| **Latencia de Carga de Pesos** | **34.8 ms** | **117.4 ms** | **466.8 ms** | **751 ms (FP32) / 219 ms (Q8) / 128 ms (Q4)** | **~8.5 s** |
| **Throughput Autorregresivo (Generate)** | **>315 tokens/s** (~3.1 ms/tok) | **~80 – 85 tokens/s** (~11.8 ms/tok) | **~18 – 20 tokens/s** (~53.0 ms/tok) | **4.57 tok/s (1T) -> 7.70 tok/s (2T Q8)** | **~0.8 – 1.0 tokens/s** |
| **Throughput Bidireccional (Inpaint)** | **>400 tokens/s** (~76 ms / $L=32$) | **~85 – 90 tokens/s** (~356 ms / $L=32$) | **~18 – 21 tokens/s** (~1,580 ms / $L=32$) | **~4.8 tok/s (1T) -> ~8.2 tok/s (2T Q8)** | **~0.9 – 1.1 tokens/s** |
| **Memoria de Estado de Trabajo ($L=128$)** | **~911 KB** | **~1,321 KB** | **~1,732 KB** | **~3,450 KB** | **~7,200 KB** |
| **Residencia en Jerarquía de Caché** | **100% en Caché L3** ($\le 32$ MB) | Residencia Parcial en L3 / DDR | Transmisión (*streaming*) desde RAM ($> 500$ MB) | Streaming DRAM (~2.48 GB FP32 / ~698 MB Q8 / ~395 MB Q4) | Transmisión Masiva RAM (>10 GB) |
| **Demanda de Ancho de Banda de Memoria** | **Limitado por Cómputo (AVX/SIMD)** | **Balanceado Caché/DRAM** | **Limitado por Bus de Memoria** | **Limitado por Bus (Optimizado Q4/Q8)** | **Extremadamente Limitado por DRAM** |
| **Caso de Uso Primario** | DMS *in silico* ultrarrápido, inpainting en tiempo real | DMS de alta resolución, cribado de péptidos | Predicción de contactos 3D, diseño *de novo* | Mapas de contacto de alta fidelidad, DMS fino | Modelos de lenguaje de proteínas de frontera |

### 4. Validación Empírica Multi-Escala y Rendimiento: FP32 vs. Q8_0 vs. Q4_0

Evaluación empírica en `test_peptidase.fasta` a través de todas las escalas de parámetros bajo ejecución vectorial SIMD nativa (4 hilos):

| Escala del Modelo | Formato de Precisión | Tamaño en Disco | Compresión Matrices | Compresión Archivo | Coseno Embeddings | Coseno Logits | Peor Coseno Matriz | Latencia (4 hilos) | Memoria Pico (RSS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ESM-2 8M** | FP32 | 30.13 MB | 1.00× | 1.00× | 1.0000 | 1.0000 | 1.0000 | 6.7 ms | 37 MB |
| | Q8_0 | 8.84 MB | 3.56× | 3.41× | 0.9998 | 0.9999 | 0.9999 | 7.9 ms | 38 MB |
| | **Q4_0** | **5.17 MB** | **6.40×** | **5.82×** | **0.9873** | **0.9850** | **0.9930** | **6.2 ms** | **38 MB** |
| **ESM-2 35M** | FP32 | 136.03 MB | 1.00× | 1.00× | 1.0000 | 1.0000 | 1.0000 | 25.5 ms | 154 MB |
| | Q8_0 | 38.86 MB | 3.56× | 3.50× | 0.9997 | 0.9999 | 0.9998 | 26.8 ms | 125 MB |
| | **Q4_0** | **22.04 MB** | **6.40×** | **6.17×** | **0.9927** | **0.9893** | **0.9913** | **20.5 ms** | **83 MB** |
| **ESM-2 150M** | FP32 | 585.11 MB | 1.00× | 1.00× | 1.0000 | 1.0000 | 1.0000 | 108.6 ms | 620 MB |
| | Q8_0 | 165.41 MB | 3.56× | 3.54× | 0.9998 | 0.9999 | 0.9998 | 108.8 ms | 316 MB |
| | **Q4_0** | **94.91 MB** | **6.40×** | **6.16×** | **0.9921** | **0.9984** | **0.9918** | **85.3 ms** | **175 MB** |
| **ESM-2 650M** | FP32 | 2,481.48 MB | 1.00× | 1.00× | 1.0000 | 1.0000 | 1.0000 | 8,410 ms | 2,624 MB |
| | Q8_0 | 704.67 MB | 3.56× | 3.52× | 0.9998 | 0.9998 | 0.9998 | 5,820 ms | 785 MB |
| | **Q4_0** | **395.28 MB** | **6.40×** | **6.28×** | **0.9924** | **0.9581** | **0.9920** | **5,530 ms** | **527 MB** |

> [!NOTE]
> **Conclusiones Clave de Fidelidad y Memoria:**
> 1. **Alivio de Ancho de Banda DRAM:** En ESM-2 650M, Q4_0 produce una **aceleración de latencia de 1.52×** frente a FP32 (5.53 s vs 8.41 s) y una **reducción de RAM del 79.8%** (de 2,624 MB a 527 MB).
> 2. **Alta Fidelidad Latente:** A través de todas las escalas, las representaciones internas mantienen un coseno $\ge 0.987$ a $0.993$. Las matrices cuantizadas individuales conservan un coseno en el peor caso $\ge 0.9913$.
> 3. **Deriva por Capas en 650M:** El coseno de logits finales de 0.9581 en 650M se debe a la de-proyección del cabezal LM sobre 33 tokens, mientras que los embeddings semánticos ocultos en las 33 capas se mantienen excepcionalmente estables con similitud de coseno $\ge 0.9917$.

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

`codon-infer` estandariza los archivos binarios de pesos bajo una nomenclatura limpia y explícita:
* **ESM-2 8M (`esm2_8m_weights.bin`):** Pre-empaquetado en la raíz del repositorio (~30.1 MB FP32) para inferencia instantánea sin configuraciones adicionales, desarrollo local y pruebas unitarias rápidas sin descargas externas.
* **ESM-2 35M (`esm2_35m_weights.bin`):** Modelo de capacidad intermedia (~134.1 MB FP32), hospedado en [GitHub Releases v1.4.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.4.0).
* **ESM-2 150M (`esm2_150m_weights.bin`):** Modelo de alta fidelidad para predicción de contactos y bioinformática con calidad de publicación (~592.6 MB FP32), hospedado en [GitHub Releases v1.4.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.4.0).
* **ESM-2 650M Q8_0 (`esm2_650m_q8_0.bin`):** Binario cuantizado nativo de alta capacidad (738.9 MB Q8_0), hospedado en [GitHub Releases v1.5.0](https://github.com/OsbaldoG/codon-infer/releases/tag/v1.5.0) para arranque zero-copy en 220 ms.
* **ESM-2 650M FP32 (`esm2_650m_weights.bin`) y ESM-2 3B (`esm2_3b_weights.bin`):** Debido al límite de 2.0 GiB por archivo en GitHub Releases, los modelos de precisión completa que superan 2 GiB (~2.60 GB para 650M FP32 y ~11.4 GB para 3B FP32) se exportan localmente directamente desde HuggingFace/fair-esm mediante `python3 scripts/export_weights.py`.

#### Descarga Directa de Pesos vía `curl`
Descargue directamente los pesos binarios deseados en la raíz del repositorio sin necesidad de reconstruirlos en Google Colab:
```bash
# Descargar pesos de ESM-2 35M (~134 MB, hospedado en v1.4.0)
curl -L -o esm2_35m_weights.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.4.0/esm2_35m_weights.bin

# Descargar pesos de ESM-2 150M (~592 MB, hospedado en v1.4.0)
curl -L -o esm2_150m_weights.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.4.0/esm2_150m_weights.bin

# Descargar pesos de ESM-2 650M Q8_0 (738.9 MB, hospedado en v1.5.0)
curl -L -o esm2_650m_q8_0.bin https://github.com/OsbaldoG/codon-infer/releases/download/v1.5.0/esm2_650m_q8_0.bin

# O utilizar el script automatizado:
./scripts/download_weights.sh 35m
./scripts/download_weights.sh 150m
./scripts/download_weights.sh 650m
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
Salida esperada:
```text
running 51 tests
test benchmark::tests::test_benchmark_report_to_json_validity ... ok
test benchmark::tests::test_discover_available_models ... ok
test export::tests::test_json_string_serializers ... ok
test fasta::tests::test_parse_multi_fasta_with_masks ... ok
test fasta::tests::test_parse_simple_fasta ... ok
test fasta::tests::test_parse_raw_sequence_fallback ... ok
test mcp::tests::test_mcp_json_parser_and_formatter ... ok
test mcp::tests::test_mcp_resolve_record_fallback ... ok
test export::tests::test_dms_row_and_csv_generation ... ok
test tests::test_attention_simd_parity_dk_16_24_32 ... ok
test tests::test_attention_simd_parity_dk_64 ... ok
test mcp::tests::test_mcp_tools_list_schema ... ok
test tests::test_esm_config_autodetection ... ok
test benchmark::tests::test_multi_scale_simd_parity ... ok
test tests::test_bidirectional_logits_entropy ... ok
test tests::test_forward_bidirectional_if_available ... ok
test tests::test_forward_pass_with_weights_if_available ... ok
test tests::test_forward_q4_cosine_similarity_8m ... ok
test mcp::tests::test_mcp_dispatch_with_weights_if_available ... ok
test tests::test_layer_norm_stability ... ok
test tests::test_log_softmax_canonical ... ok
test tests::test_mask_token_encoding_and_decoding ... ok
test tests::test_matmul_avx1_and_dispatch_parity ... ok
test tests::test_matmul_fp32_parallel_parity ... ok
test tests::test_matmul_q8_simd_parity ... ok
test tests::test_q8_0_quantization_roundtrip ... ok
test tests::test_forward_q8_cosine_similarity_8m ... ok
test tests::test_simd_benchmark_parity_run ... ok
test tests::test_softmax_sum_to_one ... ok
test tests::test_tokenizer_boundaries ... ok
test tests_q4_format::test_codq_header_valid_corrupt_incoherent ... ok
test tests::test_q8_weights_save_load_roundtrip ... ok
test tests_q4_format::test_no_ambiguity_650m_q4_vs_150m ... ok
test tests_q4_format::test_q4_0_exact_nibble_packing ... ok
test tests_q4_format::test_q4_0_roundtrip_quantize_dequantize_bound ... ok
test tests_q4_format::test_q4_0_zero_block ... ok
test tests_q4_format::test_first_block_alignment_at_offset_64 ... ok
test tests_q4_format::test_q4_weights_save_load_roundtrip_8m ... ok
test tests_q4_integration::test_forward_q4_integration_35m ... ok
test tests_q4_integration::test_forward_q4_integration_150m ... ok
test tests_q4_integration::test_forward_q4_integration_8m ... ok
test tests_q4_integration::test_forward_q8_codq_header_integration ... ok
test tests_q4_simd::test_q4_edge_case_extreme_values ... ok
test tests_q4_simd::test_q4_edge_case_small_out_dim ... ok
test tests_q4_simd::test_q4_edge_case_zero_blocks ... ok
test tests_q4_simd::test_q4_parallel_vs_sequential_identity_threads ... ok
test tests_q4_simd::test_q4_simd_parity_esm_dimensions ... ok
test tests_q4_integration::test_forward_q4_integration_650m ... ok
test tests::test_forward_650m_if_available ... ok
test tests::test_forward_650m_q8_and_multithread_if_available ... ok

test result: ok. 50 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
```

### 4. Subcomandos y Pipelines Bioinformáticos

Todos los subcomandos aceptan las banderas globales `--model <RUTA>`, `--threads <N>` y `--quantize q8_0`. `codon-infer` detecta automáticamente si el archivo binario corresponde a **ESM-2 8M**, **35M**, **150M**, **650M** o **3B** (en precisión completa FP32 o cuantizado `Q8_0`) y autoconfigura los kernels vectoriales correspondientes.

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
```

#### E. Cuantización Nativa de Modelos a Q4_0 y Q8_0 (`quantize`)
Convierte los pesos de puntos de control FP32 directamente en binarios zero-copy optimizados `Q4_0` o `Q8_0` en disco con cabeceras universales `CODQ`:
```bash
# Cuantizar pesos FP32 de ESM-2 650M (~2.48 GB) a binario directo Q4_0 (~395.28 MB, 6.28x menor):
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
* `simd`: Microbenchmarks multi-escala matriz-vector y paridad numérica AVX2+FMA vs AVX1 a través de dimensiones de proyección ($320 \to 1280$, $480 \to 1920$, $640 \to 2560$, $1280 \to 5120$, $2560 \to 10240$) y dimensiones de cabeza $d_k \in \{16, 24, 32, 64\}$.
* `mlm`: Pseudo-perplejidad evolutiva y penalizaciones comparativas entre tipo salvaje y mutaciones disruptivas.
* `dms`: Calibración de escaneo mutacional profundo en motivos catalíticos de peptidasas bacterianas.
* `contacts`: Predicción de contactos espaciales y puntuación de proximidad Cys-Cys.
* `inpainting`: Reconstrucción posicional múltiple en secuencias enmascaradas bidireccionales.
* `stress`: Prueba de estabilidad continua bajo sobrecarga de caché con cuantización dinámica de rebanadas Q8.
* `scaling`: Benchmark de escalamiento de longitud de contexto ($L = 32 \to 256$) adaptativo a la profundidad de capas del modelo.

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
* `codon_infer_dms`: Ejecuta mutagénesis de saturación *in silico* desde una secuencia cruda o archivo FASTA, identificando posiciones catalíticas críticas ($\Delta \log P \le -3.8$) y tolerancias mutacionales.
* `codon_infer_contacts`: Calcula pares de contacto epistáticos (APC) y detecta puentes disulfuro candidatos (`Cys-Cys`).

---

## 🛠️ Referencia Unificada de Opciones del CLI

### Banderas Globales
| Opción | Descripción | Valor por Defecto |
| :--- | :--- | :--- |
| `-h`, `--help` | Muestra el panel de ayuda con la guía de subcomandos | - |
| `-V`, `--version` | Muestra la versión y el motor vectorial SIMD activo | - |
| `--model <RUTA>` | Ruta al binario personalizado de pesos ESM-2 en FP32 o Q8_0 (o `all` para benchmark comparativo) | `esm2_8m_weights.bin` |
| `-t`, `--threads <N>` | Hilos de trabajo para la ejecución paralela matriz-vector | Núcleos físicos detectados |
| `--quantize <q8_0>`, `--q8`| Ejecuta o convierte utilizando cuantización por bloques `Q8_0` nativa | `auto` |
| `--export-json <RUTA>`| Exporta métricas de benchmark o análisis directamente a JSON | - |
| `--color <auto\|always\|never>` | Controla el estilizado de la terminal mediante ANSI TrueColor | `auto` |
| `--no-color` | Desactiva terminantemente los códigos de escape ANSI (estándar NO_COLOR) | `false` |

### Opciones de Pipeline
| Opción | Subcomandos Aplicables | Descripción | Valor por Defecto |
| :--- | :--- | :--- | :--- |
| `--sequence <CADENA>` | `generate`, `inpaint`, `dms`, `contacts` | Secuencia de aminoácidos suministrada de forma directa | `"M"` |
| `--fasta <RUTA>` | `generate`, `inpaint`, `dms`, `contacts` | Ruta al archivo FASTA (o `-` para leer de stdin) | - |
| `--output <RUTA>` | `quantize` | Ruta de salida para el archivo binario cuantizado (`*_q8_0.bin`) | - |
| `--export-csv <RUTA>`| `dms`, `contacts` | Ruta de destino para la matriz rectangular o tabla CSV | - |
| `--export-json <RUTA>`| `dms`, `contacts`, `benchmark` | Ruta de destino para el informe estructurado en JSON | - |
| `--window <USIZE>` | `dms` | Tamaño de ventana deslizante para proteínas largas | `128` |
| `--stride <USIZE>` | `dms` | Paso de desplazamiento de la ventana deslizante | `64` |
| `--min-separation <USIZE>` | `contacts` | Separación mínima de secuencia para contactos epistáticos | `6` |
| `--top-l <F32>` | `contacts` | Umbral de factor de contactos Top L (ej. 1.0 = $L$ contactos) | `1.0` |
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
├── Cargo.toml              # Especificación cero dependencias (v1.5.0, edición 2024)
├── esm2_8m_weights.bin     # Pesos FP32 de ESM-2 8M (7.52M parámetros, ~30 MB)
├── scripts/
│   ├── download_weights.sh # Descargador veloz de pesos desde GitHub Releases
│   └── export_weights.py   # Exportador de pesos desde HuggingFace y fair-esm
├── src/
│   ├── lib.rs              # Núcleo del motor: ThreadPool persistente, GEMV Q8_0, atención SIMD, EsmLMHead
│   ├── main.rs             # Despachador de CLI y parser de argumentos de subcomandos
│   ├── ui.rs               # Paletas TrueColor ANSI, mapas de gradiente térmico y barra de progreso
│   ├── fasta.rs            # Lector de flujos y secuencias multi-FASTA de cero copias
│   ├── export.rs           # Serializadores de datos CSV y JSON para análisis de publicación
│   ├── pipeline.rs         # Controladores de ventana deslizante para DMS, contactos e inpainting
│   ├── benchmark.rs        # Suite automatizada de estrés de hardware y validación biológica
│   ├── mcp.rs              # Servidor JSON-RPC MCP sobre stdio sin dependencias externas
│   └── mcp_main.rs         # Punto de entrada ejecutable del servidor MCP autónomo
├── LICENSE                 # Licencia MIT
├── README.md               # Documentación principal del proyecto (en inglés)
└── README_ES.md            # Documentación completa del proyecto (en español)
```

---

## 📜 Hoja de Ruta (Roadmap)

- [x] Migración completa a Rust puro desde el prototipo preliminar en C (**ADR-01**).
- [x] Runtime con estricta ausencia de asignaciones en heap durante inferencia (`RunState`).
- [x] Vectorización SIMD dual: micro-kernels `AVX2+FMA` y `AVX1 Sandy Bridge` con despacho dinámico.
- [x] Escalamiento multi-escala de la arquitectura (8M, 35M, 150M, 650M, 3B) con atención heterogénea ($d_k \in \{16, 24, 32, 64\}$).
- [x] Micro-kernels vectoriales SIMD AVX-512 para el cómputo de atención con $d_k=64$.
- [x] Formulación RoPE *split-half* y aproximación numérica precisa de GELU mediante `erf`.
- [x] Integración completa del cabezal clasificador `EsmLMHead` (denso canónico + GELU + LayerNorm + sesgo de decodificador).
- [x] Cuantización por bloques `Q8_0` para optimización de residencia en caché L3 y ancho de banda DRAM.
- [x] Motor de concurrencia persistente `ThreadPool` en `std` puro con barrera híbrida spin/futex.
- [x] Soporte nativo y validado para ESM-2 650M y serialización binaria zero-copy `Q8_0`.
- [x] Demonio estándar del protocolo Model Context Protocol (MCP) vía JSON-RPC (`codon-infer-mcp`).
- [x] Modularización desacoplada: `lib.rs` (biblioteca núcleo), `main.rs` (CLI), `mcp.rs` (servidor MCP).
- [ ] Implementación de micro-kernels SIMD ARM NEON para Apple Silicon (M1/M2/M3/M4) y Raspberry Pi.
- [ ] Cuantización sub-byte INT4 (`Q4_0`) para dispositivos biomédicos integrados de consumo ultra-bajo.

---

## 🔬 Citación y Agradecimientos

Si utiliza `codon-infer` en sus investigaciones de bioinformática, biología estructural o biología sintética, por favor cite:
* **Arquitectura Original ESM-2:** Lin, Z., et al. *"Evolutionary-scale prediction of atomic-level protein structure with a language model."* Science 379.6637 (2023): eade2574.
* **Autor y Arquitecto Principal:** Manuel Osbaldo Gastélum Islas ([@OsbaldoG](https://github.com/OsbaldoG))

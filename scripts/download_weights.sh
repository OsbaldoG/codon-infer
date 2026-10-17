#!/usr/bin/env bash
set -e

# codon-infer: Pretrained Weights Downloader from GitHub Releases
# Repository: https://github.com/OsbaldoG/codon-infer

# Note: codon-infer v1.8.0 is 100% binary-compatible with canonical v1.6.0 weight assets
# (.bin / CODQ). STRUCT-1 contact regression weights are embedded directly in 12.2 KiB of
# static Rust memory and STRUCT-2 folding is zero-trunk, so no weight re-download is needed.
WEIGHTS_TAG="${WEIGHTS_TAG:-v1.6.0}"
BASE_URL="https://github.com/OsbaldoG/codon-infer/releases/download/${WEIGHTS_TAG}"

show_help() {
    local code=${1:-0}
    echo "Usage: $0 [8m|35m|150m|650m|650m-q8|650m-q4|3b-q4|650m-fp32|3b-fp32|all]"
    echo ""
    echo "Download pretrained canonical ESM-2 weights (100% compatible with codon-infer v1.8.0)"
    echo "from GitHub Releases (${WEIGHTS_TAG}) into the current directory:"
    echo "  8m        -> esm2_8m_weights.bin     (28.7 MB FP32, canonical ${WEIGHTS_TAG} asset)"
    echo "  35m       -> esm2_35m_weights.bin    (127.9 MB FP32, canonical ${WEIGHTS_TAG} asset)"
    echo "  150m      -> esm2_150m_weights.bin   (565.2 MB FP32, canonical ${WEIGHTS_TAG} asset)"
    echo "  650m      -> esm2_650m_q8_0.bin      (704.7 MB Q8_0 CODQ, ${WEIGHTS_TAG} asset)"
    echo "  650m-q8   -> esm2_650m_q8_0.bin      (704.7 MB Q8_0 CODQ, ${WEIGHTS_TAG} asset)"
    echo "  650m-q4   -> esm2_650m_q4_0.bin      (395.4 MB Q4_0 CODQ, ${WEIGHTS_TAG} asset)"
    echo "  3b-q4     -> esm2_3b_q4_0.bin        (1.68 GiB Q4_0 CODQ, ${WEIGHTS_TAG} asset)"
    echo "  650m-fp32 -> Notice: generate via export_weights.py (>2.0 GiB GitHub limit)"
    echo "  3b-fp32   -> Notice: generate via export_weights.py (>2.0 GiB GitHub limit)"
    echo "  all       -> Download all hosted models (8M, 35M, 150M FP32; 650M Q8; 650M Q4; 3B Q4)"
    exit "${code}"
}

download_file() {
    local file=$1
    local url="${BASE_URL}/${file}"
    echo "==> Downloading ${file} from ${url} ..."
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --progress-bar -o "${file}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "${file}" "${url}"
    else
        echo "Error: Neither curl nor wget is available." >&2
        exit 1
    fi
    echo "==> Successfully downloaded ${file}"
}

TARGET="${1:-35m}"

case "${TARGET,,}" in
    8m)
        download_file "esm2_8m_weights.bin"
        ;;
    35m)
        download_file "esm2_35m_weights.bin"
        ;;
    150m)
        download_file "esm2_150m_weights.bin"
        ;;
    650m|650m-q8|q8)
        download_file "esm2_650m_q8_0.bin"
        ;;
    650m-q4|q4)
        download_file "esm2_650m_q4_0.bin"
        ;;
    3b|3b-q4)
        download_file "esm2_3b_q4_0.bin"
        ;;
    650m-fp32)
        echo "==> Notice: Due to GitHub's 2.0 GiB release asset limit, esm2_650m_weights.bin (~2.60 GB FP32) is not hosted on GitHub Releases."
        echo "    You can export it directly using Colab/HuggingFace via:"
        echo "    python3 scripts/export_weights.py --model 650M"
        ;;
    3b-fp32|3b-q8)
        echo "==> Notice: Due to GitHub's 2.0 GiB release asset limit, full FP32/Q8 checkpoints (>2.0 GiB) are not hosted on GitHub Releases."
        echo "    Use esm2_3b_q4_0.bin (1.68 GiB, hosted in v1.6.0) or export locally with scripts/export_weights.py."
        ;;
    all)
        download_file "esm2_8m_weights.bin"
        download_file "esm2_35m_weights.bin"
        download_file "esm2_150m_weights.bin"
        download_file "esm2_650m_q8_0.bin"
        download_file "esm2_650m_q4_0.bin"
        download_file "esm2_3b_q4_0.bin"
        echo "==> Note: All release v1.6.0 models downloaded successfully."
        ;;
    -h|--help|help)
        show_help 0
        ;;
    *)
        echo "Unknown target: ${TARGET}" >&2
        show_help 1
        ;;
esac

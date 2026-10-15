#!/usr/bin/env bash
set -e

# codon-infer: Pretrained Weights Downloader from GitHub Releases
# Repository: https://github.com/OsbaldoG/codon-infer

TAG="v1.4.0"
BASE_URL="https://github.com/OsbaldoG/codon-infer/releases/download/${TAG}"

show_help() {
    local code=${1:-0}
    echo "Usage: $0 [8m|35m|150m|650m|3b|all]"
    echo ""
    echo "Download pretrained ESM-2 weights from GitHub Releases into the current directory:"
    echo "  8m    -> esm2_8m_weights.bin   (~30 MB FP32)"
    echo "  35m   -> esm2_35m_weights.bin  (~133 MB FP32)"
    echo "  150m  -> esm2_150m_weights.bin (~590 MB FP32)"
    echo "  650m  -> esm2_650m_weights.bin (~2.6 GB FP32)"
    echo "  3b    -> esm2_3b_weights.bin   (~11.4 GB FP32)"
    echo "  all   -> Download all multi-scale variants"
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
    650m)
        download_file "esm2_650m_weights.bin"
        ;;
    3b)
        download_file "esm2_3b_weights.bin"
        ;;
    all)
        download_file "esm2_8m_weights.bin"
        download_file "esm2_35m_weights.bin"
        download_file "esm2_150m_weights.bin"
        download_file "esm2_650m_weights.bin"
        download_file "esm2_3b_weights.bin"
        ;;
    -h|--help|help)
        show_help 0
        ;;
    *)
        echo "Unknown target: ${TARGET}" >&2
        show_help 1
        ;;
esac

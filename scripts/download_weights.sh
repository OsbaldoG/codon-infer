#!/usr/bin/env bash
set -e

# codon-infer: Pretrained Weights Downloader from GitHub Releases
# Repository: https://github.com/OsbaldoG/codon-infer

TAG_V140="v1.4.0"
TAG_V150="v1.5.0"
BASE_URL_V140="https://github.com/OsbaldoG/codon-infer/releases/download/${TAG_V140}"
BASE_URL_V150="https://github.com/OsbaldoG/codon-infer/releases/download/${TAG_V150}"

show_help() {
    local code=${1:-0}
    echo "Usage: $0 [8m|35m|150m|650m|650m-q8|650m-fp32|3b|all]"
    echo ""
    echo "Download pretrained ESM-2 weights from GitHub Releases into the current directory:"
    echo "  8m        -> esm2_8m_weights.bin     (30.1 MB FP32, v1.4.0 release)"
    echo "  35m       -> esm2_35m_weights.bin    (134.1 MB FP32, v1.4.0 release)"
    echo "  150m      -> esm2_150m_weights.bin   (592.6 MB FP32, v1.4.0 release)"
    echo "  650m      -> esm2_650m_q8_0.bin      (738.9 MB Q8_0, v1.5.0 release)"
    echo "  650m-q8   -> esm2_650m_q8_0.bin      (738.9 MB Q8_0, v1.5.0 release)"
    echo "  650m-fp32 -> Notice: generate via export_weights.py (>2.0 GiB GitHub limit)"
    echo "  3b        -> Notice: generate via export_weights.py (>2.0 GiB GitHub limit)"
    echo "  all       -> Download all hosted release models (8M, 35M, 150M FP32 and 650M Q8_0)"
    exit "${code}"
}

download_file() {
    local tag=$1
    local file=$2
    local base_url
    if [ "${tag}" = "v1.4.0" ]; then
        base_url="${BASE_URL_V140}"
    else
        base_url="${BASE_URL_V150}"
    fi
    local url="${base_url}/${file}"
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
        download_file "v1.4.0" "esm2_8m_weights.bin"
        ;;
    35m)
        download_file "v1.4.0" "esm2_35m_weights.bin"
        ;;
    150m)
        download_file "v1.4.0" "esm2_150m_weights.bin"
        ;;
    650m|650m-q8|q8)
        download_file "v1.5.0" "esm2_650m_q8_0.bin"
        ;;
    650m-fp32)
        echo "==> Notice: Due to GitHub's 2.0 GiB release asset limit, esm2_650m_weights.bin (~2.60 GB FP32) is not hosted on GitHub Releases."
        echo "    You can export it directly from HuggingFace/fair-esm using:"
        echo "    python3 scripts/export_weights.py --model 650M"
        ;;
    3b|3b-fp32|3b-q8)
        echo "==> Notice: Due to GitHub's 2.0 GiB release asset limit, ESM-2 3B weights (~11.4 GB FP32) are not hosted on GitHub Releases."
        echo "    You can export it directly from HuggingFace/fair-esm using:"
        echo "    python3 scripts/export_weights.py --model 3B"
        ;;
    all)
        download_file "v1.4.0" "esm2_8m_weights.bin"
        download_file "v1.4.0" "esm2_35m_weights.bin"
        download_file "v1.4.0" "esm2_150m_weights.bin"
        download_file "v1.5.0" "esm2_650m_q8_0.bin"
        echo "==> Note on ESM-2 650M FP32 (~2.60 GB) and ESM-2 3B (~11.4 GB):"
        echo "    These models exceed GitHub's 2.0 GiB limit. Generate them locally with:"
        echo "    python3 scripts/export_weights.py --model 650M"
        echo "    python3 scripts/export_weights.py --model 3B"
        ;;
    -h|--help|help)
        show_help 0
        ;;
    *)
        echo "Unknown target: ${TARGET}" >&2
        show_help 1
        ;;
esac

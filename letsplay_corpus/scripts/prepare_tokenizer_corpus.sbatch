#!/usr/bin/env bash
#SBATCH --job-name=conv-bpe-corpus
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

if (( $# < 3 || $# > 4 )); then
    echo "Usage: $0 MANIFEST_CSV DATA_ROOT ANALYSIS_ROOT [SILENCE_THRESHOLD_SECONDS]" >&2
    exit 2
fi

MANIFEST_CSV="$1"
DATA_ROOT="$2"
ANALYSIS_ROOT="$3"
SILENCE_THRESHOLD_SECONDS="${4:-0.8}"
OUTPUT_DIR="$DATA_ROOT/derived/letsplay/tokenizer_corpus"

for path in "$MANIFEST_CSV" "$DATA_ROOT" "$ANALYSIS_ROOT"; do
    [[ -e "$path" ]] || {
        echo "ERROR: required path not found: $path" >&2
        exit 1
    }
done

python "$ANALYSIS_ROOT/letsplay_corpus/scripts/prepare_tokenizer_corpus.py" \
    --manifest "$MANIFEST_CSV" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --silence-threshold-seconds "$SILENCE_THRESHOLD_SECONDS"

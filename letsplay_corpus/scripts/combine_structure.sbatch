#!/usr/bin/env bash
#SBATCH --job-name=conv-combine-structure
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 MANIFEST_CSV DATA_ROOT ANALYSIS_ROOT" >&2
    exit 2
fi

MANIFEST_CSV="$1"
DATA_ROOT="$2"
ANALYSIS_ROOT="$3"
OUTPUT_DIR="$DATA_ROOT/results/letsplay_structure"

for path in "$MANIFEST_CSV" "$DATA_ROOT" "$ANALYSIS_ROOT"; do
    [[ -e "$path" ]] || {
        echo "ERROR: required path not found: $path" >&2
        exit 1
    }
done

python "$ANALYSIS_ROOT/letsplay_corpus/scripts/combine_structure.py" \
    --manifest "$MANIFEST_CSV" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR"

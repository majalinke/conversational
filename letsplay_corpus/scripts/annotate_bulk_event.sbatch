#!/usr/bin/env bash
#SBATCH --job-name=conv-syntax
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

if (( $# != 4 )); then
    echo "Usage: $0 MANIFEST_CSV DATA_ROOT ANALYSIS_ROOT TOKENIZER_ROOT" >&2
    exit 2
fi

MANIFEST_CSV="$1"
DATA_ROOT="$2"
ANALYSIS_ROOT="$3"
TOKENIZER_ROOT="$4"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-}"

if [[ -z "$TASK_INDEX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
    exit 2
fi

for path in "$MANIFEST_CSV" "$DATA_ROOT" "$ANALYSIS_ROOT" "$TOKENIZER_ROOT"; do
    [[ -e "$path" ]] || {
        echo "ERROR: required path not found: $path" >&2
        exit 1
    }
done

EVENT_ID="$(
python - "$MANIFEST_CSV" "$TASK_INDEX" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
index = int(sys.argv[2])
with manifest.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if index < 0 or index >= len(rows):
    raise SystemExit(f"array index {index} outside manifest range 0..{len(rows) - 1}")
print(rows[index]["event_id"])
PY
)"

TRANSCRIPT_HOST="$DATA_ROOT/derived/letsplay/${EVENT_ID}/raw_transcripts/${EVENT_ID}.whisper_raw.json"
OUTPUT_PARENT_HOST="$DATA_ROOT/derived/letsplay/${EVENT_ID}"
OUTPUT_DIR_HOST="$OUTPUT_PARENT_HOST/structure"
TEMP_DIR_HOST="$OUTPUT_PARENT_HOST/.structure.partial-${SLURM_JOB_ID}-${TASK_INDEX}"

[[ -s "$TRANSCRIPT_HOST" ]] || {
    echo "ERROR: transcript not found or empty: $TRANSCRIPT_HOST" >&2
    exit 1
}

valid_structure() {
    local directory="$1"
    [[ -s "$directory/summary.json" ]] || return 1
    [[ -s "$directory/segment_statistics.csv" ]] || return 1
    [[ -s "$directory/token_dependencies.csv" ]] || return 1
    [[ -s "$directory/word_sequence_statistics.csv" ]] || return 1
    [[ -s "$directory/processed_transcript.spacy.json" ]] || return 1

    python - "$directory/summary.json" "$EVENT_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
event_id = sys.argv[2]
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
valid = (
    payload.get("source_id") == event_id
    and payload.get("spacy_model") == "en_core_web_sm"
    and int(payload.get("segment_count", 0)) > 0
    and int(payload.get("token_count", 0)) > 0
    and int(payload.get("lexical_token_count", 0)) > 0
)
raise SystemExit(0 if valid else 1)
PY
}

if valid_structure "$OUTPUT_DIR_HOST"; then
    echo "status=skipped_existing_valid"
    echo "event_id=$EVENT_ID"
    echo "output=$OUTPUT_DIR_HOST"
    exit 0
fi

mkdir -p "$OUTPUT_PARENT_HOST"
rm -rf "$TEMP_DIR_HOST"
mkdir -p "$TEMP_DIR_HOST"
trap 'rm -rf "$TEMP_DIR_HOST"' EXIT

export TOKENIZER_REPO_ROOT="$TOKENIZER_ROOT"
export TOKENIZER_DATA_DIR="$DATA_ROOT"
export TOKENIZER_ANALYSIS_ROOT="$ANALYSIS_ROOT"
# The shared image is GPU-capable, but this worker runs on CPU-only nodes.
# Tell Enroot's NVIDIA hook to skip driver injection.
export NVIDIA_VISIBLE_DEVICES=void

TRANSCRIPT_CONTAINER="/workspace/data/derived/letsplay/${EVENT_ID}/raw_transcripts/${EVENT_ID}.whisper_raw.json"
TEMP_DIR_CONTAINER="/workspace/data/derived/letsplay/${EVENT_ID}/$(basename "$TEMP_DIR_HOST")"

bash "$TOKENIZER_ROOT/scripts/bwunicluster/run_gpu.sbatch" \
    python /workspace/analysis/letsplay_corpus/scripts/annotate_event.py \
        --transcript "$TRANSCRIPT_CONTAINER" \
        --output-dir "$TEMP_DIR_CONTAINER" \
        --model en_core_web_sm

valid_structure "$TEMP_DIR_HOST" || {
    echo "ERROR: structure output failed validation: $TEMP_DIR_HOST" >&2
    exit 1
}

if [[ -e "$OUTPUT_DIR_HOST" ]]; then
    mv "$OUTPUT_DIR_HOST" "${OUTPUT_DIR_HOST}.invalid.$(date +%s)"
fi
mv "$TEMP_DIR_HOST" "$OUTPUT_DIR_HOST"
trap - EXIT

echo "status=annotated"
echo "event_id=$EVENT_ID"
echo "output=$OUTPUT_DIR_HOST"

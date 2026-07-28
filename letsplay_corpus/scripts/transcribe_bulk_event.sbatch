#!/usr/bin/env bash
#SBATCH --job-name=conv-transcribe
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
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

mapfile -t EVENT_FIELDS < <(
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

row = rows[index]
print(row["event_id"])
print(row["principal_speaker_id"])
PY
)

EVENT_ID="${EVENT_FIELDS[0]}"
SPEAKER_ID="${EVENT_FIELDS[1]}"
AUDIO_HOST="$DATA_ROOT/raw_audio/letsplay/${EVENT_ID}.wav"
OUTPUT_DIR_HOST="$DATA_ROOT/derived/letsplay/${EVENT_ID}/raw_transcripts"
OUTPUT_HOST="$OUTPUT_DIR_HOST/${EVENT_ID}.whisper_raw.json"
TEMP_HOST="$OUTPUT_DIR_HOST/${EVENT_ID}.whisper_raw.partial-${SLURM_JOB_ID}.json"

[[ -s "$AUDIO_HOST" ]] || {
    echo "ERROR: audio file not found or empty: $AUDIO_HOST" >&2
    exit 1
}

valid_transcript() {
    local path="$1"
    [[ -s "$path" ]] || return 1

    python - "$path" "$EVENT_ID" <<'PY'
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
    and payload.get("model") == "large-v3"
    and isinstance(payload.get("segments"), list)
    and len(payload["segments"]) > 0
)
raise SystemExit(0 if valid else 1)
PY
}

if valid_transcript "$OUTPUT_HOST"; then
    echo "status=skipped_existing_valid"
    echo "event_id=$EVENT_ID"
    echo "speaker_id=$SPEAKER_ID"
    echo "output=$OUTPUT_HOST"
    exit 0
fi

mkdir -p "$OUTPUT_DIR_HOST"
rm -f "$TEMP_HOST"
trap 'rm -f "$TEMP_HOST"' EXIT

export TOKENIZER_REPO_ROOT="$TOKENIZER_ROOT"
export TOKENIZER_DATA_DIR="$DATA_ROOT"
export TOKENIZER_ANALYSIS_ROOT="$ANALYSIS_ROOT"

AUDIO_CONTAINER="/workspace/data/raw_audio/letsplay/${EVENT_ID}.wav"
TEMP_CONTAINER="/workspace/data/derived/letsplay/${EVENT_ID}/raw_transcripts/$(basename "$TEMP_HOST")"

bash "$TOKENIZER_ROOT/scripts/bwunicluster/run_gpu.sbatch" \
    python /workspace/analysis/letsplay_corpus/scripts/transcribe_event.py \
        --audio "$AUDIO_CONTAINER" \
        --output "$TEMP_CONTAINER" \
        --source-id "$EVENT_ID" \
        --model large-v3

valid_transcript "$TEMP_HOST" || {
    echo "ERROR: transcription output failed validation: $TEMP_HOST" >&2
    exit 1
}

mv "$TEMP_HOST" "$OUTPUT_HOST"
trap - EXIT

echo "status=transcribed"
echo "event_id=$EVENT_ID"
echo "speaker_id=$SPEAKER_ID"
echo "output=$OUTPUT_HOST"

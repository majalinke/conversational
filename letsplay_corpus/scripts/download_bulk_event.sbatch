#!/usr/bin/env bash
#SBATCH --job-name=conv-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00

set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 MANIFEST_CSV OUTPUT_DIR" >&2
    exit 2
fi

MANIFEST_CSV="$1"
OUTPUT_DIR="$2"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-}"

if [[ -z "$TASK_INDEX" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
    exit 2
fi

export PATH="$HOME/.local/bin:$PATH"

for command_name in python yt-dlp ffmpeg ffprobe; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "ERROR: required command is unavailable: $command_name" >&2
        exit 1
    }
done

[[ -f "$MANIFEST_CSV" ]] || {
    echo "ERROR: manifest not found: $MANIFEST_CSV" >&2
    exit 1
}

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
print(row["source_url"])
print(row["principal_speaker_id"])
PY
)

EVENT_ID="${EVENT_FIELDS[0]}"
SOURCE_URL="${EVENT_FIELDS[1]}"
SPEAKER_ID="${EVENT_FIELDS[2]}"
FINAL_WAV="$OUTPUT_DIR/${EVENT_ID}.wav"
STATUS_DIR="$OUTPUT_DIR/.download_status"
STATUS_JSON="$STATUS_DIR/${EVENT_ID}.json"

mkdir -p "$OUTPUT_DIR" "$STATUS_DIR"

valid_wav() {
    local path="$1"
    [[ -s "$path" ]] || return 1

    local properties
    properties="$(
        ffprobe -v error \
            -select_streams a:0 \
            -show_entries stream=codec_name,sample_rate,channels \
            -of csv=p=0:s=, \
            "$path"
    )"

    [[ "$properties" == pcm_s16le,16000,1 ]]
}

if valid_wav "$FINAL_WAV"; then
    echo "status=skipped_existing_valid"
    echo "event_id=$EVENT_ID"
    echo "speaker_id=$SPEAKER_ID"
    echo "output=$FINAL_WAV"
    exit 0
fi

if [[ -e "$FINAL_WAV" ]]; then
    mv "$FINAL_WAV" "${FINAL_WAV}.invalid.$(date +%s)"
fi

TEMP_ROOT="${TMPDIR:-/tmp}/conversational-download-${SLURM_JOB_ID:-manual}-${TASK_INDEX}"
rm -rf "$TEMP_ROOT"
mkdir -p "$TEMP_ROOT"
trap 'rm -rf "$TEMP_ROOT"' EXIT

STARTED_AT="$(date --iso-8601=seconds)"

echo "event_id=$EVENT_ID"
echo "speaker_id=$SPEAKER_ID"
echo "source_url=$SOURCE_URL"
echo "task_index=$TASK_INDEX"

yt-dlp \
    --no-playlist \
    --retries 10 \
    --fragment-retries 10 \
    --concurrent-fragments 4 \
    -f "bestaudio/best" \
    -x --audio-format wav \
    --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
    -o "$TEMP_ROOT/${EVENT_ID}.%(ext)s" \
    "$SOURCE_URL"

TEMP_WAV="$TEMP_ROOT/${EVENT_ID}.wav"
valid_wav "$TEMP_WAV" || {
    echo "ERROR: downloaded file is not mono 16 kHz PCM WAV: $TEMP_WAV" >&2
    exit 1
}

mv "$TEMP_WAV" "$FINAL_WAV"
FINISHED_AT="$(date --iso-8601=seconds)"
DURATION_SECONDS="$(
    ffprobe -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$FINAL_WAV"
)"

python - "$STATUS_JSON" "$EVENT_ID" "$SPEAKER_ID" "$SOURCE_URL" \
    "$FINAL_WAV" "$STARTED_AT" "$FINISHED_AT" "$DURATION_SECONDS" <<'PY'
import json
import sys
from pathlib import Path

(
    status_path,
    event_id,
    speaker_id,
    source_url,
    output_path,
    started_at,
    finished_at,
    duration_seconds,
) = sys.argv[1:]

payload = {
    "status": "downloaded",
    "event_id": event_id,
    "principal_speaker_id": speaker_id,
    "source_url": source_url,
    "output_path": output_path,
    "started_at": started_at,
    "finished_at": finished_at,
    "duration_seconds": float(duration_seconds),
    "audio_format": {
        "codec": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
    },
}

Path(status_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "status=downloaded"
echo "duration_seconds=$DURATION_SECONDS"
echo "output=$FINAL_WAV"

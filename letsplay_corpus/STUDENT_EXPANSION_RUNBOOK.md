# Student runbook: expand the English Let’s Play corpus

This document is the operational checklist for expanding the existing Let’s Play corpus. Follow the stages in order. Do not add, remove, rename, or replace stages, tools, models, thresholds, speakers, or output paths without written approval from the supervisor.

The target for this expansion is approximately **3 million training words** after an 80/10/10 complete-event split. At 100 words per minute, this requires about **625 total hours**, or **125 hours for each of the five existing principal speakers**. The current 106-event corpus remains intact and is extended rather than rebuilt.

## Non-negotiable rules

1. Work only with complete Let’s Play events from the five approved channels in `expansion_speaker_targets.csv`.
2. Keep the existing corpus. Never delete or overwrite raw audio, raw Whisper transcripts, structure directories, manifests, or logs.
3. Keep fillers, hesitation, pauses, slow stretches, waiting, and in-game dialogue.
4. Use faster-whisper `large-v3`. Do not substitute another model.
5. Do not add WhisperX, forced alignment, VAD, diarization, cleanup, deduplication, manual transcript correction, topic filtering, or new quality-control steps.
6. Do not listen through complete events. Selection is based on channel metadata, title rules, duration rules, and the existing automated validations.
7. Do not proceed to the next stage unless the audit command for the current stage exits successfully.
8. When any command fails, stop. Record the command, job ID, error file, and error message. Do not improvise a replacement procedure.
9. Do not change scripts while running the corpus. Code changes require a separate branch and pull request.
10. Derived data stay in the cluster workspace. Do not commit audio, transcripts, tables, logs, or generated manifests to Git.

## Required work record

Create one plain-text log before starting:

```bash
WORK=/pfs/work9/workspace/scratch/hs_mlinke-conversational
RUN="$WORK/logs/letsplay_expansion_3m"
mkdir -p "$RUN"
touch "$RUN/operator_log.md"
```

For every stage, append:

- date and time;
- exact command used;
- Slurm job ID, when applicable;
- completion counts;
- failed task IDs and the first relevant error message;
- the audit result;
- no interpretation beyond what the output shows.

A stage is not complete because jobs disappeared from `squeue`. It is complete only when `sacct` reports success and the audit passes.

## Stage 0 — synchronize the approved code

```bash
cd ~/projects/conversational
git switch main
git pull --ff-only origin main
git status --short
```

Required result: `git status --short` prints nothing. Otherwise stop.

Run the pipeline contract test:

```bash
python tests/test_pipeline_contract.py
```

Required result: exit code 0 and the final line `pipeline contract and BPE corpus fixture passed`.

## Stage 1 — build expansion manifests

This stage must preserve the existing 106 events and select only additional events.

```bash
WORK=/pfs/work9/workspace/scratch/hs_mlinke-conversational
EXP="$WORK/derived/letsplay/expansion_3m"
mkdir -p "$EXP"

python letsplay_corpus/scripts/build_expansion_manifest.py \
  --targets letsplay_corpus/expansion_speaker_targets.csv \
  --existing-manifest "$WORK/derived/letsplay/bulk_manifest.csv" \
  --known-sources letsplay_corpus/sources.yml \
  --candidate-output "$EXP/candidate_metadata.csv" \
  --new-manifest-output "$EXP/new_events_manifest.csv" \
  --full-manifest-output "$EXP/full_manifest.csv" \
  --summary-output "$EXP/manifest_summary.json"
```

Required safeguards:

- the script must exit 0;
- `full_manifest.csv` must contain every existing event exactly once;
- `new_events_manifest.csv` must contain no event from the existing manifest;
- every speaker must reach at least 125 total selected hours;
- no duplicate `event_id` or `source_video_id` may occur.

Run:

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$EXP/full_manifest.csv" \
  --data-root "$WORK" \
  --stage manifest \
  --json-output "$EXP/audit_manifest.json"
```

Stop point: send `manifest_summary.json` and `audit_manifest.json` to the supervisor. Do not download anything until the manifest is approved.

## Stage 2 — download only the new audio

Count the new events and submit one resumable CPU-array task per row:

```bash
NEW_MANIFEST="$EXP/new_events_manifest.csv"
N=$(python - "$NEW_MANIFEST" <<'PY'
import csv, sys
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
LAST=$((N - 1))
LOG_DIR="$RUN/download"
mkdir -p "$LOG_DIR"

echo "new_event_count=$N"

sbatch \
  --array="0-${LAST}%12" \
  --output="$LOG_DIR/download-%A_%a.out" \
  --error="$LOG_DIR/download-%A_%a.err" \
  letsplay_corpus/scripts/download_bulk_event.sbatch \
  "$NEW_MANIFEST" \
  "$WORK/raw_audio/letsplay"
```

After the array leaves the queue, use `sacct` with the returned job ID. All top-level array tasks must be `COMPLETED` with exit code `0:0`. Failed tasks may be resubmitted because the worker skips valid existing WAV files.

Audit:

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$EXP/full_manifest.csv" \
  --data-root "$WORK" \
  --stage audio \
  --json-output "$EXP/audit_audio.json"
```

Required result: valid audio count equals full-manifest event count. Otherwise stop.

## Stage 3 — transcribe only the new events

```bash
LOG_DIR="$RUN/transcription"
mkdir -p "$LOG_DIR"

sbatch \
  --array="0-${LAST}%8" \
  --partition=gpu_h100_short,gpu_h100 \
  --time=00:30:00 \
  --output="$LOG_DIR/transcribe-%A_%a.out" \
  --error="$LOG_DIR/transcribe-%A_%a.err" \
  letsplay_corpus/scripts/transcribe_bulk_event.sbatch \
  "$NEW_MANIFEST" \
  "$WORK" \
  "$HOME/projects/conversational" \
  "$HOME/projects/tokenizer-tuning-educational-representations"
```

Do not change `large-v3`, decoding settings, output schema, or transcript paths.

After all tasks complete, audit:

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$EXP/full_manifest.csv" \
  --data-root "$WORK" \
  --stage transcripts \
  --json-output "$EXP/audit_transcripts.json"
```

Required result: valid transcript count equals full-manifest event count. Otherwise stop.

## Stage 4 — annotate structure for only the new events

```bash
LOG_DIR="$RUN/structure"
mkdir -p "$LOG_DIR"

sbatch \
  --array="0-${LAST}%8" \
  --output="$LOG_DIR/syntax-%A_%a.out" \
  --error="$LOG_DIR/syntax-%A_%a.err" \
  letsplay_corpus/scripts/annotate_bulk_event.sbatch \
  "$NEW_MANIFEST" \
  "$WORK" \
  "$HOME/projects/conversational" \
  "$HOME/projects/tokenizer-tuning-educational-representations"
```

Audit after completion:

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$EXP/full_manifest.csv" \
  --data-root "$WORK" \
  --stage structure \
  --json-output "$EXP/audit_structure.json"
```

Required result: valid structure count equals full-manifest event count. Otherwise stop.

## Stage 5 — rebuild corpus-level structure tables

Use the full expanded manifest, not the new-events manifest:

```bash
LOG_DIR="$RUN/combine_structure"
mkdir -p "$LOG_DIR"

sbatch \
  --output="$LOG_DIR/combine-%j.out" \
  --error="$LOG_DIR/combine-%j.err" \
  letsplay_corpus/scripts/combine_structure.sbatch \
  "$EXP/full_manifest.csv" \
  "$WORK" \
  "$HOME/projects/conversational"
```

Record the job ID and verify `COMPLETED 0:0`. Do not manually concatenate CSV files.

## Stage 6 — prepare the expanded tokenizer corpus

Use the full expanded manifest and a separate output directory so the 106-event pilot corpus remains available:

```bash
LOG_DIR="$RUN/tokenizer_corpus"
TOKENIZER_OUTPUT="$WORK/derived/letsplay/tokenizer_corpus_expanded_3m"
mkdir -p "$LOG_DIR"

sbatch \
  --output="$LOG_DIR/prepare-%j.out" \
  --error="$LOG_DIR/prepare-%j.err" \
  letsplay_corpus/scripts/prepare_tokenizer_corpus.sbatch \
  "$EXP/full_manifest.csv" \
  "$WORK" \
  "$HOME/projects/conversational" \
  0.8 \
  "$TOKENIZER_OUTPUT"
```

Audit:

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$EXP/full_manifest.csv" \
  --data-root "$WORK" \
  --stage tokenizer \
  --tokenizer-output "$TOKENIZER_OUTPUT" \
  --json-output "$EXP/audit_tokenizer.json"
```

Required results:

- event count matches the full manifest;
- all events are assigned once at complete-event level;
- realized global split counts match the requested 80/10/10 counts as closely as integer event counts allow;
- every principal speaker occurs in all three splits;
- fillers remain in text;
- continuous `preceding_gap_seconds`, segment start, and segment end values are retained in the JSONL event records;
- the training corpus contains at least 3 million whitespace-delimited running-word estimates, excluding special-token markers.

If the training estimate is below 3 million words, the collection target has not been met. Stop and report the measured shortfall. Do not train a tokenizer.

## Final completion report

The task is complete only when the following files exist and every audit exits 0:

```text
expansion_3m/manifest_summary.json
expansion_3m/audit_manifest.json
expansion_3m/audit_audio.json
expansion_3m/audit_transcripts.json
expansion_3m/audit_structure.json
expansion_3m/audit_tokenizer.json
expansion_3m/full_manifest.csv
expansion_3m/new_events_manifest.csv
tokenizer_corpus_expanded_3m/summary.json
```

Send the supervisor:

1. the operator log;
2. all six audit JSON files;
3. the manifest summary;
4. Slurm job IDs for download, transcription, structure, aggregation, and tokenizer-corpus preparation;
5. only the list of failed or retried task IDs, not all successful task logs.

Do not begin BPE training. That is a separate approved task and remains blocked until corpus size, split integrity, and special-token atomicity checks pass.

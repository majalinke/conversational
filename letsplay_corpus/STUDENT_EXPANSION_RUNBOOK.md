# Student runbook: expand the English Let’s Play corpus

This is the complete operational checklist. Follow the stages in order. Do not add, remove, rename, replace, or reorder stages, tools, models, thresholds, speakers, or output paths without written approval from the supervisor.

The target is approximately **3 million training words** after an 80/10/10 complete-event split. The planning target is **625 total hours: 125 hours for each of the five existing principal speakers**. The validated 106-event corpus is preserved and extended rather than rebuilt.

## Non-negotiable rules

1. Use only complete Let’s Play events from the five approved channels in `expansion_speaker_targets.csv`.
2. Never delete or overwrite existing audio, raw Whisper transcripts, event structure directories, manifests, logs, or pilot-level combined outputs.
3. Keep fillers, hesitation, pauses, slow stretches, waiting, and in-game dialogue.
4. Use faster-whisper `large-v3`. Do not substitute another model.
5. Do not add WhisperX, forced alignment, VAD, diarization, cleanup, deduplication, manual transcript correction, topic filtering, or additional quality-control stages.
6. Do not listen through complete events. Selection uses the approved metadata, title, and duration rules.
7. Do not proceed until the audit for the current stage exits successfully.
8. On any failure, stop and record the exact command, job ID, error file, and first relevant error message. Do not improvise a replacement procedure.
9. Do not edit scripts while running the corpus. Code changes require a separate branch and pull request.
10. Do not commit generated manifests, audio, transcripts, tables, or logs to Git.
11. Do not begin BPE training. Tokenizer training is a separate approved task.

## Stage 0 — initialize the run

```bash
cd ~/projects/conversational
git switch main
git pull --ff-only origin main
git status --short
```

`git status --short` must print nothing. Otherwise stop.

Set the fixed paths once:

```bash
WORK=/pfs/work9/workspace/scratch/hs_mlinke-conversational
EXP="$WORK/derived/letsplay/expansion_3m"
RUN="$WORK/logs/letsplay_expansion_3m"
NEW_MANIFEST="$EXP/new_events_manifest.csv"
FULL_MANIFEST="$EXP/full_manifest.csv"
STRUCTURE_OUTPUT="$WORK/results/letsplay_structure_expanded_3m"
TOKENIZER_OUTPUT="$WORK/derived/letsplay/tokenizer_corpus_expanded_3m"
mkdir -p "$EXP" "$RUN"
touch "$RUN/operator_log.md"
```

Run the contract test:

```bash
python tests/test_pipeline_contract.py
```

Required result: exit code 0 and final line `pipeline contract and BPE corpus fixture passed`.

For every stage, append to `operator_log.md`:

- date and time;
- exact command;
- Slurm job ID;
- completion counts;
- failed or retried task IDs;
- first relevant error message;
- audit result.

A job is not complete because it disappeared from `squeue`. For each returned job ID run:

```bash
JOB_ID=REPLACE_WITH_RETURNED_JOB_ID
sacct -X -j "$JOB_ID" --format=JobIDRaw,State,Elapsed,ExitCode
```

Every top-level task must be `COMPLETED` with `0:0` before the audit is run.

## Stage 1 — build append-only expansion manifests

```bash
python letsplay_corpus/scripts/build_expansion_manifest.py \
  --targets letsplay_corpus/expansion_speaker_targets.csv \
  --existing-manifest "$WORK/derived/letsplay/bulk_manifest.csv" \
  --known-sources letsplay_corpus/sources.yml \
  --candidate-output "$EXP/candidate_metadata.csv" \
  --new-manifest-output "$NEW_MANIFEST" \
  --full-manifest-output "$FULL_MANIFEST" \
  --summary-output "$EXP/manifest_summary.json"
```

The script must exit 0. It fails when a speaker does not reach 125 hours or when duplicate event/video IDs occur.

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$FULL_MANIFEST" \
  --data-root "$WORK" \
  --stage manifest \
  --json-output "$EXP/audit_manifest.json"
```

Required result: `"passed": true`.

**Mandatory stop point:** send `manifest_summary.json` and `audit_manifest.json` to the supervisor. Do not download anything until the manifest is approved.

## Stage 2 — download only new audio

Count the new rows once and reuse `LAST` for the three array stages:

```bash
N=$(python - "$NEW_MANIFEST" <<'PY'
import csv, sys
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
LAST=$((N - 1))
echo "new_event_count=$N"
```

`N` must be greater than 0. Otherwise stop.

```bash
LOG_DIR="$RUN/download"
mkdir -p "$LOG_DIR"

sbatch \
  --array="0-${LAST}%12" \
  --output="$LOG_DIR/download-%A_%a.out" \
  --error="$LOG_DIR/download-%A_%a.err" \
  letsplay_corpus/scripts/download_bulk_event.sbatch \
  "$NEW_MANIFEST" \
  "$WORK/raw_audio/letsplay"
```

Check the returned job with `sacct`. Failed tasks may be resubmitted with the same command because valid WAV files are skipped. Do not delete successful outputs.

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$FULL_MANIFEST" \
  --data-root "$WORK" \
  --stage audio \
  --json-output "$EXP/audit_audio.json"
```

Required result: valid audio count equals manifest event count and `"passed": true`.

## Stage 3 — transcribe only new events

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

Do not change `large-v3`, decoding settings, output schema, or transcript paths. Check the returned job with `sacct`.

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$FULL_MANIFEST" \
  --data-root "$WORK" \
  --stage transcripts \
  --json-output "$EXP/audit_transcripts.json"
```

Required result: valid transcript count equals manifest event count and `"passed": true`.

## Stage 4 — annotate structure for only new events

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

Check the returned job with `sacct`.

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$FULL_MANIFEST" \
  --data-root "$WORK" \
  --stage structure \
  --json-output "$EXP/audit_structure.json"
```

Required result: valid structure count equals manifest event count and `"passed": true`.

## Stage 5 — rebuild expanded corpus-level structure tables

Use the full manifest and the separate expanded output path. Do not overwrite `results/letsplay_structure`.

```bash
LOG_DIR="$RUN/combine_structure"
mkdir -p "$LOG_DIR"

sbatch \
  --output="$LOG_DIR/combine-%j.out" \
  --error="$LOG_DIR/combine-%j.err" \
  letsplay_corpus/scripts/combine_structure.sbatch \
  "$FULL_MANIFEST" \
  "$WORK" \
  "$HOME/projects/conversational" \
  "$STRUCTURE_OUTPUT"
```

Check the returned job with `sacct`. Do not manually concatenate CSV files.

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$FULL_MANIFEST" \
  --data-root "$WORK" \
  --stage combined \
  --structure-output "$STRUCTURE_OUTPUT" \
  --json-output "$EXP/audit_combined.json"
```

Required result: combined event count equals manifest event count and `"passed": true`.

## Stage 6 — prepare and audit the expanded tokenizer corpus

Use the full manifest and the separate output path. Do not overwrite `derived/letsplay/tokenizer_corpus`.

```bash
LOG_DIR="$RUN/tokenizer_corpus"
mkdir -p "$LOG_DIR"

sbatch \
  --output="$LOG_DIR/prepare-%j.out" \
  --error="$LOG_DIR/prepare-%j.err" \
  letsplay_corpus/scripts/prepare_tokenizer_corpus.sbatch \
  "$FULL_MANIFEST" \
  "$WORK" \
  "$HOME/projects/conversational" \
  0.8 \
  "$TOKENIZER_OUTPUT"
```

Check the returned job with `sacct`.

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$FULL_MANIFEST" \
  --data-root "$WORK" \
  --stage tokenizer \
  --structure-output "$STRUCTURE_OUTPUT" \
  --tokenizer-output "$TOKENIZER_OUTPUT" \
  --json-output "$EXP/audit_tokenizer.json"
```

The final audit must verify all of the following:

- every manifest event has valid audio, transcript, structure, and combined outputs;
- every event is assigned exactly once at complete-event level;
- global integer split counts match the requested 80/10/10 allocation;
- every principal speaker occurs in train, validation, and test;
- continuous `start_seconds`, `end_seconds`, and `preceding_gap_seconds` values remain in event JSONL records;
- fillers remain as transcribed text;
- the training corpus contains at least 3 million whitespace-delimited running-word estimates, excluding special tokens.

If the training estimate is below 3 million, stop and report the measured shortfall. Do not train a tokenizer or change the threshold.

## Definition of done

The task is complete only when all seven audits contain `"passed": true` and these files exist:

```text
expansion_3m/manifest_summary.json
expansion_3m/audit_manifest.json
expansion_3m/audit_audio.json
expansion_3m/audit_transcripts.json
expansion_3m/audit_structure.json
expansion_3m/audit_combined.json
expansion_3m/audit_tokenizer.json
expansion_3m/full_manifest.csv
expansion_3m/new_events_manifest.csv
results/letsplay_structure_expanded_3m/summary.json
derived/letsplay/tokenizer_corpus_expanded_3m/summary.json
```

Send the supervisor:

1. `operator_log.md`;
2. all seven audit JSON files;
3. `manifest_summary.json`;
4. the five Slurm job IDs;
5. only failed or retried task IDs and their first relevant errors.

Do not start another task or add a new processing stage after this point.

# Bulk Let’s Play collection

The first completed pass contains 106 events from five principal speakers and approximately 500,000 words. It is a validated pilot corpus, not the final tokenizer-training corpus.

The active expansion target is approximately 3 million training words after an 80/10/10 complete-event split. The planning estimate is 625 total hours: 125 hours for each of the same five principal speakers. The original 106 events are preserved and additional events are appended through a versioned expansion manifest.

The operational instructions for the student assistant are in [`STUDENT_EXPANSION_RUNBOOK.md`](STUDENT_EXPANSION_RUNBOOK.md). That runbook defines the required stage order, stop points, audit commands, and completion criteria. It must be followed without adding or dropping steps.

The full bulk pipeline has two branches after transcription. The structural branch produces syntax and dependency tables for analysis. The tokenizer branch serializes speech-derived text and creates event-level training, validation and test sets for BPE training.

```text
channel metadata
→ complete-event manifest
→ 16 kHz mono audio
→ faster-whisper large-v3
→ preserved raw Whisper transcripts
   ├→ spaCy syntax and dependency tables
   │  → combined speaker- and register-level structural tables
   └→ provisional speech serialization with <SILENCE>
      → complete-event train/validation/test split
      → BPE tokenizer training corpus
      → corpus sufficiency gate
      → BPE training and tokenizer checks [not yet implemented in bulk]
```

WhisperX alignment, Silero VAD and pyannote diarization are retained only for the small timing subset. They are not part of the bulk structural pass. The tokenizer corpus uses gaps between native Whisper segments as a provisional silence measurement. It retains continuous segment start, end and preceding-gap values in the event JSONL records; the binary `<SILENCE>` marker is an additional serialization layer, not a replacement for timing information.

## 1. Initial manifest and expansion manifests

The initial manifest was built with:

```bash
python letsplay_corpus/scripts/build_bulk_manifest.py \
  --targets letsplay_corpus/bulk_speaker_targets.csv \
  --known-sources letsplay_corpus/sources.yml \
  --candidate-output /pfs/work9/workspace/scratch/hs_mlinke-conversational/derived/letsplay/bulk_candidate_metadata.csv \
  --manifest-output /pfs/work9/workspace/scratch/hs_mlinke-conversational/derived/letsplay/bulk_manifest.csv
```

The expansion must use `build_expansion_manifest.py`, not rebuild the initial manifest. It reads the existing manifest, excludes every existing event, selects only additional complete events until each speaker reaches the configured total target, and writes separate new-events and full manifests.

```bash
WORK=/pfs/work9/workspace/scratch/hs_mlinke-conversational
EXP="$WORK/derived/letsplay/expansion_3m"

python letsplay_corpus/scripts/build_expansion_manifest.py \
  --targets letsplay_corpus/expansion_speaker_targets.csv \
  --existing-manifest "$WORK/derived/letsplay/bulk_manifest.csv" \
  --known-sources letsplay_corpus/sources.yml \
  --candidate-output "$EXP/candidate_metadata.csv" \
  --new-manifest-output "$EXP/new_events_manifest.csv" \
  --full-manifest-output "$EXP/full_manifest.csv" \
  --summary-output "$EXP/manifest_summary.json"
```

The script exits nonzero when a speaker does not reach the configured target or when duplicate event or video IDs occur. No downloading may start after a failed manifest build.

## 2. Bulk processing policy

Across both branches:

- preserve complete events and raw Whisper transcripts;
- keep fillers, slow stretches, waiting, hesitation and in-game dialogue;
- use the same `large-v3` transcription settings as the pilot;
- keep all train/validation/test assignments at complete-event level;
- retain continuous segment start, end and preceding-gap values;
- do not infer `<FILLER>`, `<NOISE>`, `<OVERLAP>` or `<UNINTELLIGIBLE>` without source annotations;
- retain the project special-token inventory even when a token is not yet observed in this corpus;
- do not run forced alignment, framewise VAD or diarization unless an event is explicitly added to the timing subset;
- do not proceed between stages without a successful `audit_bulk_pipeline.py` result;
- do not train a tokenizer below the 3-million-training-word gate.

## 3. Structural annotation and aggregation

Each event receives its own `structure` directory containing segment, token-dependency and word-sequence tables. The aggregation stage combines every validated event in the supplied manifest under:

```text
results/letsplay_structure/
├── events.csv
├── segments.csv
├── token_dependencies.csv
├── word_sequences.csv
└── summary.json
```

The per-event array workers are resumable. Existing valid audio, transcript and structure outputs are skipped. Expansion processing therefore runs the workers on `new_events_manifest.csv`, while aggregation uses `full_manifest.csv`.

## 4. Prepare the BPE tokenizer corpus

The preparation stage reads the preserved raw transcripts directly. It preserves transcript order and filler words, prefixes a native Whisper segment with `<SILENCE>` when its preceding Whisper gap is at least 0.8 seconds, and retains a continuous timing record for every non-empty segment.

Splits are deterministic and use complete events. Global event counts use largest-remainder allocation for the requested 80/10/10 fractions, with every principal speaker represented in every split. The output summary records both requested fractions and realized integer counts.

For the expansion, use a separate output directory:

```bash
WORK=/pfs/work9/workspace/scratch/hs_mlinke-conversational
EXP="$WORK/derived/letsplay/expansion_3m"
TOKENIZER_OUTPUT="$WORK/derived/letsplay/tokenizer_corpus_expanded_3m"
LOG_DIR="$WORK/logs/letsplay_expansion_3m/tokenizer_corpus"
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

Outputs:

```text
tokenizer_corpus_expanded_3m/
├── train.txt
├── validation.txt
├── test.txt
├── train_events.jsonl
├── validation_events.jsonl
├── test_events.jsonl
├── split_manifest.csv
├── special_tokens.json
└── summary.json
```

The plain-text files contain one provisional prosodic unit per line. The event JSONL files retain event boundaries and per-unit `start_seconds`, `end_seconds`, `preceding_gap_seconds`, source text, serialized text and silence-insertion status.

Run the final gate before tokenizer training:

```bash
python letsplay_corpus/scripts/audit_bulk_pipeline.py \
  --manifest "$EXP/full_manifest.csv" \
  --data-root "$WORK" \
  --stage tokenizer \
  --tokenizer-output "$TOKENIZER_OUTPUT" \
  --json-output "$EXP/audit_tokenizer.json"
```

The audit fails when event assignments are incomplete, split counts drift, a speaker is missing from a split, timing metadata disappears, required files are absent, or the training estimate is below 3 million words.

Bulk BPE training, the special-token atomicity check and the dummy downstream task remain explicit incomplete stages. They must not be described as complete until their bulk implementations exist.

## 5. Pipeline parity and sufficiency guards

`letsplay_corpus/pipeline_contract.json` inventories every capability from the AH_INTERN prototype and every subsequently approved bulk safeguard. Pipeline stages and safeguards must not disappear from this file.

`tests/test_pipeline_contract.py` runs a synthetic end-to-end corpus preparation test and fails when:

- a required capability disappears from the contract;
- a deferred capability lacks an explicit reason;
- the special-token inventory changes silently;
- event-level splitting is lost;
- global split counts drift from the requested fractions;
- speaker coverage across splits is lost;
- fillers or `<SILENCE>` serialization are dropped;
- continuous timing metadata disappears;
- required BPE corpus outputs are missing;
- the 3-million-word sufficiency gate disappears.

The test runs automatically in GitHub Actions for pull requests and pushes to `main`. The pull-request template also requires an explicit parity check for pipeline changes.

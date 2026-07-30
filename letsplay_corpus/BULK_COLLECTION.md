# Bulk Let’s Play collection

The corpus targets approximately 500,000 words from five principal speakers. The first pass collects about 18 hours of complete events per speaker, corresponding to roughly 100,000 words at 100 words per minute. Actual transcript counts replace this estimate after transcription.

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
      → BPE training and tokenizer checks [not yet implemented in bulk]
```

WhisperX alignment, Silero VAD and pyannote diarization are retained only for the small timing subset. They are not part of the 500,000-word structural pass. The tokenizer corpus currently uses gaps between native Whisper segments as a provisional silence measurement, matching the AH_INTERN smoke-test serializer. It does not claim that these gaps are audio-derived pauses.

## 1. Build the candidate and selected manifests

Run from the `conversational` repository:

```bash
python letsplay_corpus/scripts/build_bulk_manifest.py \
  --targets letsplay_corpus/bulk_speaker_targets.csv \
  --known-sources letsplay_corpus/sources.yml \
  --candidate-output /pfs/work9/workspace/scratch/hs_mlinke-conversational/derived/letsplay/bulk_candidate_metadata.csv \
  --manifest-output /pfs/work9/workspace/scratch/hs_mlinke-conversational/derived/letsplay/bulk_manifest.csv
```

The script uses flat channel metadata, excludes known pilot videos, removes unsuitable durations and obvious reviews, guides, announcements, highlights and similar non-event material, then selects complete videos in channel order until each speaker reaches the requested duration target.

Outputs:

- `bulk_candidate_metadata.csv`: every inspected channel entry and its selection reason;
- `bulk_manifest.csv`: only events selected for download and transcription;
- a compact terminal summary with selected event counts, hours and estimated words per speaker.

Selection remains provisional until transcript-level checks. No complete listen-through is required.

## 2. Bulk processing policy

Across both branches:

- preserve complete events and raw Whisper transcripts;
- keep fillers, slow stretches, waiting, hesitation and in-game dialogue;
- use the same `large-v3` transcription settings as the pilot;
- keep all train/validation/test assignments at complete-event level;
- do not infer `<FILLER>`, `<NOISE>`, `<OVERLAP>` or `<UNINTELLIGIBLE>` without source annotations;
- retain the project special-token inventory even when a token is not yet observed in this corpus;
- do not run forced alignment, framewise VAD or diarization unless an event is explicitly added to the timing subset.

## 3. Structural annotation and aggregation

Each event receives its own `structure` directory containing segment, token-dependency and word-sequence tables. The aggregation stage combines the 106 validated event outputs under:

```text
results/letsplay_structure/
├── events.csv
├── segments.csv
├── token_dependencies.csv
├── word_sequences.csv
└── summary.json
```

## 4. Prepare the BPE tokenizer corpus

The preparation stage reads the preserved raw transcripts directly. It preserves transcript order and filler words, prefixes a native Whisper segment with `<SILENCE>` when its preceding Whisper gap is at least 0.8 seconds, and creates deterministic speaker-stratified splits by complete event.

Submit the CPU job from the repository root:

```bash
WORK=/pfs/work9/workspace/scratch/hs_mlinke-conversational
MANIFEST="$WORK/derived/letsplay/bulk_manifest.csv"
LOG_DIR="$WORK/logs/tokenizer_corpus"

mkdir -p "$LOG_DIR"

sbatch \
  --output="$LOG_DIR/prepare-%j.out" \
  --error="$LOG_DIR/prepare-%j.err" \
  letsplay_corpus/scripts/prepare_tokenizer_corpus.sbatch \
  "$MANIFEST" \
  "$WORK" \
  "$HOME/projects/conversational"
```

Outputs are written atomically under:

```text
derived/letsplay/tokenizer_corpus/
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

The plain-text files contain one provisional prosodic unit per line and are ready as input to a BPE trainer. The JSONL files retain event membership and boundaries. `split_manifest.csv` records the deterministic assignment of every event.

Bulk BPE training, the special-token atomicity check and the dummy downstream task remain explicit incomplete stages. They must not be described as complete until their bulk implementations exist.

## 5. Pipeline parity guard

`letsplay_corpus/pipeline_contract.json` inventories every capability from the AH_INTERN prototype and maps it to the bulk implementation or marks it explicitly as not yet implemented. Pipeline stages must not disappear from this file.

`tests/test_pipeline_contract.py` runs a synthetic end-to-end corpus preparation test and fails when:

- a prototype capability disappears from the contract;
- a deferred capability lacks an explicit reason;
- the special-token inventory changes silently;
- event-level train/validation/test splitting is lost;
- fillers or `<SILENCE>` serialization are dropped;
- required BPE corpus outputs are missing.

The test runs automatically in GitHub Actions for pull requests and pushes to `main`. The pull-request template also requires an explicit parity check for pipeline changes.

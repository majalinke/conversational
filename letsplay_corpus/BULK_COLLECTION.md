# Bulk Let’s Play collection

The structural corpus targets approximately 500,000 words from five principal speakers. The first pass collects about 18 hours of complete events per speaker, corresponding to roughly 100,000 words at 100 words per minute. Actual transcript counts will replace this estimate after transcription.

Bulk structural processing is deliberately narrower than the timing pilot:

```text
channel metadata
→ complete-event manifest
→ 16 kHz mono audio
→ faster-whisper large-v3
→ spaCy syntax and dependency tables
→ speaker- and register-level structural analyses
```

WhisperX alignment, Silero VAD and pyannote diarization are retained only for the small timing subset. They are not part of the 500,000-word structural pass.

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

For the structural pass:

- preserve complete events;
- keep slow stretches, waiting, hesitation and in-game dialogue;
- use the same `large-v3` transcription settings as the pilot;
- retain one transcript and one structural table per event;
- count actual words per speaker after spaCy processing and top up speakers below 100,000 words;
- do not run forced alignment, framewise VAD or diarization unless an event is explicitly added to the timing subset.

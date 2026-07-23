# conversational

Research code and source manifests for studying the real-time structure of naturalistic speech. The primary material is conversational Let's Play commentary; lectures provide a control condition.

The corpus is organized around complete communicative events rather than equal-duration samples. Slow sections, silence, hesitation, repair, loading or waiting periods, and other temporal structure remain part of the event unless they clearly fall outside it.

## Repository structure

- `letsplay_corpus/`: source selection, notes, and corpus-specific scripts
- `lecture_corpus/`: lecture-control source selection, notes, and scripts
- `analyses/`: data checks, temporal structure, syntax and dependencies, and individual differences
- `results/`: compact figures, tables, and summaries suitable for version control

Raw media, transcripts, model caches, and large derived datasets remain on the cluster and are not committed here. Source manifests record where material came from and how event boundaries were defined.

Preprocessing uses the separate `tokenizer-tuning-educational-representations` pipeline. Analysis records should identify the preprocessing repository commit and container image revision used for each run.

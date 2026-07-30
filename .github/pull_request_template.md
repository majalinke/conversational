## Summary

Describe the change and the outputs it affects.

## Pipeline parity

For changes to collection, transcription, annotation, corpus preparation, tokenizer training, or evaluation:

- [ ] I compared the change with `letsplay_corpus/pipeline_contract.json`.
- [ ] No prototype capability was removed silently.
- [ ] Any capability not implemented in bulk remains explicitly marked `not_yet_bulk_implemented` with a reason.
- [ ] Required output files and split rules remain covered by `tests/test_pipeline_contract.py`.
- [ ] The bulk pipeline documentation reflects the actual executable stages.

#!/usr/bin/env python3
"""Fail when prototype capabilities or BPE corpus guarantees disappear."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_CAPABILITIES = {
    "source_discovery",
    "source_manifest",
    "audio_download",
    "raw_whisper_transcription",
    "raw_transcript_preservation",
    "whisper_gap_measurement",
    "silence_serialization",
    "spacy_structure_annotation",
    "event_level_data_splits",
    "tokenizer_training_corpus",
    "bpe_tokenizer_training",
    "special_token_atomicity_check",
    "dummy_downstream_task",
}
EXPECTED_SPECIAL_TOKENS = [
    "<UNK>",
    "<FILLER>",
    "<SILENCE>",
    "<NOISE>",
    "<OVERLAP>",
    "<UNINTELLIGIBLE>",
]
ALLOWED_STATUSES = {
    "implemented",
    "not_yet_bulk_implemented",
    "not_applicable_with_reason",
}


def write_fixture_transcript(path: Path, event_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_id": event_id,
                "language": "en",
                "duration": 4.0,
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": " uh hello "},
                    {"start": 2.0, "end": 3.0, "text": "second segment"},
                ],
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "letsplay_corpus" / "pipeline_contract.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities = contract["capabilities"]
    ids = {item["id"] for item in capabilities}
    assert ids == EXPECTED_CAPABILITIES, (
        "Prototype capability inventory changed without updating the contract test: "
        f"missing={sorted(EXPECTED_CAPABILITIES - ids)}, "
        f"unexpected={sorted(ids - EXPECTED_CAPABILITIES)}"
    )
    assert contract["special_tokens"] == EXPECTED_SPECIAL_TOKENS
    for capability in capabilities:
        status = capability["bulk_status"]
        assert status in ALLOWED_STATUSES
        if status == "implemented":
            assert capability.get("bulk_implementation")
        else:
            assert capability.get("reason")

    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        data_root = temporary_path / "data"
        manifest_path = temporary_path / "manifest.csv"
        output_dir = temporary_path / "tokenizer_corpus"
        rows: list[dict[str, str]] = []
        for speaker in ("speaker_a", "speaker_b", "speaker_c"):
            for index in range(3):
                event_id = f"event_{speaker}_{index}"
                rows.append(
                    {
                        "event_id": event_id,
                        "principal_speaker_id": speaker,
                        "register": "letsplay",
                    }
                )
                write_fixture_transcript(
                    data_root
                    / "derived"
                    / "letsplay"
                    / event_id
                    / "raw_transcripts"
                    / f"{event_id}.whisper_raw.json",
                    event_id,
                )

        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        subprocess.run(
            [
                sys.executable,
                str(
                    root
                    / "letsplay_corpus"
                    / "scripts"
                    / "prepare_tokenizer_corpus.py"
                ),
                "--manifest",
                str(manifest_path),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )

        required_outputs = {
            "train.txt",
            "validation.txt",
            "test.txt",
            "train_events.jsonl",
            "validation_events.jsonl",
            "test_events.jsonl",
            "split_manifest.csv",
            "special_tokens.json",
            "summary.json",
        }
        assert required_outputs.issubset(
            {path.name for path in output_dir.iterdir()}
        )

        with (output_dir / "split_manifest.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            split_rows = list(csv.DictReader(handle))
        assert len(split_rows) == len(rows)
        assert len({row["event_id"] for row in split_rows}) == len(rows)
        by_speaker: dict[str, set[str]] = {}
        for row in split_rows:
            by_speaker.setdefault(row["principal_speaker_id"], set()).add(
                row["split"]
            )
        assert all(
            splits == {"train", "validation", "test"}
            for splits in by_speaker.values()
        )

        corpus_text = "\n".join(
            (output_dir / f"{split}.txt").read_text(encoding="utf-8")
            for split in ("train", "validation", "test")
        )
        assert "uh hello" in corpus_text
        assert "<SILENCE> second segment" in corpus_text

        summary = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["event_count"] == len(rows)
        assert summary["split_unit"] == "complete_event"
        assert summary["special_tokens"] == EXPECTED_SPECIAL_TOKENS

    print("pipeline contract and BPE corpus fixture passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

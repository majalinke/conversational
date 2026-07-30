#!/usr/bin/env python3
"""Audit cumulative Let’s Play pipeline stages against one complete-event manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STAGES = ("manifest", "audio", "transcripts", "structure", "combined", "tokenizer")
REQUIRED_STRUCTURE_FILES = (
    "summary.json",
    "segment_statistics.csv",
    "token_dependencies.csv",
    "word_sequence_statistics.csv",
    "processed_transcript.spacy.json",
)
REQUIRED_COMBINED_FILES = (
    "events.csv",
    "segments.csv",
    "token_dependencies.csv",
    "word_sequences.csv",
    "summary.json",
)
REQUIRED_TOKENIZER_FILES = (
    "train.txt",
    "validation.txt",
    "test.txt",
    "train_events.jsonl",
    "validation_events.jsonl",
    "test_events.jsonl",
    "split_manifest.csv",
    "special_tokens.json",
    "summary.json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def add_error(errors: list[str], message: str) -> None:
    if len(errors) < 100:
        errors.append(message)


def expected_split_counts(total: int) -> dict[str, int]:
    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    raw = {split: total * fraction for split, fraction in fractions.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        fractions,
        key=lambda split: (
            raw[split] - counts[split],
            split == "train",
            split == "validation",
        ),
        reverse=True,
    )
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def validate_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "csv=p=0:s=,",
            str(path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "pcm_s16le,16000,1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--structure-output", type=Path)
    parser.add_argument("--tokenizer-output", type=Path)
    parser.add_argument("--minimum-training-words", type=int, default=3_000_000)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    rows = read_csv(manifest_path)
    errors: list[str] = []
    required_columns = {
        "event_id",
        "source_video_id",
        "principal_speaker_id",
        "register",
    }
    columns = set(rows[0]) if rows else set()
    if not rows:
        add_error(errors, "manifest has no events")
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        add_error(errors, f"manifest missing columns: {missing_columns}")

    event_ids = [row.get("event_id", "").strip() for row in rows]
    video_ids = [row.get("source_video_id", "").strip() for row in rows]
    duplicate_events = sorted(
        value for value, count in Counter(event_ids).items() if value and count > 1
    )
    duplicate_videos = sorted(
        value for value, count in Counter(video_ids).items() if value and count > 1
    )
    if any(not value for value in event_ids):
        add_error(errors, "manifest contains empty event_id values")
    if any(not value for value in video_ids):
        add_error(errors, "manifest contains empty source_video_id values")
    if duplicate_events:
        add_error(errors, f"duplicate event_id values: {duplicate_events[:20]}")
    if duplicate_videos:
        add_error(errors, f"duplicate source_video_id values: {duplicate_videos[:20]}")

    stage_index = STAGES.index(args.stage)
    counts: dict[str, Any] = {
        "manifest_events": len(rows),
        "principal_speakers": len(
            {row.get("principal_speaker_id", "").strip() for row in rows}
        ),
    }

    if stage_index >= STAGES.index("audio"):
        valid_audio = 0
        for event_id in event_ids:
            path = data_root / "raw_audio" / "letsplay" / f"{event_id}.wav"
            if validate_wav(path):
                valid_audio += 1
            else:
                add_error(errors, f"invalid or missing WAV: {event_id}")
        counts["valid_audio"] = valid_audio

    if stage_index >= STAGES.index("transcripts"):
        valid_transcripts = 0
        for event_id in event_ids:
            path = (
                data_root
                / "derived"
                / "letsplay"
                / event_id
                / "raw_transcripts"
                / f"{event_id}.whisper_raw.json"
            )
            try:
                payload = read_json(path)
                valid = (
                    payload.get("source_id") == event_id
                    and payload.get("model") == "large-v3"
                    and isinstance(payload.get("segments"), list)
                    and len(payload["segments"]) > 0
                )
            except (OSError, ValueError, json.JSONDecodeError):
                valid = False
            if valid:
                valid_transcripts += 1
            else:
                add_error(errors, f"invalid or missing transcript: {event_id}")
        counts["valid_transcripts"] = valid_transcripts

    if stage_index >= STAGES.index("structure"):
        valid_structure = 0
        for event_id in event_ids:
            directory = data_root / "derived" / "letsplay" / event_id / "structure"
            valid = all(
                (directory / filename).is_file()
                and (directory / filename).stat().st_size > 0
                for filename in REQUIRED_STRUCTURE_FILES
            )
            if valid:
                try:
                    summary = read_json(directory / "summary.json")
                    valid = (
                        summary.get("source_id") == event_id
                        and summary.get("spacy_model") == "en_core_web_sm"
                        and int(summary.get("segment_count", 0)) > 0
                        and int(summary.get("token_count", 0)) > 0
                        and int(summary.get("lexical_token_count", 0)) > 0
                    )
                except (OSError, ValueError, json.JSONDecodeError, TypeError):
                    valid = False
            if valid:
                valid_structure += 1
            else:
                add_error(errors, f"invalid or missing structure output: {event_id}")
        counts["valid_structure"] = valid_structure

    if stage_index >= STAGES.index("combined"):
        structure_output = (
            args.structure_output.resolve()
            if args.structure_output
            else data_root / "results" / "letsplay_structure"
        )
        missing = [
            name for name in REQUIRED_COMBINED_FILES if not (structure_output / name).is_file()
        ]
        if missing:
            add_error(errors, f"combined structure output missing files: {missing}")
        else:
            combined_summary = read_json(structure_output / "summary.json")
            combined_event_count = int(
                combined_summary.get("totals", {}).get("event_count", -1)
            )
            if combined_event_count != len(rows):
                add_error(
                    errors,
                    "combined structure summary event_count does not match manifest",
                )
            combined_events = read_csv(structure_output / "events.csv")
            combined_ids = [row.get("event_id", "").strip() for row in combined_events]
            if len(combined_events) != len(rows) or set(combined_ids) != set(event_ids):
                add_error(
                    errors,
                    "combined events.csv does not contain every manifest event exactly once",
                )
            counts["combined_structure_events"] = len(combined_events)
            counts["combined_structure_output"] = str(structure_output)

    if stage_index >= STAGES.index("tokenizer"):
        if args.tokenizer_output is None:
            add_error(errors, "--tokenizer-output is required for tokenizer audit")
        else:
            output = args.tokenizer_output.resolve()
            missing = [
                name for name in REQUIRED_TOKENIZER_FILES if not (output / name).is_file()
            ]
            if missing:
                add_error(errors, f"tokenizer output missing files: {missing}")
            else:
                summary = read_json(output / "summary.json")
                if int(summary.get("event_count", -1)) != len(rows):
                    add_error(
                        errors,
                        "tokenizer summary event_count does not match manifest",
                    )
                split_rows = read_csv(output / "split_manifest.csv")
                split_event_ids = [
                    row.get("event_id", "").strip() for row in split_rows
                ]
                if len(split_rows) != len(rows) or set(split_event_ids) != set(event_ids):
                    add_error(
                        errors,
                        "split manifest does not assign every manifest event exactly once",
                    )
                realized = Counter(row.get("split", "") for row in split_rows)
                expected = expected_split_counts(len(rows))
                if {
                    split: realized.get(split, 0) for split in expected
                } != expected:
                    add_error(
                        errors,
                        f"global split counts differ: realized={dict(realized)}, expected={expected}",
                    )
                by_speaker: dict[str, set[str]] = defaultdict(set)
                for row in split_rows:
                    by_speaker[row.get("principal_speaker_id", "")].add(
                        row.get("split", "")
                    )
                missing_speaker_splits = sorted(
                    speaker
                    for speaker, splits in by_speaker.items()
                    if splits != {"train", "validation", "test"}
                )
                if missing_speaker_splits:
                    add_error(
                        errors,
                        "speakers missing one or more splits: "
                        f"{missing_speaker_splits}",
                    )

                timing_records = 0
                for split in ("train", "validation", "test"):
                    with (output / f"{split}_events.jsonl").open(
                        encoding="utf-8"
                    ) as handle:
                        for line_number, line in enumerate(handle, start=1):
                            record = json.loads(line)
                            units = record.get("units")
                            if not isinstance(units, list) or not units:
                                add_error(
                                    errors,
                                    f"{split}_events.jsonl line {line_number} lacks unit timing records",
                                )
                                continue
                            for unit in units:
                                if not all(
                                    key in unit
                                    for key in (
                                        "start_seconds",
                                        "end_seconds",
                                        "preceding_gap_seconds",
                                        "text",
                                        "serialized_text",
                                    )
                                ):
                                    add_error(
                                        errors,
                                        f"{split}_events.jsonl line {line_number} has incomplete unit timing metadata",
                                    )
                                    break
                                timing_records += 1
                counts["tokenizer_timing_records"] = timing_records

                special_tokens = set(
                    read_json(output / "special_tokens.json").get(
                        "special_tokens", []
                    )
                )
                train_tokens = (output / "train.txt").read_text(
                    encoding="utf-8"
                ).split()
                training_word_estimate = sum(
                    1 for token in train_tokens if token not in special_tokens
                )
                counts["training_word_estimate"] = training_word_estimate
                counts["minimum_training_words"] = args.minimum_training_words
                if training_word_estimate < args.minimum_training_words:
                    add_error(
                        errors,
                        "training corpus too small: "
                        f"{training_word_estimate} < {args.minimum_training_words}",
                    )
                counts["realized_split_counts"] = {
                    split: realized.get(split, 0)
                    for split in ("train", "validation", "test")
                }
                counts["expected_split_counts"] = expected

    report = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "required_stage": args.stage,
        "passed": not errors,
        "counts": counts,
        "errors": errors,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        output_path = args.json_output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

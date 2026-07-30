#!/usr/bin/env python3
"""Prepare deterministic event-level BPE corpora from bulk Whisper transcripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

SPECIAL_TOKENS = [
    "<UNK>",
    "<FILLER>",
    "<SILENCE>",
    "<NOISE>",
    "<OVERLAP>",
    "<UNINTELLIGIBLE>",
]

SPLITS = ("train", "validation", "test")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"event_id", "principal_speaker_id", "register"}
    missing = required.difference(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    event_ids = [row["event_id"].strip() for row in rows]
    if not event_ids:
        raise ValueError("Manifest has no events")
    duplicates = sorted(
        event_id for event_id, count in Counter(event_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate event IDs in manifest: {duplicates}")
    return rows


def split_counts(
    event_count: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int, int]:
    if event_count < 3:
        raise ValueError(
            "At least three events per speaker are required for event-level "
            f"train/validation/test splits; found {event_count}"
        )
    train_count = max(1, math.floor(event_count * train_fraction))
    validation_count = max(1, math.floor(event_count * validation_fraction))
    test_count = event_count - train_count - validation_count
    if test_count < 1:
        deficit = 1 - test_count
        train_count -= deficit
        test_count = 1
    if train_count < 1:
        raise ValueError(f"Cannot allocate non-empty splits from {event_count} events")
    assert train_count + validation_count + test_count == event_count
    return train_count, validation_count, test_count


def deterministic_split(
    manifest_rows: Sequence[dict[str, str]],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        speaker = row["principal_speaker_id"].strip()
        event_id = row["event_id"].strip()
        if not speaker:
            raise ValueError(f"Missing principal_speaker_id for {event_id}")
        by_speaker[speaker].append(event_id)

    assignment: dict[str, str] = {}
    for speaker, event_ids in sorted(by_speaker.items()):
        ordered = sorted(
            event_ids,
            key=lambda event_id: hashlib.sha256(
                f"{seed}:{speaker}:{event_id}".encode("utf-8")
            ).hexdigest(),
        )
        train_count, validation_count, _ = split_counts(
            len(ordered), train_fraction, validation_fraction
        )
        for index, event_id in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            assignment[event_id] = split
    return assignment


def serialize_event(
    transcript: dict[str, Any],
    silence_threshold_seconds: float,
) -> tuple[list[str], dict[str, int]]:
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcript has no segments")
    ordered = sorted(
        segments,
        key=lambda item: (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
        ),
    )
    units: list[str] = []
    inserted_silences = 0
    empty_segments = 0
    previous_end = 0.0
    for segment in ordered:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = normalized_text(segment.get("text", ""))
        gap = start - previous_end
        previous_end = max(previous_end, end)
        if not text:
            empty_segments += 1
            continue
        if gap >= silence_threshold_seconds:
            units.append(f"<SILENCE> {text}")
            inserted_silences += 1
        else:
            units.append(text)
    if not units:
        raise ValueError("Transcript contains no non-empty text segments")
    return units, {
        "unit_count": len(units),
        "inserted_silence_count": inserted_silences,
        "empty_segment_count": empty_segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--silence-threshold-seconds", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    if args.silence_threshold_seconds < 0:
        raise ValueError("silence threshold must be non-negative")
    if not 0 < args.train_fraction < 1:
        raise ValueError("train fraction must be between 0 and 1")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_rows = load_manifest(manifest_path)
    assignment = deterministic_split(
        manifest_rows,
        args.seed,
        args.train_fraction,
        args.validation_fraction,
    )

    temporary_dir = output_dir.with_name(f".{output_dir.name}.partial")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    split_units: dict[str, list[str]] = {split: [] for split in SPLITS}
    split_event_rows: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    event_rows: list[dict[str, Any]] = []
    aggregate = {
        split: {
            "event_count": 0,
            "unit_count": 0,
            "character_count": 0,
            "inserted_silence_count": 0,
        }
        for split in SPLITS
    }

    try:
        for manifest_row in manifest_rows:
            event_id = manifest_row["event_id"].strip()
            speaker = manifest_row["principal_speaker_id"].strip()
            register = manifest_row["register"].strip()
            split = assignment[event_id]
            transcript_path = (
                data_root
                / "derived"
                / "letsplay"
                / event_id
                / "raw_transcripts"
                / f"{event_id}.whisper_raw.json"
            )
            if not transcript_path.is_file():
                raise FileNotFoundError(f"Missing transcript: {transcript_path}")
            transcript = read_json(transcript_path)
            if str(transcript.get("source_id", "")).strip() != event_id:
                raise ValueError(
                    "Transcript source_id does not match manifest event_id: "
                    f"{transcript.get('source_id')!r} != {event_id!r}"
                )
            units, counts = serialize_event(
                transcript,
                args.silence_threshold_seconds,
            )
            characters = sum(len(unit) for unit in units)

            split_units[split].extend(units)
            split_event_rows[split].append(
                {
                    "event_id": event_id,
                    "principal_speaker_id": speaker,
                    "register": register,
                    "unit_count": counts["unit_count"],
                    "inserted_silence_count": counts[
                        "inserted_silence_count"
                    ],
                    "text": "\n".join(units),
                }
            )
            event_rows.append(
                {
                    "event_id": event_id,
                    "principal_speaker_id": speaker,
                    "register": register,
                    "split": split,
                    "transcript_path": str(transcript_path),
                    "duration_seconds": transcript.get("duration", ""),
                    "unit_count": counts["unit_count"],
                    "character_count": characters,
                    "inserted_silence_count": counts[
                        "inserted_silence_count"
                    ],
                    "empty_segment_count": counts["empty_segment_count"],
                }
            )
            aggregate[split]["event_count"] += 1
            aggregate[split]["unit_count"] += counts["unit_count"]
            aggregate[split]["character_count"] += characters
            aggregate[split]["inserted_silence_count"] += counts[
                "inserted_silence_count"
            ]

        for split in SPLITS:
            if not split_event_rows[split]:
                raise ValueError(f"Split {split!r} has no events")
            (temporary_dir / f"{split}.txt").write_text(
                "\n".join(split_units[split]) + "\n",
                encoding="utf-8",
            )
            write_jsonl(
                temporary_dir / f"{split}_events.jsonl",
                split_event_rows[split],
            )

        write_csv(
            temporary_dir / "split_manifest.csv",
            event_rows,
            [
                "event_id",
                "principal_speaker_id",
                "register",
                "split",
                "transcript_path",
                "duration_seconds",
                "unit_count",
                "character_count",
                "inserted_silence_count",
                "empty_segment_count",
            ],
        )
        write_json(
            temporary_dir / "special_tokens.json",
            {"special_tokens": SPECIAL_TOKENS},
        )
        write_json(
            temporary_dir / "summary.json",
            {
                "schema_version": 1,
                "manifest": str(manifest_path),
                "event_count": len(event_rows),
                "split_unit": "complete_event",
                "serialization_unit": (
                    "native_whisper_segment_provisional_prosodic_unit"
                ),
                "silence_marker": "<SILENCE>",
                "silence_measurement": "gap_between_native_whisper_segments",
                "silence_threshold_seconds": args.silence_threshold_seconds,
                "seed": args.seed,
                "requested_fractions": {
                    "train": args.train_fraction,
                    "validation": args.validation_fraction,
                    "test": round(
                        1.0
                        - args.train_fraction
                        - args.validation_fraction,
                        10,
                    ),
                },
                "special_tokens": SPECIAL_TOKENS,
                "splits": aggregate,
                "notes": [
                    "Fillers are preserved as transcribed text.",
                    "Only <SILENCE> is inserted automatically.",
                    (
                        "No <NOISE>, <OVERLAP>, <UNINTELLIGIBLE>, or "
                        "<FILLER> markers are inferred without source annotations."
                    ),
                    (
                        "Whisper segment gaps are provisional timing measurements, "
                        "not audio-derived pauses."
                    ),
                ],
            },
        )

        if output_dir.exists():
            backup = output_dir.with_name(f"{output_dir.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            output_dir.rename(backup)
        temporary_dir.rename(output_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "event_count": len(event_rows),
                "splits": aggregate,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

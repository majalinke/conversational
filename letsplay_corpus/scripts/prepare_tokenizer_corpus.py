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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    duplicates = sorted(event_id for event_id, count in Counter(event_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate event IDs in manifest: {duplicates}")
    return rows


def largest_remainder_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {split: total * fractions[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remaining = total - sum(counts.values())
    order = sorted(
        SPLITS,
        key=lambda split: (raw[split] - counts[split], -SPLITS.index(split)),
        reverse=True,
    )
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def deterministic_split(
    manifest_rows: Sequence[dict[str, str]],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[dict[str, str], dict[str, int]]:
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": 1.0 - train_fraction - validation_fraction,
    }
    target_counts = largest_remainder_counts(len(manifest_rows), fractions)

    by_speaker: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        speaker = row["principal_speaker_id"].strip()
        event_id = row["event_id"].strip()
        if not speaker:
            raise ValueError(f"Missing principal_speaker_id for {event_id}")
        by_speaker[speaker].append(event_id)

    speaker_count = len(by_speaker)
    if any(len(event_ids) < 3 for event_ids in by_speaker.values()):
        counts = {speaker: len(event_ids) for speaker, event_ids in by_speaker.items()}
        raise ValueError(f"At least three events per speaker are required: {counts}")
    for split in SPLITS:
        if target_counts[split] < speaker_count:
            raise ValueError(
                f"Global {split} target {target_counts[split]} cannot include all "
                f"{speaker_count} speakers"
            )

    assignment: dict[str, str] = {}
    assigned_counts = Counter()
    remaining_events: list[tuple[str, str]] = []

    for speaker, event_ids in sorted(by_speaker.items()):
        ordered = sorted(
            event_ids,
            key=lambda event_id: hashlib.sha256(
                f"{seed}:{speaker}:{event_id}".encode("utf-8")
            ).hexdigest(),
        )
        for split, event_id in zip(SPLITS, ordered[:3], strict=True):
            assignment[event_id] = split
            assigned_counts[split] += 1
        remaining_events.extend((speaker, event_id) for event_id in ordered[3:])

    remaining_targets = {
        split: target_counts[split] - assigned_counts[split] for split in SPLITS
    }
    ordered_remaining = sorted(
        remaining_events,
        key=lambda item: hashlib.sha256(
            f"{seed}:remaining:{item[0]}:{item[1]}".encode("utf-8")
        ).hexdigest(),
    )
    for _, event_id in ordered_remaining:
        eligible = [split for split in SPLITS if remaining_targets[split] > 0]
        if not eligible:
            raise AssertionError("No remaining split capacity")
        split = max(
            eligible,
            key=lambda name: (
                remaining_targets[name] / target_counts[name],
                -SPLITS.index(name),
            ),
        )
        assignment[event_id] = split
        remaining_targets[split] -= 1

    if any(remaining_targets.values()):
        raise AssertionError(f"Unfilled split targets: {remaining_targets}")
    realized = Counter(assignment.values())
    if any(realized[split] != target_counts[split] for split in SPLITS):
        raise AssertionError(f"Split allocation mismatch: {realized} != {target_counts}")
    return assignment, target_counts


def serialize_event(
    transcript: dict[str, Any],
    silence_threshold_seconds: float,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcript has no segments")
    ordered = sorted(
        enumerate(segments),
        key=lambda item: (
            float(item[1].get("start", 0.0)),
            float(item[1].get("end", 0.0)),
            item[0],
        ),
    )
    serialized_units: list[str] = []
    unit_records: list[dict[str, Any]] = []
    inserted_silences = 0
    empty_segments = 0
    previous_end = 0.0
    for original_segment_index, segment in ordered:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = normalized_text(segment.get("text", ""))
        gap = start - previous_end
        previous_end = max(previous_end, end)
        if not text:
            empty_segments += 1
            continue
        silence_inserted = gap >= silence_threshold_seconds
        serialized_text = f"<SILENCE> {text}" if silence_inserted else text
        if silence_inserted:
            inserted_silences += 1
        serialized_units.append(serialized_text)
        unit_records.append(
            {
                "segment_index": original_segment_index,
                "start_seconds": start,
                "end_seconds": end,
                "preceding_gap_seconds": gap,
                "silence_inserted": silence_inserted,
                "text": text,
                "serialized_text": serialized_text,
            }
        )
    if not serialized_units:
        raise ValueError("Transcript contains no non-empty text segments")
    return serialized_units, unit_records, {
        "unit_count": len(serialized_units),
        "inserted_silence_count": inserted_silences,
        "empty_segment_count": empty_segments,
    }


def running_word_estimate(units: Sequence[str]) -> int:
    special = set(SPECIAL_TOKENS)
    return sum(1 for token in "\n".join(units).split() if token not in special)


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
    assignment, target_counts = deterministic_split(
        manifest_rows, args.seed, args.train_fraction, args.validation_fraction
    )

    temporary_dir = output_dir.with_name(f".{output_dir.name}.partial")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    split_units: dict[str, list[str]] = {split: [] for split in SPLITS}
    split_event_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    event_rows: list[dict[str, Any]] = []
    aggregate = {
        split: {
            "event_count": 0,
            "unit_count": 0,
            "character_count": 0,
            "running_word_estimate": 0,
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
            units, unit_records, counts = serialize_event(
                transcript, args.silence_threshold_seconds
            )
            characters = sum(len(unit) for unit in units)
            words = running_word_estimate(units)

            split_units[split].extend(units)
            split_event_rows[split].append(
                {
                    "event_id": event_id,
                    "principal_speaker_id": speaker,
                    "register": register,
                    "unit_count": counts["unit_count"],
                    "running_word_estimate": words,
                    "inserted_silence_count": counts["inserted_silence_count"],
                    "units": unit_records,
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
                    "running_word_estimate": words,
                    "inserted_silence_count": counts["inserted_silence_count"],
                    "empty_segment_count": counts["empty_segment_count"],
                }
            )
            aggregate[split]["event_count"] += 1
            aggregate[split]["unit_count"] += counts["unit_count"]
            aggregate[split]["character_count"] += characters
            aggregate[split]["running_word_estimate"] += words
            aggregate[split]["inserted_silence_count"] += counts[
                "inserted_silence_count"
            ]

        for split in SPLITS:
            if not split_event_rows[split]:
                raise ValueError(f"Split {split!r} has no events")
            (temporary_dir / f"{split}.txt").write_text(
                "\n".join(split_units[split]) + "\n", encoding="utf-8"
            )
            write_jsonl(
                temporary_dir / f"{split}_events.jsonl", split_event_rows[split]
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
                "running_word_estimate",
                "inserted_silence_count",
                "empty_segment_count",
            ],
        )
        write_json(temporary_dir / "special_tokens.json", {"special_tokens": SPECIAL_TOKENS})
        write_json(
            temporary_dir / "summary.json",
            {
                "schema_version": 2,
                "manifest": str(manifest_path),
                "event_count": len(event_rows),
                "split_unit": "complete_event",
                "serialization_unit": "native_whisper_segment_provisional_prosodic_unit",
                "silence_marker": "<SILENCE>",
                "silence_measurement": "gap_between_native_whisper_segments",
                "silence_threshold_seconds": args.silence_threshold_seconds,
                "continuous_timing_fields": [
                    "start_seconds",
                    "end_seconds",
                    "preceding_gap_seconds",
                ],
                "seed": args.seed,
                "requested_fractions": {
                    "train": args.train_fraction,
                    "validation": args.validation_fraction,
                    "test": round(1.0 - args.train_fraction - args.validation_fraction, 10),
                },
                "target_event_counts": target_counts,
                "special_tokens": SPECIAL_TOKENS,
                "splits": aggregate,
                "notes": [
                    "Fillers are preserved as transcribed text.",
                    "Only <SILENCE> is inserted automatically.",
                    "No <NOISE>, <OVERLAP>, <UNINTELLIGIBLE>, or <FILLER> markers are inferred without source annotations.",
                    "Whisper segment gaps are provisional timing measurements, not audio-derived pauses.",
                    "Continuous segment start, end, and preceding-gap values are retained in event JSONL records.",
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
            {"output_dir": str(output_dir), "event_count": len(event_rows), "splits": aggregate},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

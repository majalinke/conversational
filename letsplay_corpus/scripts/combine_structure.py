#!/usr/bin/env python3
"""Combine per-event structural annotations into corpus-level analysis tables."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TABLES = (
    ("segment_statistics.csv", "segments.csv"),
    ("token_dependencies.csv", "token_dependencies.csv"),
    ("word_sequence_statistics.csv", "word_sequences.csv"),
)
ROW_METADATA_FIELDS = ("principal_speaker_id", "register")
SUMMARY_FIELDS = (
    "spacy_model",
    "segment_count",
    "token_count",
    "lexical_token_count",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def enriched_fields(fields: list[str]) -> list[str]:
    duplicates = set(fields).intersection(ROW_METADATA_FIELDS)
    if duplicates:
        raise ValueError(f"Input table already contains metadata fields: {sorted(duplicates)}")
    if "source_id" not in fields:
        raise ValueError("Input table lacks source_id")
    position = fields.index("source_id") + 1
    return fields[:position] + list(ROW_METADATA_FIELDS) + fields[position:]


def enriched_row(row: dict[str, str], manifest_row: dict[str, str]) -> dict[str, str]:
    source_id = row.get("source_id", "")
    event_id = manifest_row["event_id"]
    if source_id != event_id:
        raise ValueError(f"Unexpected source_id {source_id!r}; expected {event_id!r}")
    result = dict(row)
    result["principal_speaker_id"] = manifest_row["principal_speaker_id"]
    result["register"] = manifest_row["register"]
    return result


def combine_table(
    manifest_rows: list[dict[str, str]],
    data_root: Path,
    input_name: str,
    output_path: Path,
) -> int:
    expected_fields: list[str] | None = None
    writer: csv.DictWriter[str] | None = None
    row_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        for manifest_row in manifest_rows:
            event_id = manifest_row["event_id"]
            input_path = data_root / "derived" / "letsplay" / event_id / "structure" / input_name
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing structure table: {input_path}")

            with input_path.open(encoding="utf-8", newline="") as input_handle:
                reader = csv.DictReader(input_handle)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV has no header: {input_path}")
                fields = list(reader.fieldnames)
                if expected_fields is None:
                    expected_fields = fields
                    writer = csv.DictWriter(
                        output_handle,
                        fieldnames=enriched_fields(fields),
                        extrasaction="raise",
                    )
                    writer.writeheader()
                elif fields != expected_fields:
                    raise ValueError(
                        f"Schema mismatch in {input_path}: {fields!r} != {expected_fields!r}"
                    )

                assert writer is not None
                for row in reader:
                    writer.writerow(enriched_row(row, manifest_row))
                    row_count += 1

    return row_count


def build_event_summaries(
    manifest_rows: list[dict[str, str]], data_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_rows: list[dict[str, Any]] = []
    speaker_totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "event_count": 0,
            "duration_seconds": 0.0,
            "segment_count": 0,
            "token_count": 0,
            "lexical_token_count": 0,
        }
    )
    totals: dict[str, float | int] = {
        "event_count": 0,
        "duration_seconds": 0.0,
        "segment_count": 0,
        "token_count": 0,
        "lexical_token_count": 0,
    }

    for manifest_row in manifest_rows:
        event_id = manifest_row["event_id"]
        summary_path = data_root / "derived" / "letsplay" / event_id / "structure" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing structure summary: {summary_path}")
        summary = read_json(summary_path)
        if summary.get("source_id") != event_id:
            raise ValueError(f"Unexpected source_id in {summary_path}")

        event_row: dict[str, Any] = dict(manifest_row)
        event_row.update({field: summary[field] for field in SUMMARY_FIELDS})
        event_row["annotated_duration_seconds"] = float(summary.get("duration", 0.0))
        event_rows.append(event_row)

        speaker_id = manifest_row["principal_speaker_id"]
        event_values = {
            "event_count": 1,
            "duration_seconds": float(summary.get("duration", 0.0)),
            "segment_count": int(summary["segment_count"]),
            "token_count": int(summary["token_count"]),
            "lexical_token_count": int(summary["lexical_token_count"]),
        }
        for key, value in event_values.items():
            totals[key] += value
            speaker_totals[speaker_id][key] += value

    corpus_summary = {
        "register": "letsplay",
        "spacy_models": sorted({str(row["spacy_model"]) for row in event_rows}),
        "totals": totals,
        "speakers": dict(sorted(speaker_totals.items())),
    }
    return event_rows, corpus_summary


def write_event_summary(path: Path, rows: Iterable[dict[str, Any]], manifest_fields: list[str]) -> None:
    fields = manifest_fields + ["annotated_duration_seconds", *SUMMARY_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    manifest_fields, manifest_rows = read_csv(manifest_path)
    if not manifest_rows:
        raise ValueError("Manifest contains no events")
    event_ids = [row["event_id"] for row in manifest_rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Manifest contains duplicate event_id values")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    try:
        event_rows, corpus_summary = build_event_summaries(manifest_rows, data_root)
        write_event_summary(temp_dir / "events.csv", event_rows, manifest_fields)

        table_counts: dict[str, int] = {}
        for input_name, output_name in TABLES:
            table_counts[output_name] = combine_table(
                manifest_rows,
                data_root,
                input_name,
                temp_dir / output_name,
            )

        corpus_summary["tables"] = table_counts
        corpus_summary["manifest"] = str(manifest_path)
        write_json(temp_dir / "summary.json", corpus_summary)

        backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if output_dir.exists():
            output_dir.rename(backup_dir)
        temp_dir.rename(output_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    print(f"events={len(manifest_rows)}")
    for name, count in table_counts.items():
        print(f"{name}={count}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge syntax and audio tables segment by segment for complete events."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_merge_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "shared_merge_syntax_audio",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load merge module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def integer_field(row: dict[str, str], name: str) -> int | None:
    value = str(row.get(name, "")).strip()
    return int(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-module", type=Path, required=True)
    parser.add_argument("--syntax-csv", type=Path, required=True)
    parser.add_argument("--audio-word-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    merge = load_merge_module(args.merge_module.resolve())
    syntax_rows = read_csv(args.syntax_csv.resolve())
    audio_rows = read_csv(args.audio_word_csv.resolve())

    syntax_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    audio_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)

    for row in syntax_rows:
        segment = integer_field(row, "segment_index")
        if segment is not None:
            syntax_groups[(str(row["source_id"]), segment)].append(row)

    for row in audio_rows:
        if str(row.get("alignment_available", "")).casefold() not in {"true", "1"}:
            continue
        segment = integer_field(row, "native_whisper_segment_index")
        if segment is not None:
            audio_groups[(str(row["source_id"]), segment)].append(row)

    keys = sorted(set(syntax_groups) | set(audio_groups))
    merged_rows: list[dict[str, Any]] = []
    segment_reports: list[dict[str, Any]] = []

    for source_id, segment_index in keys:
        merged, report = merge.merge_source(
            source_id,
            syntax_groups.get((source_id, segment_index), []),
            audio_groups.get((source_id, segment_index), []),
        )
        merged_rows.extend(merged)
        report["segment_index"] = segment_index
        segment_reports.append(report)

    source_reports = []
    source_ids = sorted({source_id for source_id, _ in keys})

    for source_id in source_ids:
        reports = [
            report
            for report in segment_reports
            if report["source_id"] == source_id
        ]

        syntax_count = sum(r["syntax_row_count"] for r in reports)
        audio_count = sum(r["audio_word_count"] for r in reports)
        matched_syntax = sum(r["matched_syntax_row_count"] for r in reports)
        matched_audio = sum(r["matched_audio_word_count"] for r in reports)

        source_reports.append(
            {
                "source_id": source_id,
                "segment_count": len(reports),
                "syntax_row_count": syntax_count,
                "audio_word_count": audio_count,
                "matched_syntax_row_count": matched_syntax,
                "matched_audio_word_count": matched_audio,
                "syntax_match_rate": (
                    matched_syntax / syntax_count if syntax_count else 0.0
                ),
                "audio_word_match_rate": (
                    matched_audio / audio_count if audio_count else 0.0
                ),
                "grouped_audio_word_count": sum(
                    r["grouped_audio_word_count"] for r in reports
                ),
                "segment_reports": reports,
            }
        )

    fields = list(syntax_rows[0]) if syntax_rows else []
    extra_fields = [
        "audio_alignment_match",
        "audio_alignment_match_type",
        "audio_alignment_group_id",
        "audio_alignment_syntax_token_count",
        "audio_timing_primary_row",
        "audio_timing_shared_across_syntax_tokens",
        "audio_word",
    ]

    for field in [
        *extra_fields,
        *merge.ordered_audio_fields(audio_rows),
    ]:
        if field not in fields:
            fields.append(field)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merge.write_csv(
        output_dir / "word_sequence_audio_statistics.csv",
        merged_rows,
        fields,
    )

    report = {
        "merge_scope": "within_native_whisper_segment",
        "sources": source_reports,
    }
    (output_dir / "syntax_audio_merge_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

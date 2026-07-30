#!/usr/bin/env python3
"""Extend an existing complete-event Let’s Play manifest to larger speaker targets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_bulk_manifest import (
    DEFAULT_EXCLUDED_TITLE_TERMS,
    event_row,
    excluded_by_title,
    known_video_ids,
    normalized_duration,
    read_csv,
    video_id,
    write_csv,
    yt_dlp_channel_entries,
)


def duration_seconds(row: dict[str, str]) -> float:
    value = str(row.get("duration_seconds", "")).strip()
    if not value:
        raise ValueError(f"Missing duration_seconds for existing event {row.get('event_id')!r}")
    try:
        duration = float(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid duration_seconds for existing event {row.get('event_id')!r}: {value!r}"
        ) from error
    if duration <= 0:
        raise ValueError(f"Non-positive duration for existing event {row.get('event_id')!r}")
    return duration


def require_unique(rows: list[dict[str, Any]], field: str, label: str) -> None:
    values = [str(row.get(field, "")).strip() for row in rows]
    missing = [index for index, value in enumerate(values) if not value]
    if missing:
        raise ValueError(f"{label} has empty {field} values at rows {missing[:10]}")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} has duplicate {field} values: {duplicates[:20]}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--new-manifest-output", type=Path, required=True)
    parser.add_argument("--full-manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--known-sources", type=Path)
    args = parser.parse_args()

    targets = read_csv(args.targets.resolve())
    existing_rows = read_csv(args.existing_manifest.resolve())
    if not targets:
        raise ValueError("Expansion target file has no speakers")
    if not existing_rows:
        raise ValueError("Existing manifest has no events")

    require_unique(existing_rows, "event_id", "existing manifest")
    require_unique(existing_rows, "source_video_id", "existing manifest")

    target_speakers = [row["principal_speaker_id"].strip() for row in targets]
    if len(target_speakers) != len(set(target_speakers)):
        raise ValueError("Expansion target file contains duplicate speakers")

    existing_by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in existing_rows:
        speaker = row.get("principal_speaker_id", "").strip()
        if not speaker:
            raise ValueError(f"Missing principal_speaker_id for {row.get('event_id')!r}")
        existing_by_speaker[speaker].append(row)

    unexpected_speakers = sorted(set(existing_by_speaker) - set(target_speakers))
    missing_existing_speakers = sorted(set(target_speakers) - set(existing_by_speaker))
    if unexpected_speakers or missing_existing_speakers:
        raise ValueError(
            "Existing manifest and expansion targets do not contain the same speakers: "
            f"unexpected={unexpected_speakers}, missing={missing_existing_speakers}"
        )

    existing_ids = {row["source_video_id"].strip() for row in existing_rows}
    pilot_ids = known_video_ids(args.known_sources.resolve() if args.known_sources else None)

    candidate_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    has_shortfall = False

    for target in targets:
        speaker_id = target["principal_speaker_id"].strip()
        channel_url = target["channel_url"].strip()
        target_seconds = float(target["target_audio_hours"]) * 3600.0
        minimum_seconds = float(target["min_event_minutes"]) * 60.0
        maximum_seconds = float(target["max_event_minutes"]) * 60.0
        maximum_videos = int(target["max_channel_videos"])

        speaker_existing_rows = existing_by_speaker[speaker_id]
        existing_seconds = sum(duration_seconds(row) for row in speaker_existing_rows)
        new_seconds = 0.0
        new_count = 0

        entries = yt_dlp_channel_entries(channel_url, maximum_videos)
        for entry in entries:
            identifier = video_id(entry)
            title = str(entry.get("title") or "").strip()
            duration = normalized_duration(entry)
            if not identifier or not title:
                continue

            if identifier in existing_ids:
                status = "already_in_existing_manifest"
                reason = "event is already present in the existing bulk manifest"
            elif identifier in pilot_ids:
                status = "known_pilot_source"
                reason = "event is already listed in sources.yml"
            elif duration is None:
                status = "excluded_missing_duration"
                reason = "flat channel metadata did not include duration"
            elif duration < minimum_seconds or duration > maximum_seconds:
                status = "excluded_duration"
                reason = (
                    f"duration outside {minimum_seconds / 60:.0f}–"
                    f"{maximum_seconds / 60:.0f} minute range"
                )
            else:
                matched_term = excluded_by_title(title, DEFAULT_EXCLUDED_TITLE_TERMS)
                if matched_term is not None:
                    status = "excluded_title"
                    reason = f"title contains excluded term: {matched_term.strip()}"
                elif existing_seconds + new_seconds < target_seconds:
                    status = "selected_for_expansion"
                    reason = "selected until the speaker's total expansion target was reached"
                    new_seconds += duration
                    new_count += 1
                else:
                    status = "reserve"
                    reason = "speaker total expansion target already reached"

            row = event_row(speaker_id, channel_url, entry, status, reason)
            candidate_rows.append(row)
            if status == "selected_for_expansion":
                new_rows.append(row)

        final_seconds = existing_seconds + new_seconds
        shortfall_seconds = max(0.0, target_seconds - final_seconds)
        if shortfall_seconds > 0:
            has_shortfall = True
        summaries.append(
            {
                "principal_speaker_id": speaker_id,
                "target_audio_hours": round(target_seconds / 3600.0, 3),
                "existing_event_count": len(speaker_existing_rows),
                "existing_audio_hours": round(existing_seconds / 3600.0, 3),
                "new_event_count": new_count,
                "new_audio_hours": round(new_seconds / 3600.0, 3),
                "final_event_count": len(speaker_existing_rows) + new_count,
                "final_audio_hours": round(final_seconds / 3600.0, 3),
                "estimated_final_words_at_100_wpm": round(final_seconds * 100.0 / 60.0),
                "shortfall_audio_hours": round(shortfall_seconds / 3600.0, 3),
            }
        )

    full_rows: list[dict[str, Any]] = [*existing_rows, *new_rows]
    require_unique(new_rows, "event_id", "new expansion manifest")
    require_unique(new_rows, "source_video_id", "new expansion manifest")
    require_unique(full_rows, "event_id", "full expanded manifest")
    require_unique(full_rows, "source_video_id", "full expanded manifest")

    write_csv(args.candidate_output.resolve(), candidate_rows)
    write_csv(args.new_manifest_output.resolve(), new_rows)
    write_csv(args.full_manifest_output.resolve(), full_rows)

    summary = {
        "schema_version": 1,
        "existing_manifest": str(args.existing_manifest.resolve()),
        "target_file": str(args.targets.resolve()),
        "selection_unit": "complete_event",
        "existing_event_count": len(existing_rows),
        "new_event_count": len(new_rows),
        "full_event_count": len(full_rows),
        "estimated_full_audio_hours": round(
            sum(duration_seconds(row) for row in full_rows) / 3600.0, 3
        ),
        "estimated_full_words_at_100_wpm": round(
            sum(duration_seconds(row) for row in full_rows) * 100.0 / 60.0
        ),
        "speakers": summaries,
        "target_reached": not has_shortfall,
    }
    write_json(args.summary_output.resolve(), summary)
    print(json.dumps(summary, indent=2))

    if has_shortfall:
        print(
            "ERROR: one or more speakers did not reach the configured target; "
            "do not continue to download",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise

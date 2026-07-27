#!/usr/bin/env python3
"""Discover complete Let’s Play events and build a balanced bulk manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_EXCLUDED_TITLE_TERMS = (
    "#shorts",
    " short",
    "trailer",
    "review",
    "preview",
    "guide",
    "tutorial",
    "tips",
    "announcement",
    "channel update",
    "reaction",
    "highlights",
    "highlight reel",
    "compilation",
    "supercut",
    "stream vod",
    "livestream",
    "live stream",
    "first impressions",
    "first impression",
)

OUTPUT_FIELDS = (
    "event_id",
    "register",
    "principal_speaker_id",
    "source_platform",
    "source_url",
    "source_video_id",
    "source_channel_url",
    "title",
    "duration_seconds",
    "estimated_words_at_100_wpm",
    "publication_date",
    "event_start_seconds",
    "event_end_seconds",
    "editing_level",
    "selection_status",
    "selection_reason",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def known_video_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"(?:watch\?v=|youtu\.be/)([A-Za-z0-9_-]{6,})", text))


def yt_dlp_channel_entries(channel_url: str, maximum: int) -> list[dict[str, Any]]:
    executable = shutil.which("yt-dlp")
    if executable is None:
        raise RuntimeError("yt-dlp is not available on PATH")

    command = [
        executable,
        "--flat-playlist",
        "--dump-single-json",
        "--playlist-end",
        str(maximum),
        "--no-warnings",
        channel_url,
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)
    entries = payload.get("entries") or []
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected yt-dlp response for {channel_url}")
    return [entry for entry in entries if isinstance(entry, dict)]


def video_id(entry: dict[str, Any]) -> str:
    value = str(entry.get("id") or entry.get("url") or "").strip()
    if "watch?v=" in value:
        value = value.split("watch?v=", 1)[1].split("&", 1)[0]
    return value


def video_url(entry: dict[str, Any], identifier: str) -> str:
    value = str(entry.get("webpage_url") or entry.get("url") or "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://www.youtube.com/watch?v={identifier}"


def normalized_duration(entry: dict[str, Any]) -> float | None:
    value = entry.get("duration")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def publication_date(entry: dict[str, Any]) -> str:
    value = str(entry.get("upload_date") or "").strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return ""


def excluded_by_title(title: str, terms: tuple[str, ...]) -> str | None:
    lowered = f" {title.casefold()} "
    for term in terms:
        if term.casefold() in lowered:
            return term
    return None


def event_row(
    speaker_id: str,
    channel_url: str,
    entry: dict[str, Any],
    status: str,
    reason: str,
) -> dict[str, Any]:
    identifier = video_id(entry)
    duration = normalized_duration(entry)
    return {
        "event_id": f"letsplay_{speaker_id}_{identifier}",
        "register": "letsplay",
        "principal_speaker_id": speaker_id,
        "source_platform": "youtube",
        "source_url": video_url(entry, identifier),
        "source_video_id": identifier,
        "source_channel_url": channel_url,
        "title": str(entry.get("title") or "").strip(),
        "duration_seconds": round(duration, 3) if duration is not None else "",
        "estimated_words_at_100_wpm": (
            round(duration * 100.0 / 60.0) if duration is not None else ""
        ),
        "publication_date": publication_date(entry),
        "event_start_seconds": 0,
        "event_end_seconds": round(duration, 3) if duration is not None else "",
        "editing_level": "uncertain",
        "selection_status": status,
        "selection_reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--known-sources", type=Path)
    args = parser.parse_args()

    targets = read_csv(args.targets.resolve())
    existing_ids = known_video_ids(
        args.known_sources.resolve() if args.known_sources else None
    )

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for target in targets:
        speaker_id = target["principal_speaker_id"].strip()
        channel_url = target["channel_url"].strip()
        target_seconds = float(target["target_audio_hours"]) * 3600.0
        minimum_seconds = float(target["min_event_minutes"]) * 60.0
        maximum_seconds = float(target["max_event_minutes"]) * 60.0
        maximum_videos = int(target["max_channel_videos"])

        entries = yt_dlp_channel_entries(channel_url, maximum_videos)
        selected_seconds = 0.0
        selected_count = 0

        for entry in entries:
            identifier = video_id(entry)
            title = str(entry.get("title") or "").strip()
            duration = normalized_duration(entry)

            if not identifier or not title:
                continue

            if identifier in existing_ids:
                status = "known_source"
                reason = "already listed in sources.yml"
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
                elif selected_seconds < target_seconds:
                    status = "selected_for_bulk"
                    reason = "selected in channel order until speaker target was reached"
                    selected_seconds += duration
                    selected_count += 1
                else:
                    status = "reserve"
                    reason = "speaker duration target already reached"

            row = event_row(speaker_id, channel_url, entry, status, reason)
            candidate_rows.append(row)
            if status == "selected_for_bulk":
                selected_rows.append(row)

        summaries.append(
            {
                "principal_speaker_id": speaker_id,
                "selected_event_count": selected_count,
                "selected_audio_hours": round(selected_seconds / 3600.0, 2),
                "estimated_words_at_100_wpm": round(selected_seconds * 100.0 / 60.0),
            }
        )

    write_csv(args.candidate_output.resolve(), candidate_rows)
    write_csv(args.manifest_output.resolve(), selected_rows)

    print(json.dumps({"speakers": summaries, "selected_total": len(selected_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise

#!/usr/bin/env python3

import argparse
import importlib.metadata
import json
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    model = WhisperModel(
        args.model,
        device="cuda",
        compute_type="float16",
        download_root="/cache/faster-whisper",
    )

    segment_stream, info = model.transcribe(
        str(args.audio),
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    segments = []
    for segment in segment_stream:
        words = [
            {
                "start": word.start,
                "end": word.end,
                "word": word.word,
                "probability": word.probability,
            }
            for word in (segment.words or [])
        ]
        segments.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": words,
            }
        )

    output = {
        "source_id": args.source_id,
        "language": info.language,
        "duration": info.duration,
        "model": args.model,
        "faster_whisper_version": importlib.metadata.version("faster-whisper"),
        "vad_filter": False,
        "condition_on_previous_text": False,
        "segments": segments,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"language={info.language}")
    print(f"duration_seconds={info.duration:.3f}")
    print(f"segment_count={len(segments)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

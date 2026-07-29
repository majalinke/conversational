#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def dependency_depth(token: Any) -> int:
    depth = 0
    current = token
    visited: set[int] = set()
    while current.head.i != current.i and current.i not in visited:
        visited.add(current.i)
        current = current.head
        depth += 1
    return depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="en_core_web_sm")
    args = parser.parse_args()

    import spacy

    transcript = read_json(args.transcript)
    source_id = str(transcript["source_id"])
    language = str(transcript.get("language", "en"))
    duration = float(transcript.get("duration", 0.0))
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcript has no segments")

    nlp = spacy.load(args.model)
    texts = [str(segment.get("text", "")).strip() for segment in segments]
    docs = list(nlp.pipe(texts, batch_size=64))

    segment_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    word_sequence_rows: list[dict[str, Any]] = []
    processed_segments: list[dict[str, Any]] = []

    for segment_index, (segment, doc) in enumerate(zip(segments, docs, strict=True)):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        segment_duration = max(0.0, end - start)

        non_space_tokens = [token for token in doc if not token.is_space]
        lexical_tokens = [token for token in non_space_tokens if not token.is_punct]
        dependency_tokens = [
            token for token in lexical_tokens if token.head.i != token.i
        ]
        sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]

        sentence_index_by_token: dict[int, int] = {}
        sentence_token_index: dict[int, int] = {}
        for sentence_index, sentence in enumerate(doc.sents):
            for within_sentence_index, token in enumerate(sentence):
                sentence_index_by_token[token.i] = sentence_index
                sentence_token_index[token.i] = within_sentence_index

        processed_tokens: list[dict[str, Any]] = []
        for token in non_space_tokens:
            signed_distance = token.head.i - token.i if token.head.i != token.i else 0
            row = {
                "source_id": source_id,
                "language": language,
                "segment_index": segment_index,
                "sentence_index": sentence_index_by_token.get(token.i, -1),
                "token_index": token.i,
                "sentence_token_index": sentence_token_index.get(token.i, -1),
                "token_text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "morph": str(token.morph),
                "dependency": token.dep_,
                "head_token_index": token.head.i,
                "head_text": token.head.text,
                "dependency_depth": dependency_depth(token),
                "signed_dependency_distance": signed_distance,
                "dependency_length": abs(signed_distance),
                "is_stop": token.is_stop,
                "is_punct": token.is_punct,
                "is_sent_start": bool(token.is_sent_start),
            }
            token_rows.append(row)
            processed_tokens.append(row)

        sequence_id = f"{source_id}__whisper_segment_{segment_index:04d}"
        sequence_length = len(lexical_tokens)
        for sequence_position, token in enumerate(lexical_tokens, start=1):
            signed_distance = token.head.i - token.i if token.head.i != token.i else 0
            word_sequence_rows.append(
                {
                    "source_id": source_id,
                    "sequence_id": sequence_id,
                    "segment_index": segment_index,
                    "sequence_position": sequence_position,
                    "sequence_length": sequence_length,
                    "word": token.text,
                    "lemma": token.lemma_,
                    "coarse_pos": token.pos_,
                    "pos_tag": token.tag_,
                    "syntactic_category": token.dep_,
                    "sentence_index": sentence_index_by_token.get(token.i, -1),
                    "head_word": token.head.text,
                    "dependency_depth": dependency_depth(token),
                    "signed_dependency_distance": signed_distance,
                    "dependency_length": abs(signed_distance),
                    "sequence_duration_seconds": round(segment_duration, 3),
                }
            )

        depths = [dependency_depth(token) for token in lexical_tokens]
        dependency_lengths = [abs(token.head.i - token.i) for token in dependency_tokens]
        segment_rows.append(
            {
                "source_id": source_id,
                "language": language,
                "segment_index": segment_index,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "segment_duration_seconds": round(segment_duration, 3),
                "text": text,
                "character_count": len(text),
                "spacy_token_count": len(non_space_tokens),
                "lexical_token_count": len(lexical_tokens),
                "sentence_count": len(sentences),
                "sentence_texts_json": json.dumps(sentences, ensure_ascii=False),
                "mean_dependency_depth": round(fmean(depths), 3) if depths else 0.0,
                "max_dependency_depth": max(depths, default=0),
                "mean_dependency_length": (
                    round(fmean(dependency_lengths), 3) if dependency_lengths else 0.0
                ),
                "max_dependency_length": max(dependency_lengths, default=0),
                "lexical_tokens_per_second": (
                    round(len(lexical_tokens) / segment_duration, 3)
                    if segment_duration > 0
                    else 0.0
                ),
            }
        )
        processed_segments.append(
            {
                "segment_index": segment_index,
                "start": start,
                "end": end,
                "text": text,
                "sentences": sentences,
                "tokens": processed_tokens,
            }
        )

    output_dir = args.output_dir
    segment_fields = [
        "source_id", "language", "segment_index", "start_seconds", "end_seconds",
        "segment_duration_seconds", "text", "character_count", "spacy_token_count",
        "lexical_token_count", "sentence_count", "sentence_texts_json",
        "mean_dependency_depth", "max_dependency_depth", "mean_dependency_length",
        "max_dependency_length", "lexical_tokens_per_second",
    ]
    token_fields = [
        "source_id", "language", "segment_index", "sentence_index", "token_index",
        "sentence_token_index", "token_text", "lemma", "pos", "tag", "morph",
        "dependency", "head_token_index", "head_text", "dependency_depth",
        "signed_dependency_distance", "dependency_length", "is_stop", "is_punct",
        "is_sent_start",
    ]
    sequence_fields = [
        "source_id", "sequence_id", "segment_index", "sequence_position",
        "sequence_length", "word", "lemma", "coarse_pos", "pos_tag",
        "syntactic_category", "sentence_index", "head_word", "dependency_depth",
        "signed_dependency_distance", "dependency_length", "sequence_duration_seconds",
    ]

    write_csv(output_dir / "segment_statistics.csv", segment_rows, segment_fields)
    write_csv(output_dir / "token_dependencies.csv", token_rows, token_fields)
    write_csv(output_dir / "word_sequence_statistics.csv", word_sequence_rows, sequence_fields)
    write_json(
        output_dir / "processed_transcript.spacy.json",
        {
            "source_id": source_id,
            "language": language,
            "duration": duration,
            "spacy_model": args.model,
            "segments": processed_segments,
        },
    )
    write_json(
        output_dir / "summary.json",
        {
            "source_id": source_id,
            "language": language,
            "duration": duration,
            "spacy_model": args.model,
            "segment_count": len(segment_rows),
            "token_count": len(token_rows),
            "lexical_token_count": len(word_sequence_rows),
        },
    )

    print(f"source_id={source_id}")
    print(f"segments={len(segment_rows)}")
    print(f"tokens={len(token_rows)}")
    print(f"lexical_tokens={len(word_sequence_rows)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()

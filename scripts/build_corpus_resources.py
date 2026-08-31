#!/usr/bin/env python3
"""Build versioned frequency and multi-context JSON resources."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cefr.data.corpus_resources import (
    build_lemma_contexts,
    build_lemma_frequencies,
    file_sha256,
    normalize,
    write_resources,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-files", type=Path, nargs="+", required=True)
    parser.add_argument("--sentence-files", type=Path, nargs="+", required=True)
    parser.add_argument("--lemmas", type=Path, required=True,
                        help="UTF-8 file with one target lemma per line")
    parser.add_argument("--lemma-map", type=Path,
                        help="Optional JSON object mapping surface forms to lemmas")
    parser.add_argument("--max-contexts", type=int, default=0,
                        help="Per-lemma sentence limit; 0 keeps every unique sentence")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    lemmas = [normalize(line) for line in args.lemmas.read_text(encoding="utf-8").splitlines()]
    lemma_map = (
        json.loads(args.lemma_map.read_text(encoding="utf-8"))
        if args.lemma_map else None
    )
    frequencies, frequency_stats = build_lemma_frequencies(args.word_files, lemma_map)
    contexts, context_stats = build_lemma_contexts(
        lemmas, args.sentence_files, args.max_contexts
    )
    inputs = [*args.word_files, *args.sentence_files, args.lemmas]
    if args.lemma_map:
        inputs.append(args.lemma_map)
    write_resources(args.output_dir, frequencies, contexts, {
        "input_sha256": {str(path): file_sha256(path) for path in inputs},
        "frequency_stats": frequency_stats,
        "context_stats": context_stats,
        "frequency_semantics": (
            "aggregated mapped-lemma token count" if lemma_map
            else "surface-normalized token count; no lemmatizer map supplied"
        ),
        "context_semantics": "exact normalized token matches; unique sentences per lemma",
    })
    print(json.dumps({**frequency_stats, **context_stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Reproducible lemma-frequency and lemma-context resource builders."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


TOKEN_RE = re.compile(r"[а-яәғқңөұүһі]+", re.IGNORECASE)


def normalize(value):
    return unicodedata.normalize("NFC", str(value)).strip().casefold()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_lemma_frequencies(word_files, lemma_map=None):
    """Aggregate Leipzig ``*-words.txt`` counts onto normalized lemmas."""
    lemma_map = {normalize(k): normalize(v) for k, v in (lemma_map or {}).items()}
    frequencies, accepted, rejected, mapped = Counter(), 0, 0, 0
    for path in map(Path, word_files):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    rejected += 1
                    continue
                word = normalize(parts[1])
                try:
                    frequency = int(parts[2])
                except ValueError:
                    frequency = 0
                if frequency <= 0 or TOKEN_RE.fullmatch(word) is None:
                    rejected += 1
                    continue
                mapped += word in lemma_map
                frequencies[lemma_map.get(word, word)] += frequency
                accepted += 1
    return dict(sorted(frequencies.items())), {
        "accepted_word_types": accepted,
        "rejected_rows": rejected,
        "lemma_entries": len(frequencies),
        "accepted_tokens": sum(frequencies.values()),
        "mapped_word_types": mapped,
    }


def build_lemma_contexts(lemmas, sentence_files, max_contexts=0):
    """Collect unique exact-token sentences; zero means no per-lemma limit."""
    if max_contexts < 0:
        raise ValueError("max_contexts must be non-negative")
    targets = {normalize(lemma) for lemma in lemmas if normalize(lemma)}
    contexts = {lemma: [] for lemma in targets}
    seen = {lemma: set() for lemma in targets}
    sentence_count = 0
    for path in map(Path, sentence_files):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                sentence = line.rstrip("\n").split("\t", 1)[-1].strip()
                if not sentence:
                    continue
                sentence_count += 1
                normalized_sentence = " ".join(sentence.split())
                for lemma in set(TOKEN_RE.findall(normalize(sentence))) & targets:
                    if normalized_sentence in seen[lemma]:
                        continue
                    if max_contexts and len(contexts[lemma]) >= max_contexts:
                        continue
                    contexts[lemma].append(normalized_sentence)
                    seen[lemma].add(normalized_sentence)
    contexts = {lemma: values for lemma, values in sorted(contexts.items()) if values}
    associations = sum(map(len, contexts.values()))
    return contexts, {
        "sentences_scanned": sentence_count,
        "target_lemmas": len(targets),
        "covered_lemmas": len(contexts),
        "lemma_sentence_associations": associations,
        "max_contexts_per_lemma": max_contexts or None,
        "largest_context_bag": max(map(len, contexts.values()), default=0),
    }


def write_resources(output_dir, frequencies, contexts, manifest):
    """Write a new immutable resource directory and output checksums."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "lemma_freqs.json": frequencies,
        "lemma_sentences.json": contexts,
    }
    for name, payload in outputs.items():
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    manifest["outputs"] = {
        name: file_sha256(output_dir / name) for name in outputs
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "build_lemma_contexts",
    "build_lemma_frequencies",
    "file_sha256",
    "normalize",
    "write_resources",
]

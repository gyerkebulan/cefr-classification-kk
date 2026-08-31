#!/usr/bin/env python3
"""Audit CEFR CSV labels and train/evaluation leakage without logging text."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cefr.data.audit import audit_csv_files, write_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--lemma-column", default="lemma")
    parser.add_argument("--label-column", default="cefr")
    parser.add_argument("--context-column")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_csv_files(
        args.train,
        args.evaluation,
        lemma_column=args.lemma_column,
        label_column=args.label_column,
        context_column=args.context_column,
    )
    write_audit(args.output, report)
    print(json.dumps({
        "train_rows": report["train"]["rows"],
        "evaluation_rows": report["evaluation"]["rows"],
        "normalized_lemma_overlap": report["normalized_lemma_overlap"],
        "normalized_context_overlap": report.get("normalized_context_overlap"),
    }))


if __name__ == "__main__":
    main()

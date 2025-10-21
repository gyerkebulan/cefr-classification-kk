#!/usr/bin/env bash
set -euo pipefail
# Use a fine-tuned awesome-align model to produce word alignments.
MODEL_DIR="models/awesome_kzru"
TEST_FILE="data/parallel/test.kazru"
OUT_FILE="data/parallel/aligned.txt"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Missing $MODEL_DIR. Run scripts/train_align.sh first."
  exit 1
fi
if [ ! -f "$TEST_FILE" ]; then
  echo "Missing $TEST_FILE. Create it with 'kazakh ||| russian' lines."
  exit 1
fi

awesome-align   --model_name_or_path "$MODEL_DIR"   --data_file "$TEST_FILE"   --output_file "$OUT_FILE"   --extraction 'softmax'   --batch_size 16

echo "Wrote alignments to $OUT_FILE"

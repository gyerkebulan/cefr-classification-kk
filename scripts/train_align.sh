#!/usr/bin/env bash
set -euo pipefail
# Fine-tune awesome-align on a prepared file data/parallel/train.kazru
# Each line: <kazakh sentence> ||| <russian sentence>
DATA_FILE="data/parallel/train.kazru"
OUT_DIR="models/awesome_kzru"
EPOCHS=3
BS=16
LR=3e-5

if [ ! -f "$DATA_FILE" ]; then
  echo "Missing $DATA_FILE. Create it with 'kazakh ||| russian' lines."
  exit 1
fi

awesome-train   --output_dir "$OUT_DIR"   --model_name_or_path bert-base-multilingual-cased   --data_file "$DATA_FILE"   --batch_size $BS   --num_train_epochs $EPOCHS   --learning_rate $LR

echo "Saved fine-tuned model to $OUT_DIR"

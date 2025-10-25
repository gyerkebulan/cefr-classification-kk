# CEFR Pipelines for Kazakh ↔ Russian

This project bundles three practical pipelines that operate on Kazakh↔Russian text pairs:

1. **Alignment CLI** – return token alignments with probabilities.
2. **Word CEFR CLI** – predict the CEFR level for a single Kazakh token.
3. **Text CEFR CLI** – infer the CEFR level of a Kazakh passage.

Each action is exposed as a one‑liner script and can also be invoked from Python.

---

## Quick CLI Reference

```bash
# 1) High-confidence Kazakh bigrams aligned to a single Russian token
python -m cefr.cli align --kaz-text "Ол кітап оқып жатыр" --rus-text "Он читает книгу"
# Outputs rows with: kazakh_phrase, russian_token, kazakh_span, russian_index, alignment_conf (confidence ≥ 0.8 by default)

# 2) Kazakh word CEFR classification
python -m cefr.cli word --kaz-word "кітап"
# (kaz_word, cefr_level, confidence)

# 3) Kazakh text CEFR classification
python -m cefr.cli text --kaz-text "Ол кітап оқып жатыр. Бүгін мектепте жаңа тақырып өткен." # TODO: --rus-text 
# (kaz_text, cefr_level, confidence)

# 4) Generate silver word labels (optional)
python -m cefr.cli silver --parallel-path data/parallel/kazparc_kz_ru.csv --output-path data/labels/silver_word_labels.csv

# 5) Train word-level CEFR classifier (silver labels)
python -m cefr.cli train-word --dataset-path data/labels/silver_word_labels.csv --output-dir models/transformer_word_cefr --rebuild

# 6) Train text-level CEFR classifier (bilingual model)
python -m cefr.cli train-text
```

If `--rus-text` / `--rus-word` is omitted, the built-in translator will generate a Russian counterpart automatically (requires the translation model defined in the config).

Use `--pretty` to format JSON output.

The training commands rely on Hugging Face models/datasets. Ensure you have a valid token for any gated models (e.g. `cointegrated/rubert-tiny2`) and install the extras:

```bash
pip install transformers datasets
```

The word-level trainer consumes the silver labels in `data/labels/silver_word_labels.csv`. Regenerate them with `python -m cefr.cli silver --rebuild` after extending the parallel corpus or Russian CEFR lexicon, then rerun `train-word` to refresh the transformer model.

### Word-level transformer inference

```python
from pathlib import Path
from cefr.models.word_transformer import predict_word_level, predict_word_distribution

model_dir = Path("models/transformer_word_cefr")

print(predict_word_level("пример", model_dir=model_dir))
print(predict_word_distribution("пример", model_dir=model_dir))
```

## License

This project is intended for research use. Respect the licenses of all datasets and upstream models.

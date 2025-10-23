# CEFR Pipelines for Kazakh ↔ Russian

This project bundles three practical pipelines that operate on Kazakh↔Russian text pairs:

1. **Alignment CLI** – return token alignments with probabilities.
2. **Word CEFR CLI** – predict the CEFR level for a single Kazakh token.
3. **Text CEFR CLI** – infer the CEFR level of a Kazakh passage.

Each action is exposed as a one‑liner script and can also be invoked from Python.

---

## Quick CLI Reference

```bash
# 1) Token alignment
python -m cefr.cli align --kaz-text "Ол кітап оқып жатыр" --rus-text "Он читает книгу"

# 2) Kazakh word CEFR classification
python -m cefr.cli word --kaz-word "кітап"

# 3) Kazakh text CEFR classification
python -m cefr.cli text --kaz-text "Ол кітап оқып жатыр. Бүгін мектепте жаңа тақырып өткен."
```

If `--rus-text` / `--rus-word` is omitted, the built-in translator will generate a Russian counterpart automatically (requires the translation model defined in the config).

Use `--pretty` to format JSON output.

---

## 1. Alignment Pipeline

Command:

```bash
python -m cefr.cli align --kaz-text "Ол кітап оқып жатыр" --rus-text "Он читает книгу" --pretty
```

Example output:

```json
{
  "kaz_text": "Ол кітап оқып жатыр",
  "rus_text": "Он читает книгу",
  "translation_used": false,
  "alignments": [
    {
      "kaz_index": 0,
      "kaz_token": "Ол",
      "rus_index": 0,
      "rus_token": "Он",
      "probability": 0.82
    },
    {
      "kaz_index": 1,
      "kaz_token": "кітап",
      "rus_index": 2,
      "rus_token": "книгу",
      "probability": 0.91
    }
  ]
}
```

- `probability` is the minimum mutual attention between the aligned tokens (`≈1.0` is a strong match).
- If `translation_used` is `true`, the script translated the Kazakh text before alignment.

### Programmatic usage

```python
from cefr import load_config
from cefr.alignment import EmbeddingAligner
from cefr.text_utils import tokenize_words

cfg = load_config()
aligner = EmbeddingAligner(cfg.pipeline.alignment)
kz_tokens = tokenize_words("Ол кітап оқып жатыр")
ru_tokens = tokenize_words("Он читает книгу")

diag = aligner.diagnostics(kz_tokens, ru_tokens)
alignments = [
    (row["kaz_index"], row["rus_index"], row["joint_prob"])
    for row in diag.iter_rows(kz_tokens, ru_tokens)
    if row["is_link"]
]
```

---

## 2. Word CEFR Classifier

Command:

```bash
python -m cefr.cli word --kaz-word "кітап" --pretty
```

Example output:

```json
{
  "kaz_word": "кітап",
  "rus_token": "книгу",
  "translation_used": true,
  "cefr_level": "B1",
  "confidence": 0.82,
  "distribution": {
    "A1": 0.04,
    "A2": 0.09,
    "B1": 0.82,
    "B2": 0.04,
    "C1": 0.01,
    "C2": 0.00
  }
}
```

The script translates, aligns, and consults the Russian CEFR lexicon to estimate the level of the supplied word.

### Programmatic usage

```python
from cefr import load_config
from cefr.pipeline import TextPipeline

pipeline = TextPipeline(config=load_config().pipeline)
prediction = pipeline.predict("кітап")
alignment = prediction.word_alignments[0]
level = alignment.cefr
confidence = prediction.distribution.get(level, 0.0)
```

---

## 3. Text CEFR Classifier

Command:

```bash
python -m cefr.cli text --kaz-text "Ол кітап оқып жатыр. Бүгін мектепте жаңа тақырып өткен." --pretty
```

Example output:

```json
{
  "kaz_text": "Ол кітап оқып жатыр. Бүгін мектепте жаңа тақырып өткен.",
  "rus_text": "Он читает книгу. Сегодня в школе прошли новую тему.",
  "translation_used": true,
  "cefr_level": "B1",
  "confidence": 0.73,
  "distribution": {
    "A1": 0.02,
    "A2": 0.12,
    "B1": 0.73,
    "B2": 0.11,
    "C1": 0.02,
    "C2": 0.00
  }
}
```

The script aggregates phrase-level estimates from the alignment pipeline to deliver an overall CEFR assessment.

### Programmatic usage

```python
from cefr import load_config
from cefr.pipeline import TextPipeline

pipeline = TextPipeline(config=load_config().pipeline)
prediction = pipeline.predict("Ол кітап оқып жатыр. Бүгін мектепте жаңа тақырып өткен.")
level = prediction.average_level
confidence = prediction.distribution.get(level, 0.0)
```

---

## Environment Setup

```bash
git clone https://github.com/<your-org>/cefr-classification-kk.git
cd cefr-classification-kk
conda env create -f environment.yml
conda activate kazakh_cefr_env
python -m cefr.data.download      # optional: fetch sample parallel data
```

Ensure translation models and CEFR resources referenced in `config/default.yaml` are available locally.

---

## Repository Layout

```
alignment.py                  # CLI: token alignment
word_cefr.py                  # CLI: word-level CEFR scoring
text_cefr.py                  # CLI: text-level CEFR scoring
cefr/
  alignment.py                # alignment utilities
  cli.py                      # shared CLI entry points (align, word, text)
  pipeline.py                 # translation + alignment + scoring
  scoring.py                  # CEFR aggregation logic
  text_utils.py               # token helpers
  translation.py              # translation wrapper
  training/                   # training scripts for advanced models
data/                         # corpora and generated artefacts
models/                       # saved models (gitignored)
notebooks/                    # exploratory workflows
```

---

## License

This project is intended for research use. Respect the licenses of all datasets and upstream models.

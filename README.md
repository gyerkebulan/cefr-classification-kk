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

## License

This project is intended for research use. Respect the licenses of all datasets and upstream models.

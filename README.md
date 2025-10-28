# CEFR‑пайплайны для казахско‑русского корпуса

Набор инструментов для оценки уровней CEFR в казахских текстах через русский «пивот». Пайплайн переводит Kaz→Ru (`issai/tilmash`), выравнивает токены (awesome-align), ищет CEFR по русскому лексикону и при необходимости добавляет трансформерные модели: word‑классификатор и TF-IDF текстовый классификатор. Всё доступно через CLI и Python; параметры задаются в `config/default.yaml`.

---

## Установка и зависимости

```bash
conda env create -f environment.yml
conda activate kazakh_cefr_env
```

Для закрытых HF‑моделей выполните `huggingface-cli login`.

---

## Быстрый обзор пайплайна

1. Казахский текст переводится в русский (`issai/tilmash`).
2. `EmbeddingAligner` строит пары токенов Kaz↔Ru.
3. `CefrScorer` проверяет русские токены по словарю CEFR (lemmas → уровень).
4. Дополнительно доступны:
   - `WordCefrPipeline` — трансформер по русским токенам (`models/transformer_word_cefr`).
   - `EnsemblePipeline` — объединение лексиконного и sentence‑классификатора.
5. Любая стадия конфигурируется и вызывается из `TextPipeline` или напрямую.

---

## CLI

```bash
python -m cefr.cli align --kaz-text "..." [--rus-text "..."]   # выравнивание токенов + метрики
python -m cefr.cli word --kaz-word "..." [--rus-word "..."]    # уровень конкретного слова
python -m cefr.cli text --kaz-text "..." [--rus-text "..."]    # распределение уровней для текста
python -m cefr.cli silver --parallel-path data/parallel/...    # генерация «серебряных» меток
python -m cefr.cli train-word --dataset-path data/labels/...   # обучение word-transformer
python -m cefr.cli train-text --dataset-path data/text/...     # обучение TF-IDF классификатора
```

Все команды используют конфиг из `config/default.yaml`; добавьте `--pretty` для форматированного вывода. Используйте `--rus-text` / `--rus-word`, если хотите подать собственный перевод.

---

## Python‑API

```python
from cefr import load_config, TextPipeline

cfg = load_config()
pipeline = TextPipeline(config=cfg.pipeline)
result = pipeline.predict("Ол мектепке бара жатыр.")
print(result.average_level)          # средний уровень
print(result.word_alignment_tuples)  # [(kaz, rus, idx_kz, idx_ru, cefr), ...]
```

```python
from pathlib import Path
from cefr.models.word_transformer import predict_word_distribution

dist = predict_word_distribution("пример", model_dir=Path("models/transformer_word_cefr"))
print(dist)  # {'A1': ..., 'A2': ..., ...}
```

---

## Данные и артефакты

- `data/parallel/kazparc_kz_ru.csv` — исходный параллельный корпус.
- `data/labels/silver_word_labels.csv` — авторазметка русских токенов уровнями CEFR.
- `data/cefr/russian_cefr_sample.csv` — словарь «слово → уровень».
- `models/transformer_word_cefr/` — обученный word‑трансформер и метрики.
- `models/text_classifier/` — TF-IDF классификатор текстов и отчёт.

---

## Настройки и обучение

- Настраивайте переводчик, выравниватель, пути к моделям и веса ансамбля через `config/default.yaml` или `load_config(path=...)`.
- Обновляйте серебряные метки перед обучением трансформера (`python -m cefr.cli silver --rebuild`).
- После `train-word` или `train-text` смотрите JSON‑метрики в соответствующих каталогах `models/...`.
- `train-text` автоматически добавляет признаки распределения CEFR по токенам (используется модель из `--word-model-dir`, по умолчанию `models/transformer_word_cefr`).

---

## Тесты

```bash
pytest
```

# CEFR-пакет: перевод, выравнивание и оценка сложности

`cefr` — это Python-пакет, который объединяет всю инфраструктуру для оценки уровня владения языком (CEFR) в проектах на базе казахско-русской пары. Ниже описаны основные модули, точки расширения и сценарии использования.

---

## Структура каталога

```
cefr/
├── alignment.py          # Модуль выравнивания слов по контекстным эмбеддингам
├── cli.py                # CLI-команды для прогнозов, тренировки и диагностики
├── config.py             # dataclass-конфигурация пайплайна и переводчика
├── data/                 # доступ к лексикону CEFR и генерация «серебряных» меток
├── models/               # модели (word transformer, ru sentence classifier)
├── pipeline.py           # основная бизнес-логика для TextPipeline и ансамбля
├── scoring.py            # агрегация токенов в распределение CEFR
├── text_features.py      # вычисление числовых признаков для текстового классификатора
├── text_utils.py         # общие утилиты: токенизация, фильтры
└── training/             # скрипты обучения (слово, текст, табличные признаки)
```

---

## Конфигурация

- Базовый YAML: `config/default.yaml` (загружается через `cefr.load_config()`).
- Ключевые секции: 
  - `translator` — модель и устройство для перевода (`issai/tilmash` по умолчанию).
  - `alignment` — модель выравнивания (`aneuraz/awesome-align-with-co`, слой, порог).
  - `russian_cefr_path` — CSV со словарем «русское слово → уровень CEFR».
  - `word_model_dir` — директория с обученным трансформером для токенов.
  - `russian_model_dir` и `russian_weight` — параметры ансамбля с русским sentence-классификатором.

Настройки можно изменить програмmatically или передать альтернативный путь в `load_config(path=...)`.

---

## Основные компоненты пакета

### 1. Перевод (`translation.py`)
- Класс `Translator` оборачивает Hugging Face `TranslationPipeline`.
- Метод `translate(text)` выдаёт русскую версию казахского текста.
- Функция `get_translator(config)` кэширует экземпляры по имени модели и устройству.

### 2. Выравнивание (`alignment.py`)
- `EmbeddingAligner.align(kz_words, ru_words)` возвращает пары индексов токенов.
- `merge_kz_to_single_ru(...)` собирает казахские спаны, соответствующие одному русскому токену.
- `AlignmentDiagnostics` предоставляет вероятности и DataFrame для анализа качества связок.
- Используется HF-модель `awesome-align` (контекстные эмбеддинги + взаимное внимание).

### 3. Оценка и скоринг (`scoring.py`, `data/__init__.py`)
- `RussianCefrRepository` лениво загружает лексикон CEFR.
- `CefrScorer.score(alignments)` преобразует выровненные токены в распределение и средний уровень.
- Метод `infer_level` отдаёт уровень для отдельного русского слова.

### 4. Пайплайны (`pipeline.py`)
- `TextPipeline` стыкует перевод → выравнивание → скоринг; возвращает `TextPrediction`.
- `WordCefrPipeline` применяет обученный word-transformer к переводу.
- `EnsemblePipeline` объединяет базовое распределение (из лексикона) и вывод русской sentence-модели.
- `TextPrediction` и `WordCefrPrediction` дают доступ к распределениям, выравниваниям и вспомогательным методам.

### 5. Модели (`models/`)
- `models/word_transformer.py`: загрузка/инференс BERT-подобной модели для русских токенов (`predict_word_batch` и др.).
- `models/ru_sentence.py`: обёртка вокруг HF-классификатора предложений (уровни CEFR).

### 6. Обучение (`training/`)
- `training/word_transformer.py`: пайплайн для fine-tuning модели токенов на «серебряных» метках. Возвращает метрики (accuracy, macro-F1).
- `training/text_classification.py`: TF-IDF + логистическая регрессия с признаками (`compute_text_features`).
- `training/tabular.py`: базовые вспомогательные функции для табличных экспериментов.

### 7. CLI (`cli.py`)
- Команды: `align`, `word`, `text`, `silver`, `train-word`, `train-text`.
- Загружает конфигурацию, строит пайплайны, выводит результаты в JSON.

---

## Генерация «серебряных» меток

1. Подготовьте параллельный корпус в `data/parallel/*.csv` с колонками `kaz` и `rus`.
2. Запустите:
   ```bash
   python -m cefr.cli silver --parallel-path data/parallel/kazparc_kz_ru.csv \
     --output-path data/labels/silver_word_labels.csv
   ```
3. Скрипт:
   - Переводит каждую пару в токены (`text_utils.tokenize_words`).
   - Строит выравнивания (`EmbeddingAligner.align`).
   - Лемматизирует русские токены (pymorphy) и ищет уровни в словаре.
   - Сохраняет CSV `kaz_item`, `rus_item`, `cefr`, `kaz_sent`, `rus_sent`.

Эти данные используются как суррогат разметки для обучения трансформера.

---

## Обучение моделей

- **Слово (русский токен)**  
  ```bash
  python -m cefr.cli train-word \
    --dataset-path data/labels/silver_word_labels.csv \
    --output-dir models/transformer_word_cefr \
    --model-name cointegrated/rubert-tiny2 \
    --epochs 5
  ```
  Результаты: сохранённая модель + `word_transformer_metrics.json`.

- **Текст (двуязычный TF-IDF)**  
  ```bash
  python -m cefr.cli train-text \
    --dataset-path data/text/en_ru_cefr_corpus.csv \
    --output-dir models/text_classifier
  ```
  Файл `text_classifier_metrics.json` содержит accuracy, отчёт и матрицу ошибок.

- **Sentence-модель (HF)**  
  Используйте `RuSentenceCefrModel.from_pretrained(path)` для inference. Обучение выполняется вне пакета (например, через стандартный HF Trainer).

---

## Пример использования пакета

```python
from cefr import load_config, TextPipeline

cfg = load_config()
pipeline = TextPipeline(config=cfg.pipeline)

result = pipeline.predict("Мен ақпан айында Астанаға барамын.")
print(result.average_level)          # Средний уровень (A1–C2 или "Unknown")
print(result.distribution)           # Полное распределение по уровням
print(result.word_alignment_tuples)  # [(каз_токен, рус_токен, idx_kz, idx_ru, cefr), ...]
```

Для прямого обращения к трансформеру:

```python
from pathlib import Path
from cefr.models.word_transformer import predict_word_distribution

dist = predict_word_distribution(
    "пример",
    model_dir=Path("models/transformer_word_cefr"),
)
print(dist)
```

---

## Расширение и кастомизация

- Добавьте/подмените переводчик, изменив `PipelineConfig.translator.model_name`.
- Настройте устройства (`cuda`, `cpu`) через конфиг или параметры конструктора.
- Подключите собственный словарь CEFR, передав путь в `PipelineConfig.russian_cefr_path`.
- Можно внедрить альтернативный алгоритм скоринга, реализовав интерфейс `CefrScorer`.
- Для обслуживания нескольких языков создайте отдельные конфиги и модели в `models/`.

---

## Требования и внешние зависимости

- `transformers`, `torch`, `datasets`, `pymorphy3`/`pymorphy2`, `scikit-learn`, `pandas`, `tqdm`.
- Для загрузки HF-моделей может понадобиться токен (`huggingface-cli login`).
- Некоторые операции (выравнивание, обучение) ускоряются на GPU.

---

Если у вас есть вопросы по внутренним интерфейсам или нужна помощь с интеграцией, загляните в исходники указанных файлов или создайте issue в основном репозитории. Удачной работы с пайплайном!

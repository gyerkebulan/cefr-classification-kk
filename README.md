# Пайплайн CEFR для казахско‑русских текстов

Проект решает задачу оценки сложности текста (CEFR A1–C2) для казахского языка по параллельному переводу на русский. Пайплайн включает перевод, выравнивание слов/фраз, сопоставление с уровнем CEFR и агрегацию на уровне текста. Дополнительно можно обучить классификатор для словарного уровня CEFR.

## Как быстро начать

### 1. Установка окружения
**Conda (рекомендуется):**
```bash
conda env create -f environment.yml
conda activate kazakh_cefr_env
```

**pip / venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Загрузка параллельного корпуса KazParC
```bash
python -m src.data.download_parallel
```
Команда сохранит файл `data/parallel/kazparc_kz_ru.csv` и вернет путь к нему.

### 3. Построение «серебряных» CEFR-меток
```bash
python -m src.pipeline.build_silver_labels
```
В результате появится файл `data/labels/silver_word_labels.csv` с парами «казахская фраза → русский перевод → уровень CEFR».

### 4. Прогон пайплайна через CLI
```bash
python run_pipeline.py --text_kz "Ол кітап оқып жатыр"
```
При желании можно передать готовый перевод и миновать автоматический перевод:

```bash
python run_pipeline.py --text_kz "Ол кітап оқып жатыр" --text_ru "Он читает книгу"
```
На выходе: перевод (или переданный перевод), распределение уровней CEFR и список выровненных фраз.

### 5. Jupyter-ноутбук
В папке `notebooks/` создайте (или обновите) файл `cefr_pipeline_demo.ipynb` содержимым из `main.ipynb`. Ноутбук выполняет все шаги пайплайна в среде Colab/Kaggle или локально, используя GPU при наличии.

## Что происходит внутри

1. **Перевод** — `src/translation/translator.py` оборачивает модель `issai/tilmash`. Кэширование позволяет загружать модель один раз.
2. **Выранивание** — `src/align/mutual_align.py` формирует взаимные соответствия слов через `aneuraz/awesome-align-with-co`. Класс `EmbeddingAligner` поддерживает CPU и GPU.
3. **Сборка фраз** — `src/align/merge_phrases.py` объединяет соседние казахские токены, выровненные с одним русским словом, и возвращает `PhraseAlignment`.
4. **Доменные сервисы** — `src/domain/services.py` реализует `TextCefrPipeline`, который объединяет перевод, выравнивание и скоринг CEFR. Возвращаемый объект `TextCefrPrediction` содержит результат и удобен для сериализации.
5. **Работа с ресурсами** — `src/data/repositories.py` лениво загружает словарь «русское слово → уровень CEFR», чтобы избежать повторного чтения CSV.
6. **Серебряные метки** — `src/pipeline/build_silver_labels.py` запускает пайплайн на параллельном корпусе, дополнительно приводит русские слова к нормальной форме через `pymorphy3`.

## Настройка и GPU

- Все модули автоматически переходят на CUDA, если `torch.cuda.is_available()` возвращает `True`. При необходимости можно явно передать `device="cuda"` в `EmbeddingAligner` и `get_translator`.
- Длинные предложения автоматически пропускаются с предупреждением, чтобы избежать превышения лимита в 512 токенов у BERT.

## Обучение и дообучение

1. **Выравнивание (awesome-align)**  
   - Сформируйте файл `data/parallel/train.kazru` с строками вида `kazakh ||| russian`.  
   - Запустите  
     ```bash
     bash scripts/train_align.sh
     bash scripts/align_infer.sh
     ```  
     Это улучшит качество выравнивания и, как следствие, точность CEFR-оценки.

2. **Классификатор словарных уровней**  
   - После генерации `data/labels/silver_word_labels.csv` выполните:  
     ```bash
     python -m src.models.train_word_transformer
     ```  
   - По умолчанию используется компактный трансформер `cointegrated/rubert-tiny2`; веса и токенизатор сохраняются в `models/transformer_word_cefr/`.  
   - Для инференса доступны хелперы:
     ```python
     from pathlib import Path
     from src.models.predict_transformer_word import predict_transformer_word

     predict_transformer_word("пример", model_dir=Path("models/transformer_word_cefr"))
     ```

3. **Эксперименты**  
   - Увеличьте размер русско-CEFR словаря (`data/cefr/russian_cefr_sample.csv`), заменив его на собственный большой список.  
   - Настройте гиперпараметры в `src/domain/services.py` (слои BERT, пороги выравнивания) для лучшего баланса между покрытием и точностью.  
   - Добавьте собственные метрики и отчеты, интегрировав сервисы пайплайна в ваш продукт.

## Структура проекта

```
data/
  cefr/                        # словари CEFR
  parallel/                    # параллельные корпуса
  labels/                      # сгенерированные серебряные метки
models/
  transformer_word_cefr/       # сохраненные веса трансформер-классификатора
notebooks/
  cefr_pipeline_demo.ipynb     # jupyter-скрипт для полного пайплайна
scripts/
  train_align.sh               # обучение awesome-align
  align_infer.sh               # применение выравнителя
src/
  align/                       # выравнивание и слияние фраз
  data/                        # загрузка данных и репозитории
  domain/                      # доменные сущности и сервисы
  models/                      # обучение и инференс классификатора
  pipeline/                    # построение серебряных меток
  text/                        # текстовый CEFR-предиктор
  translation/                 # обертка над моделью перевода
run_pipeline.py                # CLI-пример
tests/                         # smoke-тесты
```

## Что можно улучшить

- **Расширение корпусов** — подключите дополнительные параллельные источники или собственные наборы данных.
- **Собственные словари** — обогащайте русско-CEFR словарь полноразмерными леммами или вручную размеченными данными.
- **Автовalidation** — добавьте pytest с моками для сервисов (`TranslationService`, `AlignmentService`) и интеграционные тесты для пайплайна.
- **Интерфейс** — заверните `TextCefrPipeline` в REST/CLI-сервис или интерфейс на Streamlit/Gradio.

## Лицензия

Проект предназначен для исследовательских целей. Соблюдайте лицензии используемых моделей, датасетов и внешних сервисов.

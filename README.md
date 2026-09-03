# AI Product Hack — Trigger Model

Командный репозиторий триггерной модели для трансграничных переводов.

В репозитории реализован воспроизводимый исследовательский pipeline: загрузка официальных дневных курсов Банка России, causal-признаки, разметка targets, walk-forward оптимизация rule-based индикаторов и независимый backtest фиксированных G0/W1-правил.

Требуется Python 3.10 или новее.

## Быстрый старт

```bash
git clone https://github.com/YYukh/AI_Product_Hack_trigger_model.git
cd AI_Product_Hack_trigger_model

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

jupyter lab
```

На Windows активация окружения выполняется командой:

```powershell
.venv\Scripts\activate
```

После запуска откройте `notebooks/test.ipynb` и выполните все ячейки сверху вниз. Запускайте Jupyter из корня репозитория либо из каталога `notebooks`: ноутбук корректно определяет корень в обоих случаях.

## Что делает загрузчик

- загружает XML-историю с официального сайта ЦБ РФ;
- поддерживает `AMD`, `KZT`, `KGS`, `TJS`, `UZS`, `USD`, `EUR`, `CNY`;
- сохраняет исходный номинал и исходное значение курса;
- нормализует значение до рублей за одну единицу валюты;
- возвращает temporal-таблицу и широкий DataFrame `дата × валюта`;
- при необходимости сохраняет исходные XML локально.

Исторический XML не содержит точного времени публикации. Поэтому `available_at` и `publication_timestamp` консервативно установлены на `00:00` следующего календарного дня, а поле `publication_timestamp_is_proxy=True` явно маркирует это допущение.

## Что делает исследовательский pipeline

- строит point-in-time календарную панель без использования будущих значений в признаках;
- рассчитывает causal-индикаторы и future outcomes для разметки;
- формирует семейства targets и оценивает их частоту и теоретический lift;
- перебирает одиночные правила и пары правил с `AND`/`OR`;
- подбирает параметры на train-части walk-forward и оценивает pooled OOS lift;
- требует не менее двух сигналов в неделю;
- фиксирует лучшие правила по discovery-периоду и отдельно тестирует G0/W1 на holdout с 2025 года.

## Структура

```text
notebooks/test.ipynb             # основной воспроизводимый notebook
src/cbr_loader.py                # загрузка и нормализация курсов ЦБ
src/market_data.py               # point-in-time market panel
src/features.py                  # causal-признаки
src/outcomes.py                  # будущие outcomes для разметки
src/targets.py                   # targets и их registry
src/target_evaluation.py         # частота и теоретический lift targets
src/indicators.py                # правила и комбинации AND/OR
src/walk_forward.py              # временные WF-разбиения
src/indicator_optimization.py    # train-оптимизация и pooled OOS-оценка
src/indicator_backtest.py        # независимый backtest фиксированных правил
tests/                           # автоматические проверки
data/raw/cbr/                    # локальный XML-кэш, не коммитится
data/processed/                  # локальные CSV/Parquet, не коммитятся
```

## Проверка

```bash
python -m unittest discover -s tests
```

Перед командной работой создавайте отдельную ветку:

```bash
git switch -c feature/short-description
```

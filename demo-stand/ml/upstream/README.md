# AI Product Hack — Trigger Model

Командный репозиторий триггерной модели для трансграничных переводов.

В репозитории реализован воспроизводимый исследовательский pipeline: загрузка официальных дневных курсов Банка России, causal-признаки, разметка targets, walk-forward оптимизация rule-based индикаторов, независимый backtest фиксированных G0/W1-правил и ML-индикатор на непрерывных составляющих лучших правил. Отдельный stateful signal pipeline день за днём обновляет просроченные rule/ML-артефакты, формирует единый JSON-вектор, пропускает его через сменяемую метамодель и считает итоговый backtest общего потока.

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

`notebooks/test.ipynb` содержит исследование и подбор конфигураций. `notebooks/02_signal_pipeline.ipynb` является самостоятельным production-shaped pipeline: он сам загружает данные, строит признаки, воспроизводит raw OOS-поток движков с 2022 года, обучает логистическую метамодель на 2022–2023, валидирует её на 2024 и считает финальный backtest с 2025 года. Второй notebook не зависит от переменных или файлов, созданных первым. Запускайте Jupyter из корня репозитория, его родительской папки либо из каталога `notebooks`.

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
- выбирает лучшие rule-архитектуры на discovery и продолжает их OOS walk-forward с переоптимизацией thresholds перед каждым test-периодом;
- выбирает частоту переобучения ML на discovery и продолжает OOS walk-forward победителя;
- для rules и ML использует rolling train-окно последних 24 месяцев вместо expanding train.

## Единый сигнал и метамодель

- в коде зафиксированы победившие rule-архитектуры, parameter grids и cadence; конкретные thresholds переобучаются на прошлом перед каждым OOS-периодом;
- HistGradientBoosting переобучается на фиксированном cadence 12 месяцев;
- для каждого rule/ML-движка хранится состояние: fitted artifact, версия, дата обучения, последняя зрелая дата train и следующая дата переобучения;
- `get_signal` отдаёт JSON-совместимый score каждого движка, включая несработавшие и технические статусы;
- `filter_signal` является стабильной границей сменяемой метамодели, а `run_signal_day` — единственной дневной точкой входа для production и исторического replay;
- время `as_of` фиксируется как 09:00 `Europe/Moscow`;
- rule confidence пересчитывается как precision правила только по уже созревшим прошлым targets, ML confidence — precision на прошлом validation-окне;
- исходный ML probability хранится отдельно как `raw_score`;
- простой confidence filter реализован отдельно и может быть заменён другой функцией без изменения upstream-кода и backtest;
- финальный JSON не содержит будущего значения target;
- разные горизонты сохраняются как отдельные события.

## Структура

```text
notebooks/test.ipynb             # основной исследовательский notebook
notebooks/02_signal_pipeline.ipynb # общий поток, meta-model и финальный backtest
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
src/ml_backtest.py               # WF-выбор cadence и backtest ML-индикатора
src/production_config.py         # frozen rules и ML production config
src/production_pipeline.py       # состояния, retraining, get_signal и replay
src/signal_contract.py           # единый evidence contract и adapters
src/meta_model.py                # сменяемая метамодель и JSON event contract
src/signal_backtest.py           # pooled backtest финального потока
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

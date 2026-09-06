# FX Signal Pipeline

Пайплайн для выбора выгодного момента перевода рублей в `AMD`, `KGS`, `KZT`,
`TJS` и `UZS`. Он объединяет интерпретируемые индикаторные правила и ML,
формирует ограниченный поток сигналов и проверяет актуальность сигнала по цене
MOEX перед его использованием.

## Что делает проект

```text
Данные ЦБ РФ и рыночный контекст
              ↓
causal-признаки и targets G0/W1
              ↓
rule-сигналы + ML-модели
              ↓
walk-forward OOS predictions
              ↓
evidence aggregation и selector
              ↓
cooldown + не более 2 сигналов за 7 дней
              ↓
holdout-оценка и проверка актуальности на MOEX
```

- `G0 / GOOD_NOW` — курс выгоден сейчас;
- `W1 / WINDOW_CLOSING` — выгодное окно может скоро закрыться;
- горизонты прогнозирования: `1`, `3`, `5`, `10` и `20` дней.

Rule confidence рассчитывается только по уже созревшим OOS-наблюдениям. Для ML
используется temporally calibrated `predict_proba`. Обучение и валидация идут
строго по времени; параметры selector и policy замораживаются до финального
holdout.

## Структура

```text
notebooks/model_pipeline.ipynb  # основной воспроизводимый ноутбук
src/                            # данные, признаки, модели, selector и policy
tests/                          # автоматические тесты
figures/                        # итоговые графики проекта
data/                           # локальный кэш данных, не коммитится
product/                        # продуктовые исследования и материалы
```

Основная программная точка входа:

```python
from src.pipeline import run_yura_pipeline
```

Архитектуру базового ML и тип selector можно менять независимо:

```python
from src.config import YuraPipelineConfig
from src.selector import build_opportunity_selector

config = YuraPipelineConfig(ml_scope="pooled")
selector = build_opportunity_selector("threshold")
```

Допустимые `ml_scope`: `pooled`, `hybrid`, `per_currency`.

Допустимые selector: `threshold`, `logistic_regression`, `extra_trees`.

## Запуск

Требуется Python 3.10 или новее.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab notebooks/model_pipeline.ipynb
```

При первом запуске данные загружаются и кэшируются в `data/raw` и
`data/processed`.

## Тесты

```bash
python -m pytest -q
```

## Важные ограничения

- курс ЦБ — ориентир, а не гарантированный курс исполнения;
- исторический lift и BPS uplift не гарантируют будущий результат;
- клиентская симуляция оценивает продуктовый эффект, но не участвует в обучении
  и формировании сигналов.

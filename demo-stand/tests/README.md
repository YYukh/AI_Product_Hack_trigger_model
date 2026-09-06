# Тесты стенда + ML-интеграции

Питон-окружение стенда живёт в Docker (локально ставить pandas/pyarrow/sklearn
не нужно). Один образ гоняет всё.

```bash
cd ~/Desktop/product\ hack/dev/demo-stand
docker build -f tests/Dockerfile -t stand-tests .
docker run --rm -v "$PWD:/w" stand-tests
```

Тесты монтируются из репозитория, образ пересобирать при изменении кода не нужно —
только при смене зависимостей.

## Что покрыто

| Файл | Что проверяет |
|---|---|
| `test_ml_common_contract.py` | `ml/common/contract.py`: маппинг ML-сценария → код текста стенда (`ML_MOMENT` на трендовом рынке, `LEVEL_LOW` на локальном минимуме, `REVERSAL_UP` для `WINDOW_CLOSING`, сезонный слот), `facts` по короткому окну перцентиля, форма `to_stand_signal` |
| `test_ml_common_pipeline.py` | `_clean_wide` (запрет NaN, дедуп, сортировка), `rates_long` (только 5 продакшн-коридоров, формат стенда) |
| `test_ml_common_artifacts.py` | том `/mldata`: ready-барьер, `wait_ready` таймаут, roundtrip parquet/joblib |
| `test_stand_rates_source.py` | `api/data_access.rates()`: курсы из парсера при `RATES_URL`, молчаливый откат на `data/rates.csv` при 500, чтение файла без конфигурации |
| `test_stand_ml_endpoints.py` | `/api/health` показывает статусы трёх ML-сервисов; `/api/ml/*` проксируют; 502 при недоступности; `_ml_probe("")` |
| `test_scenarios_real_data.py` | все сценарии из `data/scenarios.json` резолвятся в `expected_state` (file-режим), плашки без дырок; `data/signals.json` — только допустимые коды; `ML_MOMENT` есть в `texts.json` |

ML-сервисы в интеграционных тестах замоканы через `respx` — Docker с профилем
`ml` для прогона тестов поднимать не требуется.

## Тесты кода ML-команды (`ml/upstream`)

Отдельно, в их же наборе:

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml --profile ml \
  run --rm --no-deps --entrypoint "" moment-model \
  python -m unittest discover -s /app/upstream/tests
```

# AI Product Hack — Trigger Model

Командный репозиторий триггерной модели для трансграничных переводов.

Сейчас в репозитории опубликован первый изолированный слой: загрузка официальных дневных курсов Банка России. Признаки, таргеты, индикаторы, ML и policy будут добавляться отдельными проверяемыми изменениями.

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

После запуска откройте `notebooks/01_cbr_data_loading.ipynb` и выполните все ячейки.

## Что делает загрузчик

- загружает XML-историю с официального сайта ЦБ РФ;
- поддерживает `AMD`, `KZT`, `KGS`, `TJS`, `UZS`, `USD`, `EUR`, `CNY`;
- сохраняет исходный номинал и исходное значение курса;
- нормализует значение до рублей за одну единицу валюты;
- возвращает temporal-таблицу и широкий DataFrame `дата × валюта`;
- при необходимости сохраняет исходные XML локально.

Исторический XML не содержит точного времени публикации. Поэтому `available_at` и `publication_timestamp` консервативно установлены на `00:00` следующего календарного дня, а поле `publication_timestamp_is_proxy=True` явно маркирует это допущение.

## Структура

```text
notebooks/01_cbr_data_loading.ipynb  # воспроизводимый пример загрузки
src/cbr_loader.py                    # библиотечный код
tests/test_cbr_loader.py             # тест XML-парсинга и нормализации
data/raw/cbr/                        # локальный XML-кэш, не коммитится
data/processed/                      # локальные CSV/Parquet, не коммитятся
```

## Проверка

```bash
python -m unittest discover -s tests
```

Перед командной работой создавайте отдельную ветку:

```bash
git switch -c feature/short-description
```

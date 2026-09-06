# Демо-стенд: подсказка выгодного момента для трансграничного перевода

Веб-стенд, который проигрывает клиентский путь фичи как в мобильном приложении:
пуш о курсе → экран перевода → в том числе ситуация «момент уже изменился» →
перевод доведён до конца. Стенд показывает **поведение продукта**, а не метрики
отклика — конверсию, объём и каннибализацию меряет пилот.

---

## Стек

| Слой | Технологии | Роль |
|---|---|---|
| **proxy** | Caddy 2.8 (alpine) | единая точка входа `:80` / `:443`, TLS Let's Encrypt по домену, `/api/*` → `api`, всё остальное → `web` |
| **web** | React 18 · TypeScript 5.5 · Vite 5 · Tailwind 3.4 · lucide-react. Билд → статика, раздаётся nginx 1.27 | клиентский макет (рамка телефона) + лаунчер сценариев. Состояние прогона в query-строке |
| **api** | Python 3.11 · FastAPI 0.111 · Uvicorn · pandas 2.2 · httpx · pydantic 2 | вся логика стенда: расчёт состояния экрана, тексты, резервы, лог событий, проксирование к ML |
| **parser** *(опц.)* | Python 3.11 · FastAPI · pandas · `ml/upstream/src/cbr_loader.py` | парсер котировок ЦБ РФ, отдаёт курсы в формате стенда |
| **moment-model** *(опц.)* | + scikit-learn · joblib · `ml/upstream` production-pipeline | ~100 rule+ML движков «выгодного момента», walk-forward с переобучением |
| **push-model** *(опц.)* | + логистическая метамодель + частотная политика | решает, какой сигнал уходит в пуш; реализует контракт стенда |
| **ml-warmup** *(опц.)* | одноразовый job | прогоняет весь ML-конвейер один раз, кэширует артефакты в том `ml_data` |

Хранилища нет: `data/*.csv` и `data/*.json` в репозитории, резервы и события —
в памяти процесса `api` + append в `logs/events.jsonl`. ML-сервисы обмениваются
через docker-том `ml_data` (parquet/joblib).

---

## Структура проекта

```
demo-stand/
├── docker-compose.yml            базовый стек: proxy + web + api
├── docker-compose.ml.yml         оверлей: parser + moment-model + push-model + ml-warmup (профиль ml)
├── Caddyfile                     reverse-proxy :80/:443, здесь же меняется домен для TLS
├── .env.example                  все параметры стенда с дефолтами → cp .env.example .env
│
├── api/                          бэкенд (FastAPI, без БД)
│   ├── main.py                   приложение, ~22 ручки /api/*, /api/health, passthrough /api/ml/*
│   ├── config.py                 параметры из окружения
│   ├── data_access.py            чтение data/*, метаданные коридоров, помощники по ряду курсов
│   ├── signals.py                источник сигналов: файл data/signals.json или HTTP-модель (ML_URL)
│   ├── evaluate.py               ядро: правила состояний экрана (OK/DRIFT/BETTER/NEUTRAL)
│   ├── texts.py                  подстановка facts в шаблоны data/texts.json
│   ├── reserve.py                «дождаться выгодного курса»: машина состояний, ленивая переоценка
│   ├── policy.py                 коммуникационная политика: бюджет, cooldown, тихие часы
│   ├── events.py                 лог событий → logs/events.jsonl
│   ├── selftest.py               прогон инвариантов без HTTP (python -m api.selftest)
│   └── Dockerfile
│
├── web/                          фронтенд (React + Vite + Tailwind)
│   ├── src/
│   │   ├── App.tsx               оркестрация: вкладки «Песочница» / «Сценарии», экраны, вызовы api
│   │   ├── Launcher.tsx          список 7 сценариев
│   │   ├── api.ts store.ts types.ts   REST-клиент, query-состояние + форматтеры, типы
│   │   ├── components/
│   │   │   ├── Chrome.tsx        TopTabs + SandboxBar (параметры песочницы) + ScenarioBar
│   │   │   ├── Shell.tsx         PhoneFrame — рамка телефона со статус-баром
│   │   │   ├── Plaque.tsx        плашка сигнала (4 состояния, одинаковая заливка)
│   │   │   ├── PushToast.tsx     системное уведомление сверху экрана
│   │   │   └── ui.tsx            кнопки, поля, чипы
│   │   └── screens/              Home · Transfer · Settings · Reserve (R1–R5) · RecipientLimit
│   ├── tailwind.config.js        дизайн-токены (цвета, радиусы, тени)
│   ├── nginx.conf  vite.config.ts
│   └── Dockerfile                multi-stage: сборка Vite → раздача nginx
│
├── ml/                          рабочие ML-сервисы (см. ml/README.md)
│   ├── upstream/                 вендорённый снимок кода ML-команды (src/ + tests/, коммит 1898dee, НЕ модифицируется)
│   ├── common/                   тонкие обёртки: pipeline.py · contract.py · artifacts.py
│   ├── parser/ moment_model/ push_model/    FastAPI-сервисы (app.py + Dockerfile)
│   ├── warmup/run_warmup.py      одноразовый прогон конвейера → том ml_data
│   └── fallback/cbr_rates.csv    офлайн-снапшот курсов ЦБ
│
├── data/                         входные данные стенда (без БД)
│   ├── rates.csv                 дневной ряд курсов по 5 коридорам
│   ├── signals.json              предпосчитанный ответ модели (facts + scenario_code)
│   ├── scenarios.json            7 сценариев (S1–S7)
│   ├── personas.json             5 портретов получателей
│   └── texts.json                утверждённые формулировки push / plaque / forbidden
│
├── tests/                        pytest в Docker (см. tests/README.md)
│   ├── Dockerfile  conftest.py
│   └── test_*.py                 ml/common + интеграция стенда + резолв сценариев (32 теста)
│
├── tools/                        офлайн-скрипты
│   ├── gen_data.py               детерминированный синтетический ряд + сигналы
│   ├── prepare_rates.py          реальная выгрузка ЦБ РФ в rates.csv
│   └── simulate_users.py         Монте-Карло прогон профилей → simulation-stats.*
│
├── logs/events.jsonl             append-лог событий стенда (создаётся в рантайме)
└── simulation-stats.md / .json   результаты последнего прогона моделирования
```

---

## Архитектура

```
                         ┌───────────────── docker network ─────────────────┐
                         │                                                  │
  браузер ──:80/:443──►  proxy (Caddy)                                      │
                         │   ├─ /api/*  ──►  api  (FastAPI, :8000)          │
                         │   └─ /*      ──►  web  (nginx, статика React)    │
                         │                    │                             │
                         │                    │  ML_URL / RATES_URL / MOMENT_URL
                         │                    │  (пусто → файлы, HTTP → сервисы,
                         │                    │   при сбое таймаут 2–4 с → откат на файл)
                         │                    ▼                             │
                         │        parser ─┐                                 │
                         │        push-model ├─ читают ─► ml_data (том)     │
                         │        moment-model ┘            ▲                │
                         │                                  │ пишет один раз │
                         │                            ml-warmup             │
                         │                                                  │
   репозиторий ──volume──►  data/*.csv, *.json   (api: ro)                  │
   хост       ──volume──►  logs/events.jsonl     (api: rw)                  │
                         └──────────────────────────────────────────────────┘
```

**Два режима запуска.**

- **Базовый** (`docker compose up`): `proxy + web + api`. Сигналы из
  `data/signals.json`, курсы из `data/rates.csv`. Всё, что нужно для показа
  клиентского пути.
- **Полный** (`+ docker-compose.ml.yml --profile ml`): добавляются
  `parser + moment-model + push-model + ml-warmup`. Оверлей проставляет `api`
  переменные `RATES_URL` / `ML_URL` / `MOMENT_URL`, и бэкенд начинает брать
  курсы и сигналы по HTTP. Любой сбой ML → молчаливый откат на те же файлы,
  факт отката виден в `GET /api/health`.

**Ключевые принципы.**

- **Базы данных нет.** `rates.csv` грузится в pandas один раз при старте `api`,
  JSON-файлы читаются с диска (`lru_cache`). Перезапуск `api` обнуляет
  демо-сессии (резервы, события) — осознанно.
- **Сигнальный слой отделён контрактом.** `api` не знает, что внутри модели:
  только `GET /health` + `GET /signals?as_of=&corridors=` (см. ниже). Файл и
  HTTP-модель взаимозаменяемы.
- **Симулированное время.** Стенд живёт на исторических датах (`sim_date` +
  `sim_minutes` в query-строке). Реальное время — только в метках лога.
- **Стенд не выносит вердикт.** `api` отдаёт состояние, факты, числа и период;
  «выгодно / невыгодно» и глаголы будущего времени не появляются нигде.

### Жизненный цикл запроса (экран перевода)

1. Фронт открывает сессию (песочница или сценарий) → `POST /api/evaluate` с
   `{corridor, sim_date, sim_minutes, entry: PUSH|SELF, push_rate, amount_rub, …}`.
2. `evaluate.py`: берёт курс на дату (`data_access.rate_on`), считает
   `delta_bp` между курсом пуша и текущим, определяет состояние
   `OK / DRIFT / BETTER / NEUTRAL` (порог `DRIFT_THRESHOLD_BP`, TTL пуша
   `PUSH_TTL_MINUTES`).
3. Берёт сигнал дня (`signals.py` → файл или HTTP-модель), достаёт
   `scenario_code` + `facts`.
4. `texts.py` подставляет `facts` в утверждённый шаблон из `data/texts.json`.
   Нет плейсхолдера — текст не рендерится (не рендерится «с дыркой»).
5. Ответ: `{state, plaque, push_text, current_rate, recipient_gets,
   percentile_now, actions, …}`. Вывод «что это значит» делает клиент.

Назначение каждого модуля `api/` и `web/src/` — в разделе
[«Структура проекта»](#структура-проекта) выше. Ключевые помощники по ряду
курсов в `data_access.py`: `rate_on`, `window_values`, `percentile_rank`,
`next_trading_day`. Дизайн-токены фронтенда (цвета, радиусы, тени) —
`web/tailwind.config.js`, сняты со скриншотов веб-приложения; обводок нет
нигде, только заливки и отступы.

---

## Интерфейс

Две вкладки в тёмной области стенда над телефоном.

### Вкладка «Песочница» (по умолчанию)

Свободный режим: пользователь сам крутит параметры сессии и ходит по всем
экранам приложения. В свёрнутой полосе — сводка (`страна · дата · как открыл`)
и кнопки `+1 день` / `↻` (на главный экран). Разворот полосы открывает
параметры:

| Параметр | Что делает |
|---|---|
| **Страна получателя** | коридор (валюта, курс, тексты); переключается и внутри экрана «За рубеж» |
| **Дата курса** | любой торговый день из доступного ряда; `−`/`+` по дням, «последняя» |
| **Как открыл приложение** | `Сам зашёл` (экран открыт вручную) / `Пришёл по пушу` (сверху съезжает уведомление) |
| **Курс в пуше — с даты** | для входа по пушу: курс из уведомления берётся за N торговых дней до выбранной даты — так на экране видно `OK` / `BETTER` / `DRIFT` в зависимости от того, куда ушёл курс |

Ссылка на конкретную сессию: `http://localhost/?corridor=RUB_TJS&date=2026-08-14&entry=PUSH`.

### Вкладка «Сценарии»

Список из 7 подготовленных карточек. Тап запускает сценарный отыгрыш: коридор,
портрет, дата, точка входа и курс пуша заданы жёстко. В полосе стенда —
`‹ К списку`, `↻` (проиграть заново), `+1 день` (при активном резерве).
Прямая ссылка: `http://localhost/?scenario=S2`.

| ID | Название | Вход | Что показывает |
|---|---|---|---|
| **S1** | Момент актуален | пуш | пуш → тап → предзаполненная форма → подтверждение |
| **S2** | Момент изменился | пуш | курс ушёл против клиента → экран честности (обе величины, разница в валюте получателя), без предзаполнения |
| **S3** | Окно закрывается | пуш | факт о развороте вверх после месячного минимума, без «успейте» |
| **S4** | Календарный сигнал в пустую неделю | пуш | низкорисковое сезонное сообщение, когда рынок стоит |
| **S5** | Сигнала нет — контроль | сам зашёл | нейтральный факт (реальный перцентиль курса), обычный день не выдаётся за сигнал |
| **S6** | Перебор коммуникаций | пуш | третий пуш за три дня → в уведомлении ссылка на настройку частоты, а не ещё один сигнал |
| **S7** | Право вето получателя | пуш | перевод исчерпает месячный лимит получателя (опциональный сюжет, `FEATURE_RECIPIENT_LIMIT`) |

Состояния экрана перевода — `OK / DRIFT / BETTER / NEUTRAL`. Плашка сигнала —
один компонент с одинаковой заливкой во всех состояниях (зелёного и красного
нет: цвет читался бы как вердикт). Механика подачи DRIFT зафиксирована — дельта
+ контекст, порог `DRIFT_THRESHOLD_BP`.

---

## Запуск

### Стенд (без ML-контейнеров)

```bash
cp .env.example .env
docker compose up -d --build
```

Открыть `http://localhost/`. Сигналы — из `data/signals.json`, курсы — из
`data/rates.csv`.

### Полный стенд с рабочими ML-сервисами

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.ml.yml --profile ml up -d --build
```

Первый старт: одноразовый `ml-warmup` прогоняет полный production-конвейер
(walk-forward replay движков с 2020 года + обучение метамодели), **~12–15 мин**.
Артефакты кэшируются в docker-томе `ml_data` — повторные запуски мгновенные.
Сузить первый прогон: `ML_REPLAY_FROM=2024-01-01` в `.env`. Подробности —
[`ml/README.md`](ml/README.md).

### Локальная разработка

```bash
# бэкенд
cd api && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && cd ..
uvicorn api.main:app --reload --port 8000

# фронтенд (отдельный терминал)
cd web && npm install && npm run dev      # http://localhost:5173, /api → :8000
```

Быстрый прогон инвариантов без HTTP: `python -m api.selftest`.

---

## Данные

| Файл | Что |
|---|---|
| `data/rates.csv` | дневной ряд курсов, `date,corridor,rate,is_stale`. `rate` — рублей за 1 единицу валюты получателя, номинал нормирован |
| `data/signals.json` | предпосчитанный по истории ответ модели: факты и `scenario_code`, не текст |
| `data/personas.json` | 5 портретов с поведенческими параметрами |
| `data/scenarios.json` | сценарии S1–S7; `expected_state` — это тест, а не логика |
| `data/texts.json` | библиотека формулировок: `push` / `plaque` / `forbidden` / `why_forbidden` |

При запуске с профилем `ml` контейнер `ml-warmup` переписывает `data/rates.csv`
реальными курсами ЦБ и `data/scenarios.json` / `data/signals.json` — на реальные
срабатывания модели. Синтетические оригиналы сохраняются рядом в
`data/*.synthetic.*`.

### Перегенерация вручную

```bash
python3 tools/gen_data.py                                   # детерминированный синтетический ряд
python3 tools/prepare_rates.py --from 2021-01-01 --to 2026-08-31   # реальная выгрузка ЦБ (нужна сеть)
```

---

## Контракт модели пуша

Отдельный контейнер, стенд не знает, что внутри. Два метода:

```
GET /health   → {"status":"ok","model_version":"..."}
GET /signals?as_of=2026-06-15&corridors=RUB_TJS,RUB_UZS
```

```json
{
  "as_of": "2026-06-15",
  "model_version": "...",
  "signals": [{
    "date": "2026-06-15", "corridor": "RUB_TJS",
    "indicator": "level_p10", "direction": "favorable", "speed": "slow",
    "strength": 0.82, "scenario_code": "LEVEL_LOW",
    "facts": {"rate": 0.10883, "percentile": 12, "window_days": 90,
              "streak_days": 4, "change_bp": -145}
  }]
}
```

Модель отдаёт **факты и `scenario_code`**, не текст, не рекомендацию, не
вероятность. `scenario_code` ∈ `MOMENTUM_DOWN, LEVEL_LOW, REVERSAL_UP, SEASONAL,
NEUTRAL, ML_MOMENT`. Ответ на `as_of=T` не зависит от данных после `T`
(проверяется двумя вызовами на пересекающихся датах).

---

## Тесты

```bash
docker build -f tests/Dockerfile -t stand-tests .
docker run --rm -v "$PWD:/w" stand-tests
```

32 теста: `ml/common` (маппинг ML→стенд, подготовка курсов, том артефактов),
интеграция стенда (курсы из парсера + откат на файл, `/api/health` со статусами
ML-сервисов, проксирование `/api/ml/*`), резолв всех сценариев на текущих
`data/*`. ML-сервисы замоканы (`respx`), профиль `ml` не нужен. Тесты кода
ML-команды — `ml/upstream/tests`, см. [`tests/README.md`](tests/README.md).

---

## Деплой на VPS

1. Сервер: 1–2 vCPU, 2 ГБ RAM, Ubuntu 22.04+, Docker Engine + Compose plugin.
2. Docker (если нет): `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER && newgrp docker`
3. Запуск:
   ```bash
   git clone <repo> stand && cd stand
   cp .env.example .env
   docker compose up -d --build            # или с оверлеем -f docker-compose.ml.yml --profile ml
   ```
4. Фаервол: `sudo ufw allow 80 && sudo ufw allow 443`
5. Домен и TLS: в файле `Caddyfile` заменить строку `:80` на `ваш.домен.рф`
   (без `http://`), завести A-запись на IP сервера, `docker compose up -d`.
   Caddy сам получит сертификат Let's Encrypt.
6. Эксплуатация:
   ```bash
   docker compose logs -f api            # логи бэкенда
   cat logs/events.jsonl                 # лог событий стенда
   docker compose pull && docker compose up -d --build   # обновление
   docker compose down                   # остановка
   ```
7. Проверка: `curl -s http://localhost/api/health | jq` — `status: "ok"`,
   `signals_source.active` = `file` или `http`, `dates_available` с диапазоном
   ряда.

---

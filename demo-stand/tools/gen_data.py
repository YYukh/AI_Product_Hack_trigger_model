#!/usr/bin/env python3
"""
gen_data.py — детерминированный генератор данных для оффлайн-демо стенда.

Пишет:
    data/rates.csv      — дневной ряд курсов по 5 коридорам, ~2 года
    data/signals.json    — предпосчитанные сигналы, привязанные к сценариям S1..S4
    data/personas.json    — 5 портретов
    data/scenarios.json  — сценарии S1..S7
    data/texts.json      — библиотека формулировок (+ запрещённые двойники)

Это НЕ выгрузка ЦБ. Реальную выгрузку делает tools/prepare_rates.py.
Здесь ряд синтетический, но детерминированный (seed = hash(date+corridor)),
чтобы даты сценариев всегда попадали в нужные состояния и прогон был
воспроизводим. Все числа, попадающие в интерфейс, имеют источник:
либо этот ряд, либо блок facts сигнала, либо — с пометкой «параметр стенда».

Запуск:  python3 tools/gen_data.py
"""

import csv
import json
import math
import hashlib
import os
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

START = date(2024, 9, 1)
END = date(2026, 8, 31)

CORRIDORS = ["RUB_TJS", "RUB_UZS", "RUB_KGS", "RUB_AMD", "RUB_KZT"]

# База: рублей за одну единицу валюты получателя (после нормировки по номиналу).
# Значения условные, масштаб подобран так, чтобы суммы у получателя выглядели
# правдоподобно для своей валюты. recipient_gets = amount_rub / rate.
BASE = {
    "RUB_TJS": 0.108830,
    "RUB_UZS": 0.000790,
    "RUB_KGS": 0.935000,
    "RUB_AMD": 0.240000,
    "RUB_KZT": 0.185000,
}

# Сезонные пики из аналитики: UZS июль–октябрь, KGS декабрь.
SEASONAL_PEAK_MONTHS = {"RUB_UZS": (7, 10), "RUB_KGS": (12, 12)}


def rng(*parts):
    """Детерминированное число в [0,1) из строкового ключа."""
    key = "|".join(str(p) for p in parts).encode()
    h = hashlib.sha256(key).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def noise(d, corridor, scale):
    return (rng(d.isoformat(), corridor, "n") - 0.5) * 2 * scale


def month_wave(d, corridor):
    """Медленная синусоида ~30 дней. Амплитуда даёт медианный размах внутри
    месяца ≈ 5–6% (аналитика: медиана 5,9%)."""
    phase = rng(corridor, "phase") * 2 * math.pi
    day_idx = (d - START).days
    return 0.029 * math.sin(2 * math.pi * day_idx / 30.4 + phase)


def slow_trend(d, corridor):
    """Плавный тренд по всему ряду, ± несколько процентов."""
    day_idx = (d - START).days
    total = (END - START).days
    amp = 0.02 + 0.03 * rng(corridor, "trend_amp")
    dirn = 1 if rng(corridor, "trend_dir") > 0.5 else -1
    return dirn * amp * math.sin(math.pi * day_idx / total)


def seasonal(d, corridor):
    rng_ = SEASONAL_PEAK_MONTHS.get(corridor)
    if not rng_:
        return 0.0
    lo, hi = rng_
    if lo <= d.month <= hi:
        return 0.012
    return 0.0


def raw_rate(d, corridor):
    b = BASE[corridor]
    factor = 1.0 + month_wave(d, corridor) + slow_trend(d, corridor) \
        + seasonal(d, corridor) + noise(d, corridor, 0.004)
    return b * factor


# --- Точечные правки под сценарии -------------------------------------------
# Только абсолютные значения для одной ключевой пары дат сценария S2:
# состояние DRIFT считает evaluate.py как (current_rate - push_rate),
# где push_rate берётся из scenarios.json, а current_rate — из rates.csv.
# Чтобы дельта была ровно +45 б.п., курс RUB_TJS на дату среза фиксируем.
ABS_OVERRIDES = {
    ("RUB_TJS", date(2026, 6, 15)): 0.109320,   # S2: current_rate
    ("RUB_TJS", date(2026, 6, 14)): 0.108910,
    ("RUB_TJS", date(2026, 6, 13)): 0.108870,
    ("RUB_TJS", date(2026, 6, 12)): 0.108900,
}

# Косметический sculpt: привести 6 торговых дней перед датой сценария к
# монотонному тренду нужного направления (~0,45%/день). На факты сигнала это
# не влияет — факты заданы явно в write_signals() как параметры файла-заглушки.
# Влияет только на форму линии на графике рядом с датой среза.
SCULPT = [
    ("RUB_UZS", date(2026, 3, 12), 6, -0.0045),  # S1: снижение
    ("RUB_UZS", date(2026, 4, 22), 6, +0.0050),  # S3: разворот вверх
    ("RUB_KGS", date(2025, 12, 10), 8, +0.0006),  # S4: почти плоско
]


def rate_for(d, corridor):
    key = (corridor, d)
    if key in ABS_OVERRIDES:
        return ABS_OVERRIDES[key]
    return raw_rate(d, corridor)


def apply_sculpt(get, setv):
    for corridor, as_of, days, step in SCULPT:
        anchor = as_of - timedelta(days=days)
        base = get(anchor, corridor)
        if base is None:
            continue
        for i in range(days + 1):
            dd = anchor + timedelta(days=i)
            val = base * (1 + step * i)
            setv(dd, corridor, round(val, 6))


# --- rates.csv -------------------------------------------------------------
def write_rates():
    # 1. базовый ряд в словарь {(date,corridor): (price, is_stale)}
    grid = {}
    last_by_corr = {}
    d = START
    while d <= END:
        weekend = d.weekday() >= 5
        for c in CORRIDORS:
            if weekend and c in last_by_corr:
                grid[(d, c)] = (last_by_corr[c], True)
            else:
                price = round(rate_for(d, c), 6)
                last_by_corr[c] = price
                grid[(d, c)] = (price, False)
        d += timedelta(days=1)

    # 2. косметический sculpt вокруг дат сценариев
    def _get(dd, cc):
        v = grid.get((dd, cc))
        return v[0] if v else None

    def _set(dd, cc, val):
        if (dd, cc) in grid:
            grid[(dd, cc)] = (val, grid[(dd, cc)][1])

    apply_sculpt(_get, _set)

    # 3. выгрузка
    rows = []
    d = START
    while d <= END:
        for c in CORRIDORS:
            price, stale = grid[(d, c)]
            rows.append((d.isoformat(), c, f"{price:.6f}", "true" if stale else "false"))
        d += timedelta(days=1)
    with open(os.path.join(DATA, "rates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "corridor", "rate", "is_stale"])
        w.writerows(rows)
    print("rates.csv:", len(rows), "строк")
    return rows


def series(rows, corridor):
    return [(r[0], float(r[2])) for r in rows if r[1] == corridor]


def percentile_of(values, x):
    """Доля значений <= x, в процентах (0..100)."""
    if not values:
        return 50
    below = sum(1 for v in values if v <= x)
    return round(100 * below / len(values))


def window(ser, as_of, days):
    lo = (date.fromisoformat(as_of) - timedelta(days=days)).isoformat()
    return [v for (dt, v) in ser if lo <= dt <= as_of]


def streak_down(ser, as_of):
    hist = [v for (dt, v) in ser if dt <= as_of]
    n = 1
    for i in range(len(hist) - 1, 0, -1):
        if hist[i] < hist[i - 1]:
            n += 1
        else:
            break
    return n


def streak_up(ser, as_of):
    hist = [v for (dt, v) in ser if dt <= as_of]
    n = 1
    for i in range(len(hist) - 1, 0, -1):
        if hist[i] > hist[i - 1]:
            n += 1
        else:
            break
    return n


def change_bp(ser, as_of, days):
    w = window(ser, as_of, days)
    if len(w) < 2:
        return 0
    return round((w[-1] - w[0]) / w[0] * 10000)


# --- signals.json -------------------------------------------------------------
def write_signals(rows):
    # Факты заданы явно: это файл-заглушка, играющий роль ответа модели.
    # Значения согласованы с нарративом кейса (§9 аналитики). Поле rate
    # подтягивается из фактического ряда на дату среза — единственное, что
    # обязано совпасть с графиком.
    def rate_on(corridor, as_of):
        return next(v for (dt, v) in series(rows, corridor) if dt == as_of)

    spec = [
        dict(corridor="RUB_UZS", as_of="2026-03-12", indicator="momentum_down",
             direction="favorable", speed="slow", strength=0.78, code="MOMENTUM_DOWN",
             facts=dict(percentile=12, window_days=90, streak_days=4, change_bp=-160)),
        dict(corridor="RUB_TJS", as_of="2026-06-15", indicator="level_p10",
             direction="favorable", speed="slow", strength=0.82, code="LEVEL_LOW",
             facts=dict(percentile=12, window_days=90, streak_days=4, change_bp=-145)),
        dict(corridor="RUB_UZS", as_of="2026-04-22", indicator="reversal_up",
             direction="closing", speed="fast", strength=0.71, code="REVERSAL_UP",
             facts=dict(percentile=9, window_days=90, streak_days=2, change_bp=110)),
        dict(corridor="RUB_KGS", as_of="2025-12-10", indicator="seasonal",
             direction="favorable", speed="slow", strength=0.55, code="SEASONAL",
             facts=dict(percentile=52, window_days=90, streak_days=0, change_bp=12,
                        season="december")),
    ]
    signals = []
    for s in spec:
        facts = dict(s["facts"])
        facts["rate"] = round(rate_on(s["corridor"], s["as_of"]), 6)
        signals.append({
            "date": s["as_of"], "corridor": s["corridor"], "indicator": s["indicator"],
            "direction": s["direction"], "speed": s["speed"], "strength": s["strength"],
            "scenario_code": s["code"], "facts": facts,
        })
    payload = {
        "as_of": "2026-08-31",
        "model_version": "file-stub-v0.1",
        "generated_by": "tools/gen_data.py",
        "signals": signals,
    }
    with open(os.path.join(DATA, "signals.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("signals.json:", len(signals), "сигналов")
    for s in signals:
        print("  ", s["corridor"], s["date"], s["scenario_code"], s["facts"])


# --- personas.json -------------------------------------------------------------
def write_personas():
    personas = [
        {
            "id": "oybek", "name": "Ойбек, 27", "corridor": "RUB_UZS", "city": "Москва",
            "timezone": "Europe/Moscow", "typical_amount_rub": 20000, "recipient_name": "Брат",
            "recipient_phone": "+998 90 123-45-67", "recipient_bank": "Kapitalbank, Узбекистан",
            "open_delay_min": 20, "rate_sensitivity": "high", "recipient_limit": None,
            "note": "Портрет 1. Курс назван главным драйвером выбора сервиса. "
                    "Транзакционный поток, ~4 операции в месяц.",
            "assumption": "Чувствительность к курсу — оценка, параметр стенда.",
        },
        {
            "id": "daler", "name": "Далер, 34", "corridor": "RUB_TJS", "city": "Химки",
            "timezone": "Europe/Moscow", "typical_amount_rub": 20000, "recipient_name": "Жена",
            "recipient_phone": "+992 90 123-45-67", "recipient_bank": "Азизи Молия, Таджикистан",
            "open_delay_min": 360, "rate_sensitivity": "high", "recipient_limit": None,
            "note": "Портрет 2. Сменный график — пуш открывает через несколько часов. "
                    "Непереведённые деньги дома = голодная неделя, холд средств пугает.",
            "assumption": "Задержка открытия зафиксирована для воспроизводимости прогона.",
        },
        {
            "id": "farrukh", "name": "Фаррух, 41", "corridor": "RUB_UZS", "city": "Екатеринбург",
            "timezone": "Asia/Yekaterinburg", "typical_amount_rub": 45000, "recipient_name": "Отец",
            "recipient_phone": "+998 91 765-43-21", "recipient_bank": "Ipoteka Bank, Узбекистан",
            "open_delay_min": 40, "rate_sensitivity": "medium", "recipient_limit": None,
            "note": "Портрет 3. Событийная отправка раз в 1–3 месяца — «большая отправка семье». "
                    "Низкий цифровой опыт: свободная настройка условий — барьер.",
            "assumption": "Частота и сумма событийной отправки — оценка по коридору, параметр стенда.",
        },
        {
            "id": "ainura", "name": "Айнура, 29", "corridor": "RUB_KGS", "city": "Санкт-Петербург",
            "timezone": "Europe/Moscow", "typical_amount_rub": 26600, "recipient_name": "Мама",
            "recipient_phone": "+996 700 12-34-56", "recipient_bank": "Оптима Банк, Кыргызстан",
            "open_delay_min": 30, "rate_sensitivity": "medium",
            "recipient_limit": {"per_operation_kgs": 15000, "per_month_kgs": 30000,
                                "reason": "фото-идентификация получателя"},
            "note": "Портрет 4-адаптация. Средний перевод ≈ 28 500 сомов. У матери фото-идентификация — "
                    "перевод почти исчерпывает месячный лимит получателя.",
            "assumption": "Лимиты получателя — факт из кейса; сумма перевода — оценка по коридору.",
        },
        {
            "id": "arman", "name": "Арман, 38", "corridor": "RUB_AMD", "city": "Москва",
            "timezone": "Europe/Moscow", "typical_amount_rub": 30000, "recipient_name": "Сестра",
            "recipient_phone": "+374 91 12-34-56", "recipient_bank": "Ameriabank, Армения",
            "open_delay_min": 90, "rate_sensitivity": "low", "recipient_limit": None,
            "note": "Портрет 5. Армянский коридор. Поведенческих данных нет — портрет доступен, "
                    "но помечен как гипотеза.",
            "assumption": "Весь профиль поведения — гипотеза, данных по коридору AMD нет.",
        },
    ]
    with open(os.path.join(DATA, "personas.json"), "w") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print("personas.json:", len(personas))


# --- scenarios.json -------------------------------------------------------------
def write_scenarios():
    scenarios = [
        {
            "id": "S1", "title": "Момент актуален", "persona": "oybek",
            "corridor": "RUB_UZS", "as_of_date": "2026-03-12", "push_sent_at": "18:30",
            "open_delay_min": 20, "entry": "PUSH", "push_rate": 0.000770,
            "expected_state": "OK", "scenario_code": "MOMENTUM_DOWN",
            "summary": "Пуш → тап → предзаполненная форма. Один тап от уведомления до подтверждения.",
        },
        {
            "id": "S2", "title": "Момент изменился", "persona": "daler",
            "corridor": "RUB_TJS", "as_of_date": "2026-06-15", "push_sent_at": "09:10",
            "open_delay_min": 360, "entry": "PUSH", "push_rate": 0.108830,
            "expected_state": "DRIFT", "scenario_code": "LEVEL_LOW",
            "summary": "Курс ушёл против клиента на 0,45%. Экран честности вместо предзаполненной формы. "
                       "Три механики подачи — переключаются в панели.",
        },
        {
            "id": "S3", "title": "Окно закрывается", "persona": "farrukh",
            "corridor": "RUB_UZS", "as_of_date": "2026-04-22", "push_sent_at": "12:00",
            "open_delay_min": 40, "entry": "PUSH", "push_rate": 0.000778,
            "expected_state": "OK", "scenario_code": "REVERSAL_UP",
            "summary": "Факт о развороте вверх после месячного минимума. Без «успейте».",
        },
        {
            "id": "S4", "title": "Календарный сигнал в пустую неделю", "persona": "ainura",
            "corridor": "RUB_KGS", "as_of_date": "2025-12-10", "push_sent_at": "11:00",
            "open_delay_min": 30, "entry": "PUSH", "push_rate": 0.975000,
            "expected_state": "OK", "scenario_code": "SEASONAL",
            "summary": "Рынок без движения. Низкорисковое сообщение про сезонность — канал не молчит месяц.",
        },
        {
            "id": "S5", "title": "Сигнала нет — контроль", "persona": "oybek",
            "corridor": "RUB_UZS", "as_of_date": "2026-05-20", "push_sent_at": None,
            "open_delay_min": 0, "entry": "SELF", "push_rate": None,
            "expected_state": "NEUTRAL", "scenario_code": "NEUTRAL",
            "summary": "Клиент сам зашёл в обычный день. Нейтральный факт: курс в середине диапазона. "
                       "Визуальный эквивалент lift ≈ 1,0 — обычный день не выдаётся за сигнал.",
        },
        {
            "id": "S6", "title": "Перебор коммуникаций", "persona": "farrukh",
            "corridor": "RUB_UZS", "as_of_date": "2026-04-22", "push_sent_at": "12:00",
            "open_delay_min": 40, "entry": "PUSH", "push_rate": 0.000778,
            "expected_state": "OK", "scenario_code": "REVERSAL_UP",
            "summary": "В панели поднять частоту до 30/мес. Третий пуш за три дня → экран отписки. "
                       "Почему индикатор с высокой точностью и высокой частотой неприменим.",
            "panel_hint": {"push_budget_month": 30},
        },
        {
            "id": "S7", "title": "Право вето получателя", "persona": "ainura",
            "corridor": "RUB_KGS", "as_of_date": "2025-12-10", "push_sent_at": "11:00",
            "open_delay_min": 30, "entry": "PUSH", "push_rate": 0.975000,
            "expected_state": "OK", "scenario_code": "SEASONAL", "optional": True,
            "amount_rub_override": 26600,
            "summary": "Айнура отправляет ≈ 28 500 сомов, у матери фото-идентификация. "
                       "Экран предупреждает: перевод исчерпает месячный лимит получателя.",
        },
    ]
    with open(os.path.join(DATA, "scenarios.json"), "w") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)
    print("scenarios.json:", len(scenarios))


# --- texts.json -------------------------------------------------------------
def write_texts():
    texts = {
        "MOMENTUM_DOWN": {
            "push": "Курс {currency_gen} снижается {streak_days}-й день подряд",
            "plaque": "Курс {currency_gen} снижается {streak_days}-й день. "
                      "Сейчас он выгоднее, чем в {percentile_inv}% дней за {window_days} дней",
            "forbidden": "Курс скоро вырастет — переводите сейчас",
            "why_forbidden": "утверждение о будущем, которого мы не можем гарантировать",
        },
        "LEVEL_LOW": {
            "push": "Курс {currency_gen} сейчас выгоднее, чем в {percentile_inv}% дней за {window_days} дней",
            "plaque": "Курс {currency_gen} сейчас выгоднее, чем в {percentile_inv}% дней "
                      "за последние {window_days} дней",
            "forbidden": "Лучший курс за три месяца — успейте",
            "why_forbidden": "«успейте» создаёт срочность, которой в самом факте нет",
        },
        "REVERSAL_UP": {
            "push": "Курс {currency_gen} два дня растёт после месячного минимума",
            "plaque": "Курс {currency_gen} {streak_days} дня растёт после месячного минимума. "
                      "За неделю изменение {change_bp} б.п.",
            "forbidden": "Окно закрывается, вы теряете деньги",
            "why_forbidden": "давление и обещание убытка",
        },
        "SEASONAL": {
            "push": "Перед Новым годом переводы обычно растут — многие отправляют заранее",
            "plaque": "Перед Новым годом переводы в этот коридор обычно растут — многие отправляют заранее. "
                      "Курс сегодня в середине диапазона за {window_days} дней",
            "forbidden": "В декабре курс будет хуже",
            "why_forbidden": "прогноз",
        },
        "NEUTRAL": {
            "push": None,
            "plaque": "Сегодня курс {currency_gen} в середине диапазона за последние {window_days} дней",
            "forbidden": None,
            "why_forbidden": None,
        },
        "WEEK_CHANGE": {
            "push": "За неделю рубль укрепился к {currency_dat} на {change_pct}%",
            "plaque": "За неделю рубль изменился к {currency_dat} на {change_pct}%. "
                      "Курс сегодня выгоднее, чем в {percentile_inv}% дней за {window_days} дней",
            "forbidden": "Заработайте на курсе",
            "why_forbidden": "превращает перевод в инвестицию, риск индивидуальной инвестиционной рекомендации",
        },
        "DRIFT_A": {
            "plaque": "В уведомлении было {push_rate} ₽ за {currency_acc}. Сейчас {current_rate} — "
                      "на {delta_pct}% дороже. С {amount_rub} ₽ получатель получит на "
                      "{recipient_delta} {currency_short} меньше",
            "forbidden": "Ничего страшного, курс всё ещё хороший",
            "why_forbidden": "оценочное суждение вместо факта",
        },
        "DRIFT_B": {
            "plaque": "Сейчас курс {currency_gen} выгоднее, чем в {percentile}% дней за {window_days} дней",
            "forbidden": "Не переживайте, сейчас нормальный курс",
            "why_forbidden": "успокаивающая оценка вместо факта и числа",
        },
        "DRIFT_C": {
            "plaque": "В уведомлении было {push_rate} ₽ за {currency_acc}, сейчас {current_rate} — "
                      "на {delta_pct}% дороже: с {amount_rub} ₽ получатель получит на "
                      "{recipient_delta} {currency_short} меньше. "
                      "При этом курс сейчас выгоднее, чем в {percentile}% дней за {window_days} дней",
            "forbidden": "Курс подрос, но скоро отыграет — переводите",
            "why_forbidden": "прогноз плюс побуждение к действию",
        },
        "BETTER": {
            "plaque": "С момента уведомления курс {currency_gen} снизился ещё на {delta_pct}% — "
                      "сейчас он выгоднее, чем был в пуше",
            "forbidden": "Вы в плюсе, срочно переводите пока не поздно",
            "why_forbidden": "срочность и обещание, которого в факте нет",
        },
    }
    with open(os.path.join(DATA, "texts.json"), "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print("texts.json:", len(texts), "кодов")


def main():
    os.makedirs(DATA, exist_ok=True)
    rows = write_rates()
    write_signals(rows)
    write_personas()
    write_scenarios()
    write_texts()
    print("\nГотово. Файлы в", DATA)


if __name__ == "__main__":
    main()

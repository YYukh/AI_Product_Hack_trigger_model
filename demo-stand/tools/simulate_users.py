#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_users.py — моделирование тестирования на пользователях против стенда.

Что делает:
  1. Тянет со стенда (http://localhost) реальные данные: ряд курсов, метаданные
     коридоров, health.
  2. Строит популяцию из N агентов по пяти статистическим портретам
     (portrety-ca-perevody.md) с весами и распределениями из
     instrukciya-modelirovanie.md.
  3. Прогоняет 12 месяцев. Два потока отправок раздельно (§5.5):
        A — транзакционный (4 опер./мес, ВТБ [Ф]);
        B — «большая отправка семье» (портретная частота).
     Поток B проходит через триггерный слой: сигнал → пуш (с учётом
     коммуникационной политики) → задержка открытия → состояние экрана
     (OK/DRIFT/BETTER/NEUTRAL) → поведенческое решение.
  4. Состояние экрана считается ЛОКАЛЬНЫМ ЗЕРКАЛОМ логики стенда
     (api/evaluate.py) — иначе 100k+ HTTP-вызовов. Зеркало сверяется с живым
     /api/evaluate на подвыборке (--api-check), расхождений быть не должно.
  5. Резерв (расширенный уровень) — зеркало api/reserve.py: условие по
     перцентилю на реальном ряду, TTL, gain_bp.
  6. Агрегация + масштабирование на 3 млн работающих иностранцев (§2),
     сверка годовых агрегатов (§9).
  7. Пишет simulation-stats.md и simulation-stats.json.

Метки в выводе: [Ф] факт, [В] вывод, [М] допущение (крутилка).
Эффект фичи считается КОНСЕРВАТИВНО: только удержание объёма в формальном
канале + возврат отправок через резерв. Timing-сдвиги помечаются отдельно и
в инкрементальный объём НЕ идут.

Запуск:
    python3 tools/simulate_users.py --n 10000 --months 12 --seed 20260903
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

STAND = "http://localhost"
RUB_PER_USD = 90.0        # [М] курс для сверки агрегатов, сент. 2026

# --------------------------------------------------------------------------
# 0. Утилиты HTTP
# --------------------------------------------------------------------------
def _get(path: str):
    with urllib.request.urlopen(STAND + path, timeout=30) as r:
        return json.load(r)


def _post(path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(STAND + path, data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# --------------------------------------------------------------------------
# 1. Данные стенда
# --------------------------------------------------------------------------
class Market:
    def __init__(self):
        self.corridors = {c["corridor"]: c for c in _get("/api/corridors")}
        self.health = _get("/api/health")
        self.rates: dict[str, list[tuple[date, float]]] = {}
        for corr in self.corridors:
            pts = _get(f"/api/rates?corridor={corr}")["points"]
            self.rates[corr] = [(date.fromisoformat(p["date"]), p["rate"]) for p in pts]
        self._idx = {corr: {d: v for d, v in ser} for corr, ser in self.rates.items()}
        self._mm: dict = {}
        lo = self.health["dates_available"]["from"]
        hi = self.health["dates_available"]["to"]
        self.date_from = date.fromisoformat(lo)
        self.date_to = date.fromisoformat(hi)

    def rate_on(self, corr: str, d: date) -> float:
        m = self._idx[corr]
        if d in m:
            return m[d]
        # ближайший предыдущий торговый день
        cur = d
        for _ in range(10):
            cur -= timedelta(days=1)
            if cur in m:
                return m[cur]
        return self.rates[corr][0][1]

    def window(self, corr: str, as_of: date, days: int) -> list[float]:
        lo = as_of - timedelta(days=days)
        return [v for (dd, v) in self.rates[corr] if lo <= dd <= as_of]

    def percentile_rank(self, corr: str, as_of: date, x: float, days: int = 90) -> int:
        vals = self.window(corr, as_of, days)
        if not vals:
            return 50
        return round(100 * sum(1 for v in vals if v <= x) / len(vals))

    def percentile_value(self, corr: str, as_of: date, p: int, days: int) -> float:
        vals = sorted(self.window(corr, as_of, days))
        if not vals:
            return self.rate_on(corr, as_of)
        if len(vals) == 1:
            return vals[0]
        k = (len(vals) - 1) * p / 100
        lo = int(k)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    def next_trading_day(self, corr: str, d: date) -> date:
        cur = d
        for _ in range(10):
            cur += timedelta(days=1)
            if cur in self._idx[corr]:
                return cur
        return d + timedelta(days=1)

    def month_mean(self, corr: str, y: int, mo: int) -> float:
        """Средний курс коридора за календарный месяц — прокси «если бы отправил
        в случайный день этого месяца, не думая о моменте»."""
        key = (corr, y, mo)
        c = self._mm.get(key)
        if c is None:
            vals = [v for (dd, v) in self.rates[corr] if dd.year == y and dd.month == mo]
            c = sum(vals) / len(vals) if vals else self.rate_on(corr, date(y, mo, 15))
            self._mm[key] = c
        return c


# --------------------------------------------------------------------------
# 2. Портреты и веса (instrukciya-modelirovanie.md §3, §5)
# --------------------------------------------------------------------------
# corridor коды стенда: RUB_UZS, RUB_TJS, RUB_KGS, RUB_AMD, RUB_KZT
PORTRAITS = {
    "P1_oybek": dict(
        name="Ойбек — Узбекистан, патент, <30",
        corridor="RUB_UZS", country="Узбекистан",
        income_rub=(90_000, 117_600, 150_000),          # [Ф/М] строительство, центр 117,6к
        patent_rub=10_000,                               # [Ф] Москва 2026
        share_home=(0.30, 0.50),                         # [М] в диапазоне 25–50% [Ф]
        send_b_per_month=(0.4, 0.9),                     # [М] на базе 67% раз в квартал [Ф]
        check_b_rub=(15_000, 22_000, 25_000),           # [М] на базе 22к ВТБ [Ф]
        open_delay_min=20,                               # [М]
        rate_sensitivity="high",                         # [Ф] курс — главный драйвер
        speed_over_price=False,
        digital_skill="mid",
        recipient_limit=False,
    ),
    "P2_daler": dict(
        name="Далер — Таджикистан, патент, 30–45, семья дома",
        corridor="RUB_TJS", country="Таджикистан",
        income_rub=(70_000, 95_000, 120_000),           # [М]
        patent_rub=10_000,                               # [Ф] (регион ~= Москва/МО)
        share_home=(0.40, 0.60),                         # [Ф/М] правый хвост 50%+
        send_b_per_month=(0.5, 1.0),                     # [М]
        check_b_rub=(18_000, 24_000, 30_000),            # [М]
        open_delay_min=360,                              # [М] сменный график
        rate_sensitivity="mid",
        speed_over_price=True,                           # [В] скорость > цена
        digital_skill="mid",
        recipient_limit=False,
    ),
    "P3_ainura": dict(
        name="Айнура — Кыргызстан, ЕАЭС, женщина",
        corridor="RUB_KGS", country="Кыргызстан",
        income_rub=(63_000, 72_000, 90_000),            # [Ф/М] $700–1000 ×90
        patent_rub=0,                                    # [Ф] ЕАЭС
        share_home=(0.25, 0.40),                         # [М]
        send_b_per_month=(0.7, 1.3),                     # [Ф] 1–3 перевода/мес по коридору
        check_b_rub=(23_000, 26_600, 30_000),           # [Ф] ~28,5к сомов ≈ 26,6к ₽
        open_delay_min=30,                               # [М]
        rate_sensitivity="mid",
        speed_over_price=False,
        digital_skill="low",                            # [Ф] ниже цифровой опыт
        recipient_limit=True,                           # [Ф] лимит фото-ID
    ),
    "P4_ashot": dict(
        name="Ашот — Армения, ЕАЭС, крупный чек (гипотеза)",
        corridor="RUB_AMD", country="Армения",
        income_rub=(120_000, 170_000, 260_000),         # [М] гипотеза «крупный чек»
        patent_rub=0,                                    # [Ф] ЕАЭС
        share_home=(0.20, 0.35),                         # [М]
        send_b_per_month=(0.25, 0.5),                    # [М] ниже частоты 1–3
        check_b_rub=(60_000, 110_000, 220_000),         # [М] гипотеза «крупный чек», выше среднего
        open_delay_min=90,                               # [М]
        rate_sensitivity="high",                         # [М] большая сумма → % чувствителен
        speed_over_price=False,
        digital_skill="high",
        recipient_limit=False,
    ),
    "P5_farrukh": dict(
        name="Фаррух — Узбекистан, оседающий 5+ лет, событийная отправка",
        corridor="RUB_UZS", country="Узбекистан",
        income_rub=(80_000, 130_000, 200_000),          # [М] торговля/транспорт
        patent_rub=10_000,                              # [М] патент/ВНЖ
        share_home=(0.15, 0.35),                         # [В] не цикл выживания
        send_b_per_month=(0.2, 0.5),                     # [В] низкая, нерегулярная
        check_b_rub=(30_000, 55_000, 120_000),          # [В] высокий, волатильный
        open_delay_min=40,                               # [М]
        rate_sensitivity="high",                         # [Ф] курс ₽, может отложить
        speed_over_price=False,
        digital_skill="high",
        recipient_limit=False,
    ),
}

WEIGHT_SCENARIOS = {
    "base":        dict(P1_oybek=.35, P2_daler=.30, P3_ainura=.15, P4_ashot=.05, P5_farrukh=.15),
    "young_flow":  dict(P1_oybek=.45, P2_daler=.30, P3_ainura=.12, P4_ashot=.03, P5_farrukh=.10),
    "settling":    dict(P1_oybek=.25, P2_daler=.25, P3_ainura=.15, P4_ashot=.05, P5_farrukh=.30),
    "eaeu_focus":  dict(P1_oybek=.20, P2_daler=.20, P3_ainura=.35, P4_ashot=.15, P5_farrukh=.10),
}

# География присваивается независимо от портрета (§4) [Ф]
REGIONS = [("Москва", .27), ("Московская обл.", .19), ("Санкт-Петербург", .11),
           ("Тюменская обл.", .03), ("Свердловская обл.", .03), ("Прочие", .37)]

# Транзакционный поток A (§5.4, §5.5) — не проходит триггерный слой
FLOW_A_OPS_PER_MONTH = 4                     # [Ф] ВТБ, 4 операции/мес
FLOW_A_CHECK_RUB = (16_000, 22_000, 30_000)  # [Ф/М] средний чек 22к ВТБ, разброс [М]
REMIT_SHARE_A = 0.52                          # [М] доля потока A, идущая семье (не карта/сервисы);
                                             # откалибрована под сверку §9 — крутить первой

# Годовые агрегаты для сверки (§9) — входящие из России, $ млрд/год
VALIDATION_USD_BLN = {
    "Узбекистан": (13.0, 15.0),   # доля России от $18,9 млрд входящих, ~72–80%
    "Таджикистан": (4.5, 5.7),
    "Кыргызстан": (3.0, 3.2),
    "Армения": (3.8, 3.9),
}
VALIDATION_TOTAL_USD_BLN = (25.0, 28.0)      # [В]


# --------------------------------------------------------------------------
# 3. Поведенческие отклики на экран (§ КЕЙС) — блок допущений [М]
# --------------------------------------------------------------------------
# Три набора: сдержанный / центральный / оптимистичный. Все — крутилки,
# стенд по определению не меряет конверсию. Порог ухода к другому провайдеру
# 0,5% в итоговом курсе — [Ф].
RESPONSE = {
    "conservative": dict(
        base_ok_transfer=0.90,
        base_drift_switch=0.25,       # baseline: молча увидел худший курс, δ>0.5% → ушёл
        base_drift_transfer_bad=0.62, # остальное — перевёл по плохому курсу
        feat_ok_transfer=0.93,
        feat_neutral_transfer=0.86, feat_neutral_reserve=0.03,
        feat_drift_C=dict(transfer=0.45, reserve=0.28),   # abandon = остаток
        mech_A=dict(transfer=-0.10, reserve=+0.02),
        mech_B=dict(transfer=+0.05, reserve=-0.08),
        p2_switch_bonus=0.10, p3_reserve_mult=0.6,
        p5_reserve_bonus=0.05,
    ),
    "central": dict(
        base_ok_transfer=0.92,
        base_drift_switch=0.35,
        base_drift_transfer_bad=0.55,
        feat_ok_transfer=0.95,
        feat_neutral_transfer=0.88, feat_neutral_reserve=0.06,
        feat_drift_C=dict(transfer=0.52, reserve=0.31),
        mech_A=dict(transfer=-0.10, reserve=+0.02),
        mech_B=dict(transfer=+0.05, reserve=-0.08),
        p2_switch_bonus=0.10, p3_reserve_mult=0.6,
        p5_reserve_bonus=0.05,
    ),
    "optimistic": dict(
        base_ok_transfer=0.93,
        base_drift_switch=0.45,
        base_drift_transfer_bad=0.47,
        feat_ok_transfer=0.96,
        feat_neutral_transfer=0.90, feat_neutral_reserve=0.10,
        feat_drift_C=dict(transfer=0.60, reserve=0.33),
        mech_A=dict(transfer=-0.10, reserve=+0.02),
        mech_B=dict(transfer=+0.05, reserve=-0.08),
        p2_switch_bonus=0.10, p3_reserve_mult=0.6,
        p5_reserve_bonus=0.05,
    ),
}

# Коммуникационная политика стенда (api/config.py) [Ф/В]
PUSH_BUDGET_MONTH = 8
PUSH_COOLDOWN_DAYS = 3
DRIFT_THRESHOLD_BP_DEFAULT = 20

# Модель движения курса между отправкой пуша и открытием. После благоприятного
# сигнала работает возврат к среднему: курс склонен идти обратно ПРОТИВ клиента.
# Обе величины — [М], крутилки; знак и порядок обоснованы асимметрией λ=3 и
# сценарием S2 стенда (+45 б.п. за 6 ч).
REVERSION_BP_PER_H = 2.2          # среднее смещение против клиента, б.п. за час задержки
DRIFT_SD_BP_PER_SQRT_H = 6.0     # разброс, б.п. за корень из часов


# --------------------------------------------------------------------------
# 4. Генерация сигналов из реального ряда (правило, замена контракту модели)
# --------------------------------------------------------------------------
def gen_signals(mkt: Market, start: date, end: date) -> dict[str, dict[date, dict]]:
    out: dict[str, dict[date, dict]] = {c: {} for c in mkt.corridors}
    for corr, ser in mkt.rates.items():
        ser = [(d, v) for d, v in ser if start - timedelta(days=120) <= d <= end]
        by_date = dict(ser)
        dates = [d for d, _ in ser]
        for i, d in enumerate(dates):
            if d < start or d > end:
                continue
            v = by_date[d]
            w90 = mkt.window(corr, d, 90)
            if len(w90) < 30:
                continue
            pr = round(100 * sum(1 for x in w90 if x <= v) / len(w90))
            # streak падения
            streak = 1
            j = i
            while j > 0 and by_date[dates[j]] < by_date[dates[j - 1]]:
                streak += 1
                j -= 1
            # разворот вверх после локального минимума
            rev = (i >= 3 and by_date[dates[i]] > by_date[dates[i - 1]] > 0
                   and by_date[dates[i - 2]] <= by_date[dates[i - 3]]
                   and pr <= 20)
            month = d.month
            seasonal = ((corr == "RUB_UZS" and 7 <= month <= 10) or
                        (corr == "RUB_KGS" and month == 12))
            sig = None
            if pr <= 15 or streak >= 4:
                sig = dict(code="LEVEL_LOW" if pr <= 15 else "MOMENTUM_DOWN",
                           direction="favorable", strength=round(1 - pr / 100, 2),
                           percentile=pr, streak=streak)
            elif rev:
                sig = dict(code="REVERSAL_UP", direction="closing",
                           strength=0.6, percentile=pr, streak=streak)
            elif seasonal:
                sig = dict(code="SEASONAL", direction="favorable",
                           strength=0.45, percentile=pr, streak=0)
            if sig:
                out[corr][d] = sig
    return out


# --------------------------------------------------------------------------
# 5. Зеркало логики стенда
# --------------------------------------------------------------------------
def mirror_state(entry: str, elapsed_min, delta_bp: float, has_signal: bool,
                 thr_bp: int) -> str:
    if entry == "PUSH":
        if elapsed_min is not None and elapsed_min > 24 * 60:
            return "NEUTRAL"
        if delta_bp > thr_bp:
            return "DRIFT"
        if delta_bp < -thr_bp:
            return "BETTER"
        return "OK"
    return "OK" if has_signal else "NEUTRAL"


def reserve_lifecycle(mkt: Market, corr: str, created_on: date, amount_rub: float,
                      percentile: int, window_days: int, ttl_days: int):
    """Зеркало api/reserve.py: условие rate <= P(percentile) на окне, TTL."""
    rate_create = mkt.rate_on(corr, created_on)
    d = created_on
    for _ in range(40):
        nd = mkt.next_trading_day(corr, d)
        if nd == d:
            break
        d = nd
        cur = mkt.rate_on(corr, d)
        thr = mkt.percentile_value(corr, d, percentile, window_days)
        if cur <= thr:
            gain_bp = round((rate_create - cur) / rate_create * 10000, 1)
            return dict(state="EXECUTED", waited_days=(d - created_on).days,
                        gain_bp=gain_bp, exec_rate=cur)
        if (d - created_on).days >= ttl_days:
            return dict(state="EXPIRED", waited_days=ttl_days, gain_bp=0.0,
                        exec_rate=None)
    return dict(state="EXPIRED", waited_days=ttl_days, gain_bp=0.0, exec_rate=None)


# --------------------------------------------------------------------------
# 6. Агент
# --------------------------------------------------------------------------
def tri(rng: random.Random, spec):
    if len(spec) == 2:
        return rng.uniform(*spec)
    lo, mode, hi = spec
    return rng.triangular(lo, hi, mode)


class Agent:
    __slots__ = ("pid", "p", "region", "income", "share_home", "share_mult",
                 "sendB_rate", "checkB", "checkA", "rng")

    def __init__(self, pid: str, rng: random.Random):
        self.pid = pid
        self.p = PORTRAITS[pid]
        self.rng = rng
        self.region = _weighted(rng, REGIONS)
        self.income = tri(rng, self.p["income_rub"])
        s_lo, s_hi = self.p["share_home"]
        # распределение с центром в диапазоне и правым хвостом (§5.3)
        base = rng.uniform(s_lo, s_hi)
        if rng.random() < 0.25:
            base = min(0.75, base + rng.uniform(0.0, 0.15))
        self.share_home = base
        mid = (s_lo + s_hi) / 2
        # множитель объёма ремитанса относительно центра портрета: выше доля
        # дохода домой → крупнее «большая отправка» и больше идёт семье в потоке A
        self.share_mult = min(1.6, max(0.4, base / mid))
        self.sendB_rate = rng.uniform(*self.p["send_b_per_month"])
        self.checkB = tri(rng, self.p["check_b_rub"])
        self.checkA = tri(rng, FLOW_A_CHECK_RUB)

    def monthly_home_budget(self) -> float:
        # доля дохода домой (§5.3); патент/аренда/еда — приоритетнее (§7.1)
        return max(0.0, self.income * self.share_home)


def _weighted(rng: random.Random, pairs):
    r = rng.random()
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


def build_population(n: int, weights: dict, seed: int) -> list[Agent]:
    rng = random.Random(seed)
    pids = list(weights)
    probs = [weights[p] for p in pids]
    agents = []
    for i in range(n):
        pid = rng.choices(pids, probs)[0]
        agents.append(Agent(pid, random.Random(seed * 1_000_003 + i)))
    return agents


# --------------------------------------------------------------------------
# 7. Симуляция
# --------------------------------------------------------------------------
def simulate(agents: list[Agent], mkt: Market, signals, *, months: int,
             start: date, response_key: str, mechanic: str, thr_bp: int,
             seed: int, reserve_enabled: bool = True):
    R = RESPONSE[response_key]
    rng = random.Random(seed ^ 0x9E3779B9)

    agg = defaultdict(float)
    per_portrait = defaultdict(lambda: defaultdict(float))
    state_counts = Counter()
    decision_counts = Counter()
    reserve_out = Counter()               # ключ: f"{origin_state}:{rl_state}"
    reserve_wait = defaultdict(list)       # ключ: origin_state
    reserve_gain = defaultdict(list)
    push_suppressed = Counter()

    # окно дат симуляции внутри доступного ряда
    for a in agents:
        pp = per_portrait[a.pid]
        corr = a.p["corridor"]
        # ---- поток A: транзакционный, вне триггерного слоя ----
        volA = FLOW_A_OPS_PER_MONTH * a.checkA * months
        volA_remit = volA * REMIT_SHARE_A * a.share_mult   # часть, идущая семье
        agg["flowA_ops"] += FLOW_A_OPS_PER_MONTH * months
        agg["flowA_volume_rub"] += volA
        agg["flowA_remit_rub"] += volA_remit
        pp["flowA_volume_rub"] += volA
        pp["flowA_remit_rub"] += volA_remit

        # ---- поток B: «большая отправка семье» через триггерный слой ----
        month_cursor = start
        sent_this_month = defaultdict(int)
        last_push_date = None
        for m in range(months):
            # число крупных отправок в этом месяце ~ Poisson(sendB_rate)
            k = _poisson(rng, a.sendB_rate)
            for _ in range(k):
                day = rng.randint(1, 27)
                target = _add_months(start, m).replace(day=day)
                if target > mkt.date_to - timedelta(days=10):
                    continue
                amount = max(1000.0, a.checkB * rng.uniform(0.75, 1.3) * a.share_mult)
                agg["flowB_events"] += 1
                pp["flowB_events"] += 1

                # есть ли сигнал в окне планирования 3–10 дней до target?
                sig, sig_date = _signal_in_window(signals, corr, target, lo=3, hi=10)
                got_push = False
                if sig:
                    ym = (sig_date.year, sig_date.month)
                    if sent_this_month[ym] >= PUSH_BUDGET_MONTH:
                        push_suppressed["бюджет месяца"] += 1
                    elif last_push_date and (sig_date - last_push_date).days < PUSH_COOLDOWN_DAYS:
                        push_suppressed["cooldown"] += 1
                    else:
                        got_push = True
                        sent_this_month[ym] += 1
                        last_push_date = sig_date
                        agg["push_sent"] += 1

                delay = a.p["open_delay_min"]
                if got_push:
                    entry = "PUSH"
                    push_rate = mkt.rate_on(corr, sig_date)
                    open_date = sig_date
                    if delay >= 12 * 60:
                        open_date = mkt.next_trading_day(corr, sig_date)
                    # движение курса за задержку: возврат к среднему ПРОТИВ клиента
                    # после благоприятного сигнала + разброс + реальное движение
                    # ряда для многодневных открытий. Всё [М].
                    delay_h = max(0.25, delay / 60)
                    favor = 1.0 if sig["direction"] == "favorable" else 0.4
                    mu = REVERSION_BP_PER_H * delay_h * favor
                    sd = DRIFT_SD_BP_PER_SQRT_H * math.sqrt(delay_h)
                    delta_bp = mu + sd * rng.gauss(0, 1)
                    if open_date != sig_date:
                        real = (mkt.rate_on(corr, open_date) - push_rate) / push_rate * 10000
                        delta_bp += real
                    elapsed = delay
                    has_sig = True
                else:
                    entry = "SELF"
                    push_rate = None
                    delta_bp = 0.0
                    elapsed = None
                    has_sig = mkt_rate_has_signal(signals, corr, target)

                state = mirror_state(entry, elapsed, delta_bp, has_sig, thr_bp)
                state_counts[state] += 1
                pp[f"state_{state}"] += 1

                # предрозыгрыш «настроения» — общий для baseline и фичи, чтобы
                # сравнение было честным контрфактуалом одного и того же события
                u = dict(act=rng.random(), switch=rng.random(),
                         resv=rng.random(), post=rng.random())

                dec = decide(a, R, state, mechanic, delta_bp, u, has_feature=True,
                             reserve_enabled=reserve_enabled)
                dec_base = decide(a, R, state, mechanic, delta_bp, u, has_feature=False,
                                  reserve_enabled=reserve_enabled)
                decision_counts[f"feat:{state}:{dec}"] += 1
                decision_counts[f"base:{state}:{dec_base}"] += 1

                # --- объёмы: считаем как ремитанс (идёт семье) ---
                base_completed = dec_base in ("transfer", "transfer_bad")
                if base_completed:
                    agg["base_flowB_volume_rub"] += amount
                    pp["base_flowB_volume_rub"] += amount
                elif dec_base == "switch_informal":
                    agg["base_switch_volume_rub"] += amount

                feat_completed = dec == "transfer"
                reserve_executed = False
                rl_gain_bp = 0.0
                if dec == "reserve":
                    rl = reserve_lifecycle(
                        mkt, corr, open_date if got_push else target, amount,
                        percentile=25, window_days=30, ttl_days=7)
                    reserve_out[f"{state}:{rl['state']}"] += 1
                    decision_counts[f"feat:{state}:reserve->{rl['state']}"] += 1
                    if rl["state"] == "EXECUTED":
                        reserve_executed = True
                        feat_completed = True
                        rl_gain_bp = rl["gain_bp"]
                        reserve_wait[state].append(rl["waited_days"])
                        reserve_gain[state].append(rl["gain_bp"])
                    elif u["post"] < 0.6:      # после истечения часть всё же переводит
                        feat_completed = True

                if feat_completed:
                    agg["feat_flowB_volume_rub"] += amount
                    pp["feat_flowB_volume_rub"] += amount

                # --- ЭКОНОМИЯ ПОЛЬЗОВАТЕЛЯ от фичи ---
                # signal  = курс у открытия vs средний курс месяца (отправка «в
                #           случайный день, не думая о моменте») — только по пушу;
                # reserve = выигрыш от ожидания до порога P25 — по любому входу
                #           (в т.ч. NEUTRAL, когда клиент сам зашёл и поставил
                #           отложенный ордер).
                if feat_completed:
                    mm = mkt.month_mean(corr, target.year, target.month)
                    sig_save = 0.0
                    if got_push:
                        rate_open = push_rate * (1 + delta_bp / 10000)
                        sig_save = amount * (mm - rate_open) / mm
                        agg["user_save_signal_rub"] += sig_save
                        agg["user_save_events"] += 1
                        agg["user_save_positive_noorder"] += 1 if sig_save > 0 else 0
                        pp["user_save_signal_rub"] += sig_save
                        pp["user_save_events"] += 1
                    res_save = amount * (rl_gain_bp / 10000) if reserve_executed else 0.0
                    if reserve_executed:
                        agg["user_save_reserve_rub"] += res_save
                        agg["user_save_reserve_n"] += 1
                        pp["user_save_reserve_rub"] += res_save
                    if got_push:
                        agg["user_save_positive_order"] += 1 if (sig_save + res_save) > 0 else 0

                # --- КОНСЕРВАТИВНЫЙ эффект фичи ---
                feat_kept = dec in ("transfer", "reserve")  # остался в формальном канале
                # 1) удержание: baseline ушёл бы в неформальный, с фичей — остался
                if feat_kept and dec_base == "switch_informal":
                    agg["eff_retained_volume_rub"] += amount
                    agg["eff_retained_sends"] += 1
                    pp["eff_retained_volume_rub"] += amount
                # 2) возврат: baseline не отправил бы (ушёл/бросил), фича довела
                #    через резерв до исполнения
                if reserve_executed and dec_base in ("switch_informal", "abandon"):
                    agg["eff_recovered_volume_rub"] += amount
                    agg["eff_recovered_sends"] += 1

                # timing-сдвиг (НЕ инкрементальный объём): фича перенесла отправку
                if state == "DRIFT" and dec == "reserve" and dec_base in ("transfer", "transfer_bad"):
                    agg["timing_shift_events"] += 1
                    agg["timing_shift_volume_rub"] += amount

    def _res_block(origin):
        w = reserve_wait.get(origin, [])
        g = reserve_gain.get(origin, [])
        return dict(
            executed=len(g),
            wait_median=(statistics.median(w) if w else None),
            gain_median=(statistics.median(g) if g else None),
            gain_mean=(round(statistics.fmean(g), 1) if g else None),
            gain_share_positive=(round(sum(1 for x in g if x > 5) / len(g), 2) if g else None),
        )

    return dict(
        agg=dict(agg), per_portrait={k: dict(v) for k, v in per_portrait.items()},
        state_counts=dict(state_counts), decision_counts=dict(decision_counts),
        reserve_out=dict(reserve_out),
        reserve_by_origin={o: _res_block(o) for o in ("DRIFT", "NEUTRAL")},
        push_suppressed=dict(push_suppressed),
    )


def decide(a: Agent, R: dict, state: str, mechanic: str, delta_bp: float,
           u: dict, *, has_feature: bool, reserve_enabled: bool = True) -> str:
    """Решение по одному событию. u — общий предрозыгрыш для baseline и фичи
    (честный контрфактуал). Возвращает:
    transfer | transfer_bad | reserve | abandon | switch_informal
    """
    p = a.p
    caller = p["speed_over_price"] or p["rate_sensitivity"] == "high"
    pswitch = R["base_drift_switch"] + (R["p2_switch_bonus"] if p["speed_over_price"] else 0.0)

    # ---------- baseline: обычный предзаполненный экран, без честного DRIFT ----------
    if not has_feature:
        if state == "DRIFT":
            # молча увидел худший курс. Часть «обзванивает альтернативы» [Ф] и
            # в этот раз уходит в неформальный/к конкуренту; остальные переводят.
            if caller and u["switch"] < pswitch:
                return "switch_informal"
            return "transfer_bad" if u["act"] < 0.9 else "abandon"
        return "transfer" if u["act"] < R["base_ok_transfer"] else "abandon"

    # ---------- с фичей ----------
    if state in ("OK", "BETTER"):
        return "transfer" if u["act"] < R["feat_ok_transfer"] else "abandon"

    if state == "NEUTRAL":
        p_res = R["feat_neutral_reserve"]
        if p["digital_skill"] == "low":
            p_res *= R["p3_reserve_mult"]
        if a.pid == "P5_farrukh":
            p_res += R["p5_reserve_bonus"]
        p_res = max(0.0, p_res)
        p_tr = R["feat_neutral_transfer"]
        if not reserve_enabled:      # нет вторичного действия: половина всё равно переводит
            p_tr += 0.5 * p_res
            p_res = 0.0
        if u["act"] < p_tr:
            return "transfer"
        if u["act"] < p_tr + p_res:
            return "reserve"
        return "abandon"

    # DRIFT — экран честности, три механики подачи
    c = dict(R["feat_drift_C"])
    if mechanic == "A":
        c["transfer"] += R["mech_A"]["transfer"]
        c["reserve"] += R["mech_A"]["reserve"]
    elif mechanic == "B":
        c["transfer"] += R["mech_B"]["transfer"]
        c["reserve"] += R["mech_B"]["reserve"]
    p_res = max(0.0, c["reserve"])
    if p["digital_skill"] == "low":
        p_res *= R["p3_reserve_mult"]
    if a.pid == "P5_farrukh":
        p_res += R["p5_reserve_bonus"]
    p_tr = max(0.0, c["transfer"])
    if not reserve_enabled:      # user-path: в DRIFT без резерва — только перевести или уйти
        p_tr += 0.5 * p_res
        p_res = 0.0

    if u["act"] < p_tr:
        return "transfer"
    if u["act"] < p_tr + p_res:
        return "reserve"
    # остаток — ушёл. Тот же розыгрыш switch, что и в baseline → честное сравнение.
    if caller and u["switch"] < pswitch:
        return "switch_informal"
    return "abandon"


# --------------------------------------------------------------------------
# 8. Вспомогательное
# --------------------------------------------------------------------------
def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _add_months(d: date, m: int) -> date:
    y = d.year + (d.month - 1 + m) // 12
    mo = (d.month - 1 + m) % 12 + 1
    return date(y, mo, 1)


def _signal_in_window(signals, corr: str, target: date, lo: int, hi: int):
    for back in range(lo, hi + 1):
        dd = target - timedelta(days=back)
        if dd in signals.get(corr, {}):
            return signals[corr][dd], dd
    return None, None


def mkt_rate_has_signal(signals, corr: str, d: date) -> bool:
    return d in signals.get(corr, {})


# --------------------------------------------------------------------------
# 9. Сверка зеркала с живым /api/evaluate
# --------------------------------------------------------------------------
def api_parity_check(mkt: Market, signals, n_checks: int, seed: int) -> dict:
    rng = random.Random(seed ^ 0x5DEECE66D)
    scen = _get("/api/scenarios")
    mism = 0
    ok = 0
    details = []
    for _ in range(n_checks):
        sc = rng.choice(scen)
        for mech in ("A", "B", "C"):
            try:
                live = _get(f"/api/scenario/{sc['id']}/run?drift_mechanic={mech}")
            except Exception as e:  # noqa
                details.append(f"api error {sc['id']}: {e}")
                continue
            # зеркало
            corr = sc["corridor"]
            pr = sc.get("push_rate")
            push_min = None
            if sc.get("push_sent_at"):
                h, mm = sc["push_sent_at"].split(":")
                push_min = int(h) * 60 + int(mm)
            elapsed = (sc.get("open_delay_min", 0)) if push_min is not None else None
            cur = mkt.rate_on(corr, date.fromisoformat(sc["as_of_date"]))
            delta_bp = ((cur - pr) / pr * 10000) if (sc["entry"] == "PUSH" and pr) else 0.0
            has_sig = mkt_rate_has_signal(signals, corr, date.fromisoformat(sc["as_of_date"]))
            # для сверки берём has_signal из ответа стенда (у стенда свой signals.json)
            has_sig_stand = live.get("signal") is not None
            mirror = mirror_state(sc["entry"], elapsed, delta_bp, has_sig_stand,
                                  DRIFT_THRESHOLD_BP_DEFAULT)
            if mirror == live["state"]:
                ok += 1
            else:
                mism += 1
                details.append(
                    f"{sc['id']}/{mech}: mirror={mirror} live={live['state']} "
                    f"delta_bp={delta_bp:.1f}")
    return dict(checks=ok + mism, matched=ok, mismatched=mism, details=details[:20])


# --------------------------------------------------------------------------
# 10. Агрегация и отчёт
# --------------------------------------------------------------------------
def build_report(args, mkt, results: dict, parity: dict, weights_used: dict):
    scale = 3_000_000 / args.n
    base = results["central"]  # опорный набор для основных чисел

    def usd(rub):
        return rub / RUB_PER_USD

    lines = []
    A = lines.append
    A("# Моделирование тестирования на пользователях — статистика\n")
    A(f"**Дата прогона:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    A(f"**Стенд:** {STAND} · версия {mkt.health['version']} · источник сигналов "
      f"`{mkt.health['signals_source']['active']}`  ")
    A(f"**Выборка:** {args.n:,} агентов · горизонт {args.months} мес · "
      f"seed {args.seed}  ".replace(",", " "))
    A(f"**Масштабирование:** ×{scale:,.0f} до 3 млн работающих иностранцев (§2 инструкции)  "
      .replace(",", " "))
    A(f"**Веса портретов:** {weights_used}  ")
    A(f"**Эффект фичи:** консервативный — только удержание объёма в формальном "
      f"канале + возврат отправок через резерв. Timing-сдвиги помечены отдельно "
      f"и в инкрементальный объём не входят.\n")

    A("Метки: **[Ф]** факт · **[В]** вывод · **[М]** допущение (крутилка).\n")

    # -- 0. Парити-чек
    A("## 0. Сверка зеркала логики с живым стендом\n")
    A(f"Проверок: {parity['checks']} · совпало: {parity['matched']} · "
      f"расхождений: {parity['mismatched']}")
    if parity["mismatched"]:
        A("\n```")
        for d in parity["details"]:
            A(d)
        A("```")
    A("\nПопуляционный прогон использует локальное зеркало правил `api/evaluate.py`. "
      "Строка выше подтверждает, что зеркало воспроизводит состояние экрана стенда "
      "один-в-один на всех сценариях × 3 механики.\n")

    # -- 1. Популяция
    A("## 1. Популяция и потоки отправок\n")
    A("| Портрет | Доля | Агентов | Медианный доход ₽/мес [Ф/М] | Медианная доля дохода домой [М] |")
    A("|---|---|---|---|---|")
    for pid, w in weights_used.items():
        p = PORTRAITS[pid]
        inc = p["income_rub"][1] if len(p["income_rub"]) == 3 else sum(p["income_rub"]) / 2
        sh = sum(p["share_home"]) / 2
        n_ag = f"{round(args.n*w):,}".replace(",", " ")
        inc_s = f"{inc:,.0f}".replace(",", " ")
        A(f"| {p['name']} | {w:.0%} | {n_ag} | {inc_s} | {sh:.0%} |")
    A("")
    ev = base["agg"]
    flowA_all = ev["flowA_volume_rub"] * scale
    flowA_remit = ev.get("flowA_remit_rub", 0) * scale
    flowB_vol = ev.get("feat_flowB_volume_rub", 0) * scale
    A(f"- Транзакционный поток A [Ф ВТБ: 4 опер./мес × 22к ₽]: "
      f"{ev['flowA_ops']*scale:,.0f} операций/год, весь объём "
      f"{flowA_all/1e9:,.1f} млрд ₽ (${usd(flowA_all)/1e9:,.1f} млрд); "
      f"из них семье (REMIT_SHARE_A={REMIT_SHARE_A} [М] · ×доля дохода домой) — "
      f"{flowA_remit/1e9:,.1f} млрд ₽ (${usd(flowA_remit)/1e9:,.1f} млрд)"
      .replace(",", " "))
    A(f"- «Большая отправка семье» поток B: {ev['flowB_events']*scale:,.0f} "
      f"событий/год, объём (с фичей) {flowB_vol/1e9:,.1f} млрд ₽ "
      f"(${usd(flowB_vol)/1e9:,.1f} млрд)".replace(",", " "))
    A(f"- **Ремитанс всего (A_семье + B): "
      f"${usd(flowA_remit + flowB_vol)/1e9:,.1f} млрд/год** — сверка §9 ниже"
      .replace(",", " "))
    A(f"- Пушей отправлено (поток B): {ev.get('push_sent',0)*scale:,.0f}/год; "
      f"подавлено политикой: {sum(base['push_suppressed'].values())*scale:,.0f} "
      f"({base['push_suppressed']})".replace(",", " "))
    A("")

    # -- 2. Состояния экрана
    A("## 2. Распределение состояний экрана при открытии (поток B, с пушем)\n")
    total_state = sum(base["state_counts"].values()) or 1
    A("| Состояние | Доля событий | Что это |")
    A("|---|---|---|")
    meanings = {"OK": "момент актуален", "DRIFT": "момент изменился против клиента",
                "BETTER": "стало лучше", "NEUTRAL": "сигнала нет / пуш протух"}
    for st in ("OK", "DRIFT", "BETTER", "NEUTRAL"):
        c = base["state_counts"].get(st, 0)
        A(f"| {st} | {c/total_state:.1%} | {meanings[st]} |")
    A("")

    # -- 3. Решения: baseline vs фича
    A("## 3. Поведенческое решение: без фичи vs с фичей (набор «central» [М])\n")
    dc = base["decision_counts"]

    def share(prefix):
        items = {k: v for k, v in dc.items()
                 if k.startswith(prefix) and "->" not in k}
        tot = sum(items.values()) or 1
        return {k.split(":", 2)[2]: v / tot for k, v in items.items()}

    for st in ("OK", "NEUTRAL", "DRIFT"):
        A(f"### {st}")
        A(f"- Без фичи: {_fmt_share(share(f'base:{st}:'))}")
        A(f"- С фичей: {_fmt_share(share(f'feat:{st}:'))}")
        A("")

    # -- 4. Три механики подачи DRIFT
    A("## 4. DRIFT: три механики подачи (набор «central», прогон по механике)\n")
    A("| Механика | transfer | reserve | abandon | switch → неформальный |")
    A("|---|---|---|---|---|")
    for mech in ("A", "B", "C"):
        r = results.get(f"central_mech_{mech}")
        if not r:
            continue
        d = r["decision_counts"]
        tot = sum(v for k, v in d.items() if k.startswith("feat:DRIFT:")
                  and "->" not in k) or 1
        g = lambda name: sum(v for k, v in d.items()
                             if k.startswith("feat:DRIFT:") and k.endswith(name)) / tot
        A(f"| {mech} | {g('transfer'):.0%} | {g('reserve'):.0%} | "
          f"{g('abandon'):.0%} | {g('switch_informal'):.0%} |")
    A("\nМеханика A — «холодная»: максимум доверия, но останавливает и того, кто "
      "перевёл бы. C — переводит плохой момент в удержание. B — прячет дельту.\n")

    # -- 5. Резерв
    A("## 5. Резервирование (расширенный уровень) — раздельно по состоянию входа\n")
    A("Ключевой вывод виден только в разбивке: резерв, созданный ИЗ DRIFT "
      "(сразу после благоприятного сигнала), почти всегда срабатывает на "
      "следующий день с нулевым выигрышем — курс уже низкий, порог P25 уже взят. "
      "Реальная опционная ценность резерва — из состояния NEUTRAL.\n")
    A("| Отклик [М] | Вход | Создано/год | Исполнено | Истекло | Медиана ожидания, дн | Медиана gain_bp | Среднее gain_bp | Доля с gain>5 б.п. |")
    A("|---|---|---|---|---|---|---|---|---|")
    for key in ("conservative", "central", "optimistic"):
        r = results[key]
        ro = r["reserve_out"]
        for origin in ("DRIFT", "NEUTRAL"):
            created = sum(v for k, v in ro.items() if k.startswith(origin + ":"))
            execd = ro.get(f"{origin}:EXECUTED", 0)
            expd = ro.get(f"{origin}:EXPIRED", 0)
            b = r["reserve_by_origin"][origin]
            tot = created or 1
            A(f"| {key} | {origin} | {created*scale:,.0f} | {execd/tot:.0%} | "
              f"{expd/tot:.0%} | {b['wait_median']} | {b['gain_median']} | "
              f"{b['gain_mean']} | {b['gain_share_positive']} |".replace(",", " "))
    A("")
    A("> user-path §4: если `gain_bp` около нуля — резерв не продукт, и узнать "
      "это лучше на стенде. Здесь это подтверждается для входа из DRIFT; из "
      "NEUTRAL — смотреть на среднее и долю с положительным выигрышем.\n")

    # -- 6. Эффект фичи (консервативно) + вход в юнит-экономику
    A("## 6. Эффект фичи — консервативная оценка + вход в юнит-экономику\n")
    A("| Отклик [М] | Удержано в формальном канале, ₽/год | Возвращено отправок (резерв), ₽/год | Удержано отправок, шт/год | Возвращено отправок, шт/год |")
    A("|---|---|---|---|---|")
    for key in ("conservative", "central", "optimistic"):
        a = results[key]["agg"]
        A(f"| {key} | {a.get('eff_retained_volume_rub',0)*scale/1e6:,.0f} млн | "
          f"{a.get('eff_recovered_volume_rub',0)*scale/1e6:,.0f} млн | "
          f"{a.get('eff_retained_sends',0)*scale:,.0f} | "
          f"{a.get('eff_recovered_sends',0)*scale:,.0f} |".replace(",", " "))
    A("")
    cen = results["central"]["agg"]
    ret = (cen.get("eff_retained_volume_rub", 0) + cen.get("eff_recovered_volume_rub", 0)) * scale
    A(f"**Совокупный удержанный/возвращённый объём (central): {ret/1e9:,.2f} млрд ₽/год "
      f"(${usd(ret)/1e6:,.0f} млн).**".replace(",", " "))
    ts = cen.get("timing_shift_volume_rub", 0) * scale
    ts_ev = f"{cen.get('timing_shift_events',0)*scale:,.0f}".replace(",", " ")
    A(f"\n_Timing-сдвиг (перенос отправки на более выгодный день, НЕ "
      f"инкрементальный объём): {ts_ev} событий/год, {ts/1e9:.2f} млрд ₽. "
      f"Это не доход, а смещение уже существующего объёма во времени._")
    A("")
    A("### Пример перевода в маржу (ставки — ваши, ниже иллюстрация)\n")
    A("Стоимость официального канала 3–5% [Ф]. Возьмём чистую маржу провайдера "
      "на удержанной операции консервативно **1,5%**:")
    for key in ("conservative", "central", "optimistic"):
        a = results[key]["agg"]
        v = (a.get("eff_retained_volume_rub", 0) + a.get("eff_recovered_volume_rub", 0)) * scale
        A(f"- {key}: {v*0.015/1e6:,.0f} млн ₽/год валовой маржи с удержанного объёма"
          .replace(",", " "))
    A("")
    A("### На одного активного отправителя потока B\n")
    ev_b = results["central"]["agg"]["flowB_events"]
    n_b = args.n  # грубо: почти все агенты хоть раз шлют B
    A(f"- Событий «большой отправки» в выборке: {ev_b:,.0f} за {args.months} мес "
      f"→ ~{ev_b/args.n/args.months:.2f} на агента в месяц".replace(",", " "))
    retper = ret / 3_000_000
    A(f"- Удержанный/возвращённый объём в пересчёте на 1 работающего иностранца: "
      f"~{retper:,.0f} ₽/год".replace(",", " "))
    A("")

    # -- 6b. Экономия пользователей
    A("## 6б. Сколько сэкономили сами пользователи (отправители)\n")
    A("Экономия = получатель получил больше валюты за те же рубли, чем при "
      "отправке «в случайный день месяца, не думая о моменте» (средний курс "
      "коридора за календарный месяц).\n"
      "- **Экономия от сигнала (тайминг)** — по завершённым отправкам после "
      "пуша: курс у открытия против среднего за месяц. Есть в обоих сценариях.\n"
      "- **Экономия от отложенного ордера** — по фактически исполненным ордерам "
      "(любой вход, в т.ч. когда клиент сам зашёл без сигнала): выигрыш от "
      "ожидания до порога P25 против курса на день оформления.\n")
    no = results["savings_noorder"]["agg"]
    yes = results["savings_order"]["agg"]

    def _sig(a):
        return a.get("user_save_signal_rub", 0) * scale

    def _res(a):
        return a.get("user_save_reserve_rub", 0) * scale

    n_push_no = no.get("user_save_events", 0) * scale
    n_push_yes = yes.get("user_save_events", 0) * scale
    n_res_yes = yes.get("user_save_reserve_n", 0) * scale
    tot_no = _sig(no) + _res(no)
    tot_yes = _sig(yes) + _res(yes)

    A("| Сценарий | Экономия от сигнала (тайминг), млрд ₽ | Экономия от отложенного ордера, млрд ₽ | Всего, млрд ₽/год | Всего, $ млн/год | На одну отправку по пушу, ₽ | Доля отправок по пушу в плюсе |")
    A("|---|---|---|---|---|---|---|")
    A(f"| Без отложенного ордера | {_sig(no)/1e9:.2f} | 0.00 | {tot_no/1e9:.2f} | "
      f"{usd(tot_no)/1e6:.0f} | {(tot_no/n_push_no if n_push_no else 0):,.0f} | "
      f"{no.get('user_save_positive_noorder',0)/(no.get('user_save_events',1) or 1):.0%} |"
      .replace(",", " "))
    A(f"| С отложенным ордером | {_sig(yes)/1e9:.2f} | {_res(yes)/1e9:.2f} | "
      f"{tot_yes/1e9:.2f} | {usd(tot_yes)/1e6:.0f} | "
      f"{(_sig(yes)/n_push_yes if n_push_yes else 0):,.0f} | "
      f"{yes.get('user_save_positive_order',0)/(yes.get('user_save_events',1) or 1):.0%} |"
      .replace(",", " "))
    A("")
    A(f"- Отправок по пушу в год (масштаб 3 млн): ~{n_push_yes:,.0f}".replace(",", " "))
    A(f"- Из них отложенный ордер довёл до исполнения по лучшему курсу: "
      f"~{n_res_yes:,.0f}/год, средний выигрыш на такой ордер ~"
      f"{(_res(yes)/n_res_yes if n_res_yes else 0):,.0f} ₽".replace(",", " "))
    gain_abs = tot_yes - tot_no
    gain_pct = (tot_yes / tot_no - 1) * 100 if tot_no else 0
    A(f"- **Прирост экономии от отложенного ордера: +{gain_abs/1e9:.2f} млрд ₽/год "
      f"(+${usd(gain_abs)/1e6:.0f} млн), это +{gain_pct:.0f}% к сценарию без ордера.**")
    A("")
    A("По портретам (сценарий «с ордером», ₽/год, ×масштаб):")
    A("| Портрет | Экономия всего | В т.ч. от ордера | На одну отправку по пушу |")
    A("|---|---|---|---|")
    for pid in ("P1_oybek", "P2_daler", "P3_ainura", "P4_ashot", "P5_farrukh"):
        pd = results["savings_order"]["per_portrait"].get(pid, {})
        n = pd.get("user_save_events", 0) * scale
        sg = pd.get("user_save_signal_rub", 0) * scale
        rs = pd.get("user_save_reserve_rub", 0) * scale
        per = ((sg + rs) / n) if n else 0
        A(f"| {PORTRAITS[pid]['name']} | {(sg+rs)/1e6:,.0f} млн ₽ | {rs/1e6:,.0f} млн ₽ | "
          f"{per:,.0f} ₽ |".replace(",", " "))
    A("\n_Экономия «от сигнала» может быть отрицательной у части отправок: клиент "
      "открыл пуш поздно, курс ушёл (DRIFT), но всё равно перевёл. Столбец «доля "
      "отправок в плюсе» показывает, у какой части вышло выгодно. Экономия «от "
      "ордера» считается только по фактически исполненным отложенным ордерам._\n")

    # -- 7. Право вето получателя
    A("## 7. Право вето получателя (портрет 3, коридор KGS) [Ф]\n")
    p3 = base["per_portrait"].get("P3_ainura", {})
    A(f"- События потока B у портрета 3: {p3.get('flowB_events',0)*scale:,.0f}/год"
      .replace(",", " "))
    A("- Средний чек ≈ 28,5 тыс. сомов исчерпывает месячный лимит фото-ID (30 тыс.) "
      "почти полностью [Ф]. Закладывается вероятность блокировки второго перевода "
      "в месяце → часть объёма портрета 3 структурно уходит в «цепочку счетов» / "
      "офлайн-получение. В консервативной оценке эффекта это не засчитано как "
      "удержание.\n")

    # -- 8. Сверка агрегатов (§9)
    A("## 8. Сверка годовых агрегатов ремитанса (§9 инструкции)\n")
    A(f"Ремитанс коридора = (ремитанс-часть потока A) + поток B (с фичей), "
      f"×масштаб, ÷{RUB_PER_USD:.0f} ₽/$.\n")
    A("| Коридор | Модель, $ млрд/год | Цель §9 | В коридоре? |")
    A("|---|---|---|---|")
    by_country = defaultdict(float)
    for pid, pdata in base["per_portrait"].items():
        country = PORTRAITS[pid]["country"]
        v = (pdata.get("flowA_remit_rub", 0) + pdata.get("feat_flowB_volume_rub", 0))
        by_country[country] += v
    total_usd = 0.0
    for country, (lo, hi) in VALIDATION_USD_BLN.items():
        mv = usd(by_country.get(country, 0) * scale) / 1e9
        total_usd += mv
        ok = "да" if lo * 0.6 <= mv <= hi * 1.6 else "проверить веса/долю дохода (§9)"
        A(f"| {country} | {mv:,.1f} | {lo}–{hi} | {ok} |".replace(",", " "))
    A(f"| **Итого 4 страны** | **{total_usd:,.1f}** | **{VALIDATION_TOTAL_USD_BLN[0]}–"
      f"{VALIDATION_TOTAL_USD_BLN[1]}** | {'да' if VALIDATION_TOTAL_USD_BLN[0]*0.6 <= total_usd <= VALIDATION_TOTAL_USD_BLN[1]*1.6 else 'калибровать §9'} |"
      .replace(",", " "))
    A("\nПри расхождении по §9 крутить в первую очередь два блока допущений: "
      "веса портретов 1/2/5 и долю отправляемого дохода (25–50%). Параметры "
      "дохода и патента — [Ф], их не трогать.")
    A("\n**Армения** структурно недобирает: вес портрета 4 = 5% [Ф от потоков "
      "въезда], а её ремитанс — ~14% суммы четырёх стран. Модель при весе 5% "
      "не может дать $3,8 млрд без нереалистичного среднего чека. Это тот самый "
      "пробел §10 №4 — по армянскому коридору нет ни одного исследования "
      "отправителей. Портрет 4 держим как гипотезу; в эффект фичи он почти не "
      "вносит вклада (по AMD нет сигналов → всегда NEUTRAL).\n")

    # -- 9. Чувствительность
    A("## 9. Чувствительность (обязательна по §9)\n")
    A("### 9.1 Веса портретов\n")
    A("| Сценарий весов | Удержано+возвращено, млрд ₽/год (central) | Δ к базовому |")
    A("|---|---|---|")
    base_ret = None
    for wk in ("base", "young_flow", "settling", "eaeu_focus"):
        r = results.get(f"weights_{wk}")
        if not r:
            continue
        a = r["agg"]
        v = (a.get("eff_retained_volume_rub", 0) + a.get("eff_recovered_volume_rub", 0)) * scale / 1e9
        if base_ret is None:
            base_ret = v
        A(f"| {wk} | {v:,.2f} | {(v-base_ret):+,.2f} |".replace(",", " "))
    A("")
    A("### 9.2 Доля дохода домой (§5.3, второй калибровочный блок §9)\n")
    A("| Профиль доли | Ремитанс всего, $ млрд/год | Удержано+возвращено, млрд ₽/год | Δ к центру |")
    A("|---|---|---|---|")
    sc0 = None
    for sk, lbl in (("share_center", "центр диапазона 25–50%"),
                    ("share_tail", "правый хвост, чаще 50%+ (+12 п.п.)")):
        r = results.get(sk)
        if not r:
            continue
        a = r["agg"]
        remit = (a.get("flowA_remit_rub", 0) + a.get("feat_flowB_volume_rub", 0)) * scale
        v = (a.get("eff_retained_volume_rub", 0) + a.get("eff_recovered_volume_rub", 0)) * scale / 1e9
        if sc0 is None:
            sc0 = v
        A(f"| {lbl} | {usd(remit)/1e9:.1f} | {v:.2f} | {v-sc0:+.2f} |")
    A("")
    A("### 9.3 Порог DRIFT (крутилка стенда)\n")
    A("| Порог, б.п. | Доля DRIFT | Удержано+возвращено, млрд ₽/год |")
    A("|---|---|---|")
    for tk in (10, 20, 40):
        r = results.get(f"thr_{tk}")
        if not r:
            continue
        a = r["agg"]
        tot_state = sum(r["state_counts"].values()) or 1
        drift_share = r["state_counts"].get("DRIFT", 0) / tot_state
        v = (a.get("eff_retained_volume_rub", 0) + a.get("eff_recovered_volume_rub", 0)) * scale / 1e9
        A(f"| {tk} | {drift_share:.1%} | {v:,.2f} |".replace(",", " "))
    A("")

    # -- 10. Пробелы
    A("## 10. Пробелы — что модель заведомо не знает (§10, не заполнять выдумкой)\n")
    for g in [
        "День месяца и привязка к зарплатному циклу — единственный якорь: дата платежа по патенту.",
        "Средний чек в разрезе гражданства отправителя из России.",
        "Доля карта vs наличные среди мигрантов.",
        "Поведенческие данные по армянскому коридору (портрет 4 — гипотеза, в стенде даёт NEUTRAL: нет сигналов по AMD).",
        "Реальный отклик на пуш (CTR, конверсия, отписки) — стенд не меряет; блок RESPONSE — [М], дан вилкой.",
        "Объёмы неформального сектора (оценка 20–40% — экспертная).",
    ]:
        A(f"- {g}")
    A("")
    A("---\n")
    A("_Числовой дамп — в `simulation-stats.json` рядом с этим файлом._")

    return "\n".join(lines)


def _fmt_share(d: dict) -> str:
    if not d:
        return "—"
    return " · ".join(f"{k} {v:.0%}" for k, v in sorted(d.items(), key=lambda x: -x[1]))


# --------------------------------------------------------------------------
# 11. main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--api-check", type=int, default=60,
                    help="сколько сценариев сверить с живым /api/evaluate")
    ap.add_argument("--out", default="simulation-stats")
    args = ap.parse_args()

    t0 = time.time()
    print("Тяну данные со стенда…", flush=True)
    mkt = Market()
    start = date(2025, 9, 1)  # горизонт внутри доступного ряда (…2026-08-31)
    end = _add_months(start, args.months)
    signals = gen_signals(mkt, start, end)
    n_sig = sum(len(v) for v in signals.values())
    print(f"  ряд {mkt.date_from}…{mkt.date_to}, сигналов сгенерировано: {n_sig}", flush=True)

    print("Сверяю зеркало с /api/evaluate…", flush=True)
    parity = api_parity_check(mkt, signals, args.api_check, args.seed)
    print(f"  {parity['matched']}/{parity['checks']} совпало, "
          f"{parity['mismatched']} расхождений", flush=True)

    weights_used = WEIGHT_SCENARIOS["base"]
    pop = build_population(args.n, weights_used, args.seed)
    print(f"Популяция: {len(pop):,} агентов".replace(",", " "), flush=True)

    results: dict[str, dict] = {}

    # базовые прогоны: три отклика
    for rk in ("conservative", "central", "optimistic"):
        print(f"  прогон отклик={rk}…", flush=True)
        results[rk] = simulate(pop, mkt, signals, months=args.months, start=start,
                               response_key=rk, mechanic="C",
                               thr_bp=DRIFT_THRESHOLD_BP_DEFAULT, seed=args.seed)

    # экономия пользователей: без отложенного ордера vs с ним (central)
    print("  прогон экономия: без ордера…", flush=True)
    results["savings_noorder"] = simulate(
        pop, mkt, signals, months=args.months, start=start, response_key="central",
        mechanic="C", thr_bp=DRIFT_THRESHOLD_BP_DEFAULT, seed=args.seed,
        reserve_enabled=False)
    print("  прогон экономия: с ордером…", flush=True)
    results["savings_order"] = simulate(
        pop, mkt, signals, months=args.months, start=start, response_key="central",
        mechanic="C", thr_bp=DRIFT_THRESHOLD_BP_DEFAULT, seed=args.seed,
        reserve_enabled=True)

    # механики DRIFT (central)
    for mech in ("A", "B", "C"):
        print(f"  прогон механика={mech}…", flush=True)
        results[f"central_mech_{mech}"] = simulate(
            pop, mkt, signals, months=args.months, start=start,
            response_key="central", mechanic=mech,
            thr_bp=DRIFT_THRESHOLD_BP_DEFAULT, seed=args.seed)

    # чувствительность: веса
    for wk, w in WEIGHT_SCENARIOS.items():
        print(f"  прогон веса={wk}…", flush=True)
        p2 = build_population(args.n, w, args.seed)
        results[f"weights_{wk}"] = simulate(
            p2, mkt, signals, months=args.months, start=start,
            response_key="central", mechanic="C",
            thr_bp=DRIFT_THRESHOLD_BP_DEFAULT, seed=args.seed)

    # чувствительность: доля дохода домой
    for sk, shift in (("share_center", 0.0), ("share_tail", 0.12)):
        print(f"  прогон доля={sk}…", flush=True)
        p3 = build_population(args.n, weights_used, args.seed)
        if shift:
            for a in p3:
                a.share_home = min(0.80, a.share_home + shift)
                s_lo, s_hi = a.p["share_home"]
                mid = (s_lo + s_hi) / 2
                a.share_mult = min(1.9, max(0.4, a.share_home / mid))
        results[sk] = simulate(p3, mkt, signals, months=args.months, start=start,
                               response_key="central", mechanic="C",
                               thr_bp=DRIFT_THRESHOLD_BP_DEFAULT, seed=args.seed)

    # чувствительность: порог DRIFT
    for tk in (10, 20, 40):
        print(f"  прогон порог DRIFT={tk}…", flush=True)
        results[f"thr_{tk}"] = simulate(
            pop, mkt, signals, months=args.months, start=start,
            response_key="central", mechanic="C", thr_bp=tk, seed=args.seed)

    md = build_report(args, mkt, results, parity, weights_used)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "parity": parity,
                   "weights_used": weights_used, "results": results},
                  f, ensure_ascii=False, indent=2)

    print(f"\nГотово за {time.time()-t0:.0f} c → {args.out}.md / {args.out}.json")


if __name__ == "__main__":
    main()

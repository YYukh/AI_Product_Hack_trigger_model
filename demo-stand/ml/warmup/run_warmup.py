#!/usr/bin/env python3
"""ml-warmup — одноразовый прогрев ML-конвейера.

Повторяет production-цепочку ML-команды (ml/upstream/src):
    курсы ЦБ → panel/features/targets → replay движков (walk-forward с
    переобучением) → метамодель «какой сигнал в пуш» → частотная политика.

Все тяжёлые артефакты кладёт в /mldata, чтобы три сервиса стартовали мгновенно.
Также переписывает data/rates.csv (реальные курсы) и data/scenarios.json
(S1–S4 на реальные срабатывания). Синтетику сохраняет в *.synthetic.*.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/app")            # common
sys.path.insert(0, "/app/upstream")  # src

from common import artifacts as A          # noqa: E402
from common import contract as C           # noqa: E402
from common import pipeline as P           # noqa: E402

DATA_DIR = Path(os.getenv("STAND_DATA_DIR", "/data"))
REPLAY_FROM = os.getenv("ML_REPLAY_FROM", "2020-01-01")   # с какой даты движки скорят
# С какой даты грузим историю ЦБ для контекста обучения (train-окно 24 мес).
# Всегда раньше REPLAY_FROM, чтобы движки успели обучиться.
DATA_FROM = os.getenv("ML_DATA_FROM", "2020-01-01")
CBR_PREFER = os.getenv("CBR_PREFER", "auto")   # auto | live | fallback
FORCE = os.getenv("ML_WARMUP_FORCE", "false").lower() == "true"


def log(msg: str) -> None:
    print(f"[warmup {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    if A.read_ready() and not FORCE:
        log("ready.json уже есть — прогрев пропущен (ML_WARMUP_FORCE=true чтобы пересчитать)")
        return 0

    # ---------- 1. курсы ЦБ ----------
    data_from = min(DATA_FROM, REPLAY_FROM)
    log(f"загрузка курсов ЦБ ({CBR_PREFER}) c {data_from} (скоринг движков с {REPLAY_FROM})…")
    rates, cbr_source = P.load_wide_rates(
        start=data_from, prefer=CBR_PREFER, raw_dir=str(A.p("raw")),
    )
    log(f"курсы: источник={cbr_source}, {rates.index.min().date()}…{rates.index.max().date()}, "
        f"{len(rates)} дней, валюты={list(rates.columns)}")
    A.save_parquet(rates.reset_index(), "rates_wide.parquet")
    _write_stand_rates(rates)

    # ---------- 2. features / targets ----------
    log("panel → features → outcomes → targets…")
    scoring, target_registry, features, _panel = P.build_scoring_data(rates)
    log(f"scoring_data: {len(scoring)} строк, "
        f"{pd.to_datetime(scoring['available_at']).min().date()}…"
        f"{pd.to_datetime(scoring['available_at']).max().date()}")
    A.save_parquet(scoring, "scoring_data.parquet")
    A.save_parquet(target_registry, "target_registry.parquet")

    feature_cols = [c for c in features.columns
                    if c not in ("available_at", "currency", "source",
                                 "source_available_at", "is_update_day")]
    A.p("feature_columns.json").write_text(json.dumps(feature_cols, ensure_ascii=False))

    # ---------- 3. replay движков «выгодного момента» ----------
    from src.production_config import (
        FIXED_INDICATORS, INDICATOR_SPACES, ML_CONFIG, ML_FEATURE_NAMES,
        ML_MIN_SIGNALS_PER_WEEK, ML_MODEL_TYPE, ML_POOLING_MODE,
        ML_RETRAIN_MONTHS, ML_VALIDATION_MONTHS, PRODUCTION_CURRENCIES,
        PRODUCTION_HORIZONS, PRODUCTION_TARGET_FAMILIES,
        RULE_CONFIDENCE_WINDOW_MONTHS, RULE_MIN_SIGNALS_PER_WEEK,
        TRAIN_WINDOW_MONTHS,
    )
    from src.production_pipeline import (
        engine_state_registry, initialize_engine_states, replay_engine_signals,
        save_engine_states,
    )

    log(f"инициализация движков (replay c {REPLAY_FROM})…")
    states = initialize_engine_states(
        rule_configurations=FIXED_INDICATORS,
        target_registry=target_registry,
        currencies=PRODUCTION_CURRENCIES,
        target_families=PRODUCTION_TARGET_FAMILIES,
        first_score_date=REPLAY_FROM,
        train_months=TRAIN_WINDOW_MONTHS,
        ml_feature_names=ML_FEATURE_NAMES,
        ml_model_type=ML_MODEL_TYPE,
        ml_retrain_months=ML_RETRAIN_MONTHS,
        ml_pooling_mode=ML_POOLING_MODE,
    )
    log(f"{len(states)} движков; walk-forward replay (это самый долгий шаг)…")
    replay = replay_engine_signals(
        scoring,
        states=states,
        first_score_date=REPLAY_FROM,
        indicator_spaces=INDICATOR_SPACES,
        rule_min_signals_per_week=RULE_MIN_SIGNALS_PER_WEEK,
        ml_validation_months=ML_VALIDATION_MONTHS,
        ml_min_signals_per_week=ML_MIN_SIGNALS_PER_WEEK,
        ml_model_type=ML_MODEL_TYPE,
        ml_model_config=ML_CONFIG,
        ml_pooling_mode=ML_POOLING_MODE,
    )
    raw = replay.raw_signals.copy()
    raw["available_at"] = pd.to_datetime(raw["available_at"])
    log(f"replay готов: {len(raw)} строк сигналов, "
        f"{int(raw['signal'].astype(bool).sum())} срабатываний")
    save_engine_states(replay.states, str(A.p("engine_states.joblib")))
    A.save_parquet(raw, "raw_signals.parquet")
    registry = engine_state_registry(replay.states)
    A.save_parquet(registry, "engine_registry.parquet")

    # ---------- 4. метамодель «какой сигнал в пуш» ----------
    meta_kind = "confidence_filter"
    model_version = f"engines_{registry['model_version'].dropna().nunique()}v_{cbr_source}"
    push_events = pd.DataFrame()
    try:
        from src.meta_model import (
            build_meta_candidates, fit_logistic_meta_model, logistic_meta_model,
        )
        from src.production_pipeline import apply_meta_model
        from src.signal_backtest import build_evaluation_universe
        from src.signal_contract import calibrate_rule_confidence_from_oos
        from src.signal_policy import SignalPolicyConfig, apply_signal_policy

        log("калибровка confidence и обучение метамодели…")
        universe = build_evaluation_universe(
            scoring, target_registry=target_registry,
            target_families=PRODUCTION_TARGET_FAMILIES,
            currencies=PRODUCTION_CURRENCIES, start_date=REPLAY_FROM,
        )
        raw_cal = calibrate_rule_confidence_from_oos(
            raw, evaluation_universe=universe,
            window_months=RULE_CONFIDENCE_WINDOW_MONTHS,
        )
        A.save_parquet(raw_cal, "raw_signals_calibrated.parquet")

        meta_keys = ["available_at", "currency", "scenario", "target_family",
                     "target", "horizon"]
        cand = build_meta_candidates(raw_cal)
        training = cand.merge(
            universe.loc[:, meta_keys + ["target_value"]],
            on=meta_keys, how="left", validate="many_to_one",
        ).dropna(subset=["target_value"])
        log(f"обучающих примеров метамодели: {len(training)}")

        last_date = pd.to_datetime(scoring["available_at"]).max()
        meta_train_end = (last_date - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        meta_model_fn, meta_config = None, None
        try:
            fitted = fit_logistic_meta_model(
                training, train_end=meta_train_end, validation_months=12,
                min_signals_per_week=1.0, max_signals_per_week=2.0,
            )
            A.save_joblib(fitted, "meta_model.joblib")
            meta_kind = "logistic_regression"
            meta_model_fn, meta_config = logistic_meta_model, fitted
            model_version = f"logistic_meta_trained_{fitted.trained_at.date()}_{cbr_source}"
            log(f"логистическая метамодель обучена: threshold={fitted.threshold:.3f}, "
                f"val precision={fitted.validation_precision:.3f}, "
                f"lift={fitted.validation_lift:.2f}")
        except Exception as exc:  # noqa: BLE001
            log(f"логистическую метамодель обучить не вышло ({type(exc).__name__}: {exc}); "
                f"использую confidence_filter")

        if meta_model_fn is not None:
            events = apply_meta_model(raw_cal, meta_model=meta_model_fn, meta_config=meta_config)
        else:
            events = apply_meta_model(raw_cal)
        # страховка: если обученная метамодель выродилась в «почти никогда» —
        # откатываемся на детерминированный confidence-filter (путь по умолчанию
        # filter_signal в production_pipeline)
        if len(events) < 20 and meta_kind != "confidence_filter":
            log(f"метамодель дала {len(events)} событий — откат на confidence_filter")
            events = apply_meta_model(raw_cal)
            meta_kind = "confidence_filter_fallback"
            model_version = f"confidence_filter_fallback_{cbr_source}"
        if "evidence_count" not in events.columns and "evidence" in events.columns:
            events["evidence_count"] = events["evidence"].apply(len)
        log(f"событий после метамодели ({meta_kind}): {len(events)}")

        push_events = apply_signal_policy(
            events, SignalPolicyConfig(cooldown_days=3, max_signals_per_7d=2),
        )
        log(f"событий после частотной политики (cooldown 3д, ≤2/7д): {len(push_events)}")

        # Если обученная метамодель отфильтровала весь поток WINDOW_CLOSING —
        # добираем его отдельным deterministic confidence-filter проходом только
        # по W1-сигналам, чтобы сценарий «окно закрывается» жил на реальном
        # событии модели, а не на реконструкции из ряда.
        if "WINDOW_CLOSING" not in set(push_events["scenario"].astype(str)):
            w1_raw = raw_cal[raw_cal["scenario"].astype(str) == "WINDOW_CLOSING"]
            if len(w1_raw):
                w1_ev = apply_meta_model(w1_raw)  # confidence_filter по умолчанию
                if "evidence_count" not in w1_ev.columns and "evidence" in w1_ev.columns:
                    w1_ev["evidence_count"] = w1_ev["evidence"].apply(len)
                w1_push = apply_signal_policy(
                    w1_ev, SignalPolicyConfig(cooldown_days=3, max_signals_per_7d=2))
                if len(w1_push):
                    w1_push = w1_push.assign(meta_model="confidence_filter_w1_supplement")
                    push_events = pd.concat([push_events, w1_push], ignore_index=True)
                    log(f"добор WINDOW_CLOSING confidence-filter: +{len(w1_push)} событий")

        A.save_parquet(push_events, "push_events.parquet")
        try:
            from src.meta_model import market_event_records
            A.p("push_events_records.json").write_text(
                json.dumps(market_event_records(push_events), ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        log("ВНИМАНИЕ: шаг метамодели упал, сервисы будут считать её на лету:")
        traceback.print_exc()

    # ---------- 5. пересборка сценариев + data/signals.json стенда ----------
    try:
        seeds = _rewrite_scenarios(push_events, raw, rates, scoring)
        _write_stand_signals(push_events, scoring, rates, seeds, model_version)
    except Exception:  # noqa: BLE001
        log("ВНИМАНИЕ: не удалось пересобрать data/scenarios.json/signals.json, оставляю синтетические:")
        traceback.print_exc()

    # ---------- 6. ready.json ----------
    A.write_ready({
        "warmup_at": pd.Timestamp.now("UTC").isoformat(),
        "model_version": model_version,
        "replay_from": REPLAY_FROM,
        "data_from": data_from,
        "cbr_source": cbr_source,
        "meta_model": meta_kind,
        "n_engines": int(len(states)),
        "n_raw_signals": int(len(raw)),
        "n_fired": int(raw["signal"].astype(bool).sum()),
        "n_push_events": int(len(push_events)),
        "scoring_from": str(pd.to_datetime(scoring["available_at"]).min().date()),
        "scoring_to": str(pd.to_datetime(scoring["available_at"]).max().date()),
        "elapsed_s": round(time.time() - t0, 1),
    })
    log(f"готово за {time.time() - t0:.0f} c → {A.READY}")
    return 0


# ---------------------------------------------------------------------------
def _write_stand_rates(rates: pd.DataFrame) -> None:
    long = P.rates_long(rates)
    csv_path = DATA_DIR / "rates.csv"
    syn_path = DATA_DIR / "rates.synthetic.csv"
    try:
        if csv_path.exists() and not syn_path.exists():
            syn_path.write_bytes(csv_path.read_bytes())
        long.to_csv(csv_path, index=False)
        log(f"data/rates.csv переписан реальными курсами ЦБ ({len(long)} строк)")
    except OSError as exc:
        log(f"не смог записать data/rates.csv ({exc}) — том ./data не примонтирован rw?")


def _rate_on(rates: pd.DataFrame, corridor: str, d: pd.Timestamp) -> float | None:
    cur = corridor.replace("RUB_", "")
    if cur not in rates.columns:
        return None
    s = rates[cur].loc[rates.index <= d]
    return float(s.iloc[-1]) if len(s) else None


def _reversal_up_date(scoring: pd.DataFrame, corridor: str):
    """Недавняя дата, где курс 2+ дня подряд растёт И вырос за 3 дня в плюс —
    для сценария S3 «окно закрывается», чтобы текст «N дней растёт, +X б.п.»
    не противоречил фактам."""
    cur = corridor.replace("RUB_", "")
    sub = scoring.loc[scoring["currency"] == cur].sort_values("available_at")
    if "consecutive_up" not in sub.columns:
        return None
    cutoff = pd.to_datetime(sub["available_at"]).max() - pd.Timedelta(days=20)
    r3 = sub["return_3d_bps"] if "return_3d_bps" in sub.columns else 1.0
    m = sub.loc[(sub["consecutive_up"] >= 2) & (r3 > 0)
                & (pd.to_datetime(sub["available_at"]) <= cutoff)]
    return pd.Timestamp(m["available_at"].max()) if len(m) else None


def _feature_row(scoring: pd.DataFrame, corridor: str, d: pd.Timestamp):
    cur = corridor.replace("RUB_", "")
    sub = scoring.loc[
        (scoring["currency"] == cur)
        & (pd.to_datetime(scoring["available_at"]) <= d)
    ]
    return sub.iloc[-1] if len(sub) else None


def _rewrite_scenarios(push_events: pd.DataFrame, raw: pd.DataFrame,
                       rates: pd.DataFrame, scoring: pd.DataFrame) -> None:
    """S1–S4 переводим на реальные события, ушедшие в пуш (push_events).
    Персона/заголовок/entry не трогаем. S5–S7 остаются как есть."""
    sc_path = DATA_DIR / "scenarios.json"
    syn_path = DATA_DIR / "scenarios.synthetic.json"
    if not sc_path.exists():
        return
    scenarios = json.loads(sc_path.read_text(encoding="utf-8"))
    if not syn_path.exists():
        syn_path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")

    if push_events is None or not len(push_events):
        log("push_events пуст — сценарии оставлены синтетическими")
        return
    syn = {s["id"]: s for s in json.loads(syn_path.read_text(encoding="utf-8"))}
    pe = push_events.copy()
    pe["available_at"] = pd.to_datetime(pe["available_at"])
    cutoff = rates.index.max() - pd.Timedelta(days=20)

    def pick(corridor: str, want_scenario: str | None) -> pd.Timestamp | None:
        sub = pe.loc[pe["corridor"] == corridor]
        if want_scenario:
            sub = sub.loc[sub["scenario"] == want_scenario]
        sub = sub.loc[sub["available_at"] <= cutoff]
        return sub["available_at"].max() if len(sub) else None

    def engine_name(corridor: str, d: pd.Timestamp, want: str) -> str:
        f = raw.loc[(raw["corridor"] == corridor)
                    & (pd.to_datetime(raw["available_at"]) == d)
                    & raw["signal"].astype(bool)]
        if want:
            f = f.loc[f["scenario"] == want]
        return str(f["engine_name"].iloc[0]) if len(f) else ""

    changed: list[str] = []
    seeds: list[dict] = []
    for sc in scenarios:
        sid, corr = sc.get("id"), sc.get("corridor")
        if sid == "S2":
            # «момент изменился»: реальный GOOD_NOW-пуш, затем курс ушёл вверх на ~15–200 б.п.
            d0 = pick(corr, "GOOD_NOW") or pick(corr, None)
            if d0 is None:
                continue
            r0 = _rate_on(rates, corr, d0)
            cur = corr.replace("RUB_", "")
            after = rates[cur].loc[(rates.index > d0) & (rates.index <= d0 + pd.Timedelta(days=18))]
            d_open = None
            for ts, val in after.items():
                ratio = val / r0
                if 1.0015 < ratio < 1.02:
                    d_open = ts
                    break
            if d_open is None:
                up = after[after > r0 * 1.0015]
                d_open = up.index[0] if len(up) else None
            if d_open is None:
                continue
            sc["as_of_date"] = pd.Timestamp(d_open).date().isoformat()
            sc["push_rate"] = round(float(r0), 6)
            sc["scenario_code"] = "LEVEL_LOW"
            bp = round((rates[cur].loc[d_open] / r0 - 1) * 10000)
            changed.append(f"{sid}:{sc['as_of_date']} +{bp}бп")
            continue
        want = {"S1": "GOOD_NOW", "S3": "WINDOW_CLOSING", "S4": "GOOD_NOW"}.get(sid)
        if want is None:
            continue
        # S3 «окно закрывается» — сценарный, поэтому берём дату с реальным ростом
        # курса 2+ дня подряд (иначе текст «N дней растёт» противоречит фактам).
        # В общий поток push_events реальные WINDOW_CLOSING всё равно добраны выше.
        if sid == "S3":
            d = _reversal_up_date(scoring, corr)
            if d is None:
                changed.append(f"{sid}:оставлен как есть (нет подходящего разворота в ряду)")
                continue
            r = _rate_on(rates, corr, d)
            fr = _feature_row(scoring, corr, d)
            sc["as_of_date"] = pd.Timestamp(d).date().isoformat()
            sc["push_rate"] = round(float(r), 6)
            sc["scenario_code"] = "REVERSAL_UP"
            facts = C.facts_from_feature_row(fr, float(r), window_days=30)
            try:  # для «растёт» показываем именно рост за 3 дня (он > 0 по отбору)
                r3 = float(fr.get("return_3d_bps"))
                if r3 == r3:
                    facts["change_bp"] = int(round(r3))
            except (TypeError, ValueError):
                pass
            seeds.append({
                "date": sc["as_of_date"], "corridor": corr,
                "indicator": "reversal_up", "direction": "closing",
                "speed": "slow", "strength": 0.55, "scenario_code": "REVERSAL_UP",
                "facts": facts,
                "seed": True,
            })
            has_real = "WINDOW_CLOSING" in set(pe.loc[pe["corridor"] == corr, "scenario"].astype(str))
            changed.append(f"{sid}:{sc['as_of_date']} REVERSAL_UP"
                           + ("" if has_real else " (в потоке нет реального WINDOW_CLOSING)"))
            continue
        d = pick(corr, want) or pick(corr, None)
        if d is None:
            continue
        r = _rate_on(rates, corr, d)
        if not r:
            continue
        sc["as_of_date"] = pd.Timestamp(d).date().isoformat()
        sc["push_rate"] = round(float(r), 6)
        sc["scenario_code"] = "REVERSAL_UP" if want == "WINDOW_CLOSING" else \
            C.scenario_code(want, engine_name(corr, d, want), month=d.month,
                            feature_row=_feature_row(scoring, corr, d))
        changed.append(f"{sid}:{sc['as_of_date']} {sc['scenario_code']}")

    # S6 (перебор коммуникаций) и S7 (вето получателя) наследуют реальную дату
    # и push_rate от родственного сценария, чтобы не остаться на синтетической шкале
    by_id = {s["id"]: s for s in scenarios}
    for child, parent in (("S6", "S3"), ("S7", "S4")):
        c, p = by_id.get(child), by_id.get(parent)
        if c and p and c.get("entry") == "PUSH":
            for k in ("as_of_date", "push_rate", "scenario_code"):
                if k in p:
                    c[k] = p[k]
            changed.append(f"{child}:← {parent}")

    sc_path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    A.p("scenarios.ml.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=2))
    A.p("signal_seeds.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=2))
    log(f"data/scenarios.json пересобран: {', '.join(changed) if changed else 'нет подходящих push-событий, оставлено как есть'}"
        + (f"; сид-сигналов: {len(seeds)}" if seeds else ""))
    return seeds


def _write_stand_signals(push_events, scoring, rates, seeds, model_version) -> None:
    """data/signals.json из реальных push-событий (+ сиды) — чтобы file-режим
    стенда (без ML-профиля) тоже работал на реальных сигналах."""
    sig_path = DATA_DIR / "signals.json"
    syn = DATA_DIR / "signals.synthetic.json"
    if sig_path.exists() and not syn.exists():
        syn.write_bytes(sig_path.read_bytes())
    out = list(seeds or [])
    if push_events is not None and len(push_events):
        pe = push_events.copy()
        pe["available_at"] = pd.to_datetime(pe["available_at"])
        recent = pe.loc[pe["available_at"] >= rates.index.max() - pd.Timedelta(days=180)]
        for row in recent.itertuples(index=False):
            d = pd.Timestamp(row.available_at)
            corr = row.corridor
            ev_raw = getattr(row, "evidence", None)
            evidence = list(ev_raw) if ev_raw is not None else []
            eng = ""
            for e in evidence:
                if isinstance(e, dict):
                    eng = e.get("engine_name") or e.get("engine_id") or eng
                    if e.get("engine_type") == "ml":
                        break
            out.append(C.to_stand_signal(
                date=d.date().isoformat(), corridor=corr, scenario=str(row.scenario),
                engine_name=eng, horizon=int(row.horizon), strength=float(row.confidence),
                feature_row=_feature_row(scoring, corr, d),
                rate=_rate_on(rates, corr, d),
            ))
    out.sort(key=lambda s: s["date"])
    payload = {"as_of": str(rates.index.max().date()), "model_version": model_version,
               "generated_by": "ml-warmup", "signals": out}
    sig_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"data/signals.json пересобран из реальных сигналов: {len(out)} записей")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)

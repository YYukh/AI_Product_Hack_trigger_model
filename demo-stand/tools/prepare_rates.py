#!/usr/bin/env python3
"""
prepare_rates.py — выгрузка дневного ряда ЦБ РФ в data/rates.csv.

Тянет XML_dynamic.asp по каждой валюте за период, нормирует по номиналу
НА ЭТАПЕ ПОДГОТОВКИ (не в рантайме) и раскрывает выходные переносом
предыдущего рабочего дня с флагом is_stale=true.

ОБЯЗАТЕЛЬНАЯ ПРОВЕРКА: diff() по полю <Nominal>. Номинал ЦБ исторически
менялся (например, UZS: 10000 → иное). Без нормировки в ряду появится
скачок в десять раз. Скрипт печатает все точки смены номинала и, если
нормировка не убрала разрыв > 20% день-к-дню, завершается с ошибкой.

Зависимости: только стандартная библиотека.
Запуск:  python3 tools/prepare_rates.py --from 2021-01-01 --to 2026-08-31
Если сеть недоступна — используйте оффлайн-генератор tools/gen_data.py.
"""
import argparse
import csv
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# Внутренние коды валют ЦБ РФ (VAL_NM_RQ) для XML_dynamic.asp.
CBR_CODES = {
    "RUB_TJS": "R01815",  # сомони
    "RUB_UZS": "R01717",  # сум
    "RUB_KGS": "R01370",  # сом
    "RUB_AMD": "R01060",  # драм
    "RUB_KZT": "R01335",  # тенге
}
URL = ("https://www.cbr.ru/scripts/XML_dynamic.asp"
       "?date_req1={d1}&date_req2={d2}&VAL_NM_RQ={code}")


def fetch(code: str, d1: date, d2: date) -> list[tuple[date, float, int]]:
    url = URL.format(d1=d1.strftime("%d/%m/%Y"), d2=d2.strftime("%d/%m/%Y"), code=code)
    req = urllib.request.Request(url, headers={"User-Agent": "demo-stand/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    root = ET.fromstring(raw)
    out = []
    for rec in root.findall("Record"):
        d = datetime.strptime(rec.attrib["Date"], "%d.%m.%Y").date()
        nominal = int(rec.findtext("Nominal", "1").replace(" ", ""))
        value = float(rec.findtext("Value").replace(",", ".").replace(" ", ""))
        out.append((d, value, nominal))
    out.sort(key=lambda x: x[0])
    return out


def check_nominal(corridor: str, series: list[tuple[date, float, int]]) -> None:
    changes = []
    prev_nom = None
    for d, _v, nom in series:
        if prev_nom is not None and nom != prev_nom:
            changes.append((d, prev_nom, nom))
        prev_nom = nom
    if changes:
        print(f"  [{corridor}] смена номинала:")
        for d, a, b in changes:
            print(f"    {d}: {a} -> {b}")
    # после нормировки разрывов быть не должно
    norm = [(d, v / nom) for d, v, nom in series]
    for (d0, r0), (d1, r1) in zip(norm, norm[1:]):
        if r0 and abs(r1 - r0) / r0 > 0.20:
            print(f"  [{corridor}] ВНИМАНИЕ: разрыв {r0:.6f} -> {r1:.6f} "
                  f"на {d1} даже после нормировки", file=sys.stderr)
            sys.exit(2)


def expand_calendar(norm: list[tuple[date, float]], d1: date, d2: date):
    by_date = dict(norm)
    rows = []
    cur = d1
    last = None
    while cur <= d2:
        if cur in by_date:
            last = by_date[cur]
            rows.append((cur, last, False))
        elif last is not None:
            rows.append((cur, last, True))
        cur += timedelta(days=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d1", default="2021-01-01")
    ap.add_argument("--to", dest="d2", default=date.today().isoformat())
    ap.add_argument("--corridors", default=",".join(CBR_CODES))
    args = ap.parse_args()
    d1, d2 = date.fromisoformat(args.d1), date.fromisoformat(args.d2)
    corridors = [c.strip() for c in args.corridors.split(",")]

    os.makedirs(DATA, exist_ok=True)
    all_rows = []
    for corridor in corridors:
        code = CBR_CODES[corridor]
        print(f"{corridor} ({code}) {d1}..{d2}")
        series = fetch(code, d1, d2)
        if not series:
            print(f"  пусто, пропуск", file=sys.stderr)
            continue
        check_nominal(corridor, series)
        norm = [(d, round(v / nom, 6)) for d, v, nom in series]
        for d, rate, stale in expand_calendar(norm, d1, d2):
            all_rows.append((d.isoformat(), corridor, f"{rate:.6f}",
                             "true" if stale else "false"))

    path = os.path.join(DATA, "rates.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "corridor", "rate", "is_stale"])
        w.writerows(sorted(all_rows, key=lambda r: (r[1], r[0])))
    print(f"\n{path}: {len(all_rows)} строк")


if __name__ == "__main__":
    main()

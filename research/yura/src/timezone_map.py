"""Russian-subject choropleths for client time-zone simulation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests


RUSSIA_SUBJECTS_GEOJSON_URL = (
    "https://raw.githubusercontent.com/antibioticbook/"
    "russian-geo-data/master/geo.json"
)

# The GeoJSON uses GADM NAME_1 identifiers. A subject is assigned to the time
# zone of its administrative centre. This is necessarily an approximation for
# geographically wide subjects (most notably Sakha/Yakutia).
SUBJECT_TIMEZONE: dict[str, str | None] = {
    "Adygey": "MSK",
    "Altay": "KRAT",
    "Amur": "YAKT",
    "Arkhangel'sk": "MSK",
    "Astrakhan'": "SAMT",
    "Bashkortostan": "YEKT",
    "Belgorod": "MSK",
    "Bryansk": "MSK",
    "Buryat": "IRKT",
    "Chechnya": "MSK",
    "Chelyabinsk": "YEKT",
    "Chukot": "PETT",
    "Chuvash": "MSK",
    "City of St. Petersburg": "MSK",
    "Crimea": "MSK",
    "Dagestan": "MSK",
    "Gorno-Altay": "KRAT",
    "Ingush": "MSK",
    "Irkutsk": "IRKT",
    "Ivanovo": "MSK",
    "Kabardin-Balkar": "MSK",
    "Kaliningrad": None,
    "Kalmyk": "MSK",
    "Kaluga": "MSK",
    "Kamchatka": "PETT",
    "Karachay-Cherkess": "MSK",
    "Karelia": "MSK",
    "Kemerovo": "KRAT",
    "Khabarovsk": "VLAT",
    "Khakass": "KRAT",
    "Khanty-Mansiy": "YEKT",
    "Kirov": "MSK",
    "Komi": "MSK",
    "Kostroma": "MSK",
    "Krasnodar": "MSK",
    "Krasnoyarsk": "KRAT",
    "Kurgan": "YEKT",
    "Kursk": "MSK",
    "Leningrad": "MSK",
    "Lipetsk": "MSK",
    "Maga Buryatdan": "MAGT",
    "Mariy-El": "MSK",
    "Mordovia": "MSK",
    "Moscow City": "MSK",
    "Moskva": "MSK",
    "Murmansk": "MSK",
    "Nenets": "MSK",
    "Nizhegorod": "MSK",
    "North Ossetia": "MSK",
    "Novgorod": "MSK",
    "Novosibirsk": "KRAT",
    "Omsk": "OMST",
    "Orel": "MSK",
    "Orenburg": "YEKT",
    "Penza": "MSK",
    "Perm'": "YEKT",
    "Primor'ye": "VLAT",
    "Pskov": "MSK",
    "Rostov": "MSK",
    "Ryazan'": "MSK",
    "Sakha": "YAKT",
    "Sakhalin": "MAGT",
    "Samara": "SAMT",
    "Saratov": "SAMT",
    "Sevastopol'": "MSK",
    "Smolensk": "MSK",
    "Stavropol'": "MSK",
    "Sverdlovsk": "YEKT",
    "Tambov": "MSK",
    "Tatarstan": "MSK",
    "Tomsk": "KRAT",
    "Tula": "MSK",
    "Tuva": "KRAT",
    "Tver'": "MSK",
    "Tyumen'": "YEKT",
    "Udmurt": "SAMT",
    "Ul'yanovsk": "SAMT",
    "Vladimir": "MSK",
    "Volgograd": "MSK",
    "Vologda": "MSK",
    "Voronezh": "MSK",
    "Yamal-Nenets": "YEKT",
    "Yaroslavl'": "MSK",
    "Yevrey": "VLAT",
    "Zabaikalskiy Krai": "YAKT",
}


def load_russian_subject_geojson(
    cache_path: str | Path | None = None,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Load the subject-level GeoJSON once and optionally cache it locally."""
    path = Path(cache_path).expanduser() if cache_path is not None else None
    if path is not None and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        response = requests.get(RUSSIA_SUBJECTS_GEOJSON_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
    features = payload.get("features", [])
    if payload.get("type") != "FeatureCollection" or not features:
        raise ValueError("GeoJSON субъектов РФ имеет неожиданный формат")
    names = {feature.get("properties", {}).get("NAME_1") for feature in features}
    missing = set(SUBJECT_TIMEZONE).difference(names)
    if missing:
        raise ValueError(
            "В GeoJSON отсутствуют субъекты из таблицы часовых зон: "
            f"{sorted(missing)}"
        )
    return payload


def build_subject_lift_table(
    simulation_summary: pd.DataFrame,
    *,
    currency: str,
    geojson: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Join one currency's time-zone lift to all Russian subjects."""
    required = {"timezone", "currency", "lift"}
    missing = required.difference(simulation_summary.columns)
    if missing:
        raise ValueError(f"В simulation_summary отсутствуют поля: {sorted(missing)}")
    selected = simulation_summary.loc[
        simulation_summary["currency"].astype(str).eq(str(currency)),
        ["timezone", "lift"],
    ].copy()
    if selected.empty:
        known = sorted(simulation_summary["currency"].astype(str).unique())
        raise ValueError(f"Нет валюты {currency!r}; доступны: {known}")
    if selected["timezone"].duplicated().any():
        raise ValueError("На одну валюту должна приходиться одна строка на зону")
    lift_by_zone = selected.set_index("timezone")["lift"]

    russian_names: dict[str, str] = {}
    if geojson is not None:
        russian_names = {
            feature["properties"]["NAME_1"]: (
                feature["properties"].get("NL_NAME_1")
                or feature["properties"]["NAME_1"]
            )
            for feature in geojson.get("features", [])
        }
    rows = []
    for subject_id, timezone in SUBJECT_TIMEZONE.items():
        rows.append({
            "subject_id": subject_id,
            "subject": russian_names.get(subject_id, subject_id),
            "timezone": timezone,
            "currency": str(currency),
            "lift": (
                float(lift_by_zone.get(timezone, np.nan))
                if timezone is not None else np.nan
            ),
        })
    return pd.DataFrame(rows)


def build_subject_savings_table(
    simulation_summary: pd.DataFrame,
    *,
    currency: str = "ALL",
    geojson: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Join mean realized client saving (%) to every Russian subject.

    ``currency='ALL'`` pools currencies using the number of expected executed
    client transactions. BPS are converted to percent with 100 BPS = 1%.
    """
    required = {
        "timezone", "currency", "mean_realized_benefit_bps",
        "expected_client_transactions",
    }
    missing = required.difference(simulation_summary.columns)
    if missing:
        raise ValueError(f"В simulation_summary отсутствуют поля: {sorted(missing)}")
    selected = simulation_summary.copy()
    if str(currency).upper() != "ALL":
        selected = selected.loc[
            selected["currency"].astype(str).eq(str(currency))
        ].copy()
        if selected.empty:
            known = sorted(simulation_summary["currency"].astype(str).unique())
            raise ValueError(f"Нет валюты {currency!r}; доступны: {known}")

    rows: list[dict] = []
    for timezone, sample in selected.groupby("timezone", sort=False):
        weights = pd.to_numeric(
            sample["expected_client_transactions"], errors="coerce"
        ).fillna(0.0)
        values = pd.to_numeric(
            sample["mean_realized_benefit_bps"], errors="coerce"
        )
        valid = values.notna() & weights.gt(0)
        mean_bps = (
            float(np.average(values.loc[valid], weights=weights.loc[valid]))
            if valid.any() else np.nan
        )
        rows.append({
            "timezone": timezone,
            "mean_client_savings_pct": (
                mean_bps / 100.0 if pd.notna(mean_bps) else np.nan
            ),
        })
    savings_by_zone = pd.DataFrame(rows).set_index("timezone")[
        "mean_client_savings_pct"
    ]

    russian_names: dict[str, str] = {}
    if geojson is not None:
        russian_names = {
            feature["properties"]["NAME_1"]: (
                feature["properties"].get("NL_NAME_1")
                or feature["properties"]["NAME_1"]
            )
            for feature in geojson.get("features", [])
        }
    return pd.DataFrame([
        {
            "subject_id": subject_id,
            "subject": russian_names.get(subject_id, subject_id),
            "timezone": timezone,
            "currency": str(currency).upper(),
            "mean_client_savings_pct": (
                float(savings_by_zone.get(timezone, np.nan))
                if timezone is not None else np.nan
            ),
        }
        for subject_id, timezone in SUBJECT_TIMEZONE.items()
    ])


def _focus_on_russia(figure: go.Figure) -> None:
    """Set a fixed Russia-centred viewport instead of a world overview."""
    figure.update_geos(
        visible=False,
        bgcolor="#ffffff",
        projection_type="natural earth",
        projection_rotation_lon=100,
        projection_scale=2.15,
        center_lat=63,
        center_lon=100,
        lataxis_range=[40, 82],
    )


def plot_russian_timezone_lift_map(
    simulation_summary: pd.DataFrame,
    *,
    currency: str,
    geojson: dict[str, Any] | None = None,
    cache_path: str | Path | None = None,
) -> go.Figure:
    """Plot subject-level lift: 1.0 is grey, larger lift becomes red."""
    payload = geojson or load_russian_subject_geojson(cache_path)
    table = build_subject_lift_table(
        simulation_summary, currency=currency, geojson=payload
    )
    finite = table["lift"].replace([np.inf, -np.inf], np.nan).dropna()
    zmax = max(1.05, float(finite.max())) if not finite.empty else 1.05

    # Draw every subject first so excluded/missing zones remain visible.
    figure = go.Figure(go.Choropleth(
        geojson=payload,
        locations=table["subject_id"],
        featureidkey="properties.NAME_1",
        z=np.zeros(len(table)),
        colorscale=[[0.0, "#eeeeee"], [1.0, "#eeeeee"]],
        showscale=False,
        marker_line_color="#ffffff",
        marker_line_width=0.55,
        customdata=np.column_stack([
            table["subject"], table["timezone"].fillna("исключена")
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Зона: %{customdata[1]}<br>Lift: нет данных<extra></extra>"
        ),
        name="Нет данных",
    ))

    available = table.loc[table["lift"].notna()].copy()
    # Values below one stay grey as well; the exact value remains in hover.
    available["color_lift"] = available["lift"].clip(lower=1.0)
    figure.add_trace(go.Choropleth(
        geojson=payload,
        locations=available["subject_id"],
        featureidkey="properties.NAME_1",
        z=available["color_lift"],
        zmin=1.0,
        zmax=zmax,
        colorscale=[
            [0.00, "#bdbdbd"],
            [0.20, "#f2c4c4"],
            [0.55, "#df6767"],
            [1.00, "#b30000"],
        ],
        colorbar={"title": "Lift", "thickness": 16},
        marker_line_color="#ffffff",
        marker_line_width=0.55,
        customdata=np.column_stack([
            available["subject"], available["timezone"], available["lift"]
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Зона: %{customdata[1]}<br>"
            "Lift: %{customdata[2]:.2f}<extra></extra>"
        ),
        name="Lift",
    ))
    _focus_on_russia(figure)
    figure.update_layout(
        title={
            "text": f"Lift клиентских сигналов по субъектам РФ — {currency}",
            "x": 0.5,
        },
        height=720,
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        paper_bgcolor="#ffffff",
        font={"color": "#111111"},
    )
    return figure


def plot_russian_timezone_savings_map(
    simulation_summary: pd.DataFrame,
    *,
    currency: str = "ALL",
    geojson: dict[str, Any] | None = None,
    cache_path: str | Path | None = None,
) -> go.Figure:
    """Plot mean realized client saving per accepted transfer in percent."""
    payload = geojson or load_russian_subject_geojson(cache_path)
    table = build_subject_savings_table(
        simulation_summary, currency=currency, geojson=payload
    )
    finite = table["mean_client_savings_pct"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    zmax = max(0.01, float(finite.clip(lower=0).max())) if not finite.empty else 0.01

    figure = go.Figure(go.Choropleth(
        geojson=payload,
        locations=table["subject_id"],
        featureidkey="properties.NAME_1",
        z=np.zeros(len(table)),
        colorscale=[[0.0, "#eeeeee"], [1.0, "#eeeeee"]],
        showscale=False,
        marker_line_color="#ffffff",
        marker_line_width=0.55,
        customdata=np.column_stack([
            table["subject"], table["timezone"].fillna("исключена")
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Зона: %{customdata[1]}<br>Средняя выгода: нет данных<extra></extra>"
        ),
        name="Нет данных",
    ))
    available = table.loc[table["mean_client_savings_pct"].notna()].copy()
    available["color_savings_pct"] = available[
        "mean_client_savings_pct"
    ].clip(lower=0.0)
    figure.add_trace(go.Choropleth(
        geojson=payload,
        locations=available["subject_id"],
        featureidkey="properties.NAME_1",
        z=available["color_savings_pct"],
        zmin=0.0,
        zmax=zmax,
        colorscale=[
            [0.00, "#bdbdbd"],
            [0.20, "#f2c4c4"],
            [0.55, "#df6767"],
            [1.00, "#b30000"],
        ],
        colorbar={"title": "Выгода, %", "thickness": 16},
        marker_line_color="#ffffff",
        marker_line_width=0.55,
        customdata=np.column_stack([
            available["subject"], available["timezone"],
            available["mean_client_savings_pct"],
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Зона: %{customdata[1]}<br>"
            "Средняя выгода: %{customdata[2]:.3f}%<extra></extra>"
        ),
        name="Средняя выгода",
    ))
    _focus_on_russia(figure)
    currency_title = "все валюты" if str(currency).upper() == "ALL" else currency
    figure.update_layout(
        title={
            "text": (
                "Средняя выгода клиента по субъектам РФ — "
                f"{currency_title}, сценарии 09/12/15/18 МСК"
            ),
            "x": 0.5,
        },
        height=720,
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        paper_bgcolor="#ffffff",
        font={"color": "#111111"},
    )
    return figure

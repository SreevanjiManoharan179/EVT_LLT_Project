"""Presentation layer shared by all pages: theme, provenance colour system, chart/readout/table
helpers, cached data access, the active-twin state and the experiment log.

Design rule: colour encodes PROVENANCE, nothing else.  Measured (graphite), Simulated (cobalt),
Estimated (teal), Predicted (amber), Reference (slate) are used identically in every chart, tag
and legend, so a viewer can always tell where a number came from.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from . import data_access as da
from . import paths
from .ecm import ECMParams
from .thermal import ThermalParams

# ----------------------------------------------------------------------------- colour system
PROV = {
    "Measured": "#1E2A38",
    "Simulated": "#2456C9",
    "Estimated": "#0E8A7D",
    "Predicted": "#C46A12",
    "Reference": "#7A8794",
}
PROV_MEANING = {
    "Measured": "taken from the datasets",
    "Simulated": "output of the digital twin",
    "Estimated": "SOC estimators",
    "Predicted": "ML models and forecasts",
    "Reference": "derived reference values",
}
SERIES = ["#2456C9", "#0E8A7D", "#C46A12", "#7B3FA0", "#B3261E", "#4D7C0F", "#1E2A38", "#0369A1",
          "#A16207", "#BE185D"]
INK, MUTED, GRID, PANEL, BG, ALERT = "#18212B", "#5B6876", "#E4E9EE", "#FFFFFF", "#EEF1F4", "#B3261E"
FONT = "Barlow, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
FONT_COND = "'Barlow Semi Condensed', Barlow, 'Segoe UI', sans-serif"

pio.templates["g14"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=13, color=INK),
    paper_bgcolor=PANEL, plot_bgcolor=PANEL, colorway=SERIES,
    xaxis=dict(gridcolor=GRID, zerolinecolor="#CBD3DB", linecolor="#AEB8C2", ticks="outside", tickcolor="#AEB8C2",
               title=dict(font=dict(size=13, color=MUTED)), automargin=True),
    yaxis=dict(gridcolor=GRID, zerolinecolor="#CBD3DB", linecolor="#AEB8C2", ticks="outside", tickcolor="#AEB8C2",
               title=dict(font=dict(size=13, color=MUTED)), automargin=True),
    legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(size=12, color=INK)),
    margin=dict(l=64, r=24, t=36, b=52), hoverlabel=dict(font_family=FONT, bgcolor="#FFFFFF"),
))
pio.templates.default = "g14"

# ----------------------------------------------------------------------------- variable labels
LABELS = {
    "capacity_Ah": ("Discharge capacity", "Ah"), "soh_pct": ("SOH", "%"), "energy_Wh": ("Discharge energy", "Wh"),
    "mean_voltage_loaded_V": ("Mean loaded voltage", "V"), "T_max_C": ("Peak cell temperature", "°C"),
    "T_rise_C": ("Temperature rise on discharge", "K"), "t_to_Tmax_s": ("Time to peak temperature", "s"),
    "t_to_3p8V_s": ("Time to 3.8 V", "s"), "t_to_3p6V_s": ("Time to 3.6 V", "s"), "t_to_3p4V_s": ("Time to 3.4 V", "s"),
    "t_cc_s": ("CC-charge duration", "s"), "t_cv_s": ("CV-charge duration", "s"), "q_cc_Ah": ("Charge in CC phase", "Ah"),
    "Re_ohm": ("EIS Re", "Ω"), "Rct_ohm": ("EIS Rct", "Ω"), "r_dc_first_sample_ohm": ("DC resistance proxy", "Ω"),
    "mean_current_A": ("Mean load current", "A"), "ambient_C": ("Ambient temperature", "°C"),
    "ic_peak_AhV": ("ICA peak height", "Ah/V"), "ic_peak_V": ("ICA peak voltage", "V"), "ic_area_Ah": ("ICA window charge", "Ah"),
    "T_rise_charge_C": ("Temperature rise on charge", "K"), "duration_load_s": ("Discharge duration", "s"),
    "i_pack_A": ("Pack current", "A"), "v_pack_V": ("Pack voltage", "V"), "P_kW": ("Pack power", "kW"),
    "t_max_C": ("Max pack temperature", "°C"), "t_min_C": ("Min pack temperature", "°C"), "soc_bms_pct": ("BMS SOC", "%"),
    "speed_kph": ("Vehicle speed", "km/h"), "t_amb_C": ("Ambient temperature", "°C"),
}


def lab(var: str) -> str:
    name, unit = LABELS.get(var, (var, ""))
    return f"{name} ({unit})" if unit else name


# ----------------------------------------------------------------------------- CSS
CSS = """
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1360px; }
h1 { font-family: 'Barlow Semi Condensed', Barlow, sans-serif !important; font-weight: 600 !important;
     letter-spacing: -0.01em; font-size: 2.2rem !important; margin-bottom: 0 !important; }
h2, h3 { font-family: 'Barlow Semi Condensed', Barlow, sans-serif !important; font-weight: 600 !important; }
h3 { font-size: 1.35rem !important; margin-top: 0.6rem !important; }
.g14-lede { color: #5B6876; font-size: 1.02rem; max-width: 78ch; margin: 0.25rem 0 1.0rem 0; line-height: 1.5; }
.g14-tag { display: inline-block; padding: 1px 8px 2px 8px; border-radius: 3px; font-size: 0.74rem;
           font-weight: 600; color: #fff; margin-left: 6px; vertical-align: 2px; letter-spacing: 0.01em; }
.g14-ch { display: flex; align-items: baseline; flex-wrap: wrap; gap: 2px; margin: 0.35rem 0 0.15rem 2px; }
.g14-ch-t { font-family: 'Barlow Semi Condensed', Barlow, sans-serif; font-weight: 600; font-size: 1.04rem; color: #18212B; }
.g14-ch-s { width: 100%; color: #5B6876; font-size: 0.86rem; margin-top: 1px; }
.g14-note { border-left: 3px solid #2456C9; background: #F5F8FC; padding: 0.55rem 0.9rem; margin: 0.5rem 0 0.9rem 0;
            color: #2A3542; font-size: 0.92rem; line-height: 1.5; max-width: 104ch; border-radius: 0 3px 3px 0; }
.g14-warn { border-left-color: #C46A12; background: #FDF7F0; }
.g14-ro-grid { display: grid; gap: 10px; margin: 0.3rem 0 0.8rem 0; }
.g14-ro { background: #FFFFFF; border: 1px solid #DCE2E8; border-top: 3px solid #1E2A38; border-radius: 3px;
          padding: 8px 12px 9px 12px; min-width: 0; }
.g14-ro .l { color: #5B6876; font-size: 0.8rem; line-height: 1.25; }
.g14-ro .v { font-family: 'Barlow Semi Condensed', Barlow, sans-serif; font-weight: 600; font-size: 1.55rem;
             color: #18212B; line-height: 1.2; margin-top: 2px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.g14-ro .u { font-size: 0.95rem; color: #5B6876; font-weight: 500; margin-left: 3px; }
.g14-ro .s { color: #5B6876; font-size: 0.78rem; margin-top: 1px; line-height: 1.3; }
.g14-pass { font-size: 0.84rem; line-height: 1.5; color: #2A3542; }
.g14-pass table { border-collapse: collapse; width: 100%; margin-top: 4px; }
.g14-pass td { padding: 1px 0; border: none; }
.g14-pass td.k { color: #5B6876; width: 42%; }
.g14-pass td.v { font-variant-numeric: tabular-nums; text-align: right; }
.g14-pass .h { font-family: 'Barlow Semi Condensed', Barlow, sans-serif; font-weight: 600; font-size: 0.98rem; color: #18212B; }
.g14-pass .src { color: #5B6876; font-size: 0.78rem; }
.g14-legend { font-size: 0.8rem; color: #2A3542; line-height: 1.65; margin-top: 2px; }
.g14-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; vertical-align: -1px; }
.g14-legend span { color: #5B6876; }
.g14-foot { color: #7A8794; font-size: 0.78rem; margin-top: 2.4rem; padding-top: 0.7rem; border-top: 1px solid #DCE2E8;
            max-width: 110ch; line-height: 1.45; }
.g14-sheet { width: 100%; border-collapse: collapse; background: #FFFFFF; border: 1px solid #DCE2E8; font-size: 0.9rem; }
.g14-sheet th { text-align: left; font-weight: 600; color: #5B6876; font-size: 0.78rem; padding: 7px 10px;
                border-bottom: 1px solid #DCE2E8; background: #F7F9FA; }
.g14-sheet td { padding: 6px 10px; border-bottom: 1px solid #EEF1F4; vertical-align: top; }
.g14-sheet td.n { font-variant-numeric: tabular-nums; white-space: nowrap; }
.g14-sheet.params td:first-child { width: 30%; }
.g14-sheet tr:last-child td { border-bottom: none; }
div[data-testid="stExpander"] details summary p { font-weight: 600; }
</style>
"""


def page_header(title: str, lede: str):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title, anchor=False)
    st.markdown(f'<p class="g14-lede">{lede}</p>', unsafe_allow_html=True)


def tag(kind: str) -> str:
    return f'<span class="g14-tag" style="background:{PROV[kind]}">{kind}</span>'


def tags(*kinds) -> None:
    st.markdown("".join(tag(k) for k in kinds), unsafe_allow_html=True)


def note(text: str, warn: bool = False):
    st.markdown(f'<div class="g14-note{" g14-warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def section(title: str, sub: str | None = None):
    st.subheader(title, anchor=False)
    if sub:
        st.caption(sub)


def footer():
    st.markdown('<div class="g14-foot">Data: NASA Ames PCoE Li-ion Battery Aging Data Set (B. Saha and K. Goebel, '
                'public U.S. Government data) and KU Leuven BEV energy dynamics dataset (Yasko et al., 2025, '
                'doi:10.48804/8KPDTW, CC BY 4.0). Group 14 virtual laboratory for 21EEO601T Electric Vehicle '
                'Technology. All metrics are computed by this application.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- charts
def _subplot_titles_left(fig) -> int:
    """Left-align make_subplots panel titles over their axes; returns how many were found."""
    domains = {}
    for k in fig.layout:
        if k.startswith("xaxis"):
            ax = fig.layout[k]
            if ax.domain:
                domains[round(sum(ax.domain) / 2, 4)] = ax.domain[0]
    n = 0
    for a in fig.layout.annotations or []:
        if a.showarrow is False and a.xref == "paper" and a.yref == "paper" and a.text and a.x is not None:
            key = round(a.x, 4)
            if key in domains:
                a.update(x=domains[key], xanchor="left", font=dict(size=13, color=MUTED, family=FONT))
                n += 1
    return n


def chart(fig, title: str | None = None, prov=(), height: int = 380, key=None, sub: str | None = None,
          legend: str = "auto"):
    """Render a Plotly figure with an HTML header (title + provenance tags) and a legend that never
    collides with titles: horizontal row at the top for up to 7 series, vertical on the right beyond."""
    if title or prov:
        t = f'<span class="g14-ch-t">{html.escape(title)}</span>' if title else ""
        s = f'<span class="g14-ch-s">{sub}</span>' if sub else ""
        st.markdown(f'<div class="g14-ch">{t}{"".join(tag(k) for k in prov)}{s}</div>', unsafe_allow_html=True)
    fig.update_layout(title=None, height=height)
    n_panel_titles = _subplot_titles_left(fig)
    named = [tr for tr in fig.data if tr.showlegend is not False and getattr(tr, "name", None)]
    top = 12 + (26 if n_panel_titles else 0)
    if legend == "none" or len(named) <= 1:
        fig.update_layout(showlegend=False, margin=dict(t=top + 8))
    else:
        mode = legend if legend != "auto" else ("top" if len(named) <= 7 else "right")
        if mode == "top":
            fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=1, yanchor="top", yref="container",
                                          traceorder="normal"), margin=dict(t=top + 34))
        else:
            fig.update_layout(legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"),
                              margin=dict(t=top + 8, r=200))
    st.plotly_chart(fig, theme=None, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": (title or "figure")[:40]}})


plot = chart  # backwards-compatible name


def heatmap_axes(fig, x_title, y_title):
    fig.update_xaxes(type="category", title=x_title, showgrid=False, ticks="")
    fig.update_yaxes(type="category", title=y_title, showgrid=False, ticks="", autorange="reversed")


# ----------------------------------------------------------------------------- readouts & tables
def readouts(items, cols: int | None = None, accent: str | None = None):
    """Instrument-style readouts. items: (label, value, unit[, sub]) tuples."""
    cols = cols or len(items)
    cells = []
    for it in items:
        label, value, unit = it[0], it[1], it[2]
        sub = it[3] if len(it) > 3 else ""
        style = f' style="border-top-color:{accent}"' if accent else ""
        subhtml = f'<div class="s">{sub}</div>' if sub else ""
        cells.append(f'<div class="g14-ro"{style}><div class="l">{label}</div><div class="v">{value}'
                     f'<span class="u">{unit}</span></div>{subhtml}</div>')
    st.markdown(f'<div class="g14-ro-grid" style="grid-template-columns:repeat({cols},minmax(0,1fr))">{"".join(cells)}</div>',
                unsafe_allow_html=True)


def table(df: pd.DataFrame, spec: dict, height="auto", key=None):
    """st.dataframe with human labels, units and number formats. spec: col -> (label, fmt|None[, help])."""
    cfg, cols = {}, []
    for c, v in spec.items():
        if c not in df.columns:
            continue
        label, fmt_ = v[0], v[1] if len(v) > 1 else None
        helptext = v[2] if len(v) > 2 else None
        cols.append(c)
        cfg[c] = (st.column_config.NumberColumn(label, format=fmt_, help=helptext) if fmt_
                  else st.column_config.TextColumn(label, help=helptext))
    if height == "auto" and 10 < len(df) <= 25:
        height = 35 * (len(df) + 1) + 3          # show every row instead of an inner scroll bar
    st.dataframe(df[cols], column_config=cfg, hide_index=True, width="stretch", height=height, key=key)


def sheet(rows, header=("Parameter", "Value", "Basis")):
    """Datasheet-style static table."""
    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f'<td class="{"n" if j == 1 else ""}">{c}</td>' for j, c in enumerate(r)) + "</tr>"
                   for r in rows)
    cls = "g14-sheet params" if header[0] == "Parameter" else "g14-sheet"
    st.markdown(f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)


def fmt(x, nd=2, none="–"):
    try:
        return f"{x:,.{nd}f}" if np.isfinite(x) else none
    except Exception:
        return str(x)


# ----------------------------------------------------------------------------- domain checks
def domain_warnings(ambient=None, crate=None, soh=None, p: ECMParams | None = None):
    msgs = []
    if ambient is not None and (ambient < 4 or ambient > 44):
        rng = p.extra.get("calibrated_T_range_C") if p is not None else None
        extra = f"; the Arrhenius law was fitted on mean cell temperatures {rng[0]:.0f}–{rng[1]:.0f} °C" if rng else ""
        msgs.append(f"ambient {ambient} °C lies outside the NASA test conditions (4–44 °C){extra}")
    if crate is not None and crate > 2.2:
        msgs.append(f"{crate:.1f}C exceeds the highest NASA load (4 A ≈ 2.2C)")
    if soh is not None and soh < 60:
        msgs.append(f"SOH {soh} % is below the most aged NASA reference cell (≈ 58 %)")
    if msgs:
        note("Outside the calibrated domain, results are extrapolated: " + "; ".join(msgs) + ".", warn=True)


# ----------------------------------------------------------------------------- cached data
@st.cache_data(show_spinner=False)
def cells():
    return da.cells()


@st.cache_data(show_spinner=False)
def discharge_cycles():
    return da.discharge_cycles()


@st.cache_data(show_spinner=False)
def charge_cycles():
    return da.charge_cycles()


@st.cache_data(show_spinner=False)
def impedance():
    return da.impedance()


@st.cache_data(show_spinner=False, max_entries=64)
def discharge_ts(cell, idx=None):
    return da.discharge_ts(cell, idx)


@st.cache_data(show_spinner=False, max_entries=64)
def charge_ts(cell, idx=None):
    return da.charge_ts(cell, idx)


@st.cache_data(show_spinner=False, max_entries=32)
def eis(cell):
    return da.eis_spectra(cell)


@st.cache_data(show_spinner=False, max_entries=128)
def cycle(cell, idx):
    return da.cycle_dict(cell, int(idx))


@st.cache_data(show_spinner=False, max_entries=64)
def charge_after(cell, discharge_idx):
    """CC-CV charge following a discharge (SOC 0 = that discharge's cut-off), or None."""
    return da.charge_after(cell, int(discharge_idx))


@st.cache_data(show_spinner=False)
def eis_re(cell):
    return da.eis_ohmic_resistance(cell)


def bev_available() -> bool:
    return (paths.PROCESSED / "bev_sessions.parquet").exists()


@st.cache_data(show_spinner=False)
def bev_sessions():
    return pd.read_parquet(paths.PROCESSED / "bev_sessions.parquet")


@st.cache_data(show_spinner=False, max_entries=48)
def bev_ts(kind, session_id):
    df = pd.read_parquet(paths.PROCESSED / f"bev_ts_{kind}.parquet", filters=[("session_id", "==", session_id)])
    df["session_id"] = df["session_id"].astype(str)
    df["vehicle"] = df["vehicle"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def metrics_csv(name):
    p = paths.METRICS / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def metrics_parquet(name):
    p = paths.METRICS / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def rf_bundle():
    """Trained RF SOH baseline; retrained (≈ 1 s) if the pickle does not load on this scikit-learn."""
    import joblib

    from . import soh
    try:
        return joblib.load(paths.TRAINED / "rf_soh_reference.joblib")
    except Exception:
        fs = soh.DEFAULT_FEATURE_SET
        feats = soh.FEATURE_SETS[fs]
        df = soh.dataset(da.discharge_cycles(), feats, soh.REFERENCE_CELLS, False)
        return dict(model=soh.fit_full(df, feats, "Random Forest"), features=feats, feature_set=fs,
                    trained_on=soh.REFERENCE_CELLS, target="capacity_Ah")


def rf_capacity(cell: str, idx: int):
    """Capacity the RF SOH model reports for a cycle it has NOT seen: the leave-one-cell-out
    prediction for reference cells, the full model otherwise. Returns (Ah or nan, how)."""
    b = rf_bundle()
    preds = metrics_parquet("soh_loco_predictions.parquet")
    if len(preds):
        m = preds[(preds.feature_set == b.get("feature_set")) & (preds.model == "Random Forest") &
                  preds.cell_set.str.startswith("Reference") & (preds.cell_id == cell) & (preds.discharge_index == idx)]
        if len(m):
            return float(m.pred_Ah.iloc[0]), "leave-one-cell-out prediction (cell unseen in training)"
    if cell in b.get("trained_on", []):
        return np.nan, "cycle filtered out of the SOH dataset"
    row = discharge_cycles().query("cell_id == @cell and discharge_index == @idx")
    x = row[b["features"]].to_numpy(float)
    if len(x) and np.isfinite(x).all():
        return float(b["model"].predict(x)[0]), "full model (cell not in its training set)"
    return np.nan, "indicators missing for this cycle"


# ----------------------------------------------------------------------------- active twin
def default_twin() -> dict:
    p = ECMParams.load(paths.TRAINED / "ecm_reference.json")
    th = json.loads((paths.TRAINED / "thermal_reference.json").read_text())
    return dict(ecm=p.to_dict(), thermal=th, label="Reference twin, B0005 cycles 2–6 (discharge + charge)", calibrated_at="build")


def twin():
    if "twin" not in st.session_state:
        st.session_state["twin"] = default_twin()
    t = st.session_state["twin"]
    return ECMParams.from_dict(t["ecm"]), ThermalParams(**t["thermal"]), t


def twin_hash() -> str:
    _, _, t = twin()
    return hashlib.md5(json.dumps([t["ecm"], t["thermal"]], sort_keys=True, default=str).encode()).hexdigest()[:10]


def set_twin(p: ECMParams, th: ThermalParams, label: str):
    st.session_state["twin"] = dict(ecm=p.to_dict(), thermal=th.to_dict(), label=label,
                                    calibrated_at=datetime.now().strftime("%H:%M:%S"))


def twin_soh_pct(p: ECMParams) -> float:
    """SOH of the calibrated twin itself (% of the 2.0 Ah rating)."""
    return p.soh_pct


def resistance_factor(soh_pct: float, p: ECMParams) -> float:
    """Resistance multiplier at a given SOH relative to the twin's own state (growth law, see ECMParams)."""
    return p.resistance_factor(soh_pct)


def passport():
    p, th, t = twin()
    r1 = f"{p.R1*1000:.0f} / {p.R1_chg*1000:.0f} mΩ" if p.R1_chg is not None else f"{p.R1*1000:.1f} mΩ"
    rows = [("R0", f"{p.R0*1000:.1f} mΩ"), ("R1 dis / chg" if p.R1_chg is not None else "R1", r1), ("τ1", f"{p.tau1:.0f} s"),
            ("Q", f"{p.Q_Ah:.3f} Ah"), ("Ea", f"{p.Ea_R0/1000:.1f} kJ/mol"), ("C_th", f"{th.C_th:.0f} J/K"),
            ("hA", f"{th.hA*1000:.1f} mW/K"), ("dU/dT", f"{th.dUdT*1000:+.2f} mV/K"), ("Fit", f"{p.fit_rmse_mV:.1f} mV RMSE")]
    body = "".join(f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in rows)
    when = f" (calibrated {t['calibrated_at']})" if t["calibrated_at"] != "build" else ""
    st.sidebar.markdown(f'<div class="g14-pass"><div class="h">Active twin</div><div class="src">{t["label"]}{when}</div>'
                        f'<table>{body}</table></div>', unsafe_allow_html=True)
    leg = "".join(f'<div><i style="background:{c}"></i>{k} <span>{PROV_MEANING[k]}</span></div>' for k, c in PROV.items())
    st.sidebar.markdown(f'<div class="g14-legend">{leg}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- experiment log / export
def log_experiment(kind: str, config: dict, summary, data: pd.DataFrame | None = None, fig=None):
    st.session_state.setdefault("log", [])
    rec = dict(id=len(st.session_state["log"]) + 1, kind=kind, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               config=config, summary=summary)
    if data is not None:
        rec["data_csv"] = data.to_csv(index=False)
    if fig is not None:
        rec["fig_html"] = fig.to_html(include_plotlyjs="cdn", full_html=True)
    st.session_state["log"].append(rec)
    return rec["id"]


def bundle(records: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        index = []
        for r in records:
            stem = f"exp{r['id']:03d}_{r['kind'].replace(' ', '_')}"
            z.writestr(f"{stem}/config.json", json.dumps(dict(config=r["config"], summary=r["summary"], time=r["time"]),
                                                         indent=2, default=str))
            if "data_csv" in r:
                z.writestr(f"{stem}/data.csv", r["data_csv"])
            if "fig_html" in r:
                z.writestr(f"{stem}/figure.html", r["fig_html"])
            index.append(dict(id=r["id"], kind=r["kind"], time=r["time"]))
        z.writestr("index.json", json.dumps(index, indent=2))
    return buf.getvalue()


def export_bar(kind: str, config: dict, summary, data: pd.DataFrame | None, fig=None, key="x"):
    c1, c2, c3 = st.columns([1.1, 1.1, 3], vertical_alignment="center")
    with c1:
        if data is not None:
            st.download_button("Download data (CSV)", data.to_csv(index=False).encode(), f"{kind.replace(' ', '_')}.csv",
                               "text/csv", key=f"dl_{key}", icon=":material/download:", width="stretch")
    with c2:
        if st.button("Add to experiment log", key=f"log_{key}", icon=":material/note_add:", width="stretch"):
            i = log_experiment(kind, config, summary, data, fig)
            st.toast(f"Logged as experiment {i}. Export all runs from Validation & export.")
    with c3:
        n = len(st.session_state.get("log", []))
        st.caption(f"Logged runs keep configuration, summary, data and an interactive figure. "
                   f"{n} run{'s' if n != 1 else ''} logged this session.")


export_buttons = export_bar  # backwards-compatible name

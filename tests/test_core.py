"""Engine tests (run: make test, or PYTHONPATH=src python -m pytest -q tests)."""
import json

import numpy as np
import pandas as pd
import pytest

from g14lab import data_access as da
from g14lab import ecm, paths, soh, thermal
from g14lab import estimators as est
from g14lab import simulator as sim
from g14lab.ica import ica_curve, ica_features
from g14lab.sensitivity import tornado
from g14lab.thermal import ThermalParams


@pytest.fixture(scope="module")
def twin():
    p = ecm.ECMParams.load(paths.TRAINED / "ecm_reference.json")
    th = ThermalParams(**json.loads((paths.TRAINED / "thermal_reference.json").read_text()))
    return p, th


# ----------------------------------------------------------------------------- data
def test_dataset_inventory_and_sign_convention():
    cells = da.cells()
    dc = da.discharge_cycles()
    assert len(cells) == 34
    assert dc.cell_id.nunique() == 34
    c = da.cycle_dict("B0005", 10)
    assert np.median(c["I"][c["I"] > 0.2]) == pytest.approx(2.0, abs=0.05)   # +2 A = discharge
    assert abs(c["Q"] - c["Q_nasa"]) < 0.05                                  # integration matches NASA field


# ----------------------------------------------------------------------------- model physics
def test_ocv_monotone_and_bounded(twin):
    p, _ = twin
    v = p.ocv(np.linspace(0, 1, 101))
    assert np.all(np.diff(v) >= -1e-9)
    assert 3.0 < v[0] < 3.8 and 4.1 < v[-1] < 4.25


def test_arrhenius_direction(twin):
    p, _ = twin
    assert p.Ea_R0 > 0
    assert p.r0_at(5.0) > p.r0_at(25.0) > p.r0_at(45.0)


def test_simulator_charge_conservation_and_heating(twin):
    p, th = twin
    st = sim.TwinState(soc0=1.0, T0_C=25, T_amb_C=25)
    df, s1 = sim.run_protocol(p, th, st, [{"type": "cc", "current_A": 2.0, "until_V": 2.7}])
    assert 0.9 * p.Q_Ah < s1["Ah_discharged"] < 1.15 * p.Q_Ah
    assert (df.Qgen_W - df.Qrev_W >= -1e-12).all()                           # irreversible heat is never negative
    _, s2 = sim.run_protocol(p, th, st, [{"type": "cc", "current_A": 4.0, "until_V": 2.7}])
    assert s2["T_max_C"] > s1["T_max_C"]                                     # higher C-rate -> hotter


def test_cccv_charge_terminates_near_full(twin):
    p, th = twin
    df, s = sim.run_protocol(p, th, sim.TwinState(soc0=0.05, T0_C=25, T_amb_C=25),
                             [{"type": "cc", "current_A": -1.5, "until_V": 4.2},
                              {"type": "cv", "voltage_V": 4.2, "until_A": 0.02}])
    assert df.V.max() <= 4.2 + 1e-3
    assert s["soc_final_pct"] > 88                                          # near the rested full state
    assert "taper" in s["end_reason"] or "settled" in s["end_reason"]       # CV ends on current, not on the timer
    assert s["duration_min"] < 240


def test_reference_twin_is_physically_bounded(twin):
    p, th = twin
    re = p.extra.get("eis_Re_ohm")
    assert re is not None and p.R0 >= re - 1e-9                              # R0 never below the EIS ohmic resistance
    assert p.extra.get("temperature_aware") and p.T_ref_C == pytest.approx(24.0)
    assert p.extra.get("n_charges", 0) > 0                                   # charges took part in the identification


def test_polarisation_resistance_is_direction_dependent(twin):
    p, _ = twin
    assert p.R1_chg is not None
    assert float(p.r1_at(25.0, 1.0, -1.5)) == pytest.approx(p.R1_chg * float(p.arrhenius(25.0, p.Ea_R1)))
    assert float(p.r1_at(25.0, 1.0, 2.0)) == pytest.approx(float(p.r1_at(25.0)))
    sym = ecm.ECMParams.from_dict({**p.to_dict(), "R1_chg": None})             # symmetric fallback
    assert float(sym.r1_at(25.0, 1.0, -1.5)) == pytest.approx(float(sym.r1_at(25.0)))


def test_twin_reproduces_a_held_out_charge(twin):
    p, th = twin
    c = da.charge_after("B0005", 10)
    soc = ecm.soc_reference(c["t"], c["I"], c["Q"], soc0=0.0)
    vm, _ = ecm.simulate_voltage(p, c["t"], c["I"], soc, T_C=c["T"])
    assert np.sqrt(np.mean((vm - c["V"]) ** 2)) < 0.06                       # < 60 mV on the replayed charge
    ch = da.charge_cycles()
    t_meas = float(ch[(ch.cell_id == "B0005") & (ch.charge_index == c["idx"])].t_cc_s.iloc[0])
    df, _ = sim.run_protocol(p, th, sim.TwinState(soc0=0.0, T0_C=25, T_amb_C=24, soh=c["Q"] / p.Q_Ah),
                             [{"type": "cc", "current_A": -1.5, "until_V": 4.2}], dt=2.0)
    assert df.t_s.iloc[-1] == pytest.approx(t_meas, rel=0.15)               # CC phase to 4.2 V within 15 %


def test_reversible_heat_changes_sign_with_current(twin):
    p, th = twin
    th2 = ThermalParams(C_th=th.C_th, hA=th.hA, dUdT=-5e-4)
    d, _ = sim.run_protocol(p, th2, sim.TwinState(soc0=1.0), [{"type": "cc", "current_A": 2.0, "until_V": 3.0}], dt=5.0)
    c, _ = sim.run_protocol(p, th2, sim.TwinState(soc0=0.1), [{"type": "cc", "current_A": -1.5, "until_V": 4.1}], dt=5.0)
    assert d.Qrev_W.iloc[1:].mean() > 0 > c.Qrev_W.iloc[1:].mean()


def test_entropic_term_needs_both_current_directions(twin):
    p, _ = twin
    cyc = [da.cycle_dict("B0005", 3)]
    th = thermal.identify_thermal(cyc, p, [24.0])
    assert th.dUdT == 0.0 and th.C_th > 0 and th.hA > 0


# ----------------------------------------------------------------------------- estimation
@pytest.mark.parametrize("adaptive", [False, True])
def test_filters_beat_coulomb_counting_with_wrong_init(twin, adaptive):
    p, _ = twin
    c = da.cycle_dict("B0005", 30)
    ref = ecm.soc_reference(c["t"], c["I"], c["Q"])
    cc = est.coulomb_counting(c["t"], c["I"], p.Q_Ah, 0.8)
    e = est.ekf(p, c["t"], c["I"], c["V"], 0.8, Q_Ah=p.Q_Ah, T_C=c["T"], adaptive=adaptive)
    assert np.mean(np.abs(e["soc"] - ref)) < np.mean(np.abs(cc - ref))


def test_ukf_runs_and_is_bounded(twin):
    p, _ = twin
    c = da.cycle_dict("B0025", 10)
    u = est.ukf(p, c["t"], c["I"], c["V"], 0.8, Q_Ah=p.Q_Ah, T_C=c["T"])
    assert np.isfinite(u["soc"]).all() and (u["soc"] >= -0.1).all() and (u["soc"] <= 1.1).all()


# ----------------------------------------------------------------------------- SOH
def test_ica_features_track_ageing():
    ts = da.charge_ts("B0005", (2, 160))
    feats = []
    for k in (2, 160):
        s = ts[ts.charge_index == k]
        feats.append(ica_features(*ica_curve(s.t_s.to_numpy(float), s.V.to_numpy(float), -s.I.to_numpy(float))))
    assert feats[1]["ic_peak_AhV"] < feats[0]["ic_peak_AhV"]               # peak falls with ageing


def test_loco_has_no_cell_leakage_and_conformal_bounds():
    d = da.discharge_cycles()
    feats = soh.FEATURE_SETS["Charging-phase indicators (on-board measurable)"]
    df = soh.dataset(d, feats, soh.REFERENCE_CELLS, False)
    out, per_cell, pooled = soh.loco(df, feats, "Ridge regression")
    assert set(per_cell.cell_id) == set(soh.REFERENCE_CELLS)
    assert 0.0 <= pooled["conformal_coverage_90"] <= 1.0
    assert (out.conf_hi_Ah >= out.conf_lo_Ah).all()


def test_fade_models_and_backtest():
    d = da.discharge_cycles()
    g = d[(d.cell_id == "B0006") & d.usable_for_soh].sort_values("discharge_index")
    r = soh.fade_forecast(g.discharge_index.to_numpy(), g.capacity_Ah.to_numpy(), 60, model="linear")
    assert set(r["aicc"]) == set(soh.FADE_MODELS)
    assert np.isfinite(r["eol_pred"])
    bt = soh.forecast_backtest(d, cells=["B0006"], origins=(60,))
    assert isinstance(bt, pd.DataFrame) and set(bt.model) == set(soh.FADE_MODELS)


def test_sensitivity_tornado(twin):
    p, th = twin
    tor, base = tornado(p, th, sim.TwinState(soc0=1.0, T0_C=25, T_amb_C=25),
                        [{"type": "cc", "current_A": 2.0, "until_V": 2.7}], dt=5.0)
    q = tor[(tor.output == "Ah_discharged") & (tor.parameter == "Capacity Q")]
    assert (q[q.change == "+20 %"].delta.iloc[0]) > 0                        # more capacity -> more charge

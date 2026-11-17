| Table / column | Unit | Meaning | Provenance |
|---|---|---|---|
| **ts_discharge / ts_charge** | | sample-level NASA records | measured |
| t_s | s | time since start of the operation | measured |
| V | V | terminal voltage (Voltage_measured) | measured |
| I | A | current, **positive = discharge** (sign of Current_measured flipped) | measured |
| T | °C | cell temperature (Temperature_measured) | measured |
| **discharge_cycles** | | one row per discharge | derived |
| capacity_Ah | Ah | NASA-reported discharge capacity | measured (NASA) |
| soh_pct | % | 100 × capacity / 2.0 Ah rated | derived |
| ah_integrated | Ah | ∫I dt to the end of load (our integration) | derived |
| t_to_3p8V_s, t_to_3p6V_s | s | time from load start to voltage threshold | derived |
| T_rise_C, t_to_Tmax_s | K, s | temperature rise; time to peak temperature | derived |
| r_dc_first_sample_ohm | Ω | (rest V − first loaded V)/I, DC-resistance proxy at ~10–19 s | derived |
| t_cc_s, q_cc_Ah, t_cv_s | s, Ah, s | preceding charge: CC duration, CC charge, CV duration | derived |
| Re_ohm, Rct_ohm | Ω | last EIS before this discharge (NASA fitted) | measured (NASA fit) |
| ic_peak_AhV, ic_peak_V | Ah/V, V | main dQ/dV peak height and voltage of the preceding CC charge (Savitzky–Golay, 5 mV grid) | derived |
| ic_area_Ah | Ah | charge inside 3.80–4.10 V of the preceding CC charge (area under dQ/dV) | derived |
| flag_low_capacity, flag_capacity_mismatch, usable_for_soh | bool | quality flags | derived |
| **bev_ts_*** | | Tesla 1 s pack telemetry | measured |
| soc_bms_pct | % | BMS average SOC (SOCave292) | BMS-reported |
| v_pack_V, i_pack_A | V, A | pack voltage; current (+ = discharge) | measured (CAN) |
| t_min_C, t_max_C, t_amb_C | °C | min/max pack temperature, ambient | measured (CAN) |
| speed_kph, p_ac_kW, fc_i_A, fc_v_V | | speed, AC line power, DC charger current/voltage | measured (CAN) |
| p_dis_lim_kW, p_regen_lim_kW, bms_diss_kW | kW | BMS power limits and dissipation | BMS-reported |
| **bev_sessions** | | per-session summary | derived |
| q_implied_Ah | Ah | abs(∫I dt) / abs(ΔSOC_BMS), only when ΔSOC ≥ 20 % and > 90 % of the session is logged | proxy |
| kwh_per_100km | kWh/100 km | pack-terminal energy / distance (drives > 5 km) | derived |
| logged_fraction | 0–1 | share of the session duration actually logged (gaps > 10 s excluded) | derived |
| **model_metrics** | | benchmarks written by `python -m g14lab.calibration` | computed |
| param_tracking.parquet | | per-cycle R0, R1, τ1 of the reference cells (digital shadow) with EIS | computed |
| soh_benchmark.csv, soh_loco_predictions.parquet | | LOCO / K-fold SOH results and predictions with intervals | computed |
| soc_benchmark.csv, twin_validation.csv, shadow_validation.csv, forecast_backtest.csv | | estimator, twin (discharge), shadow and forecast validation | computed |
| charge_validation.csv | | held-out CC-CV charges: replayed-current V/T RMSE; CC-phase duration and charge, measured vs twin | computed |
| parameter_map.csv | | per-cycle R0, R1, τ1 of 26 cells at their mean temperature (input to the Arrhenius fit) | computed |
| **trained_models** | | ecm_reference.json (OCV knots, R0, R1, R1_chg, τ1, Ea, T_ref, growth law), thermal_reference.json (C_th, hA, dU/dT), rf_soh_reference.joblib | computed |

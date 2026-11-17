# EV Battery Digital-Twin Virtual Lab (SOC/SOH)

**Group 14, 21EEO601T Electric Vehicle Technology (LLT-1), SRM Institute of Science and Technology**
Bharathwaj Manoharan, Gokul Vasan R, Arvind Kumar

A virtual laboratory for lithium-ion **state of charge (SOC)** and **state of health (SOH)**. A cell digital twin is
*identified from measured data* (NASA PCoE ageing tests: 34 cells, 4 / 24 / 43 °C, 1 / 2 / 4 A and pulsed loads).
The lab runs electro-thermal experiments on it, compares SOC estimators under sensor faults, estimates SOH with
leakage-free machine learning, and checks everything against measured cells and two real Tesla Model Y packs
(KU Leuven, 1 s CAN telemetry).

![Overview page](outputs/figures/demo/d01_overview.png)

## Community edition

Course-assigned title: **AI-Based Virtual Laboratory for Battery Digital Twin (SOC/SOH)**.

This repository is a starting point for future student groups to study the existing lab,
reproduce its results, and develop new models and experiments. The original Group 14
reports and presentation remain in `docs/`.

- [Get the data and models](docs/DATA_SETUP.md)
- [Contribute changes](CONTRIBUTING.md)
- [Development roadmap](ROADMAP.md)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

## Quick start

Python 3.10 or newer is required. Start with the **v1.2.0** source and its matching
release data package. A Git clone or GitHub's automatic source ZIP alone does not
include the datasets or trained models.

On Linux, from a terminal:

```bash
git clone https://github.com/SreevanjiManoharan179/EVT_LLT_Project.git
cd "EVT_LLT_Project"
git switch --detach v1.2.0
curl -fL -O "https://github.com/SreevanjiManoharan179/EVT_LLT_Project/releases/download/v1.2.0/EVT_LLT_Project-v1.2.0-data.zip"
curl -fL -O "https://github.com/SreevanjiManoharan179/EVT_LLT_Project/releases/download/v1.2.0/SHA256SUMS"
sha256sum -c "SHA256SUMS"
unzip "EVT_LLT_Project-v1.2.0-data.zip" -d .
bash run.sh
```

Continue only if the checksum reports `OK`. macOS can use `shasum -a 256 -c SHA256SUMS`.
For Windows, or if `curl` or `unzip` is unavailable, download the source and data ZIPs
from the [v1.2.0 release](https://github.com/SreevanjiManoharan179/EVT_LLT_Project/releases/tag/v1.2.0),
then extract the data ZIP into the source folder beside `app.py`. On Windows, double-click
`run.bat`; on macOS, run `bash run.sh`. See [data setup](docs/DATA_SETUP.md) for the exact layout.

The launcher creates `.venv`, installs `requirements.txt`, prepares missing data/model
outputs, starts Streamlit on the first available port from 8501, and opens the browser.
The first launch needs internet for Python dependencies. With the release assets and
dependencies installed, the lab uses local data, models, and fonts. Press **Ctrl+C**
in the terminal to stop it.

Options include `--port 8600`, `--no-browser`, and `--rebuild`. Rebuilding uses the raw
archives and replaces the local processed data and model outputs. To install the Linux
desktop shortcut, use `bash run.sh --install-shortcut`.

If Ubuntu or Debian cannot create a virtual environment, install the distribution's
`python3-venv` package. For development, follow [CONTRIBUTING.md](CONTRIBUTING.md).

## The lab, page by page

| # | Page | What you can do |
|---|---|---|
| | Overview | Measured fade curves, data inventory, system diagram, demonstration path |
| 1 | Data hub & explorer | Browse every cell and Tesla session. View V/I/T traces, EIS Nyquist plots, histograms, scatter plots and Spearman correlations. See a data-quality profile and the data dictionary, and export to CSV |
| 2 | Calibrate the twin | Identify OCV(SOC), R0, R1, τ1, C_th, hA and the entropic coefficient dU/dT live from chosen discharges and the charges that follow them (EIS-bounded R0, temperature-aware fit). Compare measured vs simulated voltage and temperature on discharges and charges, see the Arrhenius map, and track how resistance ages in the digital shadow |
| 3 | Experiment bench | Run CC discharge, CC-CV charge, pulses or charge → rest → discharge, with initial SOC, C-rate, ambient and ageing set by a growth law or a measured cycle. Includes sweeps, A/B scenarios, a ±20 % tornado sensitivity, and validity warnings outside the calibrated domain |
| 4 | Thermal & fast charging | 36-run ΔT and usable-charge maps across C-rate × ambient, measured vs predicted temperature with the heat split, and Tesla DC power taper, C-rate and pack heating |
| 5 | SOC estimation | Coulomb counting vs EKF vs adaptive EKF vs UKF, with gain/offset/noise faults and a wrong initial SOC. Capacity comes from the RF SOH model, the twin or an oracle |
| 6 | SOH & ageing | Fade under each condition, incremental-capacity (dQ/dV) analysis, RF / Ridge / gradient boosting with leave-one-cell-out validation, conformal intervals and permutation importance, and a fade forecast with backtest |
| 7 | Tesla telemetry replay | Session replay with BMS power limits, BMS SOC vs Coulomb counting, energy use vs ambient over 721 drives, and pack chemistry inferred from data |
| 8 | Validation & export | Live re-validation of the active twin on 23 discharges and 10 CC-CV charges, SOC/SOH benchmarks, identified physical laws, limitations, and a ZIP export of logged experiments |

Every chart is tagged **Measured**, **Simulated**, **Estimated**, **Predicted** or **Reference**. The colours are used for provenance only.

## Engineering results (all computed by this code, `python -m g14lab.calibration`)

| Subsystem | Result |
|---|---|
| Identified twin (B0005: 5 discharges + the 5 CC-CV charges after them) | At 24 °C: R0 46.4 mΩ (bounded below by the cell's EIS Re, 44.8 mΩ), R1 220 mΩ discharge / 27 mΩ charge, τ1 108 s; voltage fit 41 mV RMSE. Thermal: C_th 106 J/K, hA 53.9 mW/K, fit 1.1 K; the identified entropic coefficient is negligible (−0.01 mV/K) |
| Temperature law | Arrhenius Ea = 22.3 kJ/mol from 41 cycles of 16 cells (R² 0.89); current dependence weak (β = 0.06) |
| Held-out discharges (median voltage RMSE) | 81 mV on other cells under the same protocol; 96 mV across the life of B0005; 161 mV at 43 °C; 208 mV at 4 °C; 190 mV at 4 A or with pulses. Temperature: median 2.1 K |
| Held-out CC-CV charges (10) | Replayed charge voltage within 35 mV RMSE (median), temperature within 1.2 K. Simulated time to reach 4.2 V at 1.5 A within 0.7 min (median) of measured for cells above 85 % SOH; overestimated by about 11 min for aged and 4 °C cells |
| Digital shadow | Tracked R0+R1 follows NASA EIS Re+Rct (Spearman ρ 0.75–0.88 for B0005/6/18; 0.31 for B0007). Growth law R/R_early = 1 + 0.53·ΔSOH (R² 0.52). One-cycle-ahead updating helps only late in life (142 → 125–127 mV), because the OCV-shape change dominates |
| SOC under faults (10 held-out cycles) | Median MAE: Coulomb counting 21.7 %, EKF 5.2 %, adaptive EKF 4.6 %, UKF 7.1 %. RF-estimated capacity performs as well as an oracle capacity (adaptive EKF 4.55 vs 4.66 %) |
| SOH, leave-one-cell-out (B0005/6/7/18) | Random Forest with charge + ICA + EIS indicators: 50.9 mAh RMSE (2.5 % SOH), R² 0.92. Ridge 47.8 mAh. The same RF scored with leaky random K-fold: 20.1 mAh |
| SOH uncertainty | Cross-conformal 90 % intervals cover 82 %; RF tree quantiles cover only 49 % |
| Fade forecasting (backtest, 16 forecasts) | Linear trend: median EOL error 8 cycles. Exponential and double-exponential models diverge (median 18–25 cycles) |
| Tesla packs | Implied capacity 167 Ah (BEV1) vs 210 Ah (BEV2). Mid-SOC slope 132 vs 972 mV/% points to LFP vs NCA. Drives below 10 °C use 19 % more energy per km |

## Architecture

![Architecture](docs/figures/system_architecture.png)

The engine (`src/g14lab`) is plain Python and fully testable; the Streamlit pages only orchestrate and plot.

```
EVT_LLT_Project/
├── app.py                     Streamlit entry point (navigation, twin passport)
├── run.sh · run.bat           one-command launchers (environment, dependencies, data check, browser)
├── requirements.txt · pyproject.toml
├── pages/                     one file per lab station (overview … validation)
├── src/g14lab/
│   ├── data_pipeline.py       NASA .mat → parquet, ICA indicators, quality flags
│   ├── telemetry.py           KU Leuven CSV → parquet sessions and summaries
│   ├── ecm.py · thermal.py    1-RC ECM, Arrhenius law, lumped thermal model, identification
│   ├── simulator.py           coupled electro-thermal protocol simulator
│   ├── estimators.py          Coulomb counting, EKF, adaptive EKF, UKF
│   ├── soh.py · ica.py        SOH models, LOCO, conformal intervals, fade models and backtest
│   ├── calibration.py         builds the twin and every validation artefact
│   ├── sensitivity.py · evaluation.py · profiling.py · data_access.py · ui.py · paths.py
├── data/   raw/ and processed/ restored from release · samples/ · metadata/
│           data_manifest.csv · data_dictionary.md
├── models/ trained_models/ and model_metrics/ restored from release
├── outputs/figures/ report/ (300 dpi figures) · demo/ (lab screenshots)    outputs/experiment_results/
├── docs/   Group14_Technical_Report.docx/.pdf (submission) · Technical_Report_Group14.md · figures/
│           course_brief/ (LLT brief, subtopics, front-page template) · presentation/ (deck, speech notes)
├── scripts/                   launch.py (used by run.sh/run.bat) · tidy.sh · lab.sh (reproduction tasks)
│                              make_report_figures.py
├── static/fonts/              Barlow (SIL OFL) for offline typography
└── tests/                     engine tests (pytest)
```

## Reproduce from the raw archives

First restore the release assets and set up the development environment described in
[CONTRIBUTING.md](CONTRIBUTING.md). The commands below are for contributors to run locally.

```bash
bash scripts/lab.sh data        # NASA + Tesla raw archives → data/processed   (~2 min)
bash scripts/lab.sh calibrate   # identification, tracking, all benchmarks     (~4 min)
bash scripts/lab.sh figures     # outputs/figures/report/*.png
bash scripts/lab.sh test        # pytest (17 engine tests)
bash scripts/lab.sh run         # start the lab (same as bash run.sh)
```

`bash run.sh --rebuild` runs the data and calibration steps and then starts the lab, in one command.

On Windows, set `PYTHONPATH=src` and run the same modules directly, e.g. `python -m g14lab.calibration`.

## Data and licences

| Dataset | Use | Licence |
|---|---|---|
| NASA PCoE Li-ion Battery Aging Data Set, `5. Battery Data Set.zip` (209,708,670 B, SHA-256 `82302a7d…f92e`) | twin identification, SOC, SOH, thermal | U.S. Government public data; cite Saha & Goebel |
| KU Leuven *BEV energy dynamics dataset* V2, doi:10.48804/8KPDTW (162,310,975 B, MD5 `cc2d6375…ff1c`) | vehicle replay, fast charging, fleet energy | CC BY 4.0 |
| Barlow typeface | interface and figures | SIL Open Font License 1.1 |

The NASA archive came from a GitHub LFS mirror, because NASA's S3 host was unreachable from the build environment.
Its SHA-256 is recorded so it can be checked against NASA's original. The KU Leuven archive came from KU Leuven RDR,
and its MD5 matches the repository record. Full provenance is in `data/data_manifest.csv`.

## Limitations

- The 1-RC model is identified from 2.5–19 s sampled data, so the R0/R1 split is only weakly identifiable; R0 is bounded below by the cell's EIS ohmic resistance. Resistances do not vary with SOC.
- There is no OCV hysteresis; the charge/discharge asymmetry is carried by the direction-dependent polarisation resistance. In CV charging the twin's current settles near 35 mA instead of tapering to the 20 mA cut-off, because its OCV saturates a few mV below 4.2 V; the simulator then ends the CV step and reports it.
- The OCV shape drifts with ageing and a fixed-OCV twin cannot follow it; this dominates late-life voltage error and makes the twin overestimate the CC charging time of aged cells.
- Thermal parameters are effective values (cell plus fixture), identified at 2 A and 1.5 A; peak temperature at 4 A is off by −7 to +3 K. This is not a thermal-runaway model.
- SOH models train on 4 reference cells, so the errors are indicative.
- Tesla SOC is BMS-reported, and implied capacity is a proxy. Cell parameters are never transferred to the packs.
- The twin is calibrated offline and replays recorded data; it is not synchronised with a live battery.

## Contributing and licence

Open an issue to discuss a change, then submit a focused pull request following
[CONTRIBUTING.md](CONTRIBUTING.md). The [roadmap](ROADMAP.md) lists possible next projects.
Preserve the distinction between measured, simulated, estimated, predicted, and reference results.

The group's original source code and original documentation are available under the
[MIT License](LICENSE). Datasets, fonts, institutional material, and other third-party
content retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
When using the lab in academic work, credit Group 14 and cite the underlying datasets.

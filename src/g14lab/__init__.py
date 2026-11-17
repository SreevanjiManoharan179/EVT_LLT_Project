"""Group 14 — EV Battery Digital-Twin Virtual Laboratory (SOC/SOH).

Computational engine, kept separate from the Streamlit interface:

* ``paths``          – project directory layout
* ``data_pipeline``  – NASA PCoE .mat -> tidy parquet tables, ICA indicators, quality flags
* ``telemetry``      – KU Leuven Tesla CSV sessions -> parquet time series and session summaries
* ``data_access``    – cached loaders (cycles, charges, EIS) used by the engine and the GUI
* ``profiling``      – dataset quality profiling
* ``ica``            – incremental-capacity (dQ/dV) curves and features
* ``ecm``            – OCV curve, 1-RC Thevenin model with direction-dependent R1, identification
* ``thermal``        – lumped thermal model (with entropic term) and its calibration
* ``simulator``      – coupled electro-thermal protocol simulator (CC, CV, rest, pulse, profile)
* ``estimators``     – Coulomb counting, EKF, adaptive EKF and UKF SOC estimators
* ``soh``            – health indicators, leave-one-cell-out ML, conformal intervals, fade forecasting
* ``calibration``    – builds the reference twin and every validation artefact
* ``sensitivity``    – one-at-a-time tornado analysis
* ``evaluation``     – metrics
* ``ui``             – Streamlit presentation helpers (charts, tables, twin passport)
"""

__version__ = "1.2.0"

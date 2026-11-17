# Contributing

Future groups are welcome to improve the battery models, experiments, interface, and
documentation. Start with a small change whose effect can be explained and checked.

## Set up a development copy

1. Fork the repository on GitHub and clone your fork.
2. Restore the v1.2.0 data and model assets using [DATA_SETUP.md](docs/DATA_SETUP.md).
3. Create a branch from `main`, for example `git switch -c improve-thermal-validation`.
4. Create a Python environment and install the dependencies below.

Linux and macOS, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The editable install makes `g14lab` importable from `src/`. The development extras
include pytest, Ruff, and Matplotlib. Python 3.10 is the declared minimum; consult
`requirements.txt` for the versions recorded for the original environment.

## Find the relevant code

| Area | Location |
| --- | --- |
| Cell data ingestion and units | `src/g14lab/data_pipeline.py`, `data_access.py` |
| Vehicle telemetry | `src/g14lab/telemetry.py` |
| Electrical and thermal physics | `src/g14lab/ecm.py`, `thermal.py`, `simulator.py` |
| SOC and SOH methods | `src/g14lab/estimators.py`, `soh.py`, `ica.py` |
| Calibration and validation | `src/g14lab/calibration.py`, `evaluation.py` |
| Streamlit interface | `app.py`, `pages/`, `src/g14lab/ui.py` |
| Existing tests | `tests/test_core.py` |

Keep physical calculations in the engine where possible. Document units, sign conventions,
assumptions, and the calibration domain. Avoid evaluating on the same cells used for training.
Keep measured quantities and model outputs clearly labelled.

## Check a change

With the development environment active and release assets restored:

```bash
python -m pytest -q tests
python -m ruff check .
```

On Windows use `.\.venv\Scripts\python.exe` in place of `python` if the environment
is not activated. Existing tests require the data and model assets; the small CSV samples
alone are insufficient. Record actual results in the pull request. If a check cannot be
run, say so and explain why. Do not describe an unrun check as passing.

For interface changes, run the lab locally and include a relevant screenshot. Add focused
tests for changed numerical behavior. For model changes, report the dataset split, units,
metrics, and any change in uncertainty or validity limits. Compare with the original
baseline without silently replacing its reported results.

## Submit a pull request

- Explain the problem and the resulting behavior.
- Keep unrelated refactoring out of the change.
- Include validation results and any known limitations.
- Update documentation when setup, behavior, or data requirements change.
- Keep environments, raw archives, processed data, model binaries, and experiment exports
  out of Git. Publish updated artifacts with a versioned release and checksums.

Use issues for reproducible bugs and proposed improvements. Include the release version,
operating system, Python version, reproduction steps, and relevant error output.

## Reuse and attribution

Original code contributions are accepted under the repository's MIT licence. Keep existing
copyright notices and third-party attribution. New datasets need their own provenance,
licence, checksums, and processing description. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

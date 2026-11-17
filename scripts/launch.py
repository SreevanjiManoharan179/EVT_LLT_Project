"""One-command launcher for the Group 14 virtual lab (run.sh and run.bat call this).

Checks that the prebuilt data and models are present (rebuilding only what is missing), picks a free
port, starts Streamlit and opens the browser as soon as the lab answers. Stop the lab with Ctrl+C.

    --port N       use port N (default: the first free port from 8501)
    --no-browser   do not open a browser tab
    --rebuild      rebuild processed data and the calibrated twin from data/raw first (about 7 min)
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
KU_ZIP = "BEV energy dynamic data_V2.zip"

# (module, what it does, files it produces, raw archives it needs, required for the lab to start)
STAGES = [
    ("g14lab.data_pipeline", "Building the NASA cell tables (about 2 min)",
     [PROCESSED / f"{n}.parquet" for n in ("cells", "discharge_cycles", "charge_cycles", "impedance",
                                           "eis_spectra", "ts_discharge", "ts_charge")],
     [RAW / "nasa_pcoe" / "5. Battery Data Set.zip"], True),
    ("g14lab.telemetry", "Building the Tesla session tables (about 1 min)",
     [PROCESSED / f"{n}.parquet" for n in ("bev_sessions", "bev_ts_drive", "bev_ts_fast_charge", "bev_ts_slow_charge")],
     [RAW / "kuleuven_bev" / KU_ZIP, ROOT.parent / KU_ZIP], False),
    ("g14lab.calibration", "Calibrating the twin and running every benchmark (about 4 min)",
     [ROOT / "models" / "trained_models" / f for f in ("ecm_reference.json", "rf_soh_reference.joblib",
                                                       "thermal_reference.json")]
     + [ROOT / "models" / "model_metrics" / f for f in ("twin_validation.csv", "charge_validation.csv",
                                                        "soc_benchmark.csv", "soh_benchmark.csv",
                                                        "soh_loco_predictions.parquet", "forecast_backtest.csv",
                                                        "param_tracking.parquet", "shadow_validation.csv",
                                                        "parameter_map.csv")],
     [], True),
]


def prepare(env: dict, rebuild: bool) -> None:
    """Rebuild only what is missing. Everything ships prebuilt, so normally this does nothing."""
    for module, what, outputs, archives, required in STAGES:
        missing = [p for p in outputs if not p.exists()]
        if not (rebuild or missing):
            continue
        if archives and not any(a.exists() for a in archives):
            where = f"'{archives[0].name}' is not in {archives[0].parent.relative_to(ROOT)}"
            names = ", ".join(p.name for p in missing)
            if missing and required:
                sys.exit(f"\nMissing {names}, which cannot be rebuilt because {where}.\n"
                         "Copy the archive there and run again.")
            if missing:
                print(f"\nNote: missing {names}, and {where}; the Tesla page will be empty.", flush=True)
            else:
                print(f"\nNote: keeping the prebuilt {module.split('.')[-1]} output because {where}.", flush=True)
            continue
        print(f"\n>> {what} ...", flush=True)
        if subprocess.run([sys.executable, "-m", module], cwd=ROOT, env=env).returncode:
            sys.exit(f"\n{module} failed; the messages above show why.")


def free_port(start: int = 8501, tries: int = 50) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if os.name != "nt":  # same rule the Streamlit server uses when it binds
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    sys.exit(f"No free port between {start} and {start + tries - 1}. Close other lab windows or use --port.")


def wait_until_up(port: int, proc: subprocess.Popen, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def open_browser(url: str) -> None:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return  # no desktop session (for example over SSH): just print the address
    try:
        webbrowser.open(url, new=2)
    except Exception:  # a missing browser must never stop the lab
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Start the Group 14 battery digital-twin virtual lab.")
    ap.add_argument("--port", type=int, help="port to use (default: the first free port from 8501)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild processed data and the calibrated twin from data/raw first (about 7 min)")
    args = ap.parse_args()

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    prepare(env, args.rebuild)

    port = args.port or free_port()
    url = f"http://localhost:{port}"
    print(f"\n>> Starting the virtual lab on {url} ...", flush=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                             "--server.port", str(port), "--server.headless", "true"], cwd=ROOT, env=env)
    try:
        if wait_until_up(port, proc):
            print(f"\n   The Battery Digital Twin lab is running at {url}\n"
                  "   Keep this window open. Press Ctrl+C here to stop the lab.\n"
                  "   Before presenting, open each page once: the first visit loads its data\n"
                  "   (Validation re-simulates 33 measured cycles and takes longest); after that, pages open at once.\n",
                  flush=True)
            if not args.no_browser:
                open_browser(url)
        elif proc.poll() is None:
            print(f"\n   The lab has not answered yet. Open {url} in a browser in a moment.\n", flush=True)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\nLab stopped.")
        return
    if proc.returncode:
        sys.exit(f"\nThe lab stopped with an error (exit code {proc.returncode}). The messages above show why.")


if __name__ == "__main__":
    main()

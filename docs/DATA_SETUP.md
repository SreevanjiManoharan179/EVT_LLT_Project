# Data and model setup

## Restore the release bundle

Use source tag **v1.2.0** with **EVT_LLT_Project-v1.2.0-data.zip** from the
[same GitHub release](https://github.com/SreevanjiManoharan179/EVT_LLT_Project/releases/tag/v1.2.0).
The bundle contains both original dataset archives, processed tables, trained reference
models, and benchmark outputs. It is a release attachment, not a Git LFS object.

Download the data ZIP and `SHA256SUMS` into the cloned project root. Verify the ZIP before
extracting it:

```bash
sha256sum -c "SHA256SUMS"
unzip "EVT_LLT_Project-v1.2.0-data.zip" -d .
sha256sum -c "docs/releases/v1.2.0-data-files.sha256"
```

Continue only when every entry reports `OK`. On macOS use `shasum -a 256 -c` in place of
`sha256sum -c`. In Windows PowerShell, print the hash and compare it with `SHA256SUMS`:

```powershell
Get-FileHash -Algorithm SHA256 "EVT_LLT_Project-v1.2.0-data.zip"
Get-Content "SHA256SUMS"
```

If the hashes match, extract the bundle:

```powershell
Expand-Archive -LiteralPath "EVT_LLT_Project-v1.2.0-data.zip" -DestinationPath .
```

For manual extraction, merge the archive's `data/` and `models/` directories into the
project root. Do not create an extra `EVT_LLT_Project-v1.2.0-data/` nesting level.
Extract into a fresh clone or back up locally modified data and models first.

The resulting layout should include:

```text
EVT_LLT_Project/
  app.py
  data/
    raw/
      nasa_pcoe/5. Battery Data Set.zip
      kuleuven_bev/BEV energy dynamic data_V2.zip
    processed/                 11 parquet tables
    samples/                   small CSV examples, already in Git
    metadata/                  original build summary, already in Git
  models/
    trained_models/            reference JSON and Random Forest model
    model_metrics/             validation and benchmark outputs
```

The archive also carries provenance and attribution under `data/release-info/`.
The raw dataset ZIPs stay zipped; the project handles extraction when rebuilding.
The launcher uses the prebuilt data and models when they are present. Samples alone
do not provide everything required by the lab or its existing tests.

## Obtain the original datasets independently

If release downloads are unavailable, obtain the archives from their original sources:

| Dataset | Source | Local destination |
| --- | --- | --- |
| NASA PCoE battery ageing | [NASA data repository, item 5](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | `data/raw/nasa_pcoe/5. Battery Data Set.zip` |
| KU Leuven BEV energy dynamics V2 | [KU Leuven RDR](https://doi.org/10.48804/8KPDTW) | `data/raw/kuleuven_bev/BEV energy dynamic data_V2.zip` |

Create the destination directories if needed. Keep the original archive filenames.
The exact local archive checksums and sizes are recorded in `data/data_manifest.csv`;
SHA-256 checksums for this release are in `docs/releases/v1.2.0-data-files.sha256`.
The KU Leuven publisher's MD5 is `cc2d63758b411c0f8d4b61bdaa1fff1c`.

After placing the raw archives, install the development dependencies as described in
[CONTRIBUTING.md](../CONTRIBUTING.md), then rebuild:

```bash
bash scripts/lab.sh data
bash scripts/lab.sh calibrate
```

On Windows, with the development environment installed:

```powershell
.\.venv\Scripts\python.exe -m g14lab.data_pipeline
.\.venv\Scripts\python.exe -m g14lab.telemetry
.\.venv\Scripts\python.exe -m g14lab.calibration
```

Rebuilding changes generated outputs; their hashes may differ from the preserved
release baseline because of dependency versions or numerical differences. Keep the
baseline intact when comparing a new model. If a serialized model cannot be loaded
with a newer dependency version, recreate it from the raw data using the calibration step.

## What belongs in Git

Commit source, documentation, samples, and provenance metadata. Raw archives, processed
data, trained models, model metrics, extracted data, local environments, and experiment
exports are ignored. Updated binary artifacts belong in a new tagged release with a
matching checksum manifest. Do not force-add them to bypass the ignore rules.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for attribution and dataset terms.

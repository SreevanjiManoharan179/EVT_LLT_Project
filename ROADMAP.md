# Development roadmap

These are proposed extensions, not implemented features or promised results. Discuss a
proposal in an issue before beginning a large change.

## Getting started

- Add a short tutorial that traces one cell from raw data through calibration to a plot.
- Add unit and sign-convention examples to the data dictionary.
- Improve empty-data messages and document setup problems reported by new users.
- Add focused synthetic-data tests that can run without downloading the full data bundle.

## Model development

- Investigate SOC-dependent resistance and an additional RC branch. Compare against the
  existing 1-RC baseline using held-out cells and report the extra identification cost.
- Explore OCV hysteresis and ageing-dependent OCV curves to address the limitations in the report.
- Validate thermal parameters under additional temperatures and currents before extending
  the stated operating domain.

## Estimation and validation

- Broaden SOH training beyond the four reference cells while preserving cell-level splits.
- Improve interval calibration and measure coverage on held-out cells.
- Compare SOC estimators under additional sensor faults with documented noise assumptions.
- Expand charge validation and explain early termination or CV taper mismatch.

## Community maintenance

- Add a small synthetic CI suite after separating it from data-dependent tests.
- Record a reproducible dependency lock for a validated release environment.
- Automate release-asset packaging and checksums once the manual release layout is established.
- Add live telemetry only with a documented interface, units, timestamp handling, and a clear
  distinction from the current offline replay.

See [README.md](README.md#limitations) for the original model limitations and
[CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

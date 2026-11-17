# Third-party notices

The root MIT licence covers the group's original source code and original documentation.
It does not relicense the datasets, fonts, institutional course materials, or third-party
content embedded in documents.

## NASA battery data

B. Saha and K. Goebel (2007), "Battery Data Set", NASA Prognostics Data Repository,
NASA Ames Research Center, Moffett Field, CA.

- [NASA repository and acknowledgement guidance](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- Archive: `data/raw/nasa_pcoe/5. Battery Data Set.zip`.
- The project's provenance manifest records the source as U.S. Government public data.
  NASA requests acknowledgement of the repository and data donors. Consult the source
  record for its terms; the MIT licence does not apply to this archive.
- The locally preserved archive was originally retrieved through the mirror documented
  in `data/data_manifest.csv`; its checksum is retained for reproducibility.
- Cell samples and processed cell tables derive from these measurements. Parsing,
  current-sign conversion, deduplication, quality flags, and feature extraction are
  described in the manifest and `src/g14lab/data_pipeline.py`.

## KU Leuven vehicle data

Mohamed Aboubacar Yasko, Adamou Moussa Issaka, Fanghao Tian, Hussain Syed Kazmi,
Johan Driesen, and Wilmar Martinez Martinez (2025), "BEV energy dynamics dataset",
KU Leuven RDR, version 2. [Dataset DOI](https://doi.org/10.48804/8KPDTW).

- Licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Archive: `data/raw/kuleuven_bev/BEV energy dynamic data_V2.zip`.
- The raw archive is preserved unchanged. The derived `data/processed/bev_*.parquet`
  tables select and rename signals, omit GPS, fill CAN signals within sessions, and
  compute summaries. These transformations are documented in `data/data_manifest.csv`
  and implemented in `src/g14lab/telemetry.py`.
- Retain this attribution and the licence link when redistributing this data or adaptations.

## Fonts and documents

- Barlow fonts in `static/fonts/` use the SIL Open Font License 1.1. The full licence is
  retained in `static/fonts/OFL-Barlow.txt`.
- The presentation embeds Public Sans and Archivo fonts. Their copyright and licence
  notices remain embedded in the presentation's document properties.
- The institution's brief and front-page template in `docs/course_brief/` retain their
  original authorship and terms. They are included as course context, not relicensed as
  the group's code.
- Other cited or embedded third-party material retains its original terms and attribution.

## Dependencies

Python dependencies keep their respective licences. They are installed separately from
`requirements.txt`; this repository does not distribute a Python environment.

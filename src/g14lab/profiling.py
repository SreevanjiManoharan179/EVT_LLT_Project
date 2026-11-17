"""Column-level data profiling used by the Dataset Explorer."""
from __future__ import annotations

import numpy as np
import pandas as pd


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        rows.append(dict(
            column=c, dtype=str(s.dtype), non_null=int(s.notna().sum()), missing=int(s.isna().sum()),
            missing_pct=round(100 * s.isna().mean(), 2) if n else np.nan, unique=int(s.nunique(dropna=True)),
            min=float(s.min()) if num and s.notna().any() else None,
            median=float(s.median()) if num and s.notna().any() else None,
            max=float(s.max()) if num and s.notna().any() else None,
            example=str(s.dropna().iloc[0])[:40] if s.notna().any() else "",
        ))
    return pd.DataFrame(rows)

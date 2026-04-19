"""
data_loader.py  —  Subsea VFM Pro
===================================
Supports two loading modes:

  1. VOLVE MODE  — load_and_prepare(filepath)
     Ingests the Equinor Volve daily production Excel file exactly as before.
     Sheet: "Daily Production Data"
     All Volve-specific column names, unit conversions, and well geometry
     are handled automatically. Backward-compatible with existing app.py usage.

  2. CLIENT MODE  — load_client_data(filepath, col_map, sheet_name)
     Accepts any Excel or CSV with arbitrary column names and well IDs.
     The caller provides a col_map dict mapping standard internal names
     (e.g. "well", "bhp_bar", "whp_bar") to whatever the client file uses.
     Returns the same (clean_df, ml_data, well_tests, summary) tuple so
     the rest of the app works without changes.

Unit conversions (confirmed from Equinor documentation):
  AVG_DOWNHOLE_PRESSURE  [bar]  → psi  × 14.5038
  AVG_WHP_P              [bar]  → psi  × 14.5038
  AVG_WHT_P              [°C]   → K    + 273.15
  BORE_OIL_VOL           [Sm³]  → STB  × 6.28981
  BORE_GAS_VOL           [Sm³]  → SCF  × 35.3147
  BORE_WAT_VOL           [Sm³]  → STB  × 6.28981
  DP_CHOKE_SIZE          [1/64 inch] — no conversion needed
"""

import pandas as pd
import numpy as np

# ── Unit factors ────────────────────────────────────────────────
BAR_TO_PSI = 14.5038
SM3_TO_STB  = 6.28981
SM3_TO_SCF  = 35.3147

# ── Volve-specific well geometry (from Equinor field reports) ────
WELL_DEPTH_M = {
    "NO 15/9-F-1 C":  2750.0,
    "NO 15/9-F-11 H": 2800.0,
    "NO 15/9-F-12 H": 2800.0,
    "NO 15/9-F-14 H": 2900.0,
    "NO 15/9-F-15 D": 2700.0,
    "NO 15/9-F-5 AH": 2700.0,
}
WELL_API = {
    "NO 15/9-F-1 C":  33.0,
    "NO 15/9-F-11 H": 33.5,
    "NO 15/9-F-12 H": 33.0,
    "NO 15/9-F-14 H": 34.0,
    "NO 15/9-F-15 D": 33.0,
    "NO 15/9-F-5 AH": 33.0,
}
VOLVE_T_RES_K = 373.0   # Reservoir temperature [K] — Volve field

# ── Volve short-name mapping ─────────────────────────────────────
VOLVE_SHORT_NAMES = {
    "NO 15/9-F-14 H": "F-14 H",
    "NO 15/9-F-12 H": "F-12 H",
    "NO 15/9-F-11 H": "F-11 H",
    "NO 15/9-F-15 D": "F-15 D",
    "NO 15/9-F-1 C":  "F-1 C",
    "NO 15/9-F-5 AH": "F-5 AH",
}

# ── Standard internal column names ──────────────────────────────
# These are the names the rest of the pipeline expects after loading.
# For Volve data these are populated automatically.
# For client data the col_map bridges client names → these names.
STANDARD_COLS = [
    "well",          # well identifier string
    "bhp",           # bottomhole pressure [psi]
    "whp",           # wellhead pressure [psi]
    "wht_k",         # wellhead temperature [K]
    "q_oil_stbd",    # oil rate [STB/D]
    "q_gas_scfd",    # gas rate [SCF/D]
    "q_wat_stbd",    # water rate [STB/D]
    "q_liq_stbd",    # liquid rate [STB/D]
    "wc",            # water cut [fraction]
    "gor",           # GOR [SCF/STB]
    "choke_size_64", # choke size [1/64 in]
    "q_true",        # target rate for ML training [STB/D]
    "depth",         # well TVD [m]
    "t_res",         # reservoir temperature [K]
    "DATEPRD",       # date (optional — used for well test sampling)
]


# ════════════════════════════════════════════════════════════════
#  MODE 1: VOLVE (original behaviour — fully backward-compatible)
# ════════════════════════════════════════════════════════════════

def load_volve_excel(filepath) -> pd.DataFrame:
    """
    Load and clean the Volve daily production Excel file.
    Returns a cleaned DataFrame with all derived columns in field units.
    Identical behaviour to the original data_loader.
    """
    raw = pd.read_excel(filepath,
                        sheet_name="Daily Production Data",
                        engine="openpyxl")

    prod = raw[
        (raw["WELL_TYPE"] == "OP") &
        (raw["BORE_OIL_VOL"] > 0) &
        (raw["AVG_WHP_P"]   > 0) &
        (raw["ON_STREAM_HRS"] >= 20)
    ].copy()

    prod["bhp"]           = prod["AVG_DOWNHOLE_PRESSURE"] * BAR_TO_PSI
    prod["whp"]           = prod["AVG_WHP_P"]              * BAR_TO_PSI
    prod["wht_k"]         = prod["AVG_WHT_P"]              + 273.15
    prod["q_oil_stbd"]    = prod["BORE_OIL_VOL"]           * SM3_TO_STB
    prod["q_gas_scfd"]    = prod["BORE_GAS_VOL"]           * SM3_TO_SCF
    prod["q_wat_stbd"]    = prod["BORE_WAT_VOL"]           * SM3_TO_STB
    prod["q_liq_stbd"]    = prod["q_oil_stbd"] + prod["q_wat_stbd"]
    prod["wc"]            = (prod["q_wat_stbd"] /
                             prod["q_liq_stbd"].clip(lower=1)).clip(0.0, 0.999)
    prod["gor"]           = (prod["q_gas_scfd"] /
                             prod["q_oil_stbd"].clip(lower=1)).clip(50.0, 5000.0)
    prod["choke_size_64"] = prod["DP_CHOKE_SIZE"].clip(0, 128)
    prod["q_true"]        = prod["q_liq_stbd"]
    prod["well"]          = prod["WELL_BORE_CODE"]
    prod["depth"]         = prod["well"].map(WELL_DEPTH_M).fillna(2800.0)
    prod["t_res"]         = VOLVE_T_RES_K

    clean = prod[
        (prod["bhp"]   > prod["whp"] + 100) &
        (prod["bhp"]   > 500) &
        (prod["wht_k"] > 273) &
        (prod["q_liq_stbd"] > 100)
    ].copy()

    return clean


def prepare_ml_dataset(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split cleaned data into per-well ML training DataFrames.
    Works for both Volve data (applies short name mapping) and
    client data (uses well names as-is).
    """
    ML_FEATS = ["bhp", "whp", "wht_k", "wc", "gor",
                "choke_size_64", "depth", "t_res", "q_true"]

    out = {}
    for code in clean["well"].unique():
        # Apply Volve short name if available, otherwise use name as-is
        short = VOLVE_SHORT_NAMES.get(code, code)
        w = clean[clean["well"] == code][ML_FEATS].dropna().copy()
        if len(w) >= 50:
            out[short] = w.reset_index(drop=True)
    return out


def prepare_well_tests(clean: pd.DataFrame,
                       sample_every_n: int = 14) -> dict[str, pd.DataFrame]:
    """
    Build per-well well-test DataFrames for AutoCalibrator.
    Works for both Volve and client data.
    If DATEPRD column is absent, samples by index position instead.
    """
    WT_COLS = ["bhp", "whp", "wht_k", "wc", "gor",
               "choke_size_64", "q_liq_stbd"]

    out = {}
    for code in clean["well"].unique():
        short = VOLVE_SHORT_NAMES.get(code, code)
        subset = clean[clean["well"] == code].copy()

        # Sort by date if available, otherwise keep existing order
        if "DATEPRD" in subset.columns:
            subset = subset.sort_values("DATEPRD")

        w = subset.iloc[::sample_every_n][WT_COLS].copy()
        w = w.rename(columns={"q_liq_stbd": "q_liq"})
        w = w[w["bhp"] > w["whp"] + 200].dropna()
        if len(w) >= 3:
            out[short] = w.reset_index(drop=True)
    return out


def field_summary(clean: pd.DataFrame) -> dict:
    """Return a plain-English summary dict for the UI."""
    date_range = "N/A"
    if "DATEPRD" in clean.columns:
        date_range = (f"{clean['DATEPRD'].min().date()} – "
                      f"{clean['DATEPRD'].max().date()}")

    return {
        "total_records":   len(clean),
        "wells":           sorted(clean["well"].unique().tolist()),
        "date_range":      date_range,
        "bhp_range_psi":   (round(clean["bhp"].min(), 0),
                            round(clean["bhp"].max(), 0)),
        "whp_range_psi":   (round(clean["whp"].min(), 0),
                            round(clean["whp"].max(), 0)),
        "wc_range":        (round(clean["wc"].min(), 3),
                            round(clean["wc"].max(), 3)),
        "gor_range_scfstb":(round(clean["gor"].min(), 0),
                            round(clean["gor"].max(), 0)),
        "qliq_range_stbd": (round(clean["q_liq_stbd"].min(), 0),
                            round(clean["q_liq_stbd"].max(), 0)),
        "per_well_counts": clean["well"].value_counts().to_dict(),
    }


def load_and_prepare(filepath) -> tuple:
    """
    One-call convenience for Volve data. Fully backward-compatible.
    Returns (clean_df, ml_datasets, well_tests, summary)
    """
    clean      = load_volve_excel(filepath)
    ml_data    = prepare_ml_dataset(clean)
    well_tests = prepare_well_tests(clean)
    summary    = field_summary(clean)
    return clean, ml_data, well_tests, summary


# ════════════════════════════════════════════════════════════════
#  MODE 2: CLIENT DATA  (new — dynamic column mapping)
# ════════════════════════════════════════════════════════════════

# Default column map for Volve-structured files
# Keys = internal standard names, Values = expected client column names
DEFAULT_CLIENT_COL_MAP = {
    "well":           "well",
    "bhp_psi":        "bhp_psi",       # already in psi
    "whp_psi":        "whp_psi",       # already in psi
    "wht_k":          "wht_k",         # already in K
    "q_oil_stbd":     "q_oil_stbd",
    "q_gas_scfd":     "q_gas_scfd",
    "q_wat_stbd":     "q_wat_stbd",
    "choke_size_64":  "choke_size_64",
    "depth_m":        None,            # optional
    "t_res_k":        None,            # optional
    "date":           None,            # optional
}


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect likely column mappings from a client DataFrame by
    fuzzy-matching common naming conventions.
    Returns a suggested col_map dict {internal_name: detected_col}.
    The UI should show this to the user for confirmation/correction.
    """
    cols = {c.lower().strip(): c for c in df.columns}

    # Patterns: internal_name → list of candidate substrings (lowercase)
    PATTERNS = {
        "well":          ["well", "wellname", "well_name", "well_id",
                          "wellbore", "bore", "uwi"],
        "bhp_psi":       ["bhp", "downhole_pressure", "bottomhole",
                          "avg_downhole", "pdh"],
        "whp_psi":       ["whp", "wellhead_pressure", "tubing_head",
                          "thp", "avg_whp"],
        "wht_k":         ["wht", "wellhead_temp", "temperature",
                          "avg_wht", "t_wellhead"],
        "q_oil_stbd":    ["oil_vol", "oil_rate", "bore_oil", "q_oil",
                          "oil_prod", "oil"],
        "q_gas_scfd":    ["gas_vol", "gas_rate", "bore_gas", "q_gas",
                          "gas_prod", "gas"],
        "q_wat_stbd":    ["wat_vol", "water_vol", "water_rate",
                          "bore_wat", "q_wat", "water"],
        "choke_size_64": ["choke", "choke_size", "dp_choke", "bean"],
        "depth_m":       ["depth", "tvd", "true_vert"],
        "t_res_k":       ["t_res", "res_temp", "reservoir_temp"],
        "date":          ["date", "dateprd", "timestamp", "time", "day"],
    }

    detected = {}
    for internal, patterns in PATTERNS.items():
        for p in patterns:
            match = next((orig for low, orig in cols.items() if p in low), None)
            if match:
                detected[internal] = match
                break
        else:
            detected[internal] = None

    return detected


def _infer_pressure_units(series: pd.Series) -> str:
    """
    Guess whether a pressure column is in bar or psi based on value range.
    Typical wellhead: 50–300 bar / 700–4500 psi
    Typical BHP: 200–600 bar / 3000–8700 psi
    """
    median = series.median()
    if median < 500:
        return "bar"
    return "psi"


def _infer_temperature_units(series: pd.Series) -> str:
    """Guess whether temperature is in °C or K."""
    median = series.median()
    return "K" if median > 200 else "C"


def _infer_volume_units(series: pd.Series, col_type: str) -> str:
    """Guess whether volumes are in Sm³ or STB/SCF."""
    median = series.median()
    # Sm³ values for oil typically 100–2000; STB 600–12000
    if col_type in ("oil", "water") and median < 2000:
        return "sm3"
    if col_type == "gas" and median < 200000:
        return "sm3"
    return "field"


def load_client_data(
    filepath,
    col_map: dict,
    sheet_name: str = 0,
    pressure_unit: str = "auto",   # "psi", "bar", or "auto"
    temp_unit: str = "auto",       # "K", "C", or "auto"
    volume_unit: str = "auto",     # "field" (STB/SCF), "sm3", or "auto"
    default_depth_m: float = 2800.0,
    default_t_res_k: float = 373.0,
    min_records_per_well: int = 50,
) -> tuple:
    """
    Load client production data with arbitrary column names and units.

    Parameters
    ----------
    filepath     : path or file-like object (Excel or CSV)
    col_map      : dict mapping internal names → client column names.
                   Use detect_columns() to generate a starting suggestion.
                   Required keys: "well", "bhp_psi"/"bhp_bar", "whp_psi"/"whp_bar",
                                  "wht_k"/"wht_c", "q_oil_stbd"/"q_oil_sm3",
                                  "q_gas_scfd"/"q_gas_sm3", "q_wat_stbd"/"q_wat_sm3"
                   Optional keys: "choke_size_64", "depth_m", "t_res_k", "date"
    sheet_name   : for Excel files, sheet name or index (default 0 = first sheet)
    pressure_unit: "psi", "bar", or "auto" (auto-detects from value range)
    temp_unit    : "K", "C", or "auto"
    volume_unit  : "field" (STB/SCF), "sm3", or "auto"
    default_depth_m   : fallback depth [m] for wells not in col_map
    default_t_res_k   : fallback reservoir temperature [K]
    min_records_per_well: minimum records to include a well in ML training

    Returns
    -------
    (clean_df, ml_datasets, well_tests, summary)  — same tuple as load_and_prepare()
    """
    # ── Load file ────────────────────────────────────────────────
    fname = getattr(filepath, "name", str(filepath))
    if fname.endswith(".csv"):
        raw = pd.read_csv(filepath)
    else:
        raw = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")

    # ── Map columns ──────────────────────────────────────────────
    prod = pd.DataFrame()
    prod["well"] = raw[col_map["well"]].astype(str).str.strip()

    # Pressure
    bhp_col = col_map.get("bhp_psi") or col_map.get("bhp_bar")
    whp_col = col_map.get("whp_psi") or col_map.get("whp_bar")
    if not bhp_col or not whp_col:
        raise ValueError("col_map must contain 'bhp_psi' (or 'bhp_bar') "
                         "and 'whp_psi' (or 'whp_bar')")

    bhp_raw = raw[bhp_col].astype(float)
    whp_raw = raw[whp_col].astype(float)

    # Auto-detect units
    pu = pressure_unit
    if pu == "auto":
        pu = _infer_pressure_units(bhp_raw)

    if pu == "bar":
        prod["bhp"] = bhp_raw * BAR_TO_PSI
        prod["whp"] = whp_raw * BAR_TO_PSI
    else:
        prod["bhp"] = bhp_raw
        prod["whp"] = whp_raw

    # Temperature
    wht_col = col_map.get("wht_k") or col_map.get("wht_c")
    if wht_col:
        wht_raw = raw[wht_col].astype(float)
        tu = temp_unit
        if tu == "auto":
            tu = _infer_temperature_units(wht_raw)
        prod["wht_k"] = wht_raw if tu == "K" else wht_raw + 273.15
    else:
        prod["wht_k"] = 348.15  # fallback 75°C

    # Volumes
    oil_col = col_map.get("q_oil_stbd") or col_map.get("q_oil_sm3")
    gas_col = col_map.get("q_gas_scfd") or col_map.get("q_gas_sm3")
    wat_col = col_map.get("q_wat_stbd") or col_map.get("q_wat_sm3")

    if not oil_col:
        raise ValueError("col_map must contain 'q_oil_stbd' or 'q_oil_sm3'")

    oil_raw = raw[oil_col].astype(float)
    gas_raw = raw[gas_col].astype(float) if gas_col else pd.Series(
        np.zeros(len(raw)), index=raw.index)
    wat_raw = raw[wat_col].astype(float) if wat_col else pd.Series(
        np.zeros(len(raw)), index=raw.index)

    vu = volume_unit
    if vu == "auto":
        vu = _infer_volume_units(oil_raw, "oil")

    if vu == "sm3":
        prod["q_oil_stbd"] = oil_raw * SM3_TO_STB
        prod["q_gas_scfd"] = gas_raw * SM3_TO_SCF
        prod["q_wat_stbd"] = wat_raw * SM3_TO_STB
    else:
        prod["q_oil_stbd"] = oil_raw
        prod["q_gas_scfd"] = gas_raw
        prod["q_wat_stbd"] = wat_raw

    # Derived
    prod["q_liq_stbd"] = prod["q_oil_stbd"] + prod["q_wat_stbd"]
    prod["wc"]  = (prod["q_wat_stbd"] /
                   prod["q_liq_stbd"].clip(lower=1)).clip(0.0, 0.999)
    prod["gor"] = (prod["q_gas_scfd"] /
                   prod["q_oil_stbd"].clip(lower=1)).clip(50.0, 5000.0)
    prod["q_true"] = prod["q_liq_stbd"]

    # Choke
    choke_col = col_map.get("choke_size_64")
    prod["choke_size_64"] = (raw[choke_col].clip(0, 128).astype(float)
                             if choke_col else 12.0)

    # Depth & reservoir temperature (per-well overrides via col_map, else defaults)
    depth_col  = col_map.get("depth_m")
    t_res_col  = col_map.get("t_res_k")
    prod["depth"] = (raw[depth_col].astype(float)
                     if depth_col else default_depth_m)
    prod["t_res"] = (raw[t_res_col].astype(float)
                     if t_res_col else default_t_res_k)

    # Date (optional)
    date_col = col_map.get("date")
    if date_col:
        prod["DATEPRD"] = pd.to_datetime(raw[date_col], errors="coerce")

    # ── Quality filter ────────────────────────────────────────────
    clean = prod[
        (prod["bhp"]        > prod["whp"] + 100) &
        (prod["bhp"]        > 500) &
        (prod["wht_k"]      > 273) &
        (prod["q_liq_stbd"] > 10) &       # relaxed from 100 for smaller fields
        (prod["q_oil_stbd"] > 0)
    ].copy().reset_index(drop=True)

    # ── Build outputs using shared functions ──────────────────────
    ml_data    = prepare_ml_dataset(clean)
    well_tests = prepare_well_tests(clean)
    summary    = field_summary(clean)

    return clean, ml_data, well_tests, summary


def get_well_configs_from_data(clean: pd.DataFrame) -> dict:
    """
    Auto-generate WELL_CONFIGS dict from loaded client data.
    Uses median values per well as defaults for the Live VFM inputs.
    This replaces the hardcoded WELL_CONFIGS in app.py for client data.
    """
    configs = {}
    for code in clean["well"].unique():
        short = VOLVE_SHORT_NAMES.get(code, code)
        w = clean[clean["well"] == code]
        configs[short] = {
            "depth":               float(w["depth"].median()),
            "diameter":            0.127,          # default 5" tubing
            "t_res":               float(w["t_res"].median()),
            "t_sea":               278.0,           # default North Sea
            "api":                 33.0,            # default
            "sg_gas":              0.72,            # default
            "productivity_index":  15.0,            # default
            "wc":                  float(w["wc"].median()),
            "gor":                 float(w["gor"].median()),
            "rho_brine":           1028.0,
            "choke_size_64":       float(w["choke_size_64"].median()),
        }
    return configs

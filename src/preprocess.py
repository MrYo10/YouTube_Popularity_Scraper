from __future__ import annotations

import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil import parser as dtp

IN_SCRAPED = Path("data/raw/scraped/scraped_flat.csv")
IN_API     = Path("data/raw/api/api_videos.csv")
OUT_DIR    = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def parse_duration_iso8601(s: str) -> float:
    """Convert ISO8601 duration 'PT#H#M#S' to minutes (float)."""
    if not isinstance(s, str):
        return np.nan
    hr = mn = sc = 0
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", s)
    if not m:
        return np.nan
    if m.group(1): hr = int(m.group(1))
    if m.group(2): mn = int(m.group(2))
    if m.group(3): sc = int(m.group(3))
    return hr * 60 + mn + sc / 60.0

def add_time_features(df: pd.DataFrame, col: str = "publish_dt") -> pd.DataFrame:
    # Ensure datetime.
    if col not in df.columns:
        df[col] = pd.NaT
    dt = pd.to_datetime(df[col], errors="coerce", utc=True)
    df[col] = dt
    df["upload_year"]  = dt.dt.year
    df["upload_month"] = dt.dt.month
    df["upload_dow"]   = dt.dt.dayofweek
    df["upload_hour"]  = dt.dt.hour
    return df

def _series_or_blank(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a string Series for column `col`, defaulting to a blank Series aligned to df.index."""
    if col in df.columns:
        s = df[col]
    else:
        s = pd.Series([""] * len(df), index=df.index, dtype=object)
    return s.astype(str).fillna("").str.strip()

def common_text_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["title"] = _series_or_blank(df, "title")
    df["description"] = _series_or_blank(df, "description")

    tags = df["tags"] if "tags" in df.columns else pd.Series([""] * len(df), index=df.index)
    tags = tags.fillna("").astype(str)
    df["tags"] = tags  # keep normalized copy for later
    df["tags_count"] = tags.map(lambda s: 0 if s == "" else len(s.split("|")))

    df["desc_length"] = df["description"].str.len()
    df["has_links_in_desc"] = df["description"].str.contains(r"http[s]?://", regex=True, na=False).astype(int)
    return df

# ----------YouTube ID extraction ----------
_ID_RX = re.compile(r"(?:v=|\/shorts\/|youtu\.be\/|\/embed\/|watch\/|\/v\/)([A-Za-z0-9_-]{11})")
def _extract_id_from_text(s: str):
    if not isinstance(s, str):
        return None
    m = _ID_RX.search(s)
    return m.group(1) if m else None

# ---------- scraped ----------
def _robust_views(df: pd.DataFrame) -> pd.Series:
    """
    Try to find a 'views' column reliably:
    - prefer 'views' if present
    - else any column whose name contains 'view'
    - else try to autodetect a numeric column with largest median
    """
    if "views" in df.columns:
        return pd.to_numeric(df["views"], errors="coerce")

    cand_cols = [c for c in df.columns if "view" in c.lower()]
    for c in cand_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(0.01 * len(df))):
            return s

    best_s, best_med = None, -1
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        med = float(s.median())
        if med > best_med:
            best_med, best_s = med, s
    return best_s if best_s is not None else pd.Series([np.nan] * len(df), index=df.index)

def preprocess_scraped(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    before = len(df)

    # Trim column names
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # Ensure video_id
    if "video_id" not in df.columns:
        if "url" in df.columns:
            df["video_id"] = df["url"].astype(str).apply(_extract_id_from_text)
        else:
            df["video_id"] = None
            str_cols = [c for c in df.columns if df[c].dtype == "object"]
            for c in str_cols:
                mask = df["video_id"].isna()
                df.loc[mask, "video_id"] = df.loc[mask, c].astype(str).apply(_extract_id_from_text)

    df = df[df["video_id"].notna()].drop_duplicates(subset=["video_id"])

    # Clean text and simple derived text feats
    for c in ["title", "uploader", "description", "category", "tags"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df = common_text_feats(df)

    # Dates & duration
    df["publish_dt"] = pd.to_datetime(df.get("publish_date"), errors="coerce", utc=True)
    if "duration_seconds" in df.columns:
        df["duration_min"] = pd.to_numeric(df["duration_seconds"], errors="coerce") / 60.0
    elif "duration_min" not in df.columns:
        df["duration_min"] = np.nan

    # Robust views/likes/comments
    df["viewCount"]    = _robust_views(df)
    df["likeCount"]    = pd.to_numeric(df.get("likes"), errors="coerce")
    df["commentCount"] = pd.to_numeric(df.get("comments"), errors="coerce")

    # Engagement
    denom = df["viewCount"].replace({0: np.nan})
    df["engagement_rate"] = (df["likeCount"].fillna(0) + df["commentCount"].fillna(0)) / denom
    df["engagement_rate"] = df["engagement_rate"].replace([np.inf, -np.inf], np.nan)

    # Time features
    df = add_time_features(df, col="publish_dt")
    ref = pd.Timestamp.utcnow()  # UTC
    df["time_since_upload_days"] = (ref - df["publish_dt"]).dt.days

    # Clip views for stability
    if df["viewCount"].notna().any():
        vq = df["viewCount"].quantile(0.998)
        df["viewCount_clipped"] = df["viewCount"].clip(upper=vq)
    else:
        df["viewCount_clipped"] = df["viewCount"]

    # Per-day views
    days = df["time_since_upload_days"].clip(lower=1)
    df["views_per_day"] = df["viewCount"] / days

    # Flags
    df["has_target_views"] = df["viewCount"].notna() & (df["viewCount"] > 0)

    after = len(df)
    return df, before, after

# ---------- api ----------
def preprocess_api(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["viewCount", "likeCount", "commentCount", "channel_subscribers", "channel_view_count", "channel_video_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df["publish_dt"] = pd.to_datetime(df.get("publish_date", pd.Series([pd.NaT] * len(df))),
                                      errors="coerce", utc=True)
    ref = pd.Timestamp.utcnow()  # tz-aware UTC
    df["time_since_upload_days"] = (ref - df["publish_dt"]).dt.days

    # Duration from ISO8601 -> minutes
    df["duration_min"] = df.get("duration_iso8601", pd.Series([np.nan] * len(df))).apply(parse_duration_iso8601)

    df = common_text_feats(df)

    denom = df["viewCount"].replace({0: np.nan})
    df["engagement_rate"] = (df["likeCount"].fillna(0) + df["commentCount"].fillna(0)) / denom
    df["engagement_rate"] = df["engagement_rate"].replace([np.inf, -np.inf], np.nan)

    if "video_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["video_id"])
        after = len(df)
        print(f"Dedup by video_id: {before} -> {after}")

    # keep only rows with usable targets
    df = df[(df["viewCount"].notna()) & (df["viewCount"] > 0)]

    vq = df["viewCount"].quantile(0.999)
    df["viewCount_clipped"] = df["viewCount"].clip(upper=vq) if pd.notna(vq) and vq > 0 else df["viewCount"]

    # views/day (handy for API-side too)
    days = df["time_since_upload_days"].clip(lower=1)
    df["views_per_day"] = df["viewCount"] / days

    df = add_time_features(df, col="publish_dt")
    return df

# ---------- main ----------
def main():
    # Read scraped (permissive: skip bad lines)
    if IN_SCRAPED.exists():
        try:
            df_scraped_raw = pd.read_csv(IN_SCRAPED, engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"Warning: failed to read {IN_SCRAPED}: {e}", file=sys.stderr)
            df_scraped_raw = pd.DataFrame()
    else:
        df_scraped_raw = pd.DataFrame()

    # Read API (required)
    try:
        df_api_raw = pd.read_csv(IN_API)
    except Exception as e:
        print(f"Error: failed to read {IN_API}: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Scraped ----
    if not df_scraped_raw.empty:
        scraped, s_before, s_after = preprocess_scraped(df_scraped_raw)
        scraped_out = OUT_DIR / "scraped_preprocessed.csv"
        scraped.to_csv(scraped_out, index=False)
        n_targets = int(scraped["has_target_views"].sum()) if "has_target_views" in scraped.columns else 0
        print(f"SCRAPED rows: {s_before} -> {s_after} after cleaning")
        print(f"SCRAPED with target views: {n_targets}/{s_after}")
        print(f"Saved: {scraped_out}")
    else:
        print("SCRAPED: no rows found (skip).")

    # ---- API ----
    a_before = len(df_api_raw)
    api = preprocess_api(df_api_raw)
    a_after  = len(api)
    api_out = OUT_DIR / "api_preprocessed.csv"
    api.to_csv(api_out, index=False)

    miss_eng = int(api["engagement_rate"].isna().sum())
    print("Saved interim preprocessed datasets.")
    print(f"API rows: {a_before} -> {a_after} after cleaning")
    print(f"Missing engagement_rate (API): {miss_eng}")
    print(f"Saved: {api_out}")

if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

IN_SCRAPED = Path("data/interim/scraped_preprocessed.csv")
IN_API     = Path("data/interim/api_preprocessed.csv")
OUT_DIR    = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TITLE_VOCAB = 3000
NGRAM_RANGE = (1, 2)

def ensure_cols(df: pd.DataFrame, cols: list[str], fill=0):
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

def safe_title_vectorize(text_series: pd.Series, max_features: int = TITLE_VOCAB, ngram_range=NGRAM_RANGE):
    """Return a DataFrame of n-grams; never crashes on empty."""
    ts = text_series.fillna("").astype(str).str.strip()
    if (ts.str.contains(r"\w", regex=True)).sum() == 0:
        return pd.DataFrame(index=range(len(ts))), []
    vec = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1
    )
    X = vec.fit_transform(ts)
    cols = [f"title__{t}" for t in vec.get_feature_names_out()]
    return pd.DataFrame(X.toarray(), columns=cols), cols

# ---------------- API features ----------------
def build_api_features(df_api: pd.DataFrame) -> pd.DataFrame:
    # Combine title+desc+tags for signal
    joined_text = (
        df_api.get("title", pd.Series([""]*len(df_api))).astype(str) + " " +
        df_api.get("description", pd.Series([""]*len(df_api))).astype(str) + " " +
        df_api.get("tags", pd.Series([""]*len(df_api))).astype(str)
    )
    X_title, title_cols = safe_title_vectorize(joined_text)

    base_cols = [
        "duration_min", "time_since_upload_days",
        "upload_year", "upload_month", "upload_dow", "upload_hour",
        "tags_count", "desc_length", "has_links_in_desc",
        "channel_subscribers", "channel_view_count", "channel_video_count",
        "views_per_day",
    ]
    df_api = ensure_cols(df_api, base_cols, fill=np.nan)

    # Targets
    df_api["target_views"] = df_api["viewCount_clipped"] if "viewCount_clipped" in df_api else df_api["viewCount"]
    df_api["target_engagement"] = df_api.get("engagement_rate", pd.Series([np.nan]*len(df_api))).replace([np.inf, -np.inf], np.nan)
    df_api["target_views_log"] = np.log1p(pd.to_numeric(df_api["target_views"], errors="coerce"))
    df_api["target_views_per_day"] = pd.to_numeric(df_api.get("views_per_day"), errors="coerce")
    df_api["target_views_per_day_log"] = np.log1p(df_api["target_views_per_day"])

    id_cols = [c for c in ["video_id", "channel_id", "category_id", "publish_date"] if c in df_api.columns]
    features = df_api[id_cols + base_cols + [
        "target_views","target_views_log","target_views_per_day","target_views_per_day_log","target_engagement"
    ]].copy()

    num_cols = [c for c in base_cols if c in features.columns]
    for c in num_cols:
        features[c] = pd.to_numeric(features[c], errors="coerce")
    if num_cols:
        features[num_cols] = features[num_cols].fillna(features[num_cols].median())

    feat = pd.concat([features.reset_index(drop=True), X_title.reset_index(drop=True)], axis=1)
    return feat

# ---------------- Scraped features ----------------
def build_scraped_features(df_s: pd.DataFrame) -> pd.DataFrame:
    joined_text = (
        df_s.get("title", pd.Series([""]*len(df_s))).astype(str) + " " +
        df_s.get("description", pd.Series([""]*len(df_s))).astype(str) + " " +
        df_s.get("tags", pd.Series([""]*len(df_s))).astype(str)
    )
    X_title, title_cols = safe_title_vectorize(joined_text)

    base_cols = [
        "duration_min", "time_since_upload_days",
        "upload_year", "upload_month", "upload_dow", "upload_hour",
        "tags_count", "desc_length", "has_links_in_desc",
        "views_per_day",
    ]
    df_s = ensure_cols(df_s, base_cols, fill=np.nan)

    # Targets (keep NaN if missing)
    if "viewCount_clipped" in df_s:
        df_s["target_views"] = df_s["viewCount_clipped"]
    else:
        df_s["target_views"] = df_s.get("viewCount", pd.Series([np.nan]*len(df_s)))

    df_s["target_engagement"] = (
        df_s.get("engagement_rate", pd.Series([np.nan]*len(df_s)))
          .replace([np.inf, -np.inf], np.nan)
    )
    df_s["target_views_log"] = np.log1p(pd.to_numeric(df_s["target_views"], errors="coerce"))
    df_s["target_views_per_day"] = pd.to_numeric(df_s.get("views_per_day"), errors="coerce")
    df_s["target_views_per_day_log"] = np.log1p(df_s["target_views_per_day"])

    id_cols = [c for c in ["video_id", "category", "publish_date"] if c in df_s.columns]
    features = df_s[id_cols + base_cols + [
        "target_views","target_views_log","target_views_per_day","target_views_per_day_log","target_engagement"
    ]].copy()

    num_cols = [c for c in base_cols if c in features.columns]
    for c in num_cols:
        features[c] = pd.to_numeric(features[c], errors="coerce")
    if num_cols:
        features[num_cols] = features[num_cols].fillna(features[num_cols].median())

    feat = pd.concat([features.reset_index(drop=True), X_title.reset_index(drop=True)], axis=1)
    return feat

def main():
    # API features
    df_api = pd.read_csv(IN_API)
    api_feat = build_api_features(df_api)
    api_out = OUT_DIR / "api_features.csv"
    api_feat.to_csv(api_out, index=False)
    print(f"Wrote {len(api_feat):,} rows → {api_out}")

    # Scraped features (if exists)
    if IN_SCRAPED.exists():
        df_scraped = pd.read_csv(IN_SCRAPED)
        if df_scraped.empty:
            (OUT_DIR / "scraped_features.csv").write_text("")  # create empty
            print("Scraped preprocessed is empty; wrote empty scraped_features.csv")
        else:
            scraped_feat = build_scraped_features(df_scraped)
            scraped_out = OUT_DIR / "scraped_features.csv"
            scraped_feat.to_csv(scraped_out, index=False)
            print(f"Wrote {len(scraped_feat):,} rows → {scraped_out}")
    else:
        print("No scraped_preprocessed.csv found; skipped scraped features.")

if __name__ == "__main__":
    main()

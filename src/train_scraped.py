from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

IN = Path("data/processed/scraped_features.csv")
OUT_MODELS  = Path("models");  OUT_MODELS.mkdir(exist_ok=True)
OUT_REPORTS = Path("reports"); OUT_REPORTS.mkdir(exist_ok=True)

# Prefer the more learnable targets first
PREFERRED_TARGETS = [
    "target_views_per_day_log",
    "target_views_log",
    "target_views",
    "target_engagement",
]
MIN_ROWS = 300
MIN_VAR  = 1e-8

def pick_target(df: pd.DataFrame, forced: str|None) -> str:
    if forced:
        if forced not in df.columns:
            raise SystemExit(f"--target {forced} not found in columns")
        s = pd.to_numeric(df[forced], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() < MIN_ROWS or float(np.nanvar(s)) < MIN_VAR:
            raise SystemExit(f"--target {forced} has too few rows or near-zero variance")
        return forced

    for t in PREFERRED_TARGETS:
        if t not in df.columns:
            continue
        s = pd.to_numeric(df[t], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() >= MIN_ROWS and float(np.nanvar(s)) >= MIN_VAR:
            return t
    raise SystemExit(f"No viable target found. Checked {PREFERRED_TARGETS} with MIN_ROWS={MIN_ROWS}, MIN_VAR={MIN_VAR}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=PREFERRED_TARGETS, help="force target column")
    args = ap.parse_args()

    if not IN.exists() or IN.stat().st_size == 0:
        raise SystemExit("scraped_features.csv missing or empty. Run preprocess + features first.")

    df = pd.read_csv(IN)
    if df.empty:
        raise SystemExit("scraped_features.csv has 0 rows.")

    target = pick_target(df, args.target)
    print(f"Using target: {target}")

    y = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = y.notna()
    print(f"Rows with '{target}': {int(keep.sum())}/{len(df)}")
    df = df.loc[keep].copy()
    y = y.loc[keep].values

    # base engineered features
    base_cols = [
        "duration_min", "time_since_upload_days",
        "upload_year", "upload_month", "upload_dow", "upload_hour",
        "tags_count", "desc_length", "has_links_in_desc",
        "views_per_day",
    ]
    have_base = [c for c in base_cols if c in df.columns]
    title_cols = [c for c in df.columns if c.startswith("title__")]
    feat_cols = have_base + title_cols
    if not feat_cols:
        raise SystemExit("No feature columns found (base or title__). Did features.py run?")

    X = df[feat_cols].copy()
    for c in have_base:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if have_base:
        X[have_base] = X[have_base].fillna(X[have_base].median())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)  # no 'squared' arg for older sklearn
        rmse = float(np.sqrt(mse))
        return {"R2": float(r2_score(y_true, y_pred)),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": rmse}

    res = {
        "model": "RandomForestRegressor",
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "target": target,
        "train": metrics(y_tr, rf.predict(X_tr)),
        "test": metrics(y_te, rf.predict(X_te)),
        "features_used": feat_cols[:10] + (["..."] if len(feat_cols) > 10 else []),
    }

    model_path = OUT_MODELS / f"rf_scraped_{target}.joblib"
    joblib.dump(rf, model_path)
    with open(OUT_MODELS / f"scraped_{target}_metrics.json", "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== RF on scraped data ===")
    print(f"Target: {target}")
    print(f"[Train] R2={res['train']['R2']:.4f} | MAE={res['train']['MAE']:.6f} | RMSE={res['train']['RMSE']:.6f}")
    print(f"[Test ] R2={res['test']['R2']:.4f} | MAE={res['test']['MAE']:.6f} | RMSE={res['test']['RMSE']:.6f}")
    print(f"Saved model → {model_path}")

if __name__ == "__main__":
    main()

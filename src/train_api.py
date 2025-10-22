from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DEFAULT = Path("data/processed/api_features.csv")
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_NUMERIC = [
    "duration_min", "time_since_upload_days",
    "upload_year", "upload_month", "upload_dow", "upload_hour",
    "tags_count", "desc_length", "has_links_in_desc",
    "channel_subscribers", "channel_view_count", "channel_video_count"
]

IGNORE_COLS = {"video_id", "channel_id", "category_id", "publish_date", "target_views", "target_engagement"}

def parse_args():
    ap = argparse.ArgumentParser(description="Train regressors on API features.")
    ap.add_argument("--in", dest="in_path", type=str, default=str(DATA_DEFAULT), help="Features CSV path.")
    ap.add_argument("--target", choices=["engagement", "views"], default="engagement", help="Which target to predict.")
    ap.add_argument("--model", choices=["rf", "xgb", "both"], default="both", help="Model(s) to train.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return ap.parse_args()

def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"❌ Features file not found: {path}. Run preprocess + features first.")
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all")
    return df

def pick_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # title n-grams + numeric bases
    title_cols = [c for c in df.columns if c.startswith("title__")]
    cols = [c for c in BASE_NUMERIC if c in df.columns] + title_cols
    X = df[cols].copy()
    # Fill numerics with medians (robust); title counts with 0
    num_cols = [c for c in BASE_NUMERIC if c in X.columns]
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
    if num_cols:
        med = X[num_cols].median()
        X[num_cols] = X[num_cols].fillna(med)
    X[title_cols] = X[title_cols].fillna(0)
    return X, cols

def evaluate(y_true, y_pred, prefix: str):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"[{prefix}] R2={r2:.4f} | MAE={mae:.6f} | RMSE={rmse:.6f}")
    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)}

def plot_importance(model, feat_cols, out_png: Path, title: str, top_k: int = 25):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    names = np.array(feat_cols)[order]
    vals = importances[order]
    plt.figure(figsize=(8, max(4, int(top_k * 0.35))))
    plt.barh(range(len(vals))[::-1], vals[::-1])
    plt.yticks(range(len(vals))[::-1], names[::-1], fontsize=8)
    plt.xlabel("Importance"); plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    print(f"Saved feature importance → {out_png}")

def train_rf(X_tr, y_tr, seed: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=500, max_depth=None, n_jobs=-1, random_state=seed
    )
    model.fit(X_tr, y_tr)
    return model

def train_xgb(X_tr, y_tr, seed: int):
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None
    model = XGBRegressor(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=seed,
        tree_method="hist"
    )
    model.fit(X_tr, y_tr)
    return model

def main():
    args = parse_args()
    in_path = Path(args.in_path)
    df = load_features(in_path)

    # choose target
    if args.target == "engagement":
        target_col = "target_engagement"
        log_target = False
    else:
        target_col = "target_views"
        log_target = True   # views are highly skewed

    # build X, y
    X, feat_cols = pick_columns(df)
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna() & np.isfinite(y)
    X = X.loc[mask]; y = y.loc[mask]

    # log transform for views
    if log_target:
        y_trn = np.log1p(y.clip(lower=0))
    else:
        y_trn = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_trn, test_size=args.test_size, random_state=args.random_state
    )

    to_train = []
    if args.model in ("rf", "both"): to_train.append(("rf", train_rf))
    if args.model in ("xgb", "both"): to_train.append(("xgb", train_xgb))

    results = {}
    for name, trainer in to_train:
        model = trainer(X_train, y_train, args.random_state)
        if model is None:
            print("xgboost not installed; skipping XGB.")
            continue

        pred_train = model.predict(X_train)
        pred_test  = model.predict(X_test)

        # inverse transform
        if log_target:
            y_train_report = np.expm1(y_train)
            y_test_report  = np.expm1(y_test)
            pred_train_rep = np.expm1(pred_train)
            pred_test_rep  = np.expm1(pred_test)
        else:
            y_train_report, y_test_report = y_train, y_test
            pred_train_rep, pred_test_rep = pred_train, pred_test

        print(f"\n=== {name.upper()} on {target_col} ===")
        res = {
            "train": evaluate(y_train_report, pred_train_rep, "Train"),
            "test":  evaluate(y_test_report,  pred_test_rep,  "Test"),
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "features_used": feat_cols
        }
        results[name] = res

        # save model
        model_path = MODELS_DIR / f"{name}_api_{target_col}.joblib"
        joblib.dump(model, model_path); print(f"Saved model → {model_path}")

        # importance plot + CSV
        imp_png = REPORTS_DIR / f"feature_importance_{name}_{target_col}.png"
        plot_importance(model, feat_cols, imp_png, f"{name.upper()} — {target_col}")

        if hasattr(model, "feature_importances_"):
            pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_})\
                .sort_values("importance", ascending=False)\
                .to_csv(MODELS_DIR / f"{name}_{target_col}_feature_importance.csv", index=False)

        # save predictions for inspection
        pred_out = REPORTS_DIR / f"predictions_{name}_{target_col}.csv"
        pd.DataFrame({
            "video_id": df.loc[mask, "video_id"].iloc[X_test.index],
            "y_true": y_test_report,
            "y_pred": pred_test_rep
        }).to_csv(pred_out, index=False)
        print(f"Saved predictions → {pred_out}")

    metrics_path = MODELS_DIR / f"api_{target_col}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

if __name__ == "__main__":
    main()

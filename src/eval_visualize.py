from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = Path("models")
PROC_DIR   = Path("data/processed")
REPORTS    = Path("reports"); REPORTS.mkdir(exist_ok=True)

PREFERRED_SCRAPED = [
    "scraped_target_views_per_day_log_metrics.json",
    "scraped_target_views_log_metrics.json",
    "scraped_target_views_metrics.json",
    "scraped_target_engagement_metrics.json",
]

def load_api_metrics() -> dict | None:
    p = MODELS_DIR / "api_target_engagement_metrics.json"
    if not p.exists():
        print(f"[WARN] Missing {p}; did you run train_api.py?")
        return None
    with open(p) as f:
        return json.load(f)

def find_scraped_metrics() -> tuple[Path | None, dict | None]:
    # try preferred order first
    for name in PREFERRED_SCRAPED:
        p = MODELS_DIR / name
        if p.exists():
            with open(p) as f:
                return p, json.load(f)

    # else, fall back to any scraped_*_metrics.json
    cands = sorted(MODELS_DIR.glob("scraped_*_metrics.json"))
    if not cands:
        print("[WARN] No scraped metrics JSON found in models/.")
        return None, None
    p = cands[-1]
    with open(p) as f:
        return p, json.load(f)

def get_test_rmse_from_api(api: dict) -> dict:
    out = {}
    if "rf" in api:
        out["RF (API-engagement)"] = api["rf"]["test"]["RMSE"]
    if "xgb" in api:
        out["XGB (API-engagement)"] = api["xgb"]["test"]["RMSE"]
    return out

def get_test_rmse_from_scraped(scraped_json: dict, scraped_fname: str) -> dict:
    # scraped layout (from our trainer): flat dict with "test": {RMSE, ...}, "target": "<name>"
    target = scraped_json.get("target", re.sub(r"^scraped_(.+)_metrics\.json$", r"\1", scraped_fname))
    rmse = scraped_json.get("test", {}).get("RMSE")
    if rmse is None:
        return {}
    nice = {
        "target_views_per_day_log": "views/day (log)",
        "target_views_log": "views (log)",
        "target_views": "views",
        "target_engagement": "engagement",
    }.get(target, target)
    return {f"RF (Scraped-{nice})": rmse}

def plot_bar(d: dict[str, float], title: str, out_png: Path):
    if not d:
        print(f"[WARN] Nothing to plot for {title}")
        return
    names = list(d.keys())
    vals  = [d[k] for k in names]
    plt.figure()
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel("RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[OK] Saved {out_png}")

def plot_feature_importance(csv_path: Path, title: str, out_png: Path, topk: int = 20):
    if not csv_path.exists():
        print(f"[WARN] Missing FI CSV: {csv_path} (skipping {title})")
        return
    df = pd.read_csv(csv_path).sort_values("importance", ascending=False).head(topk)
    plt.figure()
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[OK] Saved {out_png}")

def plot_api_trends(api_features_csv: Path):
    if not api_features_csv.exists():
        print(f"[WARN] Missing {api_features_csv}; run features.py first.")
        return
    api_df = pd.read_csv(api_features_csv)

    # --- by category ---
    if "category_id" in api_df.columns and "target_engagement" in api_df.columns:
        grp = api_df.groupby("category_id")["target_engagement"].mean().sort_values(ascending=False).head(15)
        plt.figure()
        plt.bar(grp.index.astype(str), grp.values)
        plt.title("Avg Engagement Rate by Category (API)")
        plt.xlabel("category_id")
        plt.ylabel("avg engagement")
        plt.tight_layout()
        plt.savefig(REPORTS / "trend_category.png")
        plt.close()
        print(f"[OK] Saved {REPORTS / 'trend_category.png'}")
    else:
        print("[WARN] API features missing category_id or target_engagement; skipping category trend.")

    # --- by length bucket ---
    if "duration_min" in api_df.columns and "target_engagement" in api_df.columns:
        bins = [0,5,10,20,40,120,1e9]
        labels = ["<5","5-10","10-20","20-40","40-120",">120"]
        api_df["len_bucket"] = pd.cut(api_df["duration_min"], bins=bins, labels=labels, include_lowest=True)
        grp2 = api_df.groupby("len_bucket")["target_engagement"].mean()
        plt.figure()
        plt.bar(grp2.index.astype(str), grp2.values)
        plt.title("Avg Engagement Rate by Length (API)")
        plt.xlabel("duration bucket (min)")
        plt.ylabel("avg engagement")
        plt.tight_layout()
        plt.savefig(REPORTS / "trend_length.png")
        plt.close()
        print(f"[OK] Saved {REPORTS / 'trend_length.png'}")
    else:
        print("[WARN] API features missing duration_min or target_engagement; skipping length trend.")

    # --- by age bucket ---
    if "time_since_upload_days" in api_df.columns and "target_engagement" in api_df.columns:
        bins_d = [0,7,30,90,180,365,365*3,1e9]
        labels_d = ["≤1w","≤1m","≤3m","≤6m","≤1y","≤3y",">3y"]
        api_df["age_bucket"] = pd.cut(api_df["time_since_upload_days"], bins=bins_d, labels=labels_d, include_lowest=True)
        grp3 = api_df.groupby("age_bucket")["target_engagement"].mean()
        plt.figure()
        plt.bar(grp3.index.astype(str), grp3.values)
        plt.title("Avg Engagement Rate by Age (API)")
        plt.xlabel("time since upload")
        plt.ylabel("avg engagement")
        plt.tight_layout()
        plt.savefig(REPORTS / "trend_age.png")
        plt.close()
        print(f"[OK] Saved {REPORTS / 'trend_age.png'}")
    else:
        print("[WARN] API features missing time_since_upload_days or target_engagement; skipping age trend.")

def main():
    # 1) Load metrics
    api = load_api_metrics()
    scraped_path, scraped = find_scraped_metrics()

    # 2) Build comparable RMSE chart
    comp = {}
    if api:
        comp.update(get_test_rmse_from_api(api))
    if scraped and scraped_path:
        comp.update(get_test_rmse_from_scraped(scraped, scraped_path.name))

    plot_bar(comp, "Test RMSE (lower is better)", REPORTS / "test_rmse.png")

    # 3) Feature importance charts (skip missing)
    plot_feature_importance(
        MODELS_DIR / "rf_api_target_engagement_feature_importance.csv",
        "Top 20 Feature Importance — RF (API, engagement)",
        REPORTS / "fi_api.png",
        topk=20
    )

    if scraped_path:
        m = re.match(r"scraped_(.+)_metrics\.json", scraped_path.name)
        if m:
            cand = MODELS_DIR / f"rf_scraped_{m.group(1)}_feature_importance.csv"
            plot_feature_importance(
                cand,
                f"Top 20 Feature Importance — RF (Scraped, {m.group(1)})",
                REPORTS / "fi_scraped.png",
                topk=20
            )

    # 4) Engagement trends from API features
    plot_api_trends(PROC_DIR / "api_features.csv")
    print("Saved figures to reports/.")

if __name__ == "__main__":
    main()

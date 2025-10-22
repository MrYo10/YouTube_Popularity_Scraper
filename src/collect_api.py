from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Iterable

import pandas as pd
from tqdm import tqdm
from googleapiclient.discovery import build

# --- Project utils ---
try:
    from utils import RAW_API, ensure_dirs, setup_logger, get_api_key
except Exception:
    # Fallbacks if utils isn't available for some reason
    import os, sys, logging
    from pathlib import Path
    def setup_logger(name="collect_api", level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            h = logging.StreamHandler(sys.stdout)
            import datetime as _dt
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
            h.setFormatter(fmt)
            logger.addHandler(h)
        return logger
    def ensure_dirs(*paths: Path) -> None:
        for p in paths:
            p.mkdir(parents=True, exist_ok=True)
    def get_api_key() -> str:
        key = os.getenv("YOUTUBE_API_KEY")
        if not key:
            raise RuntimeError("❌ No API key found. Set in .env or environment as YOUTUBE_API_KEY.")
        return key
    ROOT = Path(__file__).resolve().parents[1]
    RAW_API = ROOT / "data" / "raw" / "api"

# --------------------- Defaults ---------------------
DEFAULT_QUERIES = [
    "music", "gaming", "news", "technology", "movie trailer",
    "sports", "education tutorial", "travel vlog", "finance investing", "comedy"
]

# polite delay between requests (seconds)
REQ_DELAY = 0.10

# --------------------- Helpers ----------------------
def chunked(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def search_videos(youtube, query: str, max_items: int, logger) -> List[str]:
    """Search videos by keyword and return up to max_items video IDs (deduped)."""
    results: List[str] = []
    next_token = None
    while len(results) < max_items:
        resp = youtube.search().list(
            q=query,
            part="id",
            type="video",
            maxResults=50,
            pageToken=next_token
        ).execute()
        for item in resp.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                results.append(vid)
            if len(results) >= max_items:
                break
        next_token = resp.get("nextPageToken")
        if not next_token:
            break
        time.sleep(REQ_DELAY)
    # preserve order while deduping
    seen = set()
    out = []
    for v in results:
        if v not in seen:
            seen.add(v)
            out.append(v)
    logger.info(f"Search '{query}': got {len(out)} video ids")
    return out

def fetch_video_details(youtube, video_ids: List[str], logger) -> List[Dict[str, Any]]:
    """Batch fetch video details (snippet, contentDetails, statistics, topicDetails)."""
    rows: List[Dict[str, Any]] = []
    for chunk in tqdm(list(chunked(video_ids, 50)), desc="Video details", leave=False):
        resp = youtube.videos().list(
            id=",".join(chunk),
            part="snippet,contentDetails,statistics,topicDetails"
        ).execute()
        for it in resp.get("items", []):
            snip = it.get("snippet", {})
            stats = it.get("statistics", {})
            cont  = it.get("contentDetails", {})
            rows.append({
                "video_id": it.get("id"),
                "title": snip.get("title"),
                "description": snip.get("description"),
                "publish_date": snip.get("publishedAt"),
                "channel_id": snip.get("channelId"),
                "channel_title": snip.get("channelTitle"),
                "tags": "|".join(snip.get("tags", [])) if snip.get("tags") else None,
                "category_id": snip.get("categoryId"),
                "duration_iso8601": cont.get("duration"),
                "licensed_content": cont.get("licensedContent"),
                "definition": cont.get("definition"),
                "projection": cont.get("projection"),
                "viewCount": int(stats.get("viewCount", 0)) if stats.get("viewCount") else None,
                "likeCount": int(stats.get("likeCount", 0)) if stats.get("likeCount") else None,
                "commentCount": int(stats.get("commentCount", 0)) if stats.get("commentCount") else None,
            })
        time.sleep(REQ_DELAY)
    logger.info(f"Fetched details for {len(rows)} videos")
    return rows

def fetch_channel_stats(youtube, channel_ids: List[str], logger) -> Dict[str, Dict[str, Any]]:
    """Batch fetch channel statistics and return a mapping channel_id -> stats dict."""
    out: Dict[str, Dict[str, Any]] = {}
    ids = [c for c in channel_ids if c]
    if not ids:
        return out
    for chunk in tqdm(list(chunked(ids, 50)), desc="Channel stats", leave=False):
        resp = youtube.channels().list(
            id=",".join(chunk),
            part="statistics,snippet"
        ).execute()
        for it in resp.get("items", []):
            cid = it.get("id")
            stats = it.get("statistics", {}) or {}
            out[cid] = {
                "channel_subscribers": int(stats.get("subscriberCount", 0)) if stats.get("subscriberCount") else None,
                "channel_video_count": int(stats.get("videoCount", 0)) if stats.get("videoCount") else None,
                "channel_view_count": int(stats.get("viewCount", 0)) if stats.get("viewCount") else None,
            }
        time.sleep(REQ_DELAY)
    logger.info(f"Fetched stats for {len(out)} channels")
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect structured YouTube metadata via YouTube Data API v3.")
    p.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES,
                   help="Search keywords (space-separated). Default: common broad topics.")
    p.add_argument("--per-query", type=int, default=300,
                   help="Max results per keyword. Default: 300")
    p.add_argument("--test", action="store_true",
                   help="Test mode: use a small subset (2 queries x 5 items) for quick verification.")
    p.add_argument("--out", type=str, default=str(RAW_API / "api_videos.csv"),
                   help="Output CSV path. Default: data/raw/api/api_videos.csv")
    p.add_argument("--resume", action="store_true",
                   help="Resume mode: if output CSV exists, skip already-collected video_ids.")
    return p.parse_args()

# --------------------- Main -------------------------
def main():
    logger = setup_logger("collect_api")
    ensure_dirs(RAW_API)

    args = parse_args()
    out_csv = Path(args.out)

    # Test mode overrides
    queries = list(args.queries)
    per_query = args.per_query
    if args.test:
        queries = queries[:2]  # take first two
        per_query = min(per_query, 5)
        logger.info(f"TEST MODE: queries={queries}, per_query={per_query}")

    # API client
    api_key = get_api_key()
    youtube = build("youtube", "v3", developerKey=api_key)

    #load existing CSV to skip duplicates
    existing_ids = set()
    if args.resume and out_csv.exists():
        try:
            prev = pd.read_csv(out_csv, usecols=["video_id"])
            existing_ids = set(prev["video_id"].dropna().astype(str))
            logger.info(f"Resume ON: found {len(existing_ids)} existing video_ids in {out_csv.name}")
        except Exception as e:
            logger.warning(f"Could not read existing CSV for resume: {e}")

    # 1) Search collect video ids
    all_video_ids: List[str] = []
    for q in tqdm(queries, desc="Search queries"):
        vids = search_videos(youtube, q, per_query, logger)
        all_video_ids.extend(vids)

    # Dedup and apply resume skip
    video_ids = []
    seen = set()
    for vid in all_video_ids:
        if vid and (vid not in seen) and (vid not in existing_ids):
            seen.add(vid)
            video_ids.append(vid)

    logger.info(f"Collected {len(all_video_ids)} ids, deduped to {len(seen)}; "
                f"after resume-skip: {len(video_ids)} remaining")

    if not video_ids:
        logger.info("No new videos to fetch. Exiting.")
        return

    # 2) Fetch details for remaining
    rows = fetch_video_details(youtube, video_ids, logger)
    df = pd.DataFrame(rows).dropna(subset=["video_id"]).drop_duplicates(subset=["video_id"])

    # 3) Channel stats enrichment
    channels = sorted(set(df["channel_id"].dropna()))
    chmap = fetch_channel_stats(youtube, channels, logger)
    df["channel_subscribers"] = df["channel_id"].map(lambda c: chmap.get(c, {}).get("channel_subscribers"))
    df["channel_video_count"] = df["channel_id"].map(lambda c: chmap.get(c, {}).get("channel_video_count"))
    df["channel_view_count"]  = df["channel_id"].map(lambda c: chmap.get(c, {}).get("channel_view_count"))

    # 4) Merge with existing CSV if resume
    if args.resume and out_csv.exists():
        try:
            prev_full = pd.read_csv(out_csv)
            merged = pd.concat([prev_full, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["video_id"])
            merged.to_csv(out_csv, index=False)
            logger.info(f"Appended {len(df)} new rows; total now {len(merged)} → {out_csv}")
        except Exception as e:
            logger.warning(f"Failed to append; writing only new data. Reason: {e}")
            df.to_csv(out_csv, index=False)
            logger.info(f"Wrote {len(df)} rows → {out_csv}")
    else:
        df.to_csv(out_csv, index=False)
        logger.info(f"Wrote {len(df)} rows → {out_csv}")

    logger.info("Done.")

if __name__ == "__main__":
    main()

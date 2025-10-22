from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ========================== CONFIG ==========================
DEFAULT_QUERIES = [
    # Music
    "music", "kpop", "hip hop", "lofi", "edm", "bollywood songs", "latin music", "afrobeat",
    # Gaming
    "gaming", "minecraft", "fortnite", "roblox", "valorant", "fifa highlights", "speedrun",
    # Tech/Edu
    "technology", "iphone review", "pc build", "programming tutorial", "python tutorial", "ai tutorial",
    # News / Sports / General
    "world news", "nba highlights", "soccer highlights", "cricket highlights",
    # Entertainment
    "movie trailer", "anime trailer", "marvel trailer",
    # Lifestyle
    "travel vlog", "cooking recipe", "street food", "podcast", "asmr", "comedy"
]
DEFAULT_PER_QUERY = 300                        # target per query (BFS will try to reach this)
DEFAULT_OUT = Path("data/raw/scraped/scraped_flat.csv")
DEFAULT_DELAY = 0.30                           # polite delay between requests (sec)
MAX_RETRIES = 3
TIMEOUT = 15
# To prevent runaway BFS queue growth; queue cap = per_query * QUEUE_MULT
QUEUE_MULT = 6

# Fixed CSV schema
FIELDNAMES = [
    "video_id", "url", "title", "description", "uploader",
    "publish_date", "category", "duration_seconds",
    "views", "likes", "comments", "tags", "query"
]

# Rotate a few desktop UAs
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]
HEADERS_BASE = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Regexes
RX_PLAYER = re.compile(r"ytInitialPlayerResponse\s*=\s*({.*?});", re.S)
RX_VID = re.compile(r"watch\?v=([a-zA-Z0-9_\-]{11})")


# ========================== HTTP helpers ==========================
def _headers() -> Dict[str, str]:
    h = dict(HEADERS_BASE)
    h["User-Agent"] = random.choice(USER_AGENTS)
    return h

def http_get(url: str, timeout: int = TIMEOUT) -> Optional[requests.Response]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(), timeout=timeout)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        time.sleep(min(1.0 * attempt, 4.0))  # backoff
    return None


# ========================== ID extraction ==========================
def extract_ids_from_html(html: str) -> List[str]:
    """Extract YouTube video IDs from any HTML chunk (search/watch/sidebars)."""
    ids = RX_VID.findall(html)
    seen, out = set(), []
    for v in ids:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def search_video_ids(query: str, max_items: int, delay: float) -> List[str]:
    """Seed: first search page IDs (no JS continuations)."""
    url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
    resp = http_get(url)
    if not resp:
        return []
    ids = extract_ids_from_html(resp.text)
    # Cap to requested maximum
    ids = ids[:max_items]
    time.sleep(delay)
    return ids


# ========================== Watch page parsing ==========================
def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _parse_player_json(html: str) -> Optional[Dict[str, Any]]:
    m = RX_PLAYER.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def parse_video_page(video_id: str, delay: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Return (row, html). Row follows FIELDNAMES; html used to mine related IDs.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    resp = http_get(url)
    if not resp:
        return None, None

    html = resp.text
    data = _parse_player_json(html)

    row = {
        "video_id": video_id,
        "url": url,
        "title": "",
        "description": "",
        "uploader": "",
        "publish_date": None,
        "category": None,
        "duration_seconds": None,
        "views": None,
        "likes": None,       # Often hidden
        "comments": None,    # Not in this JSON
        "tags": None,
        "query": None,       # filled by caller
    }

    if data:
        vd = data.get("videoDetails") or {}
        micro = (data.get("microformat") or {}).get("playerMicroformatRenderer") or {}

        row["title"] = vd.get("title") or ""
        row["description"] = vd.get("shortDescription") or ""
        row["uploader"] = vd.get("author") or micro.get("ownerChannelName") or ""
        row["publish_date"] = micro.get("publishDate")
        row["category"] = micro.get("category")
        row["duration_seconds"] = _safe_int(vd.get("lengthSeconds"))
        row["views"] = _safe_int(vd.get("viewCount"))
        tags_list = vd.get("keywords") or []
        if tags_list:
            row["tags"] = "|".join(tags_list)
    else:
        # Fallback to minimal <meta> tags
        soup = BeautifulSoup(html, "lxml")
        title = soup.find("meta", {"name": "title"})
        desc = soup.find("meta", {"name": "description"})
        uploader = soup.find("link", {"itemprop": "name"})
        row["title"] = (title or {}).get("content", "") if title else ""
        row["description"] = (desc or {}).get("content", "") if desc else ""
        row["uploader"] = (uploader or {}).get("content", "") if uploader else ""

    time.sleep(delay)
    return row, html


# ========================== CSV helpers ==========================
def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def load_existing_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        ids: Set[str] = set()
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get("video_id")
                if vid:
                    ids.add(vid)
        return ids
    except Exception:
        return set()

def write_rows(path: Path, rows: List[Dict[str, Any]], append: bool):
    ensure_parent(path)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        # Normalize to fixed schema
        for r in rows:
            writer.writerow({k: r.get(k) for k in FIELDNAMES})


# ========================== CLI & main ==========================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Scrape YouTube search seeds + BFS related videos (no API)."
    )
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES,
                    help="Space-separated search queries. Default: a broad curated list.")
    ap.add_argument("--per-query", type=int, default=DEFAULT_PER_QUERY,
                    help="Target number of videos per query (BFS will try to reach this).")
    ap.add_argument("--test", action="store_true",
                    help="Test mode: use 2 queries × 25 items target (fast).")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help="Output CSV; default data/raw/scraped/scraped_flat.csv")
    ap.add_argument("--resume", action="store_true",
                    help="Resume: if CSV exists, skip already-scraped video_ids.")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                    help=f"Delay between requests (default {DEFAULT_DELAY}s).")
    return ap.parse_args()

def main():
    args = parse_args()
    out_path = Path(args.out)
    ensure_parent(out_path)

    queries = list(args.queries)
    per_query = int(args.per_query)
    if args.test:
        queries = queries[:2]
        per_query = min(per_query, 25)
        print(f"TEST MODE: queries={queries}, per_query={per_query}, delay={args.delay}s")

    # Resume support
    existing_ids = load_existing_ids(out_path) if args.resume else set()
    if existing_ids:
        print(f"Resume ON: found {len(existing_ids)} existing rows in {out_path.name}")

    total_written = 0
    for q in tqdm(queries, desc="Scraping"):
        # 1) Seed queue from search page
        seeds = search_video_ids(q, per_query, delay=args.delay)
        queue = deque(seeds)
        seen: Set[str] = set(existing_ids) | set(seeds)  # avoid re-adding
        collected = 0
        batch: List[Dict[str, Any]] = []

        # 2) BFS through related IDs until we hit per_query target
        while queue and collected < per_query:
            vid = queue.popleft()

            # Skip if already on disk
            if vid in existing_ids:
                continue

            row, html = parse_video_page(vid, delay=args.delay)
            if not row:
                continue
            row["query"] = q
            batch.append(row)
            collected += 1

            # Mine related IDs and enqueue (bounded growth)
            if html and len(queue) < per_query * QUEUE_MULT:
                for rvid in extract_ids_from_html(html):
                    if rvid not in seen:
                        seen.add(rvid)
                        queue.append(rvid)

            # Flush periodically
            if len(batch) >= 50:
                write_rows(out_path, batch, append=True)
                existing_ids.update(r["video_id"] for r in batch)
                total_written += len(batch)
                batch.clear()

        if batch:
            write_rows(out_path, batch, append=True)
            existing_ids.update(r["video_id"] for r in batch)
            total_written += len(batch)

    print(f"✅ Saved {total_written} scraped videos → {out_path}")

if __name__ == "__main__":
    main()

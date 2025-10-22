from __future__ import annotations
import os, sys, json, re, time, subprocess, logging
from pathlib import Path
from typing import Iterable, List, Any, Callable, Optional

# -------- Paths--------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW  = DATA / "raw"
RAW_SCRAPED = RAW / "scraped"
RAW_API     = RAW / "api"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# -------- .env loading --------
from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")  # <-- explicitly load from project root

def get_api_key() -> str:
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("No API key found. Make sure .env exists at project root with YOUTUBE_API_KEY=...")
    return key

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

# -------- Logging --------
def setup_logger(name: str = "ytproj", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

# -------- Serialization --------
def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

# -------- System helpers --------
def run_cmd(args: List[str], logger: Optional[logging.Logger] = None, text: bool = True):
    if logger: logger.debug(f"Running: {' '.join(args)}")
    return subprocess.run(args, capture_output=True, text=text)

def retry(n: int, wait_sec: float, fn: Callable[[], Any], logger: Optional[logging.Logger] = None) -> Any:
    last = None
    for i in range(1, n+1):
        try:
            return fn()
        except Exception as e:
            last = e
            if logger: logger.warning(f"Attempt {i} failed: {e}; retrying in {wait_sec}s")
            time.sleep(wait_sec)
    raise last

# -------- Misc --------
def chunked(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

_slug_rx = re.compile(r"[^a-zA-Z0-9._-]+")
def slugify(s: str) -> str:
    s = s.strip().replace(" ", "-")
    return _slug_rx.sub("-", s)

def parse_int(s: Any) -> Optional[int]:
    """Best-effort turn '1,234' or None into int, else None."""
    if s is None: return None
    if isinstance(s, (int,)): return s
    try:
        return int(str(s).replace(",", "").strip())
    except Exception:
        return None

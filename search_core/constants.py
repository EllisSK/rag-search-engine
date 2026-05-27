from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

BM25_K1 = 1.5
BM25_B = 0.75
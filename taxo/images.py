"""Image sources for the evaluators.

  manifest JSONL  {"image": "<id>", "path": "<file>"} per line
  gallery         --gallery-data /data → every blob-present image of the
                  FineTuneGallery DB (images/<sha[0:2]>/<sha>.png)
"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def from_manifest(path) -> list[tuple[str, str]]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append((r["image"], r["path"]))
    return out


def from_gallery(data_dir) -> list[tuple[str, str]]:
    data = Path(data_dir)
    db = sqlite3.connect(f"file:{data / 'db' / 'finetune.sqlite'}?mode=ro", uri=True)
    rows = db.execute("SELECT id, sha256 FROM images WHERE blob_present = 1 ORDER BY id").fetchall()
    db.close()
    return [(i, str(data / "images" / s[:2] / f"{s}.png")) for i, s in rows]


def source(args) -> list[tuple[str, str]]:
    items = from_gallery(args.gallery_data) if args.gallery_data else from_manifest(args.manifest)
    if args.limit:
        items = items[: args.limit]
    return items


def add_args(ap) -> None:
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="JSONL with image/path")
    g.add_argument("--gallery-data", help="FineTuneGallery data dir")
    ap.add_argument("--limit", type=int, default=0)


def _open(p: str) -> Image.Image:
    im = Image.open(p)
    im.load()
    return im


def batches(items, size: int, workers: int = 4):
    """Yield (ids, PIL images) with decoding overlapped with inference."""
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(items), size):
            chunk = items[i: i + size]
            ims = list(pool.map(lambda it: _open(it[1]), chunk))
            yield [c[0] for c in chunk], ims

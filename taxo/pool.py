"""Stratified calibration pool from the gallery (UC25, SPARK calibration step 1a).

Reads the FineTuneGallery DB read-only (same as images.py --gallery-data) and
writes a manifest for the evaluators — ids, sha and paths, never image copies:

  {"image": "<id>", "sha256": "<sha>", "path": "<data>/images/<sha[0:2]>/<sha>.png",
   "wd14_rating": "explicit"}

All explicit and questionable images, then sensitive and general sampled so the
pool keeps the ratio explicit : questionable : sensitive : general = 4 : 2 : 1 : 1
(scaled to the explicit count, capped by what exists).

Excluded, before any sampling:
  * every image with a row in taxonomy_blocks (quarantine or block),
  * every image with any minor signal at all — the taxonomy's wd14_signals plus
    teen, at any confidence (stricter than the server's min_conf on purpose),
  * images without a WD14 caption or without the blob.

  python taxo/pool.py --gallery-data /data --out pool.jsonl [--seed 25]
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxolib  # noqa: E402

RATIO = {"explicit": 4, "questionable": 2, "sensitive": 1, "general": 1}


def minor_tags(tax: taxolib.Taxonomy) -> set[str]:
    sig = set(tax.minor_block["wd14_signals"]) | set(tax.minor_block["values"])
    return {taxolib.normalize_tag(t) for t in sig}


def candidates(data: Path, tax: taxolib.Taxonomy) -> tuple[dict[str, list[dict]], Counter]:
    db = sqlite3.connect(f"file:{data / 'db' / 'finetune.sqlite'}?mode=ro", uri=True)
    blocked = {r[0] for r in db.execute("SELECT image_id FROM taxonomy_blocks")}
    minor = minor_tags(tax)
    by_rating: dict[str, list[dict]] = {k: [] for k in RATIO}
    skipped: Counter = Counter()
    rows = db.execute("""
        SELECT i.id, i.sha256, c.tags_json FROM images i
        JOIN captions c ON c.image_id = i.id AND c.kind = 'auto'
        WHERE i.blob_present = 1 ORDER BY i.id""")
    for iid, sha, tj in rows:
        if iid in blocked:
            skipped["blocked"] += 1
            continue
        tags = json.loads(tj or "{}")
        rating = tags.get("rating") or {}
        general = {taxolib.normalize_tag(k) for k in (tags.get("general") or {})}
        if general & minor:
            skipped["minor_tag"] += 1
            continue
        if not rating:
            skipped["no_rating"] += 1
            continue
        top = max(rating, key=rating.get)
        if top not in by_rating:
            skipped["other_rating"] += 1
            continue
        by_rating[top].append({
            "image": iid, "sha256": sha,
            "path": str(data / "images" / sha[:2] / f"{sha}.png"),
            "wd14_rating": top,
        })
    db.close()
    return by_rating, skipped


def stratify(by_rating: dict[str, list[dict]], seed: int) -> list[dict]:
    rnd = random.Random(seed)
    unit = len(by_rating["explicit"]) / RATIO["explicit"]
    out = []
    for k, w in RATIO.items():
        pool = by_rating[k]
        want = len(pool) if k in ("explicit", "questionable") else min(len(pool), round(unit * w))
        out += rnd.sample(pool, want) if want < len(pool) else pool
    rnd.shuffle(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-data", required=True, help="FINETUNE_DATA dir (db/, images/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args(argv)

    tax = taxolib.load()
    by_rating, skipped = candidates(Path(args.gallery_data), tax)
    pool = stratify(by_rating, args.seed)
    with open(args.out, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({
        "pool": dict(Counter(r["wd14_rating"] for r in pool)),
        "available": {k: len(v) for k, v in by_rating.items()},
        "skipped": dict(skipped),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())

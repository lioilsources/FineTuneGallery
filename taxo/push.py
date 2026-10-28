"""Upload combine.py output (and the SigLIP embeddings) to the gallery.

  python taxo/push.py --server http://finetune:8092 --combined combined.jsonl [--emb emb.npz]

POST /api/taxonomy/import in batches. Image ids must be gallery ids (use
--gallery-data when running the evaluators). The server recomputes the band
from its active thresholds and makes the minor-block decision itself — this
script only carries the evaluators' answers.
"""

from __future__ import annotations

import argparse
import base64
import sys

import numpy as np
import requests

import taxolib


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--combined", required=True)
    ap.add_argument("--emb")
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args(argv)

    rows = taxolib.read_jsonl(a.combined)
    if a.emb:
        z = np.load(a.emb)
        model = str(z["model"]) if "model" in z else "siglip2"
        emb = {str(i): e for i, e in zip(z["ids"], z["emb"])}
        for r in rows:
            e = emb.get(r["image"])
            if e is not None:
                r["embedding"] = {"model": model, "dim": int(e.shape[0]),
                                  "f32": base64.b64encode(e.astype("<f4").tobytes()).decode()}
    url = a.server.rstrip("/") + "/api/taxonomy/import"
    done = blocked = 0
    for i in range(0, len(rows), a.batch):
        r = requests.post(url, json={"records": rows[i: i + a.batch]}, timeout=120)
        if r.status_code != 200:
            print(f"import failed at {i}: HTTP {r.status_code} {r.text[:300]}", file=sys.stderr)
            return 1
        res = r.json()
        done += res.get("imported", 0)
        blocked += res.get("blocked", 0)
    print(f"imported {done}, blocked/quarantined {blocked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

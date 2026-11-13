"""Upload combine.py output (and the SigLIP embeddings) to the gallery.

  python taxo/push.py --server http://finetune:8092 --combined combined.jsonl [--emb emb.npz]
  python taxo/push.py --server http://finetune:8092 --bands bands.json   # conformal.py fit → thresholds

POST /api/taxonomy/import in batches. Image ids must be gallery ids (use
--gallery-data when running the evaluators). The server recomputes the band
from its active thresholds and makes the minor-block decision itself — this
script only carries the evaluators' answers.

From the GPU box the gallery is reached over its public hostname (CF Access),
never a LAN port. Credentials come from the environment only (a gitignored
.env of the batch), never from arguments, and are never printed:

  FINETUNE_URL             default for --server
  CF_ACCESS_CLIENT_ID      the batch's own CF Access service token
  CF_ACCESS_CLIENT_SECRET
  FINETUNE_WORKER_TOKEN    the gallery's WORKER_TOKEN (import + thresholds only)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np
import requests

import taxolib


def session() -> requests.Session:
    s = requests.Session()
    # CF answers a default python-requests agent with 1010 (looks like a bad token).
    s.headers["User-Agent"] = "Mozilla/5.0 (finetune-taxo-push)"
    env = os.environ
    if env.get("CF_ACCESS_CLIENT_ID") and env.get("CF_ACCESS_CLIENT_SECRET"):
        s.headers["CF-Access-Client-Id"] = env["CF_ACCESS_CLIENT_ID"]
        s.headers["CF-Access-Client-Secret"] = env["CF_ACCESS_CLIENT_SECRET"]
    if env.get("FINETUNE_WORKER_TOKEN"):
        s.headers["Authorization"] = "Bearer " + env["FINETUNE_WORKER_TOKEN"]
    return s


def send(s: requests.Session, method: str, url: str, body, tries: int = 5) -> requests.Response:
    """Retry 429/5xx and connection errors with backoff (the tunnel, not the server)."""
    for i in range(tries):
        try:
            r = s.request(method, url, json=body, timeout=120)
            if r.status_code != 429 and r.status_code < 500:
                return r
        except requests.RequestException:
            if i == tries - 1:
                raise
        time.sleep(min(60, 2 ** i))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.environ.get("FINETUNE_URL"))
    ap.add_argument("--combined")
    ap.add_argument("--bands", help="conformal.py fit output; the server refuses an invalid fit")
    ap.add_argument("--emb")
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args(argv)
    if not a.server:
        ap.error("--server or FINETUNE_URL")
    s = session()

    if a.bands:
        fit = json.loads(open(a.bands).read())
        th = {k: v for k, v in fit["thresholds"].items() if v is not None}
        r = send(s, "PUT", a.server.rstrip("/") + "/api/taxonomy/thresholds",
                 {"thresholds": th, "source": fit["source"], "fit": fit})
        print(f"thresholds: HTTP {r.status_code} {r.text[:300]}")
        return 0 if r.status_code == 200 else 1
    if not a.combined:
        ap.error("--combined or --bands")
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
        r = send(s, "POST", url, {"records": rows[i: i + a.batch]})
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

"""Merge the evaluators' JSONL into one record per image: every answer, the
consensus, the rating ensemble and the minor decision — the import format of
POST /api/taxonomy/import — plus the summary numbers.

  python taxo/combine.py --wd14 wd14.jsonl --nsfw nsfw.jsonl --siglip siglip.jsonl \
      [--vlm vlm.jsonl] [--head head.jsonl] --out combined.jsonl --stats stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import consensus
import taxolib
import temperature


def load_by_image(path) -> dict[str, list[dict]]:
    out = defaultdict(list)
    if path:
        for r in taxolib.read_jsonl(path):
            out[r["image"]].append(r)
    return out


def combine(tax, wd14, nsfw, siglip, vlm=None, head=None, consensus_from=("wd14", "siglip", "vlm")):
    vlm, head = vlm or {}, head or {}
    ids = sorted(set(wd14) | set(siglip) | set(nsfw) | set(vlm))
    out = []
    for iid in ids:
        by_eval, parts, extra = {}, {}, {}
        for src, recs in (("wd14", wd14), ("siglip", siglip), ("vlm", vlm), ("head", head)):
            for r in recs.get(iid, []):
                if r.get("taxonomy_version") != tax.version:
                    continue
                by_eval[src] = r["labels"]
                if src == "wd14":
                    parts["wd14"] = (r.get("rating") or {}).get("temperature")
                    extra["minor_signal"] = r.get("minor_signal", 0.0)
                if src == "vlm":
                    extra["vlm_rating"] = r.get("vlm_rating")
        for r in nsfw.get(iid, []):
            parts[r["evaluator"]] = (r.get("rating") or {}).get("temperature")
        ens = temperature.ensemble(tax, parts)
        cons = consensus.consensus(tax, by_eval, consensus_from)
        minor = consensus.minor_decision(tax, by_eval, extra.get("minor_signal", 0.0), ens, parts.get("wd14"))
        out.append({"image": iid, "taxonomy_version": tax.version, "labels": by_eval,
                    "consensus": cons, "rating": ens, "minor": minor, **extra})
    return out


def stats(tax, rows) -> dict:
    n = len(rows)
    node_unsure, node_answered, node_agreed = Counter(), Counter(), Counter()
    img_any_unsure = 0
    for r in rows:
        any_u = False
        for nid, c in r["consensus"].items():
            node_answered[nid] += 1
            if c["unsure"]:
                node_unsure[nid] += 1
                any_u = True
            elif c["n"] >= 2:
                node_agreed[nid] += 1
        any_u = any_u or r["rating"]["unsure"]
        img_any_unsure += any_u
    bands = Counter(r["rating"]["band"] for r in rows)
    rating_unsure = sum(1 for r in rows if r["rating"]["unsure"])
    decisions = Counter(r["minor"] or "none" for r in rows)
    total_node_answers = sum(node_answered.values())
    return {
        "images": n,
        "bands": dict(bands),
        "rating_unsure": rating_unsure,
        "rating_unsure_share": round(rating_unsure / n, 4) if n else None,
        "images_with_any_unsure_share": round(img_any_unsure / n, 4) if n else None,
        "node_unsure_share_overall": round(sum(node_unsure.values()) / total_node_answers, 4) if total_node_answers else None,
        "per_node": {
            nid: {"answered": node_answered[nid], "agreed": node_agreed[nid], "unsure": node_unsure[nid],
                  "unsure_share": round(node_unsure[nid] / node_answered[nid], 4)}
            for nid in [x.id for x in tax.nodes] if node_answered[nid]
        },
        "minor_decisions": dict(decisions),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd14", required=True)
    ap.add_argument("--nsfw", required=True)
    ap.add_argument("--siglip", required=True)
    ap.add_argument("--vlm")
    ap.add_argument("--head")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats")
    a = ap.parse_args(argv)
    tax = taxolib.load()
    rows = combine(tax, load_by_image(a.wd14), load_by_image(a.nsfw), load_by_image(a.siglip),
                   load_by_image(a.vlm), load_by_image(a.head))
    taxolib.write_jsonl(a.out, rows)
    st = stats(tax, rows)
    st["evaluators"] = ["wd14", "siglip"] + (["vlm"] if a.vlm else [])
    if a.stats:
        with open(a.stats, "w") as f:
            json.dump(st, f, indent=1)
    print(json.dumps({k: st[k] for k in ("images", "bands", "rating_unsure_share",
                                          "node_unsure_share_overall", "minor_decisions")}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

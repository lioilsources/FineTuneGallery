"""Agreement between evaluators = confidence.

Per node, from the evaluators that did not abstain:
  single:  ≥2 raters on the same value → consensus; any conflict → unsure
  ordinal: as single, but adjacent values agree (consensus = the lower-rank
           majority-side value, i.e. the median of the votes)
  multi:   value in the consensus iff ≥2 raters include it; unsure when the
           raters' sets overlap less than MULTI_MIN_JACCARD
  one rater only → tentative value, unsure (single source is not agreement)

minor_decision mirrors server/taxonomy.go (the server is the enforcement
point; this copy only produces the report numbers and guards the calibration
generator). The two are tested on the same synthetic vectors.
"""

from __future__ import annotations

from itertools import combinations

import taxolib

MULTI_MIN_JACCARD = 0.5


def _single_conf(lab: dict) -> float:
    c = lab.get("conf", 1.0)
    return float(c) if not isinstance(c, dict) else 1.0


def node_consensus(node: taxolib.Node, votes: dict[str, dict]) -> dict | None:
    """votes: evaluator → {"value", "conf"} for this node (abstentions absent)."""
    if not votes:
        return None
    if node.kind == "multi":
        sets = {e: set(v["value"]) for e, v in votes.items()}
        if len(sets) == 1:
            (e, s), = sets.items()
            return {"value": sorted(s, key=node.value_ids.index), "unsure": True, "n": 1, "why": "single source"}
        count: dict[str, int] = {}
        for s in sets.values():
            for x in s:
                count[x] = count.get(x, 0) + 1
        agreed = sorted([x for x, c in count.items() if c >= 2], key=node.value_ids.index)
        jac = []
        for a, b in combinations(sets.values(), 2):
            u = a | b
            jac.append(1.0 if not u else len(a & b) / len(u))
        mean_j = sum(jac) / len(jac)
        return {"value": agreed, "unsure": mean_j < MULTI_MIN_JACCARD, "n": len(sets),
                "jaccard": round(mean_j, 3)}

    vals = {e: v["value"] for e, v in votes.items() if v["value"] in node.value_ids}
    if not vals:
        return None
    if len(vals) == 1:
        (e, x), = vals.items()
        return {"value": x, "unsure": True, "n": 1, "why": "single source"}
    idx = sorted(node.value_ids.index(x) for x in vals.values())
    if node.kind == "ordinal":
        ok = idx[-1] - idx[0] <= 1
        med = idx[(len(idx) - 1) // 2]
        return {"value": node.value_ids[med], "unsure": not ok, "n": len(vals)}
    count = {}
    for x in vals.values():
        count[x] = count.get(x, 0) + 1
    best = max(count, key=count.get)
    return {"value": best, "unsure": len(count) > 1 or count[best] < 2, "n": len(vals)}


def consensus(tax: taxolib.Taxonomy, by_eval: dict[str, dict],
              evaluators: tuple[str, ...] = ("wd14", "siglip", "vlm")) -> dict:
    """by_eval: evaluator → labels dict (taxolib record format)."""
    out = {}
    for n in tax.nodes:
        votes = {e: by_eval[e][n.id] for e in evaluators if e in by_eval and n.id in by_eval[e]}
        c = node_consensus(n, votes)
        if c is not None:
            out[n.id] = c
    return out


def minor_decision(tax: taxolib.Taxonomy, by_eval: dict[str, dict],
                   wd14_signal: float, ensemble: dict | None,
                   wd14_temp: float | None, thresholds: dict | None = None) -> str | None:
    """None | "quarantine" | "block". See server/taxonomy.go minorDecision."""
    mb = tax.minor_block
    th = thresholds or tax.thresholds()
    minor = wd14_signal >= mb["min_conf"]
    for e, labels in by_eval.items():
        lab = labels.get("person.age_group")
        if not lab:
            continue
        conf = 1.0 if e == "human" else _single_conf(lab)
        if lab["value"] in mb["values"] and conf >= mb["min_conf"]:
            minor = True
    if not minor:
        return None
    if ensemble is not None and ensemble.get("temperature") is not None:
        if ensemble["band"] == "safe" and not ensemble["unsure"]:
            return None
        return "block"
    if wd14_temp is not None and taxolib.band_of(wd14_temp, th) == "safe":
        return None
    return "quarantine"

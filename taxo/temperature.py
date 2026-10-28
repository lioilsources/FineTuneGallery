"""Rating "temperature": WD14 rating probabilities ensembled with two binary
NSFW classifiers (Falconsai/nsfw_image_detection, AdamCodd/vit-base-nsfw-detector).

  T_wd14 = Σ p_r · level_r / Σ p_r      (r ∈ general/sensitive/questionable/explicit)
  T_bin  = p(nsfw)                      (each binary classifier)
  T      = weighted mean of the available parts (weights in taxonomy.yaml)

unsure when the raters disagree: spread of the parts > disagree_delta, or one
part lands in `safe` while another lands in `nsfw`/`explicit`. An unsure image
is never `safe` for a shared view and goes to the human queue.
"""

from __future__ import annotations

import taxolib


def wd14_temperature(tax: taxolib.Taxonomy, rating: dict[str, float]) -> float | None:
    levels = tax.rating["wd14_levels"]
    num = den = 0.0
    for k, lvl in levels.items():
        p = float(rating.get(k, 0.0))
        num += p * lvl
        den += p
    if den <= 0:
        return None
    return num / den


def ensemble(tax: taxolib.Taxonomy, parts: dict[str, float | None],
             thresholds: dict[str, float] | None = None) -> dict:
    th = thresholds or tax.thresholds()
    cfg = tax.rating["ensemble"]
    avail = {k: float(v) for k, v in parts.items() if v is not None and k in cfg["weights"]}
    if not avail:
        return {"temperature": None, "band": None, "unsure": True, "parts": parts, "reason": "no rater"}
    wsum = sum(cfg["weights"][k] for k in avail)
    t = sum(cfg["weights"][k] * v for k, v in avail.items()) / wsum
    bands = {k: taxolib.band_of(v, th) for k, v in avail.items()}
    spread = max(avail.values()) - min(avail.values())
    ranks = [taxolib.BAND_RANK[b] for b in bands.values()]
    reasons = []
    if len(avail) < len(cfg["weights"]):
        reasons.append("missing:" + ",".join(sorted(set(cfg["weights"]) - set(avail))))
    if spread > cfg["disagree_delta"]:
        reasons.append(f"spread {spread:.2f}")
    if min(ranks) == 0 and max(ranks) >= 2:
        reasons.append("safe vs nsfw")
    # A missing rater alone does not make an image unsure; disagreement does.
    unsure = any(not r.startswith("missing") for r in reasons)
    return {"temperature": round(t, 4), "band": taxolib.band_of(t, th), "unsure": unsure,
            "parts": {k: round(v, 4) for k, v in avail.items()}, "part_bands": bands,
            "reason": "; ".join(reasons)}

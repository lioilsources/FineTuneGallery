"""Temperature band thresholds with guarantees (split conformal + Wilson).

  python taxo/conformal.py fit --scores combined.jsonl --labels human.jsonl --out bands.json
  python taxo/conformal.py fit --scores combined.jsonl --pseudo wd14-argmax --out bands.json
  python taxo/conformal.py synth --n 3000          # self-check on synthetic data

Guarantees (taxonomy.yaml rating.conformal):
  explicit recall ≥ 0.99   explicit_guard = the ⌊α(n+1)⌋-th smallest
                           temperature among calibration `explicit` (α = 0.01),
                           tightened to the PAC rank (see guard_rank): with
                           probability ≥ 95 % over the calibration draw,
                           recall on new exchangeable images is ≥ 1 − α. Anything ≥ guard that the
                           point thresholds would not call explicit is sent
                           to `unsure` — a false positive into the human
                           queue, never a miss. Needs ≥ 299 explicit labels (PAC) / 99 (marginal);
                           below that the guard is 0 (everything unsure-or-
                           worse) and the fit is marked invalid.
  safe precision ≥ 0.98    safe_max = the largest threshold whose one-sided
                           95 % Wilson lower bound of precision(T ≤ t → safe)
                           is ≥ 0.98 — the same lower-bound rule the eval
                           ranks with.
The two middle cut points are point estimates (balanced accuracy).

Pseudo-labels are only for testing the tooling: the result must be re-fit on
the user's labels before it is used (the server records the fit's source).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys

import taxolib

Z95_ONE_SIDED = 1.6448536269514722
WD14_TO_BAND = {"general": "safe", "sensitive": "suggestive", "questionable": "nsfw", "explicit": "explicit"}


def wilson_lower(k: int, n: int, z: float = Z95_ONE_SIDED) -> float:
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binom(n, p) ≤ k)."""
    if k < 0:
        return 0.0
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                 + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(0, min(k, n) + 1)))


def guard_rank(n: int, alpha: float, delta: float | None) -> int:
    """How many of the lowest explicit scores may fall below the guard.

    delta None: marginal split-conformal rank ⌊α(n+1)⌋ — recall ≥ 1−α in
    expectation over calibration sets. delta (default 0.05): PAC rank — the
    largest k with P(Binom(n, α) ≤ k−1) ≤ δ, so that with probability ≥ 1−δ
    over the calibration draw the recall itself is ≥ 1−α. Stricter; it is
    the default because one calibration set is all there will be."""
    if delta is None:
        return math.floor(alpha * (n + 1))
    k = 0
    while k + 1 <= n and binom_cdf(k, n, alpha) <= delta:
        k += 1
    return k


def explicit_guard(scores_explicit: list[float], recall: float, delta: float | None = 0.05) -> tuple[float, bool]:
    n = len(scores_explicit)
    k = guard_rank(n, 1 - recall, delta)
    if k < 1:
        return 0.0, False
    return sorted(scores_explicit)[k - 1], True


def explicit_needed(recall: float, delta: float | None = 0.05) -> int:
    n = 1
    while guard_rank(n, 1 - recall, delta) < 1:
        n += 1
    return n


def safe_threshold(pairs: list[tuple[float, str]], precision: float) -> tuple[float, dict]:
    pairs = sorted(pairs)
    best, info = 0.0, {"precision": None, "precision_lower": None, "n_pred_safe": 0}
    k = n = 0
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == t:
            n += 1
            k += pairs[i][1] == "safe"
            i += 1
        lo = wilson_lower(k, n)
        if lo >= precision:
            best, info = t, {"precision": k / n, "precision_lower": lo, "n_pred_safe": n}
    return best, info


def _cut(pairs, low: set, high: set, lo_bound: float) -> float:
    """Threshold between two label groups maximising balanced accuracy."""
    xs = [(s, l in high) for s, l in pairs if l in low | high]
    nl = sum(1 for _, h in xs if not h)
    nh = len(xs) - nl
    if not nl or not nh:
        return None
    best, best_t = -1, None
    for t in sorted({s for s, _ in xs}):
        if t <= lo_bound:
            continue
        tn = sum(1 for s, h in xs if not h and s <= t)
        tp = sum(1 for s, h in xs if h and s > t)
        ba = (tn / nl + tp / nh) / 2
        if ba > best:
            best, best_t = ba, t
    return best_t


def fit(pairs: list[tuple[float, str]], tax: taxolib.Taxonomy, source: str) -> dict:
    cfg = tax.rating["conformal"]
    default = tax.thresholds()
    guard, guard_ok = explicit_guard([s for s, l in pairs if l == "explicit"], cfg["explicit_recall"])
    safe_max, safe_info = safe_threshold(pairs, cfg["safe_precision"])
    sugg = _cut(pairs, {"safe", "suggestive"}, {"nsfw", "explicit"}, safe_max) or default["suggestive_max"]
    nsfw = _cut(pairs, {"safe", "suggestive", "nsfw"}, {"explicit"}, sugg) or default["nsfw_max"]
    eps = 1e-4
    sugg = max(sugg, safe_max + eps)
    nsfw = max(nsfw, sugg + eps)
    # An invalid guard is left out, not 0: a guard of 0 would send every image
    # to unsure. The fit is then marked invalid and the server refuses it.
    th = {"safe_max": round(safe_max, 6), "suggestive_max": round(sugg, 6), "nsfw_max": round(nsfw, 6),
          "explicit_guard": round(min(guard, nsfw), 6) if guard_ok else None}
    # empirical check on the calibration set itself (optimistic; the
    # guarantee is the construction, this is the sanity print)
    ex = [s for s, l in pairs if l == "explicit"]
    g = th["explicit_guard"] if th["explicit_guard"] is not None else th["nsfw_max"]
    caught = sum(1 for s in ex if s >= g)
    unsure = sum(1 for s, _ in pairs if g <= s <= th["nsfw_max"]) if guard_ok else 0
    counts = {b: sum(1 for _, l in pairs if l == b) for b in tax.bands}
    return {
        "thresholds": th, "source": source, "n": len(pairs), "label_counts": counts,
        "valid": bool(guard_ok and safe_info["n_pred_safe"] > 0),
        "explicit_guard_valid": guard_ok,
        "explicit_needed": explicit_needed(cfg["explicit_recall"]),
        "calibration": {"explicit_recall_with_guard": caught / len(ex) if ex else None,
                        "safe": safe_info, "unsure_share_from_guard": unsure / len(pairs) if pairs else None},
        "targets": cfg,
    }


def _scores(path) -> dict[str, float]:
    out = {}
    for r in taxolib.read_jsonl(path):
        t = (r.get("rating") or {}).get("temperature")
        if t is not None:
            out[r["image"]] = t
    return out


def _pseudo_wd14(path) -> dict[str, str]:
    out = {}
    for r in taxolib.read_jsonl(path):
        if r.get("evaluator") == "wd14":
            rt = r["rating"]["parts"]["wd14"]
            out[r["image"]] = WD14_TO_BAND[max(rt, key=rt.get)]
    return out


def synth(n: int, seed: int = 1, shift: float = 0.0) -> list[tuple[float, str]]:
    """Synthetic calibration data: class-conditional Beta scores."""
    rnd = random.Random(seed)
    params = {"safe": (1.2, 12), "suggestive": (3, 6), "nsfw": (6, 4), "explicit": (12, 1.6)}
    mix = {"safe": 0.55, "suggestive": 0.2, "nsfw": 0.12, "explicit": 0.13}
    out = []
    for _ in range(n):
        b = rnd.choices(list(mix), weights=list(mix.values()))[0]
        a, bb = params[b]
        out.append((min(1.0, max(0.0, rnd.betavariate(a, bb) + shift)), b))
    return out


def evaluate(th: dict, pairs) -> dict:
    th = {**th, "explicit_guard": th["explicit_guard"] if th["explicit_guard"] is not None else th["nsfw_max"]}
    ex = [s for s, l in pairs if l == "explicit"]
    pred_safe = [(s, l) for s, l in pairs if s <= th["safe_max"]]
    return {
        "explicit_recall": sum(1 for s in ex if s >= th["explicit_guard"]) / len(ex) if ex else None,
        "safe_precision": sum(1 for _, l in pred_safe if l == "safe") / len(pred_safe) if pred_safe else None,
        "safe_coverage": len(pred_safe) / max(1, sum(1 for _, l in pairs if l == "safe")),
        "unsure_share": sum(1 for s, _ in pairs if th["explicit_guard"] <= s <= th["nsfw_max"]) / len(pairs),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--scores", required=True, help="combined.jsonl (rating.temperature)")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--labels", help="JSONL {image, rating: {band}} — the user's labels")
    g.add_argument("--pseudo", choices=["wd14-argmax"], help="tooling test only")
    f.add_argument("--wd14", help="wd14.jsonl for --pseudo wd14-argmax")
    f.add_argument("--out", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    tax = taxolib.load()

    if a.cmd == "synth":
        cal, test = synth(a.n, a.seed), synth(20000, a.seed + 1)
        res = fit(cal, tax, "synthetic")
        res["test"] = evaluate(res["thresholds"], test)
        print(json.dumps(res, indent=1))
        return 0

    scores = _scores(a.scores)
    if a.labels:
        truth = {r["image"]: r["rating"]["band"] for r in taxolib.read_jsonl(a.labels)
                 if (r.get("rating") or {}).get("band")}
        source = "human"
    else:
        truth = _pseudo_wd14(a.wd14 or a.scores)
        source = "pseudo:wd14-argmax"
    pairs = [(scores[i], b) for i, b in truth.items() if i in scores]
    res = fit(pairs, tax, source)
    with open(a.out, "w") as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps({k: res[k] for k in ("thresholds", "n", "label_counts", "valid")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

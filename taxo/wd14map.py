"""WD14 booru tags → taxonomy labels. Deterministic, free (the tags are already
in captions.tags_json for every captioned image).

The mapping lives in taxonomy.yaml (values.<v>.wd14); this module only
applies it. Tags are matched in the tags_json spelling (spaces).
"""

from __future__ import annotations

import taxolib


def _value_conf(v: taxolib.Value, tags: dict[str, float]) -> float:
    return max((tags.get(t, 0.0) for t in v.wd14), default=0.0)


def map_tags(tax: taxolib.Taxonomy, general: dict[str, float],
             min_conf: float = taxolib.DEFAULT_MIN_CONF["wd14"]) -> dict:
    tags = {taxolib.normalize_tag(k): float(c) for k, c in general.items()}
    labels: dict = {}

    subj = tax.node("subject")
    present = {}
    for v in subj.values:
        c = _value_conf(v, tags)
        neg = max((tags.get(t, 0.0) for t in subj.wd14_negative.get(v.id, [])), default=0.0)
        if c >= min_conf and c > neg:
            present[v.id] = round(c, 4)
    # An image with no subject tag at all is an abstention, not "nothing".
    if present or tags.get("no humans", 0.0) >= min_conf:
        labels["subject"] = {"value": sorted(present, key=subj.value_ids.index), "conf": present}
    has_people = "people" in present

    for n in tax.nodes:
        if n.id == "subject":
            continue
        if n.when == "people" and not has_people:
            continue
        mapped = [v for v in n.values if v.wd14]
        if not mapped:
            continue
        thr = max(min_conf, n.min_conf)
        confs = {v.id: _value_conf(v, tags) for v in mapped}
        if n.kind == "multi":
            vals = {k: round(c, 4) for k, c in confs.items() if c >= thr}
            # The tagger looked at a person and tagged none of these items:
            # that is an answer ([]), and it is what makes "no glasses" agree.
            labels[n.id] = {"value": [k for k in n.value_ids if k in vals], "conf": vals}
        else:
            best = max(confs, key=confs.get)
            if confs[best] >= thr:
                labels[n.id] = {"value": best, "conf": round(confs[best], 4)}
    return labels


def minor_signal(tax: taxolib.Taxonomy, general: dict[str, float]) -> float:
    """Highest confidence among the WD14 tags that are a minor signal alone."""
    tags = {taxolib.normalize_tag(k): float(c) for k, c in general.items()}
    sig = [taxolib.normalize_tag(t) for t in tax.minor_block["wd14_signals"]]
    return max((tags.get(t, 0.0) for t in sig), default=0.0)

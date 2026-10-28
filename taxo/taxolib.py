"""taxonomy.yaml loader and the pieces every evaluator shares.

One output format for all evaluators (one JSON object per image, JSONL):

  {"image": "<gallery image id or path>",
   "evaluator": "wd14" | "siglip" | "vlm" | "head" | "human",
   "model": "<hf repo or alias>",
   "taxonomy_version": 1,
   "labels": {"<node id>": {"value": "<v>" | ["<v>", …], "conf": 0.0–1.0}, …},
   "rating": {"temperature": 0.0–1.0, "band": "safe", "parts": {…}, "unsure": false},
   "ms": <latency per image>}

A node missing from `labels` is an abstention. `rating` is only filled by the
temperature evaluators (wd14 rating part, the two NSFW classifiers, and the
ensemble record "temperature").
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TAXONOMY_PATH = HERE / "taxonomy.yaml"


@dataclass
class Value:
    id: str
    wd14: list[str] = field(default_factory=list)
    siglip: list[str] = field(default_factory=list)
    filterable: bool = True


@dataclass
class Node:
    id: str
    kind: str  # single | multi | ordinal
    values: list[Value]
    when: str | None = None
    facet: bool = True
    human_required: bool = False
    label_cs: str = ""
    vlm: str = ""
    min_conf: float = 0.0
    wd14_negative: dict[str, list[str]] = field(default_factory=dict)

    @property
    def value_ids(self) -> list[str]:
        return [v.id for v in self.values]

    def value(self, vid: str) -> Value:
        for v in self.values:
            if v.id == vid:
                return v
        raise KeyError(f"{self.id}: unknown value {vid!r}")


@dataclass
class Taxonomy:
    version: int
    nodes: list[Node]
    rating: dict
    minor_block: dict
    source_sha256: str

    def node(self, nid: str) -> Node:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    @property
    def bands(self) -> list[str]:
        return list(self.rating["bands"])

    def thresholds(self) -> dict[str, float]:
        return dict(self.rating["default_thresholds"])


# Default per-evaluator abstention threshold when a node sets none. SigLIP
# softmax over 5–12 hypotheses rarely exceeds 0.5 even when right, so its
# floor is lower than WD14's (WD14 tags already passed the tagger's 0.35).
DEFAULT_MIN_CONF = {"wd14": 0.35, "siglip": 0.30, "vlm": 0.0, "head": 0.40, "human": 0.0}


def load(path: Path | str = TAXONOMY_PATH) -> Taxonomy:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes)
    nodes = []
    seen = set()
    for n in raw["nodes"]:
        if n["id"] in seen:
            raise ValueError(f"duplicate node {n['id']}")
        seen.add(n["id"])
        if n["kind"] not in ("single", "multi", "ordinal"):
            raise ValueError(f"{n['id']}: bad kind {n['kind']}")
        tmpl = n.get("siglip_template", "{}")
        vals = []
        for vid, spec in (n.get("values") or {}).items():
            spec = spec or {}
            hyps = spec.get("siglip") or [tmpl.format(vid.replace("_", " "))]
            vals.append(Value(
                id=vid,
                wd14=[normalize_tag(t) for t in spec.get("wd14", [])],
                siglip=list(hyps),
                filterable=bool(spec.get("filterable", True)),
            ))
        if len(vals) < 2:
            raise ValueError(f"{n['id']}: needs at least two values")
        nodes.append(Node(
            id=n["id"], kind=n["kind"], values=vals, when=n.get("when"),
            facet=bool(n.get("facet", True)),
            human_required=bool(n.get("human_required", False)),
            label_cs=n.get("label_cs", ""), vlm=n.get("vlm", ""),
            min_conf=float(n.get("min_conf", 0.0)),
            wd14_negative={k: [normalize_tag(t) for t in v]
                           for k, v in (n.get("wd14_negative") or {}).items()},
        ))
    tax = Taxonomy(
        version=int(raw["version"]), nodes=nodes, rating=raw["rating"],
        minor_block=raw["minor_block"],
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    _validate(tax)
    return tax


def _validate(tax: Taxonomy) -> None:
    th = tax.thresholds()
    if not (0 < th["safe_max"] < th["suggestive_max"] < th["nsfw_max"] < 1):
        raise ValueError("rating thresholds must be increasing inside (0, 1)")
    if tax.bands != ["safe", "suggestive", "nsfw", "explicit"]:
        raise ValueError("bands are fixed: safe, suggestive, nsfw, explicit")
    age = tax.node("person.age_group")
    for v in tax.minor_block["values"]:
        if age.value(v).filterable:
            raise ValueError(f"minor value {v} must be filterable: false")
    for n in tax.nodes:
        if n.when not in (None, "people"):
            raise ValueError(f"{n.id}: unsupported when={n.when}")


def normalize_tag(t: str) -> str:
    """WD14 spelling as stored in captions.tags_json (spaces, not underscores)."""
    return t.strip().replace("_", " ")


def band_of(temp: float, th: dict[str, float]) -> str:
    if temp <= th["safe_max"]:
        return "safe"
    if temp <= th["suggestive_max"]:
        return "suggestive"
    if temp <= th["nsfw_max"]:
        return "nsfw"
    return "explicit"


BAND_RANK = {"safe": 0, "suggestive": 1, "nsfw": 2, "explicit": 3}


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def record(image: str, evaluator: str, model: str, tax: Taxonomy,
           labels: dict | None = None, rating: dict | None = None,
           ms: float | None = None, **extra) -> dict:
    r = {"image": image, "evaluator": evaluator, "model": model,
         "taxonomy_version": tax.version, "labels": labels or {}}
    if rating is not None:
        r["rating"] = rating
    if ms is not None:
        r["ms"] = round(ms, 2)
    r.update(extra)
    return r


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

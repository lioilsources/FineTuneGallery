"""taxonomy.yaml → the generated artefacts. Run after every edit of the YAML.

  python taxo/gen_schema.py                      # writes both files below
  python taxo/gen_schema.py --wd14-csv <selected_tags.csv>   # + vocabulary check
  python taxo/gen_schema.py --check              # exit 1 if outputs are stale

Outputs (both committed, both carry the YAML's sha256 so staleness is
detectable without a YAML parser — the Go test checks it):
  server/taxonomy.json            embedded in the Go server (facets, filter
                                  validation, minor-block rule, bands)
  taxo/schema/vlm_extract.schema.json   guided-JSON schema for Qwen3-VL
                                  (vLLM response_format=json_schema)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import taxolib

ROOT = taxolib.HERE.parent
SERVER_JSON = ROOT / "server" / "taxonomy.json"
VLM_SCHEMA = taxolib.HERE / "schema" / "vlm_extract.schema.json"

UNKNOWN = "unknown"


def server_doc(tax: taxolib.Taxonomy) -> dict:
    return {
        "generated_by": "taxo/gen_schema.py — do not edit, edit taxo/taxonomy.yaml",
        "version": tax.version,
        "source_sha256": tax.source_sha256,
        "bands": tax.bands,
        "default_thresholds": tax.thresholds(),
        "wd14_levels": tax.rating["wd14_levels"],
        "minor_block": {
            "node": "person.age_group",
            "values": tax.minor_block["values"],
            "min_conf": tax.minor_block["min_conf"],
            "wd14_signals": [taxolib.normalize_tag(t) for t in tax.minor_block["wd14_signals"]],
        },
        "nodes": [
            {
                "id": n.id, "kind": n.kind, "when": n.when or "", "facet": n.facet,
                "humanRequired": n.human_required, "labelCs": n.label_cs,
                "values": [{"id": v.id, "filterable": v.filterable} for v in n.values],
            }
            for n in tax.nodes
        ],
    }


def _leaf_schema(n: taxolib.Node) -> dict:
    vals = n.value_ids
    if n.kind == "multi":
        # No uniqueItems: xgrammar (vLLM's guided decoding) does not support
        # it; eval_vlm.py de-duplicates instead.
        return {"type": "array", "items": {"type": "string", "enum": vals},
                "description": n.vlm}
    return {"type": "string", "enum": vals + [UNKNOWN], "description": n.vlm}


def vlm_schema(tax: taxolib.Taxonomy) -> dict:
    """Nested object mirroring the dotted node ids. `person` is null when no
    person is depicted, so the model is never forced to invent an age or a
    skin tone for a landscape."""
    root: dict = {}
    for n in tax.nodes:
        parts = n.id.split(".")
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = _leaf_schema(n)

    def to_schema(tree: dict) -> dict:
        props = {}
        for k, v in tree.items():
            props[k] = v if "type" in v else to_schema(v)
        return {"type": "object", "properties": props,
                "required": list(props), "additionalProperties": False}

    s = to_schema(root)
    if "person" in s["properties"]:
        s["properties"]["person"] = {"anyOf": [s["properties"]["person"], {"type": "null"}]}
    # Advisory only — the temperature is the WD14 + NSFW ensemble. Recorded so
    # the VLM's opinion on rating gets its own κ against the human labels.
    s["properties"]["rating"] = {"type": "string", "enum": tax.bands,
                                 "description": "Content rating of the image."}
    s["required"].append("rating")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"ftg-taxonomy-v{tax.version}",
        "x-source-sha256": tax.source_sha256,
        **s,
    }


def check_wd14(tax: taxolib.Taxonomy, csv_path: Path) -> list[str]:
    with open(csv_path) as f:
        vocab = {taxolib.normalize_tag(r["name"]) for r in csv.DictReader(f) if r["category"] == "0"}
    missing = []
    for n in tax.nodes:
        for v in n.values:
            missing += [f"{n.id}.{v.id}: {t}" for t in v.wd14 if t not in vocab]
        for val, tags in n.wd14_negative.items():
            missing += [f"{n.id}.!{val}: {t}" for t in tags if t not in vocab]
    for t in tax.minor_block["wd14_signals"]:
        if taxolib.normalize_tag(t) not in vocab:
            missing.append(f"minor_block: {t}")
    return missing


def dump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd14-csv", type=Path, help="selected_tags.csv of the tagger, for a vocabulary check")
    ap.add_argument("--check", action="store_true", help="fail if generated files are stale")
    a = ap.parse_args(argv)

    tax = taxolib.load()
    if a.wd14_csv:
        miss = check_wd14(tax, a.wd14_csv)
        if miss:
            print("WD14 tags not in the tagger vocabulary:\n  " + "\n  ".join(miss), file=sys.stderr)
            return 1
        print(f"wd14: all mapped tags present in {a.wd14_csv.name}")

    outs = {SERVER_JSON: dump(server_doc(tax)), VLM_SCHEMA: dump(vlm_schema(tax))}
    if a.check:
        stale = [str(p) for p, txt in outs.items() if not p.exists() or p.read_text() != txt]
        if stale:
            print("stale: " + ", ".join(stale) + " — run python taxo/gen_schema.py", file=sys.stderr)
            return 1
        return 0
    for p, txt in outs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Evaluator 3: Qwen3-VL-8B structured extraction (vLLM, guided JSON).

  python taxo/eval_vlm.py --manifest m.jsonl --out vlm.jsonl \
      --base-url http://127.0.0.1:8004/v1 --model qwen3-vl-8b [--concurrency 16]

The response is constrained by taxo/schema/vlm_extract.schema.json (generated
from taxonomy.yaml) through the OpenAI `response_format: json_schema` that
vLLM implements with xgrammar, so every answer parses and every value is one
of the taxonomy's. "unknown" and person=null are abstentions. Temperature 0;
the alias + taxonomy version is the evaluator's identity, as with the judge.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

import images
import taxolib
from gen_schema import VLM_SCHEMA

MAX_SIDE = 896  # ~1k visual tokens for Qwen3-VL; enough for hair/eyes/medium

SYSTEM = (
    "You label AI-generated images for a gallery's content taxonomy. "
    "Describe only what is visibly rendered. Use \"unknown\" when something is not visible or not "
    "determinable; set person to null when no person is depicted. Age group and skin tone describe "
    "how the figure is drawn, not a real person's identity. Answer with JSON only."
)


def _prompt(tax: taxolib.Taxonomy) -> str:
    lines = ["Fill every field of the schema for this image. Field guide:"]
    for n in tax.nodes:
        vals = ", ".join(n.value_ids)
        many = " (list, may be empty)" if n.kind == "multi" else ""
        lines.append(f"- {n.id}{many}: {n.vlm} [{vals}]")
    lines.append(f"- rating: content rating [{', '.join(tax.bands)}]")
    return "\n".join(lines)


def _data_url(path: str) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def flatten(tax: taxolib.Taxonomy, ans: dict) -> dict:
    labels = {}
    for n in tax.nodes:
        cur = ans
        for p in n.id.split("."):
            cur = cur.get(p) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is None:
            continue
        if n.kind == "multi":
            if isinstance(cur, list):
                vals = [v for v in n.value_ids if v in set(cur)]
                labels[n.id] = {"value": vals, "conf": {v: 1.0 for v in vals}}
        elif cur in n.value_ids:
            labels[n.id] = {"value": cur, "conf": 1.0}
    # person == null means "no person": make subject consistent with it
    if ans.get("person") is None and "subject" in labels and "people" in labels["subject"]["value"]:
        labels["subject"]["value"].remove("people")
    return labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    images.add_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8004/v1")
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args(argv)

    tax = taxolib.load()
    schema = json.loads(VLM_SCHEMA.read_text())
    for k in ("$schema", "title", "x-source-sha256"):
        schema.pop(k, None)
    prompt = _prompt(tax)
    items = images.source(a)
    sess = requests.Session()

    def one(item):
        iid, path = item
        body = {
            "model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": f"ftg_taxonomy_v{tax.version}", "schema": schema, "strict": True}},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(path)}},
                    {"type": "text", "text": prompt}]},
            ],
        }
        t = time.perf_counter()
        try:
            r = sess.post(a.base_url.rstrip("/") + "/chat/completions", json=body, timeout=300)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]["content"]
            ans = json.loads(msg)
            err = None
        except Exception as e:  # noqa: BLE001 — recorded, never fatal for the batch
            ans, err = {}, str(e)[:300]
        ms = (time.perf_counter() - t) * 1000
        rec = taxolib.record(iid, "vlm", a.model, tax, labels=flatten(tax, ans), ms=ms,
                             vlm_rating=ans.get("rating"))
        if err:
            rec["error"] = err
        return rec

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as pool:
        rows = list(pool.map(one, items))
    wall = time.perf_counter() - t0
    taxolib.write_jsonl(a.out, rows)
    errs = sum(1 for r in rows if "error" in r)
    print(json.dumps({"vlm_model": a.model, "images": len(rows), "errors": errs, "wall_s": round(wall, 1),
                      "img_per_h": round(len(rows) / wall * 3600) if wall else None}), file=sys.stderr)
    return 0 if errs < len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

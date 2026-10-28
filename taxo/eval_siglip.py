"""Evaluator 2: SigLIP 2 zero-shot per taxonomy node (+ the image embedding).

  python taxo/eval_siglip.py --manifest m.jsonl --out siglip.jsonl --emb-out emb.npz [--device cuda]

Per node, every value has one or more text hypotheses (taxonomy.yaml
values.<v>.siglip, default from siglip_template). A value's logit is the max
over its hypotheses; single/ordinal nodes take the softmax over values,
multi nodes keep every value whose softmax share is ≥ MULTI_REL × the top one
and ≥ MULTI_ABS / k. New values in the YAML work immediately, no training.

The L2-normalised image embedding (1152-d, float16) is saved for the
multi-head classifier (head.py) and the gallery's similar-images search.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

import images
import taxolib

REPO = "google/siglip2-so400m-patch16-512"
MULTI_REL = 0.5
MULTI_ABS = 1.5


def _feat(out):
    return out if torch.is_tensor(out) else out.pooler_output


class Siglip:
    def __init__(self, tax: taxolib.Taxonomy, device: str):
        self.tax, self.device = tax, device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = AutoProcessor.from_pretrained(REPO)
        self.model = AutoModel.from_pretrained(REPO, dtype=self.dtype).to(device).eval()
        # Text side once: every hypothesis of every value of every node.
        self.slices = {}
        texts = []
        for n in tax.nodes:
            for v in n.values:
                self.slices[(n.id, v.id)] = (len(texts), len(texts) + len(v.siglip))
                texts += v.siglip
        with torch.inference_mode():
            t = self.proc(text=texts, padding="max_length", max_length=64, return_tensors="pt", truncation=True)
            tf = _feat(self.model.get_text_features(**{k: v.to(device) for k, v in t.items()}))
            self.text = torch.nn.functional.normalize(tf.float(), dim=-1)
        self.scale = self.model.logit_scale.exp().float()
        self.bias = self.model.logit_bias.float()

    @torch.inference_mode()
    def embed(self, ims) -> torch.Tensor:
        x = self.proc(images=[im.convert("RGB") for im in ims], return_tensors="pt")["pixel_values"]
        f = _feat(self.model.get_image_features(pixel_values=x.to(self.device, self.dtype)))
        return torch.nn.functional.normalize(f.float(), dim=-1)

    def labels(self, emb: torch.Tensor) -> list[dict]:
        logits = (emb @ self.text.T) * self.scale + self.bias  # [B, T]
        out = []
        for row in logits:
            labels = {}
            people = None
            for n in self.tax.nodes:
                if n.when == "people" and not people:
                    continue
                vl = torch.stack([row[a:b].max() for a, b in (self.slices[(n.id, v.id)] for v in n.values)])
                p = vl.softmax(-1).tolist()
                probs = {v.id: round(pi, 4) for v, pi in zip(n.values, p)}
                thr = max(n.min_conf, taxolib.DEFAULT_MIN_CONF["siglip"])
                if n.kind == "multi":
                    top = max(p)
                    k = len(p)
                    keep = [v for v, pi in probs.items() if pi >= MULTI_REL * top and pi >= MULTI_ABS / k]
                    labels[n.id] = {"value": keep, "conf": {v: probs[v] for v in keep}}
                else:
                    best = max(probs, key=probs.get)
                    if probs[best] >= thr:
                        labels[n.id] = {"value": best, "conf": probs[best]}
                if n.id == "subject":
                    people = "people" in labels["subject"]["value"]
            out.append(labels)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    images.add_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb-out", help="npz with ids + float16 embeddings")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args(argv)

    tax = taxolib.load()
    items = images.source(a)
    m = Siglip(tax, a.device)
    rows, embs, ids_all = [], [], []
    t0 = time.perf_counter()
    for ids, ims in images.batches(items, a.batch):
        tb = time.perf_counter()
        e = m.embed(ims)
        labs = m.labels(e)
        ms = (time.perf_counter() - tb) * 1000 / len(ims)
        for iid, lab in zip(ids, labs):
            rows.append(taxolib.record(iid, "siglip", REPO, tax, labels=lab, ms=ms))
        embs.append(e.cpu().numpy().astype(np.float16))
        ids_all += ids
    wall = time.perf_counter() - t0
    taxolib.write_jsonl(a.out, rows)
    if a.emb_out:
        np.savez(a.emb_out, ids=np.array(ids_all), emb=np.concatenate(embs), model=REPO)
    print(json.dumps({"siglip_device": a.device, "images": len(rows), "wall_s": round(wall, 1),
                      "img_per_h": round(len(rows) / wall * 3600)}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

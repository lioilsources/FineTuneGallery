"""Temperature raters 2 and 3: two binary NSFW classifiers (ViT-base).

  python taxo/eval_nsfw.py --manifest m.jsonl --out nsfw.jsonl [--device cuda]

Writes one record per image per classifier (evaluator falconsai / adamcodd)
with rating.temperature = p(nsfw). The ensemble with WD14 is combine.py.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

import images
import taxolib

MODELS = {
    "falconsai": "Falconsai/nsfw_image_detection",
    "adamcodd": "AdamCodd/vit-base-nsfw-detector",
}


class Binary:
    def __init__(self, repo: str, device: str):
        self.repo = repo
        self.proc = AutoImageProcessor.from_pretrained(repo)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForImageClassification.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.device, self.dtype = device, dtype
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        nsfw = [i for i, l in labels.items() if "nsfw" in l]
        if len(nsfw) != 1:
            raise RuntimeError(f"{repo}: cannot find the nsfw label in {labels}")
        self.nsfw = nsfw[0]

    @torch.inference_mode()
    def __call__(self, ims) -> list[float]:
        x = self.proc(images=[im.convert("RGB") for im in ims], return_tensors="pt")["pixel_values"]
        logits = self.model(pixel_values=x.to(self.device, self.dtype)).logits.float()
        return logits.softmax(-1)[:, self.nsfw].tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    images.add_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--only", choices=list(MODELS), action="append")
    a = ap.parse_args(argv)

    tax = taxolib.load()
    items = images.source(a)
    rows = []
    for name in a.only or list(MODELS):
        m = Binary(MODELS[name], a.device)
        t0 = time.perf_counter()
        for ids, ims in images.batches(items, a.batch):
            tb = time.perf_counter()
            ps = m(ims)
            ms = (time.perf_counter() - tb) * 1000 / len(ims)
            for iid, p in zip(ids, ps):
                rows.append(taxolib.record(iid, name, m.repo, tax,
                                           rating={"temperature": round(p, 4)}, ms=ms))
        wall = time.perf_counter() - t0
        print(f"{name}[{a.device}]: {len(items)} images in {wall:.1f}s → {len(items) / wall * 3600:.0f} img/h",
              file=sys.stderr)
    taxolib.write_jsonl(a.out, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

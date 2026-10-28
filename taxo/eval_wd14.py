"""Evaluator 1: WD14 (wd-swinv2-tagger-v3, ONNX) → taxonomy + WD14 temperature.

  python taxo/eval_wd14.py --manifest m.jsonl --out wd14.jsonl [--device cuda]

Same preprocessing and thresholds as tagger/main.py (the gallery sidecar), so
these labels equal what the server derives from captions.tags_json. On the GPU
box it is the same ONNX graph on CUDAExecutionProvider and batched.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

import images
import taxolib
import temperature
import wd14map

REPO = "SmilingWolf/wd-swinv2-tagger-v3"
THRESHOLD = 0.35  # tagger/main.py TAGGER_THRESHOLD


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    # identical to tagger/main.py _preprocess
    image = image.convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")
    w, h = image.size
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(image, ((side - w) // 2, (side - h) // 2))
    if side != size:
        padded = padded.resize((size, size), Image.BICUBIC)
    return np.asarray(padded, dtype=np.float32)[:, :, ::-1]


class WD14:
    def __init__(self, device: str = "cpu", threads: int = 0):
        model_path = hf_hub_download(REPO, "model.onnx")
        df = pd.read_csv(hf_hub_download(REPO, "selected_tags.csv"))
        self.names = [n.replace("_", " ") for n in df["name"].tolist()]
        self.rating_idx = list(np.where(df["category"] == 9)[0])
        self.general_idx = list(np.where(df["category"] == 0)[0])
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=prov)
        self.size = self.sess.get_inputs()[0].shape[1]
        self.inp = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]

    def __call__(self, ims: list[Image.Image]) -> list[dict]:
        x = np.stack([preprocess(im, self.size) for im in ims])
        probs = self.sess.run(None, {self.inp: x})[0]
        out = []
        for p in probs:
            out.append({
                "rating": {self.names[i]: round(float(p[i]), 4) for i in self.rating_idx},
                "general": {self.names[i]: round(float(p[i]), 4) for i in self.general_idx if p[i] >= THRESHOLD},
            })
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    images.add_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)

    tax = taxolib.load()
    items = images.source(a)
    m = WD14(a.device, a.threads)
    rows, t0 = [], time.perf_counter()
    for ids, ims in images.batches(items, a.batch):
        tb = time.perf_counter()
        res = m(ims)
        ms = (time.perf_counter() - tb) * 1000 / len(ims)
        for iid, r in zip(ids, res):
            t = temperature.wd14_temperature(tax, r["rating"])
            rows.append(taxolib.record(
                iid, "wd14", REPO, tax,
                labels=wd14map.map_tags(tax, r["general"]),
                rating={"temperature": None if t is None else round(t, 4), "parts": {"wd14": r["rating"]}},
                ms=ms, tags=r["general"], minor_signal=round(wd14map.minor_signal(tax, r["general"]), 4)))
    wall = time.perf_counter() - t0
    taxolib.write_jsonl(a.out, rows)
    print(f"wd14[{m.provider}]: {len(rows)} images in {wall:.1f}s → {len(rows) / wall * 3600:.0f} img/h", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

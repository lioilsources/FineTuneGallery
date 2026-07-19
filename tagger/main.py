"""WD14 tagger sidecar — booru-style auto-captioning on CPU.

POST /tag  (body: raw image bytes)  →
  {"general": {tag: conf}, "character": {…}, "rating": {…}, "model": "…"}

Model: SmilingWolf/wd-swinv2-tagger-v3 (ONNX), downloaded to /cache on first
start. ~1–2 s/image on NAS-class CPU — fine for an unattended backlog.
"""

import io
import os
import threading

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, Request, Response
from huggingface_hub import hf_hub_download
from PIL import Image

MODEL_REPO = os.environ.get("TAGGER_MODEL", "SmilingWolf/wd-swinv2-tagger-v3")
THRESHOLD = float(os.environ.get("TAGGER_THRESHOLD", "0.35"))
CHARACTER_THRESHOLD = float(os.environ.get("TAGGER_CHARACTER_THRESHOLD", "0.75"))

app = FastAPI()
_lock = threading.Lock()
_session = None
_tags = None  # (names, rating_idx, general_idx, character_idx)


def _load():
    global _session, _tags
    with _lock:
        if _session is not None:
            return
        model_path = hf_hub_download(MODEL_REPO, "model.onnx")
        csv_path = hf_hub_download(MODEL_REPO, "selected_tags.csv")
        df = pd.read_csv(csv_path)
        names = df["name"].tolist()
        _tags = (
            names,
            list(np.where(df["category"] == 9)[0]),
            list(np.where(df["category"] == 0)[0]),
            list(np.where(df["category"] == 4)[0]),
        )
        _session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )


def _preprocess(image: Image.Image, size: int) -> np.ndarray:
    # Reference WD14 preprocessing: pad to square on white, resize, BGR.
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

    arr = np.asarray(padded, dtype=np.float32)[:, :, ::-1]  # RGB → BGR
    return arr[np.newaxis, :]


@app.get("/healthz")
def healthz():
    return {"status": "ok", "loaded": _session is not None}


@app.post("/tag")
async def tag(request: Request):
    body = await request.body()
    if not body:
        return Response(
            content='{"error":"empty body"}', status_code=400,
            media_type="application/json",
        )
    _load()
    try:
        image = Image.open(io.BytesIO(body))
    except Exception as e:  # noqa: BLE001
        return Response(
            content=f'{{"error":"cannot decode image: {e}"}}', status_code=400,
            media_type="application/json",
        )

    height = _session.get_inputs()[0].shape[1]
    inp = _preprocess(image, height)
    input_name = _session.get_inputs()[0].name
    with _lock:  # onnxruntime sessions are thread-safe, but keep memory flat
        probs = _session.run(None, {input_name: inp})[0][0]

    names, rating_idx, general_idx, character_idx = _tags
    out = {"general": {}, "character": {}, "rating": {}, "model": MODEL_REPO}
    for i in rating_idx:
        out["rating"][names[i]] = round(float(probs[i]), 4)
    for i in general_idx:
        p = float(probs[i])
        if p >= THRESHOLD:
            out["general"][names[i].replace("_", " ")] = round(p, 4)
    for i in character_idx:
        p = float(probs[i])
        if p >= CHARACTER_THRESHOLD:
            out["character"][names[i].replace("_", " ")] = round(p, 4)
    return out

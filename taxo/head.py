"""Multi-head classifier over frozen SigLIP 2 embeddings (distillation of the
evaluators, later of the human labels). One shared trunk, one head per
taxonomy node, one regression head for the temperature.

  python taxo/head.py train --emb emb.npz --labels combined.jsonl --out head/ \
      [--human human.jsonl] [--min-agree 2] [--epochs 200]
  python taxo/head.py predict --emb emb.npz --model head/ --out head.jsonl
  python taxo/head.py bench --model head/        # ONNX latency, batch 1 and 64

Training labels per node, in order of preference:
  human label  >  consensus (not unsure, ≥ --min-agree evaluators agreeing)
Everything else is masked out of the loss — a node nobody agreed on teaches
nothing. Temperature target = the ensemble temperature (human band midpoint
when a human rated it).

Outputs: head/head.onnx (input `emb` [B, D] float32 → `logits_<node>` per
node + `temperature` [B]), head/meta.json (node/value order, taxonomy version,
held-out metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import taxolib

BAND_MID = {"safe": 0.1, "suggestive": 0.33, "nsfw": 0.6, "explicit": 0.9}


class Head(nn.Module):
    def __init__(self, dim: int, nodes: list[taxolib.Node], hidden: int = 512):
        super().__init__()
        self.nodes = nodes
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2))
        self.heads = nn.ModuleList([nn.Linear(hidden, len(n.values)) for n in nodes])
        self.temp = nn.Linear(hidden, 1)

    def forward(self, emb):
        h = self.trunk(emb)
        return [hd(h) for hd in self.heads], torch.sigmoid(self.temp(h)).squeeze(-1)


def _targets(tax, ids, combined, human, min_agree):
    """→ per node: (y [N, k] float, mask [N] bool); temperature (t [N], mask)."""
    by = {r["image"]: r for r in combined}
    hum = {r["image"]: r for r in human}
    N = len(ids)
    ys, ms = [], []
    for n in tax.nodes:
        k = len(n.values)
        y = np.zeros((N, k), np.float32)
        m = np.zeros(N, bool)
        for i, iid in enumerate(ids):
            lab = None
            if iid in hum and n.id in hum[iid]["labels"]:
                lab = hum[iid]["labels"][n.id]["value"]
            elif iid in by:
                c = by[iid]["consensus"].get(n.id)
                if c and not c["unsure"] and c["n"] >= min_agree:
                    lab = c["value"]
            if lab is None:
                continue
            if n.kind == "multi":
                for v in lab:
                    y[i, n.value_ids.index(v)] = 1
            else:
                y[i, n.value_ids.index(lab)] = 1
            m[i] = True
        ys.append(y)
        ms.append(m)
    t = np.zeros(N, np.float32)
    tm = np.zeros(N, bool)
    for i, iid in enumerate(ids):
        if iid in hum and hum[iid].get("rating", {}).get("band"):
            t[i], tm[i] = BAND_MID[hum[iid]["rating"]["band"]], True
        elif iid in by and by[iid]["rating"].get("temperature") is not None:
            t[i], tm[i] = by[iid]["rating"]["temperature"], True
    return ys, ms, t, tm


def _f1(y_true, y_pred, multi: bool):
    """macro F1 over the values present in y_true."""
    if not len(y_true):
        return None
    if multi:
        f = []
        for j in range(y_true.shape[1]):
            tp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 1)).sum())
            fp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 0)).sum())
            fn = int(((y_pred[:, j] == 0) & (y_true[:, j] == 1)).sum())
            if tp + fn == 0:
                continue
            f.append(2 * tp / (2 * tp + fp + fn))
        return float(np.mean(f)) if f else None
    t = y_true.argmax(1)
    p = y_pred
    f = []
    for c in np.unique(t):
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def train(a) -> int:
    tax = taxolib.load()
    z = np.load(a.emb)
    ids = [str(x) for x in z["ids"]]
    X = torch.tensor(z["emb"].astype(np.float32))
    combined = taxolib.read_jsonl(a.labels)
    human = taxolib.read_jsonl(a.human) if a.human else []
    ys, ms, t, tm = _targets(tax, ids, combined, human, a.min_agree)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(len(ids) * a.val_frac))
    val, tr = perm[:n_val], perm[n_val:]

    model = Head(X.shape[1], tax.nodes, a.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    Y = [torch.tensor(y) for y in ys]
    M = [torch.tensor(m) for m in ms]
    T, TM = torch.tensor(t), torch.tensor(tm)
    tr_t = torch.tensor(tr)
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        idx = tr_t[torch.randperm(len(tr_t))]
        for b in range(0, len(idx), a.batch):
            bi = idx[b: b + a.batch]
            outs, tp = model(X[bi])
            loss = torch.zeros(())
            for n, o, y, m in zip(tax.nodes, outs, Y, M):
                mm = m[bi]
                if not mm.any():
                    continue
                if n.kind == "multi":
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(o[mm], y[bi][mm])
                else:
                    loss = loss + nn.functional.cross_entropy(o[mm], y[bi][mm].argmax(1))
            if TM[bi].any():
                loss = loss + 4 * nn.functional.mse_loss(tp[TM[bi]], T[bi][TM[bi]])
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        outs, tp = model(X[torch.tensor(val)])
    metrics = {}
    for n, o, y, m in zip(tax.nodes, outs, ys, ms):
        mv = m[val]
        if mv.sum() == 0:
            metrics[n.id] = {"n_val": 0, "n_train": int(m[tr].sum())}
            continue
        if n.kind == "multi":
            pred = (torch.sigmoid(o[torch.tensor(mv)]) >= 0.5).int().numpy()
        else:
            pred = o[torch.tensor(mv)].argmax(1).numpy()
        metrics[n.id] = {"n_val": int(mv.sum()), "n_train": int(m[tr].sum()),
                         "f1_macro_vs_labels": _f1(y[val][mv], pred, n.kind == "multi")}
    tv = tm[val]
    mae = float(np.abs(tp.numpy()[tv] - t[val][tv]).mean()) if tv.any() else None

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "taxonomy_version": tax.version, "taxonomy_sha256": tax.source_sha256,
        "embedding_model": str(z["model"]) if "model" in z else "google/siglip2-so400m-patch16-512",
        "dim": int(X.shape[1]), "hidden": a.hidden,
        "nodes": [{"id": n.id, "kind": n.kind, "values": n.value_ids} for n in tax.nodes],
        "label_source": "human+consensus" if human else "consensus pseudo-labels (re-train on human labels)",
        "train": {"images": len(tr), "val": len(val), "epochs": a.epochs, "seconds": round(train_s, 2)},
        "val_metrics": metrics, "val_temperature_mae": mae,
        "val_ids": [ids[i] for i in val],
    }
    torch.save(model.state_dict(), out / "head.pt")
    export_onnx(model, tax, X.shape[1], out / "head.onnx")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"train_s": meta["train"]["seconds"], "val_temperature_mae": mae,
                      "f1": {k: v.get("f1_macro_vs_labels") for k, v in metrics.items()}}), file=sys.stderr)
    return 0


def export_onnx(model, tax, dim, path):
    class Flat(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, emb):
            outs, t = self.m(emb)
            return (*outs, t)

    names = [f"logits_{n.id}" for n in tax.nodes] + ["temperature"]
    torch.onnx.export(Flat(model.eval()), (torch.zeros(1, dim),), str(path), input_names=["emb"],
                      output_names=names, dynamic_axes={"emb": {0: "b"}, **{k: {0: "b"} for k in names}},
                      opset_version=17, dynamo=False)


def _session(model_dir):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    return ort.InferenceSession(str(Path(model_dir) / "head.onnx"), so, providers=["CPUExecutionProvider"])


def predict(a) -> int:
    tax = taxolib.load()
    meta = json.loads((Path(a.model) / "meta.json").read_text())
    if meta["taxonomy_version"] != tax.version:
        print("head was trained on another taxonomy version", file=sys.stderr)
        return 1
    sess = _session(a.model)
    z = np.load(a.emb)
    ids = [str(x) for x in z["ids"]]
    X = z["emb"].astype(np.float32)
    rows = []
    t0 = time.perf_counter()
    outs = sess.run(None, {"emb": X})
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(ids))
    for i, iid in enumerate(ids):
        labels = {}
        for n, o in zip(tax.nodes, outs[:-1]):
            lo = o[i]
            if n.kind == "multi":
                p = 1 / (1 + np.exp(-lo))
                vals = [v for v, pi in zip(n.value_ids, p) if pi >= 0.5]
                labels[n.id] = {"value": vals, "conf": {v: round(float(pi), 4) for v, pi in zip(n.value_ids, p) if pi >= 0.5}}
            else:
                p = np.exp(lo - lo.max())
                p /= p.sum()
                j = int(p.argmax())
                if p[j] >= taxolib.DEFAULT_MIN_CONF["head"]:
                    labels[n.id] = {"value": n.value_ids[j], "conf": round(float(p[j]), 4)}
        if "subject" in labels and "people" not in labels["subject"]["value"]:
            labels = {k: v for k, v in labels.items() if not tax.node(k).when}
        rows.append(taxolib.record(iid, "head", f"ftg-head@v{tax.version}", tax, labels=labels,
                                   rating={"temperature": round(float(outs[-1][i]), 4)}, ms=ms))
    taxolib.write_jsonl(a.out, rows)
    return 0


def bench(a) -> int:
    meta = json.loads((Path(a.model) / "meta.json").read_text())
    sess = _session(a.model)
    res = {}
    for b in (1, 64):
        x = np.random.default_rng(0).standard_normal((b, meta["dim"])).astype(np.float32)
        for _ in range(20):
            sess.run(None, {"emb": x})
        n = 500 if b == 1 else 50
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(None, {"emb": x})
        res[f"batch{b}_ms_per_call"] = round((time.perf_counter() - t0) * 1000 / n, 4)
        res[f"batch{b}_ms_per_image"] = round(res[f"batch{b}_ms_per_call"] / b, 4)
    res["onnx_bytes"] = (Path(a.model) / "head.onnx").stat().st_size
    print(json.dumps(res))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--emb", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--human")
    t.add_argument("--out", required=True)
    t.add_argument("--min-agree", type=int, default=2)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("predict")
    p.add_argument("--emb", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    torch.manual_seed(getattr(a, "seed", 1))
    return {"train": train, "predict": predict, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

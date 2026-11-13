#!/usr/bin/env bash
# UC25 krok 4 na SPARKu: hlava teploty + conformal prahy z lidských labelů.
# Spouští broker jako dávku projektu finetune, nikdy ručně:
#
#   sparkctl run finetune comfyui 10 \
#     "cd ~/deploy/FineTuneGallery && taxo/run_train.sh taxo-data/human.jsonl" "UC25 hlava teploty + conformal"
#
# human.jsonl = `finetune-gallery taxobench -export-human` z galerie (jen id a labely).
# Vstupy combined.jsonl a emb.npz z run_eval.sh. Výstupy v taxo-data/private/train/:
# head/ (head.onnx, meta.json), bands.json (fit nad teplotou ensemblu, ten se
# z labelů neučí; hlava jde online až s vlastní kalibrací), report.json.
# Na stdout jde poslední řádek jen s čísly (metriky pro sparkctl done).
set -uo pipefail
cd "$(dirname "$0")/.."
H=${1:?human.jsonl}
T=taxo; E=taxo-data/private/eval; O=taxo-data/private/train
umask 077; mkdir -p $O
export HF_HUB_OFFLINE=1
PY=$T/.venv/bin/python
log() { echo "$(date -Is) $*" | tee -a $O/run.log >&2; }
log "labely $(wc -l < "$H")"
fail=0
$PY $T/head.py train --emb $E/emb.npz --labels $E/combined.jsonl --human "$H" --out $O/head \
  2>$O/head.err || fail=$((fail+1)); log "head $(tail -1 $O/head.err)"
$PY $T/conformal.py fit --scores $E/combined.jsonl --labels "$H" --out $O/bands.json \
  2>$O/conformal.err || fail=$((fail+1)); log "conformal $(tail -1 $O/conformal.err)"
$PY - "$O" <<'EOF' || fail=$((fail+1))
import json, sys
o = sys.argv[1]
meta = json.load(open(f"{o}/head/meta.json"))
fit = json.load(open(f"{o}/bands.json"))
hb = meta.get("val_band_vs_human") or {}
rep = {
    "valid": fit["valid"], "label_counts": fit["label_counts"], "thresholds": fit["thresholds"],
    "explicit_needed": fit["explicit_needed"],
    "explicit_recall_cal": fit["calibration"]["explicit_recall_with_guard"],
    "strata": fit.get("strata"),
    "head": {"val_temperature_mae": meta["val_temperature_mae"], **hb,
             "excluded_minor": meta.get("excluded_minor")},
}
json.dump(rep, open(f"{o}/report.json", "w"), indent=1)
print(json.dumps({"valid": fit["valid"], "explicit": fit["label_counts"].get("explicit"),
                  "f1": hb.get("f1_macro"), "recall_explicit": hb.get("explicit_recall"),
                  "mae": meta["val_temperature_mae"]}))
EOF
log "hotovo, chybných kroků $fail"
[ $fail -eq 0 ]

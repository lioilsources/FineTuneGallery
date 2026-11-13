#!/usr/bin/env bash
# UC25 hodnotitelé teploty na SPARKu (SPARK verze uc25/gpu_batch.sh, fáze A).
# Spouští broker jako dávku projektu finetune v profilu comfy, nikdy ručně:
#
#   sparkctl run finetune comfyui 10 \
#     "cd ~/deploy/FineTuneGallery && taxo/run_eval.sh taxo-data/manifest_all.jsonl" "UC25 hodnotitelé teploty"
#
# Manifest {"image","path"} na obrázky v taxo-data/private/. Výstupy (skóre, embeddingy)
# jdou do taxo-data/private/eval/ — jsou odvozené z private/, takže zůstávají tam.
# Na stdout jde poslední řádek jen s počty (metriky pro sparkctl done).
set -uo pipefail
cd "$(dirname "$0")/.."
M=${1:?manifest}
T=taxo; O=taxo-data/private/eval
umask 077; mkdir -p $O
export HF_HUB_OFFLINE=1
PY=$T/.venv/bin/python
log() { echo "$(date -Is) $*" | tee -a $O/run.log >&2; }
n=$(wc -l < "$M"); log "manifest $n obrázků"
fail=0
# WD14 (ONNX) z cache taggeru; ONNX GPU build pro aarch64 na PyPI není → CPU, víc vláken
HF_HUB_CACHE=$PWD/tagger-cache/hub $PY $T/eval_wd14.py --manifest "$M" --out $O/wd14.jsonl \
  --device cpu --batch 16 --threads ${WD14_THREADS:-12} 2>$O/wd14.err || fail=$((fail+1)); log "wd14 $(tail -1 $O/wd14.err)"
$PY $T/eval_nsfw.py --manifest "$M" --out $O/nsfw.jsonl --device cuda --batch 64 2>$O/nsfw.err \
  || fail=$((fail+1)); log "nsfw $(tail -1 $O/nsfw.err)"
$PY $T/eval_siglip.py --manifest "$M" --out $O/siglip.jsonl --emb-out $O/emb.npz --device cuda --batch 32 \
  2>$O/siglip.err || fail=$((fail+1)); log "siglip $(tail -1 $O/siglip.err)"
$PY $T/combine.py --wd14 $O/wd14.jsonl --nsfw $O/nsfw.jsonl --siglip $O/siglip.jsonl \
  --out $O/combined.jsonl --stats $O/stats.json 2>>$O/run.log || fail=$((fail+1))
ok=$(wc -l < $O/combined.jsonl 2>/dev/null || echo 0)
log "hotovo: $ok/$n, chybných kroků $fail"
echo "{\"images\":{\"ok\":$ok,\"failed\":$((n-ok))}}"
[ $fail -eq 0 ]

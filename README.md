# finetune-gallery

Rating platforma + LoRA fine-tune pipeline pro [Ol1nLLM](../Ol1nLLM) Image
Studio. Appka exportuje generační stromy (prompty + obrázky + metadata) sem na
NAS; ve webové galerii se výstupy hodnotí (likes, kritiky, aspect tagy,
relační kritéria) a z ohodnocených dat se staví kohya-ss datasety pro SDXL
LoRA trénink na GPU boxu. Hotová LoRA se objeví v appce automaticky (ComfyUI
`/object_info/LoraLoader`).

```
Ol1nLLM appka ──export──▶ NAS (tento repo)            GPU box
                          Go API + Svelte web         make train DATASET=…
                          SQLite + blobs              kohya sd-scripts
                          WD14 tagger (CPU)     ──▶   → ComfyUI models/loras/
```

## Struktura

```
server/   Go 1.24 backend — ingest, galerie, datasety, kohya builder;
          embedne web (go:embed webdist), modernc.org/sqlite (CGO-free)
web/      Svelte 5 + Vite SPA (build → server/webdist)
tagger/   WD14 (wd-swinv2-tagger-v3) FastAPI sidecar, CPU ONNX
train/    kohya trainer pro GPU box (Docker) + deploy do ComfyUI
```

## Deploy (NAS)

```bash
cp .env.example .env            # DATA_DIR=/volume1/finetune
docker compose up -d --build
curl localhost:8092/healthz     # {"status":"ok"} (odkomentuj ports v compose)
```

Cloudflare je **předprovisionováno** (2026-07-19): dedikovaný tunel
`finetune-nas` (`8a88b8b7-dc02-4feb-8c64-e972aa50fd28`, config v
`cloudflared/config.yml`, běží jako služba v compose), DNS CNAME
`finetune.ol1n.com`, Access aplikace `finetune` se dvěma policy — **Service
Auth** (service token appky) + **Allow** (owner SSO e-mail). Jediný ruční
krok: umístit tunnel credentials do `cloudflared/credentials.json` (není
v gitu — viz NAS deploy plán). Server sám auth neřeší; connector navíc
vynucuje Access JWT (audTag) jako defense in depth.

Appka: `FINETUNE_URL=https://finetune.ol1n.com` (default) nebo LAN
`FINETUNE_URL=http://<nas-ip>:8092` v `.env.local` u Ol1nLLM.

## Data

```
/data/images/<ab>/<sha256>.png   content-addressed bloby (dedup zdarma)
/data/thumbs/                    lazy 384px JPEG
/data/db/finetune.sqlite         metadata, ratingy, captiony, datasety
/data/datasets/<name>/           built kohya balíčky
```

Caption vrstvy: `auto` (WD14 booru tagy, nikdy se nepřepisuje) < `refined`
(LLM, v1.5) < `human`. Do tréninku jde `trigger word, <efektivní caption>`.

## Trénink (GPU box)

```bash
cd train
cp .env.example .env             # NAS_DATASETS, CKPT_DIR, LORA_DIR
make build                       # jednorázově: kohya image
make train DATASET=jewelry_v1    # rsync → sdxl_train_network.py → cp do loras/
```

Balíček je self-contained (img/ + captiony + oba tomly + README s přesným
příkazem). Výstup `<aspect>_<ecosystem>_v1.safetensors` — **jméno nesmí
obsahovat „flux"** (heuristika LoRA rodin v appce); run.sh to hlídá.

## Ekosystémy

LoRA se přenáší dobře jen uvnitř ekosystému checkpointu, na kterém se
trénovala: **pony** (Pony V6, Atomix) / **illustrious** / **sdxl-base**
(Juggernaut). Zdrojové obrázky v datasetu můžou být od jiného modelu — to je
záměrná destilace (např. juggernaut obličeje → LoRA pro illustrious) — ale
musí to být vědomá volba, UI na to upozorňuje.

## API (výběr)

```
POST /api/ingest/manifest                  → {"needed":[sha…]}
PUT  /api/ingest/images/{sha256}           raw PNG
POST /api/ingest/sessions/{id}/finalize    → {"images":N,"newBlobs":M}
GET  /api/images?model=&aspect=&score=&criterion=pose_adherence:1&cursor=
GET  /api/images/{id}                      detail + parentChain
PUT  /api/images/{id}/rating|aspects|criteria|caption
POST /api/images/{id}/autocaption
GET  /api/datasets · POST …/items/from-filter · POST …/build · GET …/download
GET  /api/stats · /api/meta · /healthz
```

## Vývoj

```bash
cd server && go run .                                # API na :8092
cd web && npm install && npm run dev                 # Vite dev + proxy na :8092
cd web && npm run build                              # → server/webdist (embed)
```

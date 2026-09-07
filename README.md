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

## Eval harness

Relační kritéria (`pose_adherence`, `source_identity`, `source_style`) hodnotí
výstup proti tomu, co ho podmiňovalo — jsou to jediné ratingy tady, které umí
odpovědět na „který checkpoint drží pózu". Záložka **Eval** je ta odpověď:
rozřež ohodnocený korpus podle libovolné dimenze (model / art styl / LoRA /
póza / session) a porovnej.

```
Eval → group by: Model → pose adherence
  Pony V6          90 %  ≥75 %  n=31    ← vede
  Juggernaut XL    86 %  ≥66 %  n=21
  Animagine XL 4  100 %  ≥29 %  n=3     ← neřadí se, pod prahem
  NoobAI XL        n/a                  ← žádný obrázek s pózou
```

Tři věci, na kterých to stojí:

- **Řadí se podle dolní meze 95% Wilsonova intervalu, ne podle poměru.**
  3/3 vypadá na poměr líp než 36/40 a na žebříčku před ním stát nesmí. Sloupec
  `min ratings` je práh, pod kterým se skupina vůbec nekoruje.
- **Pokrytí je per kritérium.** Póza se nedá hodnotit na obrázku bez pózy, a
  `source_*` na obrázku, co nevznikl z jiného. Buňka proto rozlišuje tři stavy:
  `n/a` (nelze hodnotit) ≠ `rate N` (lze, nikdo nehodnotil) ≠ `0 %`.
- **Klik na buňku vede na to, co ještě chybí.** Galerie se otevře přesně na
  neohodnocených obrázcích té skupiny (`criterion=<name>:none`) — harness řekne,
  kde je důkazů málo, a odkaz tě pošle je doplnit.

LoRA se seskupuje podle jména **a síly** (`face_v1 @ 0.40` a `face_v1 @ 1.20`
jsou dva různé experimenty). `CSV` exportuje celou matici.

### Osa `medium` — první A/B, na který je harness postavený

Stylové bloky v appce samy pojmenovávají render medium („woodblock print",
„stone relief"), ale prompt žádné nedeklaruje. Stylová matice appky
(`Ol1nLLM/docs/style-matrix.md`) zaznamenala následek: tradice, jejichž celé
tvrzení *je* medium (asyrský, mezopotámský, hebrejský reliéf), spadnou na
většině modelů na béžovou stěnu. Hypotéza převzatá z Tsumiki
(`MangaPrompts/assets/config/blocks/medium.yaml`): medium musí být vlastní,
dopředu umístěná osa, ne vlečná věta.

Sloupec `nodes.medium_id` (migrace v3) je způsob, jak to rozsoudit:

```
lab run --subject "a ballerina" --models juggernaut-xl         --styles assyrian,mesopotamian,hebrew --mediums __none,medium_illustration
lab export build/lab/<běh> --send

Eval → group by Medium, scope: style=assyrian
  medium_illustration   71 %  ≥50 %  n=21
  (no medium)           27 %  ≥11 %  n=15     ← kontrolní rameno
```

Dvě věci, které z toho dělají experiment a ne dojem:

- **Kontrolní rameno musí být dosažitelné.** `medium=none` je proto plnohodnotná
  hodnota filtru — bez ní se na obrázky, proti kterým se porovnává, nedá dostat.
- **Dolní meze musí přestat sousedit.** Při ~36 hodnoceních se interval 50–86 %
  a 11–52 % sotva rozpojí. To je minimum, ne cíl; když se meze překrývají,
  odpověď zní „ještě nevím", ne „vyrovnané".

Typický běh: vygeneruj stejné prompty přes N checkpointů → oštítkuj kritéria →
Eval → group by Model. Žebříček místo dojmů.

## Překlad promptu do jazyka modelu

Pony, Illustrious, NoobAI a Animagine jsou trénované na Danbooru tazích a
prózu čtou špatně — UGC pipeline v AiStacku to zaznamenala („no character"
→ busta, „empty item" → celý nindža), appka jim přesto posílá volný text.
Galerie proto překládá **na jednom místě**, aby appka, lab i benchmark
dostaly pro stejný vstup stejné tagy:

```
POST /api/translate   {"text": "a ballerina on pointe, seen from the side"}
→ {"text": "1girl, solo, ballerina, en pointe, from side, …",
   "tags": [...], "dropped": ["glowing aura"], "translator": "prompt-tags@v1"}
```

Tři vlastnosti, bez kterých by to rozbilo feedback loop:

- **Deterministické a přiřaditelné.** Odpověď se ukládá do `translations`
  podle `sha256(text, jazyk, verze)`; `translator` (alias@verze promptu) jde
  na node a Eval podle něj seskupuje — „pomáhá překlad?" a „je v2 lepší než
  v1?" je táž otázka nad týmž sloupcem.
- **Omezené slovníkem.** LLM si tagy vymýšlí; `GET {tagger}/tags` vrací
  `selected_tags.csv` WD14 a co v něm není, se zahodí — a **vrátí v `dropped`**,
  ne tiše.
- **Bez fallbacku.** Gateway dole = 502, ne jiný model pod stejnou verzí.
  Appka pak pošle prózu a `translator` nechá prázdný; kontrolní rameno je
  dosažitelné filtrem `translator=none`.

Konfigurace: `LLM_GATEWAY_URL` (AiStack gateway na LAN, prázdné = vypnuto),
`TRANSLATE_MODEL` (LiteLLM alias, default `prompt-tags`). Stav v `/api/meta`.

### Výběr modelu: `tagbench`

Ground truth už máš: `nodes.prompt` (co jsi napsal) a `captions.tags_json`
(co WD14 viděl na výsledku). Benchmark nechá každý alias přeložit stejné
prompty a skóruje proti taggeru — bez ručního labelování, bez stahování.

```
finetune-gallery tagbench -models prompt-tags,translate -n 100
model        n    err  recall  precision  vocab hit  tags  p50    p90
prompt-tags  100  0    71.4%   58.2%      93.1%      14.2  0.84s  1.31s
translate    100  0    68.9%   61.0%      95.4%      12.7  2.10s  3.05s
```

`recall` je hlavní číslo (našel, co v obrázku bylo); `precision` a `vocab hit`
říkají, kolik si vymyslel; `p50/p90` jestli může stát před každou generací.
`-no-cache` měří živý model místo včerejší odpovědi, `-v` vypíše páry.

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
GET  /api/images?model=&aspect=&score=&style=&lora=&lora_strength=&pose=
                &criterion=pose_adherence:1|-1|none&cursor=
GET  /api/images/{id}                      detail + parentChain
PUT  /api/images/{id}/rating|aspects|criteria|caption
POST /api/images/{id}/autocaption
GET  /api/datasets · POST …/items/from-filter · POST …/build · GET …/download
GET  /api/stats · /api/meta · /healthz
GET  /api/eval?group=model|style|lora|pose|session&min=10[&format=csv]
                                           + všechny filtry /api/images
```

## Vývoj

```bash
cd server && go run .                                # API na :8092
cd web && npm install && npm run dev                 # Vite dev + proxy na :8092
cd web && npm run build                              # → server/webdist (embed)
```

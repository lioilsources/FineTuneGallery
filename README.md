# finetune-gallery

Rating platforma + LoRA fine-tune pipeline pro [Ol1nLLM](../Ol1nLLM) Image
Studio. Appka exportuje generační stromy (prompty + obrázky + metadata) sem na
NAS; ve webové galerii se výstupy hodnotí (likes, kritiky, aspect tagy,
relační kritéria) a z ohodnocených dat se staví kohya-ss datasety pro SDXL
LoRA trénink na GPU boxu. Hotová LoRA se objeví v appce automaticky (ComfyUI
`/object_info/LoraLoader`).

```
Ol1nLLM appka ──export──▶ NAS (tento repo)      SPARK (LAN)      GPU box
                          Go API + Svelte web   WD14 tagger      make train DATASET=…
                          SQLite + blobs   ───▶ (CPU, :8097)     kohya sd-scripts
                                                                 → ComfyUI models/loras/
```

## Struktura

```
server/   Go 1.24 backend — ingest, galerie, datasety, kohya builder;
          embedne web (go:embed webdist), modernc.org/sqlite (CGO-free)
web/      Svelte 5 + Vite SPA (build → server/webdist)
tagger/   WD14 (wd-swinv2-tagger-v3) FastAPI sidecar, CPU ONNX — běží na
          SPARKu (docker-compose.spark.yml), ne na NAS, viz níž
train/    kohya trainer pro GPU box (Docker) + deploy do ComfyUI
taxo/     taxonomie obsahu (taxonomy.yaml) + hodnotitelé WD14 / SigLIP 2 /
          Qwen3-VL, teplota, konsenzus, multi-head hlava, conformal prahy
```

## Deploy (NAS + SPARK)

```bash
# na SPARKu — tagger (jednou, a znovu jen při změně tagger/)
ssh spark
cd ~/deploy/FineTuneGallery && docker compose -f docker-compose.spark.yml up -d --build
curl localhost:8097/healthz     # {"status":"ok"}

# na NAS — appka; TAGGER_URL v docker-compose.yml už ukazuje na SPARK přes LAN
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
/data/stylemaps/                 packy map stylů pro appku (viz níž)
```

### Mapy stylů

`GET /stylemaps/…` servíruje statické packy pro picker stylů v appce
(`index.json`, `<sada>/map.json`, `<sada>/atlas.webp`, `<sada>/t/<i>.webp`).
Server je nestaví ani nečte — vznikají v Ol1nLLM (`tools/stylemap`) a na NAS se
kopírují (`make stylemap-publish` tamtéž). Jen soubory: žádné výpisy adresářů,
žádné soubory začínající tečkou. Manifesty se revalidují při každém otevření,
obrázky se cachují den, protože pack se přestavuje pod stejnými jmény.

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

### VL soudce — druhý hodnotitel s κ-gate

Eval řadí jen tak rychle, jak rychle člověk kliká: při ~36 hodnoceních na
rameno se meze sotva rozpojí a každý další A/B chce desítky kliků na skupinu.
VLM, který dostane stejnou dvojici jako člověk — **REFERENCE** (OpenPose
šablona / img2img zdroj) a pak **OUTPUT** — umí backlog ohodnotit. Ale jen
tam, kde s člověkem prokazatelně souhlasí:

```
Eval → labels: Judge                                    (ilustrační, ne reálná data)
                   pose adherence          source identity        source style
                   pass · κ 0.75 ≥0.51     fail · κ 0.31 ≥0.08    uncalibrated · 12 pairs
  Pony V6          88 %  n=64 · 41 judge   n/a · gate             n/a · gate
```

- **Člověk zůstává ground truth.** Verdikty jdou do `criteria_judgments`,
  nikdy do `image_criteria`. Soudce = `alias@judge-v1`; nový prompt nebo jiný
  model za aliasem je nový, nekalibrovaný soudce.
- **κ, ne shoda.** `pose_adherence` má ~90 % up — soudce, který řekne „up" na
  všechno, má 90% shodu a nulovou informaci. Cohenovo κ odečte shodu, kterou
  dají dva hodnotitelé náhodou, a pro tohohle soudce vyjde 0.
- **Gate per kritérium, na dolní mezi.** `pass` = ≥ 40 párů (člověk ±1 ∧ soudce
  ±1), ≥ 8 lidských „down", κ ≥ 0.60 **a** dolní mez 95% intervalu ≥ 0.40.
  Méně dat = `uncalibrated`, ne „skoro pass". Prahy jsou konstanty v
  `kappa.go`; při `fail` se píše nový prompt podle nesouhlasů, prahy se nesnižují.
- **Slepý hodnotitel.** Detail ukáže verdikt soudce až **po** tvém hodnocení
  kritéria (vynucuje API, ne UI) — jinak by κ měřilo poslušnost.
- **Výběrová past.** Kalibrační vzorek nesmí vybírat soudce. `judgebench`
  losuje (seed) z lidsky ohodnocených obrázků; fronta `judge=…:unrated` se na
  verdikt nedívá. Odpověď „unsure" (abstain) se do κ ani do Evalu nepočítá,
  jen do pokrytí. Chybějící reference není abstain, ale chyba.

`labels` v Evalu: **Human** (default, beze změny), **Judge** (jen verdikty
kritérií s otevřenou gate, ostatní `n/a · gate`), **Merged** (lidský label
vyhrává, soudce doplní neohodnocené). Buňka říká, kolik je od soudce.

```
finetune-gallery judgebench -criterion pose_adherence -n 80 -v
judge judge@judge-v1 · gate: n ≥ 40, human down ≥ 8, κ ≥ 0.60, 95% low ≥ 0.40
criterion       sampled  err  abst  pairs  po    κ     95% low  down agree  gate
pose_adherence  50       0    1     49     0.92  0.75  0.51     0.80        pass
```

(čísla jsou z testovací fixture, ne z reálného běhu). Galerie: filtr
`judge=<kritérium>:disagree` (podklad pro další verzi promptu) a
`:unrated` (posouzeno soudcem, rate blind). Detail má u kritéria tlačítko
„judge"; `POST /api/judge/sweep?criterion=&n=` dá do fronty neposouzené
obrázky a první chyba gateway zbytek dávky zahodí (zavřené okno modelu).

Konfigurace: `JUDGE_MODEL` (LiteLLM alias, default `judge` = qwen36 v AiStacku,
temperature 0, thinking off), gateway sdílí `LLM_GATEWAY_URL`. Žádný fallback.

**Kalibrace na SPARKu (ručně, okno llm 17–01):**

1. AiStack: nasadit alias `judge` z `deploy/litellm_config.yaml` (restart
   litellm); `curl …:8080/v1/models` ho musí vypsat.
2. Ověřit **dva obrázky v jednom requestu** na `qwen36-agent` (jeden ruční
   `POST /api/images/{id}/judge`). Když vLLM druhý obrázek odmítne, chce to
   `--limit-mm-per-prompt '{"image":2}'` — restart produkčního kontejneru.
3. NAS redeploy (migrace v5 proběhne sama), `judgebench -n 80` per kritérium
   (~240 requestů, odhad 15–30 min sdíleného qwen36) → výsledek sem.

## Obsah: taxonomie, „teplota" a SFW brána

Co obrázek ukazuje (námět, vlasy, oči, médium, paleta, záběr …) a jak je
„horký" (spojitá **teplota 0–1** + pásma `safe / suggestive / nsfw /
explicit`) — jako facety galerie, filtr a brána před sdílením. Za flagem
`FINETUNE_TAXONOMY=1`; bez něj se galerie chová jako dřív (výjimka: blok
nezletilých, níž).

**Etika a rozsah.** Klasifikuje se **vygenerovaný obsah, ne lidé**.
`skin_tone` (tón pleti, škála světlá → tmavá po vzoru Monk/Fitzpatrick, ne
etnické kategorie — klasifikátor vidí barvu, ne původ) a `age_group` jsou
vizuální atributy vyrenderovaného obrázku pro filtrování a vyhledávání, ne
identita. Kategorie `child` (a přísněji i `teen`) existuje **jen jako blok**,
nikdy jako filtr nebo facet — server ji jako filtr odmítne i z URL. **Žádný
atribut taxonomie se nepoužívá k rankování kvality** (Eval s ní nepočítá).

**Jediný zdroj pravdy: `taxo/taxonomy.yaml`** (verzovaná). Každý uzel má
povolené hodnoty, mapování na WD14 tagy, textové hypotézy pro SigLIP, otázku
pro VLM a `human_required`. `python taxo/gen_schema.py` z ní vyrobí
`server/taxonomy.json` (embed do serveru) a `taxo/schema/vlm_extract.schema.json`
(guided JSON pro Qwen3-VL); `--wd14-csv selected_tags.csv` ověří, že každý
mapovaný tag existuje ve slovníku taggeru. Go test hlídá, že JSON odpovídá
YAML (sha256), Python test totéž přes `--check`.

**Tři hodnotitelé, jeden formát** (`taxo/`, běží na GPU boxu, ne na NAS):

| hodnotitel | co | cena |
|---|---|---|
| `eval_wd14.py` | WD14 tagy → uzly (deterministické mapování z YAML) + WD14 teplota | zdarma, tagy už jsou v `captions.tags_json` |
| `eval_siglip.py` | SigLIP 2 (`google/siglip2-so400m-patch16-512`) zero-shot per uzel + embedding | nová hodnota v YAML = hned, bez tréninku |
| `eval_vlm.py` | Qwen3-VL-8B (vLLM, `response_format: json_schema` ze schématu) | nejdražší, umí „rozcuchané" a „cel-shaded vs flat" |

Teplota = WD14 rating (pravděpodobnostně vážený průměr úrovní
general/sensitive/questionable/explicit) v ensemblu s
`Falconsai/nsfw_image_detection` a `AdamCodd/vit-base-nsfw-detector`
(`eval_nsfw.py`). Neshoda (rozptyl > 0.40, nebo jeden „safe" a jiný ≥ nsfw)
= `unsure` → fronta pro člověka. `combine.py` spojí odpovědi: **shoda = jistota**
(≥ 2 hodnotitelé stejně; u ordinálních uzlů stačí sousední hodnota), jinak
`unsure`; jediný zdroj není shoda. `push.py` to nahraje přes
`POST /api/taxonomy/import`.

**Blok nezletilých — bez výjimky a bez flagu.** `child`/`teen` od
*kteréhokoli* hodnotitele (conf ≥ 0.30), WD14 tag `child`/`loli`/`aged down`/…
nebo lidský label + cokoli jiného než jisté `safe` (pásmo ≠ safe nebo unsure)
= `block`: obrázek zmizí z gallery view (tedy ze seznamu, Evalu, datasetů,
podobných), blob vrací 451, záznam v `taxonomy_audit`. Když hodnotil zatím jen
WD14 z captionu, je to `quarantine` (taky skryté), kterou uvolní jen ensemble
s jistým `safe`; `block` se neuvolní nikdy (ani lidským „safe"). Blok kontroluje
captioner po každém WD14 captionu, import, lidský label, změna prahů a
startovní sweep. Pravidlo je čistá funkce `minorDecision` testovaná na
syntetických vektorech labelů (`taxo/testdata/minor_vectors.json`, sdílené s
Python testy) — žádné obrázky.

**Prahy s garancí** (`taxo/conformal.py`): `explicit` recall ≥ 0.99 přes
split-conformal „guard" (PAC varianta: s pravděpodobností 95 % nad
kalibračním vzorkem; potřebuje ≥ 299 lidských `explicit`) — vše nad guardem,
co body neoznačí jako explicit, jde do `unsure` (raději false positive do
fronty); `safe` precision ≥ 0.98 na Wilsonově dolní mezi (stejné pravidlo
jako Eval). `PUT /api/taxonomy/thresholds` přepočítá pásma, znovu aplikuje
blok a změnu zapíše do auditu. **Dokud nejsou lidské labely, platí ruční
defaulty z YAML** (pseudo-label fit slouží jen k testu nástroje).

**Lidské labely a κ.** `finetune-gallery taxobench -sample 300 -borderline 60`
vylosuje (seed) stratifikovaný vzorek podle model × styl + 60 hraničních
(rating unsure nebo ±0.05 od prahu) do `taxonomy_label_queue`. Filtr „label
queue" + panel Obsah v detailu; obrázek ze vzorku se ukazuje **naslepo**
(odpovědi hodnotitelů až po labelu, jako u soudce). `finetune-gallery
taxobench` pak vypíše κ per hodnotitel per uzel se **stejnou gate jako
soudce** (`kappa.go`: n ≥ 40, menšinová třída ≥ 8, κ ≥ 0.60, dolní mez ≥
0.40); u multi-label uzlů κ per hodnota. Hraniční stratum se reportuje zvlášť
(`-stratum random|borderline|all`). Efektivní label = člověk > konsenzus; u
`human_required` uzlů (věk, tón pleti, postava) jen člověk.

**Vlastní hlava** (`taxo/head.py`): jedna multi-head síť nad zmrazeným
SigLIP 2 embeddingem (uzel = hlava, teplota = regrese), trénovaná na
konsenzu (≥ 2 shodní) + lidských labelech, export ONNX. Embedding se ukládá
do `image_embeddings` (float32 BLOB; modernc/sqlite neumí načíst sqlite-vec,
20k × 1152 se prohledá hrubou silou v Go za milisekundy) → `GET
/api/images/{id}/similar`.

**Sdílené a veřejné pohledy.** `view=shared` = jen `safe` a ne `unsure`
(neohodnocené taky ne). `FINETUNE_PUBLIC_VIEW=1` to vynutí pro všechny
seznamy i bloby (instance pro StyleMap packy, merch, market) a bez taxonomie
odmítne nastartovat. Kohya export: `PUT /api/datasets/{id} {"tempMax":0.3}`
vyřadí nad prahem / unsure / neohodnocené; README datasetu nese limit, max
teplotu a počty vyřazených.

```
python taxo/eval_wd14.py   --gallery-data /data --out wd14.jsonl   --device cuda
python taxo/eval_nsfw.py   --gallery-data /data --out nsfw.jsonl   --device cuda
python taxo/eval_siglip.py --gallery-data /data --out siglip.jsonl --emb-out emb.npz --device cuda
python taxo/eval_vlm.py    --gallery-data /data --out vlm.jsonl --base-url http://gpu:8004/v1 --model qwen3-vl-8b
python taxo/combine.py --wd14 wd14.jsonl --nsfw nsfw.jsonl --siglip siglip.jsonl --vlm vlm.jsonl \
                       --out combined.jsonl --stats stats.json
python taxo/push.py --server http://finetune:8092 --combined combined.jsonl --emb emb.npz
python taxo/head.py train --emb emb.npz --labels combined.jsonl --out head/
python taxo/conformal.py fit --scores combined.jsonl --labels human.jsonl --out bands.json
python -m unittest discover -s taxo/tests
```

**Prompt lint pro Ol1nLLM (návrh, neimplementováno):** před generací
předpovědět očekávanou teplotu z promptu (malý textový klasifikátor nad
dvojicemi prompt → naměřená teplota z této galerie) → varování/blok podle
režimu appky; po generaci skutečná teplota z galerie — neshoda = model
„ujíždí". Per checkpoint statistika (kolik % promptů bez explicitního záměru
skončí nad 0.5) patří do Evalu jako argument pro výběr modelu.

## Picker modelů: sloučený, ne opsaný

Model je jediný strukturální filtr galerie, takže model, který v pickeru
chybí, je korpus, ke kterému se nedostaneš. Do 2026-09-22 byl seznam ručním
opisem registru appky (`lib/models/image_model.dart`) — a šest checkpointů
přidaných na SPARK 21. 9. tu prostě nebylo, protože se nikdo nedíval.

`/api/meta` teď seznam **skládá ze tří zdrojů**, každý zná jinou část
odpovědi:

| zdroj | co ví | `source` |
|---|---|---|
| ComfyUI `/object_info` | které soubory box právě umí načíst | `comfyui` |
| `kModels` (`server/meta.go`) | id appky, label, ekosystém LoRA, trainable | `registry` |
| `nodes.model_id` | co v galerii reálně leží | `gallery` |

Do pickeru se dostane sjednocení, takže výpadek jednoho zdroje nikoho
neschová:

- Checkpoint na boxu, který nikdo nezařadil, se objeví sám — bez labelu,
  `trainable:false`. Ekosystém ani trainable se **nehádají**: obojí je lidské
  rozhodnutí a tip by poslal trénink na špatný base model. Dopsat řádek do
  `kModels` je pak kosmetika, ne podmínka viditelnosti.
- Model, který z boxu zmizel, zůstává s `available:false` — jeho obrázky
  v galerii pořád jsou.
- Id, které existuje jen v odingestovaných řádcích, zůstává filtrovatelné.
- Video checkpointy (`ltx…` v `checkpoints/`, celá rodina `wan2.2…`) se
  přeskakují — appka generuje stills, byly by to řádky, na které nic nesedí.

Box se čte na pozadí (po startu, pak co 5 minut), takže `/api/meta` na něm
nikdy nečeká; výpadek drží poslední známý stav a hlásí se v odpovědi:

```
GET /api/meta → {"models": [...],
                 "comfyui": {"ok": true, "ckpts": 16, "checkedAt": "…"}}
```

Konfigurace: `COMFYUI_URL` (prázdné = jen registr + galerie). Míří na SPARK
**přes LAN** — `comfyui.ol1n.com` je za CF Access a na `/object_info` vrátí
403 s přihlašovací stránkou.

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
                &criterion=pose_adherence:1|-1|none&judge=pose_adherence:disagree|unrated&cursor=
GET  /api/images/{id}                      detail + parentChain
PUT  /api/images/{id}/rating|aspects|criteria|caption
POST /api/images/{id}/autocaption
GET  /api/datasets · POST …/items/from-filter · POST …/build · GET …/download
GET  /api/stats · /api/meta · /healthz
GET  /api/eval?group=model|style|lora|pose|session&min=10[&format=csv]
                [&source=human|judge|merged]  + všechny filtry /api/images
POST /api/images/{id}/judge?criterion=     verdikt VL soudce (502 bez fallbacku)
POST /api/judge/sweep?criterion=&n=        fronta neposouzených
GET  /api/judge                            gate tabule: κ, pokrytí, prahy
GET  /api/images?taxo=person.hair.color:blonde&temp_max=0.3&view=shared
                &band=&rating_unsure=&taxo_unsure=<uzel>&label_queue=1&taxo_src=
POST /api/taxonomy/import                  výstup taxo/combine.py (FINETUNE_TAXONOMY=1)
GET|PUT /api/images/{id}/taxonomy          odpovědi hodnotitelů / lidské labely
GET  /api/images/{id}/similar              kosinus nad SigLIP embeddingy
GET|PUT /api/taxonomy/thresholds           pásma teploty (PUT = audit + přepočet)
GET  /api/taxonomy/kappa?stratum=          κ per hodnotitel per uzel
GET  /api/taxonomy/audit                   bloky, uvolnění, změny prahů
```

## Vývoj

```bash
cd server && go run .                                # API na :8092
cd web && npm install && npm run dev                 # Vite dev + proxy na :8092
cd web && npm run build                              # → server/webdist (embed)
```

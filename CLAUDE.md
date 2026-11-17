# FineTuneGallery

<!-- Max 80 řádků. Co se nevejde, jde do docs/ a sem jeden odkaz. Veřejné repo: žádné LAN IP, domovské cesty, ID. -->

## Účel
Rating platforma a LoRA fine-tune pipeline pro Image Studio v Ol1nLLM: galerie
exportovaných generací, hodnocení, Eval harness, taxonomie obsahu a stavba
kohya datasetů. Běží na NAS (Go + Svelte), tagger a hodnotitelé na SPARKu.

## Build a test
```
cd server && go test ./...                      # gate backendu
cd web && npm ci && npm run build               # → server/webdist (go:embed)
python -m unittest discover -s taxo/tests       # taxonomie, minor block
cd server && go run .                           # API na :8092
cd web && npm run dev                           # Vite dev, proxy na :8092
docker compose up -d --build                    # NAS deploy (multi-stage Dockerfile)
```

## Mapa
- `server/` Go backend: ingest, galerie, Eval, soudce (κ gate), taxonomie, kohya builder; embeduje `webdist`.
- `web/` Svelte 5 + Vite SPA, build jde do `server/webdist`.
- `tagger/` WD14 FastAPI sidecar, běží na SPARKu (`docker-compose.spark.yml`).
- `taxo/` taxonomie (`taxonomy.yaml`) a hodnotitelé WD14 / SigLIP / VLM, conformal prahy.
- `train/` kohya trainer pro GPU box (`make build`, `make train DATASET=…`).
- `cloudflared/` config tunelu; credentials nejsou v gitu.
- `Prompts/` plány (např. UC25 kalibrace na SPARKu).

## Invarianty
- `taxo/taxonomy.yaml` je jediný zdroj pravdy; `server/taxonomy.json` a VLM schéma se generují `python taxo/gen_schema.py` a testy hlídají shodu (sha256).
- Blok nezletilých (`minorDecision`) platí bez flagu a bez výjimky; `child`/`teen` nikdy jako filtr ani facet.
- Člověk je ground truth: verdikty soudce jdou do `criteria_judgments`, nikdy do `image_criteria`.
- Prahy κ gate jsou konstanty v `server/kappa.go`; při `fail` se mění prompt, ne prahy.
- Jméno výstupní LoRA nesmí obsahovat „flux“ (hlídá `train/run.sh`).
- Adresy SPARKu a gateway patří do `.env`, ne do veřejného repa (`.env.example` má placeholdery).

## Kam dál
README (deploy, Eval, soudce, taxonomie, API), `Prompts/`, `.env.example`.

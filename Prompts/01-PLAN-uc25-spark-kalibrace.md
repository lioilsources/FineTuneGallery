# 01 — UC25: kalibrace teploty na SPARKu, pokračování

Zadání pro Opus: **implementuj podle plánu.** Navazuje na zdrojový plán 05 z privátního sandbox repa (cesta je v paměti projektu, `uc25-source-plan`). Jeho pravidla §0 platí celá; zkráceně jsou dole v části Mantinely. Kde se tenhle dokument od plánu 05 liší, platí tenhle.

Stav k 13. 11. 2026.

## Cíl

V galerii na NAS platí pásma teploty s conformal fitem z lidských labelů (explicit recall ≥ 0,99, safe precision ≥ 0,98) a je k nim report s čísly. Hlava nad SigLIP embeddingem je natrénovaná a změřená; do online provozu jde až s vlastní kalibrací (Past 2).

## Kde to stojí (13. 11. 2026, po kroku 0)

- Kroky 1 a 2 proběhly 28.–29. 10. (předchozí session): na SPARKu je v `taxo-data/private/gallery/` kopie celé galerie (19 740 souborů, manifest 19 918) a `run_eval.sh` doběhl bez chyb (`taxo-data/private/eval/`: `combined.jsonl`, `emb.npz`, `stats.json`).
- Ensemble: safe 3 209, suggestive 1 691, nsfw 3 145, explicit 11 873, unsure 30,5 %. Explicit je dost, **generování 1b odpadá**.
- **Pravidlo nezletilých by při importu zablokovalo 2 298 obrázků (11,5 %).** Všechny spouští jen SigLIP zero-shot věk (teen 1 189, child 1 109, většinou conf 0,3–0,6); WD14 tag nemá žádný. SigLIP věk se s WD14 shoduje jen u 33 z 16 420 obrázků. Rozhodnutí 5.
- Modely jsou na SPARKu v HF cache, venv má torch 2.14.1+cu130 s CUDA.
- Import do galerie ještě neproběhl. `FINETUNE_TAXONOMY` je v produkci vypnutý.
- Cesta ze SPARKu do galerie (Director, 13. 11.): veřejný hostname přes CF Access s **vlastním** service tokenem dávky + `WORKER_TOKEN` galerie. Žádný LAN port. Tokeny vytváří a zapisuje Olin.

Hotovo v kódu (větev `uc25/pool`):

- `WORKER_TOKEN` v galerii (`server/calibration.go`): Bearer otevře jen import a prahy; import bez tokenu se odmítne, jakmile je token nastavený.
- `taxobench -calib-hi/-calib-lo` (strata `calib:hi`/`calib:lo`, jen už ohodnocené obrázky), `-stratum calib`, `-export-human`.
- `push.py`: CF Access hlavičky, Bearer, User-Agent a retry z env.
- `head.py`: váhy (člověk 1, konsenzus 0,3), held-out stratifikovaný podle lidského pásma, F1 a recall explicit proti člověku, obrázky s rozhodnutím o nezletilosti mimo trénink.
- `conformal.py`: report per stratum s horní mezí 95 %.
- `taxo/run_train.sh`, `taxo/requirements-spark.txt`, `.gitignore`, compose + `.env.example`.

## Dvě statistické pasti

**Past 1 — výběr vzorku.** Pool bere všechny obrázky, které WD14 označil explicit nebo questionable, a z ostatních jen zlomek. Lidsky potvrzené explicit z takového poolu jsou skoro samé ty, které WD14 poznal, a na těch má ensemble vysokou teplotu (WD14 je jeho součást). Guard z nich vyjde optimistický a garance neplatí pro explicit, které WD14 minul.

Doporučení: garanci uvádět per stratum.

- Stratum `hi` (WD14 explicit + questionable) je v poolu celé. Fronta labelů z něj losuje náhodně se seedem, dokud není 300 lidských explicit. PAC guard pak pro tohle stratum platí.
- Stratum `lo` (sensitive + general): náhodný vzorek kolem 300 labelů. Report uvede, kolik explicit v něm člověk našel a jakou měly teplotu. Když 0 z n, uveď horní mez 3/n, ne „žádné nejsou“.
- `pool.py` zapisuje do manifestu stratum a pravděpodobnost výběru, ať jde podíly přepočítat na celou galerii.
- Hraniční stratum (±0,05 od prahu) reportuj zvlášť a do fitu safe prahu ho bez vah nemíchej.

**Past 2 — na čem se prahy fitují.** Prahy patří ke skóre, které v provozu rozhoduje. Ensemble se z lidských labelů neučí, do jeho kalibrace tedy smí všechny. Hlava se z nich učí: prahy pro její skóre smí vzniknout jen z obrázků, které při tréninku neviděla (out-of-fold predikce), a 299 explicit se počítá v kalibrační části.

Proto první nasazení jsou pásma z ensemblu. Hlava se trénuje a měří, online jde až v dalším PR.

## Postup

### Krok 0 — zjisti stav (jen čtení)

- SPARK: commit v deploy checkoutu, `taxo/.venv`, `taxo-data/`, HF cache, cache taggeru.
- NAS: nasazený commit, jestli DB má `taxonomy_label_queue`, jestli běží `FINETUNE_TAXONOMY=1`.
- Když ssh neprojde oprávněním, řekni to Olinovi (povolení, nebo spustí sám přes `!`). Neobcházej.

### Krok 1 — pool a přenos (PR z `uc25/pool`)

- Konzistentní snapshot DB (`VACUUM INTO` nebo `.backup`, ne `cp` živé WAL databáze) do gitignorovaného `run/`. Jsou to jen metadata. Nad ním `pool.py`.
- Výstup `available` rozhoduje: aspoň ~350 kandidátů explicit znamená, že generování 1b odpadá.
- `pool.py`: `--path-prefix` (kde obrázky budou na SPARKu), stratum a pravděpodobnost výběru v záznamu, test na syntetické DB (stratifikace, vyřazení blokovaných a signálu nezletilosti).
- Přenos obrázků poolu NAS → SPARK do `taxo-data/private/gallery/` (práva 700). Přímý rsync, pokud mezi stroji je klíč; jinak roura `tar | tar` přes M2, na jeho disk nic. Ne během renderu (page cache, `spark-gotchas.md`). Kontrola: počet souborů = počet řádků manifestu.
- `.gitignore` (mezera 9), `taxo/requirements-spark.txt` s `onnxruntime`. Torch pro aarch64 s CUDA ve verzi, jakou má ComfyUI na SPARKu; do jeho venv nic neinstaluj.

### Krok 2 — hodnotitelé na SPARKu

- Přes skill `spark-job`: `sparkctl run finetune comfyui 10 "<run_eval.sh s manifestem>" "UC25 hodnotitelé teploty"`.
- `run_eval.sh` jede s `HF_HUB_OFFLINE=1`. Nejsou-li modely v cache, zastav se u rozhodnutí 1.
- Výstupy (`combined.jsonl`, `emb.npz`, `stats.json`) zůstávají v `taxo-data/private/eval/`.
- Před importem spočítej z `combined.jsonl`, kolik obrázků pravidlo nezletilých zablokuje, a číslo řekni Olinovi. Blok je nevratný.
- Import: `push.py --combined --emb` cestou z rozhodnutí 3. Veřejný hostname chce browser `User-Agent`.

### Krok 3 — fronta labelů a export (PR `uc25/labels`)

- `taxobench -queue-file -` (manifest ze stdin, ať jde přes `docker exec -i`): náhodně se seedem do `taxonomy_label_queue`, strata `calib:hi` a `calib:lo`. V κ reportu `-stratum calib`.
- `taxobench -export-human`: JSONL `{image, rating:{band}, labels:{uzel:{value}}}`. CLI nad DB, žádný nový HTTP endpoint.
- Compose: `FINETUNE_TAXONOMY` z `.env` s prázdným defaultem, doplnit `.env.example`. Stejné soubory mění otevřený PR #8, čekej konflikt.
- Testy v Go bez sítě a bez `time.Now`.
- Pak labeluje Olin: filtr „label queue“, naslepo, kolem 600 obrázků, 1,5–2 h. Tady práce čeká; mezitím dělej krok 4.

### Krok 4 — trénink a kalibrace (PR `uc25/train`)

Kód jde napsat a otestovat na syntetických datech dřív, než budou labely.

- `head.py`: váhy vzorků, stratifikovaný held-out 20 % podle lidského pásma, out-of-fold predikce, metriky proti lidským labelům.
- `conformal.py`: fit per stratum podle Pasti 1, testy.
- Vyhodnocení: F1 per pásmo, MAE teploty, recall explicit, podíl unsure, falešné bloky; hlava vs ensemble vs samotné WD14; κ hodnotitelů proti Olinovi. Výstup jen agregáty.
- `taxo/run_train.sh` přes broker; poslední řádek stdout jsou metriky pro `sparkctl done`.
- Když lidských explicit nebude 300, fit dá slabší garanci a report ji řekne číslem.

### Krok 5 — nasazení pásem

- Před `push.py --bands` spočítej, kolik obrázků změní pásmo, a ukaž to Olinovi. Server neplatný fit odmítne.
- README: postup kalibrace a čísla. Report do sandbox repa (`reports/uc25-taxonomie-teplota.md`, část „SPARK kalibrace“), jen agregáty.
- Smazání obrázků poolu ze SPARKu až na pokyn Olina. Skóre a embeddingy zůstávají.

### Krok 1b — generování (jen při nedostatku explicit)

Zastav se u rozhodnutí 7. Postup a pravidla jsou v plánu 05, §1b a §0.2–0.3.

## Rozhodnutí Olina (stav 13. 11.)

1. Modely: jsou na SPARKu, nic se nestahuje.
2. `FINETUNE_TAXONOMY=1` na NAS: **ano**.
3. Cesta importu: rozhodl Director (viz výš). Olin založí service token v CF Zero Trust, přidá ho do Access policy aplikace finetune, vygeneruje `WORKER_TOKEN` a zapíše hodnoty do `.env` na NAS a na SPARKu.
4. Kopie obrázků na SPARKu už existuje. Po importu a tréninku ji jde smazat (hlava se trénuje z embeddingů); smazat jen na pokyn Olina.
5. 2 298 bloků jen ze SigLIP věku: **karanténa místo bloku** (Olin, 13. 11.). `minor_block.weak_evaluators: [siglip]`; signál od kohokoli dalšího je dál blok. Import tedy dá 0 bloků a 2 298 karantén (skryté, vratné). Do kalibrační fronty ani do tréninku nejdou.
6. Hlava online až s vlastní kalibrací (doporučeno, čeká na potvrzení).
7. Generování 1b: odpadá.

## Mantinely

- SPARK jen přes broker, projekt `finetune`, priorita 10. Okno jen na doslovná slova Olina.
- Obrázky poolu ani nic z `taxo-data/private/` neotvírej. Čteš manifesty a čísla. Do PR, commitů, reportu, Telegramu a `sparkctl done` jdou jen počty.
- Z `private/` nic neopouští SPARK. Výjimka: skóre a embeddingy obrázků galerie jdou importem do galerie.
- Signál nezletilosti a cokoli jiného než jisté safe: do datasetu se nebere. U generovaných se soubor maže a do auditu jde id, checkpoint, seed a důvod.
- Repo je veřejné: žádné LAN IP, domovské cesty ani názvy svazků; adresy přes `.env`.
- Modely jen se souhlasem, žádné sudo. Nasazenou migraci needituj, přidej novou.
- Merge a nasazení rozhoduje Olin.
- Gate před pushem: `go test ./...` v `server/`, `python -m unittest discover -s taxo/tests`, `python taxo/gen_schema.py --check`.

## Hotovo, když

- je nainstalovaný platný fit, nebo zdokumentovaná slabší garance s počtem explicit,
- report má agregáty per stratum a κ hodnotitelů,
- README popisuje postup kalibrace,
- mimo SPARK není žádný soubor z `private/`,
- PR čekají na Olina.

## Mimo rozsah

- LAN IP a domovská cesta, které už v compose souborech jsou. Patří do samostatného úklidu.
- VLM hodnotitel pro teplotu, prompt lint, online hlava v taggeru.

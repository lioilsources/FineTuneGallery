package main

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The minor rule is tested on synthetic label vectors only (the fixture is
// shared with taxo/tests) — never with imagery.

func TestTaxonomyJSONMatchesYAML(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "taxo", "taxonomy.yaml"))
	if err != nil {
		t.Skip("taxo/taxonomy.yaml not next to the server (vendored build)")
	}
	sum := sha256.Sum256(b)
	if got := hex.EncodeToString(sum[:]); got != taxonomy.SourceSha256 {
		t.Fatalf("server/taxonomy.json is stale (yaml %s, json %s) — run python taxo/gen_schema.py", got[:12], taxonomy.SourceSha256[:12])
	}
}

func TestMinorValuesAreNeverFilterable(t *testing.T) {
	n := taxonomy.node(taxonomy.MinorBlock.Node)
	for _, v := range taxonomy.MinorBlock.Values {
		tv, ok := n.value(v)
		if !ok || tv.Filterable {
			t.Fatalf("%s must exist and be filterable: false", v)
		}
	}
}

type minorFixture struct {
	Thresholds bandThresholds `json:"thresholds"`
	Cases      []struct {
		Name   string `json:"name"`
		Labels map[string]map[string]struct {
			Value string
			Conf  float64
		} `json:"labels"`
		WD14Signal float64 `json:"wd14_signal"`
		Ensemble   *struct {
			Temperature float64 `json:"temperature"`
			Band        string  `json:"band"`
			Unsure      bool    `json:"unsure"`
		} `json:"ensemble"`
		WD14Temp *float64 `json:"wd14_temp"`
		Want     *string  `json:"want"`
	} `json:"cases"`
}

func TestMinorDecisionOnSharedVectors(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "taxo", "testdata", "minor_vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fx minorFixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	if len(fx.Cases) < 10 {
		t.Fatalf("only %d cases", len(fx.Cases))
	}
	for _, c := range fx.Cases {
		t.Run(c.Name, func(t *testing.T) {
			in := minorInput{WD14Signal: c.WD14Signal, WD14Temp: c.WD14Temp}
			for ev, labels := range c.Labels {
				if l, ok := labels["person.age_group"]; ok {
					in.Votes = append(in.Votes, ageVote{Evaluator: ev, Value: l.Value, Conf: l.Conf})
				}
			}
			if e := c.Ensemble; e != nil {
				in.HasEnsemble, in.Unsure = true, e.Unsure
				in.Band = bandOf(e.Temperature, fx.Thresholds)
				if in.Band != e.Band {
					t.Fatalf("fixture band %s, thresholds say %s", e.Band, in.Band)
				}
			}
			got, reason := minorDecision(in, fx.Thresholds)
			want := ""
			if c.Want != nil {
				want = *c.Want
			}
			if got != want {
				t.Fatalf("decision %q, want %q (%s)", got, want, reason)
			}
			if got != "" && reason == "" {
				t.Fatal("a block needs a reason for the audit log")
			}
		})
	}
}

func TestMultiKappaEqualsTheJudgeGateOnTwoClasses(t *testing.T) {
	rnd := rand.New(rand.NewSource(3))
	var pairs [][2]string
	var a agreement
	for i := 0; i < 120; i++ {
		h, j := 1, 1
		if rnd.Float64() < 0.2 {
			h = -1
		}
		if rnd.Float64() < 0.25 {
			j = -h
		} else {
			j = h
		}
		a.add(h, j)
		pairs = append(pairs, [2]string{itoa(h + 1), itoa(j + 1)})
	}
	a.finish()
	m := multiKappa(pairs)
	if math.Abs(*a.Kappa-*m.Kappa) > 1e-12 || math.Abs(*a.Lower-*m.Lower) > 1e-12 || a.Gate != m.Gate {
		t.Fatalf("judge κ %.4f [%.4f] %s vs multi κ %.4f [%.4f] %s",
			*a.Kappa, *a.Lower, a.Gate, *m.Kappa, *m.Lower, m.Gate)
	}
}

/* ---------------- end-to-end over the HTTP handlers ---------------- */

type taxoFix struct {
	t *testing.T
	s *server
	n int
}

func newTaxoFix(t *testing.T) *taxoFix {
	t.Helper()
	taxoEnabled = true
	t.Cleanup(func() { taxoEnabled, taxoPublicView = false, false })
	f := &taxoFix{t: t, s: newTestServer(t)}
	f.exec(`INSERT INTO sessions (id, title, model_id) VALUES ('s1', 'lab', 'pony')`)
	return f
}

func (f *taxoFix) exec(q string, args ...any) {
	f.t.Helper()
	if _, err := f.s.db.Exec(q, args...); err != nil {
		f.t.Fatalf("%s: %v", q, err)
	}
}

func (f *taxoFix) addImage() (id, sha string) {
	f.t.Helper()
	f.n++
	id, sha = "t"+itoa(f.n), fakeSha(1000+f.n)
	f.exec(`INSERT INTO nodes (id, session_id, prompt, model_id, created_at) VALUES (?, 's1', 'p', 'pony', ?)`,
		"n"+id, "2026-10-28T10:00:"+itoa(10+f.n))
	f.exec(`INSERT INTO blobs (sha256, size) VALUES (?, 1)`, sha)
	f.exec(`INSERT INTO images (id, node_id, idx, sha256, size, blob_present) VALUES (?, ?, 0, ?, 1, 1)`, id, "n"+id, sha)
	// a real file, so a served blob is distinguishable from a refused one
	p := f.s.blobPath(sha)
	os.MkdirAll(filepath.Dir(p), 0o755)
	os.WriteFile(p, []byte("png"), 0o644)
	return id, sha
}

func (f *taxoFix) do(method, path string, body any) *httptest.ResponseRecorder {
	f.t.Helper()
	var rdr *bytes.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rdr = bytes.NewReader(b)
	} else {
		rdr = bytes.NewReader(nil)
	}
	req := httptest.NewRequest(method, path, rdr)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/images", f.s.handleImagesList)
	mux.HandleFunc("GET /api/meta", f.s.handleMeta)
	mux.HandleFunc("GET /img/{sha256}", f.s.handleImg)
	mux.HandleFunc("POST /api/taxonomy/import", f.s.handleTaxonomyImport)
	mux.HandleFunc("PUT /api/taxonomy/thresholds", f.s.handleThresholdsPut)
	mux.HandleFunc("GET /api/taxonomy/audit", f.s.handleTaxonomyAudit)
	mux.HandleFunc("GET /api/taxonomy/kappa", f.s.handleTaxoKappa)
	mux.HandleFunc("GET /api/images/{id}/taxonomy", f.s.handleImageTaxonomy)
	mux.HandleFunc("PUT /api/images/{id}/taxonomy", f.s.handleImageTaxonomyPut)
	mux.HandleFunc("GET /api/images/{id}/similar", f.s.handleSimilar)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	return rec
}

func (f *taxoFix) listIDs(query string) []string {
	f.t.Helper()
	rec := f.do("GET", "/api/images?"+query, nil)
	if rec.Code != http.StatusOK {
		f.t.Fatalf("GET /api/images?%s → %d %s", query, rec.Code, rec.Body.String())
	}
	var out struct {
		Items []galleryItem `json:"items"`
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	ids := []string{}
	for _, it := range out.Items {
		ids = append(ids, it.ID)
	}
	return ids
}

func rec(id string, temp float64, unsure bool, age, hair string) map[string]any {
	labels := map[string]any{"subject": map[string]any{"value": []string{"people"}, "conf": map[string]float64{"people": 0.9}}}
	cons := map[string]any{"subject": map[string]any{"value": []string{"people"}, "unsure": false}}
	if age != "" {
		labels["person.age_group"] = map[string]any{"value": age, "conf": 0.8}
		cons["person.age_group"] = map[string]any{"value": age, "unsure": false}
	}
	if hair != "" {
		labels["person.hair.color"] = map[string]any{"value": hair, "conf": 0.8}
		cons["person.hair.color"] = map[string]any{"value": hair, "unsure": false}
	}
	return map[string]any{
		"image": id, "taxonomy_version": taxonomy.Version,
		"labels":    map[string]any{"vlm": labels, "siglip": labels},
		"consensus": cons,
		"rating":    map[string]any{"temperature": temp, "unsure": unsure, "parts": map[string]float64{"wd14": temp}},
	}
}

func TestTaxonomyImportFiltersAndBlocks(t *testing.T) {
	f := newTaxoFix(t)
	adult, _ := f.addImage()
	child, childSha := f.addImage()
	hot, hotSha := f.addImage()
	kid, _ := f.addImage()

	r := f.do("POST", "/api/taxonomy/import", map[string]any{"records": []any{
		rec(adult, 0.05, false, "adult", "blonde"),
		rec(child, 0.30, false, "child", "brown"), // child + suggestive
		rec(hot, 0.80, false, "adult", "blonde"),
		rec(kid, 0.03, false, "child", "black"), // child + confident safe: allowed
	}})
	if r.Code != 200 {
		t.Fatalf("import → %d %s", r.Code, r.Body.String())
	}
	var res struct{ Imported, Blocked int }
	json.Unmarshal(r.Body.Bytes(), &res)
	if res.Imported != 4 || res.Blocked != 1 {
		t.Fatalf("imported %d blocked %d, want 4/1", res.Imported, res.Blocked)
	}

	all := f.listIDs("")
	if strings.Contains(strings.Join(all, ","), child) {
		t.Fatalf("blocked image listed: %v", all)
	}
	if got := f.listIDs("taxo=person.hair.color:blonde"); len(got) != 2 {
		t.Fatalf("blonde: %v", got)
	}
	if got := f.listIDs("taxo=person.hair.color:blonde&temp_max=0.3"); len(got) != 1 || got[0] != adult {
		t.Fatalf("blonde ≤ 0.3: %v", got)
	}
	shared := f.listIDs("view=shared")
	if len(shared) != 2 || strings.Contains(strings.Join(shared, ","), hot) {
		t.Fatalf("shared view: %v", shared)
	}
	if r := f.do("GET", "/api/images?taxo=person.age_group:child", nil); r.Code != http.StatusBadRequest {
		t.Fatalf("child filter → %d, want 400", r.Code)
	}
	if r := f.do("GET", "/img/"+childSha, nil); r.Code != http.StatusUnavailableForLegalReasons {
		t.Fatalf("blocked blob → %d, want 451", r.Code)
	}
	if r := f.do("GET", "/img/"+hotSha, nil); r.Code != http.StatusOK {
		t.Fatalf("private blob → %d", r.Code)
	}
	taxoPublicView = true
	if r := f.do("GET", "/img/"+hotSha, nil); r.Code != http.StatusNotFound {
		t.Fatalf("nsfw blob in public view → %d, want 404", r.Code)
	}
	if got := f.listIDs(""); len(got) != 2 {
		t.Fatalf("public view lists %v", got)
	}
	taxoPublicView = false

	// The audit holds the block, by id and reason only.
	var audit struct {
		Entries []map[string]any `json:"entries"`
		Blocks  map[string]int   `json:"blocks"`
	}
	json.Unmarshal(f.do("GET", "/api/taxonomy/audit", nil).Body.Bytes(), &audit)
	if audit.Blocks["block"] != 1 || len(audit.Entries) != 1 || audit.Entries[0]["imageId"] != child {
		t.Fatalf("audit: %+v", audit)
	}

	// Facets: counts of the effective label, no minor values at all.
	var meta struct {
		Taxonomy struct {
			Enabled bool        `json:"enabled"`
			Facets  []taxoFacet `json:"facets"`
		} `json:"taxonomy"`
	}
	json.Unmarshal(f.do("GET", "/api/meta", nil).Body.Bytes(), &meta)
	for _, fc := range meta.Taxonomy.Facets {
		for _, v := range fc.Values {
			if v.ID == "child" || v.ID == "teen" {
				t.Fatalf("facet lists %s", v.ID)
			}
			if fc.ID == "person.hair.color" && v.ID == "blonde" && v.Count != 2 {
				t.Fatalf("blonde count %d", v.Count)
			}
		}
	}
}

func TestTaxonomyFiltersNeedTheFlag(t *testing.T) {
	f := newTaxoFix(t)
	taxoEnabled = false
	for _, q := range []string{"taxo=style.medium:photo", "temp_max=0.3", "view=shared"} {
		if r := f.do("GET", "/api/images?"+q, nil); r.Code != http.StatusBadRequest {
			t.Fatalf("%s with the flag off → %d", q, r.Code)
		}
	}
	if r := f.do("POST", "/api/taxonomy/import", map[string]any{"records": []any{}}); r.Code != http.StatusNotFound {
		t.Fatalf("import with the flag off → %d", r.Code)
	}
	// The minor rule does not need the flag: a WD14 caption alone quarantines.
	id, sha := f.addImage()
	f.exec(`INSERT INTO captions (image_id, kind, text, tags_json) VALUES (?, 'auto', 'x', ?)`, id,
		`{"general":{"child":0.8},"rating":{"general":0.2,"sensitive":0.7,"questionable":0.1,"explicit":0}}`)
	st, err := recheckMinor(f.s.db, id)
	if err != nil || st != "quarantine" {
		t.Fatalf("state %q (%v), want quarantine", st, err)
	}
	if r := f.do("GET", "/img/"+sha, nil); r.Code != http.StatusUnavailableForLegalReasons {
		t.Fatalf("quarantined blob → %d", r.Code)
	}
}

func TestQuarantineIsReleasedOnlyBySafeEnsemble(t *testing.T) {
	f := newTaxoFix(t)
	id, _ := f.addImage()
	f.exec(`INSERT INTO captions (image_id, kind, text, tags_json) VALUES (?, 'auto', 'x', ?)`, id,
		`{"general":{"aged down":0.6},"rating":{"general":0.3,"sensitive":0.7,"questionable":0,"explicit":0}}`)
	if st, _ := recheckMinor(f.s.db, id); st != "quarantine" {
		t.Fatalf("want quarantine, got %q", st)
	}
	// Ensemble: confidently safe → released, audited.
	f.do("POST", "/api/taxonomy/import", map[string]any{"records": []any{rec(id, 0.04, false, "", "")}})
	var st string
	f.s.db.QueryRow(`SELECT state FROM taxonomy_blocks WHERE image_id = ?`, id).Scan(&st)
	if st != "" {
		t.Fatalf("still %q after a safe ensemble", st)
	}
	var kinds []string
	rows, _ := f.s.db.Query(`SELECT kind FROM taxonomy_audit ORDER BY id`)
	for rows.Next() {
		var k string
		rows.Scan(&k)
		kinds = append(kinds, k)
	}
	rows.Close()
	if strings.Join(kinds, ",") != "quarantine,release" {
		t.Fatalf("audit %v", kinds)
	}
	// A later non-safe ensemble blocks for good; a human "safe" does not lift it.
	f.do("POST", "/api/taxonomy/import", map[string]any{"records": []any{rec(id, 0.5, false, "", "")}})
	f.s.db.QueryRow(`SELECT state FROM taxonomy_blocks WHERE image_id = ?`, id).Scan(&st)
	if st != "block" {
		t.Fatalf("want block, got %q", st)
	}
	f.exec(`UPDATE image_temperature SET temperature = 0.01, band = 'safe' WHERE image_id = ?`, id)
	recheckMinor(f.s.db, id)
	f.s.db.QueryRow(`SELECT state FROM taxonomy_blocks WHERE image_id = ?`, id).Scan(&st)
	if st != "block" {
		t.Fatalf("a block must not be lifted, got %q", st)
	}
}

func TestThresholdChangeRebandsAuditsAndReappliesTheRule(t *testing.T) {
	f := newTaxoFix(t)
	kid, _ := f.addImage()
	f.do("POST", "/api/taxonomy/import", map[string]any{"records": []any{rec(kid, 0.15, false, "child", "")}})
	if ids := f.listIDs(""); len(ids) != 1 {
		t.Fatalf("safe child image hidden: %v", ids)
	}
	guard := 0.5
	r := f.do("PUT", "/api/taxonomy/thresholds", map[string]any{
		"thresholds": bandThresholds{SafeMax: 0.1, SuggestiveMax: 0.4, NsfwMax: 0.7, ExplicitGuard: &guard},
		"source":     "pseudo:test",
	})
	if r.Code != 200 {
		t.Fatalf("PUT thresholds → %d %s", r.Code, r.Body.String())
	}
	if ids := f.listIDs(""); len(ids) != 0 {
		t.Fatalf("0.15 is suggestive now — the child image must be blocked: %v", ids)
	}
	var n int
	f.s.db.QueryRow(`SELECT COUNT(*) FROM taxonomy_audit WHERE kind = 'thresholds'`).Scan(&n)
	if n != 1 {
		t.Fatalf("threshold audit rows %d", n)
	}
	// The guard: 0.6 is nsfw by the point thresholds but ≥ guard → unsure.
	adult, _ := f.addImage()
	f.do("POST", "/api/taxonomy/import", map[string]any{"records": []any{rec(adult, 0.6, false, "adult", "")}})
	var unsure int
	f.s.db.QueryRow(`SELECT unsure FROM image_temperature WHERE image_id = ?`, adult).Scan(&unsure)
	if unsure != 1 {
		t.Fatal("explicit guard did not mark 0.6 unsure")
	}
	if r := f.do("PUT", "/api/taxonomy/thresholds", map[string]any{
		"thresholds": bandThresholds{SafeMax: 0.5, SuggestiveMax: 0.4, NsfwMax: 0.7}}); r.Code != 400 {
		t.Fatalf("non-increasing thresholds → %d", r.Code)
	}
	if r := f.do("PUT", "/api/taxonomy/thresholds", map[string]any{
		"thresholds": bandThresholds{SafeMax: 0.1, SuggestiveMax: 0.4, NsfwMax: 0.7},
		"fit":        map[string]any{"valid": false}}); r.Code != 400 {
		t.Fatalf("invalid conformal fit → %d", r.Code)
	}
}

func TestHumanLabelsWinAndBlindLabelQueue(t *testing.T) {
	f := newTaxoFix(t)
	id, _ := f.addImage()
	f.do("POST", "/api/taxonomy/import", map[string]any{"records": []any{rec(id, 0.05, false, "adult", "blonde")}})
	f.exec(`INSERT INTO taxonomy_label_queue (image_id, stratum, seed) VALUES (?, 'random:pony/-', 1)`, id)

	var d map[string]any
	json.Unmarshal(f.do("GET", "/api/images/"+id+"/taxonomy", nil).Body.Bytes(), &d)
	if d["blind"] != true || d["evaluators"] != nil {
		t.Fatalf("label-queue image must be blind: %v", d)
	}
	if got := f.listIDs("label_queue=1"); len(got) != 1 {
		t.Fatalf("label queue %v", got)
	}
	r := f.do("PUT", "/api/images/"+id+"/taxonomy", map[string]any{
		"labels": map[string]any{"person.hair.color": "red"}, "rating": "safe"})
	if r.Code != http.StatusNoContent {
		t.Fatalf("PUT → %d %s", r.Code, r.Body.String())
	}
	json.Unmarshal(f.do("GET", "/api/images/"+id+"/taxonomy", nil).Body.Bytes(), &d)
	if d["blind"] != false || d["evaluators"] == nil {
		t.Fatalf("after labelling the answers are visible: %v", d)
	}
	if got := f.listIDs("taxo=person.hair.color:blonde"); len(got) != 0 {
		t.Fatalf("human said red, effective filter still blonde: %v", got)
	}
	if got := f.listIDs("taxo=person.hair.color:blonde&taxo_src=consensus"); len(got) != 1 {
		t.Fatalf("consensus source: %v", got)
	}
	if got := f.listIDs("label_queue=1"); len(got) != 0 {
		t.Fatalf("labelled image still queued: %v", got)
	}
	if r := f.do("PUT", "/api/images/"+id+"/taxonomy", map[string]any{
		"labels": map[string]any{"person.hair.color": []string{"red", "blonde"}}}); r.Code != 400 {
		t.Fatalf("two values on a single node → %d", r.Code)
	}
	// human child label + rating unsure → block
	f.exec(`UPDATE image_temperature SET unsure = 1 WHERE image_id = ?`, id)
	f.do("PUT", "/api/images/"+id+"/taxonomy", map[string]any{"labels": map[string]any{"person.age_group": "child"}})
	var st string
	f.s.db.QueryRow(`SELECT state FROM taxonomy_blocks WHERE image_id = ?`, id).Scan(&st)
	if st != "block" {
		t.Fatalf("human child + unsure: %q", st)
	}
}

func TestTaxoKappaAndLabelSample(t *testing.T) {
	f := newTaxoFix(t)
	var ids []string
	for i := 0; i < 60; i++ {
		id, _ := f.addImage()
		ids = append(ids, id)
	}
	counts, err := sampleLabelQueue(f.s.db, 30, 0, 7)
	if err != nil || counts["random"] != 30 {
		t.Fatalf("sample %v %v", counts, err)
	}
	// 50 images: vlm agrees with the human on 45 hair colours.
	colors := []string{"blonde", "brown", "black"}
	var recs []any
	for i, id := range ids[:50] {
		h := colors[i%3]
		v := h
		if i%10 == 0 {
			v = colors[(i+1)%3]
		}
		recs = append(recs, rec(id, 0.05, false, "", v))
		f.exec(`INSERT INTO image_taxonomy (image_id, node, evaluator, value, taxonomy_version) VALUES (?, 'person.hair.color', 'human', ?, ?)`,
			id, h, taxonomy.Version)
	}
	f.do("POST", "/api/taxonomy/import", map[string]any{"records": recs})
	rows, err := taxoKappa(f.s.db, "all")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range rows {
		if r.Node == "person.hair.color" && r.Evaluator == "vlm" {
			found = true
			if r.N != 50 || r.Kappa == nil || *r.Kappa < 0.8 || r.Gate != gatePass {
				t.Fatalf("hair κ row %+v", r)
			}
		}
	}
	if !found {
		t.Fatal("no hair row")
	}
}

func TestSimilarUsesEmbeddingsAndRespectsBlocks(t *testing.T) {
	f := newTaxoFix(t)
	a, _ := f.addImage()
	b, _ := f.addImage()
	c, _ := f.addImage()
	for id, v := range map[string][]float32{a: {1, 0, 0}, b: {0.9, 0.1, 0}, c: {0, 0, 1}} {
		f.exec(`INSERT INTO image_embeddings (image_id, model, dim, vec) VALUES (?, 'siglip2', 3, ?)`, id, encodeVec(v))
	}
	embIndex.invalidate()
	var out struct {
		Items []map[string]any `json:"items"`
	}
	json.Unmarshal(f.do("GET", "/api/images/"+a+"/similar?limit=2", nil).Body.Bytes(), &out)
	if len(out.Items) != 2 || out.Items[0]["id"] != b {
		t.Fatalf("similar: %v", out.Items)
	}
	f.exec(`INSERT INTO taxonomy_blocks (image_id, state, reason) VALUES (?, 'block', 'test')`, b)
	json.Unmarshal(f.do("GET", "/api/images/"+a+"/similar?limit=2", nil).Body.Bytes(), &out)
	for _, it := range out.Items {
		if it["id"] == b {
			t.Fatal("blocked image offered as similar")
		}
	}
}

func TestMigrationV6KeepsV5Rows(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v5.sqlite")
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(ON)")
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{schemaV1, migV2, migV3, migV4, migV5, `PRAGMA user_version = 5`} {
		if _, err := db.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	sha := fakeSha(9)
	for _, q := range []string{
		`INSERT INTO sessions (id, title, model_id) VALUES ('s1', 'old', 'pony')`,
		`INSERT INTO nodes (id, session_id, prompt, model_id) VALUES ('n1', 's1', 'a cat', 'pony')`,
		`INSERT INTO blobs (sha256, size) VALUES ('` + sha + `', 1)`,
		`INSERT INTO images (id, node_id, idx, sha256, size, blob_present) VALUES ('i1','n1',0,'` + sha + `',1,1)`,
		`INSERT INTO datasets (name, base_model, trigger_word) VALUES ('d1', 'pony', 'tw')`,
	} {
		if _, err := db.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()
	db, err = openDB(path)
	if err != nil {
		t.Fatalf("migrace v6 spadla: %v", err)
	}
	defer db.Close()
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM gallery`).Scan(&n); err != nil || n != 1 {
		t.Fatalf("gallery after v6: %d (%v)", n, err)
	}
	var tm *float64
	if err := db.QueryRow(`SELECT temp_max FROM datasets WHERE name = 'd1'`).Scan(&tm); err != nil || tm != nil {
		t.Fatalf("datasets.temp_max: %v %v", tm, err)
	}
}

func TestDatasetTempAdmit(t *testing.T) {
	var d datasetTemp
	lim := 0.3
	x := func(v float64) *float64 { return &v }
	cases := []struct {
		temp    *float64
		unsure  bool
		blocked bool
		limit   *float64
		want    bool
	}{
		{x(0.1), false, false, &lim, true},
		{x(0.5), false, false, &lim, false},
		{x(0.1), true, false, &lim, false},
		{nil, false, false, &lim, false},
		{nil, false, false, nil, true},
		{x(0.9), false, false, nil, true},
		{x(0.01), false, true, nil, false},
	}
	for i, c := range cases {
		if got := d.admit(c.temp, c.unsure, c.blocked, c.limit); got != c.want {
			t.Fatalf("case %d: %v", i, got)
		}
	}
	if d.Blocked != 1 || d.Excluded != 3 || d.Unrated != 1 || *d.Max != 0.9 {
		t.Fatalf("%+v", d)
	}
}

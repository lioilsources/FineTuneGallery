package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestWorkerTokenOpensOnlyItsRoutes(t *testing.T) {
	workerToken = "w-secret"
	t.Cleanup(func() { workerToken = "" })
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusNoContent) })
	h := workerAuth(ok)
	cases := []struct {
		method, path, auth string
		want               int
	}{
		{"POST", "/api/taxonomy/import", "Bearer w-secret", 204},
		{"PUT", "/api/taxonomy/thresholds", "Bearer w-secret", 204},
		{"GET", "/api/images", "Bearer w-secret", 403},       // the token reads nothing
		{"POST", "/api/taxonomy/import", "Bearer nope", 401}, // wrong token
		{"POST", "/api/taxonomy/import", "", 401},            // import is machine-only
		{"PUT", "/api/taxonomy/thresholds", "", 204},         // the UI behind CF Access
		{"GET", "/api/images", "", 204},
	}
	for _, c := range cases {
		req := httptest.NewRequest(c.method, c.path, nil)
		if c.auth != "" {
			req.Header.Set("Authorization", c.auth)
		}
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, req)
		if rr.Code != c.want {
			t.Errorf("%s %s %q → %d, want %d", c.method, c.path, c.auth, rr.Code, c.want)
		}
	}

	// Without a configured token any bearer is refused and the import stays open
	// as before (CF Access is then the only gate).
	workerToken = ""
	req := httptest.NewRequest("POST", "/api/taxonomy/import", nil)
	req.Header.Set("Authorization", "Bearer w-secret")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != 401 {
		t.Fatalf("bearer without WORKER_TOKEN → %d, want 401", rr.Code)
	}
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest("POST", "/api/taxonomy/import", nil))
	if rr.Code != 204 {
		t.Fatalf("import without WORKER_TOKEN → %d, want 204", rr.Code)
	}
}

func TestCalibSampleStrataAndHumanExport(t *testing.T) {
	f := newTaxoFix(t)
	ratings := []string{"explicit", "questionable", "sensitive", "general"}
	var recs []any
	var ids []string
	for i := 0; i < 40; i++ {
		id, _ := f.addImage()
		ids = append(ids, id)
		tags, _ := json.Marshal(map[string]any{"rating": map[string]float64{ratings[i%4]: 0.9, "general": 0.05}})
		f.exec(`INSERT INTO captions (image_id, kind, text, tagger, tags_json) VALUES (?, 'auto', 'x', 'wd14', ?)`, id, string(tags))
		if i < 36 { // the last four were never rated → not eligible
			recs = append(recs, rec(id, 0.05, false, "adult", "blonde"))
		}
	}
	if r := f.do("POST", "/api/taxonomy/import", map[string]any{"records": recs}); r.Code != 200 {
		t.Fatalf("import → %d %s", r.Code, r.Body.String())
	}
	counts, err := sampleCalibQueue(f.s.db, 10, 50, 25)
	if err != nil {
		t.Fatal(err)
	}
	if counts[calibHi] != 10 || counts[calibHi+"/available"] != 18 || counts[calibLo] != 18 {
		t.Fatalf("counts %v", counts)
	}
	var hiWrong int
	f.s.db.QueryRow(`SELECT COUNT(*) FROM taxonomy_label_queue q JOIN captions c ON c.image_id = q.image_id
		WHERE q.stratum = ? AND c.tags_json NOT LIKE '%explicit%' AND c.tags_json NOT LIKE '%questionable%'`, calibHi).Scan(&hiWrong)
	if hiWrong != 0 {
		t.Fatalf("%d non-hi images in %s", hiWrong, calibHi)
	}
	// A second draw takes only what is left, never an image twice.
	again, _ := sampleCalibQueue(f.s.db, 50, 0, 25)
	if again[calibHi] != 8 {
		t.Fatalf("second draw %v", again)
	}

	f.exec(`INSERT INTO image_taxonomy (image_id, node, evaluator, value, taxonomy_version) VALUES (?, 'rating', 'human', 'explicit', ?)`, ids[0], taxonomy.Version)
	f.exec(`INSERT INTO image_taxonomy (image_id, node, evaluator, value, taxonomy_version) VALUES (?, 'person.hair.color', 'human', 'brown', ?)`, ids[0], taxonomy.Version)
	f.exec(`INSERT INTO image_taxonomy (image_id, node, evaluator, value, taxonomy_version) VALUES (?, 'rating', 'human', 'safe', ?)`, ids[2], taxonomy.Version)
	var buf bytes.Buffer
	n, err := exportHuman(f.s.db, &buf)
	if err != nil || n != 2 {
		t.Fatalf("export n=%d err=%v", n, err)
	}
	sc := bufio.NewScanner(&buf)
	sc.Scan()
	var first struct {
		Image, Stratum string
		Rating         map[string]string
		Labels         map[string]map[string]any
	}
	if err := json.Unmarshal(sc.Bytes(), &first); err != nil {
		t.Fatal(err)
	}
	if first.Image != ids[0] || first.Rating["band"] != "explicit" || first.Labels["person.hair.color"]["value"] != "brown" || first.Stratum != calibHi {
		t.Fatalf("export line %+v", first)
	}
}

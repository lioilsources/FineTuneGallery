package main

import (
	"database/sql"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

// seedEvalCorpus builds a small but deliberately lopsided corpus:
//
//	pony            40 posed images, 36 pose_adherence up / 4 down
//	illustrious-xl   3 posed images,  3 pose_adherence up / 0 down
//	juggernaut-xl    5 unposed images, no criteria at all
//
// The point of the fixture is that illustrious wins on ratio (100% vs 90%)
// and must not win the leaderboard.
func seedEvalCorpus(t *testing.T, s *server) {
	t.Helper()
	exec := func(q string, args ...any) {
		t.Helper()
		if _, err := s.db.Exec(q, args...); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	exec(`INSERT INTO sessions (id, title, model_id) VALUES ('s1', 'Lab run', 'pony')`)

	n := 0
	add := func(model, pose, source string, score int, crit map[string]int) string {
		n++
		id := "i" + itoa(n)
		nodeID := "n" + itoa(n)
		sha := fakeSha(n)
		exec(`INSERT INTO nodes (id, session_id, prompt, model_id, pose_id, source_image_id)
		      VALUES (?, 's1', 'p', ?, ?, ?)`, nodeID, model, nullable(pose), nullable(source))
		exec(`INSERT INTO blobs (sha256, size) VALUES (?, 1)`, sha)
		exec(`INSERT INTO images (id, node_id, idx, sha256, size, blob_present)
		      VALUES (?, ?, 0, ?, 1, 1)`, id, nodeID, sha)
		if score != 0 {
			exec(`INSERT INTO ratings (image_id, score) VALUES (?, ?)`, id, score)
		}
		for c, v := range crit {
			exec(`INSERT INTO image_criteria (image_id, criterion, score) VALUES (?, ?, ?)`, id, c, v)
		}
		return id
	}

	for i := 0; i < 40; i++ {
		v := 1
		if i >= 36 {
			v = -1
		}
		add("pony", "ol1", "", 1, map[string]int{"pose_adherence": v})
	}
	for i := 0; i < 3; i++ {
		add("illustrious-xl", "ol1", "", 1, map[string]int{"pose_adherence": 1})
	}
	for i := 0; i < 5; i++ {
		add("juggernaut-xl", "", "", 0, nil)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// fakeSha is a distinct, well-formed 64-hex-digit blob id per counter value.
func fakeSha(n int) string { return fmt.Sprintf("%064x", n) }

func nullable(v string) any {
	if v == "" {
		return nil
	}
	return v
}

func evalJSON(t *testing.T, s *server, query string) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/eval?"+query, nil)
	rec := httptest.NewRecorder()
	s.handleEval(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/eval?%s → %d: %s", query, rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return out
}

func rowByLabel(t *testing.T, res map[string]any, label string) map[string]any {
	t.Helper()
	for _, r := range res["rows"].([]any) {
		row := r.(map[string]any)
		if row["label"] == label {
			return row
		}
	}
	t.Fatalf("no row labelled %q in %v", label, res["rows"])
	return nil
}

func TestWilsonPenalisesSmallSamples(t *testing.T) {
	// 3/3 looks perfect and 36/40 does not; the lower bound knows better.
	_, loSmall, hiSmall := wilson(3, 0)
	_, loBig, hiBig := wilson(36, 4)
	if loSmall >= loBig {
		t.Fatalf("3/3 lower bound %.3f should sit below 36/40's %.3f", loSmall, loBig)
	}
	if hiSmall-loSmall <= hiBig-loBig {
		t.Fatalf("3/3 interval (%.3f) should be wider than 36/40's (%.3f)", hiSmall-loSmall, hiBig-loBig)
	}
	// Known value, so a refactor of the formula cannot drift silently.
	rate, lo, hi := wilson(9, 1)
	if math.Abs(rate-0.9) > 1e-9 {
		t.Fatalf("rate = %v, want 0.9", rate)
	}
	if math.Abs(lo-0.5958) > 5e-4 || math.Abs(hi-0.9821) > 5e-4 {
		t.Fatalf("wilson(9,1) = [%.4f, %.4f], want ≈[0.5958, 0.9821]", lo, hi)
	}
	// The interval must stay inside [0,1] even at the extremes.
	if _, lo, _ := wilson(0, 5); lo < 0 {
		t.Fatalf("lower bound %v below 0", lo)
	}
	if _, _, hi := wilson(5, 0); hi > 1 {
		t.Fatalf("upper bound %v above 1", hi)
	}
}

func TestEvalLeaderIgnoresUnderpoweredGroups(t *testing.T) {
	s := newTestServer(t)
	seedEvalCorpus(t, s)

	res := evalJSON(t, s, "group=model&min=10")
	leaders := res["leaders"].(map[string]any)
	pose, ok := leaders["pose_adherence"].(map[string]any)
	if !ok {
		t.Fatalf("no pose_adherence leader: %v", leaders)
	}
	if pose["label"] != "Pony V6" {
		t.Fatalf("leader = %v, want Pony V6 — a 3/3 sample outranked a 36/40 one", pose["label"])
	}

	// The floor's own job: a group that is the only candidate still must not
	// be crowned on three samples. Scope to illustrious alone — with min=10
	// there is no leader at all, because there is no evidence yet.
	res = evalJSON(t, s, "group=model&model=illustrious-xl&min=10")
	if l, ok := res["leaders"].(map[string]any)["pose_adherence"]; ok {
		t.Fatalf("3 samples produced a leader %v; the floor should have excluded it", l)
	}
	res = evalJSON(t, s, "group=model&model=illustrious-xl&min=1")
	pose = res["leaders"].(map[string]any)["pose_adherence"].(map[string]any)
	if pose["label"] != "Illustrious XL" || pose["rated"].(float64) != 3 {
		t.Fatalf("with min=1 leader = %v, want Illustrious XL over 3 ratings", pose)
	}
	// And the lower bound still records how weak that evidence is.
	if lo := pose["lower"].(float64); lo > 0.6 {
		t.Fatalf("3/3 lower bound = %.3f; a perfect ratio on 3 samples must not read as strong", lo)
	}
}

func TestEvalEligibilityIsPerCriterion(t *testing.T) {
	s := newTestServer(t)
	seedEvalCorpus(t, s)
	res := evalJSON(t, s, "group=model")

	pony := rowByLabel(t, res, "Pony V6")
	crit := pony["criteria"].(map[string]any)
	poseCell := crit["pose_adherence"].(map[string]any)
	if got := poseCell["eligible"].(float64); got != 40 {
		t.Fatalf("pony pose_adherence eligible = %v, want 40", got)
	}
	if got := poseCell["rated"].(float64); got != 40 {
		t.Fatalf("pony pose_adherence rated = %v, want 40", got)
	}
	// No img2img anywhere, so the source criteria are not "0%% coverage" —
	// they are not applicable, and eligible must say so.
	if got := crit["source_identity"].(map[string]any)["eligible"].(float64); got != 0 {
		t.Fatalf("source_identity eligible = %v, want 0 (nothing is img2img)", got)
	}

	// Juggernaut has no poses: its pose criterion is inapplicable, not unrated.
	jug := rowByLabel(t, res, "Juggernaut XL")
	if got := jug["criteria"].(map[string]any)["pose_adherence"].(map[string]any)["eligible"].(float64); got != 0 {
		t.Fatalf("juggernaut pose_adherence eligible = %v, want 0", got)
	}
	if got := jug["n"].(float64); got != 5 {
		t.Fatalf("juggernaut n = %v, want 5", got)
	}
	if got := jug["unrated"].(float64); got != 5 {
		t.Fatalf("juggernaut unrated = %v, want 5", got)
	}
}

func TestEvalTotalsDoNotMultiplyByCriteria(t *testing.T) {
	s := newTestServer(t)
	seedEvalCorpus(t, s)
	// Two criteria on one image must not count that image twice.
	if _, err := s.db.Exec(
		`INSERT INTO image_criteria (image_id, criterion, score) VALUES ('i1', 'source_style', 1)`); err != nil {
		t.Fatal(err)
	}
	res := evalJSON(t, s, "group=model")
	total := res["total"].(map[string]any)
	if got := total["n"].(float64); got != 48 {
		t.Fatalf("total n = %v, want 48", got)
	}
}

func TestEvalRespectsGalleryFilter(t *testing.T) {
	s := newTestServer(t)
	seedEvalCorpus(t, s)
	res := evalJSON(t, s, "group=model&model=pony")
	rows := res["rows"].([]any)
	if len(rows) != 1 {
		t.Fatalf("filtered eval returned %d rows, want 1", len(rows))
	}
	if got := rows[0].(map[string]any)["n"].(float64); got != 40 {
		t.Fatalf("n = %v, want 40", got)
	}
}

func TestEvalRejectsUnknownGroup(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/eval?group=prompt", nil)
	rec := httptest.NewRecorder()
	s.handleEval(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown group → %d, want 400", rec.Code)
	}
}

func TestEvalGroupsLoraByNameAndStrength(t *testing.T) {
	s := newTestServer(t)
	seedEvalCorpus(t, s)
	for i, strength := range []float64{0.4, 1.2} {
		id, node := "L"+itoa(i), "LN"+itoa(i)
		sha := fakeSha(1000 + i)
		if _, err := s.db.Exec(`INSERT INTO nodes (id, session_id, prompt, model_id, lora_name, lora_strength)
			VALUES (?, 's1', 'p', 'pony', 'face_v1', ?)`, node, strength); err != nil {
			t.Fatal(err)
		}
		if _, err := s.db.Exec(`INSERT INTO blobs (sha256, size) VALUES (?, 1)`, sha); err != nil {
			t.Fatal(err)
		}
		if _, err := s.db.Exec(`INSERT INTO images (id, node_id, idx, sha256, size, blob_present)
			VALUES (?, ?, 0, ?, 1, 1)`, id, node, sha); err != nil {
			t.Fatal(err)
		}
	}
	res := evalJSON(t, s, "group=lora")
	labels := map[string]bool{}
	for _, r := range res["rows"].([]any) {
		labels[r.(map[string]any)["label"].(string)] = true
	}
	// Two strengths of one LoRA are two experiments, not one.
	for _, want := range []string{"face_v1 @ 0.40", "face_v1 @ 1.20", "(no LoRA)"} {
		if !labels[want] {
			t.Fatalf("missing group %q, got %v", want, labels)
		}
	}
}

func TestEvalCSVMatrix(t *testing.T) {
	s := newTestServer(t)
	seedEvalCorpus(t, s)
	req := httptest.NewRequest(http.MethodGet, "/api/eval?group=model&format=csv", nil)
	rec := httptest.NewRecorder()
	s.handleEval(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("csv → %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/csv") {
		t.Fatalf("content-type = %q", ct)
	}
	recs, err := csv.NewReader(strings.NewReader(rec.Body.String())).ReadAll()
	if err != nil {
		t.Fatalf("parse csv: %v", err)
	}
	if len(recs) != 4 { // header + three models
		t.Fatalf("csv has %d rows, want 4", len(recs))
	}
	head := recs[0]
	if head[0] != "model" {
		t.Fatalf("first column = %q, want model", head[0])
	}
	col := func(name string) int {
		for i, h := range head {
			if h == name {
				return i
			}
		}
		t.Fatalf("no column %q in %v", name, head)
		return -1
	}
	// Every row must be as wide as the header — a ragged matrix is unusable.
	for _, r := range recs[1:] {
		if len(r) != len(head) {
			t.Fatalf("row %v has %d fields, header has %d", r[0], len(r), len(head))
		}
	}
	for _, r := range recs[1:] {
		if r[1] == "Pony V6" && r[col("pose_adherence_rate")] != "0.9000" {
			t.Fatalf("pony pose_adherence_rate = %q, want 0.9000", r[col("pose_adherence_rate")])
		}
	}
}

// The harness is only useful if it can hand back the images that still need
// rating — the ones the criterion applies to and nobody has judged.
func TestCriterionNoneFilterIsTheRatingQueue(t *testing.T) {
	s := newTestServer(t)
	seedEvalCorpus(t, s)
	// One posed pony image loses its rating: that is the whole queue.
	if _, err := s.db.Exec(`DELETE FROM image_criteria WHERE image_id = 'i1'`); err != nil {
		t.Fatal(err)
	}
	where, args, bad := galleryFilter(map[string][]string{"criterion": {"pose_adherence:none"}})
	if bad != "" {
		t.Fatalf("filter rejected: %s", bad)
	}
	var n int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM gallery g WHERE `+where, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	// 5 juggernaut images are unrated too, but have no pose — they are not
	// in the queue, because the criterion cannot be judged on them.
	if n != 1 {
		t.Fatalf("pose_adherence:none matched %d images, want 1", n)
	}

	if _, _, bad := galleryFilter(map[string][]string{"criterion": {"pose_adherence:maybe"}}); bad == "" {
		t.Fatal("criterion=pose_adherence:maybe should be rejected")
	}
}

// The medium axis exists to be A/B'd: the same style rendered with an explicit
// render medium and without one. That comparison only works if the control arm
// — nodes with no medium — stays addressable in both the harness and the
// gallery, so both halves are pinned here.
func TestMediumAxisComparesAgainstItsControlArm(t *testing.T) {
	s := newTestServer(t)
	exec := func(q string, args ...any) {
		t.Helper()
		if _, err := s.db.Exec(q, args...); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	exec(`INSERT INTO sessions (id, title, model_id) VALUES ('lab', 'medium A/B', 'juggernaut-xl')`)

	// Same style (assyrian), same model: 12 with a declared medium, 8 without.
	seq := 0
	add := func(n int, medium any, up, down int) {
		for i := 0; i < n; i++ {
			seq++
			id, node, sha := fmt.Sprintf("mi%d", seq), fmt.Sprintf("mn%d", seq), fakeSha(seq)
			exec(`INSERT INTO nodes (id, session_id, prompt, model_id, style_id, medium_id)
			      VALUES (?, 'lab', 'a ballerina', 'juggernaut-xl', 'assyrian', ?)`, node, medium)
			exec(`INSERT INTO blobs (sha256, size) VALUES (?, 1)`, sha)
			exec(`INSERT INTO images (id, node_id, idx, sha256, size, blob_present)
			      VALUES (?, ?, 0, ?, 1, 1)`, id, node, sha)
			score := -1
			if i < up {
				score = 1
			}
			if i < up+down {
				exec(`INSERT INTO ratings (image_id, score) VALUES (?, ?)`, id, score)
			}
		}
	}
	add(12, "medium_illustration", 10, 2)
	add(8, nil, 2, 6)

	res := evalJSON(t, s, "group=medium&min=5")
	withMedium := rowByLabel(t, res, "medium_illustration")["n"].(float64)
	control := rowByLabel(t, res, "(no medium)")["n"].(float64)
	if withMedium != 12 || control != 8 {
		t.Fatalf("rows split %v/%v, want 12/8", withMedium, control)
	}
	leader := res["leaders"].(map[string]any)["like"].(map[string]any)
	if leader["label"] != "medium_illustration" {
		t.Fatalf("like leader = %v, want medium_illustration", leader["label"])
	}

	// The gallery must be able to serve the control arm on its own; without
	// medium=none the images the experiment compares against are unreachable.
	for _, tc := range []struct {
		filter string
		want   int
	}{
		{"none", 8},
		{"medium_illustration", 12},
	} {
		where, args, bad := galleryFilter(map[string][]string{"medium": {tc.filter}})
		if bad != "" {
			t.Fatalf("medium=%s rejected: %s", tc.filter, bad)
		}
		var n int
		if err := s.db.QueryRow(`SELECT COUNT(*) FROM gallery g WHERE `+where, args...).Scan(&n); err != nil {
			t.Fatal(err)
		}
		if n != tc.want {
			t.Fatalf("medium=%s matched %d, want %d", tc.filter, n, tc.want)
		}
	}
}

// Migrating a v2 database (styles and LoRA strengths already rated) to v3 must
// keep every row and leave the new column empty — those images are the control
// arm of the very experiment the column exists for.
func TestMigrationV3KeepsV2Rows(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v2.sqlite")

	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(ON)")
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{schemaV1, migV2, `PRAGMA user_version = 2`} {
		if _, err := db.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	sha := fakeSha(7)
	for _, q := range []string{
		`INSERT INTO sessions (id, title, model_id) VALUES ('s1', 'old', 'pony')`,
		`INSERT INTO nodes (id, session_id, prompt, model_id, style_id, lora_strength)
		 VALUES ('n1', 's1', 'a cat', 'pony', 'ukiyoe', 0.4)`,
		`INSERT INTO blobs (sha256, size) VALUES ('` + sha + `', 1)`,
		`INSERT INTO images (id, node_id, idx, sha256, size, blob_present)
		 VALUES ('i1','n1',0,'` + sha + `',1,1)`,
		`INSERT INTO ratings (image_id, score) VALUES ('i1', 1)`,
	} {
		if _, err := db.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()

	db, err = openDB(path)
	if err != nil {
		t.Fatalf("migrace v3 spadla: %v", err)
	}
	defer db.Close()

	var version int
	if err := db.QueryRow(`PRAGMA user_version`).Scan(&version); err != nil ||
		version != schemaVersion() {
		t.Fatalf("user_version = %d, want %d (%v)", version, schemaVersion(), err)
	}
	var styleID string
	var mediumID any
	var score int
	if err := db.QueryRow(
		`SELECT style_id, medium_id, score FROM gallery WHERE id = 'i1'`).
		Scan(&styleID, &mediumID, &score); err != nil {
		t.Fatalf("gallery view after v3: %v", err)
	}
	if styleID != "ukiyoe" || score != 1 {
		t.Fatalf("row changed: style=%q score=%d", styleID, score)
	}
	if mediumID != nil {
		t.Fatalf("medium_id = %v, want NULL on a pre-v3 row", mediumID)
	}
}

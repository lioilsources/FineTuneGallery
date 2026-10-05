package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func poseCell(t *testing.T, res map[string]any) map[string]any {
	t.Helper()
	return res["total"].(map[string]any)["criteria"].(map[string]any)["pose_adherence"].(map[string]any)
}

func num(cell map[string]any, key string) int {
	v, _ := cell[key].(float64)
	return int(v)
}

// calibrated is the seedCalibration corpus (gate open on pose_adherence) with
// its 50 rated images and 3 unrated ones all judged.
func calibrated(t *testing.T) *judgeFixture {
	t.Helper()
	f := newJudgeFixture(t)
	seedCalibration(f)
	if _, err := f.s.judge.sweep("pose_adherence", 1000); err != nil {
		t.Fatal(err)
	}
	close(f.s.judge.jobs)
	f.s.judge.Run()
	return f
}

func TestEvalDefaultSourceIgnoresTheJudge(t *testing.T) {
	f := calibrated(t)
	res := evalJSON(t, f.s, "group=model")
	if _, ok := res["source"]; ok {
		t.Fatal("default response grew a source key; it must stay the human-only eval")
	}
	// 45 = 40 eligible ups + the 5 stale labels on unposed images. The human
	// eval has always counted a label wherever it sits; only the judge is
	// confined to eligible images (it needs the reference).
	cell := poseCell(t, res)
	if num(cell, "up") != 45 || num(cell, "down") != 10 {
		t.Fatalf("human pose tally = %v, want 45/10", cell)
	}
	if _, ok := cell["ratedJudge"]; ok {
		t.Fatalf("human cell carries ratedJudge: %v", cell)
	}
}

func TestEvalJudgeSourceCountsOnlyGatedVerdicts(t *testing.T) {
	f := calibrated(t)
	res := evalJSON(t, f.s, "group=model&source=judge")
	if res["source"] != "judge" || res["judge"] == nil {
		t.Fatalf("source=%v judge=%v", res["source"], res["judge"])
	}
	// 53 verdicts, one abstention: 52 count. Ups: 37 + 2 (human down) + 3 unrated.
	cell := poseCell(t, res)
	if num(cell, "up") != 42 || num(cell, "down") != 10 || num(cell, "rated") != 52 ||
		num(cell, "ratedJudge") != 52 || num(cell, "ratedHuman") != 0 {
		t.Fatalf("judge pose cell = %v", cell)
	}
	// No verdicts and no gate on the source criteria: n/a, with the reason.
	src := res["total"].(map[string]any)["criteria"].(map[string]any)["source_style"].(map[string]any)
	if src["na"] != "gate: uncalibrated" {
		t.Fatalf("source_style na = %v", src["na"])
	}
	if leaders := res["leaders"].(map[string]any); leaders["source_style"] != nil {
		t.Fatal("an n/a criterion produced a leader")
	}
}

func TestEvalMergedLetsTheHumanWin(t *testing.T) {
	f := calibrated(t)
	cell := poseCell(t, evalJSON(t, f.s, "group=model&source=merged"))
	// All 55 human labels (45/10, including the 4 the judge contradicts) plus
	// the judge's 3 ups on images no human rated.
	if num(cell, "up") != 48 || num(cell, "down") != 10 ||
		num(cell, "ratedHuman") != 55 || num(cell, "ratedJudge") != 3 {
		t.Fatalf("merged pose cell = %v", cell)
	}
	if _, ok := cell["na"]; ok {
		t.Fatalf("merged cell marked n/a: %v", cell)
	}
}

func TestEvalGateDoesNotCarryOverToANewJudgeVersion(t *testing.T) {
	f := calibrated(t)
	f.s.judge = NewJudge(f.s.db, f.s.judge.gatewayURL, "vl", f.s.blobPath, f.s.judge.poses)
	res := evalJSON(t, f.s, "group=model&source=judge")
	cell := poseCell(t, res)
	if cell["na"] != "gate: uncalibrated" || num(cell, "rated") != 0 {
		t.Fatalf("vl@judge-v1 inherited judge@judge-v1's gate: %v", cell)
	}
	// merged under an uncalibrated judge is the human eval again.
	merged := poseCell(t, evalJSON(t, f.s, "group=model&source=merged"))
	if num(merged, "up") != 45 || num(merged, "ratedJudge") != 0 {
		t.Fatalf("merged under an uncalibrated judge = %v", merged)
	}
}

func TestEvalSourceErrors(t *testing.T) {
	f := newJudgeFixture(t)
	get := func(q string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		f.s.handleEval(rec, httptest.NewRequest(http.MethodGet, "/api/eval?"+q, nil))
		return rec
	}
	if rec := get("source=robot"); rec.Code != http.StatusBadRequest {
		t.Fatalf("source=robot → %d", rec.Code)
	}
	f.s.judge = NewJudge(f.s.db, "", "judge", f.s.blobPath, nil)
	if rec := get("source=judge"); rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("source=judge with the judge disabled → %d", rec.Code)
	}
	if rec := get("source=human"); rec.Code != http.StatusOK {
		t.Fatalf("source=human with the judge disabled → %d", rec.Code)
	}
}

func TestEvalCSVCarriesSourceColumns(t *testing.T) {
	f := calibrated(t)
	csvOf := func(q string) string {
		rec := httptest.NewRecorder()
		f.s.handleEval(rec, httptest.NewRequest(http.MethodGet, "/api/eval?format=csv&"+q, nil))
		return rec.Body.String()
	}
	if h := strings.SplitN(csvOf(""), "\n", 2)[0]; strings.Contains(h, "rated_judge") {
		t.Fatalf("human CSV grew judge columns: %s", h)
	}
	if h := strings.SplitN(csvOf("source=merged"), "\n", 2)[0]; !strings.Contains(h, "pose_adherence_rated_judge") {
		t.Fatalf("merged CSV header = %s", h)
	}
}

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/fstest"
)

// The judge is tested against a fake gateway that reads the OUTPUT image the
// way the real model would have to: it decodes the second data URI and answers
// from its colour. White means up, black down, grey unsure — so a fixture says
// what the judge will answer simply by what it paints.

var (
	white = color.RGBA{255, 255, 255, 255}
	black = color.RGBA{0, 0, 0, 255}
	grey  = color.RGBA{128, 128, 128, 255}
)

func pngBytes(t *testing.T, w, h int, c color.Color) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// judgeFixture is a test server with a judge wired to a fake gateway.
type judgeFixture struct {
	t        *testing.T
	s        *server
	requests atomic.Int32
	mu       sync.Mutex
	lastBody map[string]any
	reply    func(outColor color.Color) (string, int) // content, HTTP status
	n        int
}

func newJudgeFixture(t *testing.T) *judgeFixture {
	t.Helper()
	f := &judgeFixture{t: t, s: newTestServer(t)}
	f.reply = func(c color.Color) (string, int) {
		switch c {
		case white:
			return `{"verdict":"up","reason":"matches"}`, http.StatusOK
		case black:
			return `{"verdict":"down","reason":"left arm raised"}`, http.StatusOK
		}
		return `{"verdict":"unsure","reason":"figure cropped"}`, http.StatusOK
	}
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.requests.Add(1)
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		json.Unmarshal(raw, &body)
		f.mu.Lock()
		f.lastBody = body
		f.mu.Unlock()
		content, status := f.reply(outputColor(t, body))
		if status != http.StatusOK {
			http.Error(w, "upstream down", status)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": content}}},
		})
	}))
	t.Cleanup(gw.Close)
	poses := fstest.MapFS{"ol1.png": {Data: pngBytes(t, 20, 40, color.RGBA{255, 0, 0, 255})}}
	f.s.judge = NewJudge(f.s.db, gw.URL, "judge", f.s.blobPath, poses)
	judgeFilterVersion = f.s.judge.Version()
	t.Cleanup(func() { judgeFilterVersion = "" })
	f.exec(`INSERT INTO sessions (id, title, model_id) VALUES ('s1', 'Lab', 'pony')`)
	return f
}

func (f *judgeFixture) exec(q string, args ...any) {
	f.t.Helper()
	if _, err := f.s.db.Exec(q, args...); err != nil {
		f.t.Fatalf("%s: %v", q, err)
	}
}

// addImage stores a real PNG blob and its node. Returns the image id.
func (f *judgeFixture) addImage(pose, source string, c color.Color, w, h int) string {
	f.t.Helper()
	f.n++
	id, nodeID, sha := "i"+itoa(f.n), "n"+itoa(f.n), fakeSha(1000+f.n)
	f.exec(`INSERT INTO nodes (id, session_id, prompt, model_id, pose_id, source_image_id)
		VALUES (?, 's1', 'a dancer', 'pony', ?, ?)`, nodeID, nullable(pose), nullable(source))
	f.exec(`INSERT INTO blobs (sha256, size) VALUES (?, 1)`, sha)
	f.exec(`INSERT INTO images (id, node_id, idx, sha256, size, blob_present) VALUES (?, ?, 0, ?, 1, 1)`, id, nodeID, sha)
	path := f.s.blobPath(sha)
	os.MkdirAll(filepath.Dir(path), 0o755)
	if err := os.WriteFile(path, pngBytes(f.t, w, h, c), 0o644); err != nil {
		f.t.Fatal(err)
	}
	return id
}

func (f *judgeFixture) body() map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.lastBody
}

// contentParts flattens the user message into its parts.
func contentParts(body map[string]any) []map[string]any {
	var out []map[string]any
	for _, m := range body["messages"].([]any) {
		msg := m.(map[string]any)
		if msg["role"] != "user" {
			continue
		}
		for _, p := range msg["content"].([]any) {
			out = append(out, p.(map[string]any))
		}
	}
	return out
}

func decodeDataURI(t *testing.T, uri string) image.Image {
	t.Helper()
	b64, ok := strings.CutPrefix(uri, "data:image/jpeg;base64,")
	if !ok {
		t.Fatalf("not a JPEG data URI: %.40s", uri)
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		t.Fatal(err)
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func imageURIs(body map[string]any) []string {
	var out []string
	for _, p := range contentParts(body) {
		if p["type"] == "image_url" {
			out = append(out, p["image_url"].(map[string]any)["url"].(string))
		}
	}
	return out
}

// outputColor snaps the OUTPUT image's centre pixel back to the fixture
// palette (JPEG shifts it a little).
func outputColor(t *testing.T, body map[string]any) color.Color {
	uris := imageURIs(body)
	if len(uris) != 2 {
		return nil
	}
	img := decodeDataURI(t, uris[1])
	b := img.Bounds()
	r, _, _, _ := img.At(b.Dx()/2, b.Dy()/2).RGBA()
	switch {
	case r>>8 > 200:
		return white
	case r>>8 < 50:
		return black
	}
	return grey
}

func TestJudgeSendsReferenceThenOutput(t *testing.T) {
	f := newJudgeFixture(t)
	id := f.addImage("ol1", "", white, 1024, 1536)

	jd, err := f.s.judge.Judge(context.Background(), id, "pose_adherence")
	if err != nil {
		t.Fatal(err)
	}
	if jd.Verdict != 1 || jd.RefKey != "pose:ol1" || jd.Judge != "judge@judge-v1" || jd.Cached {
		t.Fatalf("judgment = %+v", jd)
	}

	body := f.body()
	if body["model"] != "judge" || body["temperature"] != float64(0) {
		t.Fatalf("model=%v temperature=%v", body["model"], body["temperature"])
	}
	var seq []string
	for _, p := range contentParts(body) {
		if p["type"] == "image_url" {
			seq = append(seq, "<image>")
		} else {
			seq = append(seq, p["text"].(string))
		}
	}
	joined := strings.Join(seq, " | ")
	if !strings.Contains(joined, "POSE ADHERENCE") ||
		!strings.Contains(joined, "REFERENCE: | <image> | OUTPUT: | <image>") {
		t.Fatalf("user content order: %s", joined)
	}

	uris := imageURIs(body)
	ref, out := decodeDataURI(t, uris[0]).Bounds(), decodeDataURI(t, uris[1]).Bounds()
	if ref.Dx() != 20 || ref.Dy() != 40 {
		t.Fatalf("reference %v, want the 20×40 pose (small images are never upscaled)", ref)
	}
	if out.Dx() != 512 || out.Dy() != judgeMaxSide {
		t.Fatalf("output %v, want 512×%d (longest side capped)", out, judgeMaxSide)
	}
}

func TestJudgeSourceCriterionShowsTheSourceImage(t *testing.T) {
	f := newJudgeFixture(t)
	src := f.addImage("", "", grey, 64, 32)
	id := f.addImage("", src, black, 64, 64)

	jd, err := f.s.judge.Judge(context.Background(), id, "source_style")
	if err != nil {
		t.Fatal(err)
	}
	if jd.Verdict != -1 || jd.RefKey != "image:"+src {
		t.Fatalf("judgment = %+v", jd)
	}
	if ref := decodeDataURI(t, imageURIs(f.body())[0]).Bounds(); ref.Dx() != 64 || ref.Dy() != 32 {
		t.Fatalf("reference %v, want the 64×32 source", ref)
	}
}

func TestParseVerdictIsTolerantOfModelHabits(t *testing.T) {
	for raw, want := range map[string]int{
		`{"verdict":"up","reason":"ok"}`:                                 1,
		"```json\n{\"verdict\": \"down\", \"reason\": \"no\"}\n```":      -1,
		`Sure! {"verdict":"UNSURE","reason":"cropped"} Hope that helps.`: 0,
		`{"verdict":" Up ","reason":""}`:                                 1,
	} {
		got, _, err := parseVerdict(raw)
		if err != nil || got != want {
			t.Errorf("parseVerdict(%q) = %d, %v; want %d", raw, got, err, want)
		}
	}
	for _, raw := range []string{`up`, `{"verdict":"yes"}`, `{"verdict":`, ``} {
		if _, _, err := parseVerdict(raw); err == nil {
			t.Errorf("parseVerdict(%q) accepted; a guessed verdict is not the model's", raw)
		}
	}
}

func countJudgments(t *testing.T, s *server) int {
	var n int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM criteria_judgments`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestJudgeUnparseableReplyStoresNothing(t *testing.T) {
	f := newJudgeFixture(t)
	f.reply = func(color.Color) (string, int) { return "I think it looks good.", http.StatusOK }
	id := f.addImage("ol1", "", white, 32, 32)
	if _, err := f.s.judge.Judge(context.Background(), id, "pose_adherence"); err == nil {
		t.Fatal("prose reply accepted")
	}
	if n := countJudgments(t, f.s); n != 0 {
		t.Fatalf("%d verdicts stored after a failed parse", n)
	}
}

func TestJudgeHasNoFallback(t *testing.T) {
	f := newJudgeFixture(t)
	f.reply = func(color.Color) (string, int) { return "", http.StatusBadGateway }
	id := f.addImage("ol1", "", white, 32, 32)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/images/"+id+"/judge?criterion=pose_adherence", nil)
	req.SetPathValue("id", id)
	f.s.handleJudgeImage(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("gateway down → %d, want 502", rec.Code)
	}
	if got := f.requests.Load(); got != 1 {
		t.Fatalf("%d gateway calls, want exactly 1 (no retry under another model)", got)
	}
	if n := countJudgments(t, f.s); n != 0 {
		t.Fatalf("%d verdicts stored with the gateway down", n)
	}
}

func TestJudgeCachesByImageCriterionAndVersion(t *testing.T) {
	f := newJudgeFixture(t)
	id := f.addImage("ol1", "", white, 32, 32)
	ctx := context.Background()
	if _, err := f.s.judge.Judge(ctx, id, "pose_adherence"); err != nil {
		t.Fatal(err)
	}
	jd, err := f.s.judge.Judge(ctx, id, "pose_adherence")
	if err != nil || !jd.Cached || jd.Verdict != 1 {
		t.Fatalf("second call = %+v, %v; want cached up", jd, err)
	}
	if got := f.requests.Load(); got != 1 {
		t.Fatalf("%d gateway calls, want 1", got)
	}
	// Another alias behind the same prompt is another judge: no cache hit.
	other, err := f.s.judge.JudgeWith(ctx, "vl", id, "pose_adherence", true)
	if err != nil || other.Cached || other.Judge != "vl@judge-v1" {
		t.Fatalf("other model = %+v, %v", other, err)
	}
	// -no-cache measures live and leaves the table alone.
	if _, err := f.s.judge.JudgeWith(ctx, "judge", id, "pose_adherence", false); err != nil {
		t.Fatal(err)
	}
	if got, n := f.requests.Load(), countJudgments(t, f.s); got != 3 || n != 2 {
		t.Fatalf("requests=%d stored=%d, want 3 and 2", got, n)
	}
}

func judgeStatus(t *testing.T, f *judgeFixture, id, criterion string) int {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/images/"+id+"/judge?criterion="+criterion, nil)
	req.SetPathValue("id", id)
	f.s.handleJudgeImage(rec, req)
	return rec.Code
}

func TestJudgeEndpointStatuses(t *testing.T) {
	f := newJudgeFixture(t)
	unposed := f.addImage("", "", white, 32, 32)
	missingPose := f.addImage("ol9", "", white, 32, 32)
	missingSource := f.addImage("", "gone", white, 32, 32)

	for _, tc := range []struct {
		id, criterion string
		want          int
	}{
		{unposed, "pose_adherence", http.StatusUnprocessableEntity},
		{unposed, "source_identity", http.StatusUnprocessableEntity},
		{missingPose, "pose_adherence", http.StatusConflict},
		{missingSource, "source_identity", http.StatusConflict},
		{"nope", "pose_adherence", http.StatusNotFound},
		{unposed, "aesthetics", http.StatusBadRequest},
	} {
		if got := judgeStatus(t, f, tc.id, tc.criterion); got != tc.want {
			t.Errorf("%s %s → %d, want %d", tc.id, tc.criterion, got, tc.want)
		}
	}
	if got := f.requests.Load(); got != 0 {
		t.Fatalf("%d gateway calls for pairs that could not be built", got)
	}
	if _, err := f.s.judge.Judge(context.Background(), missingPose, "pose_adherence"); !errors.Is(err, ErrNoReference) {
		t.Fatalf("missing pose: %v, want ErrNoReference", err)
	}
	if n := countJudgments(t, f.s); n != 0 {
		t.Fatalf("missing reference recorded as %d verdicts; it must not become an abstention", n)
	}

	f.s.judge = NewJudge(f.s.db, "", "judge", f.s.blobPath, nil)
	if got := judgeStatus(t, f, unposed, "pose_adherence"); got != http.StatusServiceUnavailable {
		t.Fatalf("disabled judge → %d, want 503", got)
	}
}

func detailJSON(t *testing.T, s *server, id string) map[string]any {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/images/"+id, nil)
	req.SetPathValue("id", id)
	s.handleImageDetail(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("detail → %d: %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	json.Unmarshal(rec.Body.Bytes(), &out)
	return out
}

// The human must not see the verdict before rating, or κ measures deference.
func TestDetailHidesVerdictUntilHumanRates(t *testing.T) {
	f := newJudgeFixture(t)
	id := f.addImage("ol1", "", black, 32, 32)
	if _, err := f.s.judge.Judge(context.Background(), id, "pose_adherence"); err != nil {
		t.Fatal(err)
	}

	d := detailJSON(t, f.s, id)
	if j := d["judgments"].(map[string]any); len(j) != 0 {
		t.Fatalf("verdict visible before the human rated: %v", j)
	}
	if h := d["judgedHidden"].([]any); len(h) != 1 || h[0] != "pose_adherence" {
		t.Fatalf("judgedHidden = %v", h)
	}

	f.exec(`INSERT INTO image_criteria (image_id, criterion, score) VALUES (?, 'pose_adherence', 1)`, id)
	d = detailJSON(t, f.s, id)
	jd, ok := d["judgments"].(map[string]any)["pose_adherence"].(map[string]any)
	if !ok || jd["verdict"] != float64(-1) || jd["reason"] != "left arm raised" {
		t.Fatalf("after rating: %v", d["judgments"])
	}
	if h := d["judgedHidden"].([]any); len(h) != 0 {
		t.Fatalf("judgedHidden after rating = %v", h)
	}
}

func TestGalleryJudgeFilters(t *testing.T) {
	f := newJudgeFixture(t)
	agree := f.addImage("ol1", "", white, 16, 16)
	disagree := f.addImage("ol1", "", black, 16, 16)
	unrated := f.addImage("ol1", "", white, 16, 16)
	abstain := f.addImage("ol1", "", grey, 16, 16)
	for _, id := range []string{agree, disagree, unrated, abstain} {
		if _, err := f.s.judge.Judge(context.Background(), id, "pose_adherence"); err != nil {
			t.Fatal(err)
		}
	}
	for _, id := range []string{agree, disagree, abstain} {
		f.exec(`INSERT INTO image_criteria (image_id, criterion, score) VALUES (?, 'pose_adherence', 1)`, id)
	}

	ids := func(filter string) []string {
		where, args, bad := galleryFilter(map[string][]string{"judge": {filter}})
		if bad != "" {
			t.Fatalf("%s: %s", filter, bad)
		}
		rows, err := f.s.db.Query(`SELECT g.id FROM gallery g WHERE `+where+` ORDER BY g.id`, args...)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()
		var out []string
		for rows.Next() {
			var id string
			rows.Scan(&id)
			out = append(out, id)
		}
		return out
	}
	if got := ids("pose_adherence:disagree"); len(got) != 1 || got[0] != disagree {
		t.Fatalf("disagree = %v, want [%s] (an abstention is not a disagreement)", got, disagree)
	}
	if got := ids("pose_adherence:unrated"); len(got) != 1 || got[0] != unrated {
		t.Fatalf("unrated = %v, want [%s]", got, unrated)
	}
	if _, _, bad := galleryFilter(map[string][]string{"judge": {"pose_adherence:agree"}}); bad == "" {
		t.Fatal("judge=…:agree accepted")
	}
	judgeFilterVersion = ""
	if _, _, bad := galleryFilter(map[string][]string{"judge": {"pose_adherence:unrated"}}); bad == "" {
		t.Fatal("judge filter accepted with the judge disabled")
	}
}

func TestJudgeSweepQueuesOnlyUnjudgedEligible(t *testing.T) {
	f := newJudgeFixture(t)
	done := f.addImage("ol1", "", white, 16, 16)
	f.addImage("ol1", "", white, 16, 16)
	f.addImage("ol1", "", black, 16, 16)
	f.addImage("", "", white, 16, 16) // no pose: not eligible
	if _, err := f.s.judge.Judge(context.Background(), done, "pose_adherence"); err != nil {
		t.Fatal(err)
	}
	queued, err := f.s.judge.sweep("pose_adherence", 100)
	if err != nil || queued != 2 {
		t.Fatalf("sweep queued %d, %v; want 2", queued, err)
	}
	close(f.s.judge.jobs)
	f.s.judge.Run() // drains synchronously once the channel is closed
	if n := countJudgments(t, f.s); n != 3 {
		t.Fatalf("%d verdicts after the sweep, want 3", n)
	}
}

func TestJudgeSweepStopsAtFirstGatewayFailure(t *testing.T) {
	f := newJudgeFixture(t)
	for range 5 {
		f.addImage("ol1", "", white, 16, 16)
	}
	f.reply = func(color.Color) (string, int) { return "", http.StatusServiceUnavailable }
	if queued, err := f.s.judge.sweep("pose_adherence", 100); err != nil || queued != 5 {
		t.Fatalf("sweep queued %d, %v", queued, err)
	}
	close(f.s.judge.jobs)
	f.s.judge.Run()
	if got := f.requests.Load(); got != 1 {
		t.Fatalf("%d gateway calls into a closed window, want 1", got)
	}
}

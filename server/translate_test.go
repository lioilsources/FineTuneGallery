package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// fakeGateway is an OpenAI-compatible chat endpoint that answers with a fixed
// tag line and counts calls — the count is how the cache test knows a second
// request never reached the model.
func fakeGateway(t *testing.T, reply string, status int) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Model       string           `json:"model"`
			Temperature float64          `json:"temperature"`
			Messages    []gatewayMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("gateway got undecodable body: %v", err)
		}
		if req.Temperature != 0 {
			t.Errorf("temperature = %v, want 0 — a warm translator is not deterministic", req.Temperature)
		}
		if len(req.Messages) == 0 || req.Messages[0].Role != "system" {
			t.Errorf("first message must be the system prompt, got %+v", req.Messages)
		}
		if status != http.StatusOK {
			w.WriteHeader(status)
			w.Write([]byte(`{"error":"upstream down"}`))
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"model": req.Model,
			"choices": []map[string]any{{
				"message": map[string]any{"role": "assistant", "content": reply},
			}},
		})
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

var testVocab = []string{
	"1girl", "solo", "ballerina", "ballet slippers", "en pointe", "leotard",
	"from side", "full body", "window", "sunlight", "long_hair", "smile",
}

func newTestTranslator(t *testing.T, gatewayURL string) (*server, *Translator) {
	t.Helper()
	s := newTestServer(t)
	tr := NewTranslator(s.db, gatewayURL, "prompt-tags", "")
	tr.SetVocab(testVocab)
	s.translator = tr
	return s, tr
}

func TestParseTagsIsTolerantOfModelHabits(t *testing.T) {
	got := parseTags("```\n1girl, Solo,  ballet_slippers,\nfrom side, from side, \"en pointe\".\n```")
	want := []string{"1girl", "solo", "ballet slippers", "from side", "en pointe"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("parseTags = %v, want %v", got, want)
	}
}

func TestFilterTagsDropsWhatTheTaggerCannotSee(t *testing.T) {
	tr := NewTranslator(nil, "http://x", "m", "")
	tr.SetVocab(testVocab)
	// Through parseTags first — filterTags trusts its input is normalised,
	// which is the production path and the only one worth testing.
	kept, dropped := filterTags(parseTags("1girl, masterpiece, score_9, ballet slippers, glowing aura of destiny"), tr.vocab.Load())
	if strings.Join(kept, ",") != "1girl,ballet slippers" {
		t.Fatalf("kept = %v", kept)
	}
	// The drops are reported, not swallowed: the benchmark's vocab-hit metric
	// and a curious human both need to see what the model invented.
	if strings.Join(dropped, ",") != "masterpiece,score 9,glowing aura of destiny" {
		t.Fatalf("dropped = %v", dropped)
	}
}

func TestTranslateCachesByTextAndVersion(t *testing.T) {
	gw, calls := fakeGateway(t, "1girl, solo, ballerina, en pointe, from side, invented tag", http.StatusOK)
	_, tr := newTestTranslator(t, gw.URL)

	first, err := tr.Translate(context.Background(), "a ballerina seen from the side")
	if err != nil {
		t.Fatal(err)
	}
	if first.Cached {
		t.Fatal("first answer reported as cached")
	}
	if first.Text != "1girl, solo, ballerina, en pointe, from side" {
		t.Fatalf("text = %q", first.Text)
	}
	if strings.Join(first.Dropped, ",") != "invented tag" {
		t.Fatalf("dropped = %v", first.Dropped)
	}
	if first.Translator != "prompt-tags@"+promptVersion {
		t.Fatalf("translator = %q", first.Translator)
	}

	// Same text again: served from the table, the model is never asked.
	second, err := tr.Translate(context.Background(), "  a ballerina seen from the side ")
	if err != nil {
		t.Fatal(err)
	}
	if !second.Cached || second.Text != first.Text {
		t.Fatalf("second = %+v, want cached copy of first", second)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("gateway called %d times, want 1 — the cache is what makes two callers agree", got)
	}

	// A different model alias is a different version and must not share the
	// entry: the benchmark compares aliases on identical prompts.
	if _, err := tr.TranslateWith(context.Background(), "other", "a ballerina seen from the side", true); err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("gateway called %d times after a second alias, want 2", got)
	}
}

func TestTranslateHasNoFallback(t *testing.T) {
	gw, _ := fakeGateway(t, "", http.StatusBadGateway)
	s, _ := newTestTranslator(t, gw.URL)

	body := bytes.NewBufferString(`{"text":"a ballerina","language":"tags"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/translate", body)
	rec := httptest.NewRecorder()
	s.handleTranslate(rec, req)
	// 502, not a quietly different answer. The app reads this as "send raw
	// prose, record translator=null" — which the eval can then tell apart.
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("gateway failure → %d, want 502; body %s", rec.Code, rec.Body.String())
	}
}

func TestTranslateRefusesWithoutVocabulary(t *testing.T) {
	gw, calls := fakeGateway(t, "1girl", http.StatusOK)
	s := newTestServer(t)
	s.translator = NewTranslator(s.db, gw.URL, "prompt-tags", "") // no SetVocab
	_, err := s.translator.Translate(context.Background(), "a ballerina")
	if !errors.Is(err, ErrNoVocabulary) {
		t.Fatalf("err = %v, want ErrNoVocabulary", err)
	}
	if calls.Load() != 0 {
		t.Fatal("model was asked although nothing could filter its answer")
	}
}

func TestTranslateEndpointShape(t *testing.T) {
	gw, _ := fakeGateway(t, "1girl, ballerina, leotard", http.StatusOK)
	s, _ := newTestTranslator(t, gw.URL)

	do := func(body string) (int, map[string]any) {
		req := httptest.NewRequest(http.MethodPost, "/api/translate", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()
		s.handleTranslate(rec, req)
		var out map[string]any
		json.Unmarshal(rec.Body.Bytes(), &out)
		return rec.Code, out
	}

	code, out := do(`{"text":"a ballerina in a leotard"}`)
	if code != 200 {
		t.Fatalf("→ %d %v", code, out)
	}
	if out["text"] != "1girl, ballerina, leotard" || out["translator"] != "prompt-tags@"+promptVersion {
		t.Fatalf("out = %v", out)
	}

	if code, _ := do(`{"text":"x","language":"prose"}`); code != 400 {
		t.Fatalf("prose → %d, want 400 (only tags is a translation)", code)
	}
	// Empty text is not an error and not a gateway call: a photo root has no
	// prompt and must not become a prompt.
	code, out = do(`{"text":"   "}`)
	if code != 200 || out["text"] != "" {
		t.Fatalf("empty → %d %v", code, out)
	}
}

func TestTranslateDisabledIs503(t *testing.T) {
	s := newTestServer(t)
	s.translator = NewTranslator(s.db, "", "prompt-tags", "")
	req := httptest.NewRequest(http.MethodPost, "/api/translate", bytes.NewBufferString(`{"text":"x"}`))
	rec := httptest.NewRecorder()
	s.handleTranslate(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("disabled → %d, want 503 (distinct from a failing gateway's 502)", rec.Code)
	}
	meta := s.translatorMeta()
	if meta["enabled"] != false {
		t.Fatalf("meta = %v", meta)
	}
}

func TestLoadVocabFromTagger(t *testing.T) {
	tagger := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tags" {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"model":   "SmilingWolf/wd-swinv2-tagger-v3",
			"general": []string{"1girl", "long_hair", "Looking At Viewer"},
		})
	}))
	defer tagger.Close()
	tr := NewTranslator(nil, "http://x", "m", tagger.URL)
	if err := tr.LoadVocab(); err != nil {
		t.Fatal(err)
	}
	if tr.VocabSize() != 3 {
		t.Fatalf("vocab size = %d", tr.VocabSize())
	}
	// Spelling is normalised on the way in, so the tagger's underscores and a
	// model's spaces meet in one form.
	kept, _ := filterTags([]string{"long hair", "looking at viewer"}, tr.vocab.Load())
	if len(kept) != 2 {
		t.Fatalf("kept = %v", kept)
	}
}

// v3 → v4 must keep every node and leave prompt_sent / translator NULL: those
// images are the raw-prose control arm of the translation A/B.
func TestMigrationV4KeepsV3Rows(t *testing.T) {
	s := newTestServer(t)
	var version int
	if err := s.db.QueryRow(`PRAGMA user_version`).Scan(&version); err != nil || version != schemaVersion() {
		t.Fatalf("user_version = %d, want %d", version, schemaVersion())
	}
	exec := func(q string, args ...any) {
		t.Helper()
		if _, err := s.db.Exec(q, args...); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	exec(`INSERT INTO sessions (id, title, model_id) VALUES ('s', 't', 'pony')`)
	exec(`INSERT INTO nodes (id, session_id, prompt, model_id) VALUES ('n', 's', 'a cat', 'pony')`)
	exec(`INSERT INTO blobs (sha256, size) VALUES (?, 1)`, fakeSha(1))
	exec(`INSERT INTO images (id, node_id, idx, sha256, size, blob_present) VALUES ('i', 'n', 0, ?, 1, 1)`, fakeSha(1))
	var sent, translator any
	if err := s.db.QueryRow(`SELECT prompt_sent, translator FROM gallery WHERE id = 'i'`).Scan(&sent, &translator); err != nil {
		t.Fatalf("gallery view after v4: %v", err)
	}
	if sent != nil || translator != nil {
		t.Fatalf("pre-v4 row got prompt_sent=%v translator=%v, want NULL", sent, translator)
	}
	// And the translations table is there for the cache.
	exec(`INSERT INTO translations (hash, text, language, translator, tags) VALUES ('h', 'x', 'tags', 'm@v1', '{"tags":[]}')`)
}

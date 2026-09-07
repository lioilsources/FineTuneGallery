package main

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"testing"
)

func seedBenchCorpus(t *testing.T, s *server) {
	t.Helper()
	exec := func(q string, args ...any) {
		t.Helper()
		if _, err := s.db.Exec(q, args...); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	exec(`INSERT INTO sessions (id, title, model_id) VALUES ('s', 'bench', 'pony')`)
	add := func(i int, prompt, tagsJSON string) {
		id, node, sha := fmt.Sprintf("bi%d", i), fmt.Sprintf("bn%d", i), fakeSha(100+i)
		exec(`INSERT INTO nodes (id, session_id, prompt, model_id) VALUES (?, 's', ?, 'pony')`, node, prompt)
		exec(`INSERT INTO blobs (sha256, size) VALUES (?, 1)`, sha)
		exec(`INSERT INTO images (id, node_id, idx, sha256, size, blob_present) VALUES (?, ?, 0, ?, 1, 1)`, id, node, sha)
		if tagsJSON != "" {
			exec(`INSERT INTO captions (image_id, kind, text, tagger, tags_json) VALUES (?, 'auto', '', 'wd14', ?)`, id, tagsJSON)
		}
	}
	// A real prompt with a real caption: truth = {1girl, ballerina, en pointe}
	// at ≥ 0.5; "window" is below threshold and must not count.
	add(1, "a ballerina on pointe by a tall window",
		`{"general":{"1girl":0.99,"ballerina":0.9,"en_pointe":0.7,"window":0.3},"character":{},"rating":{}}`)
	// Same prompt rendered again → must not be sampled twice.
	add(2, "a ballerina on pointe by a tall window",
		`{"general":{"1girl":0.99,"ballerina":0.8},"character":{},"rating":{}}`)
	// Too short to translate.
	add(3, "cat", `{"general":{"cat":0.9},"character":{},"rating":{}}`)
	// No caption yet → no ground truth.
	add(4, "a dancer resting after rehearsal in the studio", "")
}

func TestSampleBenchPairsIsOnePerPromptWithTruth(t *testing.T) {
	s := newTestServer(t)
	seedBenchCorpus(t, s)
	pairs, err := sampleBenchPairs(s.db, 100, 0.5, 4, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 1 {
		t.Fatalf("got %d pairs, want 1 (dedup by prompt, drop short, drop uncaptioned)", len(pairs))
	}
	p := pairs[0]
	if len(p.truth) != 3 {
		t.Fatalf("truth = %v, want 3 tags ≥ 0.5", p.truth)
	}
	if _, ok := p.truth["en pointe"]; !ok {
		t.Fatalf("truth spelling not normalised: %v", p.truth)
	}
}

func TestBenchScoresAgainstWD14(t *testing.T) {
	s := newTestServer(t)
	seedBenchCorpus(t, s)
	// The model finds two of the three true tags and invents one that exists
	// in the vocabulary (leotard) and one that does not.
	gw, _ := fakeGateway(t, "1girl, ballerina, leotard, sparkle dust of legend", http.StatusOK)
	tr := NewTranslator(s.db, gw.URL, "", "")
	tr.SetVocab([]string{"1girl", "ballerina", "en pointe", "leotard"})

	pairs, _ := sampleBenchPairs(s.db, 100, 0.5, 4, 1)
	res := benchModel(context.Background(), tr, "prompt-tags", pairs, false, false)
	if res.n != 1 || res.errors != 0 {
		t.Fatalf("n=%d errors=%d", res.n, res.errors)
	}
	approx := func(got, want float64, what string) {
		t.Helper()
		if math.Abs(got-want) > 1e-9 {
			t.Fatalf("%s = %.3f, want %.3f", what, got, want)
		}
	}
	approx(res.recall, 2.0/3.0, "recall")       // 1girl, ballerina of {1girl, ballerina, en pointe}
	approx(res.precision, 2.0/3.0, "precision") // 2 of 3 kept tags were in the picture
	approx(res.vocabHit, 3.0/4.0, "vocab hit")  // 3 of 4 emitted tags exist
	approx(res.avgTags, 3, "avg tags")
	if res.p50 == 0 {
		t.Fatal("uncached call should have a latency")
	}
}

func TestBenchCountsGatewayErrorsInsteadOfHiding(t *testing.T) {
	s := newTestServer(t)
	seedBenchCorpus(t, s)
	gw, _ := fakeGateway(t, "", http.StatusBadGateway)
	tr := NewTranslator(s.db, gw.URL, "", "")
	tr.SetVocab([]string{"1girl"})
	pairs, _ := sampleBenchPairs(s.db, 100, 0.5, 4, 1)
	res := benchModel(context.Background(), tr, "down", pairs, false, false)
	if res.errors != 1 || res.n != 0 {
		t.Fatalf("errors=%d n=%d — a dead alias must show as errors, not as 0%% recall", res.errors, res.n)
	}
}

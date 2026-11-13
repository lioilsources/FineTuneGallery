package main

import (
	"strings"
	"testing"
)

func TestStyleFacetsMapIsConsistent(t *testing.T) {
	if _, err := parseStyleFacets(styleFacetsJSON); err != nil {
		t.Fatal(err)
	}
	for id, s := range styleFacets.Styles {
		if s.Label == "" || s.Technique == "" {
			t.Errorf("%s: label and technique are required", id)
		}
	}
	// Every listed artist has at least one style; an orphan would be a
	// dropdown entry that can never match anything.
	used := map[string]bool{}
	for _, s := range styleFacets.Styles {
		used[s.Artist] = true
	}
	for a := range styleFacets.Artists {
		if !used[a] {
			t.Errorf("artist %s has no style", a)
		}
	}
	if _, err := parseStyleFacets([]byte(`{"epochs":[],"techniques":[{"id":"painting","label":"P"}],
		"artists":{},"styles":{"x":{"label":"X","epoch":"jurassic","technique":"painting"}}}`)); err == nil {
		t.Error("unknown epoch was accepted")
	}
}

func countWhere(t *testing.T, s *server, q map[string][]string) int {
	t.Helper()
	where, args, bad := galleryFilter(q)
	if bad != "" {
		t.Fatal(bad)
	}
	var n int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM gallery g WHERE `+where, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestFilterByStyleFacetsAndPrompt(t *testing.T) {
	s := newTestServer(t)
	ingestJSON(t, s, labManifest) // n1 ukiyoe, n2 baroque
	markBlobsPresent(t, s, shaA, shaB)

	cases := []struct {
		q    map[string][]string
		want int
	}{
		{map[string][]string{"technique": {"printmaking"}}, 1}, // ukiyoe
		{map[string][]string{"technique": {"painting"}}, 1},    // baroque
		{map[string][]string{"epoch": {"baroque"}}, 1},
		{map[string][]string{"epoch": {"c19"}, "technique": {"painting"}}, 0},
		{map[string][]string{"artist": {"hokusai"}}, 0},
		{map[string][]string{"prompt": {"BALLERINA"}}, 2},
		{map[string][]string{"prompt": {"ballerina ukiyo"}}, 1},
		{map[string][]string{"prompt": {"ballerina%"}}, 0}, // % is literal
		{map[string][]string{"prompt": {"   "}}, 2},
	}
	for _, c := range cases {
		if got := countWhere(t, s, c.q); got != c.want {
			t.Errorf("%v = %d, chci %d", c.q, got, c.want)
		}
	}
	if _, _, bad := galleryFilter(map[string][]string{"epoch": {"jurassic"}}); bad == "" {
		t.Error("unknown epoch was accepted")
	}
	if _, _, bad := galleryFilter(map[string][]string{"prompt": {strings.Repeat("a", 201)}}); bad == "" {
		t.Error("a 201-byte prompt filter was accepted")
	}
}

func TestStyleFacetsMetaOnlyOffersIngestedValues(t *testing.T) {
	m := styleFacets.facetsFor([]string{"hokusai", "monet", "not-in-map"})
	if len(m.Unclassified) != 1 || m.Unclassified[0] != "not-in-map" {
		t.Errorf("unclassified = %v", m.Unclassified)
	}
	if len(m.Artists) != 2 || m.Artists[0].ID != "monet" || m.Artists[1].ID != "hokusai" {
		t.Errorf("artists = %v", m.Artists) // sorted by label: Claude Monet, Katsushika Hokusai
	}
	if len(m.Epochs) != 1 || m.Epochs[0].ID != "c19" || m.Epochs[0].Styles != 2 {
		t.Errorf("epochs = %v", m.Epochs)
	}
	if len(m.Techniques) != 2 || len(m.Styles) != 3 {
		t.Errorf("techniques = %v, styles = %d", m.Techniques, len(m.Styles))
	}
}

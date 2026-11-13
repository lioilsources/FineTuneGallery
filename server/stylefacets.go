package main

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Every art-style preset id is split into three facets: the epoch it belongs
// to, the artist it imitates (only for artist styles) and the technique it is
// rendered in. One style carries all three, so "technique = printmaking"
// finds Hokusai and Warhol as well as plain linocut.
//
// The map is hand-classified from the app's restyle registry (MangaPrompts
// lib/config/restyle_styles.dart) for the ids that have arrived in the
// gallery. An id missing here still works with style=; meta lists it under
// "unclassified" so the gap is visible.

//go:embed stylefacets.json
var styleFacetsJSON []byte

type facetValue struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

type styleFacet struct {
	Label     string `json:"label"`
	Epoch     string `json:"epoch,omitempty"`
	Artist    string `json:"artist,omitempty"`
	Technique string `json:"technique"`
}

type styleFacetMap struct {
	Epochs     []facetValue          `json:"epochs"`
	Techniques []facetValue          `json:"techniques"`
	Artists    map[string]string     `json:"artists"`
	Styles     map[string]styleFacet `json:"styles"`
}

var styleFacets = mustStyleFacets(styleFacetsJSON)

func mustStyleFacets(raw []byte) *styleFacetMap {
	m, err := parseStyleFacets(raw)
	if err != nil {
		panic(err)
	}
	return m
}

func parseStyleFacets(raw []byte) (*styleFacetMap, error) {
	var m styleFacetMap
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("stylefacets.json: %w", err)
	}
	has := func(vs []facetValue, id string) bool {
		for _, v := range vs {
			if v.ID == id {
				return true
			}
		}
		return false
	}
	for id, s := range m.Styles {
		if s.Epoch != "" && !has(m.Epochs, s.Epoch) {
			return nil, fmt.Errorf("stylefacets.json: %s: unknown epoch %q", id, s.Epoch)
		}
		if !has(m.Techniques, s.Technique) {
			return nil, fmt.Errorf("stylefacets.json: %s: unknown technique %q", id, s.Technique)
		}
		if _, ok := m.Artists[s.Artist]; s.Artist != "" && !ok {
			return nil, fmt.Errorf("stylefacets.json: %s: unknown artist %q", id, s.Artist)
		}
	}
	return &m, nil
}

// stylesWhere returns the style ids whose facet equals val, sorted, and
// whether val is a known value of that facet at all.
func (m *styleFacetMap) stylesWhere(facet, val string) ([]string, bool) {
	known := false
	switch facet {
	case "epoch":
		for _, v := range m.Epochs {
			known = known || v.ID == val
		}
	case "technique":
		for _, v := range m.Techniques {
			known = known || v.ID == val
		}
	case "artist":
		_, known = m.Artists[val]
	}
	if !known {
		return nil, false
	}
	ids := []string{}
	for id, s := range m.Styles {
		got := map[string]string{"epoch": s.Epoch, "technique": s.Technique, "artist": s.Artist}[facet]
		if got == val {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids, true
}

// styleFacetFilter adds epoch=, artist= and technique= (AND with each other
// and with style=).
func styleFacetFilter(get func(string) string, conds []string, args []any) ([]string, []any, string) {
	for _, facet := range []string{"epoch", "artist", "technique"} {
		v := get(facet)
		if v == "" {
			continue
		}
		ids, ok := styleFacets.stylesWhere(facet, v)
		if !ok {
			return nil, nil, "unknown " + facet + " " + v
		}
		if len(ids) == 0 {
			conds = append(conds, "0=1")
			continue
		}
		conds = append(conds, "g.style_id IN (?"+strings.Repeat(",?", len(ids)-1)+")")
		for _, id := range ids {
			args = append(args, id)
		}
	}
	return conds, args, ""
}

// promptFilter matches every whitespace-separated word of prompt= as a
// case-insensitive substring of the prompt the user wrote or the one that was
// sent to the model (the translated tags).
func promptFilter(get func(string) string, conds []string, args []any) ([]string, []any, string) {
	v := strings.TrimSpace(get("prompt"))
	if v == "" {
		return conds, args, ""
	}
	if len(v) > 200 {
		return nil, nil, "prompt filter is longer than 200 bytes"
	}
	esc := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	for _, w := range strings.Fields(v) {
		pat := "%" + esc.Replace(w) + "%"
		conds = append(conds, `(g.prompt LIKE ? ESCAPE '\' OR COALESCE(g.prompt_sent, '') LIKE ? ESCAPE '\')`)
		args = append(args, pat, pat)
	}
	return conds, args, ""
}

type facetCount struct {
	ID     string `json:"id"`
	Label  string `json:"label"`
	Styles int    `json:"styles"` // ingested styles with this value
}

type styleFacetsMeta struct {
	Epochs       []facetCount `json:"epochs"`
	Artists      []facetCount `json:"artists"`
	Techniques   []facetCount `json:"techniques"`
	Styles       []styleRow   `json:"styles"`
	Unclassified []string     `json:"unclassified"`
}

type styleRow struct {
	ID string `json:"id"`
	styleFacet
}

// facetsFor narrows the map to the styles that have actually arrived: a
// facet value no image can match is not offered.
func (m *styleFacetMap) facetsFor(ingested []string) styleFacetsMeta {
	out := styleFacetsMeta{
		Epochs: []facetCount{}, Artists: []facetCount{}, Techniques: []facetCount{},
		Styles: []styleRow{}, Unclassified: []string{},
	}
	n := map[string]int{}
	for _, id := range ingested {
		s, ok := m.Styles[id]
		if !ok {
			out.Unclassified = append(out.Unclassified, id)
			out.Styles = append(out.Styles, styleRow{ID: id, styleFacet: styleFacet{Label: id}})
			continue
		}
		out.Styles = append(out.Styles, styleRow{ID: id, styleFacet: s})
		n["epoch:"+s.Epoch]++
		n["artist:"+s.Artist]++
		n["technique:"+s.Technique]++
	}
	for _, v := range m.Epochs {
		if c := n["epoch:"+v.ID]; c > 0 {
			out.Epochs = append(out.Epochs, facetCount{v.ID, v.Label, c})
		}
	}
	for _, v := range m.Techniques {
		if c := n["technique:"+v.ID]; c > 0 {
			out.Techniques = append(out.Techniques, facetCount{v.ID, v.Label, c})
		}
	}
	for id, label := range m.Artists {
		if c := n["artist:"+id]; c > 0 {
			out.Artists = append(out.Artists, facetCount{id, label, c})
		}
	}
	sort.Slice(out.Artists, func(i, j int) bool { return out.Artists[i].Label < out.Artists[j].Label })
	return out
}

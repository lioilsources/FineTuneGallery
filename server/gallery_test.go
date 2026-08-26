package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// The migration is the part that cannot be tried out on the NAS: it runs once,
// against a database that already holds every rated image. So it is exercised
// here the way it will be there — v1 first, with rows in it, then v2 on top.

func newTestServer(t *testing.T) *server {
	t.Helper()
	dir := t.TempDir()
	for _, sub := range []string{"images", "thumbs", "db"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	db, err := openDB(filepath.Join(dir, "db", "test.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return &server{db: db, dataDir: dir}
}

func TestMigrationV2KeepsExistingRows(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v1.sqlite")

	// A genuine v1 database: schema v1 only, stamped v1 — the state the NAS is
	// in right now. openDB would have applied v2 as well.
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(ON)")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(schemaV1); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`PRAGMA user_version = 1`); err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{
		`INSERT INTO sessions (id, title, model_id) VALUES ('s1', 'old', 'pony')`,
		`INSERT INTO nodes (id, session_id, prompt, model_id, seed) VALUES ('n1', 's1', 'a cat', 'pony', 42)`,
		`INSERT INTO blobs (sha256, size) VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 1)`,
		`INSERT INTO images (id, node_id, idx, sha256, size, blob_present) VALUES ('i1','n1',0,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',1,1)`,
		`INSERT INTO ratings (image_id, score) VALUES ('i1', 1)`,
	} {
		if _, err := db.Exec(q); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()

	// Reopening applies v2.
	db, err = openDB(path)
	if err != nil {
		t.Fatalf("migrace v2 spadla: %v", err)
	}
	defer db.Close()

	var version int
	if err := db.QueryRow(`PRAGMA user_version`).Scan(&version); err != nil || version != 2 {
		t.Fatalf("user_version = %d (%v)", version, err)
	}
	// The old row is untouched and now has the new columns, empty.
	var prompt string
	var styleID, loraStrength any
	var isRepose int
	if err := db.QueryRow(
		`SELECT prompt, style_id, lora_strength, is_repose FROM nodes WHERE id='n1'`).
		Scan(&prompt, &styleID, &loraStrength, &isRepose); err != nil {
		t.Fatal(err)
	}
	if prompt != "a cat" || styleID != nil || loraStrength != nil || isRepose != 0 {
		t.Errorf("uzel po migraci: %q %v %v %d", prompt, styleID, loraStrength, isRepose)
	}
	// The rating survived, and the recreated view still finds the image — a
	// view is not updated by ALTER TABLE, so this is the part that breaks if
	// the DROP/CREATE is forgotten.
	var score int
	var gotStyle any
	if err := db.QueryRow(
		`SELECT score, style_id FROM gallery WHERE id='i1'`).Scan(&score, &gotStyle); err != nil {
		t.Fatalf("pohled gallery: %v", err)
	}
	if score != 1 {
		t.Errorf("hodnocení = %d", score)
	}
	// Migrating twice must be a no-op, not a second DROP VIEW.
	db.Close()
	if db, err = openDB(path); err != nil {
		t.Fatalf("druhé otevření: %v", err)
	}
	db.Close()
}

func ingestJSON(t *testing.T, s *server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/api/ingest/manifest", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()
	s.handleIngestManifest(rec, req)
	if rec.Code != 200 {
		t.Fatalf("ingest %d: %s", rec.Code, rec.Body.String())
	}
	return rec
}

const (
	shaA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	shaB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

// labManifest is the shape tools/lab sends: one cell per node, with the style
// and LoRA strength that produced it.
const labManifest = `{
  "session": {"id":"s1","title":"baletka [lab]","modelId":"illustrious-xl","updatedAt":"2026-08-26T06:00:00Z"},
  "nodes": [
    {"id":"n1","prompt":"a ballerina, ukiyo-e","modelId":"illustrious-xl",
     "styleId":"ukiyoe","loraName":"style-usnr.safetensors","loraStrength":0.65,
     "seed":777,"steps":30,"cfg":6.0,"samplerName":"dpmpp_2m","scheduler":"karras",
     "images":[{"id":"i1","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":1,"idx":0}]},
    {"id":"n2","prompt":"a ballerina, baroque","modelId":"illustrious-xl",
     "styleId":"baroque","isRepose":true,"seed":777,
     "images":[{"id":"i2","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","size":1,"idx":0}]}
  ]
}`

func markBlobsPresent(t *testing.T, s *server, shas ...string) {
	t.Helper()
	for _, sha := range shas {
		if _, err := s.db.Exec(
			`INSERT OR IGNORE INTO blobs (sha256, size) VALUES (?, 1)`, sha); err != nil {
			t.Fatal(err)
		}
		if _, err := s.db.Exec(
			`UPDATE images SET blob_present = 1 WHERE sha256 = ?`, sha); err != nil {
			t.Fatal(err)
		}
	}
}

func TestIngestKeepsStyleAndLoraStrength(t *testing.T) {
	s := newTestServer(t)
	ingestJSON(t, s, labManifest)
	markBlobsPresent(t, s, shaA, shaB)

	req := httptest.NewRequest("GET", "/api/images?session=s1", nil)
	rec := httptest.NewRecorder()
	s.handleImagesList(rec, req)
	if rec.Code != 200 {
		t.Fatalf("%d: %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Items []struct {
			ID           string   `json:"id"`
			StyleID      *string  `json:"styleId"`
			LoraStrength *float64 `json:"loraStrength"`
			IsRepose     bool     `json:"isRepose"`
		} `json:"items"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Items) != 2 {
		t.Fatalf("položek %d", len(out.Items))
	}
	for _, it := range out.Items {
		switch it.ID {
		case "i1":
			if it.StyleID == nil || *it.StyleID != "ukiyoe" {
				t.Errorf("styl = %v", it.StyleID)
			}
			if it.LoraStrength == nil || *it.LoraStrength != 0.65 {
				t.Errorf("síla LoRA = %v", it.LoraStrength)
			}
		case "i2":
			if !it.IsRepose {
				t.Error("isRepose se ztratilo")
			}
		}
	}

	// The style facet is derived from what arrived, not from a copy of the
	// app's registry.
	styles := s.ingestedStyles()
	if len(styles) != 2 || styles[0] != "baroque" || styles[1] != "ukiyoe" {
		t.Errorf("styly = %v", styles)
	}
}

func TestFilterByStyle(t *testing.T) {
	s := newTestServer(t)
	ingestJSON(t, s, labManifest)
	markBlobsPresent(t, s, shaA, shaB)

	where, args, bad := galleryFilter(map[string][]string{"style": {"ukiyoe"}})
	if bad != "" {
		t.Fatal(bad)
	}
	var n int
	if err := s.db.QueryRow(
		`SELECT COUNT(*) FROM gallery g WHERE `+where, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("styl ukiyoe = %d obrázků", n)
	}
	// "none" is the only way to ask for the cells that ran without a LoRA —
	// in a LoRA sweep that row is the baseline.
	where, args, _ = galleryFilter(map[string][]string{"lora": {"none"}})
	if err := s.db.QueryRow(
		`SELECT COUNT(*) FROM gallery g WHERE `+where, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("bez LoRA = %d obrázků", n)
	}
}

func TestDeleteSessionRemovesBlobsButKeepsSharedOnes(t *testing.T) {
	s := newTestServer(t)
	ingestJSON(t, s, labManifest)
	markBlobsPresent(t, s, shaA, shaB)
	// A second session referencing the same PNG as the first: content-addressed
	// storage means deleting one session must not pull the file out from under
	// the other.
	ingestJSON(t, s, `{"session":{"id":"s2","title":"jiná","modelId":"pony","updatedAt":"x"},
	  "nodes":[{"id":"m1","prompt":"p","images":[{"id":"j1","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":1,"idx":0}]}]}`)
	markBlobsPresent(t, s, shaA)

	for _, sha := range []string{shaA, shaB} {
		p := s.blobPath(sha)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("png"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	req := httptest.NewRequest("DELETE", "/api/sessions/s1", nil)
	req.SetPathValue("id", "s1")
	rec := httptest.NewRecorder()
	s.handleSessionDelete(rec, req)
	if rec.Code != 200 {
		t.Fatalf("%d: %s", rec.Code, rec.Body.String())
	}
	var out struct{ Images, BlobsFreed int }
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if out.Images != 2 || out.BlobsFreed != 1 {
		t.Errorf("smazáno %+v — 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' drží druhá session, jít má jen 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'", out)
	}
	if _, err := os.Stat(s.blobPath(shaB)); !os.IsNotExist(err) {
		t.Error("osiřelý soubor zůstal na disku")
	}
	if _, err := os.Stat(s.blobPath(shaA)); err != nil {
		t.Error("sdílený soubor zmizel i pro druhou session")
	}
	var nodes int
	if err := s.db.QueryRow(
		`SELECT COUNT(*) FROM nodes WHERE session_id='s1'`).Scan(&nodes); err != nil {
		t.Fatal(err)
	}
	if nodes != 0 {
		t.Errorf("uzly po smazání: %d", nodes)
	}
}

func TestDeleteSessionRefusesWhenImagesAreInADataset(t *testing.T) {
	s := newTestServer(t)
	ingestJSON(t, s, labManifest)
	markBlobsPresent(t, s, shaA, shaB)
	if _, err := s.db.Exec(
		`INSERT INTO datasets (name, base_model, trigger_word) VALUES ('d', 'illustrious-xl', 'x')`); err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.Exec(
		`INSERT INTO dataset_items (dataset_id, image_id) VALUES (1, 'i1')`); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("DELETE", "/api/sessions/s1", nil)
	req.SetPathValue("id", "s1")
	rec := httptest.NewRecorder()
	s.handleSessionDelete(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("čekán 409, dostal jsem %d: %s", rec.Code, rec.Body.String())
	}
	var nodes int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM nodes WHERE session_id='s1'`).Scan(&nodes)
	if nodes != 2 {
		t.Errorf("odmítnuté mazání přesto sáhlo na data: %d uzlů", nodes)
	}
}

func TestMetaCoversEveryModelTheAppCanGenerateWith(t *testing.T) {
	// The list here is the app's registry (lib/models/image_model.dart). A
	// model missing from it lands in the gallery as an unfilterable string —
	// and model is the only structural filter the gallery has.
	for _, id := range []string{
		"flux-schnell", "flux-kontext", "flux-manga", "flux-fill",
		"pony", "atomix-pony-anime", "juggernaut-xl", "juggernaut-xl-lightning",
		"illustrious-xl", "noobai-xl", "wai-illustrious", "animagine-xl", "sd15",
	} {
		if modelByID(id) == nil {
			t.Errorf("registr nezná %s", id)
		}
	}
}

package main

import (
	"database/sql"
	"fmt"
	"math"
	"path/filepath"
	"testing"
)

// table builds an agreement from its four cells and finishes it.
func table(upUp, upDown, downUp, downDown int) *agreement {
	a := &agreement{}
	for range upUp {
		a.add(1, 1)
	}
	for range upDown {
		a.add(1, -1)
	}
	for range downUp {
		a.add(-1, 1)
	}
	for range downDown {
		a.add(-1, -1)
	}
	a.finish()
	return a
}

func near(t *testing.T, name string, got *float64, want float64) {
	t.Helper()
	if got == nil {
		t.Fatalf("%s = nil, want %.4f", name, want)
	}
	if math.Abs(*got-want) > 5e-4 {
		t.Fatalf("%s = %.4f, want %.4f", name, *got, want)
	}
}

func TestKappaPerfectAgreementPasses(t *testing.T) {
	a := table(36, 0, 0, 8)
	near(t, "kappa", a.Kappa, 1)
	near(t, "lower", a.Lower, 1)
	near(t, "downAgree", a.DownAgree, 1)
	if a.Gate != gatePass {
		t.Fatalf("gate = %s, want pass", a.Gate)
	}
}

// The reason the gate exists: at pose_adherence's base rate, a judge that
// says "up" to everything agrees 90% of the time and knows nothing.
func TestKappaAlwaysUpJudgeFails(t *testing.T) {
	a := table(72, 0, 8, 0)
	near(t, "po", a.Po, 0.9)
	near(t, "kappa", a.Kappa, 0)
	near(t, "downAgree", a.DownAgree, 0)
	if a.Gate != gateFail {
		t.Fatalf("gate = %s, want fail", a.Gate)
	}
}

// Hand-computed: po = .90, pe = .75² + .25² = .625, κ = .275/.375 = .7333,
// SE = sqrt(.9·.1/100)/.375 = .08, lower = .7333 − 1.96·.08 = .5765.
func TestKappaKnownTable(t *testing.T) {
	a := table(70, 5, 5, 20)
	near(t, "po", a.Po, 0.9)
	near(t, "pe", a.Pe, 0.625)
	near(t, "kappa", a.Kappa, 0.7333)
	near(t, "lower", a.Lower, 0.5765)
	near(t, "downAgree", a.DownAgree, 0.8)
	if a.Gate != gatePass {
		t.Fatalf("gate = %s, want pass", a.Gate)
	}
}

// κ = .655 clears the point threshold, but on 40 pairs the interval reaches
// down to .372 — not enough evidence yet, so the gate stays shut.
func TestKappaThinSampleFailsOnLowerBound(t *testing.T) {
	a := table(28, 2, 3, 7)
	near(t, "kappa", a.Kappa, 0.6552)
	near(t, "lower", a.Lower, 0.3724)
	if a.Gate != gateFail {
		t.Fatalf("gate = %s, want fail", a.Gate)
	}
}

func TestKappaTooFewPairsIsUncalibrated(t *testing.T) {
	if a := table(31, 0, 0, 8); a.Gate != gateUncalibrated { // n = 39, perfect
		t.Fatalf("n=39: gate = %s, want uncalibrated", a.Gate)
	}
	if a := table(40, 0, 0, 7); a.Gate != gateUncalibrated { // 7 human downs
		t.Fatalf("7 downs: gate = %s, want uncalibrated", a.Gate)
	}
}

func TestKappaUndefinedWhenBothRatersUseOneClass(t *testing.T) {
	a := table(50, 0, 0, 0)
	if a.Kappa != nil {
		t.Fatalf("kappa = %v, want nil (0/0)", *a.Kappa)
	}
	if a.DownAgree != nil {
		t.Fatal("downAgree must be nil without human downs")
	}
	if a.Gate != gateUncalibrated {
		t.Fatalf("gate = %s, want uncalibrated", a.Gate)
	}
	if empty := table(0, 0, 0, 0); empty.Gate != gateUncalibrated || empty.Po != nil {
		t.Fatalf("empty table: %+v", empty)
	}
}

// v4 → v5 runs once on the NAS against a database holding every human label.
// It must leave them alone and add the judge's table beside them.
func TestMigrationV5KeepsV4Rows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "v4.sqlite")
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(ON)")
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range migrations {
		if m.v > 4 {
			break
		}
		if _, err := db.Exec(m.sql); err != nil {
			t.Fatalf("v%d: %v", m.v, err)
		}
	}
	for _, q := range []string{
		`PRAGMA user_version = 4`,
		`INSERT INTO sessions (id, title, model_id) VALUES ('s', 't', 'pony')`,
		`INSERT INTO nodes (id, session_id, prompt, model_id, pose_id) VALUES ('n', 's', 'a cat', 'pony', 'ol1')`,
		fmt.Sprintf(`INSERT INTO blobs (sha256, size) VALUES ('%s', 1)`, fakeSha(1)),
		fmt.Sprintf(`INSERT INTO images (id, node_id, idx, sha256, size, blob_present) VALUES ('i', 'n', 0, '%s', 1, 1)`, fakeSha(1)),
		`INSERT INTO image_criteria (image_id, criterion, score) VALUES ('i', 'pose_adherence', -1)`,
	} {
		if _, err := db.Exec(q); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	if err := migrate(db); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	var version, score int
	db.QueryRow(`PRAGMA user_version`).Scan(&version)
	if version != 5 {
		t.Fatalf("user_version = %d, want 5", version)
	}
	if err := db.QueryRow(`SELECT score FROM image_criteria WHERE image_id = 'i'`).Scan(&score); err != nil || score != -1 {
		t.Fatalf("human label after v5: score=%d err=%v", score, err)
	}
	if _, err := db.Exec(`INSERT INTO criteria_judgments (image_id, criterion, judge, verdict, ref_key)
		VALUES ('i', 'pose_adherence', 'judge@judge-v1', 0, 'pose:ol1')`); err != nil {
		t.Fatalf("criteria_judgments: %v", err)
	}
	if _, err := db.Exec(`INSERT INTO criteria_judgments (image_id, criterion, judge, verdict, ref_key)
		VALUES ('i', 'source_style', 'judge@judge-v1', 2, 'image:x')`); err == nil {
		t.Fatal("verdict 2 accepted; the CHECK must refuse it")
	}
	db.Close()
}

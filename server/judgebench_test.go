package main

import (
	"bytes"
	"context"
	"image/color"
	"slices"
	"strings"
	"testing"
)

// seedCalibration builds 50 human-rated posed images whose pixels decide the
// fake judge's answer (judge_test.go), plus images the bench must ignore.
//
//	human up   40: judge up 37, down 2, unsure 1
//	human down 10: judge up 2,  down 8
//
// Hand-computed on the 49 non-abstained pairs: po = 45/49 = .918,
// pe = (39/49)² + (10/49)² = .675, κ = .749, SE = .120, lower = .513 → pass.
func seedCalibration(f *judgeFixture) {
	rate := func(id string, score int) {
		f.exec(`INSERT INTO image_criteria (image_id, criterion, score) VALUES (?, 'pose_adherence', ?)`, id, score)
	}
	for _, g := range []struct {
		human int
		paint color.Color
		count int
	}{
		{1, white, 37}, {1, black, 2}, {1, grey, 1},
		{-1, white, 2}, {-1, black, 8},
	} {
		for range g.count {
			rate(f.addImage("ol1", "", g.paint, 16, 16), g.human)
		}
	}
	// Rated but not eligible (no pose) — a stale label must not enter κ.
	for range 5 {
		rate(f.addImage("", "", black, 16, 16), 1)
	}
	// Eligible but unrated — nothing to calibrate against.
	for range 3 {
		f.addImage("ol1", "", white, 16, 16)
	}
}

func TestJudgebenchTable(t *testing.T) {
	f := newJudgeFixture(t)
	seedCalibration(f)

	row, err := benchJudge(context.Background(), f.s.judge, "judge", "pose_adherence", 80, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	a := row.agree
	if row.sampled != 50 || row.errors != 0 || row.abstained != 1 {
		t.Fatalf("sampled=%d errors=%d abstained=%d, want 50/0/1", row.sampled, row.errors, row.abstained)
	}
	if a.UpUp != 37 || a.UpDown != 2 || a.DownUp != 2 || a.DownDown != 8 {
		t.Fatalf("confusion = %+v", a)
	}
	near(t, "kappa", a.Kappa, 0.7487)
	near(t, "lower", a.Lower, 0.5127)
	near(t, "downAgree", a.DownAgree, 0.8)
	if a.Gate != gatePass || len(row.misses) != 4 {
		t.Fatalf("gate=%s misses=%d", a.Gate, len(row.misses))
	}

	var out bytes.Buffer
	printJudgebench(&out, "judge@judge-v1", []benchJudgeRow{row}, true)
	table := out.String()
	var line []string
	for _, l := range strings.Split(table, "\n") {
		if strings.HasPrefix(l, "pose_adherence ") {
			line = strings.Fields(l)
			break
		}
	}
	want := []string{"pose_adherence", "50", "0", "1", "49", "0.92", "0.75", "0.51", "0.80", "pass"}
	if !slices.Equal(line, want) {
		t.Fatalf("table row = %v\nwant        %v\n%s", line, want, table)
	}
	if !strings.Contains(table, "left arm raised") {
		t.Fatalf("-v must print the judge's reason on disagreements:\n%s", table)
	}

	// The server's gate board reads the same verdicts the bench stored.
	stats, err := judgeStats(f.s.db, f.s.judge.Version())
	if err != nil {
		t.Fatal(err)
	}
	if st := stats["pose_adherence"]; st.Judged != 50 || st.Abstained != 1 || st.Agreement.Gate != gatePass {
		t.Fatalf("judgeStats = %+v / %+v", st, st.Agreement)
	}
	if got := passedCriteria(stats); !slices.Equal(got, []string{"pose_adherence"}) {
		t.Fatalf("passed = %v", got)
	}
}

func TestJudgebenchSampleIsSeeded(t *testing.T) {
	f := newJudgeFixture(t)
	seedCalibration(f)
	a, _, _ := sampleHumanRated(f.s.db, "pose_adherence", 20, 7)
	b, _, _ := sampleHumanRated(f.s.db, "pose_adherence", 20, 7)
	c, _, _ := sampleHumanRated(f.s.db, "pose_adherence", 20, 8)
	if len(a) != 20 || !slices.Equal(a, b) {
		t.Fatalf("same seed drew different samples:\n%v\n%v", a, b)
	}
	if slices.Equal(a, c) {
		t.Fatal("different seeds drew the same sample")
	}
}

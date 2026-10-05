package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
)

// judgebench calibrates the VL judge against the human labels the gallery
// already holds.
//
//	finetune-gallery judgebench -criterion pose_adherence -n 80
//
// The sample is drawn at random (seeded) from eligible images a human has
// rated — never chosen by the judge, which would measure only the cases it
// finds easy. Each image is judged (through the verdict cache unless
// -no-cache), and the table is the same κ and gate the server applies.

type benchJudgeRow struct {
	criterion string
	sampled   int
	errors    int
	abstained int
	agree     *agreement
	misses    []benchMiss
}

type benchMiss struct {
	imageID       string
	human, judged int
	reason        string
}

func runJudgebench(args []string) int {
	fs := flag.NewFlagSet("judgebench", flag.ContinueOnError)
	criteria := fs.String("criterion", strings.Join(kCriteria, ","), "criteria to calibrate, comma-separated")
	n := fs.Int("n", 80, "human-rated images to sample per criterion")
	seed := fs.Int64("seed", 1, "sampling seed, so two runs judge the same images")
	noCache := fs.Bool("no-cache", false, "bypass and do not write the verdict cache (measures the live model)")
	verbose := fs.Bool("v", false, "print every disagreement with the judge's reason")
	dataDir := fs.String("data", env("FINETUNE_DATA", "/data"), "gallery data dir")
	gateway := fs.String("gateway", env("LLM_GATEWAY_URL", ""), "LLM gateway base URL")
	model := fs.String("model", env("JUDGE_MODEL", "judge"), "LiteLLM alias of the judge")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *gateway == "" {
		fmt.Fprintln(os.Stderr, "judgebench: -gateway (or LLM_GATEWAY_URL) is required")
		return 2
	}

	db, err := openDB(filepath.Join(*dataDir, "db", "finetune.sqlite"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "judgebench: open db: %v\n", err)
		return 1
	}
	defer db.Close()

	s := &server{db: db, dataDir: *dataDir}
	j := NewJudge(db, *gateway, *model, s.blobPath, webPoses())

	var rows []benchJudgeRow
	for _, c := range strings.Split(*criteria, ",") {
		c = strings.TrimSpace(c)
		if _, ok := judgeRubrics[c]; !ok {
			fmt.Fprintf(os.Stderr, "judgebench: unknown criterion %q\n", c)
			return 2
		}
		row, err := benchJudge(context.Background(), j, *model, c, *n, *seed, !*noCache)
		if err != nil {
			fmt.Fprintf(os.Stderr, "judgebench: %s: %v\n", c, err)
			return 1
		}
		rows = append(rows, row)
	}
	printJudgebench(os.Stdout, j.versionFor(*model), rows, *verbose)
	return 0
}

// sampleHumanRated returns up to n eligible images with a human label on the
// criterion, shuffled with the seed. Ordered by id before shuffling so the
// same seed draws the same images on any machine.
func sampleHumanRated(db *sql.DB, criterion string, n int, seed int64) ([]string, map[string]int, error) {
	rows, err := db.Query(`
		SELECT g.id, ic.score FROM gallery g
		JOIN image_criteria ic ON ic.image_id = g.id AND ic.criterion = ?
		WHERE `+criterionEligible[criterion]+`
		ORDER BY g.id`, criterion)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	var ids []string
	human := map[string]int{}
	for rows.Next() {
		var id string
		var score int
		if err := rows.Scan(&id, &score); err != nil {
			return nil, nil, err
		}
		ids = append(ids, id)
		human[id] = score
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	rand.New(rand.NewSource(seed)).Shuffle(len(ids), func(a, b int) { ids[a], ids[b] = ids[b], ids[a] })
	if len(ids) > n {
		ids = ids[:n]
	}
	return ids, human, nil
}

func benchJudge(ctx context.Context, j *Judge, model, criterion string, n int, seed int64, useCache bool) (benchJudgeRow, error) {
	row := benchJudgeRow{criterion: criterion, agree: &agreement{}}
	ids, human, err := sampleHumanRated(j.db, criterion, n, seed)
	if err != nil {
		return row, err
	}
	row.sampled = len(ids)
	for _, id := range ids {
		jd, err := j.JudgeWith(ctx, model, id, criterion, useCache)
		if err != nil {
			row.errors++
			fmt.Fprintf(os.Stderr, "  %s %s: %v\n", criterion, id, err)
			continue
		}
		if jd.Verdict == 0 {
			row.abstained++
			continue
		}
		row.agree.add(human[id], jd.Verdict)
		if jd.Verdict != human[id] {
			row.misses = append(row.misses, benchMiss{id, human[id], jd.Verdict, jd.Reason})
		}
	}
	row.agree.finish()
	return row, nil
}

func printJudgebench(w io.Writer, version string, rows []benchJudgeRow, verbose bool) {
	fmt.Fprintf(w, "judge %s · gate: n ≥ %d, human down ≥ %d, κ ≥ %.2f, 95%% low ≥ %.2f\n\n",
		version, gateMinPairs, gateMinHumanDown, gateMinKappa, gateMinKappaLower)
	f := func(p *float64) string {
		if p == nil {
			return "—"
		}
		return fmt.Sprintf("%.2f", *p)
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "criterion\tsampled\terr\tabst\tpairs\tpo\tκ\t95% low\tdown agree\tgate")
	for _, r := range rows {
		a := r.agree
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\n",
			r.criterion, r.sampled, r.errors, r.abstained, a.N,
			f(a.Po), f(a.Kappa), f(a.Lower), f(a.DownAgree), a.Gate)
	}
	tw.Flush()

	for _, r := range rows {
		a := r.agree
		fmt.Fprintf(w, "\n%s            judge up  judge down\n", r.criterion)
		fmt.Fprintf(w, "  human up    %8d  %10d\n", a.UpUp, a.UpDown)
		fmt.Fprintf(w, "  human down  %8d  %10d\n", a.DownUp, a.DownDown)
		if verbose {
			for _, m := range r.misses {
				fmt.Fprintf(w, "  %s  human %+d  judge %+d  %s\n", m.imageID, m.human, m.judged, m.reason)
			}
		}
	}
}

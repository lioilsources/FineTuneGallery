package main

import (
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
)

// taxobench is the taxonomy's κ-gate: every evaluator against the human
// labels, per node, with the same thresholds and the same interval as the VL
// judge's gate (kappa.go) — a taxonomy evaluator is a second rater like the
// judge and earns trust the same way.
//
//	finetune-gallery taxobench -sample 300 -borderline 60 -seed 1   # labelling sample
//	finetune-gallery taxobench [-stratum random|borderline|all] [-json]
//
// The sample is drawn before anyone sees an answer: proportional to the
// gallery's model × style strata, plus a borderline stratum (rating unsure or
// within ±0.05 of a band threshold) because that is where the gate decides.
// The borderline stratum is over-sampled on purpose, so κ is reported for
// "random" separately — over-sampled hard cases would understate it.
//
// Nodes with one value per image get Cohen's κ over all classes. Multi-label
// nodes get a κ per value (present / absent) — the same 2×2 statistic as the
// judge's. The minority guard of kappa.go becomes "the second most common
// human class has ≥ gateMinHumanDown labels".

type taxoAgreement struct {
	Node      string   `json:"node"`
	Value     string   `json:"value,omitempty"` // multi-label nodes: κ per value
	Evaluator string   `json:"evaluator"`
	N         int      `json:"n"`         // images both labelled
	Abstained int      `json:"abstained"` // human labelled, evaluator silent
	Minority  int      `json:"minority"`  // second most common human class
	Po        *float64 `json:"po"`
	Pe        *float64 `json:"pe"`
	Kappa     *float64 `json:"kappa"`
	Lower     *float64 `json:"lower"`
	Upper     *float64 `json:"upper"`
	Gate      string   `json:"gate"`
}

// multiKappa is Cohen's κ for any number of classes, its asymptotic 95%
// interval and the gate decision. On two classes it equals agreement.finish.
func multiKappa(pairs [][2]string) taxoAgreement {
	a := taxoAgreement{Gate: gateUncalibrated, N: len(pairs)}
	if a.N == 0 {
		return a
	}
	n := float64(a.N)
	hc, rc := map[string]float64{}, map[string]float64{}
	agree := 0.0
	for _, p := range pairs {
		hc[p[0]]++
		rc[p[1]]++
		if p[0] == p[1] {
			agree++
		}
	}
	po := agree / n
	pe := 0.0
	for c, h := range hc {
		pe += (h / n) * (rc[c] / n)
	}
	a.Po, a.Pe = &po, &pe
	counts := []int{}
	for _, h := range hc {
		counts = append(counts, int(h))
	}
	sort.Sort(sort.Reverse(sort.IntSlice(counts)))
	if len(counts) > 1 {
		a.Minority = counts[1]
	}
	if pe >= 1 {
		return a
	}
	k := (po - pe) / (1 - pe)
	se := math.Sqrt(po*(1-po)/n) / (1 - pe)
	const z = 1.959963984540054
	lo, hi := math.Max(-1, k-z*se), math.Min(1, k+z*se)
	a.Kappa, a.Lower, a.Upper = &k, &lo, &hi
	switch {
	case a.N < gateMinPairs || a.Minority < gateMinHumanDown:
		a.Gate = gateUncalibrated
	case k < gateMinKappa || lo < gateMinKappaLower:
		a.Gate = gateFail
	default:
		a.Gate = gatePass
	}
	return a
}

// taxoKappa computes the table. stratum: all | random | borderline.
func taxoKappa(db *sql.DB, stratum string) ([]taxoAgreement, error) {
	scope := ""
	switch stratum {
	case "", "all":
	case "random":
		scope = ` AND image_id IN (SELECT image_id FROM taxonomy_label_queue WHERE stratum LIKE 'random:%')`
	case "borderline":
		scope = ` AND image_id IN (SELECT image_id FROM taxonomy_label_queue WHERE stratum = 'borderline')`
	default:
		return nil, fmt.Errorf("stratum must be all|random|borderline")
	}
	// answers[evaluator][node][image] = set of values
	answers := map[string]map[string]map[string][]string{}
	rows, err := db.Query(`SELECT evaluator, node, image_id, value FROM image_taxonomy
		WHERE taxonomy_version = ? AND image_id IN (SELECT id FROM gallery)`+scope, taxonomy.Version)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var ev, node, img, val string
		if err := rows.Scan(&ev, &node, &img, &val); err != nil {
			rows.Close()
			return nil, err
		}
		if answers[ev] == nil {
			answers[ev] = map[string]map[string][]string{}
		}
		if answers[ev][node] == nil {
			answers[ev][node] = map[string][]string{}
		}
		if val != "" {
			answers[ev][node][img] = append(answers[ev][node][img], val)
		} else if _, ok := answers[ev][node][img]; !ok {
			answers[ev][node][img] = []string{}
		}
	}
	rows.Close()
	// The ensemble's band is a rater of the rating pseudo-node.
	brows, err := db.Query(`SELECT image_id, band FROM image_temperature WHERE image_id IN (SELECT id FROM gallery)` + scope)
	if err == nil {
		answers["temperature"] = map[string]map[string][]string{ratingNode: {}}
		for brows.Next() {
			var img, band string
			if brows.Scan(&img, &band) == nil {
				answers["temperature"][ratingNode][img] = []string{band}
			}
		}
		brows.Close()
	}

	human := answers["human"]
	evals := []string{"wd14", "siglip", "vlm", "head", "consensus", "temperature"}
	nodes := append([]taxoNode{}, taxonomy.Nodes...)
	nodes = append(nodes, taxoNode{ID: ratingNode, Kind: "single"})
	var out []taxoAgreement
	for _, n := range nodes {
		hn := human[n.ID]
		if len(hn) == 0 {
			continue
		}
		for _, ev := range evals {
			en := answers[ev][n.ID]
			if en == nil {
				continue
			}
			if n.Kind == "multi" {
				for _, v := range n.Values {
					var pairs [][2]string
					abst := 0
					for img, hv := range hn {
						rv, ok := en[img]
						if !ok {
							abst++
							continue
						}
						pairs = append(pairs, [2]string{yesNo(contains(hv, v.ID)), yesNo(contains(rv, v.ID))})
					}
					a := multiKappa(pairs)
					a.Node, a.Value, a.Evaluator, a.Abstained = n.ID, v.ID, ev, abst
					out = append(out, a)
				}
				continue
			}
			var pairs [][2]string
			abst := 0
			for img, hv := range hn {
				rv, ok := en[img]
				if !ok || len(rv) == 0 || len(hv) == 0 {
					abst++
					continue
				}
				pairs = append(pairs, [2]string{hv[0], rv[0]})
			}
			a := multiKappa(pairs)
			a.Node, a.Evaluator, a.Abstained = n.ID, ev, abst
			out = append(out, a)
		}
	}
	return out, nil
}

func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// GET /api/taxonomy/kappa?stratum=random
func (s *server) handleTaxoKappa(w http.ResponseWriter, r *http.Request) {
	if !taxoEnabled {
		writeErr(w, http.StatusNotFound, errTaxoDisabled.Error())
		return
	}
	rows, err := taxoKappa(s.db, r.URL.Query().Get("stratum"))
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	var labelled, queued int
	s.db.QueryRow(`SELECT COUNT(DISTINCT image_id) FROM image_taxonomy WHERE evaluator = 'human'`).Scan(&labelled)
	s.db.QueryRow(`SELECT COUNT(*) FROM taxonomy_label_queue`).Scan(&queued)
	writeJSON(w, http.StatusOK, map[string]any{"rows": rows, "humanLabelled": labelled, "labelQueue": queued})
}

// sampleLabelQueue draws the labelling sample (see the file comment).
func sampleLabelQueue(db *sql.DB, n, borderline int, seed int64) (map[string]int, error) {
	th, _ := activeBands(db)
	type cand struct {
		id, stratum string
		border      bool
	}
	rows, err := db.Query(`SELECT g.id, COALESCE(g.model_id, '?'), COALESCE(g.style_id, '-'),
		it.temperature, COALESCE(it.unsure, 0)
		FROM gallery g LEFT JOIN image_temperature it ON it.image_id = g.id
		WHERE g.id NOT IN (SELECT image_id FROM taxonomy_label_queue)
		ORDER BY g.id`)
	if err != nil {
		return nil, err
	}
	var all []cand
	for rows.Next() {
		var id, model, style string
		var t *float64
		var unsure bool
		if err := rows.Scan(&id, &model, &style, &t, &unsure); err != nil {
			rows.Close()
			return nil, err
		}
		b := unsure
		if t != nil {
			for _, x := range []float64{th.SafeMax, th.SuggestiveMax, th.NsfwMax} {
				if math.Abs(*t-x) <= 0.05 {
					b = true
				}
			}
		}
		all = append(all, cand{id, "random:" + model + "/" + style, b})
	}
	rows.Close()
	rnd := rand.New(rand.NewSource(seed))
	rnd.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })

	picked := map[string]string{}
	for _, c := range all {
		if len(picked) >= borderline {
			break
		}
		if c.border {
			picked[c.id] = "borderline"
		}
	}
	// Proportional allocation over strata (largest remainder) for the rest.
	rest := n - len(picked)
	byStratum := map[string][]string{}
	total := 0
	for _, c := range all {
		if _, ok := picked[c.id]; !ok {
			byStratum[c.stratum] = append(byStratum[c.stratum], c.id)
			total++
		}
	}
	if rest > total {
		rest = total
	}
	type alloc struct {
		stratum string
		n       int
		rem     float64
	}
	var al []alloc
	given := 0
	for st, ids := range byStratum {
		q := float64(rest) * float64(len(ids)) / float64(max(1, total))
		al = append(al, alloc{st, int(q), q - math.Floor(q)})
		given += int(q)
	}
	sort.Slice(al, func(i, j int) bool {
		if al[i].rem != al[j].rem {
			return al[i].rem > al[j].rem
		}
		return al[i].stratum < al[j].stratum
	})
	for i := 0; given < rest && i < len(al); i++ {
		al[i].n++
		given++
	}
	for _, a := range al {
		for _, id := range byStratum[a.stratum][:min(a.n, len(byStratum[a.stratum]))] {
			picked[id] = a.stratum
		}
	}
	counts := map[string]int{}
	tx, err := db.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	for id, st := range picked {
		if _, err := tx.Exec(`INSERT INTO taxonomy_label_queue (image_id, stratum, seed) VALUES (?, ?, ?)`, id, st, seed); err != nil {
			return nil, err
		}
		if st != "borderline" {
			st = "random"
		}
		counts[st]++
	}
	return counts, tx.Commit()
}

func runTaxobench(args []string) int {
	fs := flag.NewFlagSet("taxobench", flag.ContinueOnError)
	sample := fs.Int("sample", 0, "draw a labelling sample of this size into the label queue")
	borderline := fs.Int("borderline", 60, "of which from the borderline stratum")
	seed := fs.Int64("seed", 1, "sampling seed")
	stratum := fs.String("stratum", "random", "κ over: random | borderline | all")
	asJSON := fs.Bool("json", false, "print the κ table as JSON")
	dataDir := fs.String("data", env("FINETUNE_DATA", "/data"), "gallery data dir")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	db, err := openDB(filepath.Join(*dataDir, "db", "finetune.sqlite"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "taxobench: open db: %v\n", err)
		return 1
	}
	defer db.Close()

	if *sample > 0 {
		counts, err := sampleLabelQueue(db, *sample, *borderline, *seed)
		if err != nil {
			fmt.Fprintf(os.Stderr, "taxobench: %v\n", err)
			return 1
		}
		fmt.Printf("label queue += random %d, borderline %d (seed %d) — label them in the gallery: filter \"label queue\"\n",
			counts["random"], counts["borderline"], *seed)
		return 0
	}
	rows, err := taxoKappa(db, *stratum)
	if err != nil {
		fmt.Fprintf(os.Stderr, "taxobench: %v\n", err)
		return 1
	}
	if *asJSON {
		json.NewEncoder(os.Stdout).Encode(rows)
		return 0
	}
	if len(rows) == 0 {
		fmt.Println("no human taxonomy labels yet — draw a sample with -sample 300 and label it")
		return 0
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "node\tvalue\tevaluator\tn\tabst\tminority\tκ\t95% CI\tgate")
	f := func(p *float64) string {
		if p == nil {
			return "—"
		}
		return fmt.Sprintf("%.2f", *p)
	}
	for _, r := range rows {
		ci := "—"
		if r.Lower != nil {
			ci = f(r.Lower) + "…" + f(r.Upper)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\n", r.Node, r.Value, r.Evaluator, r.N,
			r.Abstained, r.Minority, f(r.Kappa), ci, strings.ToUpper(r.Gate))
	}
	tw.Flush()
	return 0
}

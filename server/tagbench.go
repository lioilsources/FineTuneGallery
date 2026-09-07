package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"
)

// tagbench picks the translator model with the gallery's own ground truth.
//
// Every rated image already carries two things: the prose the user wrote
// (nodes.prompt) and what the WD14 tagger saw in the picture that prose
// produced (captions.tags_json). A translator that turns the prose into the
// tags the picture actually contains is doing its job; one that produces
// plausible-sounding tags the picture lacks is not. So the benchmark asks
// each candidate alias to translate the same prompts and scores the answer
// against the tagger — no human labelling, no new downloads, an afternoon.
//
//	finetune-gallery tagbench -models prompt-tags,translate -n 100
//
// Recall is the headline (did the model find what was there); precision and
// the vocabulary hit rate say how much it invented; latency says whether it
// can sit in front of every generation.

type benchPair struct {
	prompt string
	truth  map[string]struct{} // WD14 general tags at or above -min-conf
}

type benchResult struct {
	model     string
	n         int
	errors    int
	recall    float64
	precision float64
	vocabHit  float64
	avgTags   float64
	p50       time.Duration
	p90       time.Duration
}

func runTagbench(args []string) int {
	fs := flag.NewFlagSet("tagbench", flag.ContinueOnError)
	models := fs.String("models", env("TRANSLATE_MODEL", "prompt-tags"), "LiteLLM aliases to compare, comma-separated")
	n := fs.Int("n", 100, "number of (prompt, image) pairs")
	minConf := fs.Float64("min-conf", 0.5, "WD14 confidence a tag needs to count as ground truth")
	minWords := fs.Int("min-words", 4, "skip prompts shorter than this — a two-word prompt has nothing to translate")
	seed := fs.Int64("seed", 1, "sampling seed, so two runs compare the same pairs")
	noCache := fs.Bool("no-cache", false, "bypass the translations table (measures the live model, not yesterday's answer)")
	verbose := fs.Bool("v", false, "print every pair with its tags")
	dataDir := fs.String("data", env("FINETUNE_DATA", "/data"), "gallery data dir")
	gateway := fs.String("gateway", env("LLM_GATEWAY_URL", ""), "LLM gateway base URL")
	tagger := fs.String("tagger", env("TAGGER_URL", "http://wd14:8000"), "tagger base URL, for the vocabulary")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *gateway == "" {
		fmt.Fprintln(os.Stderr, "tagbench: -gateway (or LLM_GATEWAY_URL) is required")
		return 2
	}

	db, err := openDB(filepath.Join(*dataDir, "db", "finetune.sqlite"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "tagbench: open db: %v\n", err)
		return 1
	}
	defer db.Close()

	tr := NewTranslator(db, *gateway, "", *tagger)
	if err := tr.LoadVocab(); err != nil {
		fmt.Fprintf(os.Stderr, "tagbench: vocabulary: %v\n", err)
		return 1
	}

	pairs, err := sampleBenchPairs(db, *n, *minConf, *minWords, *seed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tagbench: sample: %v\n", err)
		return 1
	}
	if len(pairs) == 0 {
		fmt.Fprintln(os.Stderr, "tagbench: no (prompt, caption) pairs match — rate and caption some images first")
		return 1
	}
	fmt.Printf("%d pairs · truth = WD14 general tags ≥ %.2f · vocabulary %d tags\n\n",
		len(pairs), *minConf, tr.VocabSize())

	var results []benchResult
	for _, model := range strings.Split(*models, ",") {
		model = strings.TrimSpace(model)
		if model == "" {
			continue
		}
		res := benchModel(context.Background(), tr, model, pairs, !*noCache, *verbose)
		results = append(results, res)
	}
	printBench(results)
	return 0
}

// sampleBenchPairs takes distinct prompts that have a WD14 caption, shuffled
// with the seed, and keeps n. One image per prompt: the same prompt rendered
// four times would otherwise count four times.
func sampleBenchPairs(db *sql.DB, n int, minConf float64, minWords int, seed int64) ([]benchPair, error) {
	rows, err := db.Query(`
		SELECT n.prompt, c.tags_json
		FROM nodes n
		JOIN images i   ON i.node_id = n.id AND i.blob_present = 1
		JOIN captions c ON c.image_id = i.id AND c.kind = 'auto' AND c.tags_json IS NOT NULL
		WHERE TRIM(n.prompt) != ''
		GROUP BY n.prompt
		ORDER BY n.prompt`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var all []benchPair
	for rows.Next() {
		var prompt, tagsJSON string
		if err := rows.Scan(&prompt, &tagsJSON); err != nil {
			return nil, err
		}
		if len(strings.Fields(prompt)) < minWords {
			continue
		}
		truth := truthTags(tagsJSON, minConf)
		if len(truth) == 0 {
			continue
		}
		all = append(all, benchPair{prompt: prompt, truth: truth})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	rand.New(rand.NewSource(seed)).Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	if len(all) > n {
		all = all[:n]
	}
	return all, nil
}

// truthTags reads the tagger's stored response — {"general": {tag: conf}, …}
// — and keeps the general tags at or above minConf, normalised like the
// translator's output so the two compare in one spelling.
func truthTags(tagsJSON string, minConf float64) map[string]struct{} {
	var stored struct {
		General map[string]float64 `json:"general"`
	}
	if json.Unmarshal([]byte(tagsJSON), &stored) != nil {
		return nil
	}
	out := map[string]struct{}{}
	for tag, conf := range stored.General {
		if conf >= minConf {
			out[normalizeTag(tag)] = struct{}{}
		}
	}
	return out
}

func benchModel(ctx context.Context, tr *Translator, model string, pairs []benchPair, useCache, verbose bool) benchResult {
	res := benchResult{model: model}
	var recallSum, precSum, hitSum, tagSum float64
	var lat []time.Duration
	for _, p := range pairs {
		start := time.Now()
		out, err := tr.TranslateWith(ctx, model, p.prompt, useCache)
		took := time.Since(start)
		if err != nil {
			res.errors++
			if verbose {
				fmt.Printf("  ✗ %s\n    %v\n", p.prompt, err)
			}
			continue
		}
		res.n++
		if !out.Cached {
			lat = append(lat, took)
		}
		hit := 0
		for _, t := range out.Tags {
			if _, ok := p.truth[t]; ok {
				hit++
			}
		}
		recall := float64(hit) / float64(len(p.truth))
		prec := 0.0
		if len(out.Tags) > 0 {
			prec = float64(hit) / float64(len(out.Tags))
		}
		total := len(out.Tags) + len(out.Dropped)
		vhit := 1.0
		if total > 0 {
			vhit = float64(len(out.Tags)) / float64(total)
		}
		recallSum += recall
		precSum += prec
		hitSum += vhit
		tagSum += float64(len(out.Tags))
		if verbose {
			fmt.Printf("  %s\n    → %s\n    recall %.2f  prec %.2f  dropped %s\n",
				p.prompt, out.Text, recall, prec, strings.Join(out.Dropped, ", "))
		}
	}
	if res.n > 0 {
		res.recall = recallSum / float64(res.n)
		res.precision = precSum / float64(res.n)
		res.vocabHit = hitSum / float64(res.n)
		res.avgTags = tagSum / float64(res.n)
	}
	if len(lat) > 0 {
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		res.p50 = lat[len(lat)/2]
		res.p90 = lat[(len(lat)*9)/10]
	}
	return res
}

func printBench(results []benchResult) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "model\tn\terr\trecall\tprecision\tvocab hit\ttags\tp50\tp90")
	for _, r := range results {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%.1f%%\t%.1f%%\t%.1f%%\t%.1f\t%s\t%s\n",
			r.model, r.n, r.errors, r.recall*100, r.precision*100, r.vocabHit*100, r.avgTags,
			fmtDur(r.p50), fmtDur(r.p90))
	}
	tw.Flush()
	fmt.Println("\nrecall: share of the picture's WD14 tags the translation found (headline)")
	fmt.Println("precision: share of emitted tags the picture actually had")
	fmt.Println("vocab hit: share of what the model said that exists in the tagger vocabulary")
	fmt.Println("p50/p90: gateway latency on uncached calls (— when every answer was cached)")
}

func fmtDur(d time.Duration) string {
	if d == 0 {
		return "—"
	}
	if d < time.Millisecond {
		return "<1ms" // a local stub answers in microseconds; a real gateway never will
	}
	return d.Round(time.Millisecond).String()
}

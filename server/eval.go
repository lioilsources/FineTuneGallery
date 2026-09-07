package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
)

// The eval harness. The gallery's relational criteria (pose_adherence,
// source_identity, source_style) judge an output against the thing that
// conditioned it, which makes them the only ratings here that can answer
// "which checkpoint actually holds a pose". This endpoint is that answer:
// slice the rated corpus by any conditioning dimension and compare.
//
// Two things separate it from the count table it replaces:
//
//   - Ranking is by the Wilson lower bound, not the raw ratio. 3/3 outranks
//     40/45 on ratio and must not outrank it on a leaderboard.
//   - Every cell reports how many images the criterion could apply to at all
//     (a pose criterion means nothing on an image with no pose), so a thin
//     number is visibly thin rather than silently wrong.

// evalGroupDef is one dimension the corpus can be sliced by. Expr must yield
// a single TEXT key from the gallery view aliased g; ” means "not set" and
// is surfaced as its own row rather than dropped.
type evalGroupDef struct {
	Expr string
	None string // label for the empty key
}

var evalGroupDefs = map[string]evalGroupDef{
	"model":   {"COALESCE(g.model_id, '')", "(unknown model)"},
	"style":   {"COALESCE(g.style_id, '')", "(no style)"},
	"pose":    {"COALESCE(g.pose_id, '')", "(no pose)"},
	"session": {"g.session_id", "(no session)"},
	// lora_strength is part of the identity of a LoRA run — schema v2 added
	// the column precisely because lora_name alone cannot tell a 0.4 run from
	// a 1.4 one, and grouping by name alone would throw that away again.
	"lora": {`CASE WHEN COALESCE(g.lora_name, '') = '' THEN ''
		ELSE g.lora_name || CASE WHEN g.lora_strength IS NULL THEN ''
		                         ELSE ' @ ' || printf('%.2f', g.lora_strength) END END`, "(no LoRA)"},
}

var evalGroupOrder = []string{"model", "style", "lora", "pose", "session"}

// criterionEligible restricts each criterion to the images it can be judged
// on. Without this a coverage figure is a fraction of the whole corpus and
// therefore permanently, uselessly small.
var criterionEligible = map[string]string{
	"pose_adherence":  "COALESCE(g.pose_id, '') != ''",
	"source_identity": "COALESCE(g.source_image_id, '') != ''",
	"source_style":    "COALESCE(g.source_image_id, '') != ''",
}

// evalCell is one measured proportion: successes, trials, and how many trials
// were possible.
type evalCell struct {
	Up       int      `json:"up"`
	Down     int      `json:"down"`
	Rated    int      `json:"rated"`
	Eligible int      `json:"eligible"`
	Rate     *float64 `json:"rate"`  // up/rated — nil when nothing is rated
	Lower    *float64 `json:"lower"` // Wilson 95% lower bound
	Upper    *float64 `json:"upper"`
}

func (c *evalCell) finish() {
	c.Rated = c.Up + c.Down
	if c.Rated == 0 {
		return
	}
	rate, lo, hi := wilson(c.Up, c.Down)
	c.Rate, c.Lower, c.Upper = &rate, &lo, &hi
}

type evalRow struct {
	Key      string               `json:"key"`
	Label    string               `json:"label"`
	N        int                  `json:"n"`
	Liked    int                  `json:"liked"`
	Disliked int                  `json:"disliked"`
	Unrated  int                  `json:"unrated"`
	Like     evalCell             `json:"like"`
	Criteria map[string]*evalCell `json:"criteria"`
}

// evalLeader is the headline: who wins a criterion once sample size is taken
// seriously.
type evalLeader struct {
	Key    string  `json:"key"`
	Label  string  `json:"label"`
	Rate   float64 `json:"rate"`
	Lower  float64 `json:"lower"`
	Rated  int     `json:"rated"`
	Behind *string `json:"behind"` // runner-up label, nil when there is none
}

// wilson returns the observed rate and the 95% Wilson score interval for up
// successes out of up+down trials. Callers must not pass 0 trials.
func wilson(up, down int) (rate, lower, upper float64) {
	n := float64(up + down)
	const z = 1.959963984540054 // 97.5th percentile of the standard normal
	p := float64(up) / n
	denom := 1 + z*z/n
	center := p + z*z/(2*n)
	margin := z * math.Sqrt(p*(1-p)/n+z*z/(4*n*n))
	lower = (center - margin) / denom
	upper = (center + margin) / denom
	return p, math.Max(0, lower), math.Min(1, upper)
}

// GET /api/eval?group=model&min=10&format=json|csv plus every gallery filter.
func (s *server) handleEval(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()

	groupName := q.Get("group")
	if groupName == "" {
		groupName = "model"
	}
	def, ok := evalGroupDefs[groupName]
	if !ok {
		writeErr(w, http.StatusBadRequest, "group must be one of "+strings.Join(evalGroupOrder, ", "))
		return
	}

	minRated := 10
	if v := q.Get("min"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			writeErr(w, http.StatusBadRequest, "min must be a non-negative integer")
			return
		}
		minRated = n
	}

	where, args, bad := galleryFilter(q)
	if bad != "" {
		writeErr(w, http.StatusBadRequest, bad)
		return
	}

	rows, err := s.evalRows(def, where, args)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	s.labelRows(groupName, def, rows)

	// Widest slice first: the row you are most likely to trust leads the table.
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].N != rows[j].N {
			return rows[i].N > rows[j].N
		}
		return rows[i].Label < rows[j].Label
	})

	total := evalRow{Key: "*", Label: "all", Criteria: map[string]*evalCell{}}
	for _, c := range kCriteria {
		total.Criteria[c] = &evalCell{}
	}
	for _, row := range rows {
		total.N += row.N
		total.Liked += row.Liked
		total.Disliked += row.Disliked
		total.Unrated += row.Unrated
		total.Like.Up += row.Like.Up
		total.Like.Down += row.Like.Down
		total.Like.Eligible += row.Like.Eligible
		for _, c := range kCriteria {
			t, got := total.Criteria[c], row.Criteria[c]
			t.Up += got.Up
			t.Down += got.Down
			t.Eligible += got.Eligible
		}
	}
	total.Like.finish()
	for _, c := range kCriteria {
		total.Criteria[c].finish()
	}

	if q.Get("format") == "csv" {
		writeEvalCSV(w, groupName, rows)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"group":    groupName,
		"groups":   evalGroupOrder,
		"criteria": kCriteria,
		"minRated": minRated,
		"rows":     rows,
		"total":    total,
		"leaders":  evalLeaders(rows, minRated),
	})
}

// evalRows runs the two aggregate queries — per-group totals plus criterion
// eligibility, then per-group criterion tallies — and merges them. Splitting
// them is deliberate: joining image_criteria into the totals query would
// multiply COUNT(*) by the number of criteria rated on each image.
func (s *server) evalRows(def evalGroupDef, where string, args []any) ([]*evalRow, error) {
	eligibleCols := make([]string, 0, len(kCriteria))
	for _, c := range kCriteria {
		cond, ok := criterionEligible[c]
		if !ok {
			cond = "1=1" // a criterion with no conditioning input applies to everything
		}
		eligibleCols = append(eligibleCols, fmt.Sprintf("SUM(CASE WHEN %s THEN 1 ELSE 0 END)", cond))
	}

	byKey := map[string]*evalRow{}
	var out []*evalRow

	rows, err := s.db.Query(`
		SELECT `+def.Expr+` AS k, COUNT(*),
		       SUM(CASE WHEN g.score = 1 THEN 1 ELSE 0 END),
		       SUM(CASE WHEN g.score = -1 THEN 1 ELSE 0 END),
		       SUM(CASE WHEN g.score = 0 AND g.critique = '' THEN 1 ELSE 0 END),
		       `+strings.Join(eligibleCols, ",\n		       ")+`
		FROM gallery g
		WHERE `+where+`
		GROUP BY k`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		row := &evalRow{Criteria: map[string]*evalCell{}}
		eligible := make([]int, len(kCriteria))
		scan := []any{&row.Key, &row.N, &row.Liked, &row.Disliked, &row.Unrated}
		for i := range eligible {
			scan = append(scan, &eligible[i])
		}
		if err := rows.Scan(scan...); err != nil {
			return nil, err
		}
		row.Like = evalCell{Up: row.Liked, Down: row.Disliked, Eligible: row.N}
		for i, c := range kCriteria {
			row.Criteria[c] = &evalCell{Eligible: eligible[i]}
		}
		byKey[row.Key] = row
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	crit, err := s.db.Query(`
		SELECT `+def.Expr+` AS k, ic.criterion,
		       SUM(CASE WHEN ic.score = 1 THEN 1 ELSE 0 END),
		       SUM(CASE WHEN ic.score = -1 THEN 1 ELSE 0 END)
		FROM gallery g
		JOIN image_criteria ic ON ic.image_id = g.id
		WHERE `+where+`
		GROUP BY k, ic.criterion`, args...)
	if err != nil {
		return nil, err
	}
	defer crit.Close()
	for crit.Next() {
		var key, criterion string
		var up, down int
		if err := crit.Scan(&key, &criterion, &up, &down); err != nil {
			return nil, err
		}
		row, ok := byKey[key]
		if !ok {
			continue
		}
		cell, ok := row.Criteria[criterion]
		if !ok {
			// A criterion retired from kCriteria but still on old rows: keep
			// the data visible rather than silently dropping it.
			cell = &evalCell{Eligible: row.N}
			row.Criteria[criterion] = cell
		}
		cell.Up, cell.Down = up, down
	}
	if err := crit.Err(); err != nil {
		return nil, err
	}

	for _, row := range out {
		row.Like.finish()
		for _, cell := range row.Criteria {
			cell.finish()
		}
	}
	return out, nil
}

// labelRows resolves group keys to human labels. Models come from the
// registry; sessions from their titles; everything else is already readable.
func (s *server) labelRows(groupName string, def evalGroupDef, rows []*evalRow) {
	titles := map[string]string{}
	if groupName == "session" {
		if r, err := s.db.Query(`SELECT id, title FROM sessions`); err == nil {
			defer r.Close()
			for r.Next() {
				var id, title string
				if r.Scan(&id, &title) == nil {
					titles[id] = title
				}
			}
		}
	}
	for _, row := range rows {
		switch {
		case row.Key == "":
			row.Label = def.None
		case groupName == "model":
			if m := modelByID(row.Key); m != nil {
				row.Label = m.Label
			} else {
				row.Label = row.Key
			}
		case groupName == "session":
			if t, ok := titles[row.Key]; ok && t != "" {
				row.Label = t
			} else {
				row.Label = row.Key
			}
		default:
			row.Label = row.Key
		}
	}
}

// evalLeaders picks, per criterion, the group with the highest Wilson lower
// bound among those meeting minRated. Groups below the threshold are excluded
// rather than ranked low: they are not evidence either way.
func evalLeaders(rows []*evalRow, minRated int) map[string]*evalLeader {
	out := map[string]*evalLeader{}
	rank := func(name string, cell func(*evalRow) *evalCell) {
		type cand struct {
			row  *evalRow
			cell *evalCell
		}
		var cands []cand
		for _, row := range rows {
			c := cell(row)
			if c == nil || c.Rated < minRated || c.Rated == 0 || c.Lower == nil {
				continue
			}
			cands = append(cands, cand{row, c})
		}
		if len(cands) == 0 {
			return
		}
		sort.SliceStable(cands, func(i, j int) bool { return *cands[i].cell.Lower > *cands[j].cell.Lower })
		win := cands[0]
		l := &evalLeader{
			Key: win.row.Key, Label: win.row.Label,
			Rate: *win.cell.Rate, Lower: *win.cell.Lower, Rated: win.cell.Rated,
		}
		if len(cands) > 1 {
			runnerUp := cands[1].row.Label
			l.Behind = &runnerUp
		}
		out[name] = l
	}
	rank("like", func(r *evalRow) *evalCell { return &r.Like })
	for _, c := range kCriteria {
		name := c
		rank(name, func(r *evalRow) *evalCell { return r.Criteria[name] })
	}
	return out
}

// writeEvalCSV emits the matrix as a spreadsheet — the format a comparison
// actually gets argued over in.
func writeEvalCSV(w http.ResponseWriter, groupName string, rows []*evalRow) {
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition",
		fmt.Sprintf(`attachment; filename="eval_by_%s.csv"`, groupName))

	cw := csv.NewWriter(w)
	defer cw.Flush()

	header := []string{groupName, "label", "images", "liked", "disliked", "unrated", "like_rate", "like_lower"}
	for _, c := range kCriteria {
		header = append(header, c+"_up", c+"_down", c+"_eligible", c+"_rate", c+"_lower")
	}
	cw.Write(header)

	num := func(p *float64) string {
		if p == nil {
			return ""
		}
		return strconv.FormatFloat(*p, 'f', 4, 64)
	}
	for _, row := range rows {
		rec := []string{
			row.Key, row.Label,
			strconv.Itoa(row.N), strconv.Itoa(row.Liked),
			strconv.Itoa(row.Disliked), strconv.Itoa(row.Unrated),
			num(row.Like.Rate), num(row.Like.Lower),
		}
		for _, c := range kCriteria {
			cell := row.Criteria[c]
			if cell == nil {
				cell = &evalCell{}
			}
			rec = append(rec,
				strconv.Itoa(cell.Up), strconv.Itoa(cell.Down),
				strconv.Itoa(cell.Eligible), num(cell.Rate), num(cell.Lower))
		}
		cw.Write(rec)
	}
}

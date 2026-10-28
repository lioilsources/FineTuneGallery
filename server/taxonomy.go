package main

import (
	"database/sql"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// The content taxonomy (UC25): what an image shows (subject, hair, medium …)
// and how "hot" it is (a 0–1 temperature with four bands), answered by three
// evaluators that run on the GPU box (taxo/: WD14, SigLIP 2, Qwen3-VL) and
// imported here. The vocabulary is taxo/taxonomy.yaml; this server reads the
// generated taxonomy.json and never a copy of its own.
//
// Two rules hold regardless of the feature flag:
//   - The minor block. child/teen + anything other than a confident `safe` =
//     out of every view, blob not served, audit row. The flag gates features,
//     not this.
//   - The audit log. Blocks, releases and threshold changes are written down.
// Everything else — facets, filters, the slider, import — is behind
// FINETUNE_TAXONOMY=1.

//go:embed taxonomy.json
var taxonomyJSON []byte

type taxoValue struct {
	ID         string `json:"id"`
	Filterable bool   `json:"filterable"`
}

type taxoNode struct {
	ID            string      `json:"id"`
	Kind          string      `json:"kind"` // single | multi | ordinal
	When          string      `json:"when"`
	Facet         bool        `json:"facet"`
	HumanRequired bool        `json:"humanRequired"`
	LabelCs       string      `json:"labelCs"`
	Values        []taxoValue `json:"values"`
}

func (n *taxoNode) value(id string) (taxoValue, bool) {
	for _, v := range n.Values {
		if v.ID == id {
			return v, true
		}
	}
	return taxoValue{}, false
}

type bandThresholds struct {
	SafeMax       float64  `json:"safe_max"`
	SuggestiveMax float64  `json:"suggestive_max"`
	NsfwMax       float64  `json:"nsfw_max"`
	ExplicitGuard *float64 `json:"explicit_guard,omitempty"`
}

func (t bandThresholds) valid() bool {
	ok := t.SafeMax > 0 && t.SafeMax < t.SuggestiveMax && t.SuggestiveMax < t.NsfwMax && t.NsfwMax < 1
	if t.ExplicitGuard != nil {
		ok = ok && *t.ExplicitGuard >= 0 && *t.ExplicitGuard <= 1
	}
	return ok
}

type taxonomyDoc struct {
	Version           int                `json:"version"`
	SourceSha256      string             `json:"source_sha256"`
	Bands             []string           `json:"bands"`
	DefaultThresholds bandThresholds     `json:"default_thresholds"`
	WD14Levels        map[string]float64 `json:"wd14_levels"`
	MinorBlock        struct {
		Node        string   `json:"node"`
		Values      []string `json:"values"`
		MinConf     float64  `json:"min_conf"`
		WD14Signals []string `json:"wd14_signals"`
	} `json:"minor_block"`
	Nodes []taxoNode `json:"nodes"`
}

func (t *taxonomyDoc) node(id string) *taxoNode {
	for i := range t.Nodes {
		if t.Nodes[i].ID == id {
			return &t.Nodes[i]
		}
	}
	return nil
}

var taxonomy = mustTaxonomy(taxonomyJSON)

func mustTaxonomy(b []byte) *taxonomyDoc {
	var t taxonomyDoc
	if err := json.Unmarshal(b, &t); err != nil {
		panic("taxonomy.json: " + err.Error())
	}
	if !t.DefaultThresholds.valid() || len(t.Bands) != 4 || t.node(t.MinorBlock.Node) == nil {
		panic("taxonomy.json: inconsistent — regenerate with python taxo/gen_schema.py")
	}
	return &t
}

// Feature flags. galleryFilter is a pure function shared by three handlers
// (see judgeFilterVersion), so main sets these once at startup.
var (
	taxoEnabled    = false // FINETUNE_TAXONOMY=1
	taxoPublicView = false // FINETUNE_PUBLIC_VIEW=1: every list and blob is the shared (safe-only) view
)

var errTaxoDisabled = errors.New("taxonomy disabled (set FINETUNE_TAXONOMY=1)")

// taxoEvaluators are the sources an import may write. consensus arrives in its
// own field, human only through PUT — an import can never pose as a person.
var taxoEvaluators = []string{"wd14", "siglip", "vlm", "head"}

// ratingNode is the pseudo-node a human (and the VLM, advisory) rates the band
// on. It is not part of the taxonomy's nodes: the temperature is the
// ensemble's, this is what calibrates it.
const ratingNode = "rating"

func bandOf(t float64, th bandThresholds) string {
	switch {
	case t <= th.SafeMax:
		return "safe"
	case t <= th.SuggestiveMax:
		return "suggestive"
	case t <= th.NsfwMax:
		return "nsfw"
	}
	return "explicit"
}

// guardUnsure: the conformal explicit guard sends everything at or above it
// that the point thresholds would not call explicit to the human queue.
func guardUnsure(t float64, th bandThresholds) bool {
	return th.ExplicitGuard != nil && t >= *th.ExplicitGuard && t <= th.NsfwMax
}

// activeBands is the newest temperature_bands row, or the taxonomy defaults.
func activeBands(db *sql.DB) (bandThresholds, string) {
	var th bandThresholds
	var guard sql.NullFloat64
	var source string
	err := db.QueryRow(`SELECT safe_max, suggestive_max, nsfw_max, explicit_guard, source
		FROM temperature_bands ORDER BY id DESC LIMIT 1`).
		Scan(&th.SafeMax, &th.SuggestiveMax, &th.NsfwMax, &guard, &source)
	if err != nil {
		return taxonomy.DefaultThresholds, "default"
	}
	if guard.Valid {
		g := guard.Float64
		th.ExplicitGuard = &g
	}
	return th, source
}

/* ---------------- the minor rule ---------------- */

type ageVote struct {
	Evaluator string  `json:"evaluator"`
	Value     string  `json:"value"`
	Conf      float64 `json:"conf"`
}

type minorInput struct {
	Votes       []ageVote
	WD14Signal  float64  // highest conf among minor_block.wd14_signals in the auto caption
	HasEnsemble bool     // an imported temperature exists
	Band        string   // its band under the active thresholds
	Unsure      bool     // its unsure
	WD14Temp    *float64 // WD14-only temperature from the caption's rating tags
}

// minorDecision is the rule, pure, so the tests can feed it synthetic label
// vectors (taxo/testdata/minor_vectors.json, shared with taxo/consensus.py).
// "" = no minor case or a confidently safe one; "quarantine" = minor signal
// and only WD14 has rated it (reversible once the ensemble says safe);
// "block" = minor signal and the ensemble says anything but a confident safe.
// One voice is enough for the signal: a block is not a majority decision.
func minorDecision(in minorInput, th bandThresholds) (decision, reason string) {
	mb := taxonomy.MinorBlock
	var why []string
	if in.WD14Signal >= mb.MinConf {
		why = append(why, fmt.Sprintf("wd14 minor tag %.2f", in.WD14Signal))
	}
	for _, v := range in.Votes {
		conf := v.Conf
		if v.Evaluator == "human" {
			conf = 1
		}
		if slices.Contains(mb.Values, v.Value) && conf >= mb.MinConf {
			why = append(why, fmt.Sprintf("age_group=%s by %s (%.2f)", v.Value, v.Evaluator, conf))
		}
	}
	if len(why) == 0 {
		return "", ""
	}
	if in.HasEnsemble {
		if in.Band == "safe" && !in.Unsure {
			return "", ""
		}
		state := in.Band
		if in.Unsure {
			state += ", unsure"
		}
		return "block", strings.Join(why, "; ") + "; rating " + state
	}
	if in.WD14Temp != nil && bandOf(*in.WD14Temp, th) == "safe" {
		return "", ""
	}
	if in.WD14Temp == nil {
		return "quarantine", strings.Join(why, "; ") + "; not rated yet"
	}
	return "quarantine", strings.Join(why, "; ") + fmt.Sprintf("; wd14 temperature %.2f", *in.WD14Temp)
}

// wd14FromTags reads the auto caption's tags_json: the minor signal and the
// WD14-only temperature (probability-weighted mean of the rating levels).
func wd14FromTags(tagsJSON string) (signal float64, temp *float64) {
	var tags tagResponse
	if tagsJSON == "" || json.Unmarshal([]byte(tagsJSON), &tags) != nil {
		return 0, nil
	}
	for _, t := range taxonomy.MinorBlock.WD14Signals {
		signal = math.Max(signal, tags.General[t])
		signal = math.Max(signal, tags.General[strings.ReplaceAll(t, " ", "_")])
	}
	var num, den float64
	for k, lvl := range taxonomy.WD14Levels {
		p := tags.Rating[k]
		num += p * lvl
		den += p
	}
	if den > 0 {
		t := num / den
		temp = &t
	}
	return signal, temp
}

// recheckMinor gathers everything the rule reads for one image, decides and
// applies: block upgrades a quarantine, quarantine never downgrades a block,
// "" releases a quarantine only. Every transition is audited.
func recheckMinor(db *sql.DB, imageID string) (string, error) {
	th, _ := activeBands(db)
	var in minorInput
	rows, err := db.Query(`SELECT evaluator, value, COALESCE(conf, 1) FROM image_taxonomy
		WHERE image_id = ? AND node = ?`, imageID, taxonomy.MinorBlock.Node)
	if err != nil {
		return "", err
	}
	for rows.Next() {
		var v ageVote
		if err := rows.Scan(&v.Evaluator, &v.Value, &v.Conf); err != nil {
			rows.Close()
			return "", err
		}
		in.Votes = append(in.Votes, v)
	}
	rows.Close()
	var tagsJSON sql.NullString
	db.QueryRow(`SELECT tags_json FROM captions WHERE image_id = ? AND kind = 'auto'`, imageID).Scan(&tagsJSON)
	in.WD14Signal, in.WD14Temp = wd14FromTags(tagsJSON.String)
	var unsure int
	err = db.QueryRow(`SELECT band, unsure FROM image_temperature WHERE image_id = ?`, imageID).Scan(&in.Band, &unsure)
	in.HasEnsemble = err == nil
	in.Unsure = unsure == 1

	decision, reason := minorDecision(in, th)
	var cur string
	db.QueryRow(`SELECT state FROM taxonomy_blocks WHERE image_id = ?`, imageID).Scan(&cur)
	var sha string
	db.QueryRow(`SELECT sha256 FROM images WHERE id = ?`, imageID).Scan(&sha)

	audit := func(kind string) error {
		detail, _ := json.Marshal(map[string]any{"reason": reason, "previous": cur})
		_, err := db.Exec(`INSERT INTO taxonomy_audit (kind, image_id, sha256, detail) VALUES (?, ?, ?, ?)`,
			kind, imageID, sha, string(detail))
		return err
	}
	switch {
	case decision == "block" && cur != "block":
		if _, err := db.Exec(`INSERT INTO taxonomy_blocks (image_id, state, reason) VALUES (?, 'block', ?)
			ON CONFLICT(image_id) DO UPDATE SET state = 'block', reason = excluded.reason,
			updated_at = datetime('now')`, imageID, reason); err != nil {
			return "", err
		}
		log.Printf("taxonomy: BLOCK %s (%s)", imageID, reason)
		return "block", audit("block")
	case decision == "quarantine" && cur == "":
		if _, err := db.Exec(`INSERT INTO taxonomy_blocks (image_id, state, reason) VALUES (?, 'quarantine', ?)`,
			imageID, reason); err != nil {
			return "", err
		}
		log.Printf("taxonomy: quarantine %s (%s)", imageID, reason)
		return "quarantine", audit("quarantine")
	case decision == "" && cur == "quarantine":
		if _, err := db.Exec(`DELETE FROM taxonomy_blocks WHERE image_id = ? AND state = 'quarantine'`, imageID); err != nil {
			return "", err
		}
		reason = "rated safe by the ensemble"
		return "", audit("release")
	}
	return cur, nil
}

// sweepMinors re-applies the rule to every captioned or rated image — at
// startup, and after a threshold change moved bands.
func sweepMinors(db *sql.DB) (changed int) {
	rows, err := db.Query(`SELECT id FROM images WHERE blob_present = 1 AND (
		id IN (SELECT image_id FROM captions WHERE kind = 'auto')
		OR id IN (SELECT image_id FROM image_temperature)
		OR id IN (SELECT image_id FROM image_taxonomy WHERE node = ?))`, taxonomy.MinorBlock.Node)
	if err != nil {
		log.Printf("taxonomy sweep: %v", err)
		return 0
	}
	var ids []string
	for rows.Next() {
		var id string
		if rows.Scan(&id) == nil {
			ids = append(ids, id)
		}
	}
	rows.Close()
	for _, id := range ids {
		var before string
		db.QueryRow(`SELECT state FROM taxonomy_blocks WHERE image_id = ?`, id).Scan(&before)
		after, err := recheckMinor(db, id)
		if err != nil {
			log.Printf("taxonomy sweep %s: %v", id, err)
			continue
		}
		if after != before {
			changed++
		}
	}
	return changed
}

// blobBlocked: is any image carrying this blob blocked or quarantined?
func blobBlocked(db *sql.DB, sha string) bool {
	var one int
	err := db.QueryRow(`SELECT 1 FROM taxonomy_blocks tb JOIN images i ON i.id = tb.image_id
		WHERE i.sha256 = ? LIMIT 1`, sha).Scan(&one)
	return err == nil
}

// blobSharedSafe: may a shared/public view serve this blob? Only when some
// gallery image with it is rated safe and not unsure.
func blobSharedSafe(db *sql.DB, sha string) bool {
	var one int
	err := db.QueryRow(`SELECT 1 FROM gallery g JOIN image_temperature it ON it.image_id = g.id
		WHERE g.sha256 = ? AND it.band = 'safe' AND it.unsure = 0 LIMIT 1`, sha).Scan(&one)
	return err == nil
}

/* ---------------- gallery filter ---------------- */

// taxoFilter appends the taxonomy conditions of galleryFilter. Parameters:
//
//	taxo=node:value[,node:value]   effective label (human > consensus; human
//	                               only for human_required nodes)
//	taxo_src=consensus|human|wd14|siglip|vlm|head   read one source instead
//	temp_max=0.3                   temperature ≤ x (unrated images excluded)
//	band=safe[,suggestive]         stored band
//	rating_unsure=1|0
//	taxo_unsure=<node>             consensus unsure on the node (rate-next queue)
//	label_queue=1                  in the labelling sample, not yet labelled
//	view=shared                    safe-only, not unsure (forced in public mode)
func taxoFilter(get func(string) string, conds []string, args []any) ([]string, []any, string) {
	used := false
	for _, k := range []string{"taxo", "taxo_src", "temp_max", "band", "rating_unsure", "taxo_unsure", "label_queue", "view"} {
		if get(k) != "" {
			used = true
		}
	}
	if !used && !taxoPublicView {
		return conds, args, ""
	}
	if !taxoEnabled {
		return nil, nil, errTaxoDisabled.Error()
	}
	src := get("taxo_src")
	if src != "" && src != "consensus" && src != "human" && !slices.Contains(taxoEvaluators, src) {
		return nil, nil, "taxo_src must be consensus|human|" + strings.Join(taxoEvaluators, "|")
	}
	if v := get("taxo"); v != "" {
		for _, pair := range strings.Split(v, ",") {
			nid, val, ok := strings.Cut(strings.TrimSpace(pair), ":")
			n := taxonomy.node(nid)
			if !ok || n == nil {
				return nil, nil, "taxo must be <node>:<value>[,…] with a node of the taxonomy"
			}
			tv, ok := n.value(val)
			if !ok {
				return nil, nil, fmt.Sprintf("taxo: %s has no value %q", nid, val)
			}
			// Minor values exist only as a block. Refusing here (not just hiding
			// them from the facets) makes "search for child" impossible by URL.
			if !tv.Filterable {
				return nil, nil, fmt.Sprintf("taxo: %s:%s is not a filter", nid, val)
			}
			c, a := taxoLabelCond(n, val, src)
			conds = append(conds, c)
			args = append(args, a...)
		}
	}
	if v := get("temp_max"); v != "" {
		x, err := strconv.ParseFloat(v, 64)
		if err != nil || x < 0 || x > 1 {
			return nil, nil, "temp_max must be within [0, 1]"
		}
		conds = append(conds, `EXISTS (SELECT 1 FROM image_temperature it
			WHERE it.image_id = g.id AND it.temperature <= ?)`)
		args = append(args, x)
	}
	if v := get("band"); v != "" {
		bands := strings.Split(v, ",")
		ph := make([]string, len(bands))
		for i, b := range bands {
			if !slices.Contains(taxonomy.Bands, b) {
				return nil, nil, "band must be " + strings.Join(taxonomy.Bands, "|")
			}
			ph[i] = "?"
			args = append(args, b)
		}
		conds = append(conds, `EXISTS (SELECT 1 FROM image_temperature it
			WHERE it.image_id = g.id AND it.band IN (`+strings.Join(ph, ",")+`))`)
	}
	if v := get("rating_unsure"); v == "0" || v == "1" {
		conds = append(conds, `EXISTS (SELECT 1 FROM image_temperature it
			WHERE it.image_id = g.id AND it.unsure = ?)`)
		args = append(args, v)
	}
	if v := get("taxo_unsure"); v != "" {
		if taxonomy.node(v) == nil {
			return nil, nil, "taxo_unsure must be a taxonomy node"
		}
		conds = append(conds, `EXISTS (SELECT 1 FROM image_taxonomy t
			WHERE t.image_id = g.id AND t.node = ? AND t.evaluator = 'consensus' AND t.unsure = 1)
			AND NOT EXISTS (SELECT 1 FROM image_taxonomy h
			WHERE h.image_id = g.id AND h.node = ? AND h.evaluator = 'human')`)
		args = append(args, v, v)
	}
	if get("label_queue") == "1" {
		conds = append(conds, `g.id IN (SELECT image_id FROM taxonomy_label_queue)
			AND NOT EXISTS (SELECT 1 FROM image_taxonomy h WHERE h.image_id = g.id AND h.evaluator = 'human')`)
	}
	view := get("view")
	if view != "" && view != "shared" && view != "private" {
		return nil, nil, "view must be shared|private"
	}
	if view == "shared" || taxoPublicView {
		conds = append(conds, `EXISTS (SELECT 1 FROM image_temperature it
			WHERE it.image_id = g.id AND it.band = 'safe' AND it.unsure = 0)`)
	}
	return conds, args, ""
}

// taxoLabelCond is one node:value condition for the chosen source.
func taxoLabelCond(n *taxoNode, val, src string) (string, []any) {
	switch {
	case src == "human" || (src == "" && n.HumanRequired):
		return `EXISTS (SELECT 1 FROM image_taxonomy t WHERE t.image_id = g.id
			AND t.node = ? AND t.value = ? AND t.evaluator = 'human')`, []any{n.ID, val}
	case src == "consensus":
		return `EXISTS (SELECT 1 FROM image_taxonomy t WHERE t.image_id = g.id
			AND t.node = ? AND t.value = ? AND t.evaluator = 'consensus' AND t.unsure = 0)`, []any{n.ID, val}
	case src != "":
		return `EXISTS (SELECT 1 FROM image_taxonomy t WHERE t.image_id = g.id
			AND t.node = ? AND t.value = ? AND t.evaluator = ?)`, []any{n.ID, val, src}
	}
	return `(EXISTS (SELECT 1 FROM image_taxonomy t WHERE t.image_id = g.id
			AND t.node = ? AND t.value = ? AND t.evaluator = 'human')
		OR (NOT EXISTS (SELECT 1 FROM image_taxonomy t WHERE t.image_id = g.id
			AND t.node = ? AND t.evaluator = 'human')
		AND EXISTS (SELECT 1 FROM image_taxonomy t WHERE t.image_id = g.id
			AND t.node = ? AND t.value = ? AND t.evaluator = 'consensus' AND t.unsure = 0)))`,
		[]any{n.ID, val, n.ID, n.ID, val}
}

/* ---------------- import ---------------- */

type taxoLabel struct {
	Value json.RawMessage `json:"value"`
	Conf  json.RawMessage `json:"conf"`
}

type valueConf struct {
	value string
	conf  *float64
}

// values reads the evaluator format: value is a string or a list; conf is a
// number (single) or {value: conf} (multi).
func (l taxoLabel) values() ([]valueConf, error) {
	var one string
	var many []string
	if err := json.Unmarshal(l.Value, &one); err == nil {
		many = []string{one}
	} else if err := json.Unmarshal(l.Value, &many); err != nil {
		return nil, fmt.Errorf("value must be a string or a list")
	}
	var single float64
	var per map[string]float64
	hasSingle := len(l.Conf) > 0 && json.Unmarshal(l.Conf, &single) == nil
	if !hasSingle && len(l.Conf) > 0 {
		json.Unmarshal(l.Conf, &per)
	}
	out := make([]valueConf, 0, len(many))
	for _, v := range many {
		vc := valueConf{value: v}
		if hasSingle {
			c := single
			vc.conf = &c
		} else if c, ok := per[v]; ok {
			vc.conf = &c
		}
		out = append(out, vc)
	}
	return out, nil
}

type taxoConsensus struct {
	Value  json.RawMessage `json:"value"`
	Unsure bool            `json:"unsure"`
}

type taxoImportRecord struct {
	Image           string                          `json:"image"`
	TaxonomyVersion int                             `json:"taxonomy_version"`
	Labels          map[string]map[string]taxoLabel `json:"labels"`
	Consensus       map[string]taxoConsensus        `json:"consensus"`
	Rating          *struct {
		Temperature *float64           `json:"temperature"`
		Unsure      bool               `json:"unsure"`
		Parts       map[string]float64 `json:"parts"`
	} `json:"rating"`
	VLMRating *string `json:"vlm_rating"`
	Embedding *struct {
		Model string `json:"model"`
		Dim   int    `json:"dim"`
		F32   []byte `json:"f32"` // base64 in JSON
	} `json:"embedding"`
}

// POST /api/taxonomy/import {"records":[…]} — combine.py output (taxo/push.py).
func (s *server) handleTaxonomyImport(w http.ResponseWriter, r *http.Request) {
	if !taxoEnabled {
		writeErr(w, http.StatusNotFound, errTaxoDisabled.Error())
		return
	}
	var body struct {
		Records []taxoImportRecord `json:"records"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	th, _ := activeBands(s.db)
	imported, skipped := 0, map[string]int{}
	var touched []string
	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer tx.Rollback()
	for _, rec := range body.Records {
		if rec.TaxonomyVersion != taxonomy.Version {
			skipped["version"]++
			continue
		}
		var one int
		if tx.QueryRow(`SELECT 1 FROM images WHERE id = ?`, rec.Image).Scan(&one) != nil {
			skipped["unknown image"]++
			continue
		}
		if err := importTaxoRecord(tx, rec, th, skipped); err != nil {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("%s: %v", rec.Image, err))
			return
		}
		imported++
		touched = append(touched, rec.Image)
	}
	if err := tx.Commit(); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	embIndex.invalidate()
	blocked := 0
	for _, id := range touched {
		st, err := recheckMinor(s.db, id)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		if st != "" {
			blocked++
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"imported": imported, "skipped": skipped, "blocked": blocked})
}

func importTaxoRecord(tx *sql.Tx, rec taxoImportRecord, th bandThresholds, skipped map[string]int) error {
	ins := func(node, evaluator, value string, conf *float64, unsure bool) error {
		_, err := tx.Exec(`INSERT INTO image_taxonomy (image_id, node, evaluator, value, conf, unsure, taxonomy_version)
			VALUES (?, ?, ?, ?, ?, ?, ?)`, rec.Image, node, evaluator, value, conf, unsure, taxonomy.Version)
		return err
	}
	for ev, labels := range rec.Labels {
		if !slices.Contains(taxoEvaluators, ev) {
			skipped["evaluator "+ev]++
			continue
		}
		if _, err := tx.Exec(`DELETE FROM image_taxonomy WHERE image_id = ? AND evaluator = ?`, rec.Image, ev); err != nil {
			return err
		}
		for nid, l := range labels {
			n := taxonomy.node(nid)
			if n == nil {
				skipped["unknown node"]++
				continue
			}
			vals, err := l.values()
			if err != nil {
				return fmt.Errorf("%s/%s: %w", ev, nid, err)
			}
			for _, vc := range vals {
				if _, ok := n.value(vc.value); !ok {
					skipped["unknown value"]++
					continue
				}
				if err := ins(nid, ev, vc.value, vc.conf, false); err != nil {
					return err
				}
			}
		}
	}
	if rec.VLMRating != nil && slices.Contains(taxonomy.Bands, *rec.VLMRating) {
		if err := ins(ratingNode, "vlm", *rec.VLMRating, nil, false); err != nil {
			return err
		}
	}
	if rec.Consensus != nil {
		if _, err := tx.Exec(`DELETE FROM image_taxonomy WHERE image_id = ? AND evaluator = 'consensus'`, rec.Image); err != nil {
			return err
		}
		for nid, c := range rec.Consensus {
			n := taxonomy.node(nid)
			if n == nil {
				skipped["unknown node"]++
				continue
			}
			vals, err := taxoLabel{Value: c.Value}.values()
			if err != nil {
				return fmt.Errorf("consensus/%s: %w", nid, err)
			}
			wrote := false
			for _, vc := range vals {
				if _, ok := n.value(vc.value); ok {
					if err := ins(nid, "consensus", vc.value, nil, c.Unsure); err != nil {
						return err
					}
					wrote = true
				}
			}
			if !wrote && c.Unsure {
				// Unsure with nothing agreed still has to reach the rate-next queue.
				if err := ins(nid, "consensus", "", nil, true); err != nil {
					return err
				}
			}
		}
	}
	if rec.Rating != nil && rec.Rating.Temperature != nil {
		t := *rec.Rating.Temperature
		if t < 0 || t > 1 || math.IsNaN(t) {
			return fmt.Errorf("temperature %v outside [0, 1]", t)
		}
		parts, _ := json.Marshal(rec.Rating.Parts)
		unsure := rec.Rating.Unsure || guardUnsure(t, th)
		if _, err := tx.Exec(`INSERT INTO image_temperature
			(image_id, temperature, band, disagree, unsure, parts_json, taxonomy_version)
			VALUES (?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(image_id) DO UPDATE SET temperature = excluded.temperature,
			  band = excluded.band, disagree = excluded.disagree, unsure = excluded.unsure,
			  parts_json = excluded.parts_json, taxonomy_version = excluded.taxonomy_version,
			  updated_at = datetime('now')`,
			rec.Image, t, bandOf(t, th), rec.Rating.Unsure, unsure, string(parts), taxonomy.Version); err != nil {
			return err
		}
	}
	if e := rec.Embedding; e != nil {
		if e.Dim <= 0 || len(e.F32) != 4*e.Dim {
			return fmt.Errorf("embedding: %d bytes for dim %d", len(e.F32), e.Dim)
		}
		if _, err := tx.Exec(`INSERT INTO image_embeddings (image_id, model, dim, vec) VALUES (?, ?, ?, ?)
			ON CONFLICT(image_id) DO UPDATE SET model = excluded.model, dim = excluded.dim,
			  vec = excluded.vec, updated_at = datetime('now')`, rec.Image, e.Model, e.Dim, e.F32); err != nil {
			return err
		}
	}
	return nil
}

/* ---------------- per-image read / human labels ---------------- */

type taxoSummary struct {
	Temperature *float64 `json:"temperature"`
	Band        string   `json:"band,omitempty"`
	Unsure      bool     `json:"unsure"`
}

// GET /api/images/{id}/taxonomy
func (s *server) handleImageTaxonomy(w http.ResponseWriter, r *http.Request) {
	if !taxoEnabled {
		writeErr(w, http.StatusNotFound, errTaxoDisabled.Error())
		return
	}
	id := r.PathValue("id")
	var one int
	if s.db.QueryRow(`SELECT 1 FROM gallery WHERE id = ?`, id).Scan(&one) != nil {
		writeErr(w, http.StatusNotFound, "image not found")
		return
	}
	rows, err := s.db.Query(`SELECT node, evaluator, value, conf, unsure FROM image_taxonomy
		WHERE image_id = ? ORDER BY node, evaluator, value`, id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()
	type ans struct {
		Values []string           `json:"values"`
		Conf   map[string]float64 `json:"conf,omitempty"`
		Unsure bool               `json:"unsure,omitempty"`
	}
	by := map[string]map[string]*ans{} // evaluator → node → answer
	for rows.Next() {
		var node, ev, val string
		var conf sql.NullFloat64
		var unsure bool
		if err := rows.Scan(&node, &ev, &val, &conf, &unsure); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		if by[ev] == nil {
			by[ev] = map[string]*ans{}
		}
		a := by[ev][node]
		if a == nil {
			a = &ans{Values: []string{}}
			by[ev][node] = a
		}
		if val != "" {
			a.Values = append(a.Values, val)
		}
		if conf.Valid {
			if a.Conf == nil {
				a.Conf = map[string]float64{}
			}
			a.Conf[val] = conf.Float64
		}
		a.Unsure = a.Unsure || unsure
	}
	sum := s.taxoSummaries([]string{id})[id]
	var inQueue int
	s.db.QueryRow(`SELECT 1 FROM taxonomy_label_queue WHERE image_id = ?`, id).Scan(&inQueue)
	// Blind labelling, as with the judge: an image of the labelling sample
	// shows no evaluator answer until the human has labelled it — otherwise κ
	// measures deference.
	blind := inQueue == 1 && len(by["human"]) == 0
	// nodes: the full vocabulary, minor values included — labelling one is how
	// a human triggers the block. Filters and facets never list them.
	out := map[string]any{"blind": blind, "human": by["human"], "inLabelQueue": inQueue == 1,
		"nodes": taxonomy.Nodes, "bands": taxonomy.Bands}
	if !blind {
		delete(by, "human")
		out["evaluators"] = by
		out["rating"] = sum
	}
	writeJSON(w, http.StatusOK, out)
}

// PUT /api/images/{id}/taxonomy {"labels": {"<node>": "v" | ["v", …] | null}, "rating": "safe" | null}
func (s *server) handleImageTaxonomyPut(w http.ResponseWriter, r *http.Request) {
	if !taxoEnabled {
		writeErr(w, http.StatusNotFound, errTaxoDisabled.Error())
		return
	}
	id := r.PathValue("id")
	var body struct {
		Labels map[string]json.RawMessage `json:"labels"`
		Rating *json.RawMessage           `json:"rating"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	var one int
	if s.db.QueryRow(`SELECT 1 FROM gallery WHERE id = ?`, id).Scan(&one) != nil {
		writeErr(w, http.StatusNotFound, "image not found")
		return
	}
	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer tx.Rollback()
	set := func(node string, vals []string) error {
		if _, err := tx.Exec(`DELETE FROM image_taxonomy WHERE image_id = ? AND node = ? AND evaluator = 'human'`, id, node); err != nil {
			return err
		}
		for _, v := range vals {
			if _, err := tx.Exec(`INSERT INTO image_taxonomy (image_id, node, evaluator, value, taxonomy_version)
				VALUES (?, ?, 'human', ?, ?)`, id, node, v, taxonomy.Version); err != nil {
				return err
			}
		}
		return nil
	}
	for nid, raw := range body.Labels {
		n := taxonomy.node(nid)
		if n == nil {
			writeErr(w, http.StatusBadRequest, "unknown node "+nid)
			return
		}
		var vals []string
		if string(raw) != "null" {
			vcs, err := taxoLabel{Value: raw}.values()
			if err != nil {
				writeErr(w, http.StatusBadRequest, nid+": "+err.Error())
				return
			}
			for _, vc := range vcs {
				if _, ok := n.value(vc.value); !ok {
					writeErr(w, http.StatusBadRequest, fmt.Sprintf("%s has no value %q", nid, vc.value))
					return
				}
				vals = append(vals, vc.value)
			}
			if n.Kind != "multi" && len(vals) > 1 {
				writeErr(w, http.StatusBadRequest, nid+" takes one value")
				return
			}
		}
		if err := set(nid, vals); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
	}
	if body.Rating != nil {
		var band *string
		json.Unmarshal(*body.Rating, &band)
		if band != nil && !slices.Contains(taxonomy.Bands, *band) {
			writeErr(w, http.StatusBadRequest, "rating must be "+strings.Join(taxonomy.Bands, "|"))
			return
		}
		var vals []string
		if band != nil {
			vals = []string{*band}
		}
		if err := set(ratingNode, vals); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
	}
	if err := tx.Commit(); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	// A human "child" is a signal like any other; a human "safe" rating is
	// calibration data and never lifts a block.
	if _, err := recheckMinor(s.db, id); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// taxoSummaries is the per-item temperature for the gallery grid.
func (s *server) taxoSummaries(ids []string) map[string]*taxoSummary {
	out := map[string]*taxoSummary{}
	if len(ids) == 0 {
		return out
	}
	ph := make([]string, len(ids))
	args := make([]any, len(ids))
	for i, id := range ids {
		ph[i], args[i] = "?", id
	}
	rows, err := s.db.Query(`SELECT image_id, temperature, band, unsure FROM image_temperature
		WHERE image_id IN (`+strings.Join(ph, ",")+`)`, args...)
	if err != nil {
		return out
	}
	defer rows.Close()
	for rows.Next() {
		var id string
		var t float64
		sm := &taxoSummary{}
		if rows.Scan(&id, &t, &sm.Band, &sm.Unsure) == nil {
			sm.Temperature = &t
			out[id] = sm
		}
	}
	return out
}

/* ---------------- thresholds / audit ---------------- */

// GET /api/taxonomy/thresholds
func (s *server) handleThresholdsGet(w http.ResponseWriter, r *http.Request) {
	if !taxoEnabled {
		writeErr(w, http.StatusNotFound, errTaxoDisabled.Error())
		return
	}
	th, src := activeBands(s.db)
	writeJSON(w, http.StatusOK, map[string]any{"thresholds": th, "source": src})
}

// PUT /api/taxonomy/thresholds {"thresholds": {…}, "source": "human", "fit": {…}}
// (conformal.py output). Re-bands every rated image, re-applies the minor
// rule (a moved safe_max can turn a child image's "safe" into "suggestive")
// and audits the change with old and new values.
func (s *server) handleThresholdsPut(w http.ResponseWriter, r *http.Request) {
	if !taxoEnabled {
		writeErr(w, http.StatusNotFound, errTaxoDisabled.Error())
		return
	}
	var body struct {
		Thresholds bandThresholds  `json:"thresholds"`
		Source     string          `json:"source"`
		Fit        json.RawMessage `json:"fit"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	th := body.Thresholds
	if !th.valid() {
		writeErr(w, http.StatusBadRequest, "thresholds must satisfy 0 < safe_max < suggestive_max < nsfw_max < 1")
		return
	}
	if body.Source == "" {
		body.Source = "manual"
	}
	old, oldSrc := activeBands(s.db)
	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer tx.Rollback()
	var fit *string
	if len(body.Fit) > 0 {
		f := string(body.Fit)
		fit = &f
	}
	if _, err := tx.Exec(`INSERT INTO temperature_bands (safe_max, suggestive_max, nsfw_max, explicit_guard, source, fit_json)
		VALUES (?, ?, ?, ?, ?, ?)`, th.SafeMax, th.SuggestiveMax, th.NsfwMax, th.ExplicitGuard, body.Source, fit); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	res, err := tx.Exec(`UPDATE image_temperature SET
		band = CASE WHEN temperature <= ? THEN 'safe' WHEN temperature <= ? THEN 'suggestive'
		            WHEN temperature <= ? THEN 'nsfw' ELSE 'explicit' END,
		unsure = CASE WHEN disagree = 1 THEN 1
		              WHEN ? IS NOT NULL AND temperature >= ? AND temperature <= ? THEN 1 ELSE 0 END,
		updated_at = datetime('now')`,
		th.SafeMax, th.SuggestiveMax, th.NsfwMax, th.ExplicitGuard, th.ExplicitGuard, th.NsfwMax)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	rebanded, _ := res.RowsAffected()
	detail, _ := json.Marshal(map[string]any{"old": old, "oldSource": oldSrc, "new": th, "source": body.Source})
	if _, err := tx.Exec(`INSERT INTO taxonomy_audit (kind, detail) VALUES ('thresholds', ?)`, string(detail)); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := tx.Commit(); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	changed := sweepMinors(s.db)
	writeJSON(w, http.StatusOK, map[string]any{"rebanded": rebanded, "minorChanges": changed})
}

// GET /api/taxonomy/audit?limit=100 — ids and reasons only, never thumbnails.
func (s *server) handleTaxonomyAudit(w http.ResponseWriter, r *http.Request) {
	if !taxoEnabled {
		writeErr(w, http.StatusNotFound, errTaxoDisabled.Error())
		return
	}
	limit := 100
	if n, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && n > 0 && n <= 1000 {
		limit = n
	}
	rows, err := s.db.Query(`SELECT id, at, kind, COALESCE(image_id, ''), COALESCE(sha256, ''), detail
		FROM taxonomy_audit ORDER BY id DESC LIMIT ?`, limit)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var id int64
		var at, kind, img, sha, detail string
		if rows.Scan(&id, &at, &kind, &img, &sha, &detail) == nil {
			out = append(out, map[string]any{"id": id, "at": at, "kind": kind, "imageId": img,
				"sha256": sha, "detail": json.RawMessage(detail)})
		}
	}
	counts := map[string]int{}
	crow, err := s.db.Query(`SELECT state, COUNT(*) FROM taxonomy_blocks GROUP BY state`)
	if err == nil {
		for crow.Next() {
			var st string
			var n int
			if crow.Scan(&st, &n) == nil {
				counts[st] = n
			}
		}
		crow.Close()
	}
	writeJSON(w, http.StatusOK, map[string]any{"entries": out, "blocks": counts})
}

/* ---------------- /api/meta ---------------- */

type taxoFacetValue struct {
	ID    string `json:"id"`
	Count int    `json:"count"`
}

type taxoFacet struct {
	ID            string           `json:"id"`
	Kind          string           `json:"kind"`
	LabelCs       string           `json:"labelCs"`
	HumanRequired bool             `json:"humanRequired"`
	Values        []taxoFacetValue `json:"values"`
}

// taxoMeta: facets with counts of the effective label. Minor values are not
// listed — not even with a zero.
func (s *server) taxoMeta() map[string]any {
	if !taxoEnabled {
		return map[string]any{"enabled": false}
	}
	th, src := activeBands(s.db)
	var humanOnly []string
	for _, n := range taxonomy.Nodes {
		if n.HumanRequired {
			humanOnly = append(humanOnly, "'"+n.ID+"'") // literal ids from the embedded taxonomy
		}
	}
	excl := ""
	if len(humanOnly) > 0 {
		excl = " AND t.node NOT IN (" + strings.Join(humanOnly, ",") + ")"
	}
	counts := map[string]map[string]int{}
	rows, err := s.db.Query(`
		WITH eff AS (
		  SELECT t.image_id, t.node, t.value FROM image_taxonomy t
		  WHERE t.evaluator = 'human' AND t.value != ''
		  UNION
		  SELECT t.image_id, t.node, t.value FROM image_taxonomy t
		  WHERE t.evaluator = 'consensus' AND t.unsure = 0 AND t.value != ''` + excl + `
		    AND NOT EXISTS (SELECT 1 FROM image_taxonomy h
		                    WHERE h.image_id = t.image_id AND h.node = t.node AND h.evaluator = 'human')
		)
		SELECT eff.node, eff.value, COUNT(*) FROM eff JOIN gallery g ON g.id = eff.image_id
		GROUP BY eff.node, eff.value`)
	if err == nil {
		for rows.Next() {
			var node, val string
			var n int
			if rows.Scan(&node, &val, &n) == nil {
				if counts[node] == nil {
					counts[node] = map[string]int{}
				}
				counts[node][val] = n
			}
		}
		rows.Close()
	}
	facets := []taxoFacet{}
	for _, n := range taxonomy.Nodes {
		if !n.Facet {
			continue
		}
		f := taxoFacet{ID: n.ID, Kind: n.Kind, LabelCs: n.LabelCs, HumanRequired: n.HumanRequired, Values: []taxoFacetValue{}}
		for _, v := range n.Values {
			if v.Filterable {
				f.Values = append(f.Values, taxoFacetValue{ID: v.ID, Count: counts[n.ID][v.ID]})
			}
		}
		facets = append(facets, f)
	}
	bandCounts := map[string]int{}
	var rated, unsure int
	brows, err := s.db.Query(`SELECT it.band, it.unsure, COUNT(*) FROM image_temperature it
		JOIN gallery g ON g.id = it.image_id GROUP BY it.band, it.unsure`)
	if err == nil {
		for brows.Next() {
			var b string
			var u, n int
			if brows.Scan(&b, &u, &n) == nil {
				bandCounts[b] += n
				rated += n
				if u == 1 {
					unsure += n
				}
			}
		}
		brows.Close()
	}
	return map[string]any{
		"enabled": true, "version": taxonomy.Version, "publicView": taxoPublicView,
		"bands": taxonomy.Bands, "thresholds": th, "thresholdSource": src,
		"bandCounts": bandCounts, "rated": rated, "unsure": unsure,
		"facets": facets,
	}
}

/* ---------------- similar images ---------------- */

// GET /api/images/{id}/similar?limit=24 — cosine over the SigLIP embeddings.
func (s *server) handleSimilar(w http.ResponseWriter, r *http.Request) {
	if !taxoEnabled {
		writeErr(w, http.StatusNotFound, errTaxoDisabled.Error())
		return
	}
	id := r.PathValue("id")
	limit := 24
	if n, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && n > 0 && n <= 200 {
		limit = n
	}
	idx, err := embIndex.get(s.db)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	hits, ok := idx.nearest(id, limit*2+8)
	if !ok {
		writeErr(w, http.StatusNotFound, "no embedding for this image")
		return
	}
	// Visibility goes through the gallery view (blocks) and the shared rule.
	out := []map[string]any{}
	for _, h := range hits {
		if len(out) >= limit {
			break
		}
		var sha string
		q := `SELECT g.sha256 FROM gallery g WHERE g.id = ?`
		if taxoPublicView || r.URL.Query().Get("view") == "shared" {
			q = `SELECT g.sha256 FROM gallery g JOIN image_temperature it ON it.image_id = g.id
				WHERE g.id = ? AND it.band = 'safe' AND it.unsure = 0`
		}
		if s.db.QueryRow(q, h.id).Scan(&sha) != nil {
			continue
		}
		out = append(out, map[string]any{"id": h.id, "sha256": sha, "score": math.Round(float64(h.score)*1e4) / 1e4})
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": out})
}

type embHit struct {
	id    string
	score float32
}

type embeddingIndex struct {
	ids  []string
	pos  map[string]int
	dim  int
	vecs []float32
}

func (e *embeddingIndex) nearest(id string, k int) ([]embHit, bool) {
	i, ok := e.pos[id]
	if !ok {
		return nil, false
	}
	q := e.vecs[i*e.dim : (i+1)*e.dim]
	hits := make([]embHit, 0, len(e.ids))
	for j := range e.ids {
		if j == i {
			continue
		}
		v := e.vecs[j*e.dim : (j+1)*e.dim]
		var dot float32
		for d := range q {
			dot += q[d] * v[d]
		}
		hits = append(hits, embHit{e.ids[j], dot})
	}
	sort.Slice(hits, func(a, b int) bool { return hits[a].score > hits[b].score })
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits, true
}

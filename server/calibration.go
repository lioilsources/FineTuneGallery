package main

import (
	"crypto/subtle"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"sort"
	"strings"
)

// UC25 calibration on the GPU box: the batch's own token, the calibration
// labelling sample and the export of the human labels for taxo/conformal.py
// and taxo/head.py.

// workerToken (WORKER_TOKEN) lets the GPU batch call the gallery without the
// admin's session: it reaches the gallery through CF Access with its own
// service token and then authenticates here with "Authorization: Bearer".
// The token opens exactly workerRoutes and nothing else, so it can be
// revoked separately and a leaked one cannot read the gallery.
var workerToken string

var workerRoutes = map[string]bool{
	"POST /api/taxonomy/import":    true,
	"PUT /api/taxonomy/thresholds": true,
}

// workerAccessClients (WORKER_ACCESS_CLIENT_IDS, comma list) are the CF
// Access service tokens of batches. Access lets such a token through to the
// whole hostname; here a request carrying one is held to workerRoutes, so
// the batch's CF token alone cannot read or change the gallery either.
var workerAccessClients = map[string]bool{}

func parseClientIDs(v string) map[string]bool {
	out := map[string]bool{}
	for _, id := range strings.Split(v, ",") {
		if id = strings.TrimSpace(id); id != "" {
			out[id] = true
		}
	}
	return out
}

// accessClientID is the service token's Client ID from the Access JWT
// (claim common_name), "" for a person or no Access at all. The signature is
// not checked: this only narrows what a request may do, and a request that
// bypasses Access (the LAN port) can drop the header anyway.
func accessClientID(r *http.Request) string {
	parts := strings.Split(r.Header.Get("Cf-Access-Jwt-Assertion"), ".")
	if len(parts) != 3 {
		return ""
	}
	raw, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return ""
	}
	var claims struct {
		CommonName string `json:"common_name"`
	}
	if json.Unmarshal(raw, &claims) != nil {
		return ""
	}
	return claims.CommonName
}

// machineOnly routes are refused without the worker token once one is
// configured: nobody imports evaluator answers from a browser.
var machineOnly = map[string]bool{"POST /api/taxonomy/import": true}

func workerAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		route := r.Method + " " + r.URL.Path
		if cid := accessClientID(r); workerAccessClients[cid] && !workerRoutes[route] && route != "GET /healthz" {
			writeErr(w, http.StatusForbidden, "this Access service token opens only the worker routes")
			return
		}
		bearer, hasBearer := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		switch {
		case hasBearer:
			if workerToken == "" || subtle.ConstantTimeCompare([]byte(bearer), []byte(workerToken)) != 1 {
				writeErr(w, http.StatusUnauthorized, "bad worker token")
				return
			}
			if !workerRoutes[route] {
				writeErr(w, http.StatusForbidden, "worker token does not open "+route)
				return
			}
		case workerToken != "" && machineOnly[route]:
			writeErr(w, http.StatusUnauthorized, "worker token required")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// Calibration strata of the labelling queue, by the image's WD14 rating
// (captions.tags_json): "hi" = explicit + questionable, "lo" = sensitive +
// general. Each is drawn at random (seeded) within itself, so the conformal
// guarantee can be stated per stratum (see Prompts/01-PLAN-uc25-spark-kalibrace.md,
// Past 1). Only images the evaluators already rated (image_temperature) are
// eligible: the minor rule has run on them, and the gallery view has hidden
// whatever it blocked.
const (
	calibHi = "calib:hi"
	calibLo = "calib:lo"
)

func wd14TopRating(tagsJSON string) string {
	var tags struct {
		Rating map[string]float64 `json:"rating"`
	}
	if json.Unmarshal([]byte(tagsJSON), &tags) != nil {
		return ""
	}
	top, best := "", -1.0
	for k, v := range tags.Rating {
		if v > best || (v == best && k < top) {
			top, best = k, v
		}
	}
	return top
}

func sampleCalibQueue(db *sql.DB, hi, lo int, seed int64) (map[string]int, error) {
	rows, err := db.Query(`SELECT g.id, c.tags_json FROM gallery g
		JOIN captions c ON c.image_id = g.id AND c.kind = 'auto'
		WHERE g.id IN (SELECT image_id FROM image_temperature)
		  AND g.id NOT IN (SELECT image_id FROM taxonomy_label_queue)
		ORDER BY g.id`)
	if err != nil {
		return nil, err
	}
	by := map[string][]string{}
	for rows.Next() {
		var id string
		var tj sql.NullString
		if err := rows.Scan(&id, &tj); err != nil {
			rows.Close()
			return nil, err
		}
		switch wd14TopRating(tj.String) {
		case "explicit", "questionable":
			by[calibHi] = append(by[calibHi], id)
		case "sensitive", "general":
			by[calibLo] = append(by[calibLo], id)
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	rnd := rand.New(rand.NewSource(seed))
	tx, err := db.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	counts := map[string]int{}
	for _, st := range []string{calibHi, calibLo} {
		want := map[string]int{calibHi: hi, calibLo: lo}[st]
		ids := by[st]
		rnd.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
		for _, id := range ids[:min(want, len(ids))] {
			if _, err := tx.Exec(`INSERT INTO taxonomy_label_queue (image_id, stratum, seed) VALUES (?, ?, ?)`, id, st, seed); err != nil {
				return nil, err
			}
			counts[st]++
		}
		counts[st+"/available"] = len(ids)
	}
	return counts, tx.Commit()
}

// exportHuman writes one JSONL line per human-labelled image:
//
//	{"image", "stratum", "rating": {"band"}, "labels": {"<node>": {"value"}}}
//
// the --labels format of taxo/conformal.py and --human of taxo/head.py.
// Ids and labels only, never paths or thumbnails.
func exportHuman(db *sql.DB, w io.Writer) (int, error) {
	rows, err := db.Query(`SELECT t.image_id, t.node, t.value, COALESCE(q.stratum, '')
		FROM image_taxonomy t
		LEFT JOIN taxonomy_label_queue q ON q.image_id = t.image_id
		WHERE t.evaluator = 'human' AND t.value != '' AND t.image_id IN (SELECT id FROM gallery)
		ORDER BY t.image_id, t.node, t.value`)
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	type rec struct {
		Image   string                    `json:"image"`
		Stratum string                    `json:"stratum,omitempty"`
		Rating  map[string]string         `json:"rating,omitempty"`
		Labels  map[string]map[string]any `json:"labels"`
	}
	recs := map[string]*rec{}
	multi := map[string]map[string][]string{}
	for rows.Next() {
		var id, node, val, st string
		if err := rows.Scan(&id, &node, &val, &st); err != nil {
			return 0, err
		}
		r := recs[id]
		if r == nil {
			r = &rec{Image: id, Stratum: st, Labels: map[string]map[string]any{}}
			recs[id], multi[id] = r, map[string][]string{}
		}
		if node == ratingNode {
			r.Rating = map[string]string{"band": val}
			continue
		}
		if n := taxonomy.node(node); n != nil && n.Kind == "multi" {
			multi[id][node] = append(multi[id][node], val)
			r.Labels[node] = map[string]any{"value": multi[id][node]}
		} else {
			r.Labels[node] = map[string]any{"value": val}
		}
	}
	if err := rows.Err(); err != nil {
		return 0, err
	}
	ids := make([]string, 0, len(recs))
	for id := range recs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	enc := json.NewEncoder(w)
	for _, id := range ids {
		if err := enc.Encode(recs[id]); err != nil {
			return 0, fmt.Errorf("export %s: %w", id, err)
		}
	}
	return len(ids), nil
}

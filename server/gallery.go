package main

import (
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"slices"
	"strconv"
	"strings"
)

// galleryItem is one row of the flat gallery list.
type galleryItem struct {
	ID          string          `json:"id"`
	Sha256      string          `json:"sha256"`
	Idx         int             `json:"idx"`
	NodeID      string          `json:"nodeId"`
	SessionID   string          `json:"sessionId"`
	Prompt      string          `json:"prompt"`
	ModelID     *string         `json:"modelId"`
	LoraName    *string         `json:"loraName"`
	PoseID      *string         `json:"poseId"`
	Seed        *int64          `json:"seed"`
	Score       int             `json:"score"`
	HasCritique bool            `json:"hasCritique"`
	Aspects     []string        `json:"aspects"`
	Caption     map[string]bool `json:"caption"`
	CreatedAt   *string         `json:"createdAt"`
	IsImg2img   bool            `json:"isImg2img"`
	Origin      string          `json:"origin"`
}

// galleryFilter builds the WHERE clause shared by the list endpoint and
// dataset from-filter. Cursor/limit are handled by the caller.
func galleryFilter(q map[string][]string) (where string, args []any, bad string) {
	get := func(k string) string {
		if v, ok := q[k]; ok && len(v) > 0 {
			return v[0]
		}
		return ""
	}
	conds := []string{"1=1"}
	if v := get("model"); v != "" {
		conds = append(conds, "g.model_id = ?")
		args = append(args, v)
	}
	if v := get("session"); v != "" {
		conds = append(conds, "g.session_id = ?")
		args = append(args, v)
	}
	if v := get("aspect"); v != "" {
		conds = append(conds, `EXISTS (SELECT 1 FROM image_aspects ia
			JOIN aspects a ON a.id = ia.aspect_id
			WHERE ia.image_id = g.id AND a.name = ?)`)
		args = append(args, v)
	}
	if v := get("score"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < -1 || n > 1 {
			return "", nil, "score must be -1|0|1"
		}
		conds = append(conds, "g.score = ?")
		args = append(args, n)
	}
	if get("unrated") == "1" {
		conds = append(conds, "g.score = 0 AND g.critique = ''")
	}
	if get("has_critique") == "1" {
		conds = append(conds, "g.critique != ''")
	}
	if v := get("captioned"); v == "0" || v == "1" {
		op := "EXISTS"
		if v == "0" {
			op = "NOT EXISTS"
		}
		conds = append(conds, op+" (SELECT 1 FROM captions c WHERE c.image_id = g.id)")
	}
	if v := get("criterion"); v != "" {
		name, val, ok := strings.Cut(v, ":")
		if !ok || !slices.Contains(kCriteria, name) || (val != "1" && val != "-1") {
			return "", nil, "criterion must be <name>:<1|-1>"
		}
		conds = append(conds, `EXISTS (SELECT 1 FROM image_criteria ic
			WHERE ic.image_id = g.id AND ic.criterion = ? AND ic.score = ?)`)
		args = append(args, name, val)
	}
	if v := get("from"); v != "" {
		conds = append(conds, "g.created_at >= ?")
		args = append(args, v)
	}
	if v := get("to"); v != "" {
		conds = append(conds, "g.created_at <= ?")
		args = append(args, v)
	}
	return strings.Join(conds, " AND "), args, ""
}

// GET /api/images
func (s *server) handleImagesList(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	where, args, bad := galleryFilter(q)
	if bad != "" {
		writeErr(w, http.StatusBadRequest, bad)
		return
	}

	limit := 60
	if v := q.Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 200 {
			limit = n
		}
	}

	// Keyset cursor over (COALESCE(created_at,''), id) descending.
	cursorCond := ""
	if c := q.Get("cursor"); c != "" {
		raw, err := base64.URLEncoding.DecodeString(c)
		parts := strings.SplitN(string(raw), "|", 2)
		if err != nil || len(parts) != 2 {
			writeErr(w, http.StatusBadRequest, "invalid cursor")
			return
		}
		cursorCond = " AND (COALESCE(g.created_at,''), g.id) < (?, ?)"
		args = append(args, parts[0], parts[1])
	}

	rows, err := s.db.Query(`
		SELECT g.id, g.sha256, g.idx, g.node_id, g.session_id, g.prompt,
		       g.model_id, g.lora_name, g.pose_id, g.seed, g.score,
		       g.critique != '', g.created_at, g.parent_id IS NOT NULL, g.origin
		FROM gallery g
		WHERE `+where+cursorCond+`
		ORDER BY COALESCE(g.created_at,'') DESC, g.id DESC
		LIMIT ?`, append(args, limit+1)...)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()

	items := []*galleryItem{}
	for rows.Next() {
		it := &galleryItem{Aspects: []string{}, Caption: map[string]bool{}}
		if err := rows.Scan(&it.ID, &it.Sha256, &it.Idx, &it.NodeID, &it.SessionID,
			&it.Prompt, &it.ModelID, &it.LoraName, &it.PoseID, &it.Seed,
			&it.Score, &it.HasCritique, &it.CreatedAt, &it.IsImg2img, &it.Origin); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		items = append(items, it)
	}
	var nextCursor *string
	if len(items) > limit {
		last := items[limit-1]
		created := ""
		if last.CreatedAt != nil {
			created = *last.CreatedAt
		}
		c := base64.URLEncoding.EncodeToString([]byte(created + "|" + last.ID))
		nextCursor = &c
		items = items[:limit]
	}

	if err := s.decorateItems(items); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": items, "nextCursor": nextCursor})
}

// decorateItems fills per-item aspects and caption-kind flags in two grouped
// queries (instead of 2N point lookups).
func (s *server) decorateItems(items []*galleryItem) error {
	if len(items) == 0 {
		return nil
	}
	byID := map[string]*galleryItem{}
	ph := make([]string, len(items))
	ids := make([]any, len(items))
	for i, it := range items {
		byID[it.ID] = it
		ph[i] = "?"
		ids[i] = it.ID
	}
	in := "(" + strings.Join(ph, ",") + ")"

	rows, err := s.db.Query(`
		SELECT ia.image_id, a.name FROM image_aspects ia
		JOIN aspects a ON a.id = ia.aspect_id
		WHERE ia.image_id IN `+in+` ORDER BY a.id`, ids...)
	if err != nil {
		return err
	}
	for rows.Next() {
		var id, name string
		if err := rows.Scan(&id, &name); err != nil {
			rows.Close()
			return err
		}
		byID[id].Aspects = append(byID[id].Aspects, name)
	}
	rows.Close()

	rows, err = s.db.Query(
		"SELECT image_id, kind FROM captions WHERE image_id IN "+in, ids...)
	if err != nil {
		return err
	}
	for rows.Next() {
		var id, kind string
		if err := rows.Scan(&id, &kind); err != nil {
			rows.Close()
			return err
		}
		byID[id].Caption[kind] = true
	}
	rows.Close()
	return nil
}

// GET /api/images/{id}
func (s *server) handleImageDetail(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	it := &galleryItem{Aspects: []string{}, Caption: map[string]bool{}}
	var width, height *int64
	err := s.db.QueryRow(`
		SELECT g.id, g.sha256, g.idx, g.node_id, g.session_id, g.prompt,
		       g.model_id, g.lora_name, g.pose_id, g.seed, g.score,
		       g.critique != '', g.created_at, g.parent_id IS NOT NULL, g.origin,
		       b.width, b.height
		FROM gallery g LEFT JOIN blobs b ON b.sha256 = g.sha256
		WHERE g.id = ?`, id).Scan(
		&it.ID, &it.Sha256, &it.Idx, &it.NodeID, &it.SessionID, &it.Prompt,
		&it.ModelID, &it.LoraName, &it.PoseID, &it.Seed, &it.Score,
		&it.HasCritique, &it.CreatedAt, &it.IsImg2img, &it.Origin, &width, &height)
	if err == sql.ErrNoRows {
		writeErr(w, http.StatusNotFound, "image not found")
		return
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := s.decorateItems([]*galleryItem{it}); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Full node row.
	node := map[string]any{}
	{
		var (
			prompt, origin                                     string
			modelID, lora, poseID, negative, prefix            *string
			samplerName, scheduler, createdAt                  *string
			seed, nw, nh, steps                                *int64
			cfg, denoise                                       *float64
		)
		err := s.db.QueryRow(`
			SELECT prompt, origin, model_id, lora_name, pose_id, negative_prompt,
			       positive_prefix, sampler_name, scheduler, created_at,
			       seed, width, height, steps, cfg, denoise
			FROM nodes WHERE id = ?`, it.NodeID).Scan(
			&prompt, &origin, &modelID, &lora, &poseID, &negative,
			&prefix, &samplerName, &scheduler, &createdAt,
			&seed, &nw, &nh, &steps, &cfg, &denoise)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		node = map[string]any{
			"prompt": prompt, "origin": origin, "modelId": modelID,
			"loraName": lora, "poseId": poseID, "negativePrompt": negative,
			"positivePrefix": prefix, "samplerName": samplerName,
			"scheduler": scheduler, "createdAt": createdAt, "seed": seed,
			"width": nw, "height": nh, "steps": steps, "cfg": cfg,
			"denoise": denoise,
		}
	}

	// Rating.
	rating := map[string]any{"score": 0, "critique": ""}
	{
		var score int
		var critique string
		err := s.db.QueryRow("SELECT score, critique FROM ratings WHERE image_id = ?", id).
			Scan(&score, &critique)
		if err == nil {
			rating["score"], rating["critique"] = score, critique
		} else if err != sql.ErrNoRows {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
	}

	// Criteria.
	criteria := map[string]int{}
	{
		rows, err := s.db.Query("SELECT criterion, score FROM image_criteria WHERE image_id = ?", id)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		for rows.Next() {
			var c string
			var sc int
			if err := rows.Scan(&c, &sc); err != nil {
				rows.Close()
				writeErr(w, http.StatusInternalServerError, err.Error())
				return
			}
			criteria[c] = sc
		}
		rows.Close()
	}

	// Aspect ids (the list view carries names; detail edits by id).
	aspectIDs := []int{}
	{
		rows, err := s.db.Query("SELECT aspect_id FROM image_aspects WHERE image_id = ? ORDER BY aspect_id", id)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		for rows.Next() {
			var a int
			if err := rows.Scan(&a); err != nil {
				rows.Close()
				writeErr(w, http.StatusInternalServerError, err.Error())
				return
			}
			aspectIDs = append(aspectIDs, a)
		}
		rows.Close()
	}

	// Caption layers.
	captions := map[string]any{"auto": nil, "human": nil, "refined": nil}
	{
		rows, err := s.db.Query("SELECT kind, text, tagger FROM captions WHERE image_id = ?", id)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		for rows.Next() {
			var kind, text string
			var tagger *string
			if err := rows.Scan(&kind, &text, &tagger); err != nil {
				rows.Close()
				writeErr(w, http.StatusInternalServerError, err.Error())
				return
			}
			entry := map[string]any{"text": text}
			if tagger != nil {
				entry["tagger"] = *tagger
			}
			captions[kind] = entry
		}
		rows.Close()
	}

	// Parent chain: nearest-first, following parent_id/source_image_id to root.
	type chainEntry struct {
		NodeID  string `json:"nodeId"`
		ImageID string `json:"imageId"`
		Sha256  string `json:"sha256"`
		Prompt  string `json:"prompt"`
	}
	chain := []chainEntry{}
	nodeID := it.NodeID
	for range 50 {
		var parentID, sourceImageID *string
		if err := s.db.QueryRow("SELECT parent_id, source_image_id FROM nodes WHERE id = ?", nodeID).
			Scan(&parentID, &sourceImageID); err != nil {
			break
		}
		if parentID == nil {
			break
		}
		var e chainEntry
		if sourceImageID != nil {
			// Prefer the exact source image; fall back to the parent's first image.
			err = s.db.QueryRow(`
				SELECT i.id, i.sha256, n.id, n.prompt FROM images i
				JOIN nodes n ON n.id = i.node_id
				WHERE i.id = ? AND i.blob_present = 1`, *sourceImageID).
				Scan(&e.ImageID, &e.Sha256, &e.NodeID, &e.Prompt)
		} else {
			err = sql.ErrNoRows
		}
		if err != nil {
			err = s.db.QueryRow(`
				SELECT i.id, i.sha256, n.id, n.prompt FROM images i
				JOIN nodes n ON n.id = i.node_id
				WHERE n.id = ? AND i.blob_present = 1
				ORDER BY i.idx LIMIT 1`, *parentID).
				Scan(&e.ImageID, &e.Sha256, &e.NodeID, &e.Prompt)
		}
		if err != nil {
			break
		}
		chain = append(chain, e)
		nodeID = *parentID
	}

	image := map[string]any{}
	b, _ := json.Marshal(it)
	json.Unmarshal(b, &image)
	image["width"], image["height"] = width, height

	writeJSON(w, http.StatusOK, map[string]any{
		"image":       image,
		"node":        node,
		"rating":      rating,
		"criteria":    criteria,
		"aspects":     aspectIDs,
		"captions":    captions,
		"parentChain": chain,
	})
}

// PUT /api/images/{id}/rating
func (s *server) handleRatingPut(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var body struct {
		Score    int    `json:"score"`
		Critique string `json:"critique"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	if body.Score < -1 || body.Score > 1 {
		writeErr(w, http.StatusBadRequest, "score must be -1|0|1")
		return
	}
	if !s.imageExists(w, id) {
		return
	}
	_, err := s.db.Exec(`
		INSERT INTO ratings (image_id, score, critique, updated_at)
		VALUES (?, ?, ?, datetime('now'))
		ON CONFLICT(image_id) DO UPDATE SET
		  score = excluded.score, critique = excluded.critique,
		  updated_at = datetime('now')`, id, body.Score, body.Critique)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// PUT /api/images/{id}/aspects
func (s *server) handleAspectsPut(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var body struct {
		AspectIDs []int `json:"aspectIds"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	if !s.imageExists(w, id) {
		return
	}
	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer tx.Rollback()
	if _, err := tx.Exec("DELETE FROM image_aspects WHERE image_id = ?", id); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	for _, a := range body.AspectIDs {
		if _, err := tx.Exec(
			"INSERT OR IGNORE INTO image_aspects (image_id, aspect_id) VALUES (?, ?)", id, a); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
	}
	if err := tx.Commit(); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// PUT /api/images/{id}/criteria
func (s *server) handleCriteriaPut(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var body struct {
		Criterion string `json:"criterion"`
		Score     int    `json:"score"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	if !slices.Contains(kCriteria, body.Criterion) {
		writeErr(w, http.StatusBadRequest, "unknown criterion")
		return
	}
	if body.Score < -1 || body.Score > 1 {
		writeErr(w, http.StatusBadRequest, "score must be -1|0|1")
		return
	}
	if !s.imageExists(w, id) {
		return
	}
	var err error
	if body.Score == 0 {
		_, err = s.db.Exec(
			"DELETE FROM image_criteria WHERE image_id = ? AND criterion = ?", id, body.Criterion)
	} else {
		_, err = s.db.Exec(`
			INSERT INTO image_criteria (image_id, criterion, score) VALUES (?, ?, ?)
			ON CONFLICT(image_id, criterion) DO UPDATE SET score = excluded.score`,
			id, body.Criterion, body.Score)
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// PUT /api/images/{id}/caption — human layer; empty text deletes it.
func (s *server) handleCaptionPut(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var body struct {
		Text string `json:"text"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	if !s.imageExists(w, id) {
		return
	}
	var err error
	if strings.TrimSpace(body.Text) == "" {
		_, err = s.db.Exec("DELETE FROM captions WHERE image_id = ? AND kind = 'human'", id)
	} else {
		_, err = s.db.Exec(`
			INSERT INTO captions (image_id, kind, text, updated_at)
			VALUES (?, 'human', ?, datetime('now'))
			ON CONFLICT(image_id, kind) DO UPDATE SET
			  text = excluded.text, updated_at = datetime('now')`, id, body.Text)
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// POST /api/images/{id}/autocaption
func (s *server) handleAutocaption(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !s.imageExists(w, id) {
		return
	}
	s.captioner.Enqueue(id)
	w.WriteHeader(http.StatusAccepted)
}

// GET/POST /api/aspects
func (s *server) handleAspects(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		var body struct {
			Name        string `json:"name"`
			TriggerWord string `json:"triggerWord"`
		}
		if !readJSON(w, r, &body) {
			return
		}
		body.Name = strings.TrimSpace(strings.ToLower(body.Name))
		if body.Name == "" {
			writeErr(w, http.StatusBadRequest, "name required")
			return
		}
		res, err := s.db.Exec(
			"INSERT INTO aspects (name, trigger_word) VALUES (?, ?)", body.Name, body.TriggerWord)
		if err != nil {
			writeErr(w, http.StatusConflict, "aspect exists or invalid: "+err.Error())
			return
		}
		id, _ := res.LastInsertId()
		writeJSON(w, http.StatusCreated, map[string]any{
			"id": id, "name": body.Name, "triggerWord": body.TriggerWord, "builtin": false,
		})
		return
	}
	rows, err := s.db.Query("SELECT id, name, trigger_word, builtin FROM aspects ORDER BY id")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var id int
		var name, trigger string
		var builtin bool
		if err := rows.Scan(&id, &name, &trigger, &builtin); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		out = append(out, map[string]any{
			"id": id, "name": name, "triggerWord": trigger, "builtin": builtin,
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// GET /api/sessions
func (s *server) handleSessions(w http.ResponseWriter, r *http.Request) {
	rows, err := s.db.Query(`
		SELECT s.id, s.title, s.model_id, s.last_ingested_at,
		       (SELECT COUNT(*) FROM images i JOIN nodes n ON n.id = i.node_id
		        WHERE n.session_id = s.id AND i.blob_present = 1)
		FROM sessions s ORDER BY s.last_ingested_at DESC`)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var id, title, modelID, last string
		var count int
		if err := rows.Scan(&id, &title, &modelID, &last, &count); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		out = append(out, map[string]any{
			"id": id, "title": title, "modelId": modelID,
			"lastIngestedAt": last, "imageCount": count,
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// GET /api/stats
func (s *server) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := map[string]any{}
	var total, unrated, liked, disliked int
	err := s.db.QueryRow("SELECT COUNT(*) FROM gallery").Scan(&total)
	if err == nil {
		err = s.db.QueryRow("SELECT COUNT(*) FROM gallery WHERE score = 0 AND critique = ''").Scan(&unrated)
	}
	if err == nil {
		err = s.db.QueryRow("SELECT COUNT(*) FROM gallery WHERE score = 1").Scan(&liked)
	}
	if err == nil {
		err = s.db.QueryRow("SELECT COUNT(*) FROM gallery WHERE score = -1").Scan(&disliked)
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	stats["total"], stats["unrated"], stats["liked"], stats["disliked"] = total, unrated, liked, disliked

	byAspect := []map[string]any{}
	rows, err := s.db.Query(`
		SELECT a.name,
		       COUNT(ia.image_id),
		       COALESCE(SUM(CASE WHEN g.score = 1 THEN 1 ELSE 0 END), 0)
		FROM aspects a
		LEFT JOIN image_aspects ia ON ia.aspect_id = a.id
		LEFT JOIN gallery g ON g.id = ia.image_id
		GROUP BY a.id ORDER BY a.id`)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	for rows.Next() {
		var name string
		var t, l int
		if err := rows.Scan(&name, &t, &l); err != nil {
			rows.Close()
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		byAspect = append(byAspect, map[string]any{"name": name, "total": t, "liked": l})
	}
	rows.Close()
	stats["byAspect"] = byAspect

	// The eval-harness view: which model holds pose / preserves identity.
	byModelCriterion := []map[string]any{}
	rows, err = s.db.Query(`
		SELECT COALESCE(g.model_id, '?'), ic.criterion,
		       SUM(CASE WHEN ic.score = 1 THEN 1 ELSE 0 END),
		       SUM(CASE WHEN ic.score = -1 THEN 1 ELSE 0 END)
		FROM image_criteria ic JOIN gallery g ON g.id = ic.image_id
		GROUP BY g.model_id, ic.criterion ORDER BY g.model_id, ic.criterion`)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	for rows.Next() {
		var model, criterion string
		var up, down int
		if err := rows.Scan(&model, &criterion, &up, &down); err != nil {
			rows.Close()
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		byModelCriterion = append(byModelCriterion, map[string]any{
			"modelId": model, "criterion": criterion, "up": up, "down": down,
		})
	}
	rows.Close()
	stats["byModelCriterion"] = byModelCriterion

	writeJSON(w, http.StatusOK, stats)
}

func (s *server) imageExists(w http.ResponseWriter, id string) bool {
	var ok bool
	if err := s.db.QueryRow("SELECT EXISTS(SELECT 1 FROM images WHERE id = ?)", id).Scan(&ok); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return false
	}
	if !ok {
		writeErr(w, http.StatusNotFound, "image not found")
	}
	return ok
}

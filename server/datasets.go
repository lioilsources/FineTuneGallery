package main

import (
	"archive/tar"
	"compress/gzip"
	"database/sql"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var datasetNameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{1,63}$`)

type datasetRow struct {
	ID          int64   `json:"id"`
	Name        string  `json:"name"`
	AspectID    *int64  `json:"aspectId"`
	BaseModel   string  `json:"baseModel"`
	TriggerWord string  `json:"triggerWord"`
	Repeats     int     `json:"repeats"`
	Status      string  `json:"status"`
	ConfigJSON  *string `json:"configJson,omitempty"`
	BuiltAt     *string `json:"builtAt"`
	ItemCount   int     `json:"itemCount"`
}

func (s *server) scanDataset(row *sql.Row) (*datasetRow, error) {
	var d datasetRow
	err := row.Scan(&d.ID, &d.Name, &d.AspectID, &d.BaseModel, &d.TriggerWord,
		&d.Repeats, &d.Status, &d.ConfigJSON, &d.BuiltAt, &d.ItemCount)
	if err != nil {
		return nil, err
	}
	return &d, nil
}

const datasetSelect = `
	SELECT d.id, d.name, d.aspect_id, d.base_model, d.trigger_word,
	       d.repeats, d.status, d.config_json, d.built_at,
	       (SELECT COUNT(*) FROM dataset_items di WHERE di.dataset_id = d.id)
	FROM datasets d`

// GET /api/datasets
func (s *server) handleDatasetsList(w http.ResponseWriter, r *http.Request) {
	rows, err := s.db.Query(datasetSelect + " ORDER BY d.created_at DESC")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()
	out := []datasetRow{}
	for rows.Next() {
		var d datasetRow
		if err := rows.Scan(&d.ID, &d.Name, &d.AspectID, &d.BaseModel, &d.TriggerWord,
			&d.Repeats, &d.Status, &d.ConfigJSON, &d.BuiltAt, &d.ItemCount); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		out = append(out, d)
	}
	writeJSON(w, http.StatusOK, out)
}

// POST /api/datasets
func (s *server) handleDatasetCreate(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name        string `json:"name"`
		AspectID    *int64 `json:"aspectId"`
		BaseModel   string `json:"baseModel"`
		TriggerWord string `json:"triggerWord"`
		Repeats     int    `json:"repeats"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	body.Name = strings.TrimSpace(strings.ToLower(body.Name))
	if !datasetNameRe.MatchString(body.Name) {
		writeErr(w, http.StatusBadRequest, "name must match [a-z0-9][a-z0-9_-]{1,63} (it becomes a folder)")
		return
	}
	model := modelByID(body.BaseModel)
	if model == nil || !model.Trainable {
		writeErr(w, http.StatusBadRequest, "baseModel must be a trainable SDXL model")
		return
	}
	if body.TriggerWord == "" {
		writeErr(w, http.StatusBadRequest, "triggerWord required")
		return
	}
	if body.Repeats <= 0 {
		body.Repeats = 10
	}
	res, err := s.db.Exec(`
		INSERT INTO datasets (name, aspect_id, base_model, trigger_word, repeats)
		VALUES (?, ?, ?, ?, ?)`,
		body.Name, body.AspectID, body.BaseModel, body.TriggerWord, body.Repeats)
	if err != nil {
		writeErr(w, http.StatusConflict, "create dataset: "+err.Error())
		return
	}
	id, _ := res.LastInsertId()
	d, err := s.scanDataset(s.db.QueryRow(datasetSelect+" WHERE d.id = ?", id))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, d)
}

// GET /api/datasets/{id}
func (s *server) handleDatasetDetail(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	d, err := s.scanDataset(s.db.QueryRow(datasetSelect+" WHERE d.id = ?", id))
	if err == sql.ErrNoRows {
		writeErr(w, http.StatusNotFound, "dataset not found")
		return
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	rows, err := s.db.Query(`
		SELECT di.image_id, i.sha256, COALESCE(g.score, 0)
		FROM dataset_items di
		JOIN images i ON i.id = di.image_id
		LEFT JOIN gallery g ON g.id = di.image_id
		WHERE di.dataset_id = ?`, d.ID)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer rows.Close()
	items := []map[string]any{}
	for rows.Next() {
		var imageID, sha string
		var score int
		if err := rows.Scan(&imageID, &sha, &score); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		caption, _ := s.effectiveCaption(imageID)
		items = append(items, map[string]any{
			"imageId": imageID, "sha256": sha, "caption": caption, "score": score,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"dataset": d, "items": items})
}

// PUT /api/datasets/{id}
func (s *server) handleDatasetUpdate(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var body struct {
		Name        *string `json:"name"`
		TriggerWord *string `json:"triggerWord"`
		Repeats     *int    `json:"repeats"`
		BaseModel   *string `json:"baseModel"`
		ConfigJSON  *string `json:"configJson"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	if body.Name != nil && !datasetNameRe.MatchString(*body.Name) {
		writeErr(w, http.StatusBadRequest, "invalid name")
		return
	}
	if body.BaseModel != nil {
		if m := modelByID(*body.BaseModel); m == nil || !m.Trainable {
			writeErr(w, http.StatusBadRequest, "baseModel must be a trainable SDXL model")
			return
		}
	}
	_, err := s.db.Exec(`
		UPDATE datasets SET
		  name = COALESCE(?, name),
		  trigger_word = COALESCE(?, trigger_word),
		  repeats = COALESCE(?, repeats),
		  base_model = COALESCE(?, base_model),
		  config_json = COALESCE(?, config_json)
		WHERE id = ?`,
		body.Name, body.TriggerWord, body.Repeats, body.BaseModel, body.ConfigJSON, id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// DELETE /api/datasets/{id} — removes DB rows; a built package on disk stays
// (it may already be training) and is listed in the response for manual cleanup.
func (s *server) handleDatasetDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var name string
	if err := s.db.QueryRow("SELECT name FROM datasets WHERE id = ?", id).Scan(&name); err != nil {
		writeErr(w, http.StatusNotFound, "dataset not found")
		return
	}
	if _, err := s.db.Exec("DELETE FROM datasets WHERE id = ?", id); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// POST /api/datasets/{id}/items {"add":[…],"remove":[…]}
func (s *server) handleDatasetItems(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var body struct {
		Add    []string `json:"add"`
		Remove []string `json:"remove"`
	}
	if !readJSON(w, r, &body) {
		return
	}
	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer tx.Rollback()
	for _, imgID := range body.Add {
		if _, err := tx.Exec(
			"INSERT OR IGNORE INTO dataset_items (dataset_id, image_id) VALUES (?, ?)",
			id, imgID); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
	}
	for _, imgID := range body.Remove {
		if _, err := tx.Exec(
			"DELETE FROM dataset_items WHERE dataset_id = ? AND image_id = ?",
			id, imgID); err != nil {
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

// POST /api/datasets/{id}/items/from-filter?<gallery query params>
func (s *server) handleDatasetFromFilter(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	where, args, bad := galleryFilter(r.URL.Query())
	if bad != "" {
		writeErr(w, http.StatusBadRequest, bad)
		return
	}
	res, err := s.db.Exec(`
		INSERT OR IGNORE INTO dataset_items (dataset_id, image_id)
		SELECT ?, g.id FROM gallery g WHERE `+where,
		append([]any{id}, args...)...)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	added, _ := res.RowsAffected()
	writeJSON(w, http.StatusOK, map[string]any{"added": added})
}

// GET /api/datasets/{id}/download — tar.gz of the built package.
func (s *server) handleDatasetDownload(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var name, status string
	if err := s.db.QueryRow("SELECT name, status FROM datasets WHERE id = ?", id).
		Scan(&name, &status); err != nil {
		writeErr(w, http.StatusNotFound, "dataset not found")
		return
	}
	if status != "built" {
		writeErr(w, http.StatusConflict, "dataset not built yet")
		return
	}
	root := filepath.Join(s.dataDir, "datasets", name)
	w.Header().Set("Content-Type", "application/gzip")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s.tar.gz"`, name))
	gz := gzip.NewWriter(w)
	tw := tar.NewWriter(gz)
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(filepath.Dir(root), path)
		hdr, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		hdr.Name = filepath.ToSlash(rel)
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = io.Copy(tw, f)
		return err
	})
	if err != nil {
		// Headers already sent — log only; the client sees a broken archive.
		fmt.Fprintf(os.Stderr, "dataset download %s: %v\n", name, err)
	}
	tw.Close()
	gz.Close()
}

// effectiveCaption resolves human > refined > auto for one image.
func (s *server) effectiveCaption(imageID string) (string, error) {
	var text string
	err := s.db.QueryRow(`
		SELECT text FROM captions WHERE image_id = ?
		ORDER BY CASE kind WHEN 'human' THEN 0 WHEN 'refined' THEN 1 ELSE 2 END
		LIMIT 1`, imageID).Scan(&text)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return text, err
}

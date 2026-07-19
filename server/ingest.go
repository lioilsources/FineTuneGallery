package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"image"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sync"
)

// Manifest is the app's export payload (FinetuneExportService). Node metadata
// fields are pointers: old sessions predate per-node metadata.
type Manifest struct {
	Session struct {
		ID        string `json:"id"`
		Title     string `json:"title"`
		ModelID   string `json:"modelId"`
		UpdatedAt string `json:"updatedAt"`
	} `json:"session"`
	Nodes []ManifestNode `json:"nodes"`
}

type ManifestNode struct {
	ID             string          `json:"id"`
	ParentID       *string         `json:"parentId"`
	SourceImageID  *string         `json:"sourceImageId"`
	Prompt         string          `json:"prompt"`
	Origin         *string         `json:"origin"`
	ModelID        *string         `json:"modelId"`
	LoraName       *string         `json:"loraName"`
	PoseID         *string         `json:"poseId"`
	Seed           *int64          `json:"seed"`
	NegativePrompt *string         `json:"negativePrompt"`
	PositivePrefix *string         `json:"positivePrefix"`
	Width          *int64          `json:"width"`
	Height         *int64          `json:"height"`
	Steps          *int64          `json:"steps"`
	Cfg            *float64        `json:"cfg"`
	Denoise        *float64        `json:"denoise"`
	SamplerName    *string         `json:"samplerName"`
	Scheduler      *string         `json:"scheduler"`
	CreatedAt      *string         `json:"createdAt"`
	Images         []ManifestImage `json:"images"`
}

type ManifestImage struct {
	ID     string `json:"id"`
	Sha256 string `json:"sha256"`
	Size   int64  `json:"size"`
	Idx    int    `json:"idx"`
}

var shaRe = regexp.MustCompile(`^[0-9a-f]{64}$`)

// pendingNewBlobs remembers, per session, how many blobs the last manifest
// reported missing — so finalize can echo an accurate newBlobs count without
// extra client bookkeeping. In-memory only (single user); lost on restart,
// in which case finalize just omits the count's precision (reports uploads
// since manifest as 0 — the client falls back to its own counter).
var pendingNewBlobs sync.Map // sessionID → int

// POST /api/ingest/manifest
func (s *server) handleIngestManifest(w http.ResponseWriter, r *http.Request) {
	var m Manifest
	if !readJSON(w, r, &m) {
		return
	}
	if m.Session.ID == "" {
		writeErr(w, http.StatusBadRequest, "session.id missing")
		return
	}

	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer tx.Rollback()

	_, err = tx.Exec(`
		INSERT INTO sessions (id, title, model_id, app_updated_at)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
		  title = excluded.title,
		  model_id = excluded.model_id,
		  app_updated_at = excluded.app_updated_at,
		  last_ingested_at = datetime('now')`,
		m.Session.ID, m.Session.Title, m.Session.ModelID, m.Session.UpdatedAt)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "upsert session: "+err.Error())
		return
	}

	nodeStmt, err := tx.Prepare(`
		INSERT INTO nodes (id, session_id, parent_id, source_image_id, prompt,
		                   origin, model_id, lora_name, pose_id, seed,
		                   negative_prompt, positive_prefix, width, height,
		                   steps, cfg, denoise, sampler_name, scheduler, created_at)
		VALUES (?, ?, ?, ?, ?, COALESCE(?, 'generated'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
		  prompt = excluded.prompt,
		  origin = excluded.origin,
		  model_id = COALESCE(excluded.model_id, nodes.model_id),
		  lora_name = COALESCE(excluded.lora_name, nodes.lora_name),
		  pose_id = COALESCE(excluded.pose_id, nodes.pose_id),
		  seed = COALESCE(excluded.seed, nodes.seed),
		  negative_prompt = COALESCE(excluded.negative_prompt, nodes.negative_prompt),
		  positive_prefix = COALESCE(excluded.positive_prefix, nodes.positive_prefix),
		  width = COALESCE(excluded.width, nodes.width),
		  height = COALESCE(excluded.height, nodes.height),
		  steps = COALESCE(excluded.steps, nodes.steps),
		  cfg = COALESCE(excluded.cfg, nodes.cfg),
		  denoise = COALESCE(excluded.denoise, nodes.denoise),
		  sampler_name = COALESCE(excluded.sampler_name, nodes.sampler_name),
		  scheduler = COALESCE(excluded.scheduler, nodes.scheduler),
		  created_at = COALESCE(excluded.created_at, nodes.created_at)`)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer nodeStmt.Close()

	imgStmt, err := tx.Prepare(`
		INSERT INTO images (id, node_id, idx, sha256, size, blob_present)
		VALUES (?, ?, ?, ?, ?, EXISTS(SELECT 1 FROM blobs WHERE sha256 = ?))
		ON CONFLICT(id) DO UPDATE SET
		  sha256 = excluded.sha256,
		  size = excluded.size,
		  blob_present = excluded.blob_present`)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer imgStmt.Close()

	neededSet := map[string]bool{}
	for _, n := range m.Nodes {
		if _, err := nodeStmt.Exec(
			n.ID, m.Session.ID, n.ParentID, n.SourceImageID, n.Prompt,
			n.Origin, n.ModelID, n.LoraName, n.PoseID, n.Seed,
			n.NegativePrompt, n.PositivePrefix, n.Width, n.Height,
			n.Steps, n.Cfg, n.Denoise, n.SamplerName, n.Scheduler, n.CreatedAt,
		); err != nil {
			writeErr(w, http.StatusInternalServerError, "upsert node "+n.ID+": "+err.Error())
			return
		}
		for _, img := range n.Images {
			if !shaRe.MatchString(img.Sha256) {
				writeErr(w, http.StatusBadRequest, "invalid sha256 for image "+img.ID)
				return
			}
			if _, err := imgStmt.Exec(img.ID, n.ID, img.Idx, img.Sha256, img.Size, img.Sha256); err != nil {
				writeErr(w, http.StatusInternalServerError, "upsert image "+img.ID+": "+err.Error())
				return
			}
			var have bool
			if err := tx.QueryRow("SELECT EXISTS(SELECT 1 FROM blobs WHERE sha256 = ?)", img.Sha256).Scan(&have); err != nil {
				writeErr(w, http.StatusInternalServerError, err.Error())
				return
			}
			if !have {
				neededSet[img.Sha256] = true
			}
		}
	}
	if err := tx.Commit(); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	needed := make([]string, 0, len(neededSet))
	for sha := range neededSet {
		needed = append(needed, sha)
	}
	pendingNewBlobs.Store(m.Session.ID, len(needed))
	log.Printf("ingest: manifest session=%s nodes=%d needed=%d", m.Session.ID, len(m.Nodes), len(needed))
	writeJSON(w, http.StatusOK, map[string]any{"needed": needed})
}

// PUT /api/ingest/images/{sha256} — raw PNG body, digest-verified.
func (s *server) handleIngestBlob(w http.ResponseWriter, r *http.Request) {
	sha := r.PathValue("sha256")
	if !shaRe.MatchString(sha) {
		writeErr(w, http.StatusBadRequest, "invalid sha256 path")
		return
	}

	var exists bool
	if err := s.db.QueryRow("SELECT EXISTS(SELECT 1 FROM blobs WHERE sha256 = ?)", sha).Scan(&exists); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if exists {
		io.Copy(io.Discard, http.MaxBytesReader(w, r.Body, 64<<20))
		writeJSON(w, http.StatusOK, map[string]any{"sha256": sha, "existed": true})
		return
	}

	// Stream to a temp file while hashing, verify, then move into the store.
	if err := os.MkdirAll(filepath.Join(s.dataDir, "tmp"), 0o755); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	tmp, err := os.CreateTemp(filepath.Join(s.dataDir, "tmp"), "blob-*")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, h), http.MaxBytesReader(w, r.Body, 64<<20))
	tmp.Close()
	if err != nil {
		writeErr(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != sha {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("digest mismatch: body is %s", got))
		return
	}

	// Probe dimensions (also validates it decodes as an image at all).
	var width, height any
	if f, err := os.Open(tmpPath); err == nil {
		if cfg, _, err := image.DecodeConfig(f); err == nil {
			width, height = cfg.Width, cfg.Height
		}
		f.Close()
	}

	dst := s.blobPath(sha)
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := os.Rename(tmpPath, dst); err != nil {
		writeErr(w, http.StatusInternalServerError, "store blob: "+err.Error())
		return
	}

	if _, err := s.db.Exec(`
		INSERT INTO blobs (sha256, size, width, height) VALUES (?, ?, ?, ?)
		ON CONFLICT(sha256) DO NOTHING`, sha, n, width, height); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if _, err := s.db.Exec("UPDATE images SET blob_present = 1 WHERE sha256 = ?", sha); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"sha256": sha, "size": n})
}

// POST /api/ingest/sessions/{id}/finalize
func (s *server) handleIngestFinalize(w http.ResponseWriter, r *http.Request) {
	sessionID := r.PathValue("id")
	var nodes, images int
	err := s.db.QueryRow("SELECT COUNT(*) FROM nodes WHERE session_id = ?", sessionID).Scan(&nodes)
	if err == nil {
		err = s.db.QueryRow(`
			SELECT COUNT(*) FROM images i JOIN nodes n ON n.id = i.node_id
			WHERE n.session_id = ? AND i.blob_present = 1`, sessionID).Scan(&images)
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	newBlobs := 0
	if v, ok := pendingNewBlobs.LoadAndDelete(sessionID); ok {
		newBlobs = v.(int)
	}
	if _, err := s.db.Exec(
		"INSERT INTO ingest_log (session_id, nodes, images, new_blobs) VALUES (?, ?, ?, ?)",
		sessionID, nodes, images, newBlobs); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	log.Printf("ingest: finalize session=%s nodes=%d images=%d new=%d", sessionID, nodes, images, newBlobs)
	// Wake the captioner — newly ingested images need auto captions.
	s.captioner.Kick()
	writeJSON(w, http.StatusOK, map[string]any{
		"images":   images,
		"newBlobs": newBlobs,
		"complete": true,
	})
}

func (s *server) blobPath(sha string) string {
	return filepath.Join(s.dataDir, "images", sha[:2], sha+".png")
}

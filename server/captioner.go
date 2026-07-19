package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// Captioner drives the WD14 tagger sidecar: every ingested blob gets an
// 'auto' caption layer (booru tags — exactly what SDXL/pony models eat).
// In-process queue + a startup self-heal scan; no persistent queue needed —
// missing captions are re-discovered from the DB on every start/kick.
type Captioner struct {
	db        *sql.DB
	taggerURL string // e.g. http://wd14:8000
	dataDir   string
	blobPath  func(sha string) string

	kick chan struct{}
	jobs chan string // specific image ids (manual re-runs)
}

func NewCaptioner(db *sql.DB, taggerURL, dataDir string, blobPath func(string) string) *Captioner {
	return &Captioner{
		db:        db,
		taggerURL: taggerURL,
		dataDir:   dataDir,
		blobPath:  blobPath,
		kick:      make(chan struct{}, 1),
		jobs:      make(chan string, 256),
	}
}

// Kick asks the worker to scan for uncaptioned images (cheap, coalesced).
func (c *Captioner) Kick() {
	select {
	case c.kick <- struct{}{}:
	default:
	}
}

// Enqueue schedules one specific image for (re-)captioning.
func (c *Captioner) Enqueue(imageID string) {
	select {
	case c.jobs <- imageID:
	default:
		log.Printf("captioner: job queue full, dropping %s (self-heal will catch it)", imageID)
	}
}

// Run is the single worker loop. Call in a goroutine.
func (c *Captioner) Run() {
	if c.taggerURL == "" {
		log.Printf("captioner: TAGGER_URL empty — auto-captioning disabled")
		return
	}
	// Startup self-heal after a short grace period (tagger may still be booting).
	time.Sleep(3 * time.Second)
	c.scan()
	for {
		select {
		case id := <-c.jobs:
			if err := c.caption(id); err != nil {
				log.Printf("captioner: %s: %v", id, err)
			}
		case <-c.kick:
			c.scan()
		}
	}
}

// scan captions every blob-present image lacking an 'auto' caption.
func (c *Captioner) scan() {
	rows, err := c.db.Query(`
		SELECT id FROM images
		WHERE blob_present = 1
		  AND id NOT IN (SELECT image_id FROM captions WHERE kind = 'auto')`)
	if err != nil {
		log.Printf("captioner scan: %v", err)
		return
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err == nil {
			ids = append(ids, id)
		}
	}
	rows.Close()
	if len(ids) == 0 {
		return
	}
	log.Printf("captioner: backlog %d images", len(ids))
	for _, id := range ids {
		if err := c.caption(id); err != nil {
			log.Printf("captioner: %s: %v", id, err)
			// Tagger down? Stop hammering; next kick/start retries the backlog.
			return
		}
	}
	log.Printf("captioner: backlog done")
}

type tagResponse struct {
	General   map[string]float64 `json:"general"`
	Character map[string]float64 `json:"character"`
	Rating    map[string]float64 `json:"rating"`
	Model     string             `json:"model"`
}

func (c *Captioner) caption(imageID string) error {
	var sha string
	err := c.db.QueryRow(
		"SELECT sha256 FROM images WHERE id = ? AND blob_present = 1", imageID).Scan(&sha)
	if err != nil {
		return fmt.Errorf("lookup: %w", err)
	}
	blob, err := os.ReadFile(c.blobPath(sha))
	if err != nil {
		return fmt.Errorf("read blob: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, c.taggerURL+"/tag", bytes.NewReader(blob))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "image/png")
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("tagger: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("tagger HTTP %d: %s", resp.StatusCode, string(b))
	}
	var tags tagResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return fmt.Errorf("tagger response: %w", err)
	}

	// Caption text: character tags first (identity), then general tags by
	// confidence. Rating tags are kept only in tags_json.
	text := strings.Join(append(sortedTags(tags.Character), sortedTags(tags.General)...), ", ")
	rawJSON, _ := json.Marshal(tags)

	_, err = c.db.Exec(`
		INSERT INTO captions (image_id, kind, text, tagger, tags_json, updated_at)
		VALUES (?, 'auto', ?, ?, ?, datetime('now'))
		ON CONFLICT(image_id, kind) DO UPDATE SET
		  text = excluded.text, tagger = excluded.tagger,
		  tags_json = excluded.tags_json, updated_at = datetime('now')`,
		imageID, text, tags.Model, string(rawJSON))
	if err != nil {
		return fmt.Errorf("store caption: %w", err)
	}
	log.Printf("captioner: %s ← %d tags", imageID, len(tags.General)+len(tags.Character))
	return nil
}

func sortedTags(m map[string]float64) []string {
	type kv struct {
		k string
		v float64
	}
	all := make([]kv, 0, len(m))
	for k, v := range m {
		all = append(all, kv{k, v})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].v > all[j].v })
	out := make([]string, len(all))
	for i, e := range all {
		out[i] = e.k
	}
	return out
}

// RefineCaption is the v1.5 hook: merge auto tags + the user's critique +
// the node prompt into a refined caption via the vLLM endpoint
// (llm.ol1n.com/v1/chat/completions), stored as kind='refined'. Wire a route
// to it once LLM_REFINE=1 ships; deliberately unimplemented in v1.
func (c *Captioner) RefineCaption(imageID string) error {
	return fmt.Errorf("caption refinement not implemented (v1.5)")
}

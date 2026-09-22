package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"
	"sync/atomic"
	"time"
)

// The model picker used to be a hand-copied snapshot of the app's registry.
// That is how six checkpoints added to the GPU box on 2026-09-21 stayed
// invisible here: nothing in this repo noticed, because nothing in this repo
// was looking. The catalog looks, by merging the three sources that each hold
// a different part of the answer:
//
//	ComfyUI /object_info — which model files the box can load right now.
//	  Authoritative for existence; knows nothing about the app's model ids,
//	  LoRA ecosystems, or what the kohya pipeline can train.
//	kModels (meta.go)    — exactly that missing half, keyed by file name.
//	the nodes table      — what the gallery actually holds. A model the app
//	  has since dropped still has images here, and model is the only
//	  structural filter the gallery has, so its id must stay pickable.
//
// Nothing that reaches the picker depends on all three: a checkpoint no one
// has classified still shows up (unlabelled, not trainable), and an id that
// only exists in old rows still shows up (as itself). The failure that
// started this can no longer happen silently.
type Catalog struct {
	db       *sql.DB
	comfyURL string // e.g. http://192.168.88.66:8188; "" skips the GPU box
	client   *http.Client
	snap     atomic.Pointer[CatalogSnapshot]
}

// CatalogSnapshot is one merged answer. Replaced atomically; never mutated in
// place, so /api/meta never blocks on the GPU box and never sees a half-merge.
type CatalogSnapshot struct {
	Models []ModelMeta
	Comfy  ComfyStatus

	// assets is what the last successful fetch saw, kept so a refresh that
	// cannot reach the box reuses it instead of flipping every model to
	// unavailable because SPARK happened to be rebooting.
	assets *comfyAssets
}

// ComfyStatus rides along in /api/meta so "the picker looks wrong" is a
// question the response itself answers.
type ComfyStatus struct {
	URL       string `json:"url,omitempty"`
	OK        bool   `json:"ok"`
	Stale     bool   `json:"stale,omitempty"` // serving the previous fetch
	Error     string `json:"error,omitempty"`
	Ckpts     int    `json:"ckpts"`
	CheckedAt string `json:"checkedAt,omitempty"`
}

// comfyAssets is the file-level truth from the box: the two loader node lists
// the app's graphs draw from.
type comfyAssets struct {
	Ckpts map[string]struct{} // CheckpointLoaderSimple.ckpt_name
	Unets map[string]struct{} // UNETLoader.unet_name
}

// Video checkpoints share their folders with the image ones — ltx-2.3-22b sits
// in checkpoints/, the whole wan2.2 family in diffusion_models/. The app
// generates stills, so these can never appear in the gallery; discovery skips
// them rather than filling the picker with rows that match nothing.
var videoModelRe = regexp.MustCompile(`(?i)^(ltx|wan[0-9._-]|mochi|cog|svd|hunyuan[_-]?video|animatediff)`)

var nonIDRunRe = regexp.MustCompile(`[^a-z0-9]+`)

func NewCatalog(db *sql.DB, comfyURL string) *Catalog {
	return &Catalog{
		db:       db,
		comfyURL: strings.TrimRight(comfyURL, "/"),
		client:   &http.Client{Timeout: 10 * time.Second},
	}
}

// Run keeps the snapshot fresh. Call in a goroutine: the box may be down at
// boot, and checkpoints appear whenever a Civitai download lands, so this
// retries with backoff and then refreshes every few minutes.
func (c *Catalog) Run() {
	delay := 3 * time.Second
	for {
		if err := c.Refresh(); err != nil {
			log.Printf("catalog: %v (retry in %s)", err, delay)
			time.Sleep(delay)
			if delay < time.Minute {
				delay *= 2
			}
			continue
		}
		delay = 3 * time.Second
		time.Sleep(5 * time.Minute)
	}
}

// Refresh rebuilds the snapshot. It returns an error only when the GPU box
// could not be reached — the snapshot is still replaced, because the registry
// and the gallery halves are always available and a picker built from those
// two is exactly what this server served before.
func (c *Catalog) Refresh() error {
	st := ComfyStatus{URL: c.comfyURL, CheckedAt: time.Now().UTC().Format(time.RFC3339)}

	var assets *comfyAssets
	var fetchErr error
	if c.comfyURL != "" {
		if assets, fetchErr = c.fetch(); fetchErr != nil {
			st.Error = fetchErr.Error()
			if prev := c.snap.Load(); prev != nil && prev.assets != nil {
				assets, st.Stale = prev.assets, true
			}
		} else {
			st.OK = true
		}
	}
	if assets != nil {
		st.Ckpts = len(assets.Ckpts)
	}

	c.snap.Store(&CatalogSnapshot{
		Models: mergeModels(kModels, assets, galleryModelIDs(c.db)),
		Comfy:  st,
		assets: assets,
	})
	return fetchErr
}

// fetch asks for the two loader nodes by name rather than for the whole
// /object_info document, which is megabytes of every node the box has
// installed.
func (c *Catalog) fetch() (*comfyAssets, error) {
	ckpts, err := c.loaderOptions("CheckpointLoaderSimple", "ckpt_name")
	if err != nil {
		return nil, err
	}
	// The FLUX graphs load a UNET instead of a checkpoint; a missing UNETLoader
	// is not fatal, it just leaves those rows unverifiable.
	unets, err := c.loaderOptions("UNETLoader", "unet_name")
	if err != nil {
		log.Printf("catalog: UNETLoader: %v (FLUX availability unchecked)", err)
		unets = map[string]struct{}{}
	}
	return &comfyAssets{Ckpts: ckpts, Unets: unets}, nil
}

// loaderOptions reads one combo widget's option list. ComfyUI shapes it as
// input.required.<field> = [[...options...], {...widget config...}].
func (c *Catalog) loaderOptions(node, field string) (map[string]struct{}, error) {
	resp, err := c.client.Get(c.comfyURL + "/object_info/" + node)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: HTTP %d", node, resp.StatusCode)
	}
	var body map[string]struct {
		Input struct {
			Required map[string][]json.RawMessage `json:"required"`
		} `json:"input"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("%s: %w", node, err)
	}
	spec, ok := body[node]
	if !ok {
		return nil, fmt.Errorf("%s: not installed on this ComfyUI", node)
	}
	raw, ok := spec.Input.Required[field]
	if !ok || len(raw) == 0 {
		return nil, fmt.Errorf("%s: no %s field", node, field)
	}
	var names []string
	if err := json.Unmarshal(raw[0], &names); err != nil {
		return nil, fmt.Errorf("%s.%s: %w", node, field, err)
	}
	set := make(map[string]struct{}, len(names))
	for _, n := range names {
		set[n] = struct{}{}
	}
	return set, nil
}

// mergeModels is the whole merge, pure so it can be tested without a box or a
// database. Registry order is preserved — it is curated, roughly by family —
// with discovered rows appended after it.
func mergeModels(registry []ModelMeta, comfy *comfyAssets, galleryIDs []string) []ModelMeta {
	out := make([]ModelMeta, 0, len(registry)+len(galleryIDs))
	seenID := make(map[string]bool, len(registry))
	claimed := make(map[string]bool, len(registry)) // files a registry row speaks for

	for _, m := range registry {
		m.Source = "registry"
		switch {
		case comfy == nil:
			// Nothing to check against. Report what the registry claims rather
			// than marking every model missing.
			m.Available = true
		case m.Ckpt != "":
			_, m.Available = comfy.Ckpts[m.Ckpt]
			claimed[m.Ckpt] = true
		case m.Unet != "":
			_, m.Available = comfy.Unets[m.Unet]
			claimed[m.Unet] = true
		default:
			// Neither file: not a ComfyUI model at all (the NIM-hosted FLUXes).
			m.Available = true
		}
		out = append(out, m)
		seenID[m.ID] = true
	}

	// Discovery covers checkpoints only. diffusion_models/ is almost entirely
	// WAN video plus UNETs no app graph can drive, so a UNET without a registry
	// row would be a picker entry nothing can generate.
	if comfy != nil {
		for _, f := range sortedKeys(comfy.Ckpts) {
			if claimed[f] || videoModelRe.MatchString(f) {
				continue
			}
			id := deriveModelID(f)
			if id == "" || seenID[id] {
				continue
			}
			out = append(out, ModelMeta{
				ID:    id,
				Label: deriveModelLabel(f),
				Ckpt:  f,
				// Ecosystem and Trainable stay empty on purpose: both are
				// human calls (which LoRAs transfer, what kohya can train) and
				// a guess here would send a training run at the wrong base.
				Available: true,
				Source:    "comfyui",
			})
			seenID[id] = true
		}
	}

	for _, id := range galleryIDs {
		if seenID[id] {
			continue
		}
		// Ingested under an id nothing here recognises. It is unavailable by
		// definition — no file answers to it — but its images are real and the
		// filter has to reach them.
		out = append(out, ModelMeta{ID: id, Label: id, Source: "gallery"})
		seenID[id] = true
	}
	return out
}

// deriveModelID turns a file name into a picker id. It will not match the
// app's id for the same model (the app calls CyberRealisticXL_v10.safetensors
// "cyberrealistic-xl"), which is precisely why the registry row, once someone
// adds one, claims the file and this never runs for it.
func deriveModelID(file string) string {
	base := strings.TrimSuffix(strings.TrimSuffix(file, ".safetensors"), ".ckpt")
	return strings.Trim(nonIDRunRe.ReplaceAllString(strings.ToLower(base), "-"), "-")
}

func deriveModelLabel(file string) string {
	base := strings.TrimSuffix(strings.TrimSuffix(file, ".safetensors"), ".ckpt")
	return strings.Join(strings.FieldsFunc(base, func(r rune) bool { return r == '_' }), " ")
}

// catalogSnapshot is what /api/meta serves. A server without a catalog (tests,
// or the window before the first refresh) still gets the registry and gallery
// halves — the merge does not need the box to produce an answer.
func (s *server) catalogSnapshot() CatalogSnapshot {
	if s.catalog != nil {
		if snap := s.catalog.snap.Load(); snap != nil {
			return *snap
		}
	}
	return CatalogSnapshot{Models: mergeModels(kModels, nil, galleryModelIDs(s.db))}
}

// finetune-gallery server: ingest endpoint for the Ol1nLLM app's session
// exports, a flat rating gallery (embedded Svelte SPA), WD14 auto-captioning,
// and a kohya-ss dataset builder. Runs on the NAS (no GPU); training runs on
// the GPU box via train/run.sh consuming the built packages.
package main

import (
	"database/sql"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type server struct {
	db         *sql.DB
	dataDir    string
	captioner  *Captioner
	translator *Translator
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	// Subcommands share the binary so they share the DB layer and the
	// translator — a benchmark that used a copy of either would measure the
	// copy. `finetune-gallery tagbench -h`.
	if len(os.Args) > 1 && os.Args[1] == "tagbench" {
		os.Exit(runTagbench(os.Args[2:]))
	}

	addr := env("FINETUNE_ADDR", ":8092")
	dataDir := env("FINETUNE_DATA", "/data")
	taggerURL := env("TAGGER_URL", "http://wd14:8000")
	// The LLM gateway on the LAN (AiStack gateway → LiteLLM). Empty disables
	// prompt translation; the app then sends prose and records translated=false.
	gatewayURL := env("LLM_GATEWAY_URL", "")
	translateModel := env("TRANSLATE_MODEL", "prompt-tags")

	for _, sub := range []string{"images", "thumbs", "datasets", "db", "tmp"} {
		if err := os.MkdirAll(filepath.Join(dataDir, sub), 0o755); err != nil {
			log.Fatalf("mkdir %s: %v", sub, err)
		}
	}

	db, err := openDB(filepath.Join(dataDir, "db", "finetune.sqlite"))
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	s := &server{db: db, dataDir: dataDir}
	s.captioner = NewCaptioner(db, taggerURL, dataDir, s.blobPath)
	go s.captioner.Run()
	s.translator = NewTranslator(db, gatewayURL, translateModel, taggerURL)
	if s.translator.Enabled() {
		go s.translator.Run()
	}

	mux := http.NewServeMux()

	// Ingest (Ol1nLLM app → here).
	mux.HandleFunc("POST /api/ingest/manifest", s.handleIngestManifest)
	mux.HandleFunc("PUT /api/ingest/images/{sha256}", s.handleIngestBlob)
	mux.HandleFunc("POST /api/ingest/sessions/{id}/finalize", s.handleIngestFinalize)

	// Gallery.
	mux.HandleFunc("GET /api/images", s.handleImagesList)
	mux.HandleFunc("GET /api/images/{id}", s.handleImageDetail)
	mux.HandleFunc("PUT /api/images/{id}/rating", s.handleRatingPut)
	mux.HandleFunc("PUT /api/images/{id}/aspects", s.handleAspectsPut)
	mux.HandleFunc("PUT /api/images/{id}/criteria", s.handleCriteriaPut)
	mux.HandleFunc("PUT /api/images/{id}/caption", s.handleCaptionPut)
	mux.HandleFunc("POST /api/images/{id}/autocaption", s.handleAutocaption)
	mux.HandleFunc("/api/aspects", s.handleAspects) // GET list + POST create
	mux.HandleFunc("GET /api/sessions", s.handleSessions)
	mux.HandleFunc("DELETE /api/sessions/{id}", s.handleSessionDelete)
	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("GET /api/eval", s.handleEval)
	mux.HandleFunc("POST /api/translate", s.handleTranslate)
	mux.HandleFunc("GET /api/meta", s.handleMeta)

	// Datasets.
	mux.HandleFunc("GET /api/datasets", s.handleDatasetsList)
	mux.HandleFunc("POST /api/datasets", s.handleDatasetCreate)
	mux.HandleFunc("GET /api/datasets/{id}", s.handleDatasetDetail)
	mux.HandleFunc("PUT /api/datasets/{id}", s.handleDatasetUpdate)
	mux.HandleFunc("DELETE /api/datasets/{id}", s.handleDatasetDelete)
	mux.HandleFunc("POST /api/datasets/{id}/items", s.handleDatasetItems)
	mux.HandleFunc("POST /api/datasets/{id}/items/from-filter", s.handleDatasetFromFilter)
	mux.HandleFunc("POST /api/datasets/{id}/build", s.handleDatasetBuild)
	mux.HandleFunc("GET /api/datasets/{id}/download", s.handleDatasetDownload)

	// Images + SPA.
	mux.HandleFunc("GET /img/{sha256}", s.handleImg)
	mux.HandleFunc("GET /thumb/{sha256}", s.handleThumb)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.Handle("/", s.webHandler())

	srv := &http.Server{
		Addr:              addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("finetune-gallery listening on %s (data: %s, tagger: %s, llm gateway: %q)",
		addr, dataDir, taggerURL, gatewayURL)
	log.Fatal(srv.ListenAndServe())
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		// Keep image/thumb serving out of the log noise.
		if r.URL.Path != "/healthz" && !hasPrefixAny(r.URL.Path, "/img/", "/thumb/", "/assets/") {
			log.Printf("%s %s (%s)", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
		}
	})
}

func hasPrefixAny(s string, prefixes ...string) bool {
	for _, p := range prefixes {
		if len(s) >= len(p) && s[:len(p)] == p {
			return true
		}
	}
	return false
}

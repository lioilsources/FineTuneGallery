package main

import (
	"embed"
	"io/fs"
	"net/http"
	"strings"
)

// The Svelte SPA build output (vite outDir ../server/webdist). Hash routing
// means every page loads from /, so no history-API fallback is needed —
// unknown non-API paths still serve index.html defensively.
//
//go:embed all:webdist
var webdist embed.FS

func (s *server) webHandler() http.Handler {
	sub, err := fs.Sub(webdist, "webdist")
	if err != nil {
		panic(err)
	}
	fileServer := http.FileServer(http.FS(sub))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/")
		if path != "" {
			if f, err := sub.Open(path); err == nil {
				f.Close()
				fileServer.ServeHTTP(w, r)
				return
			}
		}
		// SPA fallback.
		r.URL.Path = "/"
		fileServer.ServeHTTP(w, r)
	})
}

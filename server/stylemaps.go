package main

import (
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// Style maps are static packs built outside this server (Ol1nLLM
// `tools/stylemap`: a manifest, one atlas image, one preview per picture) and
// copied into <data>/stylemaps. The app's StyleMap widget reads them from
// here; nothing on this side interprets them.
//
//	stylemaps/index.json            which packs exist
//	stylemaps/<set>/map.json        grid, axes, pictures
//	stylemaps/<set>/atlas.webp
//	stylemaps/<set>/t/<i>.webp

// GET /stylemaps/{path...} — files only: no directory listings, no dotfiles,
// nothing outside the stylemaps directory.
func (s *server) handleStylemap(w http.ResponseWriter, r *http.Request) {
	rel := path.Clean("/" + r.PathValue("path"))
	for _, seg := range strings.Split(rel, "/") {
		if strings.HasPrefix(seg, ".") {
			writeErr(w, http.StatusNotFound, "not found")
			return
		}
	}
	full := filepath.Join(s.dataDir, "stylemaps", filepath.FromSlash(rel))
	st, err := os.Stat(full)
	if err != nil || st.IsDir() {
		writeErr(w, http.StatusNotFound, "not found")
		return
	}
	// A pack is rebuilt in place under the same names, so nothing here is
	// immutable: manifests are revalidated on every open (ServeFile answers
	// 304 from the mtime), pictures may be reused for a day.
	if strings.HasSuffix(rel, ".json") {
		w.Header().Set("Cache-Control", "no-cache")
	} else {
		w.Header().Set("Cache-Control", "public, max-age=86400")
	}
	http.ServeFile(w, r, full)
}

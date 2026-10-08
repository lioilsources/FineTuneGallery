package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestStylemapServesPackFiles(t *testing.T) {
	s := newTestServer(t)
	root := filepath.Join(s.dataDir, "stylemaps")
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("index.json", `{"packs":[]}`)
	write("artists/map.json", `{"id":"artists"}`)
	write("artists/t/7.webp", "RIFF")
	write("artists/.extract.pid", "123")
	// Next to the stylemaps directory, where a traversal would land.
	if err := os.WriteFile(filepath.Join(s.dataDir, "secret.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /stylemaps/{path...}", s.handleStylemap)

	cases := []struct {
		path  string
		code  int
		body  string
		cache string
	}{
		{"/stylemaps/index.json", 200, `{"packs":[]}`, "no-cache"},
		{"/stylemaps/artists/map.json", 200, `{"id":"artists"}`, "no-cache"},
		{"/stylemaps/artists/t/7.webp", 200, "RIFF", "public, max-age=86400"},
		{"/stylemaps/artists/t/8.webp", 404, "", ""},
		{"/stylemaps/artists/", 404, "", ""},  // no directory listing
		{"/stylemaps/artists/t", 404, "", ""}, // nor a redirect into one
		{"/stylemaps/", 404, "", ""},          //
		{"/stylemaps/artists/.extract.pid", 404, "", ""},
		{"/stylemaps/%2e%2e/secret.txt", 404, "", ""},
		{"/stylemaps/artists/%2e%2e/%2e%2e/secret.txt", 404, "", ""},
	}
	for _, c := range cases {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest("GET", c.path, nil))
		if rec.Code != c.code {
			t.Errorf("%s: status %d, want %d", c.path, rec.Code, c.code)
			continue
		}
		if c.code != 200 {
			continue
		}
		if got := rec.Body.String(); got != c.body {
			t.Errorf("%s: body %q, want %q", c.path, got, c.body)
		}
		if got := rec.Header().Get("Cache-Control"); got != c.cache {
			t.Errorf("%s: Cache-Control %q, want %q", c.path, got, c.cache)
		}
	}
}

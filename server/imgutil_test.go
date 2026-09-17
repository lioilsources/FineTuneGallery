package main

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Long-press "uložit obrázek" na mobilu není zaručené gesto, takže detail má
// tlačítko download — a to stojí a padá na hlavičce, kterou vrátí server.
func TestImgDownloadServesAttachment(t *testing.T) {
	s := newTestServer(t)
	p := s.blobPath(shaA)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("png"), 0o644); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("GET", "/img/"+shaA, nil)
	req.SetPathValue("sha256", shaA)
	rec := httptest.NewRecorder()
	s.handleImg(rec, req)
	if got := rec.Header().Get("Content-Disposition"); got != "" {
		t.Errorf("běžné zobrazení nesmí nutit stahování: %q", got)
	}

	req = httptest.NewRequest("GET", "/img/"+shaA+"?download=1&name=ftg-7-aabbccdd.png", nil)
	req.SetPathValue("sha256", shaA)
	rec = httptest.NewRecorder()
	s.handleImg(rec, req)
	if rec.Code != 200 {
		t.Fatalf("%d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Disposition"); got != `attachment; filename="ftg-7-aabbccdd.png"` {
		t.Errorf("Content-Disposition = %q", got)
	}
	if rec.Body.String() != "png" {
		t.Errorf("tělo = %q", rec.Body.String())
	}
}

func TestDownloadNameCannotBreakTheHeader(t *testing.T) {
	cases := []struct{ in, want string }{
		{"ftg-7-aabbccdd.png", "ftg-7-aabbccdd.png"},
		{"bez pripony", "bez-pripony.png"},
		{`"; rm -rf /` + "\r\nX-Evil: 1", "rm--rf-X-Evil-1.png"},
		{"", "image-" + shaA[:12] + ".png"},
		{"../../etc/passwd", "etcpasswd.png"},
	}
	for _, c := range cases {
		if got := downloadName(c.in, shaA); got != c.want {
			t.Errorf("downloadName(%q) = %q, chtěl jsem %q", c.in, got, c.want)
		}
		if strings.ContainsAny(downloadName(c.in, shaA), "\"\r\n;/\\") {
			t.Errorf("downloadName(%q) propustil znak, kterým jde rozbít hlavičku", c.in)
		}
	}
}

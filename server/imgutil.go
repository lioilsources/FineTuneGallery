package main

import (
	"image"
	"image/jpeg"
	_ "image/png"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"golang.org/x/image/draw"
)

const thumbSize = 384

// GET /img/{sha256} — full-size blob, content-addressed → cache forever.
func (s *server) handleImg(w http.ResponseWriter, r *http.Request) {
	sha := r.PathValue("sha256")
	if !shaRe.MatchString(sha) {
		writeErr(w, http.StatusBadRequest, "invalid sha256")
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	http.ServeFile(w, r, s.blobPath(sha))
}

// GET /thumb/{sha256} — 384px JPEG, generated lazily and cached on disk.
func (s *server) handleThumb(w http.ResponseWriter, r *http.Request) {
	sha := r.PathValue("sha256")
	if !shaRe.MatchString(sha) {
		writeErr(w, http.StatusBadRequest, "invalid sha256")
		return
	}
	thumbPath := s.thumbPath(sha)
	if _, err := os.Stat(thumbPath); err != nil {
		if err := makeThumb(s.blobPath(sha), thumbPath); err != nil {
			log.Printf("thumb %s: %v", sha[:8], err)
			// Fall back to the original rather than a broken tile.
			s.handleImg(w, r)
			return
		}
	}
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	http.ServeFile(w, r, thumbPath)
}

func makeThumb(srcPath, dstPath string) error {
	f, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	src, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return err
	}

	b := src.Bounds()
	wd, ht := b.Dx(), b.Dy()
	if wd > ht {
		ht = ht * thumbSize / wd
		wd = thumbSize
	} else {
		wd = wd * thumbSize / ht
		ht = thumbSize
	}
	if wd < 1 {
		wd = 1
	}
	if ht < 1 {
		ht = 1
	}
	dst := image.NewRGBA(image.Rect(0, 0, wd, ht))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, b, draw.Over, nil)

	if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
		return err
	}
	tmp := dstPath + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := jpeg.Encode(out, dst, &jpeg.Options{Quality: 85}); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	out.Close()
	return os.Rename(tmp, dstPath)
}

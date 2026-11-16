package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
)

// lineageManifest: an uploaded photo (u) edited twice (u1 → u2, u2 has no
// blob of its parent u1), a text-only chain (t → t1) and an orphan whose
// parent never arrived (o).
const lineageManifest = `{
  "session": {"id":"s1","title":"linie","modelId":"pony","updatedAt":"x"},
  "nodes": [
    {"id":"u","prompt":"photo","origin":"upload",
     "images":[{"id":"iu","sha256":"` + shaU + `","size":1,"idx":0}]},
    {"id":"u1","parentId":"u","sourceImageId":"iu","prompt":"edit 1",
     "images":[{"id":"iu1","sha256":"` + shaU1 + `","size":1,"idx":0}]},
    {"id":"u2","parentId":"u1","sourceImageId":"iu1","prompt":"edit 2",
     "images":[{"id":"iu2","sha256":"` + shaU2 + `","size":1,"idx":0}]},
    {"id":"t","prompt":"a cat",
     "images":[{"id":"it","sha256":"` + shaT + `","size":1,"idx":0}]},
    {"id":"t1","parentId":"t","sourceImageId":"it","prompt":"a cat, red",
     "images":[{"id":"it1","sha256":"` + shaT1 + `","size":1,"idx":0}]},
    {"id":"o","parentId":"gone","prompt":"orphan",
     "images":[{"id":"io","sha256":"` + shaO + `","size":1,"idx":0}]}
  ]
}`

const (
	shaU  = "1111111111111111111111111111111111111111111111111111111111111111"
	shaU1 = "2222222222222222222222222222222222222222222222222222222222222222"
	shaU2 = "3333333333333333333333333333333333333333333333333333333333333333"
	shaT  = "4444444444444444444444444444444444444444444444444444444444444444"
	shaT1 = "5555555555555555555555555555555555555555555555555555555555555555"
	shaO  = "6666666666666666666666666666666666666666666666666666666666666666"
)

func listItems(t *testing.T, s *server, query string) (int, []galleryItem) {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleImagesList(rec, httptest.NewRequest("GET", "/api/images?"+query, nil))
	var out struct {
		Items []galleryItem `json:"items"`
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out.Items
}

func TestLineageRootUploadAndTextFilter(t *testing.T) {
	s := newTestServer(t)
	ingestJSON(t, s, lineageManifest)
	// u1's PNG is missing: the chain in the detail stops there, the lineage
	// over nodes must not.
	markBlobsPresent(t, s, shaU, shaU2, shaT, shaT1, shaO)

	code, items := listItems(t, s, "")
	if code != 200 {
		t.Fatalf("list → %d", code)
	}
	by := map[string]galleryItem{}
	for _, it := range items {
		by[it.ID] = it
	}
	str := func(p *string) string {
		if p == nil {
			return "<nil>"
		}
		return *p
	}
	for _, c := range []struct {
		id, root, rootOrigin string
		upload, complete     bool
	}{
		{"iu", "u", "upload", true, true},
		{"iu2", "u", "upload", true, true},
		{"it", "t", "generated", false, true},
		{"it1", "t", "generated", false, true},
		{"io", "o", "<nil>", false, false},
	} {
		it, ok := by[c.id]
		if !ok {
			t.Fatalf("%s missing from the list", c.id)
		}
		if str(it.RootNodeID) != c.root || str(it.RootOrigin) != c.rootOrigin ||
			it.LineageHasUpload != c.upload || it.LineageComplete != c.complete {
			t.Errorf("%s: root %s/%s upload %v complete %v, want %s/%s %v %v", c.id,
				str(it.RootNodeID), str(it.RootOrigin), it.LineageHasUpload, it.LineageComplete,
				c.root, c.rootOrigin, c.upload, c.complete)
		}
	}

	ids := func(items []galleryItem) []string {
		out := []string{}
		for _, it := range items {
			out = append(out, it.ID)
		}
		slices.Sort(out)
		return out
	}
	// Text only: the orphan is out because its lineage cannot be verified.
	if _, got := listItems(t, s, "lineage=text"); !slices.Equal(ids(got), []string{"it", "it1"}) {
		t.Errorf("lineage=text → %v", ids(got))
	}
	if _, got := listItems(t, s, "lineage=upload"); !slices.Equal(ids(got), []string{"iu", "iu2"}) {
		t.Errorf("lineage=upload → %v", ids(got))
	}
	if code, _ := listItems(t, s, "lineage=maybe"); code != http.StatusBadRequest {
		t.Errorf("lineage=maybe → %d, want 400", code)
	}
}

func TestLineageSourceImageFromAnUploadCounts(t *testing.T) {
	// A text root whose child was made from an uploaded image of another node:
	// parent_id alone would call it text.
	s := newTestServer(t)
	ingestJSON(t, s, `{"session":{"id":"s1","title":"x","modelId":"pony","updatedAt":"x"},
	  "nodes":[
	    {"id":"up","prompt":"photo","origin":"upload","images":[{"id":"iup","sha256":"`+shaU+`","size":1,"idx":0}]},
	    {"id":"t","prompt":"a cat","images":[{"id":"it","sha256":"`+shaT+`","size":1,"idx":0}]},
	    {"id":"t1","parentId":"t","sourceImageId":"iup","prompt":"mix","images":[{"id":"it1","sha256":"`+shaT1+`","size":1,"idx":0}]}]}`)
	markBlobsPresent(t, s, shaU, shaT, shaT1)
	_, got := listItems(t, s, "lineage=text")
	if len(got) != 1 || got[0].ID != "it" {
		t.Fatalf("lineage=text → %+v", got)
	}
}

func TestParentChainCarriesOrigin(t *testing.T) {
	s := newTestServer(t)
	ingestJSON(t, s, lineageManifest)
	markBlobsPresent(t, s, shaU, shaU1, shaU2)
	req := httptest.NewRequest("GET", "/api/images/iu2", nil)
	req.SetPathValue("id", "iu2")
	rec := httptest.NewRecorder()
	s.handleImageDetail(rec, req)
	if rec.Code != 200 {
		t.Fatalf("detail → %d %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Image struct {
			RootOrigin       *string `json:"rootOrigin"`
			LineageHasUpload bool    `json:"lineageHasUpload"`
		} `json:"image"`
		ParentChain []struct {
			NodeID string `json:"nodeId"`
			Origin string `json:"origin"`
		} `json:"parentChain"`
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	if len(out.ParentChain) != 2 || out.ParentChain[0].NodeID != "u1" ||
		out.ParentChain[0].Origin != "generated" || out.ParentChain[1].Origin != "upload" {
		t.Fatalf("parentChain: %+v", out.ParentChain)
	}
	if out.Image.RootOrigin == nil || *out.Image.RootOrigin != "upload" || !out.Image.LineageHasUpload {
		t.Fatalf("image lineage: %+v", out.Image)
	}
}

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// fakeComfy serves the two loader nodes in ComfyUI's own shape:
// input.required.<field> = [[...options...], {...widget config...}].
func fakeComfy(t *testing.T, ckpts, unets []string) *httptest.Server {
	t.Helper()
	node := func(name, field string, opts []string) map[string]any {
		return map[string]any{
			name: map[string]any{
				"input": map[string]any{
					"required": map[string]any{
						field: []any{opts, map[string]any{"tooltip": "the model"}},
					},
				},
			},
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		switch r.URL.Path {
		case "/object_info/CheckpointLoaderSimple":
			body = node("CheckpointLoaderSimple", "ckpt_name", ckpts)
		case "/object_info/UNETLoader":
			body = node("UNETLoader", "unet_name", unets)
		default:
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func modelsByID(models []ModelMeta) map[string]ModelMeta {
	out := make(map[string]ModelMeta, len(models))
	for _, m := range models {
		out[m.ID] = m
	}
	return out
}

func TestCatalogReadsObjectInfo(t *testing.T) {
	srv := fakeComfy(t,
		[]string{"sd_xl_base_1.0.safetensors", "AutismMix_pony.safetensors"},
		[]string{"flux1-dev.safetensors"})

	assets, err := NewCatalog(nil, srv.URL).fetch()
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if len(assets.Ckpts) != 2 || len(assets.Unets) != 1 {
		t.Fatalf("přečteno %d ckpt / %d unet, čekám 2 / 1", len(assets.Ckpts), len(assets.Unets))
	}
	if _, ok := assets.Ckpts["AutismMix_pony.safetensors"]; !ok {
		t.Errorf("ckpt seznam nedorazil celý: %v", sortedKeys(assets.Ckpts))
	}
}

// A UNETLoader the box does not have must not take the checkpoints down with
// it: only FLUX availability becomes unverifiable.
func TestCatalogSurvivesMissingUNETLoader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/object_info/CheckpointLoaderSimple" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"CheckpointLoaderSimple": map[string]any{
			"input": map[string]any{"required": map[string]any{
				"ckpt_name": []any{[]string{"pony.safetensors"}, map[string]any{}},
			}},
		}})
	}))
	t.Cleanup(srv.Close)

	assets, err := NewCatalog(nil, srv.URL).fetch()
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if len(assets.Ckpts) != 1 || len(assets.Unets) != 0 {
		t.Fatalf("ckpt=%d unet=%d", len(assets.Ckpts), len(assets.Unets))
	}
}

// The reason this file exists: a checkpoint on the box that nobody has added
// to kModels still has to reach the picker.
func TestMergeDiscoversUnclassifiedCheckpoints(t *testing.T) {
	registry := []ModelMeta{
		{ID: "pony", Label: "Pony V6", Ckpt: "pony.safetensors", Ecosystem: "pony", Trainable: true},
	}
	assets := &comfyAssets{
		Ckpts: set("pony.safetensors", "BrandNew_XL_v3.safetensors"),
		Unets: set(),
	}

	got := modelsByID(mergeModels(registry, assets, nil))
	m, ok := got["brandnew-xl-v3"]
	if !ok {
		t.Fatalf("nový checkpoint se do pickeru nedostal: %v", sortedKeys(got))
	}
	if m.Label != "BrandNew XL v3" || m.Source != "comfyui" || !m.Available {
		t.Errorf("odvozená položka = %+v", m)
	}
	// Ecosystem and trainable are human calls; guessing them would aim a
	// training run at the wrong base model.
	if m.Trainable || m.Ecosystem != "" {
		t.Errorf("odvozená položka si vymyslela ecosystem/trainable: %+v", m)
	}
}

func TestMergeMarksRegistryModelsTheBoxDoesNotHave(t *testing.T) {
	registry := []ModelMeta{
		{ID: "pony", Ckpt: "pony.safetensors"},
		{ID: "gone", Ckpt: "deleted.safetensors"},
		{ID: "flux-manga", Unet: "flux1-dev.safetensors"},
		{ID: "flux-schnell"}, // NIM-hosted: no file to check
	}
	assets := &comfyAssets{Ckpts: set("pony.safetensors"), Unets: set("flux1-dev.safetensors")}

	got := modelsByID(mergeModels(registry, assets, nil))
	for id, want := range map[string]bool{
		"pony": true, "gone": false, "flux-manga": true, "flux-schnell": true,
	} {
		if got[id].Available != want {
			t.Errorf("%s: available=%v, čekám %v", id, got[id].Available, want)
		}
	}
	// Gone from the box, but its images are still in the gallery — dropping the
	// row would make them unfilterable.
	if _, ok := got["gone"]; !ok {
		t.Error("chybějící checkpoint zmizel z pickeru místo available:false")
	}
}

func TestMergeSkipsVideoCheckpoints(t *testing.T) {
	assets := &comfyAssets{
		Ckpts: set(
			"ltx-2.3-22b-dev-fp8.safetensors",
			"wan2.2_ti2v_5B_fp16.safetensors",
			"hunyuan_video_t2v.safetensors",
			"RealVisXL_V5.0.safetensors",
		),
		Unets: set(),
	}
	got := modelsByID(mergeModels(nil, assets, nil))
	if len(got) != 1 {
		t.Fatalf("picker dostal %d položek, čekám jen obrázkový model: %v", len(got), sortedKeys(got))
	}
	if _, ok := got["realvisxl-v5-0"]; !ok {
		t.Errorf("filtr videa sebral i obrázkový model: %v", sortedKeys(got))
	}
}

// The bug this all started from, in its most stubborn form: an id that exists
// only in ingested rows. No registry entry, no file on the box — and model is
// the only structural filter the gallery has.
func TestMergeKeepsGalleryOnlyModelIDs(t *testing.T) {
	registry := []ModelMeta{{ID: "pony", Ckpt: "pony.safetensors"}}
	assets := &comfyAssets{Ckpts: set("pony.safetensors"), Unets: set()}

	got := modelsByID(mergeModels(registry, assets, []string{"pony", "retired-model"}))
	m, ok := got["retired-model"]
	if !ok {
		t.Fatalf("id z galerie není v pickeru: %v", sortedKeys(got))
	}
	if m.Source != "gallery" || m.Available {
		t.Errorf("položka z galerie = %+v, čekám source=gallery a available=false", m)
	}
	if len(got) != 2 {
		t.Errorf("pony se zdvojil: %v", sortedKeys(got))
	}
}

// Without the box there is still an answer — the one this server served before
// the catalog existed.
func TestMergeWithoutComfyReportsRegistryAsAvailable(t *testing.T) {
	got := modelsByID(mergeModels([]ModelMeta{{ID: "pony", Ckpt: "pony.safetensors"}}, nil, nil))
	if !got["pony"].Available {
		t.Error("bez ComfyUI se modely tváří jako chybějící")
	}
}

// A rebooting SPARK must not empty the picker's availability flags.
func TestCatalogServesLastGoodAssetsWhenBoxGoesDown(t *testing.T) {
	up := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !up {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"CheckpointLoaderSimple": map[string]any{
			"input": map[string]any{"required": map[string]any{
				"ckpt_name": []any{[]string{"ponyDiffusionV6XL_v6StartWithThisOne.safetensors"}, map[string]any{}},
			}},
		}})
	}))
	t.Cleanup(srv.Close)

	c := NewCatalog(nil, srv.URL)
	if err := c.Refresh(); err != nil {
		t.Fatalf("první refresh: %v", err)
	}
	if !c.snap.Load().Comfy.OK {
		t.Fatal("první refresh se netváří jako úspěšný")
	}

	up = false
	if err := c.Refresh(); err == nil {
		t.Fatal("výpadek boxu se neohlásil jako chyba")
	}
	snap := c.snap.Load()
	if !snap.Comfy.Stale || snap.Comfy.Error == "" {
		t.Errorf("stav = %+v, čekám stale s popsanou chybou", snap.Comfy)
	}
	if !modelsByID(snap.Models)["pony"].Available {
		t.Error("výpadek boxu shodil dostupnost modelu, místo aby držel poslední známý stav")
	}
}

func TestDeriveModelID(t *testing.T) {
	for file, want := range map[string]string{
		"CyberRealisticXL_v10.safetensors": "cyberrealisticxl-v10",
		"sd_xl_base_1.0.safetensors":       "sd-xl-base-1-0",
		"HassakuXL_Illustrious_v3.4.ckpt":  "hassakuxl-illustrious-v3-4",
		"__weird__.safetensors":            "weird",
	} {
		if got := deriveModelID(file); got != want {
			t.Errorf("deriveModelID(%q) = %q, čekám %q", file, got, want)
		}
	}
}

func set(names ...string) map[string]struct{} {
	out := make(map[string]struct{}, len(names))
	for _, n := range names {
		out[n] = struct{}{}
	}
	return out
}

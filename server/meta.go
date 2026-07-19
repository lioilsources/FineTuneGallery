package main

import "net/http"

// ModelMeta mirrors the app's model registry (Ol1nLLM lib/models/image_model.dart).
// Ecosystem drives LoRA compatibility: a LoRA transfers well within its
// ecosystem (pony ↔ atomix), poorly across. Trainable marks the SDXL family
// covered by the v1 kohya pipeline.
type ModelMeta struct {
	ID        string `json:"id"`
	Label     string `json:"label"`
	Ckpt      string `json:"ckpt,omitempty"` // checkpoint filename on the GPU box
	Ecosystem string `json:"ecosystem"`
	Trainable bool   `json:"trainable"`
}

var kModels = []ModelMeta{
	{ID: "flux-schnell", Label: "FLUX Schnell", Ecosystem: "flux"},
	{ID: "flux-kontext", Label: "FLUX Kontext", Ecosystem: "flux"},
	{ID: "flux-manga", Label: "FLUX manga", Ecosystem: "flux"},
	{ID: "pony", Label: "Pony V6", Ckpt: "ponyDiffusionV6XL_v6StartWithThisOne.safetensors", Ecosystem: "pony", Trainable: true},
	{ID: "juggernaut-xl", Label: "Juggernaut XL", Ckpt: "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", Ecosystem: "sdxl-base", Trainable: true},
	{ID: "illustrious-xl", Label: "Illustrious XL", Ckpt: "Illustrious-XL-v2.0.safetensors", Ecosystem: "illustrious", Trainable: true},
	{ID: "atomix-pony-anime", Label: "Atomix Pony Anime", Ckpt: "atomixPonyAnimeXL_v30.safetensors", Ecosystem: "pony", Trainable: true},
	{ID: "sd15", Label: "SD 1.5", Ckpt: "v1-5-pruned-emaonly-fp16.safetensors", Ecosystem: "sd15"},
}

var kCriteria = []string{"pose_adherence", "source_identity", "source_style"}

func modelByID(id string) *ModelMeta {
	for i := range kModels {
		if kModels[i].ID == id {
			return &kModels[i]
		}
	}
	return nil
}

func (s *server) handleMeta(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"models":   kModels,
		"criteria": kCriteria,
	})
}

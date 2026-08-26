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
	{ID: "noobai-xl", Label: "NoobAI XL", Ckpt: "noobai-xl-eps11.safetensors", Ecosystem: "illustrious", Trainable: true},
	{ID: "wai-illustrious", Label: "WAI Illustrious", Ckpt: "wai-nsfw-illustrious.safetensors", Ecosystem: "illustrious", Trainable: true},
	{ID: "animagine-xl", Label: "Animagine XL 4", Ckpt: "animagine-xl-40-opt.safetensors", Ecosystem: "sdxl-base", Trainable: true},
	{ID: "atomix-pony-anime", Label: "Atomix Pony Anime", Ckpt: "atomixPonyAnimeXL_v30.safetensors", Ecosystem: "pony", Trainable: true},
	// Distilled 4-step checkpoint: fine to generate with and to rate, but a
	// LoRA for this family is trained on Juggernaut base, not on the
	// distillation — hence not trainable here.
	{ID: "juggernaut-xl-lightning", Label: "Juggernaut XL Lightning", Ckpt: "Juggernaut-XL-Lightning_4Steps.safetensors", Ecosystem: "sdxl-base"},
	{ID: "flux-fill", Label: "FLUX Fill", Ecosystem: "flux"},
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
		"styles":   s.ingestedStyles(),
	})
}

// ingestedStyles lists the art-style ids that have actually arrived, rather
// than a copy of the app's 40-entry preset registry. A registry copy here
// would be a second source of truth that goes stale the moment the app adds a
// style; the ids are readable on their own and this list can never be wrong.
func (s *server) ingestedStyles() []string {
	rows, err := s.db.Query(
		`SELECT DISTINCT style_id FROM nodes WHERE style_id IS NOT NULL AND style_id != '' ORDER BY style_id`)
	if err != nil {
		return []string{}
	}
	defer rows.Close()
	out := []string{}
	for rows.Next() {
		var v string
		if rows.Scan(&v) == nil {
			out = append(out, v)
		}
	}
	return out
}

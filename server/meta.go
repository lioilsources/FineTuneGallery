package main

import (
	"database/sql"
	"net/http"
)

// ModelMeta is one row of the model picker. Ckpt/Unet are the join key with
// ComfyUI: the file the GPU box loads. Everything else is what ComfyUI cannot
// know — the app's model id (what ingest writes into nodes.model_id, so the
// only thing the gallery filter can match on), the human label, the LoRA
// ecosystem (a LoRA transfers well within one — pony ↔ atomix — poorly across)
// and whether the v1 kohya pipeline can train on it.
//
// Available/Source are filled in by the catalog merge, not by hand.
type ModelMeta struct {
	ID        string `json:"id"`
	Label     string `json:"label"`
	Ckpt      string `json:"ckpt,omitempty"` // CheckpointLoaderSimple ckpt_name
	Unet      string `json:"unet,omitempty"` // UNETLoader unet_name (the FLUX graphs)
	Ecosystem string `json:"ecosystem"`
	Trainable bool   `json:"trainable"`

	// Available: the GPU box can load this model right now. Models that do not
	// run on ComfyUI at all (the NIM-hosted FLUXes carry neither Ckpt nor Unet)
	// have nothing to check against and stay true.
	Available bool `json:"available"`
	// Source: where this row came from — "registry" (the table below),
	// "comfyui" (a checkpoint on the box nobody has classified yet) or
	// "gallery" (an id that only survives in ingested rows).
	Source string `json:"source"`
}

// kModels is the bridge between the app's registry (Ol1nLLM
// lib/models/image_model.dart) and the files on the GPU box. It is not the
// list of models — catalog.go derives that — so a checkpoint missing here
// still reaches the picker, just without a label or an ecosystem.
//
// Ids must match the app's: ingest stores them verbatim in nodes.model_id and
// the gallery filter compares strings.
var kModels = []ModelMeta{
	{ID: "flux-schnell", Label: "FLUX Schnell", Ecosystem: "flux"},
	{ID: "flux-kontext", Label: "FLUX Kontext", Ecosystem: "flux"},
	{ID: "flux-manga", Label: "FLUX manga", Unet: "flux1-dev.safetensors", Ecosystem: "flux"},
	{ID: "pony", Label: "Pony V6", Ckpt: "ponyDiffusionV6XL_v6StartWithThisOne.safetensors", Ecosystem: "pony", Trainable: true},
	{ID: "juggernaut-xl", Label: "Juggernaut XL", Ckpt: "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", Ecosystem: "sdxl-base", Trainable: true},
	{ID: "cyberrealistic-xl", Label: "CyberRealistic XL", Ckpt: "CyberRealisticXL_v10.safetensors", Ecosystem: "sdxl-base", Trainable: true},
	{ID: "realvis-xl", Label: "RealVis XL V5", Ckpt: "RealVisXL_V5.0.safetensors", Ecosystem: "sdxl-base", Trainable: true},
	{ID: "lustify-zenith", Label: "Lustify ZENITH", Ckpt: "Lustify_ZENITH_V9.safetensors", Ecosystem: "sdxl-base", Trainable: true},
	{ID: "sdxl-base", Label: "SDXL base 1.0", Ckpt: "sd_xl_base_1.0.safetensors", Ecosystem: "sdxl-base", Trainable: true},
	{ID: "illustrious-xl", Label: "Illustrious XL", Ckpt: "Illustrious-XL-v2.0.safetensors", Ecosystem: "illustrious", Trainable: true},
	{ID: "noobai-xl", Label: "NoobAI XL", Ckpt: "noobai-xl-eps11.safetensors", Ecosystem: "illustrious", Trainable: true},
	{ID: "wai-illustrious", Label: "WAI Illustrious", Ckpt: "wai-nsfw-illustrious.safetensors", Ecosystem: "illustrious", Trainable: true},
	{ID: "hassaku-illustrious", Label: "Hassaku XL Illustrious", Ckpt: "HassakuXL_Illustrious_v3.4.safetensors", Ecosystem: "illustrious", Trainable: true},
	{ID: "animagine-xl", Label: "Animagine XL 4", Ckpt: "animagine-xl-40-opt.safetensors", Ecosystem: "sdxl-base", Trainable: true},
	{ID: "atomix-pony-anime", Label: "Atomix Pony Anime", Ckpt: "atomixPonyAnimeXL_v30.safetensors", Ecosystem: "pony", Trainable: true},
	{ID: "autismmix-pony", Label: "AutismMix Pony", Ckpt: "AutismMix_pony.safetensors", Ecosystem: "pony", Trainable: true},
	// Distilled 4-step checkpoint: fine to generate with and to rate, but a
	// LoRA for this family is trained on Juggernaut base, not on the
	// distillation — hence not trainable here.
	{ID: "juggernaut-xl-lightning", Label: "Juggernaut XL Lightning", Ckpt: "Juggernaut-XL-Lightning_4Steps.safetensors", Ecosystem: "sdxl-base"},
	{ID: "flux-fill", Label: "FLUX Fill", Unet: "flux1-fill-dev-fp8.safetensors", Ecosystem: "flux"},
	{ID: "sd15", Label: "SD 1.5", Ckpt: "v1-5-pruned-emaonly-fp16.safetensors", Ecosystem: "sd15"},
}

var kCriteria = []string{"pose_adherence", "source_identity", "source_style"}

// modelByID answers from the registry alone, deliberately: it gates dataset
// creation and the kohya config, and those need a hand-classified ecosystem
// and trainable flag. A checkpoint the catalog discovered on the GPU box has
// neither, so it must not become a training base by accident.
func modelByID(id string) *ModelMeta {
	for i := range kModels {
		if kModels[i].ID == id {
			return &kModels[i]
		}
	}
	return nil
}

func (s *server) handleMeta(w http.ResponseWriter, r *http.Request) {
	snap := s.catalogSnapshot()
	writeJSON(w, http.StatusOK, map[string]any{
		"models":     snap.Models,
		"comfyui":    snap.Comfy,
		"criteria":   kCriteria,
		"styles":     s.ingestedStyles(),
		"translator": s.translatorMeta(),
		"judge":      s.judgeMeta(),
	})
}

// ingestedStyles lists the art-style ids that have actually arrived, rather
// than a copy of the app's 40-entry preset registry. A registry copy here
// would be a second source of truth that goes stale the moment the app adds a
// style; the ids are readable on their own and this list can never be wrong.
func (s *server) ingestedStyles() []string {
	return distinctNodeColumn(s.db, "style_id")
}

// galleryModelIDs is the same idea for models, and the reason the picker can
// no longer lose one: whatever ingest wrote is pickable, classified or not.
// Takes the db rather than the server because the catalog refresher has one
// and no server.
func galleryModelIDs(db *sql.DB) []string {
	return distinctNodeColumn(db, "model_id")
}

// col is a literal at every call site — never request data.
func distinctNodeColumn(db *sql.DB, col string) []string {
	if db == nil {
		return []string{}
	}
	rows, err := db.Query(
		`SELECT DISTINCT ` + col + ` FROM nodes WHERE ` + col + ` IS NOT NULL AND ` + col + ` != '' ORDER BY ` + col)
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

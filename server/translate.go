package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"sync/atomic"
	"time"
)

// The prompt translator.
//
// The booru-trained SDXL checkpoints (Pony, Illustrious, NoobAI, Animagine)
// were trained on Danbooru tags and read prose badly — the app's own UGC
// pipeline recorded "no character" producing a bust and "empty item"
// producing a whole ninja, until the templates were rewritten as tags. The
// app sends them the user's free text in prose, so the same loss is happening
// on every generation there.
//
// An LLM can do the translation, but only on three conditions, all of which
// this file exists to enforce in one place rather than in every caller:
//
//   - Deterministic and attributable. Same input, same output, forever: the
//     result is cached by a hash of (text, language, translator version), so
//     the phone, the lab and the benchmark receive identical tags and the
//     eval harness can group on the version that produced them.
//   - Constrained to the vocabulary. An LLM invents tags freely; the WD14
//     tagger's selected_tags.csv is the canonical list of what these
//     checkpoints actually learned. Anything outside it is dropped, and the
//     drop is reported rather than hidden.
//   - No fallback. When the gateway is down the answer is an error, never a
//     different model's translation — that would change the output under the
//     same version string. Failing open (sending raw prose) is the caller's
//     decision and the caller records it as translated=false.

// promptVersion is bumped whenever the system prompt or the parsing changes,
// because either changes the output for the same input.
const promptVersion = "v1"

// systemPrompt is the whole of the translator's behaviour. Quality and rating
// tags are excluded on purpose: the app prepends the model's own positivePrefix
// (score_9… for Pony, masterpiece… for Illustrious), and a duplicate there is
// worse than none.
const systemPrompt = `You convert an image-generation prompt written in natural language into Danbooru-style tags for an SDXL anime/illustration checkpoint (Pony, Illustrious, NoobAI).

Rules:
- Output ONLY one line of comma-separated tags. No prose, no explanation, no numbering.
- Never output quality or rating tags (no masterpiece, best quality, score_9, safe, nsfw, explicit).
- Use canonical Danbooru spelling: lowercase, spaces instead of underscores (long hair, looking at viewer, from side).
- Begin with the subject count: 1girl, 1boy, 2girls, multiple girls, no humans.
- Translate every concrete visual element the prompt contains: subject, hair, eyes, expression, clothing, pose and body position, setting, lighting, composition and camera, art medium.
- Prefer the specific tag over the generic one (ballet slippers, not shoes).
- Do not invent details the prompt does not contain. Do not add style tags unless the prompt names a style.
- 8 to 25 tags.`

// fewShot anchors the format. Two examples is enough for instruction-tuned
// models; more would spend tokens on every call.
var fewShot = []gatewayMessage{
	{Role: "user", Content: "a ballerina on pointe in an empty rehearsal studio, morning light from a tall window, seen from the side"},
	{Role: "assistant", Content: "1girl, solo, ballerina, ballet slippers, en pointe, leotard, tutu, hair bun, standing on one leg, from side, full body, indoors, dance studio, wooden floor, window, sunlight, morning, empty room"},
	{Role: "user", Content: "close-up portrait of a young man with short dark hair and freckles, smiling at the camera, wearing a denim jacket, city street at dusk behind him"},
	{Role: "assistant", Content: "1boy, solo, portrait, close-up, short hair, black hair, freckles, smile, looking at viewer, denim jacket, upper body, outdoors, city, street, dusk, evening, blurry background"},
}

type gatewayMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Translation is one answer, cached or fresh.
type Translation struct {
	Text       string   `json:"text"`       // the tags joined with ", " — what goes into the prompt
	Tags       []string `json:"tags"`       // the same, as a list
	Dropped    []string `json:"dropped"`    // what the LLM said and the vocabulary refused
	Translator string   `json:"translator"` // model alias @ prompt version
	Cached     bool     `json:"cached"`
}

// vocabSet is the tagger's general-tag list as a lookup. Replaced atomically
// on reload; never mutated in place.
type vocabSet struct {
	set   map[string]struct{}
	model string
}

type Translator struct {
	db         *sql.DB
	gatewayURL string // LiteLLM via the gateway, e.g. http://spark:8080; "" disables
	model      string // LiteLLM alias, e.g. prompt-tags
	taggerURL  string
	client     *http.Client
	vocab      atomic.Pointer[vocabSet]
}

// ErrTranslatorDisabled is returned when no gateway is configured — distinct
// from a gateway that is configured but failing, which the caller may retry.
var ErrTranslatorDisabled = errors.New("translator disabled: LLM_GATEWAY_URL not set")

// ErrNoVocabulary is returned while the tagger's tag list has not arrived.
// Translating without the filter would emit unconstrained tags under a version
// string that promises constrained ones, so it is refused instead.
var ErrNoVocabulary = errors.New("translator: vocabulary not loaded yet (tagger /tags unreachable)")

func NewTranslator(db *sql.DB, gatewayURL, model, taggerURL string) *Translator {
	t := &Translator{
		db:         db,
		gatewayURL: strings.TrimRight(gatewayURL, "/"),
		model:      model,
		taggerURL:  strings.TrimRight(taggerURL, "/"),
		client:     &http.Client{Timeout: 45 * time.Second},
	}
	return t
}

func (t *Translator) Enabled() bool { return t.gatewayURL != "" }

// Version is what gets recorded on every node this translator touched.
func (t *Translator) Version() string { return t.versionFor(t.model) }

func (t *Translator) versionFor(model string) string { return model + "@" + promptVersion }

func (t *Translator) VocabSize() int {
	if v := t.vocab.Load(); v != nil {
		return len(v.set)
	}
	return 0
}

// Run keeps the vocabulary loaded. Call in a goroutine: the tagger boots
// slowly and may not be there at start, so this retries with backoff and then
// refreshes hourly (a tagger model swap changes the list).
func (t *Translator) Run() {
	if t.taggerURL == "" {
		return
	}
	delay := 3 * time.Second
	for {
		if err := t.LoadVocab(); err != nil {
			log.Printf("translator: vocab: %v (retry in %s)", err, delay)
			time.Sleep(delay)
			if delay < time.Minute {
				delay *= 2
			}
			continue
		}
		time.Sleep(time.Hour)
	}
}

// LoadVocab fetches GET {tagger}/tags and swaps the set in.
func (t *Translator) LoadVocab() error {
	resp, err := t.client.Get(t.taggerURL + "/tags")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("tagger /tags: HTTP %d", resp.StatusCode)
	}
	var body struct {
		Model   string   `json:"model"`
		General []string `json:"general"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return fmt.Errorf("tagger /tags: %w", err)
	}
	if len(body.General) == 0 {
		return errors.New("tagger /tags: empty vocabulary")
	}
	set := make(map[string]struct{}, len(body.General))
	for _, g := range body.General {
		set[normalizeTag(g)] = struct{}{}
	}
	t.vocab.Store(&vocabSet{set: set, model: body.Model})
	log.Printf("translator: vocabulary loaded (%d tags from %s)", len(set), body.Model)
	return nil
}

// SetVocab installs a vocabulary directly (tests, or a file-backed list).
func (t *Translator) SetVocab(tags []string) {
	set := make(map[string]struct{}, len(tags))
	for _, g := range tags {
		set[normalizeTag(g)] = struct{}{}
	}
	t.vocab.Store(&vocabSet{set: set, model: "manual"})
}

// Translate turns prose into tags with the default model, through the cache.
func (t *Translator) Translate(ctx context.Context, text string) (Translation, error) {
	return t.TranslateWith(ctx, t.model, text, true)
}

// TranslateWith is Translate with the model chosen per call — the benchmark
// compares aliases side by side — and the cache optional.
func (t *Translator) TranslateWith(ctx context.Context, model, text string, useCache bool) (Translation, error) {
	if !t.Enabled() {
		return Translation{}, ErrTranslatorDisabled
	}
	vocab := t.vocab.Load()
	if vocab == nil {
		return Translation{}, ErrNoVocabulary
	}
	text = strings.TrimSpace(text)
	version := t.versionFor(model)
	if text == "" {
		return Translation{Text: "", Tags: []string{}, Dropped: []string{}, Translator: version}, nil
	}

	key := cacheKey(text, "tags", version)
	if useCache {
		if tr, ok := t.cached(key); ok {
			tr.Translator = version
			tr.Cached = true
			return tr, nil
		}
	}

	raw, err := t.callGateway(ctx, model, text)
	if err != nil {
		return Translation{}, err
	}
	tags, dropped := filterTags(parseTags(raw), vocab)
	tr := Translation{
		Text:       strings.Join(tags, ", "),
		Tags:       tags,
		Dropped:    dropped,
		Translator: version,
	}
	if useCache {
		t.store(key, text, version, tr)
	}
	return tr, nil
}

func cacheKey(text, language, version string) string {
	sum := sha256.Sum256([]byte(text + "\x00" + language + "\x00" + version))
	return hex.EncodeToString(sum[:])
}

func (t *Translator) cached(key string) (Translation, bool) {
	var tagsJSON string
	err := t.db.QueryRow(`SELECT tags FROM translations WHERE hash = ?`, key).Scan(&tagsJSON)
	if err != nil {
		return Translation{}, false
	}
	var stored struct {
		Tags    []string `json:"tags"`
		Dropped []string `json:"dropped"`
	}
	if json.Unmarshal([]byte(tagsJSON), &stored) != nil {
		return Translation{}, false
	}
	if stored.Tags == nil {
		stored.Tags = []string{}
	}
	if stored.Dropped == nil {
		stored.Dropped = []string{}
	}
	return Translation{Text: strings.Join(stored.Tags, ", "), Tags: stored.Tags, Dropped: stored.Dropped}, true
}

func (t *Translator) store(key, text, version string, tr Translation) {
	payload, _ := json.Marshal(map[string]any{"tags": tr.Tags, "dropped": tr.Dropped})
	if _, err := t.db.Exec(`
		INSERT OR IGNORE INTO translations (hash, text, language, translator, tags)
		VALUES (?, ?, 'tags', ?, ?)`, key, text, version, string(payload)); err != nil {
		log.Printf("translator: cache store: %v", err)
	}
}

// callGateway is one OpenAI-compatible chat completion. temperature 0 is set
// here; enable_thinking=false is set on the LiteLLM route for the alias, so
// the gateway sees a plain request it already knows how to treat.
func (t *Translator) callGateway(ctx context.Context, model, text string) (string, error) {
	msgs := make([]gatewayMessage, 0, len(fewShot)+2)
	msgs = append(msgs, gatewayMessage{Role: "system", Content: systemPrompt})
	msgs = append(msgs, fewShot...)
	msgs = append(msgs, gatewayMessage{Role: "user", Content: text})
	body, _ := json.Marshal(map[string]any{
		"model":       model,
		"messages":    msgs,
		"temperature": 0,
		"max_tokens":  200,
		"stream":      false,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		t.gatewayURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer dummy")
	resp, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("gateway: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("gateway: HTTP %d: %s", resp.StatusCode, snippet(raw))
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || len(out.Choices) == 0 {
		return "", fmt.Errorf("gateway: unexpected response: %s", snippet(raw))
	}
	return out.Choices[0].Message.Content, nil
}

func snippet(b []byte) string {
	s := strings.TrimSpace(string(b))
	if len(s) > 160 {
		s = s[:160] + "…"
	}
	return s
}

// parseTags reads the model's line back as a list. Tolerant of what
// instruction-tuned models do around a list — a code fence, a trailing
// period, tags split over lines — and strict about the tags themselves.
func parseTags(raw string) []string {
	raw = strings.ReplaceAll(raw, "```", "")
	raw = strings.ReplaceAll(raw, "\n", ",")
	seen := map[string]bool{}
	var out []string
	for _, part := range strings.Split(raw, ",") {
		tag := normalizeTag(part)
		if tag == "" || seen[tag] {
			continue
		}
		seen[tag] = true
		out = append(out, tag)
	}
	return out
}

// normalizeTag is the one spelling everything is compared in: lowercase,
// single spaces, no underscores, no wrapping punctuation.
func normalizeTag(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Trim(s, `"'.;:()[]{}`)
	s = strings.ReplaceAll(s, "_", " ")
	s = strings.Join(strings.Fields(s), " ")
	return s
}

// filterTags keeps only what the tagger vocabulary knows, in the model's
// order, and reports the rest. The cap protects the CLIP token budget: 40
// tags is already more than a prompt should carry.
func filterTags(tags []string, vocab *vocabSet) (kept, dropped []string) {
	kept, dropped = []string{}, []string{}
	for _, tag := range tags {
		if _, ok := vocab.set[tag]; ok {
			if len(kept) < 40 {
				kept = append(kept, tag)
			}
			continue
		}
		dropped = append(dropped, tag)
	}
	return kept, dropped
}

// POST /api/translate  {"text": "...", "language": "tags"}
func (s *server) handleTranslate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text     string `json:"text"`
		Language string `json:"language"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.Language == "" {
		req.Language = "tags"
	}
	if req.Language != "tags" {
		writeErr(w, http.StatusBadRequest, "language must be 'tags'")
		return
	}
	if s.translator == nil || !s.translator.Enabled() {
		writeErr(w, http.StatusServiceUnavailable, ErrTranslatorDisabled.Error())
		return
	}
	tr, err := s.translator.Translate(r.Context(), req.Text)
	switch {
	case err == nil:
		writeJSON(w, http.StatusOK, tr)
	case errors.Is(err, ErrNoVocabulary):
		writeErr(w, http.StatusServiceUnavailable, err.Error())
	default:
		// The gateway failed. This is deliberately a 502 and not a fallback:
		// the caller sends raw prose and records translated=false, so the
		// eval can tell those images apart from translated ones.
		writeErr(w, http.StatusBadGateway, err.Error())
	}
}

// translatorMeta is what /api/meta reports, so the app can decide whether to
// call /api/translate at all instead of finding out per generation.
func (s *server) translatorMeta() map[string]any {
	if s.translator == nil || !s.translator.Enabled() {
		return map[string]any{"enabled": false}
	}
	return map[string]any{
		"enabled": true,
		"version": s.translator.Version(),
		"vocab":   s.translator.VocabSize(),
	}
}

// sortedKeys is a small helper for deterministic output in tagbench and tests.
func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

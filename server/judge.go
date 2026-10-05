package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/png"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"golang.org/x/image/draw"
)

// The VL judge: a second rater for the relational criteria.
//
// pose_adherence, source_identity and source_style judge an output against the
// thing that conditioned it, and the eval harness can only rank as fast as a
// human clicks. A vision-language model shown the same pair — reference first,
// output second — can rate the backlog, but only where it provably agrees with
// the human. So the judge is held apart from the human labels on every axis:
//
//   - Separate table. Verdicts go to criteria_judgments, never image_criteria;
//     the human label is what the judge is calibrated against.
//   - Versioned like the translator. judge = alias@promptVersion; a new prompt
//     or a new model behind the alias is a new, uncalibrated judge.
//   - Gated per criterion (kappa.go). Verdicts reach the eval only for a
//     criterion whose κ against the human clears the gate.
//   - Blind human. The detail view shows a verdict only once the human has
//     rated that criterion; otherwise κ would measure deference.
//   - No fallback. Gateway down is an error, never a different model's verdict
//     under the same version string.

// judgePromptVersion is bumped whenever a prompt, the image preparation or the
// parsing changes — any of them changes the verdict for the same pair.
const judgePromptVersion = "judge-v1"

// judgeMaxSide is the longest edge sent to the model. Enough to read a pose
// and a face; two of them stay well inside the model's context.
const judgeMaxSide = 768

// judgeFilterVersion is the judge version the gallery's judge= filter reads.
// galleryFilter is a pure function shared by three handlers, and the version
// is fixed for the life of the process, so main sets it once at startup.
var judgeFilterVersion = ""

const judgeSystemPrompt = `You are a strict visual rater for an image-generation evaluation. You are shown two images: first the REFERENCE, then the OUTPUT that was generated from it. You judge exactly one criterion, described by the user.

Answer with ONLY a JSON object, no prose before or after:
{"verdict": "up" | "down" | "unsure", "reason": "<one short sentence>"}

- "up": the OUTPUT satisfies the criterion.
- "down": it clearly does not.
- "unsure": the images do not let you decide (subject hidden, cropped away, unreadable). Do not use "unsure" to avoid a hard call.
Judge only the criterion. Image quality, aesthetics and prompt content are irrelevant unless the criterion names them.`

// judgeRubrics is what each criterion means, in the words a human rater uses.
var judgeRubrics = map[string]string{
	"pose_adherence": `Criterion: POSE ADHERENCE.
The REFERENCE is an OpenPose skeleton (coloured limbs on black). Does the main figure in the OUTPUT hold the same body pose?
- "up": the overall body configuration matches — stance, leg placement, arm positions and torso orientation correspond to the skeleton. Small differences in hands, fingers, head tilt or camera distance are fine.
- "down": a limb is clearly in a different position (an arm raised instead of lowered, legs together instead of apart, sitting instead of standing), or the figure is not a full body where the skeleton is.`,
	"source_identity": `Criterion: SOURCE IDENTITY.
The REFERENCE is the source image the OUTPUT was edited or re-rendered from. Is the main subject of the OUTPUT recognisably the same individual?
- "up": same person/character — face shape, features, hair, distinguishing marks match; a change of pose, expression, outfit or style is fine.
- "down": the subject reads as a different individual (different face, different age or build, features swapped).`,
	"source_style": `Criterion: SOURCE STYLE.
The REFERENCE is the source image the OUTPUT was edited or re-rendered from. Does the OUTPUT keep the visual style of the REFERENCE?
- "up": same rendering style — medium (photo, painting, anime, 3D), line work, shading, palette and level of detail match; content may differ.
- "down": the style visibly changed (photo became illustration, flat colours became painterly, palette replaced).`,
}

var (
	// ErrJudgeDisabled: no gateway configured — distinct from a failing one.
	ErrJudgeDisabled = errors.New("judge disabled: LLM_GATEWAY_URL not set")
	// ErrNotEligible: the criterion cannot apply to this image (no pose, no
	// source) — the same rule the eval uses for coverage.
	ErrNotEligible = errors.New("criterion does not apply to this image")
	// ErrNoReference: it applies, but the reference cannot be loaded. Never
	// recorded as an abstention — the judge did not see the pair.
	ErrNoReference = errors.New("reference image unavailable")
	errImageGone   = errors.New("image not found")
)

// Judgment is one verdict, cached or fresh.
type Judgment struct {
	ImageID   string `json:"imageId"`
	Criterion string `json:"criterion"`
	Judge     string `json:"judge"`
	Verdict   int    `json:"verdict"` // 1 up, -1 down, 0 abstain
	Reason    string `json:"reason"`
	RefKey    string `json:"refKey"`
	Cached    bool   `json:"cached"`
}

type Judge struct {
	db         *sql.DB
	gatewayURL string
	model      string
	blobPath   func(sha string) string
	poses      fs.FS // <pose_id>.png — the SPA's own pose thumbnails
	client     *http.Client
	jobs       chan judgeJob
}

type judgeJob struct{ imageID, criterion string }

func NewJudge(db *sql.DB, gatewayURL, model string, blobPath func(string) string, poses fs.FS) *Judge {
	return &Judge{
		db:         db,
		gatewayURL: strings.TrimRight(gatewayURL, "/"),
		model:      model,
		blobPath:   blobPath,
		poses:      poses,
		client:     &http.Client{Timeout: 120 * time.Second},
		jobs:       make(chan judgeJob, 1024),
	}
}

// webPoses is the pose template set the gallery already serves at
// /poses/<id>.png. The judge reads the same files the human looks at.
func webPoses() fs.FS {
	sub, err := fs.Sub(webdist, "webdist/poses")
	if err != nil {
		panic(err)
	}
	return sub
}

func (j *Judge) Enabled() bool { return j != nil && j.gatewayURL != "" }

func (j *Judge) Version() string { return j.versionFor(j.model) }

func (j *Judge) versionFor(model string) string { return model + "@" + judgePromptVersion }

// reference loads the output image and its reference for one criterion.
func (j *Judge) reference(imageID, criterion string) (outSha, refKey string, ref image.Image, err error) {
	elig, ok := criterionEligible[criterion]
	if !ok {
		return "", "", nil, fmt.Errorf("unknown criterion %q", criterion)
	}
	var poseID, sourceID string
	var eligible bool
	err = j.db.QueryRow(`
		SELECT g.sha256, COALESCE(g.pose_id, ''), COALESCE(g.source_image_id, ''), `+elig+`
		FROM gallery g WHERE g.id = ?`, imageID).Scan(&outSha, &poseID, &sourceID, &eligible)
	if errors.Is(err, sql.ErrNoRows) {
		return "", "", nil, errImageGone
	}
	if err != nil {
		return "", "", nil, err
	}
	if !eligible {
		return "", "", nil, ErrNotEligible
	}

	switch criterion {
	case "pose_adherence":
		refKey = "pose:" + poseID
		if strings.ContainsAny(poseID, `/\`) || !fs.ValidPath(poseID+".png") {
			return "", "", nil, fmt.Errorf("%w: bad pose id %q", ErrNoReference, poseID)
		}
		b, err := fs.ReadFile(j.poses, poseID+".png")
		if err != nil {
			return "", "", nil, fmt.Errorf("%w: pose %s: %v", ErrNoReference, poseID, err)
		}
		ref, _, err = image.Decode(bytes.NewReader(b))
		if err != nil {
			return "", "", nil, fmt.Errorf("%w: pose %s: %v", ErrNoReference, poseID, err)
		}
	default: // source_*: the exact image the node was made from (GenImage.id)
		refKey = "image:" + sourceID
		var srcSha string
		err := j.db.QueryRow(`SELECT sha256 FROM images WHERE id = ? AND blob_present = 1`, sourceID).Scan(&srcSha)
		if err != nil {
			return "", "", nil, fmt.Errorf("%w: source %s: not in the gallery", ErrNoReference, sourceID)
		}
		ref, err = decodeFile(j.blobPath(srcSha))
		if err != nil {
			return "", "", nil, fmt.Errorf("%w: source %s: %v", ErrNoReference, sourceID, err)
		}
	}
	return outSha, refKey, ref, nil
}

func decodeFile(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

// jpegDataURI downscales to judgeMaxSide (never up) and encodes for the
// OpenAI-style image_url content part.
func jpegDataURI(src image.Image) (string, error) {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w > judgeMaxSide || h > judgeMaxSide {
		if w >= h {
			h, w = max(1, h*judgeMaxSide/w), judgeMaxSide
		} else {
			w, h = max(1, w*judgeMaxSide/h), judgeMaxSide
		}
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, b, draw.Src, nil)
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 88}); err != nil {
		return "", err
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

// Judge rates one image on one criterion with the default model, through the
// cache.
func (j *Judge) Judge(ctx context.Context, imageID, criterion string) (Judgment, error) {
	return j.JudgeWith(ctx, j.model, imageID, criterion, true)
}

// JudgeWith picks the model per call and makes the cache optional — judgebench
// with -no-cache measures the live model and leaves the stored verdicts alone.
func (j *Judge) JudgeWith(ctx context.Context, model, imageID, criterion string, useCache bool) (Judgment, error) {
	if !j.Enabled() {
		return Judgment{}, ErrJudgeDisabled
	}
	if _, ok := judgeRubrics[criterion]; !ok {
		return Judgment{}, fmt.Errorf("unknown criterion %q", criterion)
	}
	version := j.versionFor(model)
	if useCache {
		if jd, ok := j.cached(imageID, criterion, version); ok {
			return jd, nil
		}
	}
	outSha, refKey, ref, err := j.reference(imageID, criterion)
	if err != nil {
		return Judgment{}, err
	}
	out, err := decodeFile(j.blobPath(outSha))
	if err != nil {
		return Judgment{}, fmt.Errorf("output blob: %w", err)
	}
	raw, err := j.callGateway(ctx, model, criterion, ref, out)
	if err != nil {
		return Judgment{}, err
	}
	verdict, reason, err := parseVerdict(raw)
	if err != nil {
		return Judgment{}, err
	}
	jd := Judgment{ImageID: imageID, Criterion: criterion, Judge: version,
		Verdict: verdict, Reason: reason, RefKey: refKey}
	if useCache {
		if _, err := j.db.Exec(`
			INSERT INTO criteria_judgments (image_id, criterion, judge, verdict, reason, ref_key)
			VALUES (?, ?, ?, ?, ?, ?)
			ON CONFLICT(image_id, criterion, judge) DO UPDATE SET
			  verdict = excluded.verdict, reason = excluded.reason,
			  ref_key = excluded.ref_key, created_at = datetime('now')`,
			imageID, criterion, version, verdict, reason, refKey); err != nil {
			return Judgment{}, fmt.Errorf("store verdict: %w", err)
		}
	}
	return jd, nil
}

func (j *Judge) cached(imageID, criterion, version string) (Judgment, bool) {
	jd := Judgment{ImageID: imageID, Criterion: criterion, Judge: version, Cached: true}
	err := j.db.QueryRow(`
		SELECT verdict, reason, ref_key FROM criteria_judgments
		WHERE image_id = ? AND criterion = ? AND judge = ?`, imageID, criterion, version).
		Scan(&jd.Verdict, &jd.Reason, &jd.RefKey)
	return jd, err == nil
}

// callGateway sends one pair. Both images are named in the text right before
// them, so the model cannot confuse which is which.
func (j *Judge) callGateway(ctx context.Context, model, criterion string, ref, out image.Image) (string, error) {
	refURI, err := jpegDataURI(ref)
	if err != nil {
		return "", err
	}
	outURI, err := jpegDataURI(out)
	if err != nil {
		return "", err
	}
	type part = map[string]any
	content := []part{
		{"type": "text", "text": judgeRubrics[criterion]},
		{"type": "text", "text": "REFERENCE:"},
		{"type": "image_url", "image_url": part{"url": refURI}},
		{"type": "text", "text": "OUTPUT:"},
		{"type": "image_url", "image_url": part{"url": outURI}},
		{"type": "text", "text": "Answer with the JSON object only."},
	}
	body, _ := json.Marshal(map[string]any{
		"model": model,
		"messages": []part{
			{"role": "system", "content": judgeSystemPrompt},
			{"role": "user", "content": content},
		},
		"temperature": 0,
		"max_tokens":  200,
		"stream":      false,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		j.gatewayURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer dummy")
	resp, err := j.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("gateway: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("gateway: HTTP %d: %s", resp.StatusCode, snippet(raw))
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Choices) == 0 {
		return "", fmt.Errorf("gateway: unexpected response: %s", snippet(raw))
	}
	return parsed.Choices[0].Message.Content, nil
}

// parseVerdict reads the model's JSON back. Tolerant of the habits around it —
// a code fence, a sentence before or after, upper case — and strict about the
// verdict itself: anything but up/down/unsure is an error, not an abstention,
// because a verdict the parser guessed is not the model's verdict.
func parseVerdict(raw string) (verdict int, reason string, err error) {
	start, end := strings.Index(raw, "{"), strings.LastIndex(raw, "}")
	if start < 0 || end < start {
		return 0, "", fmt.Errorf("judge: no JSON object in reply: %s", snippet([]byte(raw)))
	}
	var out struct {
		Verdict string `json:"verdict"`
		Reason  string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(raw[start:end+1]), &out); err != nil {
		return 0, "", fmt.Errorf("judge: unparseable reply: %s", snippet([]byte(raw)))
	}
	reason = strings.TrimSpace(out.Reason)
	switch strings.ToLower(strings.TrimSpace(out.Verdict)) {
	case "up":
		return 1, reason, nil
	case "down":
		return -1, reason, nil
	case "unsure":
		return 0, reason, nil
	}
	return 0, "", fmt.Errorf("judge: verdict %q is not up/down/unsure", out.Verdict)
}

// Run works the sweep queue. Call in a goroutine. The first gateway failure
// drops the rest of the batch: the model runs in a time window, and a closed
// window would otherwise turn into a thousand logged errors.
func (j *Judge) Run() {
	for job := range j.jobs {
		_, err := j.Judge(context.Background(), job.imageID, job.criterion)
		switch {
		case err == nil, errors.Is(err, ErrNotEligible), errors.Is(err, ErrNoReference), errors.Is(err, errImageGone):
			if err != nil {
				log.Printf("judge: %s %s: %v", job.imageID, job.criterion, err)
			}
		default:
			dropped := 0
			for len(j.jobs) > 0 {
				<-j.jobs
				dropped++
			}
			log.Printf("judge: %s %s: %v — dropped %d queued", job.imageID, job.criterion, err, dropped)
		}
	}
}

// sweep queues up to n eligible images this judge has not seen yet. Which
// images get judged does not depend on any verdict, so it cannot bias the
// calibration.
func (j *Judge) sweep(criterion string, n int) (int, error) {
	elig := criterionEligible[criterion]
	rows, err := j.db.Query(`
		SELECT g.id FROM gallery g
		WHERE `+elig+` AND NOT EXISTS (SELECT 1 FROM criteria_judgments cj
		  WHERE cj.image_id = g.id AND cj.criterion = ? AND cj.judge = ?)
		ORDER BY COALESCE(g.created_at, '') DESC, g.id DESC
		LIMIT ?`, criterion, j.Version(), n)
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	queued := 0
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return queued, err
		}
		select {
		case j.jobs <- judgeJob{id, criterion}:
			queued++
		default:
			return queued, nil // queue full; the next sweep picks up the rest
		}
	}
	return queued, rows.Err()
}

// judgeStat is one criterion's standing for one judge version.
type judgeStat struct {
	Judged    int        `json:"judged"`    // verdicts stored, abstentions included
	Abstained int        `json:"abstained"` // verdict 0
	Agreement *agreement `json:"agreement"` // on images the human rated too
}

// judgeStats computes every criterion's agreement table for a judge version.
// It reads the gallery view, so images whose blob went missing drop out the
// same way they drop out of the eval.
func judgeStats(db *sql.DB, version string) (map[string]*judgeStat, error) {
	out := map[string]*judgeStat{}
	for _, c := range kCriteria {
		out[c] = &judgeStat{Agreement: &agreement{}}
	}
	rows, err := db.Query(`
		SELECT cj.criterion, cj.verdict, COALESCE(ic.score, 0)
		FROM criteria_judgments cj
		JOIN gallery g ON g.id = cj.image_id
		LEFT JOIN image_criteria ic ON ic.image_id = cj.image_id AND ic.criterion = cj.criterion
		WHERE cj.judge = ?`, version)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var criterion string
		var verdict, human int
		if err := rows.Scan(&criterion, &verdict, &human); err != nil {
			return nil, err
		}
		st, ok := out[criterion]
		if !ok {
			continue
		}
		st.Judged++
		if verdict == 0 {
			st.Abstained++
			continue
		}
		if human != 0 {
			st.Agreement.add(human, verdict)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for _, st := range out {
		st.Agreement.finish()
	}
	return out, nil
}

// passedCriteria lists the criteria whose gate is open for a judge version.
func passedCriteria(stats map[string]*judgeStat) []string {
	var out []string
	for _, c := range kCriteria {
		if st := stats[c]; st != nil && st.Agreement.Gate == gatePass {
			out = append(out, c)
		}
	}
	return out
}

// POST /api/images/{id}/judge?criterion=pose_adherence
func (s *server) handleJudgeImage(w http.ResponseWriter, r *http.Request) {
	criterion := r.URL.Query().Get("criterion")
	if !slices.Contains(kCriteria, criterion) {
		writeErr(w, http.StatusBadRequest, "criterion must be one of "+strings.Join(kCriteria, ", "))
		return
	}
	if !s.judge.Enabled() {
		writeErr(w, http.StatusServiceUnavailable, ErrJudgeDisabled.Error())
		return
	}
	jd, err := s.judge.Judge(r.Context(), r.PathValue("id"), criterion)
	switch {
	case err == nil:
		writeJSON(w, http.StatusOK, jd)
	case errors.Is(err, errImageGone):
		writeErr(w, http.StatusNotFound, err.Error())
	case errors.Is(err, ErrNotEligible):
		writeErr(w, http.StatusUnprocessableEntity, err.Error())
	case errors.Is(err, ErrNoReference):
		writeErr(w, http.StatusConflict, err.Error())
	default:
		// Gateway or reply failure. 502 and nothing stored — the same
		// no-fallback rule as the translator.
		writeErr(w, http.StatusBadGateway, err.Error())
	}
}

// POST /api/judge/sweep?criterion=pose_adherence&n=100
func (s *server) handleJudgeSweep(w http.ResponseWriter, r *http.Request) {
	criterion := r.URL.Query().Get("criterion")
	if !slices.Contains(kCriteria, criterion) {
		writeErr(w, http.StatusBadRequest, "criterion must be one of "+strings.Join(kCriteria, ", "))
		return
	}
	n := 100
	if v := r.URL.Query().Get("n"); v != "" {
		var err error
		if n, err = strconv.Atoi(v); err != nil || n < 1 || n > 1000 {
			writeErr(w, http.StatusBadRequest, "n must be 1..1000")
			return
		}
	}
	if !s.judge.Enabled() {
		writeErr(w, http.StatusServiceUnavailable, ErrJudgeDisabled.Error())
		return
	}
	queued, err := s.judge.sweep(criterion, n)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"queued": queued, "judge": s.judge.Version()})
}

// GET /api/judge — the gate board: per criterion, how much the judge has seen
// and how well it agrees with the human.
func (s *server) handleJudgeStatus(w http.ResponseWriter, r *http.Request) {
	if s.judge == nil {
		writeJSON(w, http.StatusOK, map[string]any{"enabled": false})
		return
	}
	stats, err := judgeStats(s.db, s.judge.Version())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"enabled":  s.judge.Enabled(),
		"version":  s.judge.Version(),
		"criteria": stats,
		"thresholds": map[string]any{
			"minPairs": gateMinPairs, "minHumanDown": gateMinHumanDown,
			"minKappa": gateMinKappa, "minKappaLower": gateMinKappaLower,
		},
	})
}

// judgeMeta is the /api/meta entry, after translatorMeta.
func (s *server) judgeMeta() map[string]any {
	if !s.judge.Enabled() {
		return map[string]any{"enabled": false}
	}
	return map[string]any{"enabled": true, "version": s.judge.Version()}
}

// visibleJudgments returns the judge's verdicts for one image, but only for
// criteria the human has already rated. Showing a verdict first would turn
// the human into the judge's echo and inflate κ — blindness is enforced here,
// at the API, not left to the UI.
func (s *server) visibleJudgments(imageID string, humanRated map[string]int) (map[string]Judgment, error) {
	out := map[string]Judgment{}
	if s.judge == nil {
		return out, nil
	}
	rows, err := s.db.Query(`
		SELECT criterion, verdict, reason, ref_key FROM criteria_judgments
		WHERE image_id = ? AND judge = ?`, imageID, s.judge.Version())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		jd := Judgment{ImageID: imageID, Judge: s.judge.Version(), Cached: true}
		if err := rows.Scan(&jd.Criterion, &jd.Verdict, &jd.Reason, &jd.RefKey); err != nil {
			return nil, err
		}
		if _, rated := humanRated[jd.Criterion]; rated {
			out[jd.Criterion] = jd
		}
	}
	return out, rows.Err()
}

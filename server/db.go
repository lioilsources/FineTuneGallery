package main

import (
	"database/sql"
	_ "embed"
	"fmt"
	"log"

	_ "modernc.org/sqlite"
)

//go:embed schema.sql
var schemaV1 string

// migV2 adds the three conditioning fields the app and the lab were already
// sending and this schema silently dropped: the art-style preset, the LoRA
// strength (lora_name alone cannot tell a 0.4 run from a 1.4 one), and the
// keep-the-pose flow. The gallery view has to be recreated — a view does not
// pick up columns added to its tables.
const migV2 = `
ALTER TABLE nodes ADD COLUMN style_id TEXT;
ALTER TABLE nodes ADD COLUMN lora_strength REAL;
ALTER TABLE nodes ADD COLUMN is_repose INTEGER NOT NULL DEFAULT 0;

DROP VIEW gallery;
CREATE VIEW gallery AS
SELECT i.id, i.sha256, i.idx, i.node_id, n.session_id, n.prompt,
       n.positive_prefix, n.model_id, n.lora_name, n.lora_strength,
       n.style_id, n.pose_id, n.is_repose, n.seed,
       n.created_at, n.parent_id, n.source_image_id, n.origin,
       COALESCE(r.score, 0)     AS score,
       COALESCE(r.critique, '') AS critique
FROM images i
JOIN nodes n ON n.id = i.node_id
LEFT JOIN ratings r ON r.image_id = i.id
WHERE i.blob_present = 1;

CREATE INDEX nodes_style ON nodes(style_id);
`

// migV3 adds the render-medium axis. The app's art-style blocks name a medium
// of their own ("woodblock print", "stone relief") while the prompt never
// declares one, so the two compete and the lab's style matrix recorded the
// result: the traditions whose whole claim is a medium ("assyrian stone
// relief in earth tones") collapse to a beige wall on most models. Making the
// medium an explicit, front-loaded axis is the hypothesis; this column is how
// the eval harness gets to judge it, by grouping the same style with and
// without one.
//
// The gallery view has to be recreated — a view does not pick up columns added
// to its tables.
const migV3 = `
ALTER TABLE nodes ADD COLUMN medium_id TEXT;

DROP VIEW gallery;
CREATE VIEW gallery AS
SELECT i.id, i.sha256, i.idx, i.node_id, n.session_id, n.prompt,
       n.positive_prefix, n.model_id, n.lora_name, n.lora_strength,
       n.style_id, n.medium_id, n.pose_id, n.is_repose, n.seed,
       n.created_at, n.parent_id, n.source_image_id, n.origin,
       COALESCE(r.score, 0)     AS score,
       COALESCE(r.critique, '') AS critique
FROM images i
JOIN nodes n ON n.id = i.node_id
LEFT JOIN ratings r ON r.image_id = i.id
WHERE i.blob_present = 1;

CREATE INDEX nodes_medium ON nodes(medium_id);
`

// migV4 is the prompt translator's footprint.
//
// The booru-trained SDXL checkpoints (Pony, Illustrious, NoobAI, Animagine)
// read Danbooru tags; the app sends them prose. Translating prose into tags
// with an LLM is only defensible if the translation is deterministic and
// attributable, so the gallery does it in one place and remembers every
// answer: `translations` is keyed by a hash of (text, language, translator
// version), which makes the app, the lab and the benchmark all receive the
// same tags for the same input.
//
// On the node, prompt_sent is what actually reached the model (the user's
// text stays in prompt so the intent is never lost) and translator names the
// version that produced it, NULL when none did. The eval harness groups on
// translator, so "does translation help" is a measured question rather than
// an opinion. There is no separate translated flag: it would be a second
// column for the fact translator already carries.
const migV4 = `
CREATE TABLE translations (
  hash        TEXT PRIMARY KEY,
  text        TEXT NOT NULL,
  language    TEXT NOT NULL,
  translator  TEXT NOT NULL,
  tags        TEXT NOT NULL,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

ALTER TABLE nodes ADD COLUMN prompt_sent TEXT;
ALTER TABLE nodes ADD COLUMN translator TEXT;

DROP VIEW gallery;
CREATE VIEW gallery AS
SELECT i.id, i.sha256, i.idx, i.node_id, n.session_id, n.prompt,
       n.prompt_sent, n.translator,
       n.positive_prefix, n.model_id, n.lora_name, n.lora_strength,
       n.style_id, n.medium_id, n.pose_id, n.is_repose, n.seed,
       n.created_at, n.parent_id, n.source_image_id, n.origin,
       COALESCE(r.score, 0)     AS score,
       COALESCE(r.critique, '') AS critique
FROM images i
JOIN nodes n ON n.id = i.node_id
LEFT JOIN ratings r ON r.image_id = i.id
WHERE i.blob_present = 1;

CREATE INDEX nodes_translator ON nodes(translator);
`

// migV5 is the second rater's footprint: a VLM judging the relational
// criteria against the same reference a human sees (pose template, img2img
// source).
//
// Its verdicts live here and never in image_criteria. The human label is the
// ground truth the judge is calibrated against; writing the judge into that
// table would make it grade its own homework. judge is alias@promptVersion,
// so a new prompt or model starts an uncalibrated column instead of mixing
// into the old one. verdict 0 is an abstention ("cannot tell") — it counts
// towards coverage, never towards agreement or the eval. ref_key records what
// the judge was shown, so a verdict can always be traced to its reference.
const migV5 = `
CREATE TABLE criteria_judgments (
  image_id   TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  criterion  TEXT NOT NULL,
  judge      TEXT NOT NULL,
  verdict    INTEGER NOT NULL CHECK(verdict IN (-1, 0, 1)),
  reason     TEXT NOT NULL DEFAULT '',
  ref_key    TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (image_id, criterion, judge)
);
CREATE INDEX criteria_judgments_judge ON criteria_judgments(judge, criterion);
`

// migV6 is the content taxonomy's footprint (taxo/taxonomy.yaml, UC25).
//
// image_taxonomy holds every evaluator's answer per node, one row per value
// (a multi-label node is several rows), next to the consensus and the human
// label — never instead of them, for the same reason criteria_judgments is
// not image_criteria: the human label is what the evaluators are calibrated
// against. taxonomy_version is stored per row so a vocabulary change starts a
// new column of answers instead of mixing into the old one.
//
// image_temperature is the rating ensemble. band is stored (not derived per
// query) so the gallery filter stays a plain comparison; a threshold change
// rewrites it in one UPDATE, and that change is itself audited. disagree is
// the ensemble's own unsure, unsure adds the conformal explicit guard.
//
// image_embeddings is the frozen SigLIP 2 vector the multi-head classifier and
// "similar images" read. modernc/sqlite cannot load sqlite-vec (no C
// extensions), so it is a float32 BLOB with a brute-force scan in Go: 20k
// images × 1152 dims is ~25 MFLOP, milliseconds.
//
// taxonomy_blocks is the minor rule: an image here is out of the gallery view
// — and so out of the list, the eval, every dataset — and its blob is not
// served. "quarantine" is reversible (only WD14 had spoken), "block" is not.
// The view checks by sha256, so a re-ingested copy of the same blob stays out.
// taxonomy_audit has no foreign key on purpose: it outlives the image.
const migV6 = `
CREATE TABLE image_taxonomy (
  image_id         TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  node             TEXT NOT NULL,
  evaluator        TEXT NOT NULL,  -- wd14 | siglip | vlm | head | consensus | human
  value            TEXT NOT NULL,  -- '' = consensus row with no agreed value
  conf             REAL,
  unsure           INTEGER NOT NULL DEFAULT 0,
  taxonomy_version INTEGER NOT NULL,
  updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (image_id, node, evaluator, value)
);
CREATE INDEX image_taxonomy_lookup ON image_taxonomy(evaluator, node, value);

CREATE TABLE image_temperature (
  image_id         TEXT PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  temperature      REAL NOT NULL CHECK(temperature >= 0 AND temperature <= 1),
  band             TEXT NOT NULL CHECK(band IN ('safe', 'suggestive', 'nsfw', 'explicit')),
  disagree         INTEGER NOT NULL DEFAULT 0,
  unsure           INTEGER NOT NULL DEFAULT 0,
  parts_json       TEXT NOT NULL DEFAULT '{}',
  taxonomy_version INTEGER NOT NULL,
  updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX image_temperature_band ON image_temperature(band, unsure);

CREATE TABLE image_embeddings (
  image_id   TEXT PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  model      TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,          -- little-endian float32 × dim, L2-normalised
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE temperature_bands (
  id             INTEGER PRIMARY KEY,
  safe_max       REAL NOT NULL,
  suggestive_max REAL NOT NULL,
  nsfw_max       REAL NOT NULL,
  explicit_guard REAL,               -- conformal recall guard; NULL = none
  source         TEXT NOT NULL,      -- default | pseudo:… | human | manual
  fit_json       TEXT,
  created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE taxonomy_blocks (
  image_id   TEXT PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  state      TEXT NOT NULL CHECK(state IN ('quarantine', 'block')),
  reason     TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE taxonomy_audit (
  id       INTEGER PRIMARY KEY,
  at       TEXT NOT NULL DEFAULT (datetime('now')),
  kind     TEXT NOT NULL,            -- block | quarantine | release | thresholds
  image_id TEXT,
  sha256   TEXT,
  detail   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX taxonomy_audit_at ON taxonomy_audit(at);

-- The human labelling sample (taxobench -sample): stratified, seeded, chosen
-- before anyone sees an evaluator's answer.
CREATE TABLE taxonomy_label_queue (
  image_id   TEXT PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  stratum    TEXT NOT NULL,
  seed       INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Highest temperature a kohya export may contain; NULL = no limit.
ALTER TABLE datasets ADD COLUMN temp_max REAL;

DROP VIEW gallery;
CREATE VIEW gallery AS
SELECT i.id, i.sha256, i.idx, i.node_id, n.session_id, n.prompt,
       n.prompt_sent, n.translator,
       n.positive_prefix, n.model_id, n.lora_name, n.lora_strength,
       n.style_id, n.medium_id, n.pose_id, n.is_repose, n.seed,
       n.created_at, n.parent_id, n.source_image_id, n.origin,
       COALESCE(r.score, 0)     AS score,
       COALESCE(r.critique, '') AS critique
FROM images i
JOIN nodes n ON n.id = i.node_id
LEFT JOIN ratings r ON r.image_id = i.id
WHERE i.blob_present = 1
  AND NOT EXISTS (SELECT 1 FROM taxonomy_blocks tb
                  JOIN images bi ON bi.id = tb.image_id
                  WHERE bi.sha256 = i.sha256);
`

// openDB opens (creating if needed) the SQLite database and applies pending
// migrations. Single-user service: one *sql.DB with WAL is all we need.
func openDB(path string) (*sql.DB, error) {
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=foreign_keys(ON)", path)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	// modernc/sqlite serializes writes; a small pool avoids SQLITE_BUSY churn.
	db.SetMaxOpenConns(4)
	if err := migrate(db); err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

type mig struct {
	v   int
	sql string
}

var migrations = []mig{
	{1, schemaV1},
	{2, migV2},
	{3, migV3},
	{4, migV4},
	{5, migV5},
	{6, migV6},
}

// schemaVersion is the version openDB brings a database up to.
func schemaVersion() int { return migrations[len(migrations)-1].v }

func migrate(db *sql.DB) error {
	var version int
	if err := db.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		return fmt.Errorf("read user_version: %w", err)
	}
	migs := migrations
	for _, m := range migs {
		if version >= m.v {
			continue
		}
		log.Printf("db: migrating to schema v%d", m.v)
		tx, err := db.Begin()
		if err != nil {
			return err
		}
		if _, err := tx.Exec(m.sql); err != nil {
			tx.Rollback()
			return fmt.Errorf("migration v%d: %w", m.v, err)
		}
		if _, err := tx.Exec(fmt.Sprintf("PRAGMA user_version = %d", m.v)); err != nil {
			tx.Rollback()
			return fmt.Errorf("set user_version v%d: %w", m.v, err)
		}
		if err := tx.Commit(); err != nil {
			return err
		}
	}
	return nil
}

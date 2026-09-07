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

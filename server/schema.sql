-- FINETUNE gallery schema (SQLite). Applied via PRAGMA user_version migration
-- in db.go — this file is schema version 1 and must stay idempotent-free
-- (plain CREATEs); later migrations append ALTERs in db.go.

CREATE TABLE sessions (
  id                TEXT PRIMARY KEY,             -- ImageSession.id from the app
  title             TEXT NOT NULL,
  model_id          TEXT NOT NULL,                -- session-level (latest) model
  app_updated_at    TEXT,
  first_ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
  last_ingested_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE nodes (
  id              TEXT PRIMARY KEY,               -- GenNode.id
  session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  parent_id       TEXT,                           -- null = root
  source_image_id TEXT,                           -- exact parent image edited (img2img)
  prompt          TEXT NOT NULL,
  origin          TEXT NOT NULL DEFAULT 'generated', -- 'generated' | 'upload'
  model_id        TEXT,                           -- per-node snapshot (new app builds)
  lora_name       TEXT,
  pose_id         TEXT,
  seed            INTEGER,
  negative_prompt TEXT,
  positive_prefix TEXT,
  width           INTEGER,
  height          INTEGER,
  steps           INTEGER,
  cfg             REAL,
  denoise         REAL,
  sampler_name    TEXT,
  scheduler       TEXT,
  created_at      TEXT
);
CREATE INDEX nodes_session ON nodes(session_id);

CREATE TABLE images (
  id           TEXT PRIMARY KEY,                  -- GenImage.id
  node_id      TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  idx          INTEGER NOT NULL DEFAULT 0,        -- batch index (seed = node.seed [+idx on NIM])
  sha256       TEXT NOT NULL,
  size         INTEGER,
  blob_present INTEGER NOT NULL DEFAULT 0,
  UNIQUE(node_id, idx)
);
CREATE INDEX images_sha ON images(sha256);
CREATE INDEX images_node ON images(node_id);

-- Content-addressed blob index; the PNG lives at images/<sha[0:2]>/<sha>.png.
CREATE TABLE blobs (
  sha256     TEXT PRIMARY KEY,
  size       INTEGER NOT NULL,
  width      INTEGER,
  height     INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE ratings (
  image_id   TEXT PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  score      INTEGER NOT NULL DEFAULT 0 CHECK(score IN (-1, 0, 1)),
  critique   TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE aspects (
  id           INTEGER PRIMARY KEY,
  name         TEXT NOT NULL UNIQUE,
  trigger_word TEXT NOT NULL DEFAULT '',
  builtin      INTEGER NOT NULL DEFAULT 0
);
INSERT INTO aspects (name, trigger_word, builtin) VALUES
  ('hair', '', 1),
  ('jewelry', '', 1),
  ('lingerie', '', 1),
  ('pose', '', 1),
  ('environment', '', 1),
  ('face', '', 1),
  ('hands', '', 1),
  ('composition', '', 1),
  ('style', '', 1);

CREATE TABLE image_aspects (
  image_id  TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  aspect_id INTEGER NOT NULL REFERENCES aspects(id) ON DELETE CASCADE,
  PRIMARY KEY (image_id, aspect_id)
);

-- Relational criteria: output judged against its conditioning input
-- (pose template / img2img source), not the image alone. Absent row = unrated.
CREATE TABLE image_criteria (
  image_id  TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  criterion TEXT NOT NULL CHECK(criterion IN ('pose_adherence', 'source_identity', 'source_style')),
  score     INTEGER NOT NULL CHECK(score IN (-1, 1)),
  PRIMARY KEY (image_id, criterion)
);

-- Caption layers. 'auto' (WD14) is never overwritten by edits; 'human' and
-- 'refined' (LLM) are separate rows. Effective caption = human > refined > auto.
CREATE TABLE captions (
  image_id   TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL CHECK(kind IN ('auto', 'human', 'refined')),
  text       TEXT NOT NULL,
  tagger     TEXT,                                -- model name for kind='auto'
  tags_json  TEXT,                                -- raw {tag: confidence} for kind='auto'
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (image_id, kind)
);

CREATE TABLE datasets (
  id           INTEGER PRIMARY KEY,
  name         TEXT NOT NULL UNIQUE,
  aspect_id    INTEGER REFERENCES aspects(id),
  base_model   TEXT NOT NULL,                     -- app model id (pony | illustrious-xl | …)
  trigger_word TEXT NOT NULL,
  repeats      INTEGER NOT NULL DEFAULT 10,
  status       TEXT NOT NULL DEFAULT 'draft',     -- draft | built (v2: queued | training | done)
  config_json  TEXT,                              -- optional kohya overrides
  built_at     TEXT,
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE dataset_items (
  dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  image_id   TEXT NOT NULL REFERENCES images(id),
  caption    TEXT NOT NULL DEFAULT '',            -- frozen snapshot at build time
  PRIMARY KEY (dataset_id, image_id)
);

CREATE TABLE ingest_log (
  id         INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  at         TEXT NOT NULL DEFAULT (datetime('now')),
  nodes      INTEGER,
  images     INTEGER,
  new_blobs  INTEGER
);

-- The FLAT gallery: one row per downloadable generated image.
CREATE VIEW gallery AS
SELECT i.id, i.sha256, i.idx, i.node_id, n.session_id, n.prompt,
       n.positive_prefix, n.model_id, n.lora_name, n.pose_id, n.seed,
       n.created_at, n.parent_id, n.source_image_id, n.origin,
       COALESCE(r.score, 0)    AS score,
       COALESCE(r.critique, '') AS critique
FROM images i
JOIN nodes n ON n.id = i.node_id
LEFT JOIN ratings r ON r.image_id = i.id
WHERE i.blob_present = 1;

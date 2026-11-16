package main

import (
	"database/sql"
	"strings"
)

// Lineage of a node: every node it was derived from, over parent_id and over
// the node of the exact source image (img2img). It walks nodes, not images
// with a blob, so an ancestor whose PNG never arrived still counts: an upload
// at the root stays visible even when the chain in the detail cannot show it.
//
// lineageAnc is the recursive CTE body for the node given by the SQL
// expression start; UNION (not UNION ALL) makes a cycle terminate.
func lineageAnc(start string) string {
	return `anc(id, parent, src, origin) AS (
		SELECT n.id, n.parent_id, n.source_image_id, n.origin FROM nodes n WHERE n.id = ` + start + `
		UNION
		SELECT p.id, p.parent_id, p.source_image_id, p.origin FROM anc a
		JOIN nodes p ON p.id = a.parent
		             OR p.id = (SELECT si.node_id FROM images si WHERE si.id = a.src))`
}

// lineageBroken: some step points at a node or source image the gallery does
// not have, so nothing certain can be said about what lies beyond it.
const lineageBroken = `EXISTS (SELECT 1 FROM anc a WHERE
		(a.parent IS NOT NULL AND NOT EXISTS (SELECT 1 FROM nodes WHERE id = a.parent))
		OR (a.src IS NOT NULL AND NOT EXISTS (SELECT 1 FROM images WHERE id = a.src)))`

// lineageTextCond is the lineage=text filter: a complete lineage made only of
// generated nodes, i.e. no uploaded image anywhere upstream.
func lineageTextCond() string {
	return `NOT EXISTS (WITH RECURSIVE ` + lineageAnc("g.node_id") + `
		SELECT 1 FROM anc WHERE origin != 'generated' OR ` + lineageBroken + `)`
}

// lineageUploadCond is lineage=upload: some node upstream (or the image's own)
// is an upload.
func lineageUploadCond() string {
	return `EXISTS (WITH RECURSIVE ` + lineageAnc("g.node_id") + `
		SELECT 1 FROM anc WHERE origin = 'upload')`
}

type lineageInfo struct {
	RootNodeID *string // top of the parent_id chain that exists in the gallery
	RootOrigin *string // origin of the root; nil when the chain is broken
	HasUpload  bool
	Complete   bool
}

// lineages computes lineageInfo for a set of node ids in one query.
func lineages(db *sql.DB, nodeIDs []string) (map[string]lineageInfo, error) {
	out := map[string]lineageInfo{}
	if len(nodeIDs) == 0 {
		return out, nil
	}
	ph := make([]string, len(nodeIDs))
	args := make([]any, len(nodeIDs))
	for i, id := range nodeIDs {
		ph[i], args[i] = "?", id
	}
	// The root follows parent_id only (the tree in the app); upload and
	// completeness look at every edge, the source image's node included.
	rows, err := db.Query(`
		SELECT x.id,
		  (WITH RECURSIVE up(id, parent) AS (
		     SELECT x.id, x.parent_id
		     UNION
		     SELECT p.id, p.parent_id FROM up JOIN nodes p ON p.id = up.parent)
		   SELECT id FROM up WHERE parent IS NULL
		      OR NOT EXISTS (SELECT 1 FROM nodes WHERE id = up.parent) LIMIT 1),
		  (WITH RECURSIVE `+lineageAnc("x.id")+`
		   SELECT EXISTS (SELECT 1 FROM anc WHERE origin = 'upload')),
		  (WITH RECURSIVE `+lineageAnc("x.id")+`
		   SELECT NOT `+lineageBroken+`)
		FROM nodes x WHERE x.id IN (`+strings.Join(ph, ",")+`)`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var id string
		var li lineageInfo
		if err := rows.Scan(&id, &li.RootNodeID, &li.HasUpload, &li.Complete); err != nil {
			return nil, err
		}
		out[id] = li
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	// Root origin: only for a root that really is one (parent_id NULL).
	for id, li := range out {
		if li.RootNodeID == nil {
			continue
		}
		var origin string
		var parent *string
		if err := db.QueryRow(`SELECT origin, parent_id FROM nodes WHERE id = ?`, *li.RootNodeID).
			Scan(&origin, &parent); err != nil {
			return nil, err
		}
		if parent == nil {
			li.RootOrigin = &origin
		}
		out[id] = li
	}
	return out, nil
}

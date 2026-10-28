package main

import (
	"database/sql"
	"encoding/binary"
	"math"
	"sync"
)

// embIndex keeps every SigLIP embedding in memory for "similar images":
// 20k × 1152 float32 is ~90 MB, loaded on first use and dropped on import.
// Only one model's vectors are comparable — the most common model wins and the
// rest are ignored until re-embedded.
var embIndex = &embCache{}

type embCache struct {
	mu  sync.Mutex
	idx *embeddingIndex
}

func (c *embCache) invalidate() {
	c.mu.Lock()
	c.idx = nil
	c.mu.Unlock()
}

func (c *embCache) get(db *sql.DB) (*embeddingIndex, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.idx != nil {
		return c.idx, nil
	}
	var model string
	var dim int
	err := db.QueryRow(`SELECT model, dim FROM image_embeddings GROUP BY model, dim
		ORDER BY COUNT(*) DESC LIMIT 1`).Scan(&model, &dim)
	if err == sql.ErrNoRows {
		c.idx = &embeddingIndex{pos: map[string]int{}}
		return c.idx, nil
	}
	if err != nil {
		return nil, err
	}
	rows, err := db.Query(`SELECT image_id, vec FROM image_embeddings WHERE model = ? AND dim = ?`, model, dim)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	idx := &embeddingIndex{pos: map[string]int{}, dim: dim}
	for rows.Next() {
		var id string
		var blob []byte
		if err := rows.Scan(&id, &blob); err != nil {
			return nil, err
		}
		if len(blob) != 4*dim {
			continue
		}
		idx.pos[id] = len(idx.ids)
		idx.ids = append(idx.ids, id)
		for i := 0; i < dim; i++ {
			idx.vecs = append(idx.vecs, math.Float32frombits(binary.LittleEndian.Uint32(blob[4*i:])))
		}
	}
	c.idx = idx
	return idx, rows.Err()
}

// encodeVec is the storage format, for tests and tools.
func encodeVec(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

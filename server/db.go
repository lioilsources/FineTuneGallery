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

func migrate(db *sql.DB) error {
	var version int
	if err := db.QueryRow("PRAGMA user_version").Scan(&version); err != nil {
		return fmt.Errorf("read user_version: %w", err)
	}
	type mig struct {
		v   int
		sql string
	}
	migs := []mig{
		{1, schemaV1},
		// Future: {2, "ALTER TABLE …"},
	}
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

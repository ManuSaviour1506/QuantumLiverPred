-- V11 new-evaluation benchmark runs (kept separate from the historical metrics recorded in the model cards).
-- migrate:up
CREATE TABLE benchmark_runs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    dataset TEXT NOT NULL,
    split TEXT,
    max_slices INTEGER,
    model_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    n_slices INTEGER,
    result TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX ix_benchmark_runs_created ON benchmark_runs (created_at);
-- migrate:down
DROP TABLE benchmark_runs;

-- Four-paradigm comparison: uploads (validated inputs), runs, and one row per model per run (benchmark data).
-- migrate:up
CREATE TABLE comparison_uploads (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    input_name TEXT,
    input_sha256 TEXT NOT NULL,
    input_format TEXT,
    input_bytes INTEGER,
    modality TEXT NOT NULL DEFAULT 'CT',
    source TEXT NOT NULL DEFAULT 'upload',
    sample_id TEXT,
    context TEXT NOT NULL DEFAULT '{}',
    analysis TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    expires_at TEXT
);
CREATE INDEX ix_comparison_uploads_user ON comparison_uploads (user_id, created_at);

CREATE TABLE comparison_runs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    upload_id TEXT NOT NULL REFERENCES comparison_uploads(id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_models TEXT NOT NULL DEFAULT '[]',
    n_models INTEGER,
    n_completed INTEGER,
    n_failed INTEGER,
    wall_ms REAL,
    fastest_family TEXT,
    slowest_family TEXT,
    majority_label TEXT,
    agreement_pct REAL,
    summary TEXT NOT NULL DEFAULT '{}',
    result TEXT NOT NULL DEFAULT '{}',
    hardware TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX ix_comparison_runs_user ON comparison_runs (user_id, created_at);

CREATE TABLE comparison_model_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES comparison_runs(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    model_version TEXT,
    family TEXT NOT NULL,
    status TEXT NOT NULL,
    label TEXT,
    probability REAL,
    threshold REAL,
    confidence REAL,
    queue_ms REAL,
    features_ms REAL,
    inference_ms REAL,
    total_ms REAL,
    cold_start INTEGER,
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX ix_cmr_run ON comparison_model_results (run_id);
CREATE INDEX ix_cmr_model ON comparison_model_results (model_id, status, created_at);
-- migrate:down
DROP TABLE comparison_model_results;
DROP TABLE comparison_runs;
DROP TABLE comparison_uploads;

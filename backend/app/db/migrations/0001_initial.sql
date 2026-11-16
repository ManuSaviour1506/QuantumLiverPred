-- Initial schema: users and roles, authentication tokens, datasets, models, training runs, experiments,
-- predictions, prediction artifacts, background jobs and audit logs.
-- Portable between SQLite and PostgreSQL: TEXT ids and ISO-8601 UTC timestamps, INTEGER booleans, TEXT JSON.
-- Large files (images, model weights) are never stored in rows; rows hold storage keys and SHA-256 hashes.

-- migrate:up
CREATE TABLE roles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT ''
);
INSERT INTO roles (id, name, description) VALUES ('role_admin', 'admin', 'Full access: user management, all records, model registry synchronisation');
INSERT INTO roles (id, name, description) VALUES ('role_researcher', 'researcher', 'Run predictions, upload datasets and run experiments; sees own records');
INSERT INTO roles (id, name, description) VALUES ('role_viewer', 'viewer', 'Read-only: models, analytics, research results and own predictions');

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    role_id TEXT NOT NULL REFERENCES roles(id),
    is_active INTEGER NOT NULL DEFAULT 1,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    last_login_at TEXT,
    preferences TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE refresh_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    family_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    replaced_by TEXT
);
CREATE INDEX ix_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX ix_refresh_tokens_family ON refresh_tokens(family_id);

CREATE TABLE token_denylist (
    jti TEXT PRIMARY KEY,
    expires_at TEXT NOT NULL
);

CREATE TABLE password_reset_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT
);

CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    owner_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    visibility TEXT NOT NULL DEFAULT 'private',
    source TEXT NOT NULL,
    license TEXT NOT NULL DEFAULT '',
    provenance TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE dataset_versions (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    storage_key TEXT,
    content_sha256 TEXT,
    n_files INTEGER NOT NULL DEFAULT 0,
    manifest_columns TEXT NOT NULL DEFAULT '[]',
    validation TEXT NOT NULL DEFAULT '{}',
    statistics TEXT NOT NULL DEFAULT '{}',
    preprocessing TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    created_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    validated_at TEXT,
    UNIQUE (dataset_id, version)
);

CREATE TABLE dataset_files (
    id TEXT PRIMARY KEY,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    row_index INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    label TEXT,
    patient_id TEXT,
    scan_id TEXT,
    volume_id TEXT,
    slice_id TEXT,
    modality TEXT,
    mask_path TEXT,
    split TEXT,
    data_source TEXT,
    sha256 TEXT,
    width INTEGER,
    height INTEGER,
    file_status TEXT NOT NULL DEFAULT 'unchecked',
    issues TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX ix_dataset_files_version ON dataset_files(dataset_version_id, row_index);

CREATE TABLE models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    family TEXT NOT NULL,
    architecture TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE model_versions (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    deployment_status TEXT NOT NULL,
    training_status TEXT,
    training_date TEXT,
    dataset_version TEXT,
    execution_mode TEXT,
    artifact_path TEXT,
    artifact_sha256 TEXT,
    size_bytes INTEGER,
    metrics TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    registered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (model_id, version)
);

CREATE TABLE training_runs (
    id TEXT PRIMARY KEY,
    model_id TEXT,
    model_version TEXT,
    run_name TEXT NOT NULL,
    status TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',
    seed INTEGER,
    dataset_version TEXT,
    data_split TEXT NOT NULL DEFAULT '{}',
    metrics TEXT NOT NULL DEFAULT '{}',
    environment TEXT NOT NULL DEFAULT '{}',
    artifact_path TEXT,
    log_path TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_by TEXT
);

CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    owner_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',
    results TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    dataset_version_id TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE predictions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    model_version TEXT,
    dataset_version TEXT,
    execution_mode TEXT,
    status TEXT NOT NULL,
    label TEXT,
    probability REAL,
    threshold REAL,
    result TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    input_name TEXT,
    input_sha256 TEXT,
    input_format TEXT,
    input_bytes INTEGER,
    processing_ms REAL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    expires_at TEXT
);
CREATE INDEX ix_predictions_user_created ON predictions(user_id, created_at);
CREATE INDEX ix_predictions_model ON predictions(model_id);
CREATE INDEX ix_predictions_status ON predictions(status);

CREATE TABLE prediction_artifacts (
    id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    mime TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX ix_prediction_artifacts_prediction ON prediction_artifacts(prediction_id);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    claimed_by TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX ix_jobs_status_created ON jobs(status, created_at);

CREATE TABLE audit_logs (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    ip_hash TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX ix_audit_logs_created ON audit_logs(created_at);
CREATE INDEX ix_audit_logs_user ON audit_logs(user_id);

-- migrate:down
DROP TABLE audit_logs;
DROP TABLE jobs;
DROP TABLE prediction_artifacts;
DROP TABLE predictions;
DROP TABLE experiments;
DROP TABLE training_runs;
DROP TABLE model_versions;
DROP TABLE models;
DROP TABLE dataset_files;
DROP TABLE dataset_versions;
DROP TABLE datasets;
DROP TABLE password_reset_tokens;
DROP TABLE token_denylist;
DROP TABLE refresh_tokens;
DROP TABLE users;
DROP TABLE roles;

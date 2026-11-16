# Security and privacy

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

This is a research platform. It has had no external security review or penetration test, and it is not certified for
handling identifiable patient data (HIPAA, GDPR special-category data or similar). Do not upload identifiable patient
images. If you must process real clinical images, de-identify them first and deploy under your institution's controls.

## Implemented controls

| Area | Implementation | Tested by |
|---|---|---|
| Passwords | Argon2id (m = 19 MiB, t = 2, p = 1; PHC string), constant-time verify, dummy verify for unknown e-mails; ≥ 10 characters, two character classes, common-password list | `backend/tests/test_auth.py`, `test_units.py` |
| Sessions | JWT HS256 access tokens (15 min, `iss`, `jti`), opaque refresh tokens (stored as SHA-256) rotated on every use; re-use of an old refresh token revokes the whole token family; logout adds the access-token `jti` to a deny list | `test_auth.py` |
| Brute force | per-IP rate limits (sign-in / registration / password reset 10/min, token refresh 60/min in its own bucket since V11 — refresh tokens are single-use and rotated, and every page load refreshes —, predictions 30/min per user, other endpoints 300/min; all configurable), account lock-out after 5 failures for 15 min | `test_auth.py` |
| Password reset | single-use, 1-hour tokens (stored hashed), same answer for known and unknown addresses, all sessions revoked after reset; disabled (503) unless mail is configured | `test_auth.py` |
| Authorisation | roles `admin`, `researcher`, `viewer`; every endpoint declares public / any user / roles; users see only their own predictions (admins all); viewers cannot run predictions | `test_auth.py`, `test_predictions.py` |
| Uploads | size limits enforced while streaming (before the body is fully read) and per file; content detected from bytes and fully decoded before anything is stored; storage keys are generated (never derived from file names); dataset archives: no absolute paths, `..`, symlinks, size / count limits | `test_predictions.py`, `test_datasets.py` |
| Stored images | outside the web root; served only through authenticated endpoints that check ownership; deleted with the prediction; automatic expiry after `RETENTION_DAYS` (`manage purge-expired`) | `test_predictions.py` |
| Training use | uploaded images are never used for training; datasets are used only when an administrator starts a run on them | code review |
| Errors | uniform JSON errors with a request id; no stack traces, server paths or secrets in responses | `test_system.py` |
| Logs | JSON; passwords, tokens and authorisation headers are redacted; IP addresses are stored as keyed hashes in the audit log; uploaded file names are not logged | `test_units.py` |
| Headers | CSP (`script-src 'self'`, `frame-ancestors 'none'`, `object-src 'none'`), `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, COOP, HSTS in production, `Cache-Control: no-store` on the API | `test_system.py`, E2E (no CSP violations) |
| Secrets | none in the code or the ZIP; production refuses to start without `JWT_SECRET` (≥ 32 chars) and `DATABASE_URL`, or with synthetic models enabled; `.env` is git- and docker-ignored | `test_system.py`, deployment simulation |
| Model artifacts | SHA-256 verified against the model card before loading; NumPy arrays loaded with `allow_pickle=False`; scikit-learn models (joblib) are loaded only from the project's own `artifacts/` folder, never from uploads | `tests/test_v10_engine.py` |
| Audit | sign-in (success and failure), registration, logout, refresh-token reuse, password changes and resets, admin changes to users, predictions (created, deleted, expired), dataset upload / registration / validation / deletion and experiment creation are recorded in `audit_logs` | `test_models_experiments_analytics.py` |

## Operator responsibilities

* Terminate TLS in front of the API; set `CORS_ORIGINS` to your exact origins.
* Keep `JWT_SECRET` and database credentials in the platform's secret store; rotate `JWT_SECRET` to end all sessions.
* Set `FORWARDED_ALLOW_IPS` to your proxy's address range only (see `DEPLOYMENT.md`).
* Back up the database and storage together; restrict access to both.
* Review `RETENTION_DAYS` against your data-protection requirements and schedule `purge-expired`.
* Keep dependencies patched (`pip list --outdated`, `npm outdated`).

## Reporting

Report suspected vulnerabilities privately to the project owner; do not open public issues containing exploit details.

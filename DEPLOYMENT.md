# Deployment

> This platform is intended for academic research and software experimentation only. It has not been clinically
> validated and must not be used for diagnosis, treatment decisions or emergency medical decisions. Model outputs are
> experimental predictions and require independent clinical validation.

Three supported layouts:

| Layout | Files | When |
|---|---|---|
| A. Docker Compose (PostgreSQL + API + worker + nginx) | `docker-compose.yml`, `docker/backend.Dockerfile`, `docker/frontend.Dockerfile`, `docker/nginx.conf` | one server / VM, local demo |
| B. Render (single Docker web service serving API + SPA, managed PostgreSQL) | `render.yaml`, `docker/allinone.Dockerfile` | managed hosting, same origin |
| C. Vercel frontend + API elsewhere (e.g. layout B) | `frontend/vercel.json` | static SPA on a CDN, API proxied |

**What was tested in the build environment** (no Docker daemon and no cloud accounts there): `docker compose config`
validates; the container start sequence was reproduced without Docker — same file layout (`/srv` tree), same
`PYTHONPATH`, `ENVIRONMENT=production`, PostgreSQL 16, `python -m app.db.migrate upgrade`, the exact `uvicorn` CMD
string and the `HEALTHCHECK` command — and `scripts/smoke_test_deployment.py` passed against it (health, readiness,
SPA with security headers, register / login, model list without synthetic models, a real prediction on a CT slice,
PDF report, deletion, logout). The images themselves were **not built** and Render / Vercel deployments were **not
performed**; run the smoke test after your first deployment.

## Before any deployment

1. Restore the model weights (`INSTALLATION.md` §2) — the images copy `artifacts/` as-is.
2. Create `.env` from `.env.example`. Never commit it. Required in production:
   * `JWT_SECRET` — ≥ 32 random characters (`python -c "import secrets;print(secrets.token_urlsafe(48))"`);
   * `DATABASE_URL` — PostgreSQL;
   * `ALLOW_SYNTHETIC_MODELS` must be `false` (the API refuses to start otherwise).
3. Decide retention: `RETENTION_DAYS` (default 30). Schedule `python -m app.manage purge-expired` daily (cron, a
   Render cron job, or a systemd timer); it deletes expired uploads, generated images and prediction records.
4. Decide registration: `ALLOW_REGISTRATION=false` for closed deployments; create accounts with
   `python -m app.manage create-admin`.

## Command reference

| Task | Command in this project |
|---|---|
| Start the complete local application | `docker compose up --build` |
| Run database migrations | `docker compose exec backend python -m app.db.migrate upgrade` (the backend also runs it at start; there is no Alembic — see `RESEARCH_LIMITATIONS.md` §3) |
| Run backend tests | `docker compose exec backend python -m pytest` |
| Run frontend tests | `cd frontend && npm test` |
| Build frontend | `cd frontend && npm run build` |
| Check backend health | `curl http://localhost:8000/health` (readiness incl. database and model engine: `/health/ready`) |

## A. Docker Compose

```bash
cp .env.example .env            # set JWT_SECRET, POSTGRES_PASSWORD; keep WORKER_MODE=external
docker compose up --build -d
docker compose exec backend python -m app.manage create-admin --email you@example.org
python scripts/smoke_test_deployment.py --base-url http://localhost:8080 --register \
    --image dataset/sample_data/lits_sample/images/test/ct_lits_4277.png --expect-production
```

* `http://localhost:8080` — SPA through nginx (proxies `/api`, `/health`, `/docs`, `/openapi.json` to the API).
* The API is also published on `127.0.0.1:8000` (this machine only).
* Health checks: PostgreSQL `pg_isready`; API `GET /health` (Docker `HEALTHCHECK`); nginx `GET /healthz`. The worker
  starts after the API is healthy (the API applies migrations at start).
* Volumes: `pgdata` (database), `storage` (uploads + generated images, shared by API and worker).
* Datasets: the project's `dataset/` folder is mounted read-only at `/data` (`DATASET_ROOTS=/data`), so the in-project
  copy is `/data/liver_ct_lits_verified_v8` (also `LTI_DATASET_DIR` inside the containers, for the training scripts).
  Register it once: `docker compose exec backend python -m app.manage register-dataset --path
  /data/liver_ct_lits_verified_v8 --name "LiTS verified (V8)" --if-missing --validate-now`. To mount another host
  folder instead, set `LTI_DATASET_HOST_DIR=/path/on/host` before `docker compose up`.
* TLS: put the stack behind a TLS-terminating proxy (Caddy, Traefik, a cloud load balancer) and set
  `CORS_ORIGINS` / `FRONTEND_BASE_URL` to the public https origin.
* `FORWARDED_ALLOW_IPS` is set to the private Docker ranges so the API trusts `X-Forwarded-For` only from nginx, which
  overwrites the header with the real client address (rate limits and audit use it).

## B. Render

1. Push the project to a Git repository (without `.env`, with the restored weights or Git LFS for files > 100 MB).
2. Render Dashboard → **New → Blueprint** → select the repository. `render.yaml` creates:
   * `lti-db` — Render Postgres 16;
   * `liver-tumour-intelligence` — Docker web service built from `docker/allinone.Dockerfile`, `plan: 1c-2g`, health
     check `/health`, pre-deploy command `python -m app.db.migrate upgrade`, `JWT_SECRET` generated by Render,
     `DATABASE_URL` from the database, a 10 GB persistent disk at `/var/data` for uploads.
3. Enter `FRONTEND_BASE_URL` (your `https://….onrender.com` URL) when prompted.
4. After the first deploy: open a Shell on the service and run `python -m app.manage create-admin --email …`, then run
   the smoke test from your machine against the public URL.

Platform constraints (from Render's documentation, checked Nov 2026): a service with a persistent disk cannot be scaled
to several instances and has no zero-downtime deploys; pre-deploy commands need a paid instance type; free instances
have no persistent disk. For several instances, set `STORAGE_BACKEND=s3` (+ bucket settings), remove the disk and run a
separate worker service with `WORKER_MODE=external` — the S3 backend is implemented but was not tested against a real
bucket. `WORKER_MODE=inline` (the Render default here) runs jobs in threads inside the web service.
`FORWARDED_ALLOW_IPS` defaults to private ranges; if every user shares one rate-limit bucket after deployment, the
platform proxy is not in that range — adjust it.

## C. Vercel (frontend only)

1. Deploy the API first (layout A behind TLS, or B) and note its https origin.
2. In `frontend/vercel.json` replace every `YOUR-BACKEND-HOST` with that host. Requests to `/api/*` and `/health*`
   are proxied by Vercel (external rewrites), so the SPA stays same-origin: no CORS and the strict CSP
   (`connect-src 'self'`) still applies. API responses are excluded from Vercel's CDN cache
   (`x-vercel-enable-rewrite-caching: 0`; the API also sends `Cache-Control: no-store`).
3. Vercel → New Project → import the repository, **Root Directory `frontend`**. Build command, install command and
   output directory (`dist`) come from `vercel.json`. All other paths fall back to `index.html` (client-side routing).
4. On the API set `FRONTEND_BASE_URL` to the Vercel URL (password-reset links).

Alternative without the proxy: build with `API_BASE_URL=https://api.example.org`, set `CORS_ORIGINS` on the API to the
Vercel origin, and add the API origin to `connect-src` in the CSP header of `vercel.json`.

## Operations

| Task | Command |
|---|---|
| Apply / inspect / roll back migrations | `python -m app.db.migrate upgrade` · `status` · `downgrade` (latest) or `downgrade --to <version to keep>` |
| Create an administrator | `python -m app.manage create-admin --email …` (password from `LTI_ADMIN_PASSWORD` or a prompt) |
| Register a server-side dataset | `python -m app.manage register-dataset --path /data/… --name …` |
| Re-sync the model registry after adding artifacts | `python -m app.manage sync-models` |
| Purge expired uploads / predictions | `python -m app.manage purge-expired` |
| Run the job worker | `python -m app.worker` |
| Logs | JSON lines on stdout; each request has an `X-Request-ID`; uploaded file names and image content are never logged |
| Backups | back up the PostgreSQL database and the storage volume together |

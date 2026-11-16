# Liver Tumour Intelligence Platform V11 — all-in-one image (API + model engine + built web app, SQLite by default).
#   docker build -t lti-v11 .
#   docker run --rm -p 8000:8000 -e JWT_SECRET=<48+ random chars> -v lti-data:/var/lib/lti lti-v11   -> http://localhost:8000
#   add  -v /path/to/liver_ct_lits_verified_v8:/srv/dataset/liver_ct_lits_verified_v8:ro  to use the full dataset
# Database (SQLite) and uploaded scans persist in the /var/lib/lti volume.
# Multi-service deployment with PostgreSQL + worker + nginx: docker compose up --build  (see docker-compose.yml).
# CPU only — no GPU / CUDA needed. Research use only; not a medical device.
FROM node:22-alpine AS web
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
LABEL org.opencontainers.image.title="Liver Tumour Intelligence Platform" org.opencontainers.image.version="11.0.0"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app && mkdir -p /var/lib/lti && chown app /var/lib/lti
WORKDIR /srv
COPY backend/requirements.txt backend/requirements-prod.txt /srv/backend/
RUN pip install -r /srv/backend/requirements-prod.txt
COPY src /srv/src
COPY configs /srv/configs
COPY results /srv/results
COPY artifacts /srv/artifacts
COPY models /srv/models
COPY data /srv/data
COPY dataset/sample_data /srv/dataset/sample_data
COPY dataset/manifests /srv/dataset/manifests
COPY dataset/dataset_manifest.json /srv/dataset/dataset_manifest.json
COPY backend /srv/backend
COPY scripts /srv/scripts
# The three ImageNet backbone weights are bundled in artifacts/weights; this verifies them (and downloads only if absent).
ARG FETCH_WEIGHTS=true
RUN if [ "$FETCH_WEIGHTS" = "true" ]; then python /srv/scripts/fetch_weights.py || echo "WARNING: backbone weights missing"; \
    else python /srv/scripts/fetch_weights.py --check || true; fi
COPY --from=web /app/dist /srv/frontend/dist
ENV PYTHONPATH=/srv:/srv/backend DATABASE_URL=sqlite:////var/lib/lti/lti.db STORAGE_DIR=/var/lib/lti/storage \
    FRONTEND_DIST=/srv/frontend/dist FORWARDED_ALLOW_IPS=127.0.0.1
WORKDIR /srv/backend
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4).status == 200 else 1)"
CMD ["sh", "-c", "python -m app.db.migrate upgrade && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips=\"$FORWARDED_ALLOW_IPS\""]

# Single-service image: the API also serves the built SPA (same origin, no CORS). Used by render.yaml.
# Build context: project root.
FROM node:22-alpine AS web
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app && mkdir -p /srv/data /var/data && chown app /srv/data /var/data
WORKDIR /srv
COPY backend/requirements.txt backend/requirements-prod.txt /srv/backend/
RUN pip install -r /srv/backend/requirements-prod.txt
COPY src /srv/src
COPY configs /srv/configs
COPY results /srv/results
COPY artifacts /srv/artifacts
COPY dataset/sample_data /srv/dataset/sample_data
COPY backend /srv/backend
COPY scripts/fetch_weights.py /srv/scripts/fetch_weights.py
ARG FETCH_WEIGHTS=true
RUN if [ "$FETCH_WEIGHTS" = "true" ]; then python /srv/scripts/fetch_weights.py || echo "WARNING: backbone weights missing"; \
    else python /srv/scripts/fetch_weights.py --check || true; fi
COPY --from=web /app/dist /srv/frontend/dist
ENV PYTHONPATH=/srv:/srv/backend STORAGE_DIR=/srv/data/storage FRONTEND_DIST=/srv/frontend/dist FORWARDED_ALLOW_IPS=127.0.0.1
WORKDIR /srv/backend
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4).status == 200 else 1)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips=\"$FORWARDED_ALLOW_IPS\""]

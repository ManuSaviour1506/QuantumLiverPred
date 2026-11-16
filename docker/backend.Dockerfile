# Backend API / worker image (CPU-only). Build context: project root.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app && mkdir -p /srv/data && chown app /srv/data
WORKDIR /srv
COPY backend/requirements.txt backend/requirements-prod.txt backend/requirements-test.txt /srv/backend/
# INSTALL_TEST_DEPS=true (the Compose default) adds pytest + requests so `docker compose exec backend python -m pytest` works
ARG INSTALL_TEST_DEPS=false
RUN pip install -r /srv/backend/requirements-prod.txt \
 && if [ "$INSTALL_TEST_DEPS" = "true" ]; then pip install -r /srv/backend/requirements-test.txt; fi
COPY src /srv/src
COPY configs /srv/configs
COPY results /srv/results
COPY artifacts /srv/artifacts
COPY dataset/sample_data /srv/dataset/sample_data
COPY backend /srv/backend
COPY scripts/fetch_weights.py /srv/scripts/fetch_weights.py
# ImageNet backbone weights: verified if present in the build context, otherwise downloaded from the official timm
# releases (FETCH_WEIGHTS=false skips the download; the registry then reports the dependent models as unavailable).
ARG FETCH_WEIGHTS=true
RUN if [ "$FETCH_WEIGHTS" = "true" ]; then python /srv/scripts/fetch_weights.py || echo "WARNING: backbone weights missing"; \
    else python /srv/scripts/fetch_weights.py --check || true; fi
ENV PYTHONPATH=/srv:/srv/backend STORAGE_DIR=/srv/data/storage FRONTEND_DIST=/srv/frontend-not-served
WORKDIR /srv/backend
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4).status == 200 else 1)"
# FORWARDED_ALLOW_IPS: address/CIDR list of the reverse proxy whose X-Forwarded-For is trusted (client IP for rate limits/audit).
ENV FORWARDED_ALLOW_IPS=127.0.0.1
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips=\"$FORWARDED_ALLOW_IPS\""]

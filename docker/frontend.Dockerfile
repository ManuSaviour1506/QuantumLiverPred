# Frontend: build the SPA with Node, serve it with nginx (which also proxies /api to the backend).
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
RUN npm install --no-audit --no-fund
COPY frontend/ ./
ARG API_BASE_URL=""
ENV API_BASE_URL=${API_BASE_URL}
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=4s CMD wget -qO- http://127.0.0.1/healthz >/dev/null || exit 1

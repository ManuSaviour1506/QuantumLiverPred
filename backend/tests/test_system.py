"""Health checks, OpenAPI, docs, security headers, structured errors, CORS."""
import requests


def test_health_and_ready(live):
    r = requests.get(live["url"] + "/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers.get("x-request-id")
    r = requests.get(live["url"] + "/health/ready")
    assert r.status_code == 200, r.text
    assert r.json()["checks"]["database"] == "ok" and r.json()["checks"]["migrations"] >= 1


def test_openapi_and_docs(live):
    spec = requests.get(live["url"] + "/openapi.json").json()
    assert spec["openapi"].startswith("3.1")
    for p in ("/api/v1/auth/register", "/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/auth/me",
              "/api/v1/auth/logout", "/api/v1/predictions", "/api/v1/predictions/{prediction_id}",
              "/api/v1/predictions/{prediction_id}/report", "/api/v1/datasets/upload", "/api/v1/datasets",
              "/api/v1/datasets/{dataset_id}", "/api/v1/datasets/{dataset_id}/validate",
              "/api/v1/datasets/{dataset_id}/statistics", "/api/v1/models", "/api/v1/models/{model_id}",
              "/api/v1/models/{model_id}/metrics", "/api/v1/models/{model_id}/status", "/api/v1/experiments",
              "/api/v1/experiments/{experiment_id}", "/api/v1/experiments/{experiment_id}/results",
              "/api/v1/analytics/overview", "/api/v1/analytics/model-comparison",
              "/api/v1/analytics/prediction-distribution"):
        assert p in spec["paths"], p
    assert "bearerAuth" in spec["components"]["securitySchemes"]
    assert "TokenOut" in spec["components"]["schemas"]
    assert spec["paths"]["/api/v1/predictions"]["post"]["requestBody"]["content"]["multipart/form-data"]
    html = requests.get(live["url"] + "/docs")
    assert html.status_code == 200 and "swagger-ui" in html.text


def test_security_headers_and_errors(live):
    r = requests.get(live["url"] + "/api/v1/auth/me")
    assert r.status_code == 401
    body = r.json()["error"]
    assert body["code"] == "unauthorized" and body["request_id"]
    for h in ("x-content-type-options", "x-frame-options", "content-security-policy", "referrer-policy"):
        assert h in r.headers
    assert r.headers.get("cache-control") == "no-store"
    r = requests.get(live["url"] + "/api/v1/does-not-exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert "Traceback" not in r.text


def test_cors_preflight(live):
    r = requests.options(live["url"] + "/api/v1/auth/login", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r = requests.options(live["url"] + "/api/v1/auth/login", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") is None


def test_system_info(live):
    j = requests.get(live["url"] + "/api/v1/system/info").json()
    assert j["disclaimer"].startswith("This platform is intended for academic research")
    modes = {m["id"]: m for m in j["execution_modes"]}
    assert modes["custom_numpy_simulator"]["available"] is True
    assert modes["quantum_hardware"]["available"] is False
    assert "secret" not in str(j).lower()

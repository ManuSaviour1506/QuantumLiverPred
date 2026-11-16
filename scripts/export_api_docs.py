#!/usr/bin/env python3
"""Generate docs/openapi.json and API_DOCUMENTATION.md from the running application's OpenAPI document
(so the reference always matches the code).   python scripts/export_api_docs.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "docs-generation-only-" + "x" * 32)

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def schema_name(s: dict | None) -> str:
    if not s:
        return ""
    if "$ref" in s:
        return s["$ref"].rsplit("/", 1)[-1]
    if s.get("type") == "array":
        return f"list[{schema_name(s.get('items'))}]"
    return s.get("type", "object")


def main() -> None:
    app = create_app(Settings.from_env(), start_background=False)
    spec = app.state.openapi
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "openapi.json").write_text(json.dumps(spec, indent=2, default=str))
    lines = [f"# API reference — {spec['info']['title']} {spec['info']['version']}", "",
             "Generated from the application's OpenAPI document by `scripts/export_api_docs.py`. Interactive docs: "
             "`/docs` (Swagger UI); machine-readable: `/openapi.json` and `docs/openapi.json`.", "",
             "Auth column: **public** = no token; **user** = any signed-in user (Bearer access token); "
             "**researcher** / **admin** = role required.", ""]
    by_tag: dict[str, list] = {}
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            by_tag.setdefault((op.get("tags") or ["other"])[0], []).append((path, method, op))
    for tag, ops in by_tag.items():
        lines += [f"## {tag}", "", "| Method | Path | Auth | Summary | Request | Response |", "|---|---|---|---|---|---|"]
        for path, method, op in ops:
            auth = op.get("x-required-role") or ("user" if op.get("security") else "public")
            body = op.get("requestBody", {}).get("content", {})
            req = ", ".join(f"{ct.split(';')[0]}" + (f" `{schema_name(v.get('schema'))}`" if schema_name(v.get("schema")) else "")
                            for ct, v in body.items())
            ok = next((r for c, r in op.get("responses", {}).items() if c.startswith("2")), {})
            resp = schema_name(next(iter(ok.get("content", {}).values()), {}).get("schema")) if ok else ""
            lines.append(f"| {method.upper()} | `{path}` | {auth} | {op.get('summary', '')} | {req} | {resp} |")
        lines.append("")
    lines += ["## Errors", "", "Every error has the same JSON shape: "
              "`{\"error\": {\"code\": \"...\", \"message\": \"...\", \"request_id\": \"...\"}}` "
              "(plus `details` for validation errors). Server paths, secrets and stack traces are never returned.", ""]
    (ROOT / "API_DOCUMENTATION.md").write_text("\n".join(lines))
    print(f"wrote docs/openapi.json and API_DOCUMENTATION.md ({len(spec['paths'])} paths)")


if __name__ == "__main__":
    main()

import "./setup";
import assert from "node:assert/strict";
import { test } from "node:test";
import { api, ApiError, buildUrl, request, tokens } from "../src/lib/api";

type Call = { url: string; init: any };

function mockFetch(handler: (url: string, init: any) => { status: number; body?: any; headers?: Record<string, string> }) {
  const calls: Call[] = [];
  (globalThis as any).fetch = async (url: string, init: any = {}) => {
    calls.push({ url, init });
    const r = handler(url, init);
    const headers = new Headers({ "content-type": "application/json", ...(r.headers || {}) });
    return new Response(r.body === undefined ? null : JSON.stringify(r.body), { status: r.status, headers });
  };
  return calls;
}

const tokenOut = (access: string, refresh: string) => ({ access_token: access, token_type: "bearer", expires_in: 900,
  refresh_token: refresh, refresh_expires_at: "x", user: { id: "u", email: "a@b.org", full_name: "A", role: "researcher",
    is_active: true, created_at: "x", preferences: {} } });

test("buildUrl drops empty query values", () => {
  assert.equal(buildUrl("/api/v1/predictions", { q: "a b", page: 2, model_id: "", x: null }), "/api/v1/predictions?q=a+b&page=2");
});

test("structured errors become ApiError with field errors", async () => {
  mockFetch(() => ({ status: 422, body: { error: { code: "validation_error", message: "Request validation failed",
    details: [{ field: "body.email", message: "not a valid e-mail address" }], request_id: "r1" } } }));
  await assert.rejects(api.post("/api/v1/auth/register", {}), (e: any) => {
    assert.ok(e instanceof ApiError);
    assert.equal(e.status, 422);
    assert.equal(e.code, "validation_error");
    assert.equal(e.requestId, "r1");
    assert.deepEqual(e.fieldErrors(), { email: "not a valid e-mail address" });
    return true;
  });
});

test("bearer token is sent and a 401 triggers one refresh and a retry", async () => {
  tokens.set(tokenOut("old-access", "refresh-1") as any);
  let refreshed = 0;
  const calls = mockFetch((url, init) => {
    if (url.endsWith("/auth/refresh")) { refreshed++; return { status: 200, body: tokenOut("new-access", "refresh-2") }; }
    const auth = init.headers?.Authorization;
    return auth === "Bearer new-access" ? { status: 200, body: { ok: true } } : { status: 401, body: { error: { code: "unauthorized", message: "expired" } } };
  });
  const r = await request("/api/v1/auth/me");
  assert.deepEqual(r, { ok: true });
  assert.equal(refreshed, 1);
  assert.equal(calls.filter((c) => c.url.endsWith("/auth/me")).length, 2);
  assert.equal(localStorage.getItem("lti.refresh"), "refresh-2");
});

test("failed refresh signs the user out", async () => {
  tokens.set(tokenOut("a", "bad-refresh") as any);
  mockFetch((url) => url.endsWith("/auth/refresh") ? { status: 401, body: { error: { code: "unauthorized", message: "no" } } }
    : { status: 401, body: { error: { code: "unauthorized", message: "expired" } } });
  await assert.rejects(request("/api/v1/auth/me"), (e: any) => e.status === 401);
  assert.equal(tokens.access, null);
  assert.equal(localStorage.getItem("lti.refresh"), null);
});

test("network failures are reported clearly", async () => {
  (globalThis as any).fetch = async () => { throw new TypeError("fetch failed"); };
  await assert.rejects(api.get("/health"), (e: any) => e.code === "network_error" && e.status === 0);
});

import "./setup";
import assert from "node:assert/strict";
import { test } from "node:test";
import { fmtBytes, fmtMs, fmtNum, fmtPct, labelTone, statusTone, humanize } from "../src/lib/format";
import { matchPath } from "../src/lib/router";
import { checkImageFile, loginSchema, registerSchema, resetSchema } from "../src/lib/schemas";
import { fetchQuery, invalidate, keyOf } from "../src/lib/query";

test("formatting helpers", () => {
  assert.equal(fmtNum(0.98765, 3), "0.988");
  assert.equal(fmtNum(null), "—");
  assert.equal(fmtPct(0.5), "50.0%");
  assert.equal(fmtBytes(2048), "2.0 KB");
  assert.equal(fmtMs(1500), "1.50 s");
  assert.equal(labelTone("Tumour detected"), "danger");
  assert.equal(labelTone("No tumour detected"), "success");
  assert.equal(labelTone("Uncertain / unable to classify"), "warning");
  assert.equal(statusTone("succeeded"), "success");
  assert.equal(statusTone("missing_files"), "warning");
  assert.equal(humanize("not_trained"), "Not trained");
});

test("router path matching", () => {
  assert.deepEqual(matchPath("/app/predictions/:id", "/app/predictions/pred_1"), { id: "pred_1" });
  assert.equal(matchPath("/app/predictions/:id", "/app/predictions"), null);
  assert.deepEqual(matchPath("/app", "/app?x=1"), {});
  assert.equal(matchPath("/app", "/app/models"), null);
  assert.deepEqual(matchPath("/docs/*", "/docs/a/b"), {});
});

test("form schemas", () => {
  assert.equal(loginSchema.safeParse({ email: "bad", password: "x" }).success, false);
  assert.equal(loginSchema.safeParse({ email: "a@b.org", password: "x" }).success, true);
  const weak = registerSchema.safeParse({ full_name: "A", email: "a@b.org", password: "short", confirm: "short", accept: true });
  assert.equal(weak.success, false);
  const mismatch = registerSchema.safeParse({ full_name: "A", email: "a@b.org", password: "Good-Pass-123", confirm: "Other-Pass-123", accept: true });
  assert.equal(mismatch.success, false);
  assert.ok(!mismatch.success && mismatch.error.issues.some((i) => i.path[0] === "confirm"));
  const noAccept = registerSchema.safeParse({ full_name: "A", email: "a@b.org", password: "Good-Pass-123", confirm: "Good-Pass-123", accept: false });
  assert.equal(noAccept.success, false);
  assert.equal(registerSchema.safeParse({ full_name: "A", email: "a@b.org", password: "Good-Pass-123", confirm: "Good-Pass-123", accept: true }).success, true);
  assert.equal(resetSchema.safeParse({ new_password: "Good-Pass-123", confirm: "Good-Pass-123" }).success, true);
});

test("client-side image checks", () => {
  const f = (name: string, size: number) => ({ name, size }) as File;
  assert.equal(checkImageFile(f("slice.png", 1000), 20), null);
  assert.equal(checkImageFile(f("vol.nii.gz", 1000), 20), null);
  assert.equal(checkImageFile(f("scan.dcm", 1000), 20), null);
  assert.match(checkImageFile(f("notes.txt", 10), 20)!, /Unsupported/);
  assert.match(checkImageFile(f("x.png", 0), 20)!, /empty/);
  assert.match(checkImageFile(f("x.png", 30 * 1024 * 1024), 20)!, /larger/);
});

test("query cache de-duplicates and invalidates", async () => {
  let calls = 0;
  const fn = async () => { calls++; return 42; };
  const k = keyOf(["models", 1]);
  const [a, b] = await Promise.all([fetchQuery(k, fn), fetchQuery(k, fn)]);
  assert.equal(a, 42);
  assert.equal(b, 42);
  assert.equal(calls, 1);
  invalidate(["models"]);
  assert.equal(await fetchQuery(k, fn), 42);
  assert.equal(calls, 2);
});

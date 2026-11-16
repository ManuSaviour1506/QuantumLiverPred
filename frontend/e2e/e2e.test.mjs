// Browser end-to-end test of the complete user flow against a running backend that serves the built frontend.
// No mocked API: every request goes to the real server, which runs real models.
//   E2E_BASE_URL=http://127.0.0.1:8765 E2E_SAMPLE_DIR=../dataset/sample_data/lits_sample npm run test:e2e
import assert from "node:assert/strict";
import { test, before, after } from "node:test";
import path from "node:path";
import fs from "node:fs";
import { chromium } from "playwright";

const BASE = process.env.E2E_BASE_URL || "http://127.0.0.1:8765";
const SAMPLE = path.resolve(process.env.E2E_SAMPLE_DIR || "../dataset/sample_data/lits_sample");
const OUT = path.resolve(process.env.E2E_OUT || "e2e/output");
const email = `e2e_${Date.now()}@example.org`;
const password = "E2e-Test-Pass-42";
let browser, page;
const consoleErrors = [];

before(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1360, height: 900 }, acceptDownloads: true });
  page = await ctx.newPage();
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  page.on("pageerror", (e) => consoleErrors.push(String(e)));
});

after(async () => { await browser?.close(); });

async function shot(name) { await page.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: true }); }

test("landing page", async () => {
  await page.goto(BASE + "/");
  await page.getByRole("heading", { level: 1 }).waitFor();
  assert.match(await page.textContent("body"), /academic research and software experimentation only/);
  await shot("01_landing");
});

test("registration validation and account creation", async () => {
  await page.goto(BASE + "/register");
  await page.getByRole("button", { name: "Create account" }).click();
  await page.getByText("Your name is required").waitFor();
  await page.fill("#f-full_name", "E2E Researcher");
  await page.fill("#f-email", email);
  await page.fill("#f-password", "short");
  await page.fill("#f-confirm", "short");
  await page.getByRole("button", { name: "Create account" }).click();
  await page.getByText("At least 10 characters").waitFor();
  await page.fill("#f-password", password);
  await page.fill("#f-confirm", password);
  await page.check("#f-accept");
  await page.getByRole("button", { name: "Create account" }).click();
  await page.waitForURL(BASE + "/app");
  await page.getByText("Registered models").first().waitFor();          // V10.1+: /app is the comparison overview
  await page.goto(BASE + "/app/activity");                                // the earlier personal dashboard
  await page.getByRole("heading", { name: /Welcome/ }).waitFor();
  await shot("02_dashboard_empty");
});

test("protected routes require a session (separate browser context)", async () => {
  const ctx = await browser.newContext();
  const p = await ctx.newPage();
  await p.goto(BASE + "/app/predictions");
  await p.waitForURL(/\/login\?next=/);
  await ctx.close();
});

test("real prediction on CT slices with a 2.5-D CNN", async () => {
  await page.goto(BASE + "/app/workspace");
  const select = page.locator("#f-model");
  await select.waitFor();
  await page.waitForFunction(() => document.querySelectorAll("#f-model option").length > 1);
  await select.selectOption("cnn25d_v9_a4_res_se_focal");
  await page.setInputFiles("#file-ct-slice", path.join(SAMPLE, "images/test/ct_lits_4278.png"));
  const run = page.getByRole("button", { name: "Run prediction" });
  assert.equal(await run.isDisabled(), true, "run must stay disabled until adjacent slices are added");
  await page.setInputFiles("#file-previous-slice", path.join(SAMPLE, "images/test/ct_lits_4277.png"));
  await page.setInputFiles("#file-next-slice", path.join(SAMPLE, "images/test/ct_lits_4279.png"));
  await shot("03_workspace");
  await run.click();
  await page.waitForURL(/\/app\/predictions\/pred_/, { timeout: 30000 });
  await page.waitForFunction(() => /Model probability|Output withheld|The prediction failed/.test(document.body.innerText),
    null, { timeout: 120000 });
  const body = await page.textContent("body");
  assert.match(body, /Tumour detected|No tumour detected|Uncertain \/ unable to classify/);
  assert.match(body, /Model probability/);
  await page.locator('img[alt="Patch probability map"]').waitFor({ timeout: 30000 });
  await page.locator('img[alt="Grad-CAM heat map"]').waitFor({ timeout: 30000 });
  await shot("04_prediction_result");
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "PDF report" }).click()]);
  const pdf = path.join(OUT, "report.pdf");
  await dl.saveAs(pdf);
  assert.equal(fs.readFileSync(pdf).subarray(0, 4).toString(), "%PDF");
});

test("history lists the prediction and exports CSV", async () => {
  await page.goto(BASE + "/app/predictions");
  await page.getByRole("table").waitFor();
  assert.match(await page.textContent("table"), /ct_lits_4278\.png/);
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export CSV" }).click()]);
  const csv = path.join(OUT, "history.csv");
  await dl.saveAs(csv);
  assert.match(fs.readFileSync(csv, "utf8"), /cnn25d_v9_a4_res_se_focal/);
  await shot("05_history");
});

test("models, comparison, datasets, research and settings pages load real data", async () => {
  await page.goto(BASE + "/app/models");                                  // V10.1+: the model registry
  await page.getByRole("table").first().waitFor();
  await page.getByRole("link", { name: "cnn25d_v9_a4_res_se_focal" }).first().click();
  await page.getByText("Saved evaluations").waitFor();
  await shot("06_models");
  await page.goto(BASE + "/app/compare");
  await page.getByText("All metrics").or(page.getByText("No V10 evaluations saved yet")).first().waitFor();
  if (process.env.E2E_EXPECT_V10 === "1") {
    // the trained V10 models must be listed with their nested-evaluation numbers
    await page.getByRole("tab", { name: /V10 nested evaluation \((?!0\))\d+\)/ }).waitFor();
    await page.getByText("Mean within-unit ROC-AUC", { exact: true }).first().waitFor();
    await page.getByRole("link", { name: /VQC \(4 qubits/ }).first().waitFor();
  }
  await shot("07_compare_v10");
  await page.getByRole("tab", { name: /V9 fixed split/ }).click();
  await page.getByText("All metrics").waitFor();
  await shot("07b_compare_v9");
  await page.goto(BASE + "/app/datasets");
  await page.getByRole("heading", { name: "Dataset management" }).waitFor();
  await page.goto(BASE + "/app/research");
  await page.getByRole("tab", { name: "Quantum circuits" }).waitFor();
  await page.getByRole("tab", { name: "Experiments & logs" }).click();
  await page.getByText("Launch an experiment").waitFor();
  await shot("08_research");
  await page.goto(BASE + "/app/settings");
  await page.getByRole("tab", { name: "API status & about" }).click();
  await page.getByText("Quantum execution modes").waitFor();
  assert.match(await page.textContent("body"), /Quantum hardware/);
  await page.getByRole("tab", { name: "Theme" }).click();
  await page.getByRole("button", { name: "Dark", exact: true }).click();   // exact: the V10.1 header toggle says "Theme: …"
  assert.equal(await page.evaluate(() => document.documentElement.classList.contains("dark")), true);
  await shot("09_settings_dark");
  await page.getByRole("button", { name: "Light", exact: true }).click();
});

test("responsive layout on a phone-sized viewport", async () => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(BASE + "/app");
  const menu = page.getByRole("button", { name: "Open menu" });
  await menu.waitFor();
  await menu.click();
  await page.getByRole("link", { name: "Execution History" }).last().click();   // V10.1+ navigation
  await page.waitForURL(BASE + "/app/history");
  await shot("10_mobile_history");
  await page.setViewportSize({ width: 1360, height: 900 });
});

test("logout ends the session", async () => {
  await page.goto(BASE + "/app");
  await page.getByRole("button", { name: /Sign out/ }).click();
  await page.waitForURL(BASE + "/login");
  await page.goto(BASE + "/app/models");
  await page.waitForURL(/\/login/);
});

test("no browser console errors or CSP violations", () => {
  const real = consoleErrors.filter((e) => !/Failed to load resource: the server responded with a status of 401/.test(e));
  assert.deepEqual(real, []);
});

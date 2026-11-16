import "./setup";
import assert from "node:assert/strict";
import { test } from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { BarChart, LineChart } from "../src/components/charts/charts";
import { Disclaimer, DISCLAIMER } from "../src/components/layout/Disclaimer";
import { Badge, EmptyState, Pagination } from "../src/components/ui/misc";
import { Button } from "../src/components/ui/button";
import { AuthProvider } from "../src/lib/auth";
import Landing from "../src/pages/Landing";

test("bar chart renders one bar per non-null value with accessible labels", () => {
  const html = renderToStaticMarkup(<BarChart title="AUC by model" data={[{ label: "A", value: 0.9 }, { label: "B", value: null }, { label: "C", value: 0.7 }]} />);
  assert.match(html, /aria-label="AUC by model"/);
  assert.equal((html.match(/<rect /g) || []).length, 2);
  assert.match(html, /<caption>AUC by model<\/caption>/);
});

test("horizontal bar chart shows n/a for missing values", () => {
  const html = renderToStaticMarkup(<BarChart horizontal title="t" data={[{ label: "A", value: null }]} />);
  assert.match(html, /n\/a/);
});

test("horizontal bars start at the axis minimum, never over the labels (domain without 0)", () => {
  const html = renderToStaticMarkup(<BarChart horizontal title="AUC" domain={[0.5, 1]}
    data={[{ label: "Logistic regression", value: 0.966 }, { label: "VQC", value: 0.878 }]} />);
  const xs = [...html.matchAll(/<rect x="([-\d.]+)"[^>]*width="([\d.]+)"/g)].map((m) => [Number(m[1]), Number(m[2])]);
  assert.equal(xs.length, 2);
  const labelW = Math.max(96, "Logistic regression".length * 7.1 + 14);
  for (const [x, w] of xs) {
    assert.ok(Math.abs(x - labelW) < 0.01, `bar starts at ${x}, labels end at ${labelW}`);
    assert.ok(w > 0 && x + w <= 600 - 44 + 0.01);
  }
});

test("line chart draws a path per series", () => {
  const html = renderToStaticMarkup(<LineChart title="noise" series={[{ name: "a", points: [{ x: 0, y: 1 }, { x: 1, y: 0.5 }] }, { name: "b", points: [{ x: 0, y: 0.2 }, { x: 1, y: null }] }]} />);
  assert.equal((html.match(/<path /g) || []).length, 2);
});

test("disclaimer text is the required wording", () => {
  const html = renderToStaticMarkup(<Disclaimer />);
  assert.ok(html.includes("academic research and software experimentation only"));
  assert.ok(DISCLAIMER.startsWith("This platform is intended for academic research"));
});

test("basic components", () => {
  assert.match(renderToStaticMarkup(<Badge tone="danger">Tumour detected</Badge>), /Tumour detected/);
  assert.match(renderToStaticMarkup(<EmptyState title="No predictions yet" />), /No predictions yet/);
  assert.match(renderToStaticMarkup(<Button loading>Save</Button>), /aria-busy="true"/);
  const pg = renderToStaticMarkup(<Pagination page={1} pages={3} total={45} onPage={() => {}} />);
  assert.match(pg, /Page 1 of 3/);
  assert.match(pg, /45 results/);
});

test("landing page renders the disclaimer and the limits of the evidence", () => {
  const html = renderToStaticMarkup(<AuthProvider><Landing /></AuthProvider>);
  assert.ok(html.includes("Research use only"));
  assert.ok(html.includes("Claim patient-level performance"));
  assert.ok(html.includes("Quantum and hybrid research"));
});

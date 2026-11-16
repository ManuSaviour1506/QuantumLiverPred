// Production build: esbuild bundles src/main.tsx; Tailwind CSS 4 compiles src/styles.css with the class candidates
// found in the sources (the same compiler the official Tailwind CLI / Vite plugin use). Output: dist/.
//   API_BASE_URL=https://api.example.org npm run build     (default: same origin as the page)
import { build } from "esbuild";
import { compile } from "tailwindcss";
import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir, rm, readdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const src = path.join(root, "src");
const out = path.join(root, process.env.OUT_DIR || "dist");
const require = createRequire(import.meta.url);
const watch = process.argv.includes("--watch");

async function walk(dir) {
  const items = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(items.map((d) => (d.isDirectory() ? walk(path.join(dir, d.name)) : [path.join(dir, d.name)])));
  return files.flat();
}

export async function buildCss() {
  const css = await readFile(path.join(src, "styles.css"), "utf8");
  const compiler = await compile(css, {
    base: src,
    async loadStylesheet(id, base) {
      const file = id === "tailwindcss" ? require.resolve("tailwindcss/index.css")
        : id.startsWith("tailwindcss/") ? require.resolve(id) : path.resolve(base, id);
      return { path: file, base: path.dirname(file), content: await readFile(file, "utf8") };
    },
  });
  const files = (await walk(src)).filter((f) => /\.(tsx?|html)$/.test(f));
  const candidates = new Set();
  for (const f of files) {
    const text = await readFile(f, "utf8");
    for (const tok of text.split(/[\s"'`{}()<>,;=]+/)) if (tok && tok.length < 120) candidates.add(tok);
  }
  return compiler.build([...candidates]);
}

export async function buildAll({ minify = true } = {}) {
  const t0 = Date.now();
  await rm(out, { recursive: true, force: true });
  await mkdir(path.join(out, "assets"), { recursive: true });
  const res = await build({
    entryPoints: [path.join(src, "main.tsx")], bundle: true, format: "esm", target: ["es2020"],
    jsx: "automatic", minify, sourcemap: true, write: false, outdir: path.join(out, "assets"), entryNames: "app",
    define: { "process.env.NODE_ENV": JSON.stringify(minify ? "production" : "development"),
              __API_BASE__: JSON.stringify(process.env.API_BASE_URL || ""), __APP_VERSION__: JSON.stringify("11.0.0") },
    logLevel: "warning", legalComments: "none",
  });
  const hash = createHash("sha256");
  for (const f of res.outputFiles) {
    await writeFile(f.path, f.contents);
    if (f.path.endsWith(".js")) hash.update(f.contents);
  }
  const css = await buildCss();
  await writeFile(path.join(out, "assets", "app.css"), css);
  hash.update(css);
  const v = hash.digest("hex").slice(0, 10);
  const html = (await readFile(path.join(root, "index.html"), "utf8"))
    .replace("/assets/app.css", `/assets/app.css?v=${v}`).replace("/src/main.tsx", `/assets/app.js?v=${v}`);
  await writeFile(path.join(out, "index.html"), html);
  await writeFile(path.join(out, "assets", "theme-init.js"), await readFile(path.join(root, "public", "theme-init.js")));
  const js = res.outputFiles.find((f) => f.path.endsWith("app.js"));
  console.log(`built dist/ in ${Date.now() - t0} ms — app.js ${(js.contents.length / 1024).toFixed(0)} KB, app.css ${(css.length / 1024).toFixed(0)} KB`);
}

if (process.argv[1] === fileURLToPath(import.meta.url) && !watch) {
  buildAll().catch((e) => { console.error(e); process.exit(1); });
}

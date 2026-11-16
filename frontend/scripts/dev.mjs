// Development server: rebuilds on change and serves dist/ on http://localhost:5173, proxying /api, /health,
// /openapi.json and /docs to the backend (BACKEND_URL, default http://localhost:8000).
import http from "node:http";
import { watch } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildAll } from "./build.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const backend = new URL(process.env.BACKEND_URL || "http://localhost:8000");
const port = Number(process.env.PORT || 5173);
const types = { ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".map": "application/json",
  ".svg": "image/svg+xml", ".png": "image/png" };

await buildAll({ minify: false });
let timer;
watch(path.join(root, "src"), { recursive: true }, () => {
  clearTimeout(timer);
  timer = setTimeout(() => buildAll({ minify: false }).catch((e) => console.error(e.message)), 120);
});

http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (/^\/(api\/|health|openapi\.json|docs)/.test(url.pathname)) {
    const p = http.request({ hostname: backend.hostname, port: backend.port, path: req.url, method: req.method,
      headers: { ...req.headers, host: backend.host } }, (r) => { res.writeHead(r.statusCode, r.headers); r.pipe(res); });
    p.on("error", () => { res.writeHead(502, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { code: "backend_unreachable", message: `backend not reachable at ${backend}` } })); });
    req.pipe(p);
    return;
  }
  let file = path.join(dist, decodeURIComponent(url.pathname));
  if (!file.startsWith(dist)) { res.writeHead(400); return res.end(); }
  try { if (!(await stat(file)).isFile()) throw 0; } catch { file = path.join(dist, "index.html"); }
  res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream", "cache-control": "no-cache" });
  res.end(await readFile(file));
}).listen(port, () => console.log(`dev server: http://localhost:${port}  (backend ${backend.href})`));

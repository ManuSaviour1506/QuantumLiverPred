// REST client: bearer access token kept in memory, refresh token in localStorage, one transparent refresh on 401,
// structured ApiError for every non-2xx response, upload progress via XMLHttpRequest.
import type { TokenOut } from "./types";

declare const __API_BASE__: string;
export const API_BASE: string = typeof __API_BASE__ !== "undefined" ? __API_BASE__ : "";
const REFRESH_KEY = "lti.refresh";

export class ApiError extends Error {
  status: number;
  code: string;
  details: unknown;
  requestId?: string;
  constructor(status: number, code: string, message: string, details?: unknown, requestId?: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }
  fieldErrors(): Record<string, string> {
    const out: Record<string, string> = {};
    if (Array.isArray(this.details)) {
      for (const d of this.details as { field?: string; message?: string }[]) {
        if (d.field) out[d.field.split(".").pop() as string] = d.message || "invalid";
      }
    }
    return out;
  }
}

type Listener = (t: TokenOut | null) => void;

class TokenStore {
  access: string | null = null;
  private listeners = new Set<Listener>();
  get refresh(): string | null {
    try { return localStorage.getItem(REFRESH_KEY); } catch { return null; }
  }
  set(t: TokenOut | null) {
    this.access = t ? t.access_token : null;
    try {
      if (t) localStorage.setItem(REFRESH_KEY, t.refresh_token);
      else localStorage.removeItem(REFRESH_KEY);
    } catch { /* storage unavailable: session lasts for this tab only */ }
    this.listeners.forEach((l) => l(t));
  }
  subscribe(l: Listener) {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }
}

export const tokens = new TokenStore();
let refreshing: Promise<boolean> | null = null;

export async function parseError(res: Response): Promise<ApiError> {
  let body: any = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  const e = body?.error;
  return new ApiError(res.status, e?.code || `http_${res.status}`, e?.message || res.statusText || "Request failed",
    e?.details, e?.request_id || res.headers.get("x-request-id") || undefined);
}

export async function refreshSession(): Promise<boolean> {
  const rt = tokens.refresh;
  if (!rt) return false;
  if (!refreshing) {
    refreshing = (async () => {
      try {
        const res = await fetch(API_BASE + "/api/v1/auth/refresh", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ refresh_token: rt }),
        });
        if (!res.ok) { tokens.set(null); return false; }
        tokens.set((await res.json()) as TokenOut);
        return true;
      } catch { return false; }
    })().finally(() => { refreshing = null; });
  }
  return refreshing;
}

export interface RequestOptions {
  method?: string;
  body?: unknown;
  form?: FormData;
  query?: Record<string, string | number | boolean | null | undefined>;
  raw?: boolean;
  auth?: boolean;
}

export function buildUrl(path: string, query?: RequestOptions["query"]): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query || {})) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  const s = qs.toString();
  return API_BASE + path + (s ? `?${s}` : "");
}

export async function request<T = any>(path: string, opts: RequestOptions = {}, retry = true): Promise<T> {
  const headers: Record<string, string> = {};
  if (opts.auth !== false && tokens.access) headers.Authorization = `Bearer ${tokens.access}`;
  let body: BodyInit | undefined;
  if (opts.form) body = opts.form;
  else if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  let res: Response;
  try {
    res = await fetch(buildUrl(path, opts.query), { method: opts.method || (body ? "POST" : "GET"), headers, body });
  } catch {
    throw new ApiError(0, "network_error", "The API is not reachable. Check your connection or the server status.");
  }
  if (res.status === 401 && retry && opts.auth !== false && tokens.refresh) {
    if (await refreshSession()) return request<T>(path, opts, false);
  }
  if (!res.ok) throw await parseError(res);
  if (opts.raw) return res as unknown as T;
  if (res.status === 204) return undefined as T;
  const ct = res.headers.get("content-type") || "";
  return (ct.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

export const api = {
  get: <T = any>(path: string, query?: RequestOptions["query"]) => request<T>(path, { query }),
  post: <T = any>(path: string, body?: unknown) => request<T>(path, { method: "POST", body: body ?? {} }),
  patch: <T = any>(path: string, body: unknown) => request<T>(path, { method: "PATCH", body }),
  del: (path: string) => request<void>(path, { method: "DELETE" }),
};

/** Authenticated download of a binary resource (PDF, CSV, PNG) as an object URL or saved file. */
export async function fetchBlob(path: string, query?: RequestOptions["query"]): Promise<Blob> {
  const res = await request<Response>(path, { query, raw: true });
  return res.blob();
}

export async function download(path: string, filename: string, query?: RequestOptions["query"]): Promise<void> {
  const blob = await fetchBlob(path, query);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Multipart upload with progress (0..1). Retries once after a token refresh. */
export function upload<T = any>(path: string, form: FormData, onProgress?: (f: number) => void, retry = true): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", API_BASE + path);
    if (tokens.access) xhr.setRequestHeader("Authorization", `Bearer ${tokens.access}`);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
    xhr.onerror = () => reject(new ApiError(0, "network_error", "Upload failed: the API is not reachable"));
    xhr.onload = async () => {
      if (xhr.status === 401 && retry && tokens.refresh && (await refreshSession())) {
        upload<T>(path, form, onProgress, false).then(resolve, reject);
        return;
      }
      let body: any = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(new ApiError(xhr.status, body?.error?.code || `http_${xhr.status}`,
        body?.error?.message || "Upload failed", body?.error?.details, body?.error?.request_id));
    };
    xhr.send(form);
  });
}

// Minimal query cache (the role TanStack Query plays): keyed results shared between components, loading / error
// state, refetch on demand or at an interval, and invalidation by key prefix after mutations.
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

type Entry = { data?: unknown; error?: unknown; status: "idle" | "loading" | "success" | "error"; updated: number;
  promise?: Promise<unknown> };

const cache = new Map<string, Entry>();
const subs = new Map<string, Set<() => void>>();

function emit(key: string) {
  subs.get(key)?.forEach((f) => f());
}

function subscribe(key: string, f: () => void) {
  if (!subs.has(key)) subs.set(key, new Set());
  subs.get(key)!.add(f);
  return () => { subs.get(key)?.delete(f); };
}

const EMPTY: Entry = { status: "idle", updated: 0 };

export function keyOf(parts: unknown[]): string {
  return JSON.stringify(parts);
}

export async function fetchQuery<T>(key: string, fn: () => Promise<T>): Promise<T> {
  const cur = cache.get(key);
  if (cur?.promise) return cur.promise as Promise<T>;
  const p = fn();
  cache.set(key, { ...(cur || EMPTY), status: cur?.data !== undefined ? "success" : "loading", promise: p });
  emit(key);
  try {
    const data = await p;
    cache.set(key, { data, status: "success", updated: Date.now() });
    return data;
  } catch (error) {
    cache.set(key, { data: cur?.data, error, status: "error", updated: Date.now() });
    throw error;
  } finally {
    emit(key);
  }
}

export function invalidate(prefix: unknown[]) {
  const p = JSON.stringify(prefix).slice(0, -1);
  for (const k of [...cache.keys()]) {
    if (k.startsWith(p)) {
      cache.set(k, { ...(cache.get(k) as Entry), updated: 0 });
      emit(k);
    }
  }
}

export function clearCache() {
  cache.clear();
  for (const k of subs.keys()) emit(k);
}

export interface QueryResult<T> {
  data: T | undefined;
  error: unknown;
  isLoading: boolean;
  isError: boolean;
  refetch: () => Promise<T | undefined>;
}

export function useQuery<T>(parts: unknown[], fn: () => Promise<T>,
  opts: { enabled?: boolean; refetchInterval?: number | false; staleMs?: number } = {}): QueryResult<T> {
  const key = keyOf(parts);
  const enabled = opts.enabled !== false;
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const entry = useSyncExternalStore((f) => subscribe(key, f), () => cache.get(key) || EMPTY, () => EMPTY);
  const run = useCallback(() => fetchQuery<T>(key, () => fnRef.current()).catch(() => undefined), [key]);
  const stale = opts.staleMs ?? 15000;
  useEffect(() => {
    if (!enabled) return;
    const e = cache.get(key);
    if (!e || (!e.promise && Date.now() - e.updated > stale)) run();
  }, [key, enabled, entry.updated, run, stale]);
  useEffect(() => {
    if (!enabled || !opts.refetchInterval) return;
    const t = setInterval(run, opts.refetchInterval);
    return () => clearInterval(t);
  }, [enabled, opts.refetchInterval, run]);
  return { data: entry.data as T | undefined, error: entry.error, isLoading: enabled && entry.data === undefined &&
    entry.status !== "error", isError: entry.status === "error", refetch: run as () => Promise<T | undefined> };
}

export function useMutation<A, R>(fn: (a: A) => Promise<R>, opts: { onSuccess?: (r: R, a: A) => void } = {}) {
  const [state, setState] = useState<{ pending: boolean; error: unknown; data?: R }>({ pending: false, error: null });
  const mutate = useCallback(async (a: A) => {
    setState({ pending: true, error: null });
    try {
      const r = await fn(a);
      setState({ pending: false, error: null, data: r });
      opts.onSuccess?.(r, a);
      return r;
    } catch (error) {
      setState({ pending: false, error });
      throw error;
    }
  }, [fn, opts]);
  return { ...state, mutate };
}

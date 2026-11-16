// Small history-API router (the role React Router plays): path patterns with :params, links, navigation, search
// params, scroll restoration on navigation and a not-found fallback.
import { createContext, useCallback, useContext, useEffect, useMemo, useSyncExternalStore, type ReactNode,
  type AnchorHTMLAttributes, type MouseEvent } from "react";

function subscribe(f: () => void) {
  window.addEventListener("popstate", f);
  window.addEventListener("lti:navigate", f);
  return () => {
    window.removeEventListener("popstate", f);
    window.removeEventListener("lti:navigate", f);
  };
}

const getLocation = () => window.location.pathname + window.location.search;

export function navigate(to: string, opts: { replace?: boolean } = {}) {
  if (opts.replace) window.history.replaceState(null, "", to);
  else window.history.pushState(null, "", to);
  window.dispatchEvent(new Event("lti:navigate"));
  window.scrollTo({ top: 0 });
}

export function matchPath(pattern: string, path: string): Record<string, string> | null {
  const p = pattern.split("/").filter(Boolean);
  const s = path.split("?")[0].split("/").filter(Boolean);
  if (pattern.endsWith("/*")) {
    if (s.length < p.length - 1) return null;
  } else if (p.length !== s.length) return null;
  const params: Record<string, string> = {};
  for (let i = 0; i < p.length; i++) {
    if (p[i] === "*") return params;
    if (p[i].startsWith(":")) params[p[i].slice(1)] = decodeURIComponent(s[i]);
    else if (p[i] !== s[i]) return null;
  }
  return params;
}

const RouteCtx = createContext<{ params: Record<string, string>; location: string }>({ params: {}, location: "/" });

export function useLocation(): string {
  return useSyncExternalStore(subscribe, getLocation, () => "/");
}

export function useParams<T extends Record<string, string> = Record<string, string>>(): T {
  return useContext(RouteCtx).params as T;
}

export function useSearchParams(): [URLSearchParams, (next: Record<string, string | number | undefined | null>) => void] {
  const loc = useLocation();
  const params = useMemo(() => new URLSearchParams(loc.split("?")[1] || ""), [loc]);
  const set = useCallback((next: Record<string, string | number | undefined | null>) => {
    const p = new URLSearchParams(window.location.search);
    for (const [k, v] of Object.entries(next)) {
      if (v === undefined || v === null || v === "") p.delete(k);
      else p.set(k, String(v));
    }
    const qs = p.toString();
    navigate(window.location.pathname + (qs ? `?${qs}` : ""), { replace: true });
  }, []);
  return [params, set];
}

export interface RouteDef { path: string; element: (params: Record<string, string>) => ReactNode }

export function Routes({ routes, fallback }: { routes: RouteDef[]; fallback: ReactNode }) {
  const location = useLocation();
  const path = location.split("?")[0];
  for (const r of routes) {
    const params = matchPath(r.path, path);
    if (params) return <RouteCtx.Provider value={{ params, location }}>{r.element(params)}</RouteCtx.Provider>;
  }
  return <>{fallback}</>;
}

export function Link({ to, replace, onClick, ...rest }: AnchorHTMLAttributes<HTMLAnchorElement> & { to: string;
  replace?: boolean }) {
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey ||
        rest.target === "_blank") return;
    e.preventDefault();
    navigate(to, { replace });
  };
  return <a href={to} onClick={handle} {...rest} />;
}

export function Redirect({ to }: { to: string }) {
  useEffect(() => { navigate(to, { replace: true }); }, [to]);
  return null;
}

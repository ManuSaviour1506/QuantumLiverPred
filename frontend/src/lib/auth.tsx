// Authentication state backed by the real API (JWT access token + rotating refresh token). No client-side fake auth:
// the user is only "signed in" after the server returns tokens, and every page re-validates through the API.
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, refreshSession, tokens } from "./api";
import { clearCache } from "./query";
import type { TokenOut, User } from "./types";

interface AuthState {
  user: User | null;
  ready: boolean;
  login: (email: string, password: string) => Promise<User>;
  register: (email: string, password: string, fullName: string) => Promise<void>;
  logout: () => Promise<void>;
  setUser: (u: User) => void;
}

const Ctx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const off = tokens.subscribe((t) => setUser(t ? t.user : null));
    (async () => {
      if (tokens.refresh && (await refreshSession())) {
        try { setUser(await api.get<User>("/api/v1/auth/me")); } catch { tokens.set(null); }
      }
      setReady(true);
    })();
    return () => { off(); };
  }, []);
  const login = useCallback(async (email: string, password: string) => {
    const t = await api.post<TokenOut>("/api/v1/auth/login", { email, password });
    clearCache();
    tokens.set(t);
    return t.user;
  }, []);
  const register = useCallback(async (email: string, password: string, fullName: string) => {
    await api.post("/api/v1/auth/register", { email, password, full_name: fullName });
  }, []);
  const logout = useCallback(async () => {
    try { await api.post("/api/v1/auth/logout", { refresh_token: tokens.refresh }); } catch { /* already invalid */ }
    tokens.set(null);
    clearCache();
  }, []);
  return <Ctx.Provider value={{ user, ready, login, register, logout, setUser }}>{children}</Ctx.Provider>;
}

export function useAuth(): AuthState {
  const c = useContext(Ctx);
  if (!c) throw new Error("useAuth outside AuthProvider");
  return c;
}

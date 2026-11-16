import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

export type Theme = "light" | "dark" | "system";
const Ctx = createContext<{ theme: Theme; setTheme: (t: Theme) => void; resolved: "light" | "dark" }>({
  theme: "system", setTheme: () => {}, resolved: "light" });

function systemDark() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    try { return (localStorage.getItem("lti.theme") as Theme) || "dark"; } catch { return "dark"; }
  });
  const resolved: "light" | "dark" = theme === "system" ? (systemDark() ? "dark" : "light") : theme;
  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark");
  }, [resolved]);
  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    try { localStorage.setItem("lti.theme", t); } catch { /* ignore */ }
  }, []);
  return <Ctx.Provider value={{ theme, setTheme, resolved }}>{children}</Ctx.Provider>;
}

export const useTheme = () => useContext(Ctx);

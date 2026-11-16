import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { LuCircleAlert, LuCircleCheck, LuInfo, LuX } from "react-icons/lu";
import { cn } from "../../lib/utils";

type Tone = "success" | "error" | "info";
interface Toast { id: number; tone: Tone; message: string }
const Ctx = createContext<(tone: Tone, message: string) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const push = useCallback((tone: Tone, message: string) => {
    const id = Date.now() + Math.random();
    setItems((x) => [...x, { id, tone, message }]);
    setTimeout(() => setItems((x) => x.filter((t) => t.id !== id)), 5000);
  }, []);
  return (
    <Ctx.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[min(92vw,380px)] flex-col gap-2" aria-live="polite">
        {items.map((t) => {
          const Icon = t.tone === "success" ? LuCircleCheck : t.tone === "error" ? LuCircleAlert : LuInfo;
          return (
            <div key={t.id} className={cn("card pointer-events-auto flex animate-fade-in items-start gap-3 p-3 text-sm",
              t.tone === "error" && "border-red-300 dark:border-red-900")}>
              <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", t.tone === "success" ? "text-emerald-600" :
                t.tone === "error" ? "text-red-600" : "text-sky-600")} />
              <p className="flex-1">{t.message}</p>
              <button aria-label="Dismiss" onClick={() => setItems((x) => x.filter((i) => i.id !== t.id))}
                className="text-muted hover:text-ink"><LuX /></button>
            </div>
          );
        })}
      </div>
    </Ctx.Provider>
  );
}

export const useToast = () => useContext(Ctx);

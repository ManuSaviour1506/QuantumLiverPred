import { LuShieldAlert } from "react-icons/lu";
import { cn } from "../../lib/utils";

export const DISCLAIMER = "This platform is intended for academic research and software experimentation only. It has " +
  "not been clinically validated and must not be used for diagnosis, treatment decisions or emergency medical " +
  "decisions. Model outputs are experimental predictions and require independent clinical validation.";

export function Disclaimer({ compact = false, className }: { compact?: boolean; className?: string }) {
  return (
    <div role="note" className={cn("flex gap-3 rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3 text-amber-900 " +
      "dark:border-amber-900/70 dark:bg-amber-950/30 dark:text-amber-100", compact ? "text-xs" : "text-sm", className)}>
      <LuShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <p><span className="font-semibold">Research use only. </span>{DISCLAIMER}</p>
    </div>
  );
}

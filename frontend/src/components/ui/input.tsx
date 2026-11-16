import { forwardRef, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes,
  type TextareaHTMLAttributes } from "react";
import { cn } from "../../lib/utils";

const base = "w-full rounded-lg border border-line bg-panel px-3 text-sm text-ink placeholder:text-muted/70 " +
  "focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20 disabled:opacity-60 " +
  "aria-[invalid=true]:border-red-500";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input ref={ref} className={cn(base, "h-10", className)} {...p} />
));
Input.displayName = "Input";

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...p }, ref) => <textarea ref={ref} className={cn(base, "min-h-20 py-2", className)} {...p} />);
Textarea.displayName = "Textarea";

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, children, ...p }, ref) => (
    <select ref={ref} className={cn(base, "h-10 pr-8", className)} {...p}>{children}</select>
  ));
Select.displayName = "Select";

export function Label({ htmlFor, children, hint }: { htmlFor?: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label htmlFor={htmlFor} className="mb-1.5 flex items-baseline justify-between gap-2 text-sm font-medium text-ink">
      <span>{children}</span>
      {hint && <span className="text-xs font-normal text-muted">{hint}</span>}
    </label>
  );
}

export function FieldError({ id, children }: { id?: string; children?: ReactNode }) {
  if (!children) return null;
  return <p id={id} role="alert" className="mt-1 text-xs text-red-600 dark:text-red-400">{children}</p>;
}

export function Field({ label, name, error, hint, children }: { label: string; name: string; error?: string;
  hint?: ReactNode; children: ReactNode }) {
  return (
    <div>
      <Label htmlFor={`f-${name}`} hint={hint}>{label}</Label>
      {children}
      <FieldError id={`f-${name}-error`}>{error}</FieldError>
    </div>
  );
}

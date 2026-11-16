// Form state with Zod validation (the role React Hook Form + zodResolver play): typed values, per-field errors after
// blur or submit, server-side field errors merged in, submitting flag.
import { useCallback, useState, type ChangeEvent, type FormEvent } from "react";
import type { ZodType } from "zod";
import { ApiError } from "./api";

export function useZodForm<T extends Record<string, any>>(schema: ZodType<T>, initial: T) {
  const [values, setValues] = useState<T>(initial);
  const [errors, setErrors] = useState<Partial<Record<keyof T | "_form", string>>>({});
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({});
  const [submitting, setSubmitting] = useState(false);

  const validate = useCallback((v: T) => {
    const r = schema.safeParse(v);
    if (r.success) return {} as Partial<Record<keyof T, string>>;
    const out: Partial<Record<keyof T, string>> = {};
    for (const issue of r.error.issues) {
      const k = issue.path[0] as keyof T;
      if (k !== undefined && !out[k]) out[k] = issue.message;
    }
    return out;
  }, [schema]);

  const setValue = useCallback(<K extends keyof T>(k: K, v: T[K]) => {
    setValues((prev) => {
      const next = { ...prev, [k]: v };
      if (touched[k]) setErrors((e) => ({ ...e, [k]: validate(next)[k], _form: undefined }));
      return next;
    });
  }, [touched, validate]);

  const register = useCallback(<K extends keyof T>(k: K) => ({
    name: String(k),
    id: `f-${String(k)}`,
    value: values[k] as any,
    checked: typeof values[k] === "boolean" ? (values[k] as boolean) : undefined,
    "aria-invalid": errors[k] ? true : undefined,
    "aria-describedby": errors[k] ? `f-${String(k)}-error` : undefined,
    onChange: (e: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
      const t = e.target as HTMLInputElement;
      setValue(k, (t.type === "checkbox" ? t.checked : t.value) as T[K]);
    },
    onBlur: () => {
      setTouched((t) => ({ ...t, [k]: true }));
      setErrors((e) => ({ ...e, [k]: validate(values)[k] }));
    },
  }), [values, errors, setValue, validate]);

  const handleSubmit = (fn: (v: T) => Promise<void> | void) => async (e?: FormEvent) => {
    e?.preventDefault();
    const errs = validate(values);
    setTouched(Object.fromEntries(Object.keys(values).map((k) => [k, true])) as any);
    if (Object.keys(errs).length) {
      setErrors(errs);
      return;
    }
    setSubmitting(true);
    setErrors({});
    try {
      await fn(schema.parse(values));
    } catch (err) {
      if (err instanceof ApiError) {
        const fe = err.fieldErrors();
        setErrors({ ...(fe as any), _form: err.message });
      } else {
        setErrors({ _form: (err as Error)?.message || "Something went wrong" } as any);
      }
    } finally {
      setSubmitting(false);
    }
  };

  return { values, errors, register, setValue, handleSubmit, submitting, setErrors, reset: () => setValues(initial) };
}

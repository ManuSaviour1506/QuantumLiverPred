import { useEffect, useState, type ReactNode } from "react";
import { LuMailCheck } from "react-icons/lu";
import { ThemeToggle } from "../components/layout/AppShell";
import { Disclaimer } from "../components/layout/Disclaimer";
import { Logo } from "../components/layout/Logo";
import { Button } from "../components/ui/button";
import { Field, Input } from "../components/ui/input";
import { Alert } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useZodForm } from "../lib/form";
import { useDocumentTitle } from "../lib/hooks";
import { Link, navigate, useSearchParams } from "../lib/router";
import { forgotSchema, loginSchema, registerSchema, resetSchema } from "../lib/schemas";
import type { SystemInfo } from "../lib/types";

function AuthLayout({ title, subtitle, children, footer }: { title: string; subtitle?: string; children: ReactNode;
  footer?: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col bg-surface">
      <div className="flex items-center justify-between px-4 py-3 sm:px-6">
        <Link to="/" className="flex items-center gap-2 font-semibold"><Logo className="h-7 w-7" /> Liver Tumour Intelligence</Link>
        <ThemeToggle />
      </div>
      <div className="flex flex-1 items-start justify-center px-4 pb-10 pt-6 sm:pt-12">
        <div className="w-full max-w-md">
          <div className="card p-6 sm:p-8">
            <h1 className="text-xl font-semibold">{title}</h1>
            {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
            <div className="mt-6">{children}</div>
          </div>
          {footer && <div className="mt-4 text-center text-sm text-muted">{footer}</div>}
          <Disclaimer compact className="mt-4" />
        </div>
      </div>
    </div>
  );
}

export function Login() {
  useDocumentTitle("Sign in");
  const { login } = useAuth();
  const [params] = useSearchParams();
  const f = useZodForm(loginSchema, { email: "", password: "" });
  return (
    <AuthLayout title="Sign in" subtitle="Use your platform account."
      footer={<>No account? <Link to="/register" className="font-medium text-brand-700 hover:underline dark:text-brand-300">Create one</Link></>}>
      <form noValidate onSubmit={f.handleSubmit(async (v) => {
        await login(v.email, v.password);
        const next = params.get("next");
        navigate(next && next.startsWith("/app") ? next : "/app");
      })} className="space-y-4">
        {f.errors._form && <Alert tone="danger">{f.errors._form}</Alert>}
        <Field label="E-mail" name="email" error={f.errors.email}>
          <Input type="email" autoComplete="email" {...f.register("email")} />
        </Field>
        <Field label="Password" name="password" error={f.errors.password}
          hint={<Link to="/forgot-password" className="text-brand-700 hover:underline dark:text-brand-300">Forgot password?</Link>}>
          <Input type="password" autoComplete="current-password" {...f.register("password")} />
        </Field>
        <Button type="submit" className="w-full" loading={f.submitting}>Sign in</Button>
      </form>
    </AuthLayout>
  );
}

export function Register() {
  useDocumentTitle("Create account");
  const { register, login } = useAuth();
  const toast = useToast();
  const [info, setInfo] = useState<SystemInfo | null>(null);
  useEffect(() => { api.get<SystemInfo>("/api/v1/system/info").then(setInfo).catch(() => {}); }, []);
  const f = useZodForm(registerSchema, { full_name: "", email: "", password: "", confirm: "", accept: false });
  if (info && !info.registration_open) {
    return <AuthLayout title="Registration closed"><Alert tone="info">Self-registration is disabled on this server. Ask an administrator for an account.</Alert></AuthLayout>;
  }
  return (
    <AuthLayout title="Create a researcher account" subtitle="Accounts can run predictions, upload datasets and launch experiments."
      footer={<>Already registered? <Link to="/login" className="font-medium text-brand-700 hover:underline dark:text-brand-300">Sign in</Link></>}>
      <form noValidate onSubmit={f.handleSubmit(async (v) => {
        await register(v.email, v.password, v.full_name);
        await login(v.email, v.password);
        toast("success", "Account created — welcome!");
        navigate("/app");
      })} className="space-y-4">
        {f.errors._form && <Alert tone="danger">{f.errors._form}</Alert>}
        <Field label="Full name" name="full_name" error={f.errors.full_name}><Input autoComplete="name" {...f.register("full_name")} /></Field>
        <Field label="E-mail" name="email" error={f.errors.email}><Input type="email" autoComplete="email" {...f.register("email")} /></Field>
        <Field label="Password" name="password" error={f.errors.password} hint="10+ characters, mixed">
          <Input type="password" autoComplete="new-password" {...f.register("password")} />
        </Field>
        <Field label="Confirm password" name="confirm" error={f.errors.confirm}>
          <Input type="password" autoComplete="new-password" {...f.register("confirm")} />
        </Field>
        <div>
          <label className="flex items-start gap-2 text-sm text-muted">
            <input type="checkbox" className="mt-1 h-4 w-4 accent-brand-600" {...f.register("accept")} />
            I understand this is research software that must not be used for diagnosis or treatment decisions.
          </label>
          {f.errors.accept && <p className="mt-1 text-xs text-red-600" role="alert">{f.errors.accept}</p>}
        </div>
        <Button type="submit" className="w-full" loading={f.submitting}>Create account</Button>
      </form>
    </AuthLayout>
  );
}

export function ForgotPassword() {
  useDocumentTitle("Reset password");
  const [sent, setSent] = useState(false);
  const f = useZodForm(forgotSchema, { email: "" });
  return (
    <AuthLayout title="Reset your password" footer={<Link to="/login" className="hover:underline">Back to sign in</Link>}>
      {sent ? (
        <Alert tone="success" title="Check your inbox"><LuMailCheck className="hidden" />If an account exists for that address, a reset link valid for one hour has been sent.</Alert>
      ) : (
        <form noValidate onSubmit={f.handleSubmit(async (v) => { await api.post("/api/v1/auth/forgot-password", v); setSent(true); })} className="space-y-4">
          {f.errors._form && <Alert tone={f.errors._form.includes("not configured") ? "info" : "danger"}>{f.errors._form}</Alert>}
          <Field label="E-mail" name="email" error={f.errors.email}><Input type="email" autoComplete="email" {...f.register("email")} /></Field>
          <Button type="submit" className="w-full" loading={f.submitting}>Send reset link</Button>
        </form>
      )}
    </AuthLayout>
  );
}

export function ResetPassword() {
  useDocumentTitle("Choose a new password");
  const [params] = useSearchParams();
  const token = params.get("token") || "";
  const toast = useToast();
  const f = useZodForm(resetSchema, { new_password: "", confirm: "" });
  return (
    <AuthLayout title="Choose a new password">
      {!token ? <Alert tone="danger">This reset link is incomplete. Request a new one.</Alert> : (
        <form noValidate onSubmit={f.handleSubmit(async (v) => {
          await api.post("/api/v1/auth/reset-password", { token, new_password: v.new_password });
          toast("success", "Password updated. Sign in with your new password.");
          navigate("/login");
        })} className="space-y-4">
          {f.errors._form && <Alert tone="danger">{f.errors._form}</Alert>}
          <Field label="New password" name="new_password" error={f.errors.new_password}><Input type="password" autoComplete="new-password" {...f.register("new_password")} /></Field>
          <Field label="Confirm password" name="confirm" error={f.errors.confirm}><Input type="password" autoComplete="new-password" {...f.register("confirm")} /></Field>
          <Button type="submit" className="w-full" loading={f.submitting}>Update password</Button>
        </form>
      )}
    </AuthLayout>
  );
}

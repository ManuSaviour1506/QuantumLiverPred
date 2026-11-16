import { useState } from "react";
import { LuMonitor, LuMoon, LuSun } from "react-icons/lu";
import { PageHeader } from "../components/layout/AppShell";
import { Disclaimer } from "../components/layout/Disclaimer";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Field, Input, Select } from "../components/ui/input";
import { Alert, Badge, Skeleton, Table, Tabs, td, th } from "../components/ui/misc";
import { useToast } from "../components/ui/toast";
import { api, API_BASE } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useZodForm } from "../lib/form";
import { fmtDate, statusTone } from "../lib/format";
import { useDocumentTitle } from "../lib/hooks";
import { invalidate, useQuery } from "../lib/query";
import { changePasswordSchema, profileSchema } from "../lib/schemas";
import { useTheme, type Theme } from "../lib/theme";
import type { Page, SystemInfo, User } from "../lib/types";
import { cn } from "../lib/utils";

type Tab = "profile" | "security" | "appearance" | "system" | "users";

function Profile() {
  const { user, setUser } = useAuth();
  const toast = useToast();
  const f = useZodForm(profileSchema, { full_name: user?.full_name || "" });
  return (
    <Card><CardHeader title="Profile" /><CardBody>
      <form noValidate className="max-w-md space-y-4" onSubmit={f.handleSubmit(async (v) => {
        const u = await api.patch<User>("/api/v1/users/me", v);
        setUser(u);
        toast("success", "Profile updated");
      })}>
        {f.errors._form && <Alert tone="danger">{f.errors._form}</Alert>}
        <Field label="Full name" name="full_name" error={f.errors.full_name}><Input {...f.register("full_name")} /></Field>
        <div className="text-sm text-muted">E-mail: <b className="text-ink">{user?.email}</b> · role <Badge tone="navy">{user?.role}</Badge> · member since {fmtDate(user?.created_at, false)}</div>
        <Button type="submit" loading={f.submitting}>Save</Button>
      </form>
    </CardBody></Card>
  );
}

function Security() {
  const toast = useToast();
  const f = useZodForm(changePasswordSchema, { current_password: "", new_password: "", confirm: "" });
  return (
    <Card><CardHeader title="Change password" description="Other sessions are signed out after a change." /><CardBody>
      <form noValidate className="max-w-md space-y-4" onSubmit={f.handleSubmit(async (v) => {
        await api.post("/api/v1/users/me/password", { current_password: v.current_password, new_password: v.new_password });
        f.reset();
        toast("success", "Password changed");
      })}>
        {f.errors._form && <Alert tone="danger">{f.errors._form}</Alert>}
        <Field label="Current password" name="current_password" error={f.errors.current_password}><Input type="password" autoComplete="current-password" {...f.register("current_password")} /></Field>
        <Field label="New password" name="new_password" error={f.errors.new_password}><Input type="password" autoComplete="new-password" {...f.register("new_password")} /></Field>
        <Field label="Confirm new password" name="confirm" error={f.errors.confirm}><Input type="password" autoComplete="new-password" {...f.register("confirm")} /></Field>
        <Button type="submit" loading={f.submitting}>Change password</Button>
      </form>
      <ul className="mt-6 space-y-1 text-xs text-muted">
        <li>• Passwords are stored as Argon2id hashes; access tokens last 15 minutes, refresh tokens rotate on every use.</li>
        <li>• Re-using an old refresh token signs out every session of the account.</li>
        <li>• Uploaded images are private to you (and administrators) and are deleted after the retention period.</li>
      </ul>
    </CardBody></Card>
  );
}

function Appearance() {
  const { theme, setTheme } = useTheme();
  const opts: { id: Theme; label: string; icon: typeof LuSun }[] = [{ id: "light", label: "Light", icon: LuSun }, { id: "dark", label: "Dark", icon: LuMoon }, { id: "system", label: "System", icon: LuMonitor }];
  return (
    <Card><CardHeader title="Theme" /><CardBody className="flex flex-wrap gap-3">
      {opts.map((o) => (
        <button key={o.id} onClick={() => { setTheme(o.id); api.patch("/api/v1/users/me", { preferences: { theme: o.id } }).catch(() => {}); }}
          aria-pressed={theme === o.id} className={cn("flex w-36 flex-col items-center gap-2 rounded-xl border p-4 text-sm",
            theme === o.id ? "border-brand-500 ring-2 ring-brand-500/20" : "border-line hover:border-brand-300")}>
          <o.icon className="h-5 w-5" /> {o.label}
        </button>))}
    </CardBody></Card>
  );
}

function SystemPanel() {
  const info = useQuery(["system-info"], () => api.get<SystemInfo>("/api/v1/system/info"));
  const ready = useQuery(["ready"], () => fetch(API_BASE + "/health/ready").then((r) => r.json()));
  const i = info.data;
  if (!i) return <Skeleton className="h-64" />;
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card><CardHeader title="API status" /><CardBody className="space-y-2 text-sm">
        {[["Readiness", <Badge key="r" tone={ready.data?.status === "ready" ? "success" : "warning"}>{ready.data?.status || "…"}</Badge>],
          ["Database", i.database.dialect], ["Storage", `${i.storage.backend}${i.storage.persistent === true ? " (persistent)" : ""}`],
          ["Background jobs", i.worker_mode === "inline" ? "in the API process" : "separate worker process"],
          ["Model engine", <Badge key="e" tone={statusTone(i.engine.state)}>{i.engine.state}</Badge>],
          ["Servable models", `${i.engine.models_servable ?? "—"} of ${i.engine.models_total ?? "—"}`],
          ["Password reset e-mail", i.password_reset_available ? "configured" : "not configured"],
          ["Self-registration", i.registration_open ? "open" : "closed"]].map(([k, v]) => (
          <div key={k as string} className="flex justify-between gap-3 border-b border-line/60 pb-1.5"><span className="text-muted">{k}</span><span className="text-right">{v}</span></div>))}
      </CardBody></Card>
      <Card><CardHeader title="Quantum execution modes" /><CardBody className="space-y-2 text-sm">
        {i.execution_modes.map((m) => (
          <div key={m.id} className="flex items-start justify-between gap-3 border-b border-line/60 pb-1.5">
            <span>{m.label}{m.note && <span className="block text-xs text-muted">{m.note}</span>}</span>
            <Badge tone={m.available ? "success" : "neutral"}>{m.available ? "available" : "not available"}</Badge></div>))}
        <p className="pt-2 text-xs text-muted">Optional packages: {Object.entries(i.optional_packages).map(([k, v]) => `${k} ${v ? "✓" : "✗"}`).join(" · ")}</p>
      </CardBody></Card>
      <Card className="lg:col-span-2"><CardHeader title="Application information" /><CardBody className="space-y-3 text-sm">
        <p><b>{i.app}</b> · version {i.version} · environment {i.environment}</p>
        <p className="text-muted">Upload limit {i.limits.max_upload_mb} MB per image, {i.limits.max_dataset_upload_mb} MB per dataset archive; uploads are kept {i.limits.retention_days} days. Image formats: {i.limits.image_formats.join(", ")}.</p>
        <p className="text-muted">API documentation: <a className="text-brand-700 underline dark:text-brand-300" href="/docs" target="_blank" rel="noreferrer">/docs</a> (OpenAPI at /openapi.json).</p>
        <Disclaimer compact />
      </CardBody></Card>
    </div>
  );
}

function UsersPanel() {
  const toast = useToast();
  const { user: me } = useAuth();
  const [q, setQ] = useState("");
  const list = useQuery(["users", q], () => api.get<Page<User>>("/api/v1/users", { q, page_size: 50 }));
  return (
    <Card><CardHeader title="Users" description="Administrators can change roles and disable accounts." action={<Input aria-label="Search users" placeholder="Search" className="w-56" value={q} onChange={(e) => setQ(e.target.value)} />} />
      {list.isLoading ? <Skeleton className="m-4 h-32" /> : (
        <Table><thead><tr>{["Name", "E-mail", "Role", "Active", "Last sign-in"].map((h) => <th key={h} className={th}>{h}</th>)}</tr></thead>
          <tbody>{(list.data?.items || []).map((u) => (
            <tr key={u.id}><td className={td}>{u.full_name}</td><td className={td}>{u.email}</td>
              <td className={td}><Select aria-label={`Role of ${u.email}`} className="h-8 w-32" value={u.role} disabled={u.id === me?.id}
                onChange={async (e) => { try { await api.patch(`/api/v1/users/${u.id}`, { role: e.target.value }); invalidate(["users"]); toast("success", "Role updated"); } catch (err) { toast("error", (err as Error).message); } }}>
                {["admin", "researcher", "viewer"].map((r) => <option key={r}>{r}</option>)}</Select></td>
              <td className={td}><input type="checkbox" aria-label={`Active: ${u.email}`} className="h-4 w-4 accent-brand-600" checked={u.is_active} disabled={u.id === me?.id}
                onChange={async (e) => { try { await api.patch(`/api/v1/users/${u.id}`, { is_active: e.target.checked }); invalidate(["users"]); } catch (err) { toast("error", (err as Error).message); } }} /></td>
              <td className={td}>{fmtDate(u.last_login_at)}</td></tr>))}</tbody></Table>
      )}
    </Card>
  );
}

export default function Settings() {
  useDocumentTitle("Settings");
  const { user } = useAuth();
  const [tab, setTab] = useState<Tab>("profile");
  const tabs: { id: Tab; label: string }[] = [{ id: "profile", label: "Profile" }, { id: "security", label: "Security" },
    { id: "appearance", label: "Theme" }, { id: "system", label: "API status & about" }];
  if (user?.role === "admin") tabs.push({ id: "users", label: "Users" });
  return (
    <>
      <PageHeader title="Settings" />
      <div className="mb-5"><Tabs value={tab} onChange={setTab} tabs={tabs} /></div>
      {tab === "profile" && <Profile />}
      {tab === "security" && <Security />}
      {tab === "appearance" && <Appearance />}
      {tab === "system" && <SystemPanel />}
      {tab === "users" && <UsersPanel />}
    </>
  );
}

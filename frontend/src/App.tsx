import type { ReactNode } from "react";
import { LuLoaderCircle } from "react-icons/lu";
import { AppShell } from "./components/layout/AppShell";
import { ToastProvider } from "./components/ui/toast";
import { AuthProvider, useAuth } from "./lib/auth";
import { Redirect, Routes } from "./lib/router";
import { ThemeProvider } from "./lib/theme";
import Landing from "./pages/Landing";
import { ForgotPassword, Login, Register, ResetPassword } from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import Overview from "./pages/Overview";
import Upload from "./pages/Upload";
import LiveComparison from "./pages/LiveComparison";
import RunDetails from "./pages/RunDetails";
import Benchmarks from "./pages/Benchmarks";
import Registry from "./pages/Registry";
import RunHistory from "./pages/RunHistory";
import Reports from "./pages/Reports";
import SystemMonitor from "./pages/SystemMonitor";
import Workspace from "./pages/Workspace";
import PredictionDetail from "./pages/PredictionDetail";
import History from "./pages/History";
import Models from "./pages/Models";
import Compare from "./pages/Compare";
import Datasets, { DatasetDetail } from "./pages/Datasets";
import Research from "./pages/Research";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";

function Protected({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth();
  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center text-muted" role="status">
        <LuLoaderCircle className="mr-2 h-5 w-5 animate-spin" /> Checking your session…
      </div>
    );
  }
  if (!user) return <Redirect to={`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`} />;
  return <AppShell>{children}</AppShell>;
}

function GuestOnly({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth();
  if (ready && user) return <Redirect to="/app" />;
  return <>{children}</>;
}

const routes = [
  { path: "/", element: () => <Landing /> },
  { path: "/login", element: () => <GuestOnly><Login /></GuestOnly> },
  { path: "/register", element: () => <GuestOnly><Register /></GuestOnly> },
  { path: "/forgot-password", element: () => <ForgotPassword /> },
  { path: "/reset-password", element: () => <ResetPassword /> },
  { path: "/app", element: () => <Protected><Overview /></Protected> },
  { path: "/app/upload", element: () => <Protected><Upload /></Protected> },
  { path: "/app/live", element: () => <Protected><LiveComparison /></Protected> },
  { path: "/app/live/:id", element: () => <Protected><LiveComparison /></Protected> },
  { path: "/app/runs", element: () => <Protected><RunDetails /></Protected> },
  { path: "/app/runs/:id", element: () => <Protected><RunDetails /></Protected> },
  { path: "/app/benchmarks", element: () => <Protected><Benchmarks /></Protected> },
  { path: "/app/models", element: () => <Protected><Registry /></Protected> },
  { path: "/app/models/:id", element: () => <Protected><Models /></Protected> },
  { path: "/app/history", element: () => <Protected><RunHistory /></Protected> },
  { path: "/app/reports", element: () => <Protected><Reports /></Protected> },
  { path: "/app/datasets", element: () => <Protected><Datasets /></Protected> },
  { path: "/app/datasets/:id", element: () => <Protected><DatasetDetail /></Protected> },
  { path: "/app/settings", element: () => <Protected><Settings /></Protected> },
  { path: "/app/system", element: () => <Protected><SystemMonitor /></Protected> },
  // earlier single-model pages, kept working and linked from the new pages
  { path: "/app/activity", element: () => <Protected><Dashboard /></Protected> },
  { path: "/app/workspace", element: () => <Protected><Workspace /></Protected> },
  { path: "/app/predictions", element: () => <Protected><History /></Protected> },
  { path: "/app/predictions/:id", element: () => <Protected><PredictionDetail /></Protected> },
  { path: "/app/compare", element: () => <Protected><Compare /></Protected> },
  { path: "/app/research", element: () => <Protected><Research /></Protected> },
];

export default function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <AuthProvider>
          <Routes routes={routes} fallback={<NotFound />} />
        </AuthProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}

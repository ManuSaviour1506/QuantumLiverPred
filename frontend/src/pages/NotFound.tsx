import { Button } from "../components/ui/button";
import { Link } from "../lib/router";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-3 bg-surface p-6 text-center">
      <p className="text-sm font-semibold text-brand-600">404</p>
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="text-muted">The page you requested does not exist.</p>
      <Link to="/"><Button variant="outline">Back to the start page</Button></Link>
    </div>
  );
}

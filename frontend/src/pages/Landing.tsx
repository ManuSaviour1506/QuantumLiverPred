import { LuArrowRight, LuAtom, LuBrainCircuit, LuDatabase, LuFlaskConical, LuLayers, LuScanLine, LuShieldCheck,
  LuChartColumn, LuGitBranch } from "react-icons/lu";
import { ThemeToggle } from "../components/layout/AppShell";
import { Disclaimer } from "../components/layout/Disclaimer";
import { Logo } from "../components/layout/Logo";
import { Button } from "../components/ui/button";
import { useAuth } from "../lib/auth";
import { useDocumentTitle } from "../lib/hooks";
import { Link } from "../lib/router";

const CAPS = [
  { icon: LuScanLine, title: "Imaging workspace", text: "Upload a CT slice (PNG, JPEG, DICOM, NIfTI) with optional adjacent slices, run a selected model and inspect patch-level scores, explanation maps and quality checks." },
  { icon: LuLayers, title: "Model registry", text: "Every model is integrity-checked by SHA-256 before it serves a prediction. Untrained or failed models are listed with the reason — never replaced by a placeholder." },
  { icon: LuChartColumn, title: "Honest comparison", text: "Metrics come only from saved evaluations, shown with their split and evaluation unit: the V9 fixed split and the V10 nested leave-one-unit-out evaluation." },
  { icon: LuDatabase, title: "Dataset management", text: "Upload or register datasets, validate files, labels, identifiers, duplicates and leakage, and version every manifest." },
  { icon: LuFlaskConical, title: "Experiments", text: "Launch quantum-circuit diagnostics, noise studies and grouped nested evaluations as background jobs; download reports." },
  { icon: LuShieldCheck, title: "Secure by design", text: "Argon2id passwords, rotating refresh tokens, per-user access control, private storage, audit logs and configurable retention." },
];

const MODELS = [
  ["Classical ML", "Logistic regression, decision tree, random forest, extra trees, RBF-SVM, KNN, histogram gradient boosting on radiomics and frozen ImageNet features"],
  ["Deep learning", "2.5-D residual squeeze-excitation CNN trained from scratch; ResNet-50, EfficientNet-B0 and DenseNet-121 with ImageNet weights, last block fine-tuned"],
  ["Quantum ML", "Variational quantum classifier with a dead-parameter-free ansatz and quantum-kernel SVMs (ZZ and product feature maps) on an exact NumPy simulator"],
  ["Hybrid", "Encoder → VQC → head, feature fusion, late fusion, classical twin with matched capacity and ablations (random circuit, no encoder)"],
];

export default function Landing() {
  useDocumentTitle("Research platform");
  const { user } = useAuth();
  return (
    <div className="min-h-screen bg-surface">
      <header className="sticky top-0 z-20 border-b border-line bg-panel/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center gap-3 px-4 sm:px-6">
          <Logo />
          <span className="font-semibold">Liver Tumour Intelligence <span className="font-normal text-muted">V11</span></span>
          <nav className="ml-6 hidden gap-5 text-sm text-muted md:flex" aria-label="Sections">
            <a href="#capabilities" className="hover:text-ink">Capabilities</a>
            <a href="#models" className="hover:text-ink">Models</a>
            <a href="#quantum" className="hover:text-ink">Quantum research</a>
            <a href="#method" className="hover:text-ink">Methodology</a>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <ThemeToggle />
            {user ? <Link to="/app"><Button size="sm">Open dashboard</Button></Link> : (
              <>
                <Link to="/login"><Button variant="ghost" size="sm">Sign in</Button></Link>
                <Link to="/register"><Button size="sm">Create account</Button></Link>
              </>
            )}
          </div>
        </div>
      </header>

      <section className="relative overflow-hidden bg-navy-900 text-white dark:bg-navy-950">
        <div className="mx-auto grid max-w-6xl gap-10 px-4 py-20 sm:px-6 lg:grid-cols-[1.15fr_0.85fr] lg:py-24">
          <div>
            <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/15 px-3 py-1 text-xs text-navy-100">
              <LuGitBranch className="h-3.5 w-3.5" /> Research platform · classical, deep, quantum and hybrid models
            </p>
            <h1 className="text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">
              Reproducible liver tumour classification research, end to end.
            </h1>
            <p className="mt-5 max-w-xl text-lg text-navy-100/85">
              Run trained models on CT slices, compare them on saved evaluations with their evaluation unit, manage
              datasets and launch quantum experiments — with the limits of the evidence stated next to every number.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to={user ? "/app/workspace" : "/register"}><Button size="lg">Get started <LuArrowRight /></Button></Link>
              <a href="#method"><Button size="lg" variant="outline" className="border-white/25 bg-transparent text-white hover:bg-white/10">Read the methodology</Button></a>
            </div>
          </div>
          <div className="card border-white/10 bg-white/[0.04] p-6 text-sm text-navy-100 shadow-none">
            <p className="text-xs font-semibold uppercase tracking-wider text-brand-300">What the platform will not do</p>
            <ul className="mt-3 space-y-2.5">
              <li>• Present a model score as a diagnosis or as calibrated clinical confidence.</li>
              <li>• Show a metric that was not produced by a saved evaluation.</li>
              <li>• Claim patient-level performance: the data have no patient identifiers.</li>
              <li>• Label a simulation as quantum hardware, or claim a quantum advantage without evidence.</li>
              <li>• Return a prediction from a model that is missing, untrained or fails its integrity check.</li>
            </ul>
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <Disclaimer className="-mt-6 relative" />
      </div>

      <section id="capabilities" className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
        <h2 className="text-2xl font-semibold">Platform capabilities</h2>
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {CAPS.map((c) => (
            <div key={c.title} className="card p-5">
              <c.icon className="h-6 w-6 text-brand-600 dark:text-brand-400" aria-hidden />
              <h3 className="mt-3 font-semibold">{c.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{c.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="models" className="border-y border-line bg-panel">
        <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
          <h2 className="text-2xl font-semibold">AI / ML model overview</h2>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {MODELS.map(([t, d]) => (
              <div key={t} className="rounded-xl border border-line p-5">
                <h3 className="flex items-center gap-2 font-semibold"><LuBrainCircuit className="h-5 w-5 text-navy-500" /> {t}</h3>
                <p className="mt-1.5 text-sm text-muted">{d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="quantum" className="mx-auto grid max-w-6xl gap-8 px-4 py-16 sm:px-6 lg:grid-cols-2">
        <div>
          <h2 className="flex items-center gap-2 text-2xl font-semibold"><LuAtom className="h-6 w-6 text-brand-600" /> Quantum and hybrid research</h2>
          <p className="mt-3 text-muted">Quantum models are compared with classical controls that see exactly the same inputs, under the same
            nested evaluation, seeds and tuning budget. Circuit resources, kernel geometry, gradient variance and noise
            sensitivity are reported alongside accuracy.</p>
        </div>
        <ul className="space-y-3 text-sm">
          <li className="card p-4"><b>Execution mode is always shown.</b> Every result says whether it came from the custom NumPy simulator, a PennyLane or Qiskit simulator, or hardware — this installation has simulators only.</li>
          <li className="card p-4"><b>No advantage is assumed.</b> Pre-declared comparisons with exact sign-flip tests across scan-group units; a non-significant difference is never reported as equivalence.</li>
          <li className="card p-4"><b>Noise is modelled, not hidden.</b> Density-matrix simulation with depolarising and readout noise and finite shots, with hypothetical (not device-calibrated) parameters.</li>
        </ul>
      </section>

      <section id="method" className="border-t border-line bg-panel">
        <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
          <h2 className="text-2xl font-semibold">Dataset and research methodology</h2>
          <div className="mt-6 grid gap-6 text-sm leading-relaxed text-muted md:grid-cols-3">
            <p><b className="text-ink">Data.</b> 1,745 axial CT slices from the LiTS benchmark (via a public Hugging Face derivative, CC BY-NC-SA 4.0), labels re-derived from expert masks, and 2,449 liver-tissue patches.</p>
            <p><b className="text-ink">Units.</b> No patient identifiers exist; slices are grouped into reconstructed scan groups (8 evaluation units). Results are patch-level within units and are never described as patient-level.</p>
            <p><b className="text-ink">Evaluation.</b> Nested leave-one-unit-out cross-validation; preprocessing, tuning, calibration and thresholds chosen on training units only; five seeds; per-unit and pooled metrics.</p>
          </div>
        </div>
      </section>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-6 text-xs text-muted sm:px-6">
          <span>Liver Tumour Intelligence Platform V11 · academic research software</span>
          <span>LiTS: Bilic et al., Medical Image Analysis 84 (2023) 102680</span>
        </div>
      </footer>
    </div>
  );
}

export type Role = "admin" | "researcher" | "viewer";

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: Role;
  is_active: boolean;
  created_at: string;
  last_login_at?: string | null;
  preferences: Record<string, unknown>;
}

export interface TokenOut {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  refresh_token: string;
  refresh_expires_at: string;
  user: User;
}

export interface PageMeta { page: number; page_size: number; total: number; pages: number }
export interface Page<T> { items: T[]; meta: PageMeta }

export interface Evaluation {
  name?: string;
  status?: string;
  split?: Record<string, string[]> | string;
  evaluation_unit?: string;
  metrics?: Record<string, number | number[] | null | Record<string, number>>;
  per_unit_auc?: Record<string, number | null>;
  per_group_auc?: Record<string, number | null>;
  caveats?: string[];
  units?: string[];
}

export interface Model {
  id: string;
  name: string;
  family: string;
  architecture: string;
  task: string;
  description: string;
  version: string;
  source: string;
  status: string;
  deployment_status: string;
  servable: boolean;
  reason?: string | null;
  training_status?: string | null;
  training_date?: string | null;
  dataset_version?: string | null;
  execution_mode?: string | null;
  artifact_path?: string | null;
  artifact_sha256?: string | null;
  size_bytes?: number | null;
  role?: string | null;
  input_contract: { requires_context?: boolean; context_slices_each_side?: number; formats?: string[]; features?: string };
  metrics: { evaluations?: Evaluation[] };
  metadata: Record<string, any>;
}

export interface Artifact { kind: string; mime: string; size_bytes: number; url: string }

export interface PredictionSummary {
  id: string;
  user_id: string;
  model_id: string;
  model_name?: string | null;
  model_version?: string | null;
  status: "queued" | "running" | "succeeded" | "failed";
  label?: string | null;
  probability?: number | null;
  threshold?: number | null;
  input_name?: string | null;
  input_format?: string | null;
  processing_ms?: number | null;
  execution_mode?: string | null;
  created_at: string;
  finished_at?: string | null;
}

export interface Prediction extends PredictionSummary {
  dataset_version?: string | null;
  input_sha256_prefix?: string | null;
  input_bytes?: number | null;
  started_at?: string | null;
  expires_at?: string | null;
  error?: string | null;
  result: Record<string, any>;
  artifacts: Artifact[];
  disclaimer: string;
}

export interface DatasetVersion {
  id: string;
  version: number;
  status: string;
  n_files: number;
  content_sha256?: string | null;
  manifest_columns: string[];
  validation: Record<string, any>;
  statistics: Record<string, any>;
  notes: string;
  created_at: string;
  validated_at?: string | null;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  owner_id?: string | null;
  visibility: string;
  source: string;
  license: string;
  provenance: Record<string, any>;
  created_at: string;
  updated_at: string;
  latest_version?: DatasetVersion | null;
  versions: DatasetVersion[];
}

export interface DatasetFile {
  row_index: number;
  image_path: string;
  label?: string | null;
  patient_id?: string | null;
  scan_id?: string | null;
  volume_id?: string | null;
  slice_id?: string | null;
  modality?: string | null;
  split?: string | null;
  sha256?: string | null;
  width?: number | null;
  height?: number | null;
  file_status: string;
  issues: string[];
}

export interface Experiment {
  id: string;
  name: string;
  kind: string;
  source: string;
  status: string;
  owner_id?: string | null;
  config: Record<string, any>;
  error?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  has_results: boolean;
}

export interface SystemInfo {
  app: string;
  version: string;
  environment: string;
  disclaimer: string;
  database: { dialect: string };
  storage: Record<string, any>;
  worker_mode: string;
  engine: Record<string, any>;
  optional_packages: Record<string, boolean>;
  execution_modes: { id: string; label: string; available: boolean; hardware: boolean; note?: string }[];
  registration_open: boolean;
  password_reset_available: boolean;
  limits: { max_upload_mb: number; max_dataset_upload_mb: number; retention_days: number; image_formats: string[] };
}

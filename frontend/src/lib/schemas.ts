import { z } from "zod";

const email = z.string().trim().min(1, "E-mail is required").email("Enter a valid e-mail address").max(254);
export const passwordRule = z.string().min(10, "At least 10 characters").max(256)
  .refine((v) => [/[a-z]/, /[A-Z]/, /\d/].filter((r) => r.test(v)).length >= 2,
    "Mix at least two of: lower-case, upper-case, digits");

export const loginSchema = z.object({ email, password: z.string().min(1, "Password is required") });
export const registerSchema = z.object({
  full_name: z.string().trim().min(1, "Your name is required").max(120),
  email,
  password: passwordRule,
  confirm: z.string(),
  accept: z.boolean().refine((v) => v, "Please confirm you understand this is a research platform"),
}).refine((v) => v.password === v.confirm, { message: "Passwords do not match", path: ["confirm"] });
export const forgotSchema = z.object({ email });
export const resetSchema = z.object({ new_password: passwordRule, confirm: z.string() })
  .refine((v) => v.new_password === v.confirm, { message: "Passwords do not match", path: ["confirm"] });
export const changePasswordSchema = z.object({ current_password: z.string().min(1, "Required"),
  new_password: passwordRule, confirm: z.string() })
  .refine((v) => v.new_password === v.confirm, { message: "Passwords do not match", path: ["confirm"] });
export const profileSchema = z.object({ full_name: z.string().trim().min(1, "Required").max(120) });
export const datasetSchema = z.object({
  name: z.string().trim().min(1, "Name is required").max(120),
  description: z.string().max(2000),
  license: z.string().max(200),
  data_source: z.string().max(500),
});
export const experimentSchema = z.object({
  name: z.string().trim().min(1, "Name is required").max(120),
  kind: z.enum(["quantum_diagnostics", "noise_study", "nested_evaluation"]),
  n_qubits: z.coerce.number().int().min(2, "2–8").max(8, "2–8"),
  layers: z.coerce.number().int().min(1, "1–4").max(4, "1–4"),
  entangler: z.enum(["brick", "ring", "linear", "full"]),
  dataset_id: z.string(),
});

export const MAX_IMAGE_MB_DEFAULT = 20;
export const IMAGE_ACCEPT = ".png,.jpg,.jpeg,.dcm,.dicom,.nii,.gz";

export function checkImageFile(f: File, maxMb: number): string | null {
  const n = f.name.toLowerCase();
  if (!/\.(png|jpe?g|dcm|dicom|nii|nii\.gz)$/.test(n)) return "Unsupported type: use PNG, JPEG, DICOM (.dcm) or NIfTI (.nii / .nii.gz)";
  if (f.size === 0) return "The file is empty";
  if (f.size > maxMb * 1024 * 1024) return `The file is larger than ${maxMb} MB`;
  return null;
}

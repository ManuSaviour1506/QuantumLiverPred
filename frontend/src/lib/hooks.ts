import { useEffect, useState } from "react";
import { fetchBlob } from "./api";

/** Load an authenticated image (prediction artifacts are private) as an object URL. */
export function useAuthImage(path: string | null | undefined): { src: string | null; error: boolean } {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    if (!path) return;
    let url: string | null = null;
    let alive = true;
    fetchBlob(path).then((b) => {
      if (!alive) return;
      url = URL.createObjectURL(b);
      setSrc(url);
    }).catch(() => alive && setError(true));
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [path]);
  return { src, error };
}

export function useObjectUrl(file: File | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file || !/\.(png|jpe?g)$/i.test(file.name)) { setUrl(null); return; }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url;
}

export function useDocumentTitle(title: string) {
  useEffect(() => { document.title = `${title} · Liver Tumour Intelligence`; }, [title]);
}

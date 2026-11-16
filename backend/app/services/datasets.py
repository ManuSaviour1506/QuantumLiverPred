"""Datasets: upload (zip with manifest.csv) or registration of a server-side folder, versioning, validation as a
background job, manifest viewer and statistics. Image files stay in storage or in the registered folder; rows hold
paths, labels, identifiers and hashes only."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from app.core.apikit import UploadedFile
from app.core.errors import ApiError, forbidden, not_found
from app.core.security import iso, new_id, utcnow
from app.db.database import loads
from app.services import audit

log = logging.getLogger("lti.datasets")

FILE_COLS = ["image_path", "label", "patient_id", "scan_id", "volume_id", "slice_id", "modality", "mask_path",
             "split", "data_source"]


class DatasetService:
    def __init__(self, settings, db, storage, jobs):
        self.settings, self.db, self.storage, self.jobs = settings, db, storage, jobs

    # ------------------------------------------------------------------ access
    def _visible(self, user: dict) -> tuple[str, list]:
        if user["role"] == "admin":
            return "", []
        return " WHERE (d.owner_id = ? OR d.visibility = 'shared')", [user["id"]]

    def _get_row(self, c, user: dict, dataset_id: str, write: bool = False) -> dict:
        row = c.one("SELECT * FROM datasets d WHERE d.id = ?", (dataset_id,))
        if row is None:
            raise not_found("Dataset")
        if user["role"] != "admin" and row["owner_id"] != user["id"]:
            if write or row["visibility"] != "shared":
                raise not_found("Dataset")
        return row

    @staticmethod
    def _version_out(v: dict) -> dict:
        return {"id": v["id"], "version": v["version"], "status": v["status"], "n_files": v["n_files"],
                "content_sha256": v["content_sha256"], "manifest_columns": loads(v["manifest_columns"], []),
                "validation": loads(v["validation"], {}), "statistics": loads(v["statistics"], {}),
                "preprocessing": loads(v["preprocessing"], {}), "notes": v["notes"], "created_at": v["created_at"],
                "validated_at": v["validated_at"]}

    def _out(self, c, row: dict, with_versions: bool = True) -> dict:
        vs = c.all("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version DESC", (row["id"],))
        versions = [self._version_out(v) for v in vs]
        return {"id": row["id"], "name": row["name"], "description": row["description"], "owner_id": row["owner_id"],
                "visibility": row["visibility"], "source": row["source"], "license": row["license"],
                "provenance": loads(row["provenance"], {}), "created_at": row["created_at"],
                "updated_at": row["updated_at"], "latest_version": versions[0] if versions else None,
                "versions": versions if with_versions else []}

    def list(self, user: dict) -> list[dict]:
        where, params = self._visible(user)
        with self.db.session() as c:
            rows = c.all("SELECT * FROM datasets d" + where + " ORDER BY d.created_at DESC", params)
            return [self._out(c, r, with_versions=False) for r in rows]

    def get(self, user: dict, dataset_id: str) -> dict:
        with self.db.session() as c:
            return self._out(c, self._get_row(c, user, dataset_id))

    # ------------------------------------------------------------------ creation
    def _insert_rows(self, c, version_id: str, rows: list[dict]) -> None:
        for i, r in enumerate(rows):
            c.run("INSERT INTO dataset_files (id, dataset_version_id, row_index, image_path, label, patient_id, scan_id, "
                  "volume_id, slice_id, modality, mask_path, split, data_source) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (new_id("df_"), version_id, i, *[r.get(k) or None for k in FILE_COLS]))

    def _new_version(self, c, dataset_id: str, user_id: str | None, storage_key: str | None, rows: list[dict],
                     cols: list[str], notes: str = "", extra_rows_meta: dict | None = None) -> str:
        from src.v10.data.ingest import content_hash
        n = int(c.scalar("SELECT COALESCE(MAX(version), 0) FROM dataset_versions WHERE dataset_id = ?",
                         (dataset_id,)) or 0) + 1
        vid = new_id("dv_")
        c.run("INSERT INTO dataset_versions (id, dataset_id, version, status, storage_key, content_sha256, n_files, "
              "manifest_columns, notes, created_by, created_at, preprocessing) "
              "VALUES (?, ?, ?, 'uploaded', ?, ?, ?, ?, ?, ?, ?, ?)",
              (vid, dataset_id, n, storage_key, content_hash(rows), len(rows), cols, notes, user_id, iso(utcnow()),
               extra_rows_meta or {}))
        self._insert_rows(c, vid, rows)
        return vid

    def upload(self, user: dict, file: UploadedFile, name: str, description: str, license_: str, data_source: str,
               ip: str | None) -> dict:
        from src.v10.data.ingest import IngestError, find_manifest, read_manifest, safe_extract
        if user["role"] == "viewer":
            raise forbidden("Viewers cannot upload datasets")
        if not file.filename.lower().endswith(".zip"):
            raise ApiError(415, "unsupported_media_type", "Upload a .zip archive containing the images and manifest.csv")
        did = new_id("ds_")
        tmp = Path(tempfile.mkdtemp(dir=Path(self.settings.storage_dir) / "tmp"))
        try:
            try:
                files = safe_extract(file.path, tmp)
            except (IngestError, OSError, Exception) as e:
                raise ApiError(422, "invalid_archive", f"Archive rejected: {e}")
            man = find_manifest(tmp)
            if man is None:
                raise ApiError(422, "manifest_missing", "The archive has no manifest.csv; see DATASET_GUIDE.md for "
                                                       "the required columns (image_path, label, ...)")
            base = man.parent
            try:
                rows, cols, errors = read_manifest(man)
            except IngestError as e:
                raise ApiError(422, "invalid_manifest", str(e))
            key = f"datasets/{did}/v1"
            for rel in files:
                src = tmp / rel
                try:
                    inner = src.resolve().relative_to(base.resolve())
                except ValueError:
                    continue
                self.storage.put_file(f"{key}/files/{inner.as_posix()}", src)
            self.storage.put_file(f"{key}/upload.zip", file.path)
            now = iso(utcnow())
            with self.db.session() as c:
                with c.transaction():
                    c.run("INSERT INTO datasets (id, name, description, owner_id, visibility, source, license, "
                          "provenance, created_at, updated_at) VALUES (?, ?, ?, ?, 'private', 'upload', ?, ?, ?, ?)",
                          (did, name, description, user["id"], license_,
                           {"uploaded_archive_sha256": file.sha256, "data_source": data_source,
                            "manifest_errors": errors[:50]}, now, now))
                    vid = self._new_version(c, did, user["id"], key, rows, cols, "uploaded archive")
                    self.jobs.enqueue(c, "dataset_validation", vid)
                    audit.log(c, "dataset.uploaded", user["id"], "dataset", did, ip, {"rows": len(rows)},
                              secret=self.settings.jwt_secret)
            self.jobs.wake()
            return self.get(user, did)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def register_local(self, actor: dict | None, path: Path, name: str, layout: str = "lits_verified",
                       visibility: str = "shared", license_: str = "", validate_now: bool = False) -> dict:
        """Register a dataset folder already on the server (admin / CLI). The folder must be inside DATASET_ROOTS."""
        from src.v10.data.ingest import find_manifest, lits_verified_manifest, read_manifest
        path = Path(path).expanduser().resolve()
        roots = [Path(r).expanduser().resolve() for r in self.settings.dataset_roots]
        if not any(path == r or r in path.parents for r in roots):
            raise forbidden("Folder is not inside a configured DATASET_ROOTS location")
        if not path.is_dir():
            raise not_found("Dataset folder")
        if layout == "lits_verified":
            rows = lits_verified_manifest(path)
            cols = FILE_COLS
            provenance = {"layout": "lits_verified", "path_hint": path.name,
                          "source": "seongun/liver-tumor-classification (Hugging Face), derived from LiTS",
                          "citation": "Bilic et al., Medical Image Analysis 84, 102680 (2023)",
                          "identifiers": "no patient or volume identifiers in the source; scan_id = reconstructed "
                                         "scan group"}
            license_ = license_ or "CC BY-NC-SA 4.0"
        else:
            man = find_manifest(path)
            if man is None:
                raise ApiError(422, "manifest_missing", "No manifest.csv in the folder")
            rows, cols, _ = read_manifest(man)
            provenance = {"layout": "manifest", "path_hint": path.name}
        did = new_id("ds_")
        now = iso(utcnow())
        with self.db.session() as c:
            with c.transaction():
                c.run("INSERT INTO datasets (id, name, description, owner_id, visibility, source, license, provenance, "
                      "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'registered_folder', ?, ?, ?, ?)",
                      (did, name, "Registered server-side folder", actor["id"] if actor else None, visibility,
                       license_, {**provenance, "root": str(path)}, now, now))
                vid = self._new_version(c, did, actor["id"] if actor else None, None, rows, cols,
                                        f"registered from folder ({layout})",
                                        {"expected_pixel_sha12": [r.get("_expected_sha12", "") for r in rows]}
                                        if layout == "lits_verified" else None)
                self.jobs.enqueue(c, "dataset_validation", vid)
                audit.log(c, "dataset.registered", actor["id"] if actor else None, "dataset", did, None,
                          {"rows": len(rows), "layout": layout}, secret=self.settings.jwt_secret)
        self.jobs.wake()
        if validate_now:
            self.execute_validation(vid)
        with self.db.session() as c:
            return self._out(c, c.one("SELECT * FROM datasets WHERE id = ?", (did,)))

    # ------------------------------------------------------------------ validation
    def request_validation(self, user: dict, dataset_id: str, version: int | None, ip: str | None) -> dict:
        with self.db.session() as c:
            self._get_row(c, user, dataset_id, write=user["role"] != "admin")
            if version is None:
                v = c.one("SELECT id FROM dataset_versions WHERE dataset_id = ? ORDER BY version DESC LIMIT 1",
                          (dataset_id,))
            else:
                v = c.one("SELECT id FROM dataset_versions WHERE dataset_id = ? AND version = ?", (dataset_id, version))
            if v is None:
                raise not_found("Dataset version")
            c.run("UPDATE dataset_versions SET status = 'validating' WHERE id = ?", (v["id"],))
            jid = self.jobs.enqueue(c, "dataset_validation", v["id"])
            audit.log(c, "dataset.validation_requested", user["id"], "dataset", dataset_id, ip,
                      secret=self.settings.jwt_secret)
        self.jobs.wake()
        return {"job_id": jid, "status": "queued", "message": "Validation queued; refresh the dataset to see results"}

    def _root_of(self, c, v: dict) -> Path:
        ds = c.one("SELECT source, provenance FROM datasets WHERE id = ?", (v["dataset_id"],))
        if ds["source"] == "registered_folder":
            return Path(loads(ds["provenance"], {})["root"])
        if v["storage_key"] is None:
            raise RuntimeError("dataset version has no storage location")
        if self.storage.kind != "local":
            raise RuntimeError("validation of S3-stored datasets needs a local copy (not implemented offline)")
        return self.storage.path(f"{v['storage_key']}/files")

    def execute_validation(self, version_id: str) -> None:
        from src.v10.data.ingest import content_hash, validate
        with self.db.session() as c:
            v = c.one("SELECT * FROM dataset_versions WHERE id = ?", (version_id,))
            if v is None:
                return
            c.run("UPDATE dataset_versions SET status = 'validating' WHERE id = ?", (version_id,))
            files = c.all("SELECT id, row_index, " + ", ".join(FILE_COLS) +
                          " FROM dataset_files WHERE dataset_version_id = ? ORDER BY row_index", (version_id,))
            root = self._root_of(c, v)
        rows = [{k: (f[k] or "") for k in FILE_COLS} for f in files]
        expected = loads(v["preprocessing"], {}).get("expected_pixel_sha12")
        if expected and len(expected) == len(rows):
            for r, e in zip(rows, expected):
                r["_expected_sha12"] = e
        try:
            report, per_file = validate(rows, root)
        except Exception as e:
            with self.db.session() as c:
                c.run("UPDATE dataset_versions SET status = 'failed', validation = ?, validated_at = ? WHERE id = ?",
                      ({"passed": False, "error": f"{type(e).__name__}: {e}"}, iso(utcnow()), version_id))
            raise
        stats = {k: report[k] for k in ("label_distribution", "split_distribution", "group_distribution",
                                        "label_by_split", "image_sizes", "intensity", "identifier_availability")}
        status = "validated" if report["passed"] else ("missing_files" if report["file_status"].get("missing")
                                                       else "validation_failed")
        with self.db.session() as c:
            with c.transaction():
                for f, pf in zip(files, per_file):
                    c.run("UPDATE dataset_files SET file_status = ?, issues = ?, sha256 = ?, width = ?, height = ? "
                          "WHERE id = ?", (pf["file_status"], pf["issues"], pf["sha256"], pf["width"], pf["height"],
                                           f["id"]))
                c.run("UPDATE dataset_versions SET status = ?, validation = ?, statistics = ?, validated_at = ?, "
                      "content_sha256 = ? WHERE id = ?",
                      (status, report, stats, iso(utcnow()), content_hash(rows, per_file), version_id))

    # ------------------------------------------------------------------ viewer and statistics
    def statistics(self, user: dict, dataset_id: str) -> dict:
        d = self.get(user, dataset_id)
        v = d["latest_version"] or {}
        return {"dataset_id": dataset_id, "version": v.get("version"), "status": v.get("status"),
                "n_files": v.get("n_files"), "statistics": v.get("statistics") or {},
                "validation": v.get("validation") or {}}

    def files(self, user: dict, dataset_id: str, q) -> tuple[list[dict], int]:
        with self.db.session() as c:
            self._get_row(c, user, dataset_id)
            if q.version:
                v = c.one("SELECT id FROM dataset_versions WHERE dataset_id = ? AND version = ?", (dataset_id, q.version))
            else:
                v = c.one("SELECT id FROM dataset_versions WHERE dataset_id = ? ORDER BY version DESC LIMIT 1",
                          (dataset_id,))
            if v is None:
                return [], 0
            where, params = ["dataset_version_id = ?"], [v["id"]]
            if q.status:
                where.append("file_status = ?")
                params.append(q.status)
            if q.label:
                where.append("label = ?")
                params.append(q.label)
            if q.q:
                where.append("(LOWER(image_path) LIKE ? OR LOWER(COALESCE(scan_id, '')) LIKE ?)")
                params += [f"%{q.q.lower()}%"] * 2
            w = " WHERE " + " AND ".join(where)
            total = int(c.scalar("SELECT COUNT(*) FROM dataset_files" + w, params) or 0)
            rows = c.all("SELECT * FROM dataset_files" + w + " ORDER BY row_index LIMIT ? OFFSET ?",
                         (*params, q.page_size, (q.page - 1) * q.page_size))
        for r in rows:
            r["issues"] = loads(r["issues"], [])
        return rows, total

    def delete(self, user: dict, dataset_id: str, ip: str | None) -> None:
        with self.db.session() as c:
            row = self._get_row(c, user, dataset_id, write=True)
            with c.transaction():
                c.run("DELETE FROM datasets WHERE id = ?", (dataset_id,))
                audit.log(c, "dataset.deleted", user["id"], "dataset", dataset_id, ip, secret=self.settings.jwt_secret)
        if row["source"] == "upload":
            self.storage.delete_prefix(f"datasets/{dataset_id}")

    def patch_dataset_root(self, dataset_id: str) -> Path | None:
        """Folder of a registered LiTS-layout dataset (used by experiments that need the patch arrays)."""
        with self.db.session() as c:
            ds = c.one("SELECT source, provenance FROM datasets WHERE id = ?", (dataset_id,))
        if ds and ds["source"] == "registered_folder":
            p = Path(loads(ds["provenance"], {}).get("root", ""))
            if (p / "patches" / "patches_v8.npz").is_file():
                return p
        return None

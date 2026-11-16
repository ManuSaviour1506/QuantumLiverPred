"""Dataset upload, validation job, statistics, manifest viewer, archive safety and folder registration."""
import io
import zipfile

import pytest

from conftest import wait_for

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v10.data.synthetic import dataset_zip, make_dataset  # noqa: E402


def _validated(client, did):
    def check():
        d = client.get(f"/api/v1/datasets/{did}").json()
        v = d["latest_version"]
        return d if v and v["status"] not in ("uploaded", "validating") else None
    return wait_for(check, 120)


def test_upload_validate_and_view(researcher, viewer):
    entries = make_dataset(10, 4)
    z = dataset_zip(entries)
    r = researcher.post("/api/v1/datasets/upload", files={"file": ("syn.zip", z, "application/zip")},
                        data={"name": "Synthetic phantoms", "license": "test fixture", "data_source": "synthetic"})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    d = _validated(researcher, did)
    v = d["latest_version"]
    assert v["status"] == "validated", v
    val = v["validation"]
    assert val["passed"] and val["rows"] == 20 and val["file_status"]["ok"] == 20
    assert val["label_distribution"] == {"phantom_A": 10, "phantom_B": 10}
    assert any("patient identifiers" in w for w in val["warnings"])
    assert val["leakage"]["scan_id"]["ids_in_multiple_splits"] == 0
    stats = researcher.get(f"/api/v1/datasets/{did}/statistics").json()
    assert stats["statistics"]["group_distribution"]
    files = researcher.get(f"/api/v1/datasets/{did}/files?page_size=5").json()
    assert files["meta"]["total"] == 20 and len(files["items"]) == 5 and files["items"][0]["sha256"]
    assert viewer.get(f"/api/v1/datasets/{did}").status_code == 404
    assert viewer.post("/api/v1/datasets/upload", files={"file": ("syn.zip", z, "application/zip")},
                       data={"name": "x"}).status_code == 403
    assert researcher.post(f"/api/v1/datasets/{did}/validate", json={}).status_code == 202


def test_missing_files_duplicates_and_leakage(researcher):
    entries = make_dataset(6, 3)
    rows = [e[2] for e in entries]
    rows[1] = dict(rows[1], image_path="images/does_not_exist.png")
    dup_name, dup_data, dup_row = entries[0]
    extra = {"images/copy_of_first.png": dup_data}
    rows.append(dict(dup_row, image_path="images/copy_of_first.png", split="test"))
    rows[2] = dict(rows[2], label="")
    import csv
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    z = dataset_zip(entries, {**extra, "manifest.csv": s.getvalue().encode()}, manifest=False)
    did = researcher.post("/api/v1/datasets/upload", files={"file": ("bad.zip", z, "application/zip")},
                          data={"name": "faulty"}).json()["id"]
    v = _validated(researcher, did)["latest_version"]
    val = v["validation"]
    assert v["status"] == "missing_files" and not val["passed"]
    assert val["file_status"]["missing"] == 1
    assert val["label_distribution"]["(missing)"] == 1
    assert val["exact_duplicate_groups"] >= 1
    assert val["leakage"]["scan_id"]["ids_in_multiple_splits"] >= 1
    assert val["unlisted_images"] >= 1


def test_archive_safety(researcher):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.png", b"x")
        z.writestr("manifest.csv", "image_path,label\n../../evil.png,a\n")
    r = researcher.post("/api/v1/datasets/upload", files={"file": ("evil.zip", buf.getvalue(), "application/zip")},
                        data={"name": "evil"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_archive"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("images/a.png", b"x")
    r = researcher.post("/api/v1/datasets/upload", files={"file": ("nomanifest.zip", buf.getvalue(), "application/zip")},
                        data={"name": "nomanifest"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "manifest_missing"
    r = researcher.post("/api/v1/datasets/upload", files={"file": ("x.tar", b"abc", "application/x-tar")},
                        data={"name": "x"})
    assert r.status_code == 415


def test_register_real_sample_folder(live, admin, researcher):
    ds = live["dataset"]
    if ds is None:
        pytest.skip("real dataset sample not available")
    assert researcher.post("/api/v1/datasets/register",
                           json={"path": str(ds), "name": "x", "layout": "lits_verified"}).status_code == 403
    assert admin.post("/api/v1/datasets/register",
                      json={"path": "/etc", "name": "x", "layout": "manifest"}).status_code == 403
    r = admin.post("/api/v1/datasets/register", json={"path": str(ds), "name": "LiTS verified sample",
                                                      "layout": "lits_verified"})
    assert r.status_code == 201, r.text
    d = _validated(admin, r.json()["id"])
    val = d["latest_version"]["validation"]
    n = val["rows"]
    assert val["file_status"].get("ok", 0) + val["file_status"].get("warning", 0) == n
    assert val["identifier_availability"]["patient_id"] == 0
    assert val["pixel_hash_verification"]["checked"] == n and val["pixel_hash_verification"]["matched"] == n
    assert d["license"] == "CC BY-NC-SA 4.0"
    # shared registered dataset is visible to researchers
    assert researcher.get(f"/api/v1/datasets/{d['id']}").status_code == 200

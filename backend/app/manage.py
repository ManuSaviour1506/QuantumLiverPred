"""Administration CLI.

    python -m app.manage create-admin --email admin@example.org --name "Admin"   (password from LTI_ADMIN_PASSWORD
                                                                                  or an interactive prompt)
    python -m app.manage sync-models          load the engine and mirror the model registry into the database
    python -m app.manage register-dataset --path ../dataset/liver_ct_lits_verified_v8 --name "LiTS verified (V8)" [--validate-now] [--if-missing]
    python -m app.manage purge-expired        delete predictions past RETENTION_DAYS (with their images)
    python -m app.manage import-recorded      import saved result files as recorded experiments
    python -m app.manage run-jobs             process queued jobs once and exit
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys

from app.core.config import Settings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.manage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("create-admin")
    a.add_argument("--email", required=True)
    a.add_argument("--name", default="Administrator")
    sub.add_parser("sync-models")
    r = sub.add_parser("register-dataset")
    r.add_argument("--path", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--layout", default="lits_verified", choices=["lits_verified", "manifest"])
    r.add_argument("--validate-now", action="store_true")
    r.add_argument("--if-missing", action="store_true", help="do nothing if a dataset with this name exists")
    sub.add_parser("purge-expired")
    sub.add_parser("import-recorded")
    sub.add_parser("run-jobs")
    args = ap.parse_args(argv)

    from app.main import create_app
    app = create_app(Settings.from_env(), start_background=False)
    st = app.state
    if args.cmd == "create-admin":
        from app.schemas.auth import normalise_email
        pw = os.environ.get("LTI_ADMIN_PASSWORD") or getpass.getpass("Password for the admin account: ")
        with st.db.session() as c:
            with c.transaction():
                row = st.auth_service.create_user(c, normalise_email(args.email), pw, args.name, "admin")
        print(f"admin account created: {row['id']}")
    elif args.cmd == "sync-models":
        print(st.model_service.sync())
        print(st.engine.status())
    elif args.cmd == "register-dataset":
        from pathlib import Path
        if args.if_missing:
            with st.db.session() as c:
                existing = c.scalar("SELECT id FROM datasets WHERE name = ?", (args.name,))
            if existing:
                print(f"dataset '{args.name}' already registered ({existing})")
                return 0
        d = st.dataset_service.register_local(None, Path(args.path), args.name, args.layout,
                                              validate_now=args.validate_now)
        v = d["latest_version"]
        print(f"registered {d['id']} version {v['version']}: {v['n_files']} files, status {v['status']}")
    elif args.cmd == "purge-expired":
        print(f"deleted {st.prediction_service.purge_expired()} expired predictions")
    elif args.cmd == "import-recorded":
        print(f"imported {st.experiment_service.import_recorded()} recorded experiments")
    elif args.cmd == "run-jobs":
        print(f"processed {st.jobs.drain()} jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

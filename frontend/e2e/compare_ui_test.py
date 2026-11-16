"""Browser end-to-end test of the four-paradigm comparison flow against a running server (real API, real models).

    E2E_BASE_URL=http://127.0.0.1:8765 python frontend/e2e/compare_ui_test.py      # needs: pip install playwright
                                                                                   # + a Chromium/Chrome browser
Checks: sign-up/sign-in, sample upload with validation panel, Run all models, the four family panels appear and reach
"Completed" without a page refresh, fastest / slowest family are shown, the live stream was used, history lists the
run, registry and benchmarking pages load, V11: resource card and fastest / slowest model on the live page, the
system monitor shows measured metrics, a new evaluation run started from the UI completes with per-role metrics kept
apart from the historical results, and the browser console has no errors. Screenshots go to e2e/output/.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8765")
OUT = Path(os.environ.get("E2E_OUT", Path(__file__).parent / "output"))
CHROME = os.environ.get("E2E_CHROME")          # optional executable path (e.g. /usr/bin/google-chrome)


async def main() -> int:
    from playwright.async_api import async_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    email, pw = f"e2e_cmp_{int(time.time())}@example.org", "E2e-Test-Pass-42"
    r = requests.post(BASE + "/api/v1/auth/register", json={"email": email, "password": pw, "full_name": "E2E Compare"})
    assert r.status_code in (201, 409), r.text
    async with async_playwright() as p:
        kw = {"executable_path": CHROME, "args": ["--no-sandbox"]} if CHROME else {}
        br = await p.chromium.launch(**kw)
        pg = await br.new_page(viewport={"width": 1500, "height": 1000})
        errors: list[str] = []
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        streamed = []
        pg.on("request", lambda rq: streamed.append(rq.url) if rq.url.endswith("/stream") or "/stream?" in rq.url else None)

        await pg.goto(BASE + "/login")
        await pg.fill("input[type=email]", email)
        await pg.fill("input[type=password]", pw)
        await pg.keyboard.press("Enter")
        await pg.wait_for_url("**/app", timeout=20000)
        await pg.wait_for_selector("text=Registered models")

        await pg.goto(BASE + "/app/upload?tab=sample")
        await pg.click("text=ct_lits_4278.png")
        await pg.wait_for_selector("text=Liver region validation (U-Net)", timeout=60000)
        await pg.wait_for_selector("text=models eligible for this input")
        await pg.screenshot(path=str(OUT / "cmp_01_upload.png"), full_page=True)

        await pg.click("button:has-text('Run all models')")
        await pg.wait_for_url("**/app/live/run_*", timeout=20000)
        for fam in ("Classical ML", "Deep Learning", "Quantum ML", "Hybrid ML"):
            await pg.wait_for_selector(f"section[aria-label='{fam}']", timeout=20000)
        await pg.wait_for_selector("text=Completed", timeout=120000)
        end = time.time() + 180                      # the app's CSP forbids eval, so poll from Python
        while True:
            text = await pg.inner_text("body")
            low = text.lower()                       # inner_text applies CSS text-transform (uppercase labels)
            if "fastest model family" in low and "measuring…" not in low and "ran concurrently for" in low:
                break
            assert time.time() < end, "run did not finish in time"
            await pg.wait_for_timeout(500)
        assert "Executing four family pipelines" not in text or "Completed" in text
        for fam in ("Classical ML", "Deep Learning", "Quantum ML", "Hybrid ML"):
            panel = await pg.inner_text(f"section[aria-label='{fam}']")
            assert "Completed" in panel, f"{fam} panel did not complete: {panel[:200]}"
        assert "ran concurrently for" in text, "concurrency measurement missing"
        low = text.lower()
        assert "resource utilisation during the run" in low, "V11 resource card missing"
        assert "fastest" in low and "slowest" in low, "fastest / slowest model missing"
        await pg.screenshot(path=str(OUT / "cmp_02_live.png"), full_page=True)
        assert streamed, "the live page never opened the event stream"

        for path, needle in (("/app/history", "Saved comparison runs"), ("/app/models", "Every trained model"),
                             ("/app/benchmarks", "Accuracy & latency"), ("/app/runs", "Model results")):
            await pg.goto(BASE + path)
            await pg.wait_for_selector(f"text={needle}", timeout=20000)
        await pg.screenshot(path=str(OUT / "cmp_03_bench.png"), full_page=True)

        # V11: system monitor shows measured process metrics of the running server
        await pg.goto(BASE + "/app/system")
        await pg.wait_for_selector("text=API server & model engine", timeout=20000)
        await pg.wait_for_selector("text=Process CPU", timeout=20000)
        await pg.wait_for_selector("text=Model engine", timeout=20000)
        await pg.screenshot(path=str(OUT / "cmp_04_system.png"), full_page=True)

        # V11: a new evaluation run on the bundled labelled sample, started from the UI, metrics split by role
        await pg.goto(BASE + "/app/benchmarks")
        await pg.wait_for_selector("text=New evaluation runs", timeout=20000)
        await pg.wait_for_selector("button:has-text('Start evaluation'):not([disabled])", timeout=20000)
        await pg.click("button:has-text('Start evaluation')")
        end = time.time() + 900
        while True:
            low = (await pg.inner_text("body")).lower()
            if "roc-auc" in low and "held-out test" in low and "tp/fn/fp/tn" in low:
                break
            assert time.time() < end, "new evaluation run did not finish in time"
            await pg.wait_for_timeout(1000)
        assert "historical (recorded) evaluations" in low, "historical results not kept separate"
        await pg.screenshot(path=str(OUT / "cmp_05_new_evaluation.png"), full_page=True)
        await br.close()
    assert not errors, f"browser console errors: {errors[:5]}"
    print("compare UI e2e: OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

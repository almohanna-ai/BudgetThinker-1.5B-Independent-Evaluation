"""Package every study output into one zip (with a SHA-256 manifest) and download it locally.

Run in Colab after (or during) a run:  %run -i src/bt_export.py
Only completed units are included: unit files are written atomically, *.tmp files are skipped.
Set DOWNLOAD = False beforehand to build the zip without triggering the browser download.
"""
import hashlib
import json
import os
import time
import zipfile

BASE = globals().get("BASE", "/content/BudgetThinker_eval")
DOWNLOAD = globals().get("DOWNLOAD", True)
INCLUDE = ["configs", "raw_generations", "metrics", "reports", "logs", "src"]
NOTEBOOK = "/content/bt_bundle/notebooks/BudgetThinker_Independent_Eval.ipynb"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


stamp = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
out = f"/content/BudgetThinker_eval_results_{stamp}.zip"
files_meta, n_units = [], 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for d in INCLUDE:
        for dp, _, fs in os.walk(os.path.join(BASE, d)):
            for f in sorted(fs):
                if f.endswith(".tmp"):
                    continue
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, BASE).replace(os.sep, "/")
                z.write(p, rel)
                files_meta.append({"path": rel, "bytes": os.path.getsize(p), "sha256": _sha256(p)})
                n_units += f.endswith(".jsonl.gz")
    if os.path.exists(NOTEBOOK):
        z.write(NOTEBOOK, "notebooks/BudgetThinker_Independent_Eval.ipynb")
    z.writestr("MANIFEST.json", json.dumps({
        "created_utc": stamp, "base": BASE, "n_files": len(files_meta),
        "n_generation_units": n_units, "files": files_meta}, indent=1))

with zipfile.ZipFile(out) as z:
    assert z.testzip() is None, "zip integrity check failed"
size_mb = os.path.getsize(out) / 2**20
print(f"{out}\n{len(files_meta)} files, {n_units} generation units, {size_mb:.1f} MB, sha256 {_sha256(out)[:16]}")

if DOWNLOAD:
    try:
        from google.colab import files
        files.download(out)
    except ImportError:
        print("Not running in Colab: zip kept at", out)

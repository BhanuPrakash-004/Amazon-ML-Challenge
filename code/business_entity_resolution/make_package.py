"""Build <team>_submission.zip: output TSVs + code + README/requirements + docs.

Excludes: datasets, cache/, embeddings, FAISS indexes, checkpoints, logs, venvs.
Run from repo root:  python code/business_entity_resolution/make_package.py --team MYTEAM
"""
import argparse
import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "code", "business_entity_resolution")
SKIP_DIRS = {"cache", "__pycache__", ".git", "results", "checkpoints", "models"}
SKIP_EXT = {".npy", ".bin", ".log", ".faiss", ".idx"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "%s_submission.zip" % a.team)
    files = []
    for rel in ("output/matching_results.tsv", "output/candidate_pairs.tsv",
                "code/business_entity_resolution/README.md",
                "code/business_entity_resolution/requirements.txt",
                "Documentation_template.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            raise FileNotFoundError("missing required %s (run pipeline first)" % rel)
        files.append((p, rel))
    for dp, dn, fn in os.walk(os.path.join(SRC, "src")):
        if any(s in dp for s in SKIP_DIRS):
            continue
        for f in fn:
            if f.endswith(".py") and not f.startswith("test_"):
                p = os.path.join(dp, f)
                files.append((p, os.path.relpath(p, ROOT)))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, rel in files:
            z.write(p, rel)
    print("wrote %s (%d files)" % (out, len(files)))


if __name__ == "__main__":
    main()

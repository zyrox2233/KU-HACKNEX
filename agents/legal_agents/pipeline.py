"""
pipeline.py
===========
Thin wrapper over orchestrator.py: Review -> Research -> Drafting -> Verification.

    python pipeline.py --docs fir.txt case.pdf --type bail
    python pipeline.py --docs fir.txt --type bail --save pipeline.json
(Use --save instead of '>' redirection on Windows PowerShell, which writes UTF-16.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestrator import LegalOrchestrator, show


def run(docs, doc_type, verified_by=None, verified_on=None):
    orch = LegalOrchestrator()
    orch.load_documents(docs)
    orch.run_full(doc_type, verified_by, verified_on)
    return orch.export()


def main():
    ap = argparse.ArgumentParser(description="Review -> Research -> Draft -> Verify pipeline")
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--verified-by"); ap.add_argument("--verified-on")
    ap.add_argument("--save", help="Write pipeline JSON to this file")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    out = run(a.docs, a.type, a.verified_by, a.verified_on)
    if a.save:
        Path(a.save).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"Saved pipeline JSON to {a.save}")
    if a.json:
        print(json.dumps(out, indent=2, default=str))
    elif out.get("draft"):
        from legal_drafting_agent import _pretty_print
        from legal_verification_agent import _pretty
        _pretty_print(out["draft"])
        if out.get("verification"):
            _pretty(out["verification"])
        print(f"RELEASE STATUS: {out['release_status']}\n")


if __name__ == "__main__":
    main()

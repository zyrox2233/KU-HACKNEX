"""
pipeline.py
===========
Runs all three agents end to end:  Review -> Research -> Drafting

    python pipeline.py --docs fir.txt case.pdf --type bail
    python pipeline.py --docs fir.txt --type bail --verified-by "Adv. Name" --verified-on 2026-10-08 --json
"""
from __future__ import annotations

import argparse
import json

from legal_research_agent import LegalResearchAgent
from legal_case_review_agent import CaseReviewAgent
from legal_drafting_agent import LegalDraftingAgent, register_case_from_review, _pretty_print


def run(docs, doc_type, case_id="case_from_docs", verified_by=None, verified_on=None):
    research = LegalResearchAgent()                       # one shared instance
    review = CaseReviewAgent(research_agent=research)
    drafter = LegalDraftingAgent(research_agent=research, review_agent=review)

    report = review.review_and_research(docs)             # Agent 1 -> Agent 2
    register_case_from_review(case_id, report, verified_by, verified_on)   # Agent 1 -> Agent 3
    draft = drafter.draft(doc_type, case_id)              # Agent 3 (+ Agent 2 cross-checks)
    return {"review": report, "research": report["research"], "draft": draft}


def main():
    ap = argparse.ArgumentParser(description="Review -> Research -> Draft pipeline")
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--case", default="case_from_docs")
    ap.add_argument("--verified-by"); ap.add_argument("--verified-on")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    out = run(a.docs, a.type, a.case, a.verified_by, a.verified_on)
    if a.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        _pretty_print(out["draft"])


if __name__ == "__main__":
    main()

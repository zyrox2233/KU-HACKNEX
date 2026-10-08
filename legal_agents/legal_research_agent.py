"""
legal_research_agent.py   (Agent 2 of 3)
========================================
Closed-corpus legal research (RAG via TF-IDF). Never invents laws or citations.

Inter-links
  <- legal_case_review_agent : calls research_from_review(report) with a review report
  <- legal_drafting_agent    : calls research_for_case(case) and verify_citation(doc_id)

Run standalone:
    python legal_research_agent.py --query "punishment for murder" --facts "..."
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any

from common import tokenize
from corpus import load_corpus, get_verified_source


# ---------------- Retriever ----------------
class TfidfRetriever:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.doc_tokens = [tokenize(self.doc_text(d)) for d in docs]
        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] += 1
        self.N = max(len(docs), 1)

    @staticmethod
    def doc_text(d: dict[str, Any]) -> str:
        parts = [d.get("title", ""), d.get("section", ""), d.get("text", ""),
                 d.get("holding", ""), d.get("citation", "")]
        return " ".join(p for p in parts if p)

    def _idf(self, tok: str) -> float:
        return math.log((self.N + 1) / (self.df.get(tok, 0) + 1)) + 1.0

    def _score(self, q: list[str], d: list[str]) -> float:
        if not q or not d:
            return 0.0
        qc, dc = Counter(q), Counter(d)
        s = sum((qn * self._idf(t)) * (dc[t] * self._idf(t)) for t, qn in qc.items() if t in dc)
        return s / math.sqrt(len(d) + 1)

    def retrieve(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        q = tokenize(query)
        scored = sorted(((self._score(q, dt), d) for dt, d in zip(self.doc_tokens, self.docs)),
                        key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:k] if s > 0]


# ---------------- Issue spotter ----------------
ISSUE_KEYWORDS: dict[str, list[str]] = {
    "murder": ["murder", "302", "killed", "homicide"],
    "death penalty": ["death penalty", "capital punishment", "rarest of rare"],
    "culpable homicide": ["culpable homicide", "304", "not amounting"],
    "circumstantial evidence": ["circumstantial", "panchsheel", "chain of evidence"],
    "confession": ["confession", "164", "voluntary"],
    "fir": ["fir", "first information", "154", "cognizable"],
    "bail": ["bail", "437", "438", "439", "anticipatory"],
    "appeal": ["appeal", "374", "386"],
}


def spot_issues(query: str) -> list[str]:
    q = query.lower()
    found = [i for i, kws in ISSUE_KEYWORDS.items() if any(k in q for k in kws)]
    return found or ["general legal query"]


# ---------------- Verifier ----------------
def _claim_matches(claim: str, doc: dict[str, Any]) -> bool:
    ct = set(tokenize(claim))
    dt = set(tokenize(TfidfRetriever.doc_text(doc)))
    return bool(ct) and len(ct & dt) / len(ct) >= 0.4


def verify_claim(claim: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    for doc in docs:
        if doc.get("verified") is True and _claim_matches(claim, doc):
            return {"claim": claim, "verified": True, "source_id": doc["doc_id"],
                    "citation": doc.get("citation")
                    or f"Section {doc.get('section','')} {doc.get('title','')}".strip()}
    return {"claim": claim, "verified": False, "source_id": None, "citation": "Not verified."}


# ---------------- Reasoner ----------------
def connect_to_case(docs: list[dict[str, Any]], case_facts: str) -> str:
    if not docs:
        return "Not verified - no trusted source retrieved."
    lines = []
    for d in docs:
        if d["type"] == "statute":
            lines.append(f"- Section {d.get('section','')} {d['title']} applies: {d.get('text','')[:220]}")
        else:
            lines.append(f"- {d['title']} {d.get('citation','')} holds: {d.get('holding','')[:220]}")
    if case_facts:
        lines.append(f"\nCase facts considered: {case_facts.strip()}")
    return "\n".join(lines)


@dataclass
class LegalReport:
    legal_issue: list[str] = field(default_factory=list)
    relevant_laws: list[dict[str, Any]] = field(default_factory=list)
    relevant_judgments: list[dict[str, Any]] = field(default_factory=list)
    connection_to_case: str = ""
    verified_sources: list[dict[str, Any]] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    query_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------- Agent ----------------
class LegalResearchAgent:
    def __init__(self) -> None:
        self.corpus = load_corpus()
        self.retriever = TfidfRetriever(self.corpus)

    # core API
    def research(self, query: str, case_facts: str = "") -> dict[str, Any]:
        issues = spot_issues(query)
        docs = self.retriever.retrieve(query, k=8)
        laws = [d for d in docs if d["type"] == "statute"]
        judgments = [d for d in docs if d["type"] == "judgment"]

        verified, unverified = [], []
        for d in docs:
            claim = (d.get("text", "") if d["type"] == "statute" else d.get("holding", ""))[:120]
            res = verify_claim(claim, [d])
            (verified if res["verified"] else unverified).append(res if res["verified"] else claim)

        return LegalReport(
            legal_issue=issues, relevant_laws=laws, relevant_judgments=judgments,
            connection_to_case=connect_to_case(docs, case_facts),
            verified_sources=verified, unverified=unverified, query_used=query,
        ).to_dict()

    # ---- link: Drafting agent uses this to double-check every citation ----
    def verify_citation(self, doc_id: str) -> dict[str, Any]:
        doc = get_verified_source(doc_id)
        if not doc:
            return {"doc_id": doc_id, "verified": False, "citation": "Not verified.", "source_url": None}
        return {"doc_id": doc_id, "verified": True, "source_url": doc["source_url"],
                "citation": doc.get("citation")
                or f"Section {doc.get('section','')} {doc.get('title','')}".strip()}

    # ---- link: Review agent -> Research agent ----
    def research_from_review(self, review_report: dict[str, Any]) -> dict[str, Any]:
        issues = sorted({i["issue"] for i in review_report.get("legal_issues", [])})
        sections = [s["value"] for s in review_report.get("identifiers", {}).get("sections", [])]
        facts = " ".join(f["fact"] for f in review_report.get("key_facts", [])[:5])
        query = " ".join(issues + sections) or facts
        if not query.strip():
            return {"error": "Review report contained no issues, sections or facts to research."}
        return self.research(query, case_facts=facts)

    # ---- link: Drafting agent -> Research agent ----
    def research_for_case(self, case: dict[str, Any]) -> dict[str, Any]:
        sections = [str(s) for s in case.get("sections_invoked", [])]
        facts = case.get("facts_narrative") or ""
        query = " ".join(sections + [facts[:300]]).strip()
        if not query:
            return {"relevant_laws": [], "relevant_judgments": []}
        return self.research(query, case_facts=facts)


# ---------------- CLI ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Legal Research Agent (India)")
    ap.add_argument("--query", type=str)
    ap.add_argument("--facts", type=str, default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    query = a.query or input("Legal query: ").strip()
    facts = a.facts or (input("Case facts (optional): ").strip() if a.query is None else "")

    r = LegalResearchAgent().research(query, facts)
    if a.json or not sys.stdout.isatty():
        print(json.dumps(r, indent=2)); return

    print("\n" + "=" * 70 + "\nLEGAL RESEARCH REPORT\n" + "=" * 70)
    print("\n[LEGAL ISSUE]"); [print(f"  - {i}") for i in r["legal_issue"]]
    print("\n[RELEVANT LAWS / SECTIONS]")
    if not r["relevant_laws"]: print("  Not verified.")
    for l in r["relevant_laws"]:
        print(f"  - Section {l.get('section','')} {l['title']}\n    {l.get('text','')[:200]}...")
    print("\n[RELEVANT JUDGMENTS / PRECEDENTS]")
    if not r["relevant_judgments"]: print("  Not verified.")
    for j in r["relevant_judgments"]:
        print(f"  - {j['title']} {j.get('citation','')} ({j.get('court','')})\n    {j.get('holding','')[:200]}...")
    print("\n[CONNECTION TO THE CASE]\n" + r["connection_to_case"])
    print("\n[VERIFIED SOURCES]")
    for s in r["verified_sources"]: print(f"  - {s['claim'][:60]}... -> {s['citation']} [{s['source_id']}]")
    print("\n[UNVERIFIED / MISSING INFORMATION]")
    if not r["unverified"]: print("  None.")
    for u in r["unverified"]: print(f"  - Not verified: {u[:80]}...")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

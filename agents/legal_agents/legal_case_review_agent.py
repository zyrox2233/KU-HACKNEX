"""
legal_case_review_agent.py   (Agent 1 of 3)
===========================================
Reads uploaded case documents (PDF/TXT) and produces a page-cited case summary.
Only uses text present in the documents; emits NOT_FOUND when absent.

Inter-links
  -> legal_research_agent : review_and_research() sends issues/sections for research
  -> legal_drafting_agent : its report dict feeds register_case_from_review()

Run standalone:
    python legal_case_review_agent.py --docs case.pdf fir.txt [--research] [--json]
PDF support: pip install pdfplumber   (TXT needs nothing)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from common import NOT_FOUND, dedupe
from legal_research_agent import LegalResearchAgent


# ---------------- Loader ----------------
@dataclass
class Page:
    doc_name: str
    page_no: int
    text: str


def load_documents(paths: list[str]) -> list[Page]:
    pages: list[Page] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            pages.append(Page(path.name, 0, f"[FILE NOT FOUND: {path}]"))
            continue
        sfx = path.suffix.lower()
        if sfx == ".pdf":
            pages.extend(_load_pdf(path))
        elif sfx in (".txt", ".md"):
            pages.extend(_load_txt(path))
        else:
            pages.append(Page(path.name, 0, f"[UNSUPPORTED FILE TYPE: {sfx}]"))
    return pages


def _load_pdf(path: Path) -> list[Page]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return [Page(path.name, 0, "[PDF SUPPORT REQUIRES: pip install pdfplumber]")]
    out = []
    with pdfplumber.open(str(path)) as pdf:
        for i, pg in enumerate(pdf.pages, start=1):
            out.append(Page(path.name, i, pg.extract_text() or ""))
    return out


def _load_txt(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split("\f") if "\f" in text else [text]
    return [Page(path.name, i + 1, c) for i, c in enumerate(chunks)]


# ---------------- Patterns ----------------
DATE_PATTERNS = [
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
]

_NAME = r"([A-Z][A-Za-z.]+(?: [A-Z][A-Za-z.]+){0,3})"
PARTY_PATTERNS = {
    "complainant": [r"(?i:complainant|informant|petitioner|plaintiff)[ \t]*[:\-]?[ \t]*" + _NAME],
    "accused": [r"(?i:accused|respondent|defendant)[ \t]*[:\-]?[ \t]*" + _NAME],
    "witness": [r"(?i:witness|pw[ \t]*\d+|dw[ \t]*\d+)[ \t]*[:\-]?[ \t]*" + _NAME],
}

SECTION_PATTERN = r"\b(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]?)\s*(?:of\s+)?(IPC|CrPC|BNS|BNSS|Evidence Act|IEA)?"
FIR_PATTERN = r"\b(?:FIR|F\.I\.R\.)\s*(?:No\.?|Number)?\s*[:\-]?\s*(\d[0-9/\-A-Za-z]*)"
CASE_NO_PATTERN = r"\b(?:Case|Crl\.?|Civil|W\.P\.|SLP)\s*(?:No\.?|Number)\s*[:\-]?\s*(\d[0-9/\-A-Za-z]*)"
COURT_PATTERN = r"\b(High Court of [A-Za-z ]+|Supreme Court of India|District Court [A-Za-z ]+|Sessions Court [A-Za-z ]+|Magistrate [A-Za-z ]*Court)\b"

EVIDENCE_KEYWORDS = [
    "exhibit", "exh.", "marked as", "seized", "recovered", "post-mortem", "postmortem",
    "FSL", "forensic", "DNA", "fingerprint", "CCTV", "call detail record", "CDR",
    "medical report", "injury report", "dying declaration", "confession", "statement of",
    "testimony", "affidavit", "weapon", "knife", "pistol", "firearm", "blood sample",
]

LEGAL_ISSUE_KEYWORDS = {
    "murder": ["murder", "302", "killed", "homicide"],
    "culpable homicide": ["culpable homicide", "304", "not amounting"],
    "death penalty": ["death penalty", "capital punishment", "rarest of rare"],
    "bail": ["bail", "437", "438", "439", "anticipatory"],
    "circumstantial evidence": ["circumstantial", "panchsheel", "chain of evidence"],
    "confession": ["confession", "164", "voluntary", "retracted"],
    "fir": ["first information", "FIR", "154", "cognizable"],
    "domestic violence": ["domestic violence", "498A", "dowry"],
    "cheating": ["cheating", "420", "dishonest inducement"],
    "theft": ["theft", "379", "stolen"],
    "evidence admissibility": ["admissible", "inadmissible", "exclusion"],
    "limitation": ["limitation", "time-barred", "barred by time"],
    "jurisdiction": ["jurisdiction", "territorial", "pecuniary"],
}

CONTRADICTION_SIGNALS = [
    ("admitted", "denied"), ("present", "absent"), ("before", "after"),
    ("voluntary", "coerced"), ("signed", "unsigned"), ("informed", "uninformed"),
    ("agree", "disagree"), ("yes", "no"), ("confirmed", "refuted"),
]


# ---------------- Extractors ----------------
def _context(text: str, start: int, end: int, window: int = 80) -> str:
    return " ".join(text[max(0, start - window): min(len(text), end + window)].split())


def _find_all(pattern: str, page: Page, flags: int = 0, group: int = 0) -> list[dict[str, Any]]:
    return [{
        "value": m.group(group).strip(),
        "match": m.group(0).strip(),
        "document": page.doc_name,
        "page": page.page_no,
        "context": _context(page.text, m.start(), m.end()),
    } for m in re.finditer(pattern, page.text, flags)]


def extract_dates(pages):
    out = []
    for p in pages:
        for pat in DATE_PATTERNS:
            out.extend(_find_all(pat, p))
    return dedupe(out)


def extract_parties(pages):
    out = {k: [] for k in PARTY_PATTERNS}
    for role, pats in PARTY_PATTERNS.items():
        for p in pages:
            for pat in pats:
                out[role].extend(_find_all(pat, p, group=1))
    return {k: dedupe(v) for k, v in out.items()}


def extract_case_identifiers(pages):
    out: dict[str, list] = {"fir": [], "case_number": [], "court": [], "sections": []}
    for p in pages:
        out["fir"].extend(_find_all(FIR_PATTERN, p, re.IGNORECASE, group=1))
        out["case_number"].extend(_find_all(CASE_NO_PATTERN, p, group=1))
        out["court"].extend(_find_all(COURT_PATTERN, p, group=1))
        for m in re.finditer(SECTION_PATTERN, p.text):
            out["sections"].append({
                "value": f"Section {m.group(1)} {m.group(2) or ''}".strip(),
                "document": p.doc_name, "page": p.page_no,
                "context": _context(p.text, m.start(), m.end()),
            })
    return {k: dedupe(v) for k, v in out.items()}


def extract_evidence(pages):
    out = []
    for p in pages:
        low = p.text.lower()
        for kw in EVIDENCE_KEYWORDS:
            idx = low.find(kw.lower())
            while idx != -1:
                out.append({"keyword": kw, "snippet": _context(p.text, idx, idx + len(kw)),
                            "document": p.doc_name, "page": p.page_no})
                idx = low.find(kw.lower(), idx + 1)
    return dedupe(out)


def extract_legal_issues(pages):
    out = []
    for p in pages:
        low = p.text.lower()
        for issue, kws in LEGAL_ISSUE_KEYWORDS.items():
            for kw in kws:
                idx = low.find(kw.lower())
                if idx != -1:
                    out.append({"issue": issue, "matched_term": kw,
                                "snippet": _context(p.text, idx, idx + len(kw)),
                                "document": p.doc_name, "page": p.page_no})
                    break
    return dedupe(out)


def extract_key_facts(pages, max_facts: int = 15):
    verbs = ["committed", "attacked", "assaulted", "murdered", "killed", "filed", "lodged",
             "registered", "arrested", "recovered", "seized", "recorded", "deposed",
             "testified", "stated", "admitted", "denied", "convicted", "acquitted", "sentenced"]
    facts = []
    for p in pages:
        for s in re.split(r"(?<=[.!?])\s+", p.text):
            sc = " ".join(s.split())
            if len(sc) < 25:
                continue
            if any(v in sc.lower() for v in verbs) or any(re.search(dp, sc) for dp in DATE_PATTERNS):
                facts.append({"fact": sc, "document": p.doc_name, "page": p.page_no})
            if len(facts) >= max_facts:
                return facts
    return facts


def _share_topic(a: str, b: str) -> bool:
    stop = {"the", "a", "an", "is", "was", "were", "of", "to", "in", "on", "and", "or",
            "for", "by", "with", "that", "this", "it", "as"}
    ta = {t for t in re.findall(r"[a-z]+", a) if t not in stop and len(t) > 3}
    tb = {t for t in re.findall(r"[a-z]+", b) if t not in stop and len(t) > 3}
    return len(ta & tb) >= 3


def detect_contradictions(pages):
    sents = []
    for p in pages:
        for s in re.split(r"(?<=[.!?])\s+", p.text):
            sc = " ".join(s.split())
            if len(sc) >= 20:
                sents.append({"text": sc, "lower": sc.lower(), "document": p.doc_name, "page": p.page_no})
    out = []
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            a, b = sents[i], sents[j]
            if a["document"] == b["document"] and a["page"] == b["page"]:
                continue
            for pos, neg in CONTRADICTION_SIGNALS:
                if pos in a["lower"] and neg in b["lower"] and _share_topic(a["lower"], b["lower"]):
                    out.append({"statement_a": a["text"], "source_a": f"{a['document']} p.{a['page']}",
                                "statement_b": b["text"], "source_b": f"{b['document']} p.{b['page']}",
                                "signal_pair": f"{pos} vs {neg}"})
    return dedupe(out)[:20]


REQUIRED_FIELDS = [
    ("Case number / FIR number", ["fir", "case_number"]), ("Court", ["court"]),
    ("Parties (complainant / accused)", ["complainant", "accused"]), ("Dates", ["dates"]),
    ("Sections invoked", ["sections"]), ("Evidence", ["evidence"]), ("Key facts", ["key_facts"]),
]


def check_missing(extracted: dict[str, Any]) -> list[str]:
    return [label for label, keys in REQUIRED_FIELDS
            if not any(isinstance(extracted.get(k), list) and extracted.get(k) for k in keys)]


# ---------------- Report ----------------
@dataclass
class CaseReviewReport:
    key_facts: list = field(default_factory=list)
    parties: dict = field(default_factory=dict)
    important_dates: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    legal_issues: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    missing_information: list = field(default_factory=list)
    identifiers: dict = field(default_factory=dict)
    document_index: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    research: dict = field(default_factory=dict)   # filled by Research agent

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_report(all_pages: list[Page]) -> CaseReviewReport:
    warnings = [p.text for p in all_pages if p.page_no == 0]
    pages = [p for p in all_pages if p.page_no != 0]

    ids = extract_case_identifiers(pages)
    parties = extract_parties(pages)
    dates = extract_dates(pages)
    evidence = extract_evidence(pages)
    issues = extract_legal_issues(pages)
    facts = extract_key_facts(pages)
    contradictions = detect_contradictions(pages)

    missing = check_missing({
        "fir": ids["fir"], "case_number": ids["case_number"], "court": ids["court"],
        "complainant": parties["complainant"], "accused": parties["accused"],
        "dates": dates, "sections": ids["sections"], "evidence": evidence, "key_facts": facts,
    })
    return CaseReviewReport(
        key_facts=facts, parties=parties, important_dates=dates, evidence=evidence,
        legal_issues=issues, contradictions=contradictions, missing_information=missing,
        identifiers=ids, warnings=warnings,
        document_index=[{"document": p.doc_name, "page": p.page_no, "chars": len(p.text)} for p in pages],
    )


# ---------------- Agent ----------------
class CaseReviewAgent:
    def __init__(self, research_agent: LegalResearchAgent | None = None) -> None:
        self.research_agent = research_agent or LegalResearchAgent()

    def review(self, paths: list[str]) -> dict[str, Any]:
        return build_report(load_documents(paths)).to_dict()

    # link: Review -> Research
    def review_and_research(self, paths: list[str]) -> dict[str, Any]:
        report = self.review(paths)
        report["research"] = self.research_agent.research_from_review(report)
        return report


# ---------------- CLI ----------------
def _ref(doc: str, page: int) -> str:
    return f"{doc} p.{page}"


def _pretty_print(r: dict[str, Any]) -> None:
    print("\n" + "=" * 78 + "\nLEGAL CASE REVIEW REPORT\n" + "=" * 78)
    for w in r["warnings"]:
        print(f"  WARNING: {w}")
    print("\n[DOCUMENT INDEX]")
    if not r["document_index"]: print(f"  {NOT_FOUND}")
    for d in r["document_index"]: print(f"  - {d['document']} | page {d['page']} | {d['chars']} chars")

    print("\n[CASE IDENTIFIERS]")
    for key, label in [("fir", "FIR"), ("case_number", "Case No."), ("court", "Court"), ("sections", "Sections")]:
        items = r["identifiers"].get(key, [])
        if not items: print(f"  {label}: {NOT_FOUND}")
        for it in items: print(f"  {label}: {it['value']}  ({_ref(it['document'], it['page'])})")

    print("\n[PARTIES]")
    for role in ("complainant", "accused", "witness"):
        items = r["parties"].get(role, [])
        if not items: print(f"  {role.title()}: {NOT_FOUND}")
        for it in items: print(f"  {role.title()}: {it['value']}  ({_ref(it['document'], it['page'])})")

    for title, key, fmt in [
        ("IMPORTANT DATES", "important_dates", lambda d: f"{d['value']}  ({_ref(d['document'], d['page'])})"),
        ("KEY FACTS", "key_facts", lambda f: f"{f['fact']}  ({_ref(f['document'], f['page'])})"),
        ("EVIDENCE", "evidence", lambda e: f"[{e['keyword']}] {e['snippet']}  ({_ref(e['document'], e['page'])})"),
        ("LEGAL ISSUES", "legal_issues",
         lambda i: f"{i['issue']} (matched: '{i['matched_term']}')  ({_ref(i['document'], i['page'])})"),
    ]:
        print(f"\n[{title}]")
        if not r[key]: print(f"  {NOT_FOUND}")
        for it in r[key]: print(f"  - {fmt(it)}")

    print("\n[CONTRADICTIONS]")
    if not r["contradictions"]: print(f"  {NOT_FOUND}")
    for c in r["contradictions"]:
        print(f"  - Signal: {c['signal_pair']}\n      A: {c['statement_a']}  ({c['source_a']})"
              f"\n      B: {c['statement_b']}  ({c['source_b']})")

    print("\n[MISSING INFORMATION]")
    if not r["missing_information"]: print("  None detected in the provided documents.")
    for m in r["missing_information"]: print(f"  - {m}: {NOT_FOUND}")

    if r.get("research"):
        rs = r["research"]
        print("\n[LINKED RESEARCH (from Legal Research Agent)]")
        if "error" in rs: print(f"  {rs['error']}")
        else:
            for l in rs["relevant_laws"]: print(f"  - Section {l.get('section','')} {l['title']}")
            for j in rs["relevant_judgments"]: print(f"  - {j['title']} {j.get('citation','')}")
    print("\n" + "=" * 78 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Legal Case Review Agent")
    ap.add_argument("--docs", nargs="+")
    ap.add_argument("--research", action="store_true", help="Also run the Legal Research Agent")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    docs = a.docs or input("Enter document paths (space-separated): ").split()

    agent = CaseReviewAgent()
    r = agent.review_and_research(docs) if a.research else agent.review(docs)
    if a.json or not sys.stdout.isatty():
        print(json.dumps(r, indent=2, default=str))
    else:
        _pretty_print(r)


if __name__ == "__main__":
    main()

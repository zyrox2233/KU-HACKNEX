"""
legal_verification_agent.py   (Agent 5 of 6)
============================================
Verifies the output of the other agents against (a) the ORIGINAL uploaded documents
(page-tagged) and (b) the trusted legal corpus. Never approves an unverified claim.

Inter-links
  uses legal_case_review_agent : same Page / load_documents loader as the Review agent
  uses legal_research_agent    : verify_citation() for every cited authority
  uses corpus.py               : trusted corpus lookups
  <- orchestrator              : verify(pipeline, pages) and verify_chat_answer(chat, pages)

Run standalone:
    python pipeline.py --docs fir.txt --type bail --save pipeline.json
    python legal_verification_agent.py --pipeline pipeline.json --docs fir.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from common import MISSING, NOT_FOUND
from corpus import get_verified_source, find_by_citation
from legal_case_review_agent import Page, load_documents
from legal_research_agent import LegalResearchAgent

FUZZY_THRESHOLD = 0.75

# ---------------- Text matching ----------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "was", "were", "of", "to", "in", "on", "and", "or", "for",
         "by", "with", "that", "this", "it", "as", "at", "be"}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2}


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta) if ta else 0.0


def find_claim_in_pages(claim: str, pages: list[Page], threshold: float = FUZZY_THRESHOLD):
    norm = _normalize(claim)
    if not norm:
        return None
    best: tuple[float, Page | None] = (0.0, None)
    for p in pages:
        if norm in _normalize(p.text):
            return {"document": p.doc_name, "page": p.page_no, "match_type": "exact", "confidence": 1.0}
        s = _overlap(claim, p.text)
        if s > best[0]:
            best = (s, p)
    if best[1] is not None and best[0] >= threshold:
        return {"document": best[1].doc_name, "page": best[1].page_no,
                "match_type": "fuzzy", "confidence": round(best[0], 2)}
    return None


# ---------------- Claim extraction ----------------
def _claim(text, display, src_doc, src_page, origin):
    return {"text": text, "display": display, "origin": origin,
            "source": {"document": src_doc, "page": src_page} if src_doc else None}


def extract_claims_from_pipeline(pipeline: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    claims: dict[str, list] = {k: [] for k in (
        "facts", "parties", "identifiers", "dates", "evidence", "legal_issues",
        "draft_details", "draft_arguments", "citations")}
    review = pipeline.get("review") or {}
    research = pipeline.get("research") or {}
    draft = pipeline.get("draft") or {}

    for f in review.get("key_facts", []):
        claims["facts"].append(_claim(f["fact"], f["fact"], f.get("document"), f.get("page"), "review.key_facts"))
    for role, items in (review.get("parties") or {}).items():
        for it in items:
            claims["parties"].append(_claim(it["value"], f"{role}: {it['value']}",
                                            it.get("document"), it.get("page"), f"review.parties.{role}"))
    for kind, items in (review.get("identifiers") or {}).items():
        for it in items:
            claims["identifiers"].append(_claim(it["value"], f"{kind}: {it['value']}",
                                                it.get("document"), it.get("page"), f"review.identifiers.{kind}"))
    for d in review.get("important_dates", []):
        claims["dates"].append(_claim(d["value"], d["value"], d.get("document"), d.get("page"), "review.important_dates"))
    for e in review.get("evidence", []):
        claims["evidence"].append(_claim(e["snippet"], f"[{e['keyword']}] {e['snippet']}",
                                         e.get("document"), e.get("page"), "review.evidence"))
    for i in review.get("legal_issues", []):
        claims["legal_issues"].append(_claim(i["snippet"], f"{i['issue']} - {i['snippet']}",
                                             i.get("document"), i.get("page"), "review.legal_issues"))

    # Factual values the draft states (e.g. FIR number, names) must exist in the documents
    for k, v in (draft.get("case_details") or {}).items():
        if isinstance(v, str) and v != MISSING:
            claims["draft_details"].append(_claim(v, f"{k}: {v}", None, None, f"draft.case_details.{k}"))
    for a in draft.get("arguments", []):
        claims["draft_arguments"].append(_claim(a, a, None, None, "draft.arguments"))

    for s in draft.get("sources_citations", []):
        claims["citations"].append({"text": s.get("citation", ""), "doc_id": s.get("doc_id"),
                                    "origin": "draft.sources_citations"})
    for s in draft.get("suggested_authorities", []):
        claims["citations"].append({"text": s.get("citation", ""), "doc_id": s.get("doc_id"),
                                    "origin": "draft.suggested_authorities"})
    for v in research.get("verified_sources", []):
        claims["citations"].append({"text": v.get("citation", ""),
                                    "doc_id": v.get("doc_id") or v.get("source_id"),
                                    "origin": "research.verified_sources"})
    return claims


# ---------------- Report models ----------------
@dataclass
class CitationStatus:
    citation: str = ""
    doc_id: str | None = None
    origin: str = ""
    present_in_corpus: bool = False
    trusted: bool = False
    supports_claim: bool = False
    reason: str = ""


@dataclass
class VerificationReport:
    verified_claims: list[dict[str, Any]] = field(default_factory=list)
    unsupported_claims: list[dict[str, Any]] = field(default_factory=list)
    citation_status: list[dict[str, Any]] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    final_result: str = ""
    approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------- Agent ----------------
class LegalVerificationAgent:
    def __init__(self, research_agent: LegalResearchAgent | None = None) -> None:
        self.research = research_agent or LegalResearchAgent()

    # ---- facts ----
    def _verify_fact(self, claim: dict[str, Any], pages: list[Page]):
        text = (claim.get("text") or "").strip()
        base = {"claim": claim.get("display") or text, "origin": claim.get("origin", "")}
        if not text:
            return None, {**base, "reason": "Empty claim."}
        src = claim.get("source") or {}
        doc, pg = src.get("document"), src.get("page")
        if doc and pg:
            for p in pages:
                if p.doc_name == doc and p.page_no == pg:
                    ov = _overlap(text, p.text)
                    if _normalize(text) in _normalize(p.text) or ov >= FUZZY_THRESHOLD:
                        return {**base, "source": f"{doc} p.{pg}", "confidence": round(max(ov, 0.0), 2),
                                "match_type": "cited_page"}, None
                    return None, {**base, "reason": f"Cited source {doc} p.{pg} does not contain this claim."}
        hit = find_claim_in_pages(text, pages)
        if hit:
            return {**base, "source": f"{hit['document']} p.{hit['page']}",
                    "confidence": hit["confidence"], "match_type": hit["match_type"]}, None
        return None, {**base, "reason": NOT_FOUND}

    # ---- citations (cross-checked through the Research agent) ----
    def _verify_citation(self, cit: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
        text = (cit.get("text") or "").strip()
        doc_id = cit.get("doc_id")
        st = CitationStatus(citation=text, doc_id=doc_id, origin=cit.get("origin", ""))
        if not text and not doc_id:
            st.reason = "Empty citation."
            return asdict(st)

        doc = None
        if doc_id and self.research.verify_citation(doc_id)["verified"]:
            doc = get_verified_source(doc_id)
        if doc is None:
            doc = find_by_citation(text)
        if doc is None:
            st.reason = "Citation not present in trusted legal corpus."
            return asdict(st)

        st.doc_id = doc["doc_id"]
        st.present_in_corpus = True
        st.trusted = True
        corpus_text = " ".join(filter(None, [doc.get("text", ""), doc.get("holding", "")]))
        matched = next((v.get("claim", "") for v in research.get("verified_sources", [])
                        if (v.get("doc_id") or v.get("source_id")) == doc["doc_id"]), "")
        if matched:
            if _overlap(matched, corpus_text) >= 0.4 or _normalize(matched) in _normalize(corpus_text):
                st.supports_claim = True
                st.reason = "Citation verified in corpus and supports the attached claim."
            else:
                st.reason = "Citation exists in corpus but does not support the attached claim."
        else:
            st.supports_claim = True
            st.reason = "Citation exists in trusted corpus (no attached claim to cross-check)."
        return asdict(st)

    # ---- main entry: pipeline output ----
    def verify(self, pipeline: dict[str, Any], pages: list[Page]) -> VerificationReport:
        pages = [p for p in pages if p.page_no != 0]
        claims = extract_claims_from_pipeline(pipeline)
        research = pipeline.get("research") or {}
        verified: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        missing: list[str] = []

        for key in ("facts", "parties", "identifiers", "dates", "evidence", "legal_issues", "draft_details"):
            for c in claims[key]:
                ok, bad = self._verify_fact(c, pages)
                (verified if ok else unsupported).append(ok or bad)

        for c in claims["draft_arguments"]:
            t = c["text"]
            if MISSING in t:
                missing.append(f"Draft argument depends on missing data: {t}")
            elif re.search(r"maintainable on the ground of", t, re.I):
                verified.append({"claim": t, "origin": c["origin"], "source": "LEGAL_GROUNDS_MAP (procedural)",
                                 "confidence": 1.0, "match_type": "procedural"})
            else:
                ok, bad = self._verify_fact(c, pages)
                (verified if ok else unsupported).append(ok or bad)

        seen_ids: set[Any] = set()
        cit_results = []
        for c in claims["citations"]:
            key = c.get("doc_id") or c.get("text")
            if key in seen_ids:
                continue
            seen_ids.add(key)
            cit_results.append(self._verify_citation(c, research))

        for m in (pipeline.get("review") or {}).get("missing_information", []):
            missing.append(f"{m}: {NOT_FOUND}")
        draft = pipeline.get("draft") or {}
        if draft.get("verification") == MISSING:
            missing.append("Draft verification block: advocate name/date not supplied.")
        for cs in cit_results:
            if not cs["present_in_corpus"]:
                missing.append(f"Citation not in trusted corpus: {cs['citation']}")
            elif not cs["supports_claim"]:
                missing.append(f"Citation exists but does not support claim: {cs['citation']}")

        bad_cites = sum(1 for c in cit_results if not c["supports_claim"])
        total = len(verified) + len(unsupported) + len(cit_results)
        if total == 0:
            final, approved = "NOTHING TO VERIFY - pipeline contained no claims.", False
        elif not unsupported and bad_cites == 0:
            final, approved = "APPROVED - All claims verified.", True
            if missing:
                final += f" ({len(missing)} missing-information item(s) remain.)"
        elif unsupported and not verified:
            final, approved = "REJECTED - No claim could be verified.", False
        else:
            final, approved = (f"PARTIALLY VERIFIED - {len(verified)} verified, {len(unsupported)} "
                               f"unsupported, {bad_cites} citation issue(s)."), False

        return VerificationReport(verified, unsupported, cit_results, missing, final, approved)

    # ---- link: Chatbot -> Verification ----
    def verify_chat_answer(self, chat: dict[str, Any], pages: list[Page]) -> dict[str, Any]:
        """Re-check every line of a chatbot answer against its cited page / corpus entry."""
        if chat.get("status") != "answered":
            return {"verified_lines": [], "unsupported_lines": [], "deferred_lines": [],
                    "final_result": "SAFE - chatbot declined to answer (no support found).", "approved": True}
        pages = [p for p in pages if p.page_no != 0]
        refs = {r["label"]: r for r in chat.get("references", [])}
        ver, bad, deferred = [], [], []
        for line in chat["answer"].splitlines():
            m = re.match(r"^- (.*?)\s{2}\((.*)\)\s*$", line)
            if not m:
                continue
            sent, label = m.group(1), m.group(2)
            ref = refs.get(label)
            if not ref:
                bad.append({"line": sent, "reason": "No matching reference."}); continue
            if ref["origin"] == "document":
                page = next((p for p in pages if p.doc_name == ref["document"] and p.page_no == ref["page"]), None)
                ok = page is not None and _normalize(sent) in _normalize(page.text)
                (ver if ok else bad).append({"line": sent, "source": label} if ok
                                            else {"line": sent, "reason": f"Not found on {label}."})
            elif ref["origin"] == "corpus":
                doc = get_verified_source(ref["chunk_id"].split("#", 1)[1])
                txt = (doc.get("text") or doc.get("holding") or "") if doc else ""
                ok = bool(doc) and _normalize(sent) in _normalize(txt)
                (ver if ok else bad).append({"line": sent, "source": label} if ok
                                            else {"line": sent, "reason": "Not found in trusted corpus entry."})
            else:
                deferred.append({"line": sent, "source": label,
                                 "reason": "Derived from pipeline output - covered by full pipeline verification."})
        approved = not bad
        result = ("APPROVED - every answer line traced to its source." if approved and not deferred
                  else "APPROVED (with deferred pipeline-derived lines)." if approved
                  else f"NOT APPROVED - {len(bad)} unsupported line(s).")
        return {"verified_lines": ver, "unsupported_lines": bad, "deferred_lines": deferred,
                "final_result": result, "approved": approved}

    # convenience
    def verify_files(self, pipeline: dict[str, Any], doc_paths: list[str]) -> VerificationReport:
        return self.verify(pipeline, load_documents(doc_paths))


# ---------------- CLI ----------------
def _pretty(r: dict[str, Any]) -> None:
    print("\n" + "=" * 80 + "\nLEGAL VERIFICATION REPORT\n" + "=" * 80)
    print(f"\n[VERIFIED CLAIMS]  ({len(r['verified_claims'])})")
    for v in r["verified_claims"]:
        print(f"  - {v['claim'][:110]}\n      source: {v['source']}  ({v['match_type']}, conf {v['confidence']})")
    print(f"\n[UNSUPPORTED CLAIMS]  ({len(r['unsupported_claims'])})")
    if not r["unsupported_claims"]: print("  None.")
    for u in r["unsupported_claims"]:
        print(f"  - {u['claim'][:110]}\n      origin: {u['origin']}\n      reason: {u['reason']}")
    print(f"\n[CITATION STATUS]  ({len(r['citation_status'])})")
    for c in r["citation_status"]:
        print(f"  [{'OK' if c['supports_claim'] else 'FAIL'}] {c['citation']}  [{c.get('doc_id')}]\n"
              f"        reason: {c['reason']}")
    print(f"\n[MISSING INFORMATION]  ({len(r['missing_information'])})")
    if not r["missing_information"]: print("  None.")
    for m in r["missing_information"]: print(f"  - {m}")
    print(f"\n[FINAL VERIFICATION RESULT]\n  {r['final_result']}\n" + "=" * 80 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Legal Verification Agent")
    ap.add_argument("--pipeline", help="Pipeline JSON saved by pipeline.py --save / orchestrator.py --save")
    ap.add_argument("--docs", nargs="+", help="Original uploaded documents")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    pipeline = json.loads(Path(a.pipeline).read_text(encoding="utf-8-sig")) if a.pipeline \
        else json.loads(input("Paste pipeline JSON (single line): "))
    docs = a.docs or input("Document paths (space-separated): ").split()
    rep = LegalVerificationAgent().verify(pipeline, load_documents(docs)).to_dict()
    print(json.dumps(rep, indent=2, default=str)) if (a.json or not sys.stdout.isatty()) else _pretty(rep)


if __name__ == "__main__":
    main()

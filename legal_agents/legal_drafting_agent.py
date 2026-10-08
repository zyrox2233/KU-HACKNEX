"""
legal_drafting_agent.py   (Agent 3 of 3)
========================================
Drafts legal documents ONLY from registered case facts + the trusted corpus.
Any absent field renders as [MISSING INFORMATION].

Inter-links
  <- legal_case_review_agent : register_case_from_review() turns a review report into case facts
                               (with page-level provenance)
  -> legal_research_agent    : every cited authority is cross-verified; extra authorities
                               are suggested via research_for_case()

Run standalone:
    python legal_drafting_agent.py --type bail --case case_001
    python legal_drafting_agent.py --type bail --docs case.txt --verified-by "Adv. X" --verified-on 2026-10-08
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any

from common import MISSING, safe
from corpus import get_verified_source
from legal_research_agent import LegalResearchAgent
from legal_case_review_agent import CaseReviewAgent


# ---------------- Verified fact store ----------------
def _blank_case() -> dict[str, Any]:
    return {
        "case_title": None, "fir_number": None, "police_station": None, "district": None,
        "state": None, "date_of_incident": None, "date_of_fir": None, "complainant_name": None,
        "accused_name": None, "victim_name": None, "facts_narrative": None,
        "sections_invoked": [], "witnesses": [], "evidence": [], "arrest_date": None,
        "custody_duration": None, "verified_by": None, "verified_on": None,
        "provenance": {},
    }


VERIFIED_FACTS: dict[str, dict[str, Any]] = {"case_001": _blank_case()}


def get_verified_case(case_id: str) -> dict[str, Any] | None:
    return VERIFIED_FACTS.get(case_id)


def _first(items: list[dict[str, Any]]) -> tuple[Any, str | None]:
    if not items:
        return None, None
    it = items[0]
    return it["value"], f"{it['document']} p.{it['page']}"


# link: Review agent -> Drafting agent
def register_case_from_review(
    case_id: str,
    review_report: dict[str, Any],
    verified_by: str | None = None,
    verified_on: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Populate the fact store from a Case Review report. Only values literally extracted from the
    documents are used; each carries a provenance reference. Fields the documents don't show
    (police station, district, victim, title, ...) stay None -> [MISSING INFORMATION].
    The draft's verification block stays MISSING until a human supplies verified_by/verified_on.
    """
    case = _blank_case()
    prov: dict[str, str] = {}

    ids = review_report.get("identifiers", {})
    parties = review_report.get("parties", {})

    for fld, items in [("fir_number", ids.get("fir", [])),
                       ("complainant_name", parties.get("complainant", [])),
                       ("accused_name", parties.get("accused", []))]:
        val, ref = _first(items)
        if val:
            case[fld], prov[fld] = val, ref

    case["sections_invoked"] = [s["value"] for s in ids.get("sections", [])]
    case["witnesses"] = [w["value"] for w in parties.get("witness", [])]
    case["evidence"] = [f"{e['keyword']}: {e['snippet']} ({e['document']} p.{e['page']})"
                        for e in review_report.get("evidence", [])]

    facts = review_report.get("key_facts", [])
    if facts:
        case["facts_narrative"] = " ".join(f"{f['fact']} [{f['document']} p.{f['page']}]" for f in facts)
        prov["facts_narrative"] = "; ".join(sorted({f"{f['document']} p.{f['page']}" for f in facts}))

    case["verified_by"], case["verified_on"] = verified_by, verified_on
    case["provenance"] = prov
    case.update(overrides or {})          # human-supplied corrections win
    VERIFIED_FACTS[case_id] = case
    return case


# ---------------- Legal grounds ----------------
LEGAL_GROUNDS_MAP: dict[str, dict[str, list[str]]] = {
    "fir": {"statutes": ["crpc_154"], "judgments": ["sc_lalita_kumari_2013"],
            "principles": ["cognizable offence", "mandatory registration of FIR"]},
    "bail": {"statutes": ["crpc_437", "crpc_439"], "judgments": [],
             "principles": ["presumption of innocence", "no flight risk", "no tampering with evidence"]},
    "anticipatory_bail": {"statutes": ["crpc_438"], "judgments": [],
                          "principles": ["apprehension of arrest", "cooperation with investigation"]},
    "petition_302": {"statutes": ["ipc_302", "crpc_154"],
                     "judgments": ["sc_bachan_singh_1980", "sc_machhi_singh_1983"],
                     "principles": ["rarest of rare", "balance sheet of aggravating and mitigating circumstances"]},
    "appeal": {"statutes": ["crpc_374", "crpc_386"], "judgments": ["sc_sharad_birdhichand_1984"],
               "principles": ["error of law", "perversity of finding"]},
    "legal_notice": {"statutes": [], "judgments": [], "principles": ["cause of action", "opportunity to remedy"]},
    "affidavit": {"statutes": [], "judgments": [], "principles": ["deponent's personal knowledge", "truth verification"]},
    "vakalatnama": {"statutes": [], "judgments": [], "principles": ["authority to appear", "client consent"]},
    "plaint": {"statutes": [], "judgments": [], "principles": ["cause of action", "jurisdiction", "valuation"]},
    "written_statement": {"statutes": [], "judgments": [], "principles": ["denial of plaint averments", "affirmative defences"]},
}

TITLE_MAP = {
    "fir": "First Information Report", "bail": "Application for Bail",
    "anticipatory_bail": "Application for Anticipatory Bail",
    "petition_302": "Petition under Section 302 IPC", "appeal": "Memorandum of Appeal",
    "legal_notice": "Legal Notice", "affidavit": "Affidavit", "vakalatnama": "Vakalatnama",
    "plaint": "Plaint", "written_statement": "Written Statement",
}

PRAYERS = {
    "fir": ["Register an FIR under the applicable sections of the Indian Penal Code, 1860.",
            "Provide a copy of the FIR free of cost to the informant as mandated by Section 154 CrPC."],
    "bail": ["Release the applicant on bail in the above-mentioned case on such terms and "
             "conditions as this Hon'ble Court may deem fit."],
    "anticipatory_bail": ["Grant anticipatory bail to the applicant in the event of arrest in the "
                          "above-mentioned case."],
    "petition_302": ["Convict the accused under Section 302 IPC and award the appropriate sentence "
                     "in accordance with law."],
    "appeal": ["Set aside the impugned judgment and order of the lower court.",
               "Acquit the appellant of all charges."],
    "legal_notice": ["Comply with the demands stated herein within the stipulated time.",
                     "Failing compliance, the sender will be constrained to initiate appropriate legal proceedings."],
    "affidavit": ["That this affidavit is filed in support of the accompanying application."],
    "vakalatnama": ["That the advocate named herein is authorised to appear, act, and plead on behalf of the client."],
    "plaint": ["Decree the suit in favour of the plaintiff with costs."],
    "written_statement": ["Dismiss the suit with exemplary costs."],
}


# ---------------- Draft model ----------------
@dataclass
class LegalDraft:
    document_title: str = MISSING
    status: str = "DRAFT - for advocate review; not for filing"
    case_details: dict[str, Any] = field(default_factory=dict)
    facts: str = MISSING
    legal_grounds: list[str] = field(default_factory=list)
    applicable_laws: list[dict[str, Any]] = field(default_factory=list)
    suggested_authorities: list[dict[str, Any]] = field(default_factory=list)  # from Research agent
    arguments: list[str] = field(default_factory=list)
    prayer_relief: list[str] = field(default_factory=list)
    verification: str = MISSING
    sources_citations: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------- Agent ----------------
class LegalDraftingAgent:
    def __init__(self, research_agent: LegalResearchAgent | None = None,
                 review_agent: CaseReviewAgent | None = None) -> None:
        self.research = research_agent or LegalResearchAgent()
        self.review = review_agent or CaseReviewAgent(self.research)

    # --- builders ---
    @staticmethod
    def _case_details(case: dict[str, Any]) -> dict[str, Any]:
        keys = ["case_title", "fir_number", "police_station", "district", "state",
                "date_of_incident", "date_of_fir", "complainant_name", "accused_name", "victim_name"]
        return {k: safe(case.get(k)) for k in keys}

    def _applicable_laws(self, doc_type: str) -> list[dict[str, Any]]:
        g = LEGAL_GROUNDS_MAP.get(doc_type, {})
        laws = []
        for doc_id in g.get("statutes", []) + g.get("judgments", []):
            src = get_verified_source(doc_id)
            check = self.research.verify_citation(doc_id)          # link: cross-verify
            if src and check["verified"]:
                laws.append({"doc_id": doc_id, "title": src["title"], "section": src.get("section"),
                             "citation": check["citation"], "source_url": check["source_url"],
                             "text_or_holding": src.get("text") or src.get("holding"),
                             "cross_verified": True})
        return laws

    def _suggested(self, case: dict[str, Any], already: list[dict[str, Any]]) -> list[dict[str, Any]]:
        have = {l["doc_id"] for l in already}
        res = self.research.research_for_case(case)               # link: Research agent
        out = []
        for d in res.get("relevant_laws", []) + res.get("relevant_judgments", []):
            if d["doc_id"] not in have:
                out.append({"doc_id": d["doc_id"], "title": d["title"], "section": d.get("section"),
                            "citation": d.get("citation")
                            or f"Section {d.get('section','')} {d['title']}".strip(),
                            "origin": "research_agent", "note": "Suggested only - advocate to confirm relevance."})
        return out

    @staticmethod
    def _arguments(doc_type: str, case: dict[str, Any]) -> list[str]:
        args = [f"That the present application is maintainable on the ground of {g}."
                for g in LEGAL_GROUNDS_MAP.get(doc_type, {}).get("principles", [])]
        if doc_type in ("bail", "anticipatory_bail"):
            c = case.get("custody_duration")
            args.append(f"That the accused has been in custody for {c}, which merits consideration."
                        if c else "That the period of custody, if any, is [MISSING INFORMATION] and requires verification.")
        return args or [MISSING]

    @staticmethod
    def _verification(case: dict[str, Any], doc_type: str) -> str:
        who, when = safe(case.get("verified_by")), safe(case.get("verified_on"))
        if who == MISSING or when == MISSING:
            return MISSING
        return (f"Verified by {who} on {when}. This draft is prepared solely from verified case "
                f"facts and verified legal sources in the {doc_type} matter.")

    # --- main API ---
    def draft(self, doc_type: str, case_id: str) -> dict[str, Any]:
        doc_type = doc_type.lower().strip()
        if doc_type not in LEGAL_GROUNDS_MAP:
            return {"error": f"Unsupported document type: {doc_type}",
                    "supported": list(LEGAL_GROUNDS_MAP)}
        case = get_verified_case(case_id)
        if not case:
            return {"error": f"Case '{case_id}' not found in verified fact store."}

        laws = self._applicable_laws(doc_type)
        sources = [{"doc_id": l["doc_id"], "title": l["title"], "citation": l["citation"]} for l in laws] \
            or [{"doc_id": None, "title": MISSING, "citation": MISSING}]

        return LegalDraft(
            document_title=TITLE_MAP[doc_type],
            case_details=self._case_details(case),
            facts=safe(case.get("facts_narrative")),
            legal_grounds=LEGAL_GROUNDS_MAP[doc_type]["principles"] or [MISSING],
            applicable_laws=laws,
            suggested_authorities=self._suggested(case, laws),
            arguments=self._arguments(doc_type, case),
            prayer_relief=PRAYERS.get(doc_type, [MISSING]),
            verification=self._verification(case, doc_type),
            sources_citations=sources,
            provenance=case.get("provenance", {}),
        ).to_dict()

    # link: Review -> Drafting in one call
    def draft_from_documents(self, doc_type: str, paths: list[str], case_id: str = "case_from_docs",
                             verified_by: str | None = None, verified_on: str | None = None,
                             overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        report = self.review.review(paths)
        register_case_from_review(case_id, report, verified_by, verified_on, overrides)
        return self.draft(doc_type, case_id)


# ---------------- CLI ----------------
def _pretty_print(d: dict[str, Any]) -> None:
    if "error" in d:
        print(f"ERROR: {d['error']}")
        if "supported" in d: print("Supported types:", ", ".join(d["supported"]))
        return
    print("\n" + "=" * 72 + f"\nDOCUMENT TITLE: {d['document_title']}\nSTATUS: {d['status']}\n" + "=" * 72)
    print("\n[CASE DETAILS]")
    for k, v in d["case_details"].items(): print(f"  {k.replace('_',' ').title()}: {v}")
    print(f"\n[FACTS]\n  {d['facts']}")
    print("\n[LEGAL GROUNDS]"); [print(f"  - {g}") for g in d["legal_grounds"]]
    print("\n[APPLICABLE LAWS]")
    if not d["applicable_laws"]: print(f"  {MISSING}")
    for l in d["applicable_laws"]:
        print(f"  - {l['citation']}  (cross-verified: {l['cross_verified']})")
        if l.get("text_or_holding"): print(f"    {l['text_or_holding'][:180]}...")
    if d["suggested_authorities"]:
        print("\n[SUGGESTED AUTHORITIES - from Research Agent]")
        for s in d["suggested_authorities"]: print(f"  - {s['citation']}  [{s['doc_id']}]")
    print("\n[ARGUMENTS]"); [print(f"  - {a}") for a in d["arguments"]]
    print("\n[PRAYER / RELIEF]"); [print(f"  - {p}") for p in d["prayer_relief"]]
    print(f"\n[VERIFICATION]\n  {d['verification']}")
    print("\n[SOURCES / CITATIONS]"); [print(f"  - {s['citation']}  [{s['doc_id']}]") for s in d["sources_citations"]]
    if d["provenance"]:
        print("\n[FACT PROVENANCE (from Case Review Agent)]")
        for k, v in d["provenance"].items(): print(f"  - {k}: {v}")
    print("\n" + "=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Legal Drafting Agent (India)")
    ap.add_argument("--type"); ap.add_argument("--case")
    ap.add_argument("--docs", nargs="+", help="Case documents; runs the Review Agent first")
    ap.add_argument("--verified-by"); ap.add_argument("--verified-on")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    doc_type = a.type or input("Document type: ").strip()
    agent = LegalDraftingAgent()
    if a.docs:
        draft = agent.draft_from_documents(doc_type, a.docs, a.case or "case_from_docs",
                                           a.verified_by, a.verified_on)
    else:
        draft = agent.draft(doc_type, a.case or input("Case ID: ").strip())

    if a.json or not sys.stdout.isatty():
        print(json.dumps(draft, indent=2, default=str))
    else:
        _pretty_print(draft)


if __name__ == "__main__":
    main()

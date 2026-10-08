"""
orchestrator.py   (Agent 6 of 6 - the controller)
=================================================
Controls all five agents, passes data between them, enforces order and guard-rails:

    Review -> Research -> Drafting -> Verification        (+ RAG Chatbot over everything)

Guard-rails
  * Drafting needs a Review of the uploaded documents first (auto-run if missing).
  * Verification re-checks every claim against the ORIGINAL pages + trusted corpus.
  * A draft is only marked READY FOR ADVOCATE REVIEW when Verification approves it.
  * Chatbot answers are re-verified line by line before being shown as verified.
  * Every step is written to an audit log.

Usage
  python orchestrator.py --docs fir.txt --task full --type bail --save out.json
  python orchestrator.py --docs fir.txt --task review
  python orchestrator.py --docs fir.txt --task research --query "bail in murder case"
  python orchestrator.py --docs fir.txt --task chat --question "Who is the accused?"
  python orchestrator.py --docs fir.txt            # interactive mode (type /help)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from legal_research_agent import LegalResearchAgent
from legal_case_review_agent import CaseReviewAgent, load_documents, _pretty_print as _print_review
from legal_drafting_agent import (LegalDraftingAgent, LEGAL_GROUNDS_MAP,
                                  register_case_from_review, _pretty_print as _print_draft)
from legal_verification_agent import LegalVerificationAgent, _pretty as _print_verification
from legal_rag_chatbot import LegalRAGChatbot

DEFAULT_CASE_ID = "case_from_docs"


class LegalOrchestrator:
    def __init__(self) -> None:
        # one shared Research agent -> every other agent talks to the same instance
        self.research_agent = LegalResearchAgent()
        self.review_agent = CaseReviewAgent(research_agent=self.research_agent)
        self.drafting_agent = LegalDraftingAgent(research_agent=self.research_agent,
                                                 review_agent=self.review_agent)
        self.verification_agent = LegalVerificationAgent(research_agent=self.research_agent)
        self.chatbot = LegalRAGChatbot([])
        self.docs: list[str] = []
        self.state: dict[str, Any] = {"review": None, "research": None, "draft": None,
                                      "verification": None, "doc_type": None}
        self.log: list[dict[str, Any]] = []

    # ---------------- plumbing ----------------
    def _log(self, step: str, agent: str, status: str, detail: str = "") -> None:
        self.log.append({"time": datetime.now().isoformat(timespec="seconds"),
                         "step": step, "agent": agent, "status": status, "detail": detail})

    def _need_docs(self) -> dict[str, Any] | None:
        if not self.docs:
            return {"error": "No documents loaded. Use /load <paths> or pass --docs."}
        return None

    def _refresh_chatbot_context(self) -> None:
        self.chatbot.add_pipeline_context(self.export())

    # ---------------- agent tasks ----------------
    def load_documents(self, paths: list[str]) -> dict[str, Any]:
        self.docs = list(paths)
        self.state = {k: None for k in self.state}
        self.chatbot = LegalRAGChatbot(self.docs)
        pages = load_documents(self.docs)
        warns = [p.text for p in pages if p.page_no == 0] + self.chatbot.warnings
        self._log("load", "orchestrator", "ok" if not warns else "warning",
                  f"{len([p for p in pages if p.page_no])} page(s)")
        return {"pages": len([p for p in pages if p.page_no]), "warnings": warns}

    def do_review(self) -> dict[str, Any]:
        if (err := self._need_docs()): return err
        rep = self.review_agent.review(self.docs)
        self.state["review"] = rep
        self._log("review", "CaseReviewAgent", "ok")
        # Review -> Research hand-off
        self.state["research"] = self.research_agent.research_from_review(rep)
        rep["research"] = self.state["research"]
        self._log("research(from review)", "LegalResearchAgent", "ok")
        self._refresh_chatbot_context()
        return rep

    def do_research(self, query: str | None = None) -> dict[str, Any]:
        if query:
            res = self.research_agent.research(query)
        else:
            if not self.state["review"]:
                if (err := self._need_docs()): return err
                self.do_review()
            res = self.state["research"]
        self.state["research"] = res
        self._log("research", "LegalResearchAgent", "ok", query or "from review")
        self._refresh_chatbot_context()
        return res

    def do_draft(self, doc_type: str, verified_by: str | None = None, verified_on: str | None = None,
                 overrides: dict[str, Any] | None = None, case_id: str = DEFAULT_CASE_ID) -> dict[str, Any]:
        doc_type = doc_type.lower().strip()
        if doc_type not in LEGAL_GROUNDS_MAP:
            return {"error": f"Unsupported document type: {doc_type}", "supported": list(LEGAL_GROUNDS_MAP)}
        if (err := self._need_docs()): return err
        if not self.state["review"]:
            self.do_review()
        # Review -> Drafting hand-off (facts carry page-level provenance)
        register_case_from_review(case_id, self.state["review"], verified_by, verified_on, overrides)
        draft = self.drafting_agent.draft(doc_type, case_id)
        self.state["draft"], self.state["doc_type"] = draft, doc_type
        self.state["verification"] = None          # any old verification is now stale
        self._log("draft", "LegalDraftingAgent", "error" if "error" in draft else "ok", doc_type)
        self._refresh_chatbot_context()
        return draft

    def do_verify(self) -> dict[str, Any]:
        if (err := self._need_docs()): return err
        if not self.state["review"]:
            self.do_review()
        pages = load_documents(self.docs)
        report = self.verification_agent.verify(self.export(include_verification=False), pages).to_dict()
        self.state["verification"] = report
        self._log("verify", "LegalVerificationAgent", "approved" if report["approved"] else "not_approved",
                  report["final_result"])
        self._refresh_chatbot_context()
        return report

    def do_chat(self, question: str, verify: bool = True) -> dict[str, Any]:
        ans = self.chatbot.ask(question)
        self._log("chat", "LegalRAGChatbot", ans["status"], question[:60])
        if verify:
            ans["verification"] = self.verification_agent.verify_chat_answer(ans, load_documents(self.docs))
            self._log("verify(chat)", "LegalVerificationAgent",
                      "approved" if ans["verification"]["approved"] else "not_approved")
        return ans

    def run_full(self, doc_type: str, verified_by: str | None = None, verified_on: str | None = None,
                 overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Review -> Research -> Drafting -> Verification, with a release gate."""
        self.do_review()
        draft = self.do_draft(doc_type, verified_by, verified_on, overrides)
        if "error" in draft:
            return self.export()
        self.do_verify()
        return self.export()

    # ---------------- release gate / export ----------------
    def release_status(self) -> str:
        v, d = self.state["verification"], self.state["draft"]
        if not d or "error" in (d or {}):
            return "NO DRAFT"
        if not v:
            return "NOT VERIFIED - run verification before use"
        if not v["approved"]:
            return "BLOCKED - verification did not approve; fix unsupported claims first"
        if d.get("verification") == "[MISSING INFORMATION]":
            return "READY FOR ADVOCATE REVIEW - advocate verification block still pending"
        return "READY FOR ADVOCATE REVIEW"

    def export(self, include_verification: bool = True) -> dict[str, Any]:
        out = {"review": self.state["review"], "research": self.state["research"],
               "draft": self.state["draft"]}
        if include_verification:
            out["verification"] = self.state["verification"]
            out["release_status"] = self.release_status()
            out["audit_log"] = self.log
        return out

    def status(self) -> str:
        s = self.state
        mark = lambda x: "done" if x else "-"
        return (f"documents : {', '.join(self.docs) or '(none)'}\n"
                f"review    : {mark(s['review'])}\nresearch  : {mark(s['research'])}\n"
                f"draft     : {mark(s['draft'])} {s['doc_type'] or ''}\n"
                f"verified  : {mark(s['verification'])}\nrelease   : {self.release_status()}")

    # ---------------- natural-language router ----------------
    def detect_doc_type(self, text: str) -> str | None:
        t = text.lower()
        for key in sorted(LEGAL_GROUNDS_MAP, key=len, reverse=True):
            pat = r"\b" + key.replace("_", "[ _]") + r"\b"
            if re.search(pat, t):
                return key
        return None

    def route(self, text: str) -> str:
        t = text.lower()
        if re.search(r"full pipeline|end[- ]to[- ]end|run (all|everything)", t):
            return "full"
        if re.search(r"\b(verify|validate|audit|fact[- ]check)\b", t):
            return "verify"
        if re.search(r"\b(draft|prepare|write|generate|create)\b", t) and self.detect_doc_type(t):
            return "draft"
        if re.search(r"\b(review|summar\w+|key facts|timeline|extract)\b", t):
            return "review"
        if re.search(r"\b(research|precedent|case law|applicable law|which (law|section)s?|judgments?)\b", t):
            return "research"
        return "chat"

    def handle(self, text: str) -> tuple[str, Any]:
        intent = self.route(text)
        if intent == "full":
            dt = self.detect_doc_type(text) or self.state["doc_type"]
            if not dt:
                return "error", {"error": "Say which document, e.g. 'run full pipeline for bail'."}
            return "full", self.run_full(dt)
        if intent == "verify":
            return "verify", self.do_verify()
        if intent == "draft":
            return "draft", self.do_draft(self.detect_doc_type(text))
        if intent == "review":
            return "review", self.do_review()
        if intent == "research":
            return "research", self.do_research(text if not self.docs or "from" not in text.lower() else None)
        return "chat", self.do_chat(text)


# ---------------- presentation ----------------
def _print_research(r: dict[str, Any]) -> None:
    if "error" in r:
        print(f"ERROR: {r['error']}"); return
    print("\n" + "=" * 70 + "\nLEGAL RESEARCH\n" + "=" * 70)
    print("Issues:", ", ".join(r.get("legal_issue", [])) or "-")
    for l in r.get("relevant_laws", []):
        print(f"  - Section {l.get('section','')} {l['title']}")
    for j in r.get("relevant_judgments", []):
        print(f"  - {j['title']} {j.get('citation','')}")
    if not r.get("relevant_laws") and not r.get("relevant_judgments"):
        print("  Not verified - nothing retrieved.")
    print()


def _print_chat(a: dict[str, Any]) -> None:
    print("\n" + a["answer"])
    for r in a.get("references", []):
        print(f"  source: {r['label']}  [{r['origin']}]")
    v = a.get("verification")
    if v:
        print(f"  verification: {v['final_result']}")
    print()


def show(kind: str, result: Any, orch: LegalOrchestrator) -> None:
    if isinstance(result, dict) and "error" in result and kind != "full":
        print(f"ERROR: {result['error']}")
        if "supported" in result: print("Supported:", ", ".join(result["supported"]))
        return
    if kind == "review": _print_review(result)
    elif kind == "research": _print_research(result)
    elif kind == "draft": _print_draft(result)
    elif kind == "verify": _print_verification(result)
    elif kind == "chat": _print_chat(result)
    elif kind == "full":
        if result.get("review"): _print_review(result["review"])
        if result.get("draft"): _print_draft(result["draft"])
        if result.get("verification"): _print_verification(result["verification"])
        print(f"RELEASE STATUS: {result.get('release_status')}\n")


HELP = """
Commands                                   Or just type naturally, e.g.
  /load <paths...>      load documents       "summarise the case"
  /review               case review          "which sections apply to murder"
  /research [query]     legal research       "draft a bail application"
  /draft <type>         draft a document     "verify everything"
  /verify               verify all outputs   "who is the accused?"   (-> chatbot)
  /full <type>          run all stages
  /chat <question>      ask the chatbot     /status  /log  /save <file>  /help  /quit
Draft types: """ + ", ".join(LEGAL_GROUNDS_MAP) + "\n"


def interactive(orch: LegalOrchestrator) -> None:
    print("\nLEGAL ORCHESTRATOR - type /help for commands.\n" + orch.status() + "\n")
    while True:
        try:
            line = input("orchestrator> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/help": print(HELP); continue
        if line == "/status": print(orch.status() + "\n"); continue
        if line == "/log":
            for e in orch.log: print(f"  {e['time']}  {e['agent']:<22} {e['step']:<22} {e['status']}  {e['detail']}")
            print(); continue
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd == "/load":
            r = orch.load_documents(arg.split()); print(f"Loaded {r['pages']} page(s).", *r["warnings"], sep="\n  ") ; continue
        if cmd == "/save":
            Path(arg or "pipeline.json").write_text(json.dumps(orch.export(), indent=2, default=str), encoding="utf-8")
            print(f"Saved to {arg or 'pipeline.json'}\n"); continue
        if cmd == "/review": show("review", orch.do_review(), orch)
        elif cmd == "/research": show("research", orch.do_research(arg or None), orch)
        elif cmd == "/draft": show("draft", orch.do_draft(arg), orch)
        elif cmd == "/verify": show("verify", orch.do_verify(), orch)
        elif cmd == "/full": show("full", orch.run_full(arg), orch)
        elif cmd == "/chat": show("chat", orch.do_chat(arg), orch)
        elif line.startswith("/"): print("Unknown command. Type /help.")
        else:
            kind, res = orch.handle(line)
            show(kind, res, orch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Legal Orchestrator Agent")
    ap.add_argument("--docs", nargs="*", default=[])
    ap.add_argument("--task", choices=["full", "review", "research", "draft", "verify", "chat"])
    ap.add_argument("--type", help="Document type for draft/full (e.g. bail, fir, appeal)")
    ap.add_argument("--query", help="Research query")
    ap.add_argument("--question", help="Chatbot question")
    ap.add_argument("--verified-by"); ap.add_argument("--verified-on")
    ap.add_argument("--save", help="Write full pipeline JSON (UTF-8) to this file")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    orch = LegalOrchestrator()
    if a.docs:
        info = orch.load_documents(a.docs)
        for w in info["warnings"]:
            print(f"WARNING: {w}", file=sys.stderr)

    if not a.task:
        interactive(orch); return

    t = a.task
    if t == "full":     res = orch.run_full(a.type or "bail", a.verified_by, a.verified_on)
    elif t == "review": res = orch.do_review()
    elif t == "research": res = orch.do_research(a.query)
    elif t == "draft":  res = orch.do_draft(a.type or "bail", a.verified_by, a.verified_on)
    elif t == "verify":
        orch.do_review(); orch.do_draft(a.type or "bail", a.verified_by, a.verified_on); res = orch.do_verify()
    else:               res = orch.do_chat(a.question or input("Question: "))

    if a.save:
        Path(a.save).write_text(json.dumps(orch.export(), indent=2, default=str), encoding="utf-8")
        print(f"Saved pipeline JSON to {a.save}", file=sys.stderr)
    if a.json or not sys.stdout.isatty():
        print(json.dumps(res, indent=2, default=str))
    else:
        show(t, res, orch)


if __name__ == "__main__":
    main()

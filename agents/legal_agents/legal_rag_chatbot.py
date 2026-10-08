"""
legal_rag_chatbot.py   (Agent 4 of 6)
=====================================
Answers questions ONLY from (a) uploaded documents, (b) the shared trusted corpus,
and (c) optional pipeline context (review / research / draft / verification output).
Every answer line carries a source label; unsupported questions get NOT_FOUND.

Inter-links
  <- orchestrator             : creates it with the loaded docs and feeds pipeline context
  <- legal_verification_agent : verify_chat_answer() re-checks every answer line
  uses corpus.py              : same trusted corpus as Research / Drafting agents

Run standalone:
    python legal_rag_chatbot.py --docs fir.txt
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from corpus import load_corpus

CHAT_NOT_FOUND = "Not found in the available sources."


# ---------------- Chunks ----------------
@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_no: int
    text: str
    origin: str = "document"      # "document" | "corpus" | "pipeline"
    aliases: str = ""             # index-only terms (e.g. "CrPC"); never shown in answers

    def index_text(self) -> str:
        """Text used for matching: body + (for corpus/pipeline chunks) the label and aliases."""
        return self.text if self.origin == "document" else f"{self.text} {self.doc_name} {self.aliases}"


def load_documents(paths: list[str]) -> tuple[list[Chunk], list[str]]:
    chunks: list[Chunk] = []
    warnings: list[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            warnings.append(f"File not found: {path}")
            continue
        sfx = path.suffix.lower()
        if sfx == ".pdf":
            c, w = _load_pdf(path)
            chunks.extend(c); warnings.extend(w)
        elif sfx in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, pg in enumerate(text.split("\f") if "\f" in text else [text], start=1):
                chunks.extend(_split(pg, path.name, i))
        else:
            warnings.append(f"Unsupported file type: {path.name}")
    return chunks, warnings


def _load_pdf(path: Path) -> tuple[list[Chunk], list[str]]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return [], ["PDF support requires: pip install pdfplumber"]
    out: list[Chunk] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, pg in enumerate(pdf.pages, start=1):
            out.extend(_split(pg.extract_text() or "", path.name, i))
    return out, []


def _split(text: str, doc: str, page: int, target: int = 600) -> list[Chunk]:
    if not text.strip():
        return []
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]
    chunks, buf, idx = [], "", 0
    for para in paras:
        if len(buf) + len(para) <= target:
            buf += ("\n\n" if buf else "") + para
        else:
            chunks.append(Chunk(f"{doc}#p{page}#{idx}", doc, page, buf)); idx += 1; buf = para
    if buf:
        chunks.append(Chunk(f"{doc}#p{page}#{idx}", doc, page, buf))
    return chunks


def corpus_to_chunks() -> list[Chunk]:
    out = []
    for d in load_corpus():
        label = (f"Section {d.get('section','')} {d['title']}".strip() if d["type"] == "statute"
                 else f"{d['title']} {d.get('citation','')}".strip())
        title = d["title"].lower()
        alias = "IPC" if "penal" in title else "CrPC" if "criminal procedure" in title else ""
        out.append(Chunk(f"corpus#{d['doc_id']}", label, 0, d.get("text") or d.get("holding") or "",
                         "corpus", aliases=alias))
    return out


# ---------------- Retriever ----------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "was", "were", "of", "to", "in", "on", "and", "or", "for",
         "by", "with", "that", "this", "it", "as", "at", "be",
         # question / filler words must never count as topical evidence
         "what", "who", "whom", "whose", "which", "when", "where", "why", "how", "does", "did",
         "are", "can", "could", "would", "should", "will", "shall", "there", "any", "about",
         "tell", "give", "show", "from", "have", "has", "had", "been", "being", "please",
         "explain", "describe", "list", "whether", "into", "than", "then", "also", "such"}


def _tok(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2]


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens = [_tok(c.index_text()) for c in chunks]
        self.df: Counter = Counter()
        for tk in self.tokens:
            for t in set(tk):
                self.df[t] += 1
        self.N = max(len(chunks), 1)

    def _idf(self, t: str) -> float:
        return math.log((self.N + 1) / (self.df.get(t, 0) + 1)) + 1.0

    def _score(self, qt: list[str], dt: list[str]) -> float:
        if not qt or not dt:
            return 0.0
        qc, dc = Counter(qt), Counter(dt)
        s = sum((qn * self._idf(t)) * (dc[t] * self._idf(t)) for t, qn in qc.items() if t in dc)
        return s / math.sqrt(len(dt) + 1)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[float, Chunk]]:
        qt = _tok(query)
        if not qt:
            return []
        scored = sorted(((self._score(qt, dt), c) for dt, c in zip(self.tokens, self.chunks)),
                        key=lambda x: x[0], reverse=True)
        return [(s, c) for s, c in scored[:k] if s > 0]


# ---------------- Conversation memory ----------------
@dataclass
class Turn:
    role: str
    text: str
    refs: list[dict[str, Any]] = field(default_factory=list)


class Conversation:
    def __init__(self, max_turns: int = 8):
        self.turns: list[Turn] = []
        self.max_turns = max_turns

    def add_user(self, text: str) -> None:
        self._append(Turn("user", text))

    def add_assistant(self, text: str, refs: list[dict[str, Any]]) -> None:
        self._append(Turn("assistant", text, refs))

    def _append(self, t: Turn) -> None:
        self.turns.append(t)
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    def context_text(self, n: int = 3) -> str:
        return " ".join(t.text for t in self.turns[-n * 2:] if t.role == "user")

    def last_refs(self) -> list[dict[str, Any]]:
        for t in reversed(self.turns):
            if t.role == "assistant" and t.refs:
                return t.refs
        return []


# ---------------- Chatbot ----------------
@dataclass
class ChatAnswer:
    answer: str = CHAT_NOT_FOUND
    references: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    status: str = "not_found"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LegalRAGChatbot:
    def __init__(self, doc_paths: list[str] | None = None, k: int = 5):
        self.docs, self.warnings = load_documents(doc_paths or [])
        self.corpus = corpus_to_chunks()
        self.pipeline_chunks: list[Chunk] = []
        self.k = k
        self.conversation = Conversation()
        self._rebuild()

    def _rebuild(self) -> None:
        self.all_chunks = self.docs + self.corpus + self.pipeline_chunks
        self.retriever = Retriever(self.all_chunks)

    # link: Orchestrator -> Chatbot (so users can ask about review/research/draft/verification)
    def add_pipeline_context(self, pipeline: dict[str, Any]) -> None:
        ch: list[Chunk] = []
        review = pipeline.get("review") or {}
        research = pipeline.get("research") or {}
        draft = pipeline.get("draft") or {}
        ver = pipeline.get("verification") or {}

        if review.get("missing_information"):
            ch.append(Chunk("pipe#review_missing", "Review: missing information", 0,
                            "Missing information in the case documents: "
                            + "; ".join(review["missing_information"]) + ".", "pipeline"))
        auth = [f"Section {l.get('section','')} {l['title']}" for l in research.get("relevant_laws", [])] + \
               [f"{j['title']} {j.get('citation','')}" for j in research.get("relevant_judgments", [])]
        if auth:
            ch.append(Chunk("pipe#research", "Research: relevant authorities", 0,
                            "Research found these relevant authorities: " + "; ".join(auth) + ".", "pipeline"))
        if draft and "error" not in draft:
            title = draft.get("document_title", "draft")
            lines = [f"The draft document is titled {title}.",
                     "Prayer in the draft: " + " ".join(draft.get("prayer_relief", [])),
                     "Legal grounds in the draft: " + "; ".join(draft.get("legal_grounds", [])) + "."]
            laws = [l.get("citation") or l.get("title") for l in draft.get("applicable_laws", [])]
            if laws:
                lines.append("Applicable laws cited in the draft: " + "; ".join(laws) + ".")
            lines.append("Verification status of the draft: " + str(draft.get("verification")) + ".")
            ch.append(Chunk("pipe#draft", f"Draft: {title}", 0, " ".join(lines), "pipeline"))
        if ver.get("final_result"):
            ch.append(Chunk("pipe#verification", "Verification: final result", 0,
                            "Verification result: " + ver["final_result"], "pipeline"))
        self.pipeline_chunks = ch
        self._rebuild()

    def ask(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            self.conversation.add_user(question)
            self.conversation.add_assistant(CHAT_NOT_FOUND, [])
            return ChatAnswer(status="empty_question").to_dict()
        self.conversation.add_user(question)

        query = f"{self.conversation.context_text(n=3)} {question}".strip()
        hits = self.retriever.retrieve(query, k=self.k)
        if not hits:
            self.conversation.add_assistant(CHAT_NOT_FOUND, [])
            return ChatAnswer(status="no_retrieval").to_dict()

        q_terms = set(_tok(question))
        filtered = [(s, c) for s, c in hits
                    if len(q_terms & set(_tok(c.index_text()))) >= 1 or c.origin != "document"] or hits

        text, refs = self._compose(question, filtered)
        if text is None:
            self.conversation.add_assistant(CHAT_NOT_FOUND, [])
            return ChatAnswer(status="not_found",
                              retrieved_chunks=[self._ref(c) for _, c in filtered]).to_dict()

        ans = ChatAnswer(answer=text, references=refs,
                         confidence=min(1.0, round(filtered[0][0] / 5.0, 2)),
                         retrieved_chunks=[self._ref(c) for _, c in filtered], status="answered")
        self.conversation.add_assistant(ans.answer, refs)
        return ans.to_dict()

    def _compose(self, question: str, hits: list[tuple[float, Chunk]]):
        q_terms = set(_tok(question))
        if not q_terms:
            return None, []
        evidence = []
        for score, c in hits:
            for s in re.split(r"(?<=[.!?])\s+", c.text):
                sc = " ".join(s.split())
                s_terms = set(_tok(sc)) | (set(_tok(c.doc_name + " " + c.aliases)) if c.origin != "document" else set())
                if len(sc) >= 20 and q_terms & s_terms:
                    evidence.append((len(q_terms & s_terms) + score, c, sc))
        if not evidence:
            return None, []
        evidence.sort(key=lambda x: x[0], reverse=True)
        lines, refs, seen = [], [], set()
        for _, c, sent in evidence[:6]:
            if sent in seen:
                continue
            seen.add(sent)
            ref = self._ref(c)
            lines.append(f"- {sent}  ({ref['label']})")
            if ref not in refs:
                refs.append(ref)
        return "Based on the available sources:\n" + "\n".join(lines), refs

    @staticmethod
    def _ref(c: Chunk) -> dict[str, Any]:
        if c.origin == "document":
            return {"label": f"{c.doc_name} p.{c.page_no}", "document": c.doc_name,
                    "page": c.page_no, "origin": "document", "chunk_id": c.chunk_id}
        return {"label": c.doc_name, "document": None, "page": None,
                "origin": c.origin, "chunk_id": c.chunk_id}


# ---------------- CLI ----------------
def _print_answer(a: dict[str, Any]) -> None:
    print("\n" + "-" * 60 + "\nANSWER\n" + "-" * 60 + "\n" + a["answer"])
    if a.get("references"):
        print("\nSOURCES")
        for r in a["references"]:
            print(f"  - {r['label']}  [{r['origin']}]")
    print(f"\nconfidence: {a.get('confidence', 0.0)}  status: {a.get('status','')}\n" + "-" * 60 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Legal RAG Chatbot")
    ap.add_argument("--docs", nargs="*")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    paths = a.docs
    if paths is None and sys.stdin.isatty():
        raw = input("Document paths (space-separated, blank for corpus-only): ").strip()
        paths = raw.split() if raw else []
    bot = LegalRAGChatbot(paths or [], k=a.k)
    for w in bot.warnings:
        print(f"WARNING: {w}")
    if sys.stdin.isatty() and not a.json:
        print("\nLEGAL RAG CHATBOT   (commands: :quit :refs :clear :docs)\n")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not line:
            continue
        if line in (":quit", ":exit", ":q"):
            break
        if line == ":clear":
            bot.conversation = Conversation(); print("Conversation cleared.\n"); continue
        if line == ":refs":
            for r in bot.conversation.last_refs():
                print(f"  - {r['label']}  [{r['origin']}]")
            continue
        if line == ":docs":
            seen = sorted({(c.doc_name, c.page_no) for c in bot.docs})
            print("\n".join(f"  - {d} p.{p}" for d, p in seen) or "  (none)"); continue
        ans = bot.ask(line)
        print(json.dumps(ans, indent=2, default=str)) if (a.json or not sys.stdout.isatty()) else _print_answer(ans)


if __name__ == "__main__":
    main()

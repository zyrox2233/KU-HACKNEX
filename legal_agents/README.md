# Legal Agents (India)

Run everything from inside this folder (plain sibling imports, no install needed).

| File | Role |
|---|---|
| `common.py` | Shared constants (`MISSING`, `NOT_FOUND`), tokenizer, helpers |
| `corpus.py` | Single trusted legal corpus (statutes + judgments) |
| `legal_case_review_agent.py` | Agent 1 - extracts page-cited facts from PDF/TXT |
| `legal_research_agent.py` | Agent 2 - closed-corpus RAG research + citation verification |
| `legal_drafting_agent.py` | Agent 3 - drafts documents from registered facts + verified law |
| `pipeline.py` | Runs Review -> Research -> Drafting together |

## How they connect
Review --(report)--> Research   : `LegalResearchAgent.research_from_review(report)`
Review --(report)--> Drafting   : `register_case_from_review(case_id, report)`
Drafting --(each citation)--> Research : `verify_citation(doc_id)`, `research_for_case(case)`

## Usage
    python legal_case_review_agent.py --docs fir.txt case.pdf --research
    python legal_research_agent.py --query "bail in murder case"
    python legal_drafting_agent.py --type bail --docs fir.txt --verified-by "Adv. X" --verified-on 2026-10-08
    python pipeline.py --docs fir.txt --type bail
PDF input needs `pip install pdfplumber`.

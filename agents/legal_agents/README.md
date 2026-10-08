# Legal Agents (India) - 6-agent system

Run everything from INSIDE this folder (plain sibling imports, nothing to install).
Needs Python 3.9+.  PDF input needs:  pip install pdfplumber

## Files
| File | Role |
|---|---|
| `orchestrator.py` | **Controller** - runs and links all agents, release gate, audit log, interactive mode |
| `legal_case_review_agent.py` | Extracts page-cited facts, parties, dates, evidence, issues |
| `legal_research_agent.py` | Closed-corpus legal research + citation verification |
| `legal_drafting_agent.py` | Drafts FIR / bail / appeal / etc. from verified facts + verified law |
| `legal_verification_agent.py` | Re-checks every claim vs. the original pages + trusted corpus |
| `legal_rag_chatbot.py` | Q&A grounded in documents + corpus + pipeline output |
| `pipeline.py` | One-command wrapper over the orchestrator |
| `common.py`, `corpus.py` | Shared helpers and the single trusted legal corpus |
| `sample_fir.txt` | Sample document to test with |

## Data flow
    Documents -> Review -> Research -> Drafting -> Verification -> release gate
                    \________________________________/
                   Chatbot answers questions over all of it
                   (and every chatbot answer is re-verified line by line)

## Quick start
    python orchestrator.py --docs sample_fir.txt --task full --type bail
    python orchestrator.py --docs sample_fir.txt                  (interactive; type /help)
    python orchestrator.py --docs sample_fir.txt --task chat --question "Who is the accused?"
    python pipeline.py --docs sample_fir.txt --type bail --save pipeline.json
    python legal_verification_agent.py --pipeline pipeline.json --docs sample_fir.txt

Windows PowerShell: use --save, not ">" redirection (PowerShell writes UTF-16).

## Interactive mode examples
    /load fir.txt case.pdf        /review        /research bail in murder case
    /draft bail                   /verify        /full anticipatory_bail
    summarise the case            draft a bail application        verify everything
    who is the complainant?       (anything else goes to the chatbot)

## Rules enforced
* Nothing outside the documents or trusted corpus is used; absent data shows as [MISSING INFORMATION].
* A draft is "READY FOR ADVOCATE REVIEW" only if Verification approves it; otherwise BLOCKED.
* Output is a draft for a qualified advocate - never file it unreviewed.
* Corpus covers IPC/CrPC only; cases after 1 July 2024 fall under BNS/BNSS/BSA (add those texts).

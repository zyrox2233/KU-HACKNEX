"""
corpus.py
=========
Single trusted legal corpus shared by the Research and Drafting agents.
Closed allow-list: only entries with verified=True AND a source_url on a
trusted domain are ever served.
"""
from __future__ import annotations

from typing import Any

from common import TRUSTED_DOMAINS

_IC_IPC = "https://www.indiacode.nic.in/handle/123456789/2263"
_IC_CRPC = "https://www.indiacode.nic.in/handle/123456789/1521"
_SCI = "https://main.sci.gov.in/judgments"


def _statute(doc_id, title, section, text, url):
    return {"doc_id": doc_id, "type": "statute", "title": title, "section": section,
            "text": text, "jurisdiction": "India", "source_url": url, "verified": True}


def _judgment(doc_id, title, citation, year, holding):
    return {"doc_id": doc_id, "type": "judgment", "title": title, "citation": citation,
            "court": "Supreme Court of India", "year": year, "holding": holding,
            "source_url": _SCI, "verified": True}


IPC, CRPC = "Indian Penal Code, 1860", "Code of Criminal Procedure, 1973"

CORPUS: list[dict[str, Any]] = [
    # ---------------- STATUTES ----------------
    _statute("ipc_300", IPC, "300",
        "Murder. Except in the cases hereinafter excepted, culpable homicide is murder, if the "
        "act by which the death is caused is done with the intention of causing death; or with "
        "the intention of causing such bodily injury as the offender knows to be likely to cause "
        "the death of the person to whom the harm is caused; or with the intention of causing "
        "bodily injury to any person and the bodily injury intended to be inflicted is sufficient "
        "in the ordinary course of nature to cause death; or with the knowledge that the act is so "
        "imminently dangerous that it must in all probability cause death. Five exceptions reduce "
        "murder to culpable homicide not amounting to murder: grave and sudden provocation, "
        "exceeding right of private defence, public servant exceeding power, sudden fight, and consent.",
        _IC_IPC),
    _statute("ipc_302", IPC, "302",
        "Punishment for murder. Whoever commits murder shall be punished with death, or "
        "imprisonment for life, and shall also be liable to fine.", _IC_IPC),
    _statute("ipc_304", IPC, "304",
        "Punishment for culpable homicide not amounting to murder. Part I: if the act is done with "
        "intention to cause death or such bodily injury as is likely to cause death - imprisonment "
        "for life, or imprisonment up to 10 years, and fine. Part II: if the act is done with "
        "knowledge that it is likely to cause death but without intention - imprisonment up to 10 "
        "years, or fine, or both.", _IC_IPC),
    _statute("crpc_154", CRPC, "154",
        "Information in cognizable cases. Every information relating to the commission of a "
        "cognizable offence, if given orally, shall be reduced to writing, read over to the "
        "informant, and signed by the person giving it. A copy shall be given free of cost to the "
        "informant.", _IC_CRPC),
    _statute("crpc_161", CRPC, "161",
        "Examination of witnesses by police. A police officer may examine orally any person "
        "supposed to be acquainted with the facts and circumstances of the case. The person is "
        "bound to answer truly, except where answers would expose them to criminal charge, penalty "
        "or forfeiture. Statements may be recorded in writing or by audio-video electronic means.",
        _IC_CRPC),
    _statute("crpc_162", CRPC, "162",
        "Statements to police not to be signed - use of such statements in evidence. No statement "
        "made to a police officer in the course of an investigation shall, if reduced to writing, "
        "be signed by the person making it. Such statements may be used only to contradict a "
        "witness as provided in Section 145 of the Indian Evidence Act, 1872.", _IC_CRPC),
    _statute("crpc_164", CRPC, "164",
        "Recording of confessions and statements. A Magistrate may record a confession or "
        "statement. The Magistrate must apply judicial mind to ascertain the voluntariness of the "
        "confession and must comply strictly with statutory requirements.", _IC_CRPC),
    _statute("crpc_374", CRPC, "374", "Appeals from convictions.", _IC_CRPC),
    _statute("crpc_386", CRPC, "386", "Powers of the Appellate Court.", _IC_CRPC),
    _statute("crpc_437", CRPC, "437",
        "When bail may be taken in case of non-bailable offence.", _IC_CRPC),
    _statute("crpc_438", CRPC, "438",
        "Direction for grant of bail to person apprehending arrest.", _IC_CRPC),
    _statute("crpc_439", CRPC, "439",
        "Special powers of High Court or Court of Session regarding bail.", _IC_CRPC),

    # ---------------- JUDGMENTS ----------------
    _judgment("sc_bachan_singh_1980", "Bachan Singh v. State of Punjab", "(1980) 2 SCC 684", 1980,
        "The Constitution Bench upheld the constitutional validity of the death penalty but "
        "restricted its application to the rarest of rare cases where the crime is of such "
        "exceptional depravity that no other sentence would be adequate. Both the circumstances of "
        "the crime and the circumstances of the offender must be considered, and a balance sheet "
        "of aggravating and mitigating circumstances must be drawn."),
    _judgment("sc_machhi_singh_1983", "Machhi Singh v. State of Punjab", "(1983) 3 SCC 470", 1983,
        "Elaborated on what constitutes rarest of rare cases: manner of crime, motive, professional "
        "approach, nature of victim, and possibility of reformation. Death penalty is an "
        "exception; life imprisonment is the rule."),
    _judgment("sc_bariyar_2009",
        "Santosh Kumar Satishbhushan Bariyar v. State of Maharashtra", "(2009) 6 SCC 498", 2009,
        "Established a two-part test: first, determine whether the case falls under the rarest of "
        "rare doctrine; second, consider life imprisonment as the initial alternative. If death "
        "penalty is chosen, reasons must be given explaining why the convict cannot be reformed."),
    _judgment("sc_sharad_birdhichand_1984",
        "Sharad Birdhichand Sarda v. State of Maharashtra", "(1984) 4 SCC 116", 1984,
        "Established the panchsheel (five golden principles) for conviction based on circumstantial "
        "evidence: (1) circumstances must be fully established; (2) facts must be consistent only "
        "with the hypothesis of guilt; (3) circumstances must be conclusive; (4) they must exclude "
        "every possible hypothesis except guilt; (5) there must be a complete chain of evidence. "
        "Suspicion, however strong, cannot take the place of proof."),
    _judgment("sc_lalita_kumari_2013", "Lalita Kumari v. Government of U.P.", "(2014) 2 SCC 1", 2013,
        "Registration of FIR under Section 154 CrPC is mandatory when information discloses "
        "commission of a cognizable offence. The obligation to register FIR is the first step to "
        "access to justice and upholds the rule of law."),
    _judgment("sc_rabindra_kumar_pal", "Rabindra Kumar Pal @ Dara Singh v. Republic of India",
        "(2011) 2 SCC 490", 2011,
        "A Magistrate recording a confession must satisfy his conscience that the statement is not "
        "made under extraneous influence. The Magistrate must disclose that he is a Magistrate and "
        "must make a real endeavour to ascertain the voluntary character of the confession."),
]


def is_trusted(doc: dict[str, Any]) -> bool:
    url = doc.get("source_url", "")
    return doc.get("verified") is True and any(d in url for d in TRUSTED_DOMAINS)


def load_corpus() -> list[dict[str, Any]]:
    return [d for d in CORPUS if is_trusted(d)]


def get_verified_source(doc_id: str) -> dict[str, Any] | None:
    for d in CORPUS:
        if d["doc_id"] == doc_id and is_trusted(d):
            return d
    return None

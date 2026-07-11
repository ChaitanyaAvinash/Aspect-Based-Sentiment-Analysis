"""SemEval-2014 (and 2015/16 Opinions) XML parser into normalized ABSAExamples."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import get_args

from absa.data.schema import ABSAExample, AspectCategory, AspectTerm, Domain, Polarity

_VALID_POLARITIES = set(get_args(Polarity))


def _norm_polarity(raw: str | None, *, drop_conflict: bool) -> str | None:
    if raw is None:
        return None
    p = raw.strip().lower()
    if p in _VALID_POLARITIES:
        return p
    if p == "conflict":
        return None if drop_conflict else "neutral"
    return None  # unknown label -> skip


def _parse_span(elem: ET.Element) -> tuple[int, int]:
    try:
        start = int(elem.attrib.get("from", "-1"))
        end = int(elem.attrib.get("to", "-1"))
    except ValueError:
        return (-1, -1)
    if start == 0 and end == 0:  # SemEval marks implicit targets as 0,0
        return (-1, -1)
    return (start, end)


def _sentence_to_example(
    sentence: ET.Element, domain: Domain, *, drop_conflict: bool
) -> ABSAExample | None:
    sent_id = sentence.attrib.get("id", "")
    text_elem = sentence.find("text")
    text = (text_elem.text or "") if text_elem is not None else ""
    if not text:
        return None

    terms: list[AspectTerm] = []
    categories: list[AspectCategory] = []

    # SemEval-2014: <aspectTerms><aspectTerm .../></aspectTerms>
    for at in sentence.iter("aspectTerm"):
        pol = _norm_polarity(at.attrib.get("polarity"), drop_conflict=drop_conflict)
        term = at.attrib.get("term", "").strip()
        if not term or pol is None:
            continue
        start, end = _parse_span(at)
        terms.append(AspectTerm(term=term, polarity=pol, start=start, end=end))  # type: ignore[arg-type]

    for ac in sentence.iter("aspectCategory"):
        pol = _norm_polarity(ac.attrib.get("polarity"), drop_conflict=drop_conflict)
        cat = ac.attrib.get("category", "").strip()
        if not cat or pol is None:
            continue
        categories.append(AspectCategory(category=cat, polarity=pol))  # type: ignore[arg-type]

    # SemEval-2015/16: <Opinions><Opinion target=.. category=.. polarity=.. from=.. to=..>
    for op in sentence.iter("Opinion"):
        pol = _norm_polarity(op.attrib.get("polarity"), drop_conflict=drop_conflict)
        if pol is None:
            continue
        target = op.attrib.get("target", "").strip()
        if target and target.upper() != "NULL":
            start, end = _parse_span(op)
            terms.append(AspectTerm(term=target, polarity=pol, start=start, end=end))  # type: ignore[arg-type]
        category = op.attrib.get("category", "").strip()
        if category:
            categories.append(AspectCategory(category=category, polarity=pol))  # type: ignore[arg-type]

    # Drop terms whose recorded span does not actually match (annotation noise).
    terms = [t for t in terms if t.aligns_with(text)]
    return ABSAExample(
        id=sent_id, text=text, domain=domain, aspect_terms=terms, aspect_categories=categories
    )


def parse_semeval_xml(
    path: str | Path, domain: Domain = "other", *, drop_conflict: bool = True
) -> list[ABSAExample]:
    """Parse a SemEval XML file into normalized examples."""
    tree = ET.parse(path)
    root = tree.getroot()
    examples: list[ABSAExample] = []
    for sentence in root.iter("sentence"):
        ex = _sentence_to_example(sentence, domain, drop_conflict=drop_conflict)
        if ex is not None:
            examples.append(ex)
    return examples


def parse_semeval_string(
    xml: str, domain: Domain = "other", *, drop_conflict: bool = True
) -> list[ABSAExample]:
    """Parse SemEval XML from a string (used in tests)."""
    root = ET.fromstring(xml)
    examples: list[ABSAExample] = []
    for sentence in root.iter("sentence"):
        ex = _sentence_to_example(sentence, domain, drop_conflict=drop_conflict)
        if ex is not None:
            examples.append(ex)
    return examples


__all__ = ["parse_semeval_string", "parse_semeval_xml"]

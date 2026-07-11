"""Tests for the data layer: schema, preprocessing, parser, io, splits, stats."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from absa.data import (
    ABSAExample,
    AspectCategory,
    AspectTerm,
    SplitRatios,
    clean_text,
    label_distribution,
    load_sample,
    looks_english,
    make_splits,
    parse_semeval_string,
    parse_semeval_xml,
    read_jsonl,
    spans_to_bio,
    tokenize,
    write_jsonl,
)
from absa.data.preprocessing import normalize_whitespace, strip_html


# schema
def test_aspect_term_alignment() -> None:
    t = AspectTerm(term="pizza", polarity="positive", start=4, end=9)
    assert t.aligns_with("The pizza was good")
    assert not t.aligns_with("The sushi was good")
    assert t.span == (4, 9)
    assert t.is_explicit


def test_implicit_term_is_explicit_false() -> None:
    t = AspectTerm(term="service", polarity="neutral")
    assert not t.is_explicit
    assert t.aligns_with("anything")  # implicit terms never fail alignment


def test_example_rejects_misaligned_span() -> None:
    with pytest.raises(ValidationError):
        ABSAExample(
            id="x",
            text="The pizza was good",
            aspect_terms=[AspectTerm(term="sushi", polarity="positive", start=4, end=9)],
        )


# preprocessing
def test_strip_html_and_entities() -> None:
    out = strip_html("<b>Nice</b> &amp; cheap")
    assert "<" not in out and ">" not in out
    assert "&amp;" not in out
    assert "Nice" in out and "& cheap" in out


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("  a\t b\n c ") == "a b c"


def test_clean_text_truncates_and_guards() -> None:
    assert clean_text("<p>hi   there</p>") == "hi there"
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]
    long = "a " * 2000
    assert len(clean_text(long, max_chars=50)) <= 50


def test_tokenize_offsets_roundtrip() -> None:
    text = "The battery life is great."
    for tok in tokenize(text):
        assert text[tok.start : tok.end] == tok.text


def test_spans_to_bio() -> None:
    text = "The battery life is great"
    toks = tokenize(text)
    tags = spans_to_bio(toks, [(4, 16)])  # "battery life"
    assert tags == ["O", "B-ASP", "I-ASP", "O", "O"]


def test_spans_to_bio_ignores_null_span() -> None:
    toks = tokenize("nothing here")
    assert spans_to_bio(toks, [(-1, -1)]) == ["O", "O"]


def test_looks_english() -> None:
    assert looks_english("The food was great")
    assert not looks_english("这家餐厅的食物很好吃")


# SemEval parser
SEMEVAL_2014 = """
<sentences>
  <sentence id="1">
    <text>The pizza was great but service slow.</text>
    <aspectTerms>
      <aspectTerm term="pizza" polarity="positive" from="4" to="9"/>
      <aspectTerm term="service" polarity="negative" from="24" to="31"/>
    </aspectTerms>
    <aspectCategories>
      <aspectCategory category="food" polarity="positive"/>
      <aspectCategory category="service" polarity="conflict"/>
    </aspectCategories>
  </sentence>
</sentences>
"""


def test_parse_semeval_2014() -> None:
    examples = parse_semeval_string(SEMEVAL_2014, domain="restaurants")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.domain == "restaurants"
    assert {t.term for t in ex.aspect_terms} == {"pizza", "service"}
    # 'conflict' category is dropped by default.
    assert [c.category for c in ex.aspect_categories] == ["food"]


def test_parse_semeval_2016_opinions() -> None:
    xml = (
        '<sentences><sentence id="2"><text>Great sushi here</text>'
        '<Opinions><Opinion target="sushi" category="FOOD#QUALITY" '
        'polarity="positive" from="6" to="11"/></Opinions></sentence></sentences>'
    )
    examples = parse_semeval_string(xml)
    assert examples[0].aspect_terms[0].term == "sushi"
    assert examples[0].aspect_categories[0].category == "FOOD#QUALITY"


def test_parser_drops_misaligned_terms() -> None:
    xml = (
        '<sentences><sentence id="3"><text>The pizza was good</text>'
        '<aspectTerms><aspectTerm term="sushi" polarity="positive" from="4" to="9"/>'
        "</aspectTerms></sentence></sentences>"
    )
    # 'sushi' span points at 'pizza' -> dropped, leaving a valid example.
    examples = parse_semeval_string(xml)
    assert examples[0].aspect_terms == []


def test_parse_semeval_xml_file(tmp_path: Path) -> None:
    p = tmp_path / "s.xml"
    p.write_text(SEMEVAL_2014, encoding="utf-8")
    examples = parse_semeval_xml(p, domain="restaurants")
    assert len(examples) == 1
    assert {t.term for t in examples[0].aspect_terms} == {"pizza", "service"}


# io
def test_jsonl_roundtrip(tmp_path: Path) -> None:
    examples = [
        ABSAExample(
            id="a",
            text="Good food",
            domain="restaurants",
            aspect_terms=[AspectTerm(term="food", polarity="positive", start=5, end=9)],
            aspect_categories=[AspectCategory(category="food", polarity="positive")],
        )
    ]
    p = tmp_path / "x.jsonl"
    assert write_jsonl(examples, p) == 1
    assert read_jsonl(p) == examples


def test_read_jsonl_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "nope.jsonl")


# committed sample
def test_sample_loads_and_aligns() -> None:
    examples = load_sample()
    assert len(examples) >= 30
    for ex in examples:
        assert ex.domain in {"restaurants", "laptops"}
        for t in ex.aspect_terms:
            assert t.aligns_with(ex.text)
            assert t.polarity in {"positive", "negative", "neutral"}


# splits
def test_split_ratios_validation() -> None:
    with pytest.raises(ValueError):
        SplitRatios(train=0.5, val=0.4, test=0.4)


def test_make_splits_deterministic_and_complete() -> None:
    examples = load_sample()
    a = make_splits(examples, seed=42)
    b = make_splits(examples, seed=42)
    for key in ("train", "val", "test"):
        assert [e.id for e in a[key]] == [e.id for e in b[key]]

    all_ids = {e.id for e in examples}
    split_ids = {e.id for split in a.values() for e in split}
    assert split_ids == all_ids  # nothing lost
    total = sum(len(v) for v in a.values())
    assert total == len(examples)  # nothing duplicated


def test_make_splits_different_seed_differs() -> None:
    examples = load_sample()
    a = make_splits(examples, seed=1)
    b = make_splits(examples, seed=2)
    assert [e.id for e in a["train"]] != [e.id for e in b["train"]]


# stats
def test_label_distribution() -> None:
    dist = label_distribution(load_sample())
    assert dist["num_examples"] >= 30
    assert set(dist["term_polarity"]).issubset({"positive", "negative", "neutral"})  # type: ignore[arg-type]
    assert dist["num_aspect_terms"] > 0

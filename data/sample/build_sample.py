"""Build the committed tiny sample set (`data/sample/reviews.jsonl`).

Hand-written, SYNTHETIC restaurant + laptop reviews (not real SemEval data) so
tests and the demo run fully offline. Character spans are computed here (via a
case-insensitive search that stores the exact surface form) so they always
align with the text — never count offsets by hand.

Run:  python data/sample/build_sample.py
"""

from __future__ import annotations

from absa.data.io import SAMPLE_PATH, write_jsonl
from absa.data.schema import ABSAExample, AspectCategory, AspectTerm, Domain, Polarity

# (text, [(term, polarity), ...], [(category, polarity), ...])
Raw = tuple[str, list[tuple[str, Polarity]], list[tuple[str, Polarity]]]

RESTAURANTS: list[Raw] = [
    (
        "The pizza was delicious but the service was painfully slow.",
        [("pizza", "positive"), ("service", "negative")],
        [("food", "positive"), ("service", "negative")],
    ),
    (
        "Great ambience and friendly staff, though a bit pricey.",
        [("ambience", "positive"), ("staff", "positive")],
        [("ambience", "positive"), ("service", "positive"), ("price", "negative")],
    ),
    (
        "The sushi is fresh and the drinks are cheap.",
        [("sushi", "positive"), ("drinks", "positive")],
        [("food", "positive"), ("price", "positive")],
    ),
    (
        "Waited an hour for a cold burger.",
        [("burger", "negative")],
        [("food", "negative"), ("service", "negative")],
    ),
    (
        "Lovely decor but the food was bland.",
        [("decor", "positive"), ("food", "negative")],
        [("ambience", "positive"), ("food", "negative")],
    ),
    (
        "The waiter was attentive and the desserts were amazing.",
        [("waiter", "positive"), ("desserts", "positive")],
        [("service", "positive"), ("food", "positive")],
    ),
    (
        "Overpriced and the portions are tiny.",
        [("portions", "negative")],
        [("price", "negative"), ("food", "negative")],
    ),
    ("The coffee is average, nothing special.", [("coffee", "neutral")], [("food", "neutral")]),
    (
        "Reservations were easy and the location is convenient.",
        [("location", "positive")],
        [("misc", "positive"), ("service", "positive")],
    ),
    (
        "The pasta was undercooked and the sauce too salty.",
        [("pasta", "negative"), ("sauce", "negative")],
        [("food", "negative")],
    ),
    (
        "Cozy atmosphere with reasonable prices.",
        [("atmosphere", "positive"), ("prices", "positive")],
        [("ambience", "positive"), ("price", "positive")],
    ),
    ("The steak was cooked perfectly.", [("steak", "positive")], [("food", "positive")]),
    ("Rude host and a long wait to be seated.", [("host", "negative")], [("service", "negative")]),
    (
        "The menu has plenty of vegetarian options.",
        [("menu", "neutral")],
        [("food", "neutral"), ("misc", "neutral")],
    ),
    (
        "Good food but the tables are cramped.",
        [("food", "positive"), ("tables", "negative")],
        [("food", "positive"), ("ambience", "negative")],
    ),
    ("The bill came quickly and was accurate.", [("bill", "positive")], [("service", "positive")]),
    (
        "Their wine selection is impressive.",
        [("wine selection", "positive")],
        [("food", "positive")],
    ),
    (
        "Not worth the price for such small servings.",
        [("servings", "negative")],
        [("price", "negative"), ("food", "negative")],
    ),
]

LAPTOPS: list[Raw] = [
    (
        "The battery life is excellent and lasts all day.",
        [("battery life", "positive")],
        [("battery", "positive")],
    ),
    ("The keyboard feels cheap and mushy.", [("keyboard", "negative")], [("design", "negative")]),
    (
        "Super fast boot time and responsive apps.",
        [("boot time", "positive")],
        [("performance", "positive")],
    ),
    (
        "The screen is bright but the resolution is disappointing.",
        [("screen", "positive"), ("resolution", "negative")],
        [("display", "positive")],
    ),
    (
        "It runs hot and the fan is noisy.",
        [("fan", "negative")],
        [("performance", "negative"), ("design", "negative")],
    ),
    ("The trackpad is smooth and accurate.", [("trackpad", "positive")], [("design", "positive")]),
    (
        "Customer support was unhelpful and slow.",
        [("Customer support", "negative")],
        [("support", "negative")],
    ),
    ("Great value for the price.", [("price", "positive")], [("price", "positive")]),
    (
        "The build quality feels premium and sturdy.",
        [("build quality", "positive")],
        [("design", "positive")],
    ),
    ("The display has poor viewing angles.", [("display", "negative")], [("display", "negative")]),
    (
        "Storage is decent but the RAM is limited.",
        [("Storage", "neutral"), ("RAM", "negative")],
        [("misc", "neutral"), ("performance", "negative")],
    ),
    (
        "The laptop is lightweight and easy to carry.",
        [("laptop", "positive")],
        [("design", "positive"), ("misc", "positive")],
    ),
    (
        "Frequent crashes make it frustrating to use.",
        [("crashes", "negative")],
        [("performance", "negative")],
    ),
    ("The webcam quality is mediocre.", [("webcam", "neutral")], [("design", "neutral")]),
    ("Ports are plentiful and well placed.", [("Ports", "positive")], [("design", "positive")]),
    (
        "The charger is bulky but charges fast.",
        [("charger", "negative")],
        [("design", "negative"), ("performance", "positive")],
    ),
    (
        "Software updates install without issues.",
        [("Software updates", "neutral")],
        [("support", "neutral")],
    ),
]


def _find_span(text: str, term: str) -> tuple[int, int, str]:
    """Locate ``term`` in ``text`` (case-insensitive); return span + surface form."""
    idx = text.find(term)
    if idx < 0:
        idx = text.lower().find(term.lower())
    if idx < 0:
        raise ValueError(f"Term {term!r} not found in text {text!r}")
    return idx, idx + len(term), text[idx : idx + len(term)]


def _build_group(rows: list[Raw], domain: Domain, prefix: str) -> list[ABSAExample]:
    examples: list[ABSAExample] = []
    for i, (text, terms, cats) in enumerate(rows, start=1):
        aspect_terms = []
        for term, polarity in terms:
            start, end, surface = _find_span(text, term)
            aspect_terms.append(AspectTerm(term=surface, polarity=polarity, start=start, end=end))
        aspect_categories = [AspectCategory(category=c, polarity=p) for c, p in cats]
        examples.append(
            ABSAExample(
                id=f"{prefix}-{i:03d}",
                text=text,
                domain=domain,
                aspect_terms=aspect_terms,
                aspect_categories=aspect_categories,
            )
        )
    return examples


def build() -> list[ABSAExample]:
    """Return all sample examples (restaurants + laptops)."""
    return _build_group(RESTAURANTS, "restaurants", "r") + _build_group(LAPTOPS, "laptops", "l")


def main() -> None:
    examples = build()
    n = write_jsonl(examples, SAMPLE_PATH)
    print(f"Wrote {n} sample examples to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()

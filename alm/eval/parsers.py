"""Extract numeric values and MCQ letters from generated text.

Both extractors return None on failure; never raise. Callers count failures into
a `validity_rate` field, matching LLM4Mat-Bench's reporting convention.

Leak signatures (markdown image embeds, URLs) are scrubbed BEFORE matching, so a
digit hidden inside `i.imgur.com/6Z7Z7Z7.png` or `materialsproject.org/materials/101112`
won't be falsely parsed as a numeric prediction. `detect_leak` exposes the same
signature check so callers can count leak rate alongside parse-fail rate.
"""
import re

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_PAREN_LETTER = re.compile(r"\(([A-Za-z])\)")
_ANSWER_LETTER = re.compile(r"answer\s*(?:is|:)?\s*\(?([A-Za-z])\)?", re.I)
_DOTTED_LETTER = re.compile(r"\b([A-Za-z])\.")
_LONE_LETTER = re.compile(r"(?:^|\n|\s)([A-Za-z])(?:\b|$)")
_NUM_SANITY = 1e6   # materials properties don't legitimately exceed this

# Leak signatures — Qwen3-base pretraining priors that the model falls back on
# when uncertain. We strip these from the input before extraction (so a digit
# inside a URL hash is never returned as a number) and report them via
# detect_leak so callers can split parse-fail vs leak-fail in metrics.
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_NULL_LITERAL_RE = re.compile(r"^\s*(nan|null|none|n/a)\s*(eV|GPa|Å|[a-zA-Z/%]*)?\s*$", re.I)


def _scrub(text):
    """Strip markdown image blocks and URLs so digits/letters inside them
    aren't extracted as predictions."""
    text = _MD_IMG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return text


def detect_leak(text):
    """True iff the output emits the Qwen3-base fallback patterns
    (markdown image embed or any http(s) URL). These are wrong-format outputs
    regardless of whether a number happens to also be present."""
    if not text:
        return False
    if _MD_IMG_RE.search(text):
        return True
    if _URL_RE.search(text):
        return True
    return False


def extract_number(text):
    """First signed float in text (after URL/markdown scrub), with a sanity cap.
    Returns None on null literals (`nan`, `null`, `none`, `n/a`) and on
    pathologically large numbers so they don't poison MAE/RMSE."""
    if text is None:
        return None
    if _NULL_LITERAL_RE.match(text):
        return None
    text = _scrub(text)
    m = _NUM_RE.search(text)
    if not m:
        return None
    v = float(m.group(0))
    if not (v == v) or v == float("inf") or v == float("-inf") or abs(v) > _NUM_SANITY:
        return None
    return v


def extract_choice(text, choices=("A", "B", "C", "D")):
    if not text:
        return None
    text = _scrub(text)
    upper_choices = {c.upper() for c in choices}
    for pat in (_PAREN_LETTER, _ANSWER_LETTER, _DOTTED_LETTER):
        m = pat.search(text)
        if m:
            c = m.group(1).upper()
            if c in upper_choices:
                return c
    last = None
    for m in _LONE_LETTER.finditer(text):
        c = m.group(1).upper()
        if c in upper_choices:
            last = c
    return last


if __name__ == "__main__":
    cases_num = [
        ("the formation energy is -1.234 eV/atom", -1.234),
        ("approximately 2.5 GPa", 2.5),
        ("2.4e-3", 2.4e-3),
        ("no number here", None),
        ("", None),
        (None, None),
        # Leak cases — the digit inside imgur hash / MP id should NOT be returned.
        ("![](https://i.imgur.com/6Z7Z7Z7.png)", None),
        ("![](https://i.imgur.com/6YJ6Z7K.png)", None),
        ("![](https://www.materialsproject.org/materials/101112)", None),
        # Leaked-but-also-genuine: the number should still be extractable; the
        # caller is expected to drop the row via detect_leak instead of trusting
        # this number. Document the contract here.
        ("![](https://i.imgur.com/HASH.png) {\"density\": 4.6}", 4.6),
        ("![](https://www.materialsproject.org/materials/101112) {\"density\": 4.6}", 4.6),
        # Schema-corruption cases (the "!" prefix from arxiv-bucket bleed-through):
        ("!{\"k\": 1.5}", 1.5),
        ("!nan eV", None),
        ("nan eV", None),
        ("null", None),
        ("None", None),
    ]
    n_pass = 0
    for text, want in cases_num:
        got = extract_number(text)
        ok = (got is None and want is None) or (got is not None and want is not None and abs(got - want) < 1e-9)
        n_pass += ok
        print(f"{'PASS' if ok else 'FAIL'} extract_number({text!r}) = {got}  want {want}")

    cases_choice = [
        ("(B)", "B"),
        ("Answer: D.", "D"),
        ("Answer is C", "C"),
        ("the answer is c", "C"),
        ("A.", "A"),
        ("...so the answer is\n\nB", "B"),
        ("AB", None),
        ("", None),
        (None, None),
        # URL/markdown should not bleed letters from inside URLs into the choice.
        ("![](https://i.imgur.com/6Z7Z7Z7.png)", None),
        # genuine letter outside a leak block should still match.
        ("![](https://i.imgur.com/HASH.png)\nAnswer: B", "B"),
    ]
    for text, want in cases_choice:
        got = extract_choice(text)
        ok = got == want
        n_pass += ok
        print(f"{'PASS' if ok else 'FAIL'} extract_choice({text!r}) = {got}  want {want}")

    cases_leak = [
        ("![](https://i.imgur.com/6Z7Z7Z7.png)", True),
        ("![](https://www.materialsproject.org/materials/101112) {\"x\":1}", True),
        ("https://example.com/foo", True),
        ("plain old text with -1.234 eV", False),
        ("", False),
        (None, False),
        ("see https://arxiv.org/abs/1234.56789 for refs", True),  # any URL counts
    ]
    for text, want in cases_leak:
        got = detect_leak(text)
        ok = got == want
        n_pass += ok
        print(f"{'PASS' if ok else 'FAIL'} detect_leak({text!r}) = {got}  want {want}")

    total = len(cases_num) + len(cases_choice) + len(cases_leak)
    print(f"\n{n_pass}/{total} tests passed")
    if n_pass < total:
        raise SystemExit(1)

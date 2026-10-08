"""Amharic text primitives: syllables, homogenisation, comparison.

A leaf module with no imports from the rest of the pipeline, so both
:mod:`app.pipeline.qc` (measuring a run) and
:mod:`app.pipeline.dialogue_context` (budgeting a line) can use it without the two
ending up in an import cycle.

Why syllables, and not phonemes
-------------------------------
Amharic is written in the Ethiopic (Fidel) syllabary, where **one character is one
syllable**. That makes a syllable count available from the text alone - no
grapheme-to-phoneme model, no pronunciation lexicon, no forced aligner. That
matters because there is no standard Amharic G2P and no neural one, so a
phoneme-based design would have nothing to stand on; a syllable-based one works
today. It is the unit the timing budget and the delivery-rate measurement are both
expressed in.

Homophone families
------------------
The script carries several letters that are pronounced identically in modern
Amharic (ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ, ጸ/ፀ). A difference between two of them is an orthographic
choice, not a pronunciation error, so folding them before a comparison stops a
measurement from punishing spelling. Folding is applied at *comparison* time only:
the budget counts what is written, because that is what is synthesized.
"""

from __future__ import annotations

#: Unicode ranges that each encode exactly one syllable. The Ethiopic block's
#: punctuation (U+1360-U+1368, U+137D-U+137F), digits (U+1369-U+137C) and
#: combining marks (U+135D-U+135F) are excluded on purpose: they are characters,
#: but they are not things a performer pronounces as a syllable.
SYLLABLE_RANGES: tuple[tuple[int, int], ...] = ((0x1200, 0x135A), (0x1380, 0x139F))

#: Characters pronounced identically in modern Amharic, each family written with the
#: representative that folding keeps. The representative is not "more correct" than
#: the others - it only has to be consistent.
HOMOPHONE_FAMILIES: tuple[str, ...] = ("ሀሃሐሓኀኃ", "ሰሠ", "አዐ", "ጸፀ")


class InvalidTextError(ValueError):
    """An Amharic text primitive was handed something it cannot use."""


def is_syllable(character: str) -> bool:
    """``True`` when ``character`` is a single Ethiopic syllable character."""

    if not isinstance(character, str) or len(character) != 1:
        return False
    codepoint = ord(character)
    return any(low <= codepoint <= high for low, high in SYLLABLE_RANGES)


def count_syllables(text: str) -> int:
    """Return the number of syllables in ``text``.

    One Fidel character is one syllable, so this counts the characters inside
    :data:`SYLLABLE_RANGES`. Latin letters, digits, punctuation and the Ethiopic
    combining marks contribute nothing.
    """

    if not isinstance(text, str):
        raise InvalidTextError(f"syllable counting needs text, got {type(text).__name__}")

    return sum(1 for character in text if is_syllable(character))


def fold_homophones(text: str) -> str:
    """Return ``text`` with every homophone family replaced by its representative."""

    if not isinstance(text, str):
        raise InvalidTextError(f"homophone folding needs text, got {type(text).__name__}")

    folded = text
    for family in HOMOPHONE_FAMILIES:
        for character in family[1:]:
            folded = folded.replace(character, family[0])
    return folded


def syllable_sequence(text: str) -> str:
    """Return only the syllables of ``text``, homophones folded.

    This is what two strings are reduced to before they are compared, so a
    comparison measures pronunciation rather than spelling or formatting.
    """

    return "".join(character for character in fold_homophones(text) if is_syllable(character))


__all__ = [
    "HOMOPHONE_FAMILIES",
    "SYLLABLE_RANGES",
    "InvalidTextError",
    "count_syllables",
    "fold_homophones",
    "is_syllable",
    "syllable_sequence",
]

"""
text_cleaner.py

Cleans PDF-extracted text — removes weird symbols and normalizes whitespace.
"""

import re


# Map of garbage symbols → clean replacement
SYMBOL_MAP = {
    "\u00a7": " ",       # § → space
    "\u2116": "• ",      # № → bullet
    "\u2192": " → ",     # →
    "\u2666": "• ",      # ♦ → bullet
    "\u25a1": "[ ] ",    # □ → checkbox
    "\u25a0": "[■] ",    # ■ → filled checkbox
    "\u2713": "✓ ",      # ✓
    "\u2717": "✗ ",      # ✗
    "\u2022": "• ",      # •
    "\u2013": "–",       # en dash
    "\u2014": "—",       # em dash
    "\u2026": "...",     # ellipsis
    "\u00d7": "×",       # ×
    "\u00b7": "·",       # ·
    "\u200b": "",        # zero-width space
    "\u200c": "",        # zero-width non-joiner
    "\u200d": "",        # zero-width joiner
    "\ufeff": "",        # BOM
    "\u00a0": " ",       # non-breaking space
}


def clean_text(text: str) -> str:
    """Clean raw PDF text — remove junk symbols, normalize whitespace."""
    if not text:
        return ""

    # Replace known symbols
    for old, new in SYMBOL_MAP.items():
        text = text.replace(old, new)

    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)

    return text.strip()


def clean_answer(text: str) -> str:
    """More aggressive cleaning for final answer display."""
    if not text:
        return ""

    text = clean_text(text)

    # Fix joined words like "Owner§s" → "Owner's"
    text = re.sub(r"(\w)\u00a7(\w)", r"\1'\2", text)

    # Remove sequences like "II" separators → "\n• "
    text = re.sub(r"\s*II\s+", "\n• ", text)

    # Remove leftover multiple spaces
    text = re.sub(r" +", " ", text)

    # Fix broken punctuation
    text = text.replace(" .", ".").replace(" ,", ",")

    return text.strip()
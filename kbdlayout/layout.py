"""
Bidirectional character mapping between US English (QWERTY) and
Ukrainian (Enhanced) keyboard layouts.

If someone meant to type Ukrainian but had the English layout active, each
Ukrainian letter they intended becomes the Latin letter sitting on the same
physical key. `en_to_uk` and `uk_to_en` simulate that mistake on existing text
so we can generate "wrong-layout" training examples from clean corpora.
"""

_LETTERS = {
    'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г',
    'i': 'ш', 'o': 'щ', 'p': 'з',
    'a': 'ф', 's': 'і', 'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о',
    'k': 'л', 'l': 'д',
    'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и', 'n': 'т', 'm': 'ь',
}

# Non-letter keys whose character differs between layouts.
_PUNCT = {
    # Unshifted
    '[': 'х', ']': 'ї',
    ';': 'ж', "'": 'є',
    ',': 'б', '.': 'ю', '/': '.',
    '`': "'",
    # Shifted (Shift+<key> on US ↔ Shift+<key> on UK Enhanced)
    '@': '"', '#': '№', '$': ';', '^': ':',
    '&': '?', '?': ',',
}

_EN_TO_UK_LETTERS: dict[str, str] = {}
for _en, _uk in _LETTERS.items():
    _EN_TO_UK_LETTERS[_en] = _uk
    _EN_TO_UK_LETTERS[_en.upper()] = _uk.upper()

_EN_TO_UK: dict[str, str] = {**_EN_TO_UK_LETTERS, **_PUNCT}
_UK_TO_EN: dict[str, str] = {v: k for k, v in _EN_TO_UK.items()}
_UK_TO_EN_LETTERS: dict[str, str] = {v: k for k, v in _EN_TO_UK_LETTERS.items()}


def en_to_uk(text: str, letters_only: bool = False) -> str:
    """Convert text as if EN-layout keys were pressed but UK layout was active.

    `letters_only=True` skips punctuation keys that share characters between
    layouts (",", ".", ";", "[", "]", etc.), which gives nicer conversions for
    real messages where the user only mistyped the letters.
    """
    table = _EN_TO_UK_LETTERS if letters_only else _EN_TO_UK
    return "".join(table.get(c, c) for c in text)


def uk_to_en(text: str, letters_only: bool = False) -> str:
    """Convert text as if UK-layout keys were pressed but EN layout was active."""
    table = _UK_TO_EN_LETTERS if letters_only else _UK_TO_EN
    return "".join(table.get(c, c) for c in text)


if __name__ == "__main__":
    print("UK text typed on EN layout (uk_en):")
    for s in ["Привіт, як справи?", "об'єкт", "Дякую!"]:
        print(f"  {s!r} -> {uk_to_en(s)!r}")
    print("EN text typed on UK layout (en_uk):")
    for s in ["Hello, how are you?", "don't worry", "Thanks!"]:
        print(f"  {s!r} -> {en_to_uk(s)!r}")

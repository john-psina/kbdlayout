"""kbdlayout — keyboard layout detector and corrector.

Detects whether text was typed with the wrong keyboard layout
(Ukrainian ↔ English) and optionally converts it to the intended text.

Typical usage:
    from kbdlayout import LayoutPredictor

    p = LayoutPredictor()
    print(p.convert("Ghbdsn, zr cghfdb?"))   # → "Привіт, як справи?"
    print(p.advise("Руддщб, рщц фку нщг?"))  # → {'wrong_layout': True, ...}
"""

from .layout import en_to_uk, uk_to_en
from .predict import LayoutPredictor

__all__ = ["LayoutPredictor", "en_to_uk", "uk_to_en"]
__version__ = "0.1.0"

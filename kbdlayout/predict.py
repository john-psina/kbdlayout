"""Inference helper for the trained layout classifier."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .layout import en_to_uk, uk_to_en
from .model import LABELS, CharVocab, LayoutClassifier

ROOT = Path(__file__).parent
DEFAULT_CKPT = ROOT / "checkpoints" / "best.pt"

MIN_TEXT_LEN = 4


class LayoutPredictor:
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.vocab = CharVocab.from_dict(ckpt["vocab"])
        self.max_len: int = ckpt["max_len"]
        self.model = LayoutClassifier(vocab_size=len(self.vocab)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def probs(self, text: str) -> dict[str, float]:
        if len(text) < MIN_TEXT_LEN:
            return {l: 0.25 for l in LABELS}
        ids = torch.tensor(
            [self.vocab.encode(text, self.max_len)], device=self.device
        )
        logits = self.model(ids)
        ps = F.softmax(logits, dim=1).squeeze(0).tolist()
        return {l: p for l, p in zip(LABELS, ps)}

    def advise(self, text: str) -> dict:
        ps = self.probs(text)
        label = max(ps, key=lambda k: ps[k])
        intent_lang, actual_layout = label.split("_")
        wrong = intent_lang != actual_layout
        other = "en" if actual_layout == "uk" else "uk"
        return {
            "wrong_layout": wrong,
            "text_layout": actual_layout,
            "need_layout": intent_lang if wrong else other,
            "confidence": round(ps[label], 4),
        }

    def advise_per_word(self, text: str) -> list[dict]:
        """For mixed-layout messages: classify each whitespace token separately."""
        out = []
        for w in text.split():
            stripped = w.strip(".,!?;:'\"()[]{}")
            if len(stripped) < MIN_TEXT_LEN:
                out.append({"text": w, "advice": None})
            else:
                out.append({"text": w, "advice": self.advise(stripped)})
        return out

    def convert(self, text: str, letters_only: bool = True, convert_urls: bool = True) -> str:
        """If text appears wrong-layout, return its layout-corrected form.

        `letters_only=True` (default) preserves punctuation so that
        commas/periods stay as commas/periods.  Pass `letters_only=False` to
        also remap punctuation keys that differ between layouts.

        `convert_urls=True` (default) leaves URLs in the text untouched.
        """
        advice = self.advise(text)
        if not advice["wrong_layout"]:
            return text
        if advice["need_layout"] == "uk":
            return en_to_uk(text, letters_only=letters_only, convert_urls=convert_urls)
        return uk_to_en(text, letters_only=letters_only, convert_urls=convert_urls)


def _fmt_probs(ps: dict[str, float]) -> str:
    return ", ".join(f"{k}={v*100:5.2f}%" for k, v in ps.items())



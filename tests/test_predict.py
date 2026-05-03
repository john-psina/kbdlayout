"""Tests for LayoutPredictor — requires a trained checkpoint.

All tests in this file are skipped automatically if the checkpoint file
kbdlayout/checkpoints/best.pt is not present.
"""

import pytest
from pathlib import Path

CKPT = Path(__file__).parent.parent / "kbdlayout" / "checkpoints" / "best.pt"

requires_checkpoint = pytest.mark.skipif(
    not CKPT.exists(),
    reason="trained checkpoint not found at kbdlayout/checkpoints/best.pt",
)


@requires_checkpoint
class TestAdvise:
    @pytest.fixture(scope="class")
    def predictor(self):
        from kbdlayout.predict import LayoutPredictor
        return LayoutPredictor()

    def test_returns_expected_keys(self, predictor):
        result = predictor.advise("Ghbdsn, zr cghfdb?")
        assert set(result) == {"wrong_layout", "text_layout", "need_layout", "confidence"}

    def test_wrong_layout_is_bool(self, predictor):
        result = predictor.advise("Ghbdsn, zr cghfdb?")
        assert isinstance(result["wrong_layout"], bool)

    def test_text_layout_valid_value(self, predictor):
        result = predictor.advise("Ghbdsn, zr cghfdb?")
        assert result["text_layout"] in ("en", "uk")

    def test_confidence_in_range(self, predictor):
        result = predictor.advise("Ghbdsn, zr cghfdb?")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_correct_layout_not_flagged(self, predictor):
        result = predictor.advise("Hello, how are you?")
        assert result["wrong_layout"] is False

    def test_wrong_layout_detected(self, predictor):
        # "Ghbdsn" is Ukrainian typed on EN layout
        result = predictor.advise("Ghbdsn, zr cghfdb?")
        assert result["wrong_layout"] is True


@requires_checkpoint
class TestConvert:
    @pytest.fixture(scope="class")
    def predictor(self):
        from kbdlayout.predict import LayoutPredictor
        return LayoutPredictor()

    def test_returns_string(self, predictor):
        assert isinstance(predictor.convert("Ghbdsn, zr cghfdb?"), str)

    def test_correct_layout_unchanged(self, predictor):
        text = "Hello, how are you?"
        assert predictor.convert(text) == text

    def test_accepts_convert_urls_param(self, predictor):
        predictor.convert("Hello world", convert_urls=True)
        predictor.convert("Hello world", convert_urls=False)

    def test_url_preserved_during_conversion(self, predictor):
        text = "Ghbdsn https://example.com zr cghfdb?"
        result = predictor.convert(text, convert_urls=True)
        assert "https://example.com" in result

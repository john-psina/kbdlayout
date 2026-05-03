"""Tests for kbdlayout.layout — character mapping and URL preservation."""

import pytest
from kbdlayout.layout import en_to_uk, uk_to_en


class TestEnToUk:
    def test_basic_letters(self):
        assert en_to_uk("qwerty") == "йцукен"

    def test_uppercase(self):
        assert en_to_uk("Q") == "Й"
        assert en_to_uk("Hello") == "Руддщ"

    def test_known_phrase(self):
        assert en_to_uk("Ghbdsn") == "Привіт"

    def test_unmapped_chars_pass_through(self):
        assert en_to_uk("123 !") == "123 !"

    def test_empty_string(self):
        assert en_to_uk("") == ""

    def test_letters_only_false_converts_punctuation(self):
        assert en_to_uk(",", letters_only=False) == "б"
        assert en_to_uk(".", letters_only=False) == "ю"

    def test_letters_only_true_preserves_punctuation(self):
        assert en_to_uk(",", letters_only=True) == ","
        assert en_to_uk(".", letters_only=True) == "."

    # --- convert_urls ---

    def test_url_preserved_by_default(self):
        result = en_to_uk("hello https://example.com world")
        assert "https://example.com" in result

    def test_non_url_parts_are_converted(self):
        result = en_to_uk("hello https://example.com world")
        assert result.startswith(en_to_uk("hello"))
        assert result.endswith(en_to_uk("world"))

    def test_url_at_start(self):
        result = en_to_uk("https://example.com hello")
        assert result.startswith("https://example.com")

    def test_url_at_end(self):
        result = en_to_uk("hello https://example.com")
        assert result.endswith("https://example.com")

    def test_multiple_urls_preserved(self):
        result = en_to_uk("a https://one.com b https://two.org c")
        assert "https://one.com" in result
        assert "https://two.org" in result

    def test_text_without_url_unchanged_behavior(self):
        text = "hello world"
        assert en_to_uk(text, convert_urls=True) == en_to_uk(text, convert_urls=False)

    def test_convert_urls_false_converts_url_too(self):
        url = "https://example.com"
        assert en_to_uk(url, convert_urls=True) == url
        assert en_to_uk(url, convert_urls=False) != url


class TestUkToEn:
    def test_basic_letters(self):
        assert uk_to_en("йц") == "qw"

    def test_uppercase(self):
        assert uk_to_en("Й") == "Q"

    def test_unmapped_chars_pass_through(self):
        assert uk_to_en("123 !") == "123 !"

    def test_empty_string(self):
        assert uk_to_en("") == ""

    def test_letters_only_false_converts_punctuation(self):
        assert uk_to_en("б", letters_only=False) == ","

    def test_letters_only_true_preserves_punctuation(self):
        assert uk_to_en("б", letters_only=True) == "б"

    def test_url_preserved_by_default(self):
        result = uk_to_en("Привіт https://example.com світ")
        assert "https://example.com" in result

    def test_convert_urls_false_converts_url_too(self):
        url = "https://example.com"
        assert uk_to_en(url, convert_urls=True) == url
        assert uk_to_en(url, convert_urls=False) != url


class TestRoundTrip:
    """en_to_uk → uk_to_en має повертати початковий текст (для літер)."""

    @pytest.mark.parametrize("text", [
        "hello",
        "Hello World",
        "qwerty",
        "QWERTY",
        "abcdefghijklmnopqrstuvwxyz",
    ])
    def test_letters_roundtrip(self, text: str):
        assert uk_to_en(en_to_uk(text, letters_only=True), letters_only=True) == text

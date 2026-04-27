"""Command-line interface for the keyboard-layout detector.

Examples:
    python cli.py "Ghbdsn, zr cghfdb?"
    python cli.py --convert "Ghbdsn, zr cghfdb?"
    python cli.py --per-word "D nt,t ' Steam?"
    python cli.py --json "some text"
    echo "line1\nline2" | python cli.py
    python cli.py                       # interactive
"""

import argparse
import json
import sys
from pathlib import Path

from .layout import en_to_uk, uk_to_en
from .predict import LayoutPredictor


def _fix_one(text: str, advice: dict | None, full: bool = False) -> str:
    """Convert if wrong-layout. `full=True` also converts in-token punctuation,
    which is the right call when the whole token is one wrong-layout chunk
    (per-word mode); `full=False` (default) preserves punctuation."""
    if not advice or not advice["wrong_layout"]:
        return text
    if advice["need_layout"] == "uk":
        return en_to_uk(text, letters_only=not full)
    return uk_to_en(text, letters_only=not full)


def _fmt_human(text: str, advice: dict | None) -> str:
    if advice is None:
        return f"  {text!r}: <too short>"
    flag = "WRONG" if advice["wrong_layout"] else "ok"
    return (
        f"  {text!r}: [{flag}] "
        f"shown={advice['text_layout']} "
        f"should={advice['need_layout']} "
        f"({advice['confidence']*100:.1f}%)"
    )


def _process(predictor: LayoutPredictor, text: str, args) -> None:
    text = text.strip()
    if not text:
        return

    if args.per_word:
        items = predictor.advise_per_word(text)
        if args.json:
            print(json.dumps(items, ensure_ascii=False))
            return
        print(text)
        for item in items:
            print(_fmt_human(item["text"], item["advice"]))
        if args.convert:
            fixed = " ".join(_fix_one(it["text"], it["advice"], full=True) for it in items)
            print(f"  -> {fixed}")
        return

    advice = predictor.advise(text)
    if args.json:
        out = {"text": text, **advice}
        if args.convert:
            out["fixed"] = _fix_one(text, advice)
        print(json.dumps(out, ensure_ascii=False))
        return

    print(_fmt_human(text, advice))
    if args.convert and advice["wrong_layout"]:
        print(f"  -> {_fix_one(text, advice)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Detect & fix wrong keyboard layout.")
    parser.add_argument("text", nargs="?", help="text to analyze (omit for stdin / interactive)")
    parser.add_argument("--per-word", action="store_true",
                        help="classify each whitespace token (for mixed-layout messages)")
    parser.add_argument("--convert", action="store_true",
                        help="also print the layout-corrected version")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument("--ckpt", type=Path, default=None, help="path to model checkpoint")
    args = parser.parse_args()

    predictor = LayoutPredictor(ckpt_path=args.ckpt) if args.ckpt else LayoutPredictor()

    if args.text:
        _process(predictor, args.text, args)
        return

    if not sys.stdin.isatty():
        for line in sys.stdin:
            _process(predictor, line, args)
        return

    print("Interactive mode (Ctrl+C to exit).", file=sys.stderr)
    try:
        while True:
            line = input("> ")
            _process(predictor, line, args)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()

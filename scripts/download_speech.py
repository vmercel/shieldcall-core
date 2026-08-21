#!/usr/bin/env python3
"""Download Mini LibriSpeech (OpenSLR 31, dev-clean-2) into ./data."""

from __future__ import annotations

from shieldcall.eval.speech_data import download_mini_librispeech, list_speakers, speech_available


def main() -> None:
    root = download_mini_librispeech()
    print(f"speech root: {root}")
    print(f"available: {speech_available()}")
    print(f"speakers: {len(list_speakers())}")


if __name__ == "__main__":
    main()

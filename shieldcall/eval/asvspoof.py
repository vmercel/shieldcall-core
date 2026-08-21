"""Optional ASVspoof Logical Access loader.

This project does not redistribute ASVspoof. Set ``SHIELDCALL_ASVSPOOF_ROOT``
to a local copy (the directory that contains ``LA/`` or the protocol files)
and ``load_asvspoof_la()`` will yield evaluation samples.

If the corpus is absent, callers must skip the protocol rather than
fabricate numbers.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterator, List, Optional

from .harness import EvalSample
from .speech_data import load_audio, DEFAULT_TARGET_SR


PROTOCOL_CANDIDATES = (
    "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    "ASVspoof2019.LA.cm.eval.trl.txt",
)

FLAC_CANDIDATES = (
    "ASVspoof2019_LA_eval/flac",
    "LA/ASVspoof2019_LA_eval/flac",
    "flac",
)


def asvspoof_root() -> Optional[Path]:
    env = os.environ.get("SHIELDCALL_ASVSPOOF_ROOT")
    if not env:
        return None
    p = Path(env)
    return p if p.exists() else None


def available() -> bool:
    root = asvspoof_root()
    if root is None:
        return False
    return any((root / c).exists() for c in PROTOCOL_CANDIDATES)


def _find_protocol(root: Path) -> Path:
    for c in PROTOCOL_CANDIDATES:
        p = root / c
        if p.exists():
            return p
    raise FileNotFoundError(f"No ASVspoof LA protocol file under {root}")


def _find_flac_dir(root: Path) -> Path:
    for c in FLAC_CANDIDATES:
        p = root / c
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No ASVspoof flac directory under {root}")


def iter_asvspoof_la(
    subset: str = "eval",
    max_bona: int = 200,
    max_spoof: int = 200,
    target_sr: int = DEFAULT_TARGET_SR,
) -> Iterator[EvalSample]:
    root = asvspoof_root()
    if root is None:
        raise FileNotFoundError("SHIELDCALL_ASVSPOOF_ROOT is not set")
    protocol = _find_protocol(root)
    flac_dir = _find_flac_dir(root)
    n_bona = n_spoof = 0
    for line in protocol.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # speaker, utt, system, -, key
        utt_id = parts[1]
        key = parts[-1].lower()
        is_spoof = key in {"spoof", "fake"}
        if not is_spoof and n_bona >= max_bona:
            continue
        if is_spoof and n_spoof >= max_spoof:
            continue
        wav = flac_dir / f"{utt_id}.flac"
        if not wav.exists():
            wav = flac_dir / f"{utt_id}.wav"
        if not wav.exists():
            continue
        audio, sr = load_audio(wav, target_sr=target_sr)
        yield EvalSample(
            audio=audio,
            sample_rate=sr,
            is_synthetic=is_spoof,
            transcript="",
            condition=subset,
            family="spoof" if is_spoof else "bona_fide",
        )
        if is_spoof:
            n_spoof += 1
        else:
            n_bona += 1
        if n_bona >= max_bona and n_spoof >= max_spoof:
            break


def load_asvspoof_la(**kwargs) -> List[EvalSample]:
    return list(iter_asvspoof_la(**kwargs))

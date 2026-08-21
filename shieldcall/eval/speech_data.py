"""Load real speech for evaluation.

Default corpus: OpenSLR Mini LibriSpeech ``dev-clean-2`` (Panayotov et al.).
This is bona fide read speech, not telephone audio. We downsample to 8 kHz
and optionally pass it through the telephone channel simulator.

ASVspoof is supported separately in ``asvspoof.py`` when the user has
licensed that corpus. This module never silently substitutes sine-wave
proxies for missing speech: callers must handle the missing-data case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import tarfile
import urllib.request
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


MINI_LIBRISPEECH_URLS = (
    "http://www.openslr.org/resources/31/dev-clean-2.tar.gz",
    "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
)

DEFAULT_TARGET_SR = 8000


@dataclass(frozen=True)
class SpeechClip:
    """One bona fide utterance with speaker metadata."""

    path: Path
    speaker_id: str
    chapter_id: str
    utt_id: str
    audio: np.ndarray
    sample_rate: int
    transcript: str = ""


def repo_data_root() -> Path:
    env = os.environ.get("SHIELDCALL_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def librispeech_root(data_root: Optional[Path] = None) -> Path:
    root = data_root or repo_data_root()
    # tarball extracts to data/LibriSpeech/dev-clean-2/...
    candidate = root / "LibriSpeech" / "dev-clean-2"
    if candidate.is_dir():
        return candidate
    nested = root / "dev-clean-2"
    return nested


def _resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32)
    g = math.gcd(int(orig_sr), int(target_sr))
    y = resample_poly(x.astype(np.float64), target_sr // g, orig_sr // g)
    peak = np.max(np.abs(y)) + 1e-8
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def load_audio(path: Path, target_sr: int = DEFAULT_TARGET_SR) -> Tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile is required to load FLAC/WAV. pip install soundfile")
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = audio.astype(np.float32)
    audio = _resample(audio, int(sr), target_sr)
    return audio, target_sr


def _parse_transcripts(chapter_dir: Path) -> dict[str, str]:
    trans: dict[str, str] = {}
    for txt in chapter_dir.glob("*.trans.txt"):
        for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                trans[parts[0]] = parts[1].lower()
    return trans


def iter_librispeech_paths(root: Optional[Path] = None) -> Iterable[Tuple[Path, str, str, str, str]]:
    if root is None:
        base = librispeech_root()
    elif (root / "LibriSpeech" / "dev-clean-2").is_dir():
        base = root / "LibriSpeech" / "dev-clean-2"
    elif (root / "dev-clean-2").is_dir():
        base = root / "dev-clean-2"
    else:
        base = root
    if not base.is_dir():
        return
    for speaker_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        speaker_id = speaker_dir.name
        for chapter_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
            trans = _parse_transcripts(chapter_dir)
            for flac in sorted(chapter_dir.glob("*.flac")):
                utt_id = flac.stem
                yield flac, speaker_id, chapter_dir.name, utt_id, trans.get(utt_id, "")


def list_speakers(root: Optional[Path] = None) -> List[str]:
    speakers = []
    seen = set()
    for _, spk, *_ in iter_librispeech_paths(root):
        if spk not in seen:
            seen.add(spk)
            speakers.append(spk)
    return speakers


def speech_available(root: Optional[Path] = None) -> bool:
    try:
        return next(iter(iter_librispeech_paths(root)), None) is not None
    except Exception:
        return False


def load_speaker_disjoint(
    n_train_speakers: int = 8,
    n_test_speakers: int = 8,
    utt_per_speaker: int = 6,
    min_seconds: float = 1.5,
    max_seconds: float = 8.0,
    target_sr: int = DEFAULT_TARGET_SR,
    seed: int = 0,
    root: Optional[Path] = None,
) -> Tuple[List[SpeechClip], List[SpeechClip]]:
    """Load bona fide clips with a speaker-disjoint train/test split.

    Raises FileNotFoundError if Mini LibriSpeech is not on disk.
    """
    speakers = list_speakers(root)
    if len(speakers) < n_train_speakers + n_test_speakers:
        raise FileNotFoundError(
            f"Need at least {n_train_speakers + n_test_speakers} speakers; "
            f"found {len(speakers)}. Run scripts/download_speech.py"
        )
    rng = np.random.RandomState(seed)
    order = list(speakers)
    rng.shuffle(order)
    train_sp = set(order[:n_train_speakers])
    test_sp = set(order[n_train_speakers : n_train_speakers + n_test_speakers])

    by_speaker: dict[str, List[Tuple[Path, str, str, str]]] = {s: [] for s in train_sp | test_sp}
    for path, spk, chap, utt, transcript in iter_librispeech_paths(root):
        if spk in by_speaker:
            by_speaker[spk].append((path, chap, utt, transcript))

    def collect(spk_ids: Sequence[str]) -> List[SpeechClip]:
        clips: List[SpeechClip] = []
        for spk in sorted(spk_ids):
            items = list(by_speaker[spk])
            rng.shuffle(items)
            taken = 0
            for path, chap, utt, transcript in items:
                audio, sr = load_audio(path, target_sr=target_sr)
                dur = len(audio) / float(sr)
                if dur < min_seconds:
                    continue
                if dur > max_seconds:
                    audio = audio[: int(max_seconds * sr)]
                clips.append(
                    SpeechClip(
                        path=path,
                        speaker_id=spk,
                        chapter_id=chap,
                        utt_id=utt,
                        audio=audio,
                        sample_rate=sr,
                        transcript=transcript,
                    )
                )
                taken += 1
                if taken >= utt_per_speaker:
                    break
        return clips

    return collect(sorted(train_sp)), collect(sorted(test_sp))


def download_mini_librispeech(data_root: Optional[Path] = None, timeout: int = 120) -> Path:
    """Download OpenSLR mini LibriSpeech if missing. Returns extract root."""
    root = data_root or repo_data_root()
    root.mkdir(parents=True, exist_ok=True)
    if speech_available(root / "LibriSpeech" / "dev-clean-2"):
        return librispeech_root()
    tarball = root / "dev-clean-2.tar.gz"
    if not tarball.exists() or tarball.stat().st_size < 1_000_000:
        last_err: Optional[Exception] = None
        for url in MINI_LIBRISPEECH_URLS:
            try:
                urllib.request.urlretrieve(url, tarball)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if last_err is not None and not tarball.exists():
            raise RuntimeError(
                "Could not download Mini LibriSpeech. Run: "
                "curl -L -o data/dev-clean-2.tar.gz "
                "http://www.openslr.org/resources/31/dev-clean-2.tar.gz"
            ) from last_err
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(root)
    return librispeech_root()

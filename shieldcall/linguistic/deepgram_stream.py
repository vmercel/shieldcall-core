"""
Deepgram streaming ASR provider behind the ASRBridge interface.

Design notes (read before touching the threading):

- The telephone frame path calls ``push_audio`` once per 25 ms frame, up to
  ~100 times per second. It must never block on the network. All websocket
  I/O therefore lives on a single background worker thread; ``push_audio``
  only enqueues PCM bytes and drains already-arrived transcript fragments.
  Provider round-trip latency shows up as transcript delay, never as added
  frame latency. ``scripts/bench_asr_latency.py`` proves this.
- Hard provider failures (auth rejection, network loss) surface as an
  exception from ``push_audio`` so the ``GatedASR`` circuit breaker in
  ``shieldcall.runtime`` can trip and shed ASR fail-open, exactly as it does
  for any other failing bridge.
- The transport is injectable: production uses the real Deepgram websocket,
  tests and the latency bench use ``LoopbackTransport`` with a configurable
  simulated RTT. No test or benchmark needs a Deepgram API key or network.

Audio format: 16-bit PCM mono at 16 kHz. Input at other sample rates is
resampled with linear interpolation (good enough for telephony speech;
the provider re-models it anyway).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from .asr_bridge import ASRBridge, TranscriptFragment

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"
_PROVIDER_SR = 16000
_CHUNK_MS = 100  # audio queued per push is forwarded in ~100 ms bites


class TransportError(RuntimeError):
    """The streaming transport is down (auth/network/protocol)."""


class StreamingTransport:
    """Minimal interface the adapter needs from a streaming provider."""

    def connect(self) -> None: ...
    def send_audio(self, pcm16: bytes) -> None: ...
    def recv(self, timeout: float) -> Optional[Dict]: ...
    def close(self) -> None: ...


class DeepgramWSTransport(StreamingTransport):
    """Real Deepgram ``/v1/listen`` websocket transport (lazy import)."""

    def __init__(self, api_key: str, model: str = "nova-2", language: str = "en"):
        if not api_key:
            raise TransportError("DEEPGRAM_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.language = language
        self._ws = None

    def connect(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise TransportError(
                "the 'websockets' package is required for Deepgram streaming; "
                "pip install -r requirements.txt"
            ) from exc
        url = (
            f"{DEEPGRAM_URL}?model={self.model}&language={self.language}"
            "&encoding=linear16&sample_rate=16000&channels=1"
            "&smart_format=true&interim_results=true&endpointing=300"
        )
        try:
            self._ws = connect(url, additional_headers={"Authorization": f"Token {self.api_key}"})
        except Exception as exc:
            raise TransportError(f"deepgram connect failed: {exc}") from exc

    def send_audio(self, pcm16: bytes) -> None:
        try:
            self._ws.send(pcm16)
        except Exception as exc:
            raise TransportError(f"deepgram send failed: {exc}") from exc

    def recv(self, timeout: float) -> Optional[Dict]:
        try:
            raw = self._ws.recv(timeout=timeout)
        except TimeoutError:
            return None
        except Exception as exc:
            raise TransportError(f"deepgram recv failed: {exc}") from exc
        if isinstance(raw, bytes):
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass


class LoopbackTransport(StreamingTransport):
    """
    Test/benchmark transport: no network. Every audio chunk pushed is
    "transcribed" after ``latency_ms`` as a canned final fragment, so the
    bench can measure frame-path overhead vs end-to-end transcript delay
    without an API key.
    """

    def __init__(self, latency_ms: float = 120.0, transcript: str = "hello this is a test"):
        self.latency_ms = latency_ms
        self.transcript = transcript
        self._due: List[tuple[float, Dict]] = []
        self._lock = threading.Lock()
        self.sent_bytes = 0
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def send_audio(self, pcm16: bytes) -> None:
        if not self.connected:
            raise TransportError("loopback not connected")
        due = time.monotonic() + self.latency_ms / 1000.0
        msg = {
            "type": "Results",
            "is_final": True,
            "channel": {"alternatives": [{"transcript": self.transcript, "confidence": 0.99}]},
            "start": 0.0,
            "duration": 1.0,
        }
        with self._lock:
            self._due.append((due, msg))
        self.sent_bytes += len(pcm16)

    def recv(self, timeout: float) -> Optional[Dict]:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                ready = [m for d, m in self._due if d <= now]
                self._due = [(d, m) for d, m in self._due if d > now]
            if ready:
                return ready[0]
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def close(self) -> None:
        self.connected = False


def _to_pcm16_mono_16k(samples: np.ndarray, sample_rate: int) -> bytes:
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        return b""
    if sample_rate != _PROVIDER_SR:
        src_t = np.arange(x.size, dtype=np.float64)
        dst_n = max(1, int(round(x.size * _PROVIDER_SR / sample_rate)))
        dst_t = np.linspace(0, x.size - 1, dst_n)
        x = np.interp(dst_t, src_t, x)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def _fragment_from_message(msg: Dict, fallback_ts: float) -> Optional[TranscriptFragment]:
    if not isinstance(msg, dict) or msg.get("type") != "Results":
        return None
    try:
        alt = msg["channel"]["alternatives"][0]
        text = (alt.get("transcript") or "").strip()
    except (KeyError, IndexError, TypeError):
        return None
    if not text:
        return None
    return TranscriptFragment(
        text=text,
        timestamp_sec=float(msg.get("start", fallback_ts)),
        is_final=bool(msg.get("is_final", True)),
        confidence=float(alt.get("confidence", 0.0) or 0.0),
    )


class DeepgramStreamingASR(ASRBridge):
    """
    Streaming ASR provider for the sidecar telephone path.

    ``push_audio`` is non-blocking: audio is queued for the worker thread
    and only already-arrived fragments are returned. On transport failure
    the worker records the error and ``push_audio`` raises ``TransportError``
    so ``GatedASR`` trips the circuit breaker.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        transport_factory: Optional[Callable[[], StreamingTransport]] = None,
        model: str = "nova-2",
        language: str = "en",
    ):
        self._factory: Callable[[], StreamingTransport]
        if transport_factory is not None:
            self._factory = transport_factory
        else:
            self._factory = lambda: DeepgramWSTransport(api_key or "", model=model, language=language)
        self._audio_q: "queue.Queue[bytes]" = queue.Queue()
        self._frag_q: "queue.Queue[TranscriptFragment]" = queue.Queue()
        self._error: Optional[BaseException] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_ts = 0.0

    # -- worker ---------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._error = None
            self._stop.clear()
            self._worker = threading.Thread(target=self._pump, name="deepgram-asr", daemon=True)
            self._worker.start()

    def _pump(self) -> None:
        try:
            transport = self._factory()
            transport.connect()
        except Exception as exc:
            self._error = exc
            return
        try:
            while not self._stop.is_set():
                try:
                    while True:
                        chunk = self._audio_q.get_nowait()
                        transport.send_audio(chunk)
                except queue.Empty:
                    pass
                try:
                    # Short poll quantum: bounds transcript-delay quantization
                    # without burning CPU (20 wakeups/s on one thread).
                    msg = transport.recv(timeout=0.05)
                except TransportError as exc:
                    self._error = exc
                    return
                if msg is not None:
                    frag = _fragment_from_message(msg, self._last_ts)
                    if frag is not None:
                        self._frag_q.put(frag)
        finally:
            try:
                transport.close()
            except Exception:
                pass

    # -- ASRBridge ------------------------------------------------------
    def push_audio(
        self, samples: np.ndarray, sample_rate: int, timestamp_sec: float
    ) -> List[TranscriptFragment]:
        if self._error is not None:
            raise TransportError(f"asr transport down: {self._error}")
        self._ensure_worker()
        if self._error is not None:
            raise TransportError(f"asr transport down: {self._error}")
        self._last_ts = timestamp_sec
        pcm = _to_pcm16_mono_16k(samples, sample_rate)
        if pcm:
            self._audio_q.put(pcm)
        out: List[TranscriptFragment] = []
        try:
            while True:
                out.append(self._frag_q.get_nowait())
        except queue.Empty:
            pass
        return out

    def push_text(self, text: str, timestamp_sec: float, is_final: bool = True) -> TranscriptFragment:
        # Out-of-process transcripts bypass the provider entirely.
        return TranscriptFragment(text=text, timestamp_sec=timestamp_sec, is_final=is_final)

    def reset(self) -> None:
        try:
            while True:
                self._frag_q.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                self._audio_q.get_nowait()
        except queue.Empty:
            pass

    def close(self) -> None:
        self._stop.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout=2.0)

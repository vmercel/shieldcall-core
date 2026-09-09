"""Detector application service.

Use cases the phone UI calls. Wraps SidecarRuntime. Does not hang up.
If a call is unknown or the worker is shedding, returns MONITOR (fail-open).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..acoustic.features import FEATURE_DIM
from ..acoustic.scorer import AcousticScore
from ..domain.model import CallSnapshot, Capabilities, DetectorDecision
from ..eval.corpora.independent_scripts import independent_scripts
from ..fusion.engine import FusionEngine
from ..linguistic.scorer import LinguisticFraudScorer, LinguisticScore
from ..runtime.runtime import SidecarRuntime
from ..runtime.session import CallSession, SessionEvent


class DetectorApplication:
    def __init__(self, runtime: SidecarRuntime):
        self.runtime = runtime

    def capabilities(self) -> Dict[str, Any]:
        return Capabilities().as_dict()

    def health(self) -> Dict[str, Any]:
        h = self.runtime.health()
        return {
            "live": h.live,
            "ready": h.ready,
            "active_calls": h.active_calls,
            "max_calls": h.max_calls,
            "shed_total": h.shed_total,
            "asr_breaker": h.asr_breaker,
            "ms_per_frame_ewma": h.ms_per_frame_ewma,
            "detail": h.detail,
            "channel_twin": False,
            "actuation": "recommend_only",
            "fail_open": True,
        }

    def ready(self) -> Dict[str, Any]:
        h = self.health()
        return {"ready": bool(h["ready"]), "detail": h["detail"]}

    def open_call(self, call_id: str) -> Dict[str, Any]:
        sess = self.runtime.open_call(call_id)
        return {"call_id": sess.call_id, "shed": sess.shed}

    def close_call(self, call_id: str) -> Dict[str, Any]:
        trace = self.runtime.close_call(call_id)
        return {"call_id": call_id, "n_decisions": len(trace or [])}

    def list_calls(self) -> Dict[str, Any]:
        return {"calls": [self._snapshot(s).as_dict() for s in self.runtime.list_calls()]}

    def get_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        sess = self.runtime.get_call(call_id)
        if sess is None:
            return None
        return self._snapshot(sess).as_dict()

    def decision(self, call_id: str) -> Optional[Dict[str, Any]]:
        sess = self.runtime.get_call(call_id)
        if sess is None:
            return None
        return self._decision(sess).as_event()

    def ingest_transcript(self, call_id: str, text: str, t: float = 0.0) -> Optional[Dict[str, Any]]:
        sess = self.runtime.get_call(call_id)
        if sess is None:
            return None
        sess.push_transcript(text, float(t))
        hop = np.zeros(int(8000 * 0.05), dtype=np.float32)
        events = sess.push_audio(hop, 8000)
        return self._decision(sess, events).as_event()

    def ingest_audio(self, call_id: str, samples: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        sess = self.runtime.get_call(call_id)
        if sess is None:
            return None
        t0 = time.perf_counter()
        events = sess.push_audio(samples, int(sample_rate) or 8000)
        self.runtime.observe_frame_ms((time.perf_counter() - t0) * 1000.0)
        return self._decision(sess, events).as_event()

    def ingest_chunk(
        self,
        call_id: str,
        text: str = "",
        t: float = 0.0,
        samples: Optional[np.ndarray] = None,
        sample_rate: int = 8000,
    ) -> Optional[Dict[str, Any]]:
        """One analyzed window: language and optional PCM. What the live-call UI sends."""
        sess = self.runtime.get_call(call_id)
        if sess is None:
            return None
        if text and text.strip():
            sess.push_transcript(text, float(t))
        if samples is not None and len(samples) > 0:
            events = sess.push_audio(samples, int(sample_rate) or 8000)
        else:
            hop = np.zeros(int(8000 * 0.05), dtype=np.float32)
            events = sess.push_audio(hop, 8000)
        return self._decision(sess, events).as_event()

    def inject(self, call_id: str, turns: List[str]) -> Optional[Dict[str, Any]]:
        sess = self.runtime.get_call(call_id)
        if sess is None:
            return None
        sr = 8000
        last = None
        for i, text in enumerate(turns):
            sess.push_transcript(text, float(i) * 0.8)
            noise = (0.01 * np.random.RandomState(i).randn(sr // 5)).astype(np.float32)
            last = sess.push_audio(noise, sr)
        ev = self._decision(sess, last).as_event()
        ev["n_turns"] = len(turns)
        return ev

    def scripts(self) -> Dict[str, Any]:
        items = []
        for s in independent_scripts():
            items.append(
                {
                    "id": s.script_id,
                    "family": s.family,
                    "is_scam": s.is_scam,
                    "n_turns": len(s.turns),
                    "cell": "se" if s.is_scam else "safe",
                    "label": s.script_id,
                }
            )
        return {"scripts": items}

    def score_linguistic(self, text: str, t: float = 0.0) -> Dict[str, Any]:
        score = LinguisticFraudScorer().update(text or "", float(t))
        return {
            "fraud": float(score.fraud_prob),
            "confidence": float(score.confidence),
            "stage": score.discourse_stage,
            "groups": list(score.active_groups),
            "escalation": float(score.escalation_factor),
            "progression_depth": int(score.progression_depth),
        }

    def score_acoustic(self, samples: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        from ..pipeline import PipelineConfig, ShieldCallPipeline

        pipe = ShieldCallPipeline(PipelineConfig(channel=None, fuse_every_n_frames=1))
        events = pipe.push_audio(samples, int(sample_rate) or 8000)
        synth = 0.0
        conf = 0.0
        for ev in reversed(events):
            if ev.acoustic is not None:
                synth = float(ev.acoustic.synthetic_prob)
                conf = float(ev.acoustic.confidence)
                break
        return {"synth": synth, "confidence": conf}

    def score_fuse(self, fraud: float, synth: float, t: float = 0.0) -> Dict[str, Any]:
        engine = FusionEngine(use_conformal=True)
        li = LinguisticScore(
            timestamp_sec=float(t),
            fraud_prob=float(np.clip(fraud, 0.0, 1.0)),
            confidence=0.8,
        )
        ac = AcousticScore(
            timestamp_sec=float(t),
            frame_index=0,
            synthetic_prob=float(np.clip(synth, 0.0, 1.0)),
            confidence=0.8,
            is_speech=True,
            features=np.zeros(FEATURE_DIM, dtype=np.float32),
        )
        engine.update_linguistic(li)
        engine.update_acoustic(ac)
        risk = engine.fuse(float(t))
        return {
            "fraud": float(risk.linguistic_fraud_prob),
            "synth": float(risk.acoustic_synth_prob),
            "risk": float(risk.risk_score),
            "tier": risk.tier,
            "regime": risk.regime,
            "or_label": float(FusionEngine.calibrated_or(synth, fraud)),
        }

    def _snapshot(self, sess: CallSession) -> CallSnapshot:
        return CallSnapshot(
            call_id=sess.call_id,
            shed=sess.shed,
            closed=sess.closed,
            n_turns=int(sess.n_turns),
            frames=int(sess.frames),
            last_action=sess.last_action.value,
            last_text=sess.last_text,
            last=self._decision(sess),
        )

    def _decision(self, sess: CallSession, events: Optional[List[SessionEvent]] = None) -> DetectorDecision:
        risk = None
        explanation = ""
        if events:
            for ev in reversed(events):
                if ev.pipeline is not None and ev.pipeline.risk is not None:
                    risk = ev.pipeline.risk
                    explanation = getattr(risk, "explanation", "") or ""
                    break
        return DetectorDecision(
            call_id=sess.call_id,
            action=sess.last_action.value,
            shed=sess.shed,
            t=float(risk.timestamp_sec) if risk else 0.0,
            synth=float(risk.acoustic_synth_prob) if risk else 0.0,
            fraud=float(risk.linguistic_fraud_prob) if risk else 0.0,
            risk=float(risk.risk_score) if risk else 0.0,
            regime=risk.regime if risk else "",
            tier=risk.tier if risk else "SAFE",
            stage=risk.discourse_stage if risk else "",
            n_turns=int(sess.n_turns),
            last_text=sess.last_text,
            explanation=explanation,
        )

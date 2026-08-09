"""
End-to-end streaming demo of the dual-stream ShieldCall engine.

Generates synthetic audio + scheduled transcript fragments so the full
stack (TCT  ->  STRF  ->  SDTG  ->  CSCF  ->  CSR  ->  CTE) can be exercised without
a live microphone or ASR service.
"""

from __future__ import annotations

import numpy as np

from shieldcall.pipeline import ShieldCallPipeline, PipelineConfig
from shieldcall.audio.channel import ChannelConfig, CodecProfile
from shieldcall.linguistic.asr_bridge import ScheduledTranscriptASR


def make_tone(sr: int, duration: float, freq: float = 180.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.3 * np.sin(2 * np.pi * freq * t)
    x += 0.1 * np.sin(2 * np.pi * 2 * freq * t)
    x += 0.02 * rng.randn(len(t))
    return x.astype(np.float32)


def main() -> None:
    sr = 8000
    config = PipelineConfig(
        target_sr=sr,
        channel=ChannelConfig(profile=CodecProfile.G711_ULAW, snr_db=30.0, seed=7),
        use_conformal=True,
        fuse_every_n_frames=5,
    )

    transcript_schedule = [
        (0.5, "Hello, this is a courtesy call from your bank."),
        (2.0, "We have detected unusual activity on your account."),
        (3.5, "To protect you we need you to verify your social security number immediately."),
        (5.0, "Please purchase gift cards and read the numbers back to me. Do not tell anyone."),
        (6.5, "There is an active warrant if you do not comply within the next hour."),
    ]
    asr = ScheduledTranscriptASR(transcript_schedule)
    pipe = ShieldCallPipeline(config=config, asr=asr)

    audio = np.concatenate(
        [
            make_tone(sr, 2.0, 160.0, seed=1),
            make_tone(sr, 3.0, 190.0, seed=2),
            make_tone(sr, 2.0, 170.0, seed=3),
        ]
    )

    print("ShieldCall Core v0.2  -  Dual-stream streaming demo")
    print("Subsystems: TCT | STRF | SDTG | CSCF | PMA | CSR | CTE")
    print("=" * 72)

    for ev in pipe.stream(audio, sr, chunk_ms=100.0):
        risk = ev.risk
        if risk is None:
            continue
        if risk.tier != "SAFE" or (ev.frame and ev.frame.frame_index % 20 == 0):
            band = f"[{risk.conformal_lower:.2f},{risk.conformal_upper:.2f}]"
            print(
                f"t={risk.timestamp_sec:5.2f}s  "
                f"risk={risk.risk_score:.3f}{band}  "
                f"tier={risk.tier:12s}  "
                f"synth={risk.acoustic_synth_prob:.2f}  "
                f"fraud={risk.linguistic_fraud_prob:.2f}  "
                f"coact={risk.coactivation:.2f}  "
                f"stage={risk.discourse_stage or '-':10s}  "
                f"reg={risk.regime}"
            )
            if risk.tier in ("HIGH_RISK", "SUSPICIOUS", "ABSTAIN") and risk.threat_explanation:
                if risk.threat_explanation.counterfactuals:
                    cf = risk.threat_explanation.counterfactuals[0]
                    print(f"         CTE: {cf.detail}")

    cov = pipe.coverage.snapshot()
    print("=" * 72)
    print(
        f"Coverage debt index={cov.debt_index:.3f}  "
        f"mean_gap={cov.mean_gap:.3f}  high_gap_rate={cov.high_gap_rate:.3f}"
    )
    print("Demo finished. Core pipeline is operational.")
    print("Next: train STRF prototypes on real telephony corpora; attach production ASR.")


if __name__ == "__main__":
    main()

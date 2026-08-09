from shieldcall.linguistic.scorer import LinguisticFraudScorer
from shieldcall.linguistic.discourse import ScamDiscourseGraph, STAGES
from shieldcall.linguistic.asr_bridge import ScheduledTranscriptASR


def test_benign_low_score():
    scorer = LinguisticFraudScorer()
    s = scorer.update("Hello, how are you today? Looking forward to our meeting.", 0.0)
    assert s.fraud_prob < 0.35


def test_scam_high_score():
    scorer = LinguisticFraudScorer()
    scorer.update("Hello, this is a courtesy call from the IRS.", 0.0)
    scorer.update("We detected unusual activity on your social security number.", 1.0)
    s = scorer.update(
        "You must purchase gift cards immediately and do not tell anyone. There is a warrant.",
        2.0,
    )
    assert s.fraud_prob > 0.5
    assert s.progression_depth >= 3
    assert len(s.active_groups) >= 1


def test_discourse_progression():
    g = ScamDiscourseGraph()
    stages = []
    for t, text in [
        (0, "Hello, calling from your bank"),
        (1, "We found unusual activity"),
        (2, "You must act immediately"),
        (3, "Give me your social security number"),
        (4, "Buy gift cards now"),
        (5, "Do not tell anyone"),
        (6, "There is a warrant for your arrest"),
    ]:
        st = g.update(text, float(t))
        stages.append(st.stage)
    assert st.path_score > 0.4
    assert st.progression_depth >= 4
    assert all(s in STAGES for s in stages)


def test_scheduled_asr():
    asr = ScheduledTranscriptASR([(0.5, "hello"), (1.5, "world")])
    import numpy as np
    out = asr.push_audio(np.zeros(100), 8000, 0.4)
    assert out == []
    out = asr.push_audio(np.zeros(100), 8000, 0.6)
    assert len(out) == 1 and out[0].text == "hello"
    out = asr.push_audio(np.zeros(100), 8000, 2.0)
    assert len(out) == 1 and out[0].text == "world"

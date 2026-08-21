"""Agent policy tests: scripted percepts, no audio, no LLM."""

from shieldcall.agent import Action, DefenseAgent, Hypothesis, Perception


def _run(percents):
    ag = DefenseAgent()
    decisions = [ag.step(p) for p in percents]
    return ag, decisions


def test_benign_monitors():
    ag, decs = _run(
        [
            Perception(t, synth=0.1, fraud=0.05, risk=0.1)
            for t in (0.5, 1.0, 1.5, 2.0)
        ]
    )
    assert ag.belief.mode() == Hypothesis.BENIGN
    assert decs[-1].action == Action.MONITOR
    assert ag.challenges_used == 0


def test_social_engineering_warns_instead_of_wasting_liveness():
    """A human scammer will pass a nonce. Warn/escalate; do not burn the challenge."""
    ag, decs = _run(
        [
            Perception(0.5, synth=0.15, fraud=0.2, risk=0.2),
            Perception(1.5, synth=0.18, fraud=0.75, risk=0.7, regime="social_engineering"),
            Perception(2.5, synth=0.16, fraud=0.88, risk=0.85, regime="social_engineering"),
        ]
    )
    assert ag.belief.mode() == Hypothesis.SOCIAL_ENGINEERING
    assert decs[-1].action in (Action.WARN, Action.ESCALATE)
    assert ag.challenges_used == 0


def test_synthetic_probe_may_challenge():
    ag, decs = _run(
        [
            Perception(0.5, synth=0.4, fraud=0.1, risk=0.25),
            Perception(1.5, synth=0.82, fraud=0.12, risk=0.70, regime="deepfake_probe"),
            Perception(2.5, synth=0.88, fraud=0.10, risk=0.75, regime="deepfake_probe"),
        ]
    )
    assert ag.belief.mode() in (Hypothesis.SYNTHETIC_FULL, Hypothesis.UNKNOWN_FAMILY)
    assert any(d.action in (Action.CHALLENGE, Action.WARN, Action.ESCALATE) for d in decs)


def test_handoff_hypothesis_escalates():
    ag, decs = _run(
        [
            Perception(1.0, synth=0.25, fraud=0.4, handoff_score=0.2, handoff_pvalue=0.4),
            Perception(
                3.0,
                synth=0.55,
                fraud=0.7,
                handoff_score=0.85,
                handoff_pvalue=0.04,
                risk=0.8,
            ),
            Perception(
                4.0,
                synth=0.6,
                fraud=0.75,
                handoff_score=0.9,
                handoff_pvalue=0.03,
                risk=0.85,
            ),
        ]
    )
    assert ag.belief.p[Hypothesis.HANDOFF] >= ag.belief.p[Hypothesis.BENIGN]
    assert any(d.action in (Action.ESCALATE, Action.WARN, Action.CHALLENGE) for d in decs)


def test_unknown_family_can_adapt():
    ag, decs = _run(
        [
            Perception(1.0, synth=0.45, fraud=0.2, coverage_gap=0.85, risk=0.4),
            Perception(2.0, synth=0.5, fraud=0.15, coverage_gap=0.9, risk=0.45),
            Perception(3.0, synth=0.48, fraud=0.12, coverage_gap=0.92, risk=0.44),
        ]
    )
    assert ag.belief.mode() in (Hypothesis.UNKNOWN_FAMILY, Hypothesis.SYNTHETIC_FULL)
    assert any(d.action in (Action.ADAPT, Action.CHALLENGE, Action.ESCALATE) for d in decs)


def test_belief_is_a_distribution():
    ag = DefenseAgent()
    ag.step(Perception(0.0, synth=0.3, fraud=0.3))
    s = sum(ag.belief.p.values())
    assert abs(s - 1.0) < 1e-6


def test_trace_is_serializable_and_llm_free():
    import shieldcall.agent.agent as mod

    assert "openai" not in mod.__file__
    ag, _ = _run([Perception(0.4, synth=0.2, fraud=0.1)])
    blob = ag.trace_dicts()
    assert blob[0]["action"] in {a.value for a in Action}
    assert "rationale" in blob[0]


def test_challenge_tool_issues_nonce():
    ag = DefenseAgent()
    d = ag.step(Perception(1.0, synth=0.2, fraud=0.9, regime="social_engineering", risk=0.8))
    # May escalate instead of challenge depending on threat mass; both are allowed
    if d.action == Action.CHALLENGE:
        assert "nonce" in d.tool.data
        assert ag.challenges_used == 1

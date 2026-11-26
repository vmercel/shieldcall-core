"""Factorial discourse model (ADR-017, TDD v2.0 sections 7.3, 7.4, 7.5).

Ground truth is synthetic turns with known content, so every
expected classification below is derived from the turn text,
not asserted from the code under test. Nothing here touches
licensed ASVspoof data, and no model is downloaded: the
lexicons are explicit pattern lists in the module.
"""

import hashlib
import json
import math

import pytest

from shieldcall.linguistic.discourse import LEXICON_LOCK
from shieldcall.linguistic.factorial import (
    ACTION_EMISSIONS,
    ACTION_LEXICON_LOCK,
    ACTIONS,
    CallTranscript,
    DialogueTurn,
    FactorialConfig,
    FactorialDiscourseModel,
    HARMFUL_ACTIONS,
    TACTIC_EMISSIONS,
    TACTIC_LEXICON_LOCK,
    TfidfPrefixBaseline,
    compare_lead_time,
    evaluate_lead_time,
    factorial_scorer_for,
    harmful_probability,
    requested_action_active,
    threshold_at_false_alert_budget,
)


def _model(**config_kwargs):
    return FactorialDiscourseModel(FactorialConfig(**config_kwargs))


def _turn(text, t=0.0, speaker="far"):
    return DialogueTurn(text=text, timestamp_sec=t, speaker=speaker)


def _run(model, turns):
    return [model.update(t.text, t.timestamp_sec, t.speaker) for t in turns]


# ---------------------------------------------------------------- locks


def test_lexicon_locks_are_stable_16_hex_and_distinct():
    for lock in (TACTIC_LEXICON_LOCK, ACTION_LEXICON_LOCK, LEXICON_LOCK):
        assert len(lock) == 16
        int(lock, 16)
    assert TACTIC_LEXICON_LOCK != ACTION_LEXICON_LOCK
    assert TACTIC_LEXICON_LOCK != LEXICON_LOCK
    recomputed = hashlib.sha256(
        json.dumps(TACTIC_EMISSIONS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    assert recomputed == TACTIC_LEXICON_LOCK
    recomputed_actions = hashlib.sha256(
        json.dumps(ACTION_EMISSIONS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    assert recomputed_actions == ACTION_LEXICON_LOCK


# ---------------------------------------------------------------- actions


def test_each_action_classified_from_a_far_end_turn():
    cases = [
        (
            "Please tell me your social security number and date of birth.",
            "personal_data",
            False,
        ),
        (
            "Read me the verification code you just received.",
            "one_time_code_or_credential",
            True,
        ),
        (
            "You must buy a gift card and read the numbers to me.",
            "payment_or_gift_card",
            True,
        ),
        (
            "Send the money by wire transfer to this account.",
            "wire_or_crypto_transfer",
            True,
        ),
        (
            "Install TeamViewer and grant me access to your computer.",
            "remote_access",
            True,
        ),
    ]
    for text, expected_action, harmful in cases:
        state = _model().update(text, 0.0, "far")
        assert state.action == expected_action, text
        assert (state.action in HARMFUL_ACTIONS) == harmful, text
        assert requested_action_active(state) == harmful, text
        assert abs(sum(state.action_probs.values()) - 1.0) < 1e-9
        assert abs(sum(state.stage_probs.values()) - 1.0) < 1e-6


def test_personal_data_is_not_harmful_per_tdd_eq_16():
    state = _model().update("What is your bank account and routing number?", 0.0, "far")
    assert state.action == "personal_data"
    assert state.action not in HARMFUL_ACTIONS
    assert state.harm_probability < 0.5
    assert not requested_action_active(state)


def test_harmful_actions_set_is_all_but_none_and_personal_data():
    assert HARMFUL_ACTIONS == set(ACTIONS) - {"none", "personal_data"}
    assert "none" not in HARMFUL_ACTIONS


def test_benign_turn_requests_nothing():
    state = _model().update(
        "Hello, just a reminder about your appointment tomorrow. "
        "No action is required. Have a nice day.",
        0.0,
        "far",
    )
    assert state.action == "none"
    assert state.fraud_score < 0.3
    assert state.stage in ("BENIGN", "GREETING")


# ---------------------------------------------------------------- guard


def test_near_end_request_never_counts_as_an_action():
    model = _model()
    before = model.update("Hello, good morning.", 0.0, "far")
    belief_before = dict(before.action_probs)
    state = model.update("Can I pay by card? I need to send money today.", 10.0, "near")
    assert state.action == "none"
    assert state.harm_probability == 0.0
    assert state.guard_speaker_role is True
    assert state.guard_applied is True
    # The action belief is frozen by a user turn.
    assert state.action_probs == pytest.approx(belief_before)


def test_reported_speech_is_down_weighted_on_tdd_example():
    direct = _model().update("Buy a gift card now and read the numbers.", 0.0, "far")
    reported = _model().update("They told me to buy gift cards.", 0.0, "far")
    assert reported.guard_reported_speech is True
    assert reported.guard_applied is True
    assert reported.fraud_score < direct.fraud_score
    # Guarded evidence still moves the belief toward the
    # mentioned action, but the turn is not classified as a
    # request: the MAP action stays none, so the cascade
    # proxy would not fire on a reported mention.
    assert reported.action == "none"
    assert (
        reported.action_probs["payment_or_gift_card"]
        > direct.action_probs["payment_or_gift_card"] / 4
    )
    assert reported.action_probs["payment_or_gift_card"] > 1.0 / len(ACTIONS)


def test_negation_is_down_weighted_on_tdd_example():
    direct = _model().update("Tell me your verification code.", 0.0, "far")
    negated = _model().update("We will never ask for your code.", 0.0, "far")
    assert negated.guard_negated_request is True
    assert negated.fraud_score < direct.fraud_score


def test_user_initiated_request_is_down_weighted():
    alone = _model()
    alone.update("Hello, good morning.", 0.0, "far")
    direct = alone.update("Buy a gift card now.", 10.0, "far")

    initiated = _model()
    initiated.update("Hello, good morning.", 0.0, "far")
    initiated.update("Should I buy a gift card to pay this?", 10.0, "near")
    echoed = initiated.update("Yes, buy a gift card now.", 20.0, "far")

    assert echoed.guard_user_initiated is True
    assert echoed.fraud_score < direct.fraud_score


def test_guard_does_not_delete_strong_evidence():
    # A guarded turn with repeated harmful requests still moves
    # the belief; down-weighting is not deletion. The MAP action
    # stays none for a reported mention, but the harmful
    # probability rises well above its prior.
    state = _model().update(
        "They told me to buy gift cards, gift cards, gift cards right now.", 0.0, "far"
    )
    assert state.guard_reported_speech is True
    assert state.action_probs["payment_or_gift_card"] > 1.0 / len(ACTIONS)
    assert state.harm_probability > 0.24  # the prior harmful total


# ---------------------------------------------------------------- stage dwell


def test_stage_dwell_floor_holds_one_alarming_turn():
    model = _model(min_stage_turns=2)
    # The dwell floor applies from the first turn: one turn
    # proposes a stage, the second consecutive turn commits it.
    first_greeting = model.update(
        "Hello, good morning, this is a courtesy call.", 0.0, "far"
    )
    assert first_greeting.stage_map == "GREETING"
    assert first_greeting.stage == "BENIGN"
    second_greeting = model.update(
        "Hello, good morning, this is a courtesy call.", 10.0, "far"
    )
    assert second_greeting.stage == "GREETING"
    first = model.update(
        "This is Agent Smith from the IRS calling about your account.", 20.0, "far"
    )
    assert first.stage_map == "AUTHORITY"
    assert first.stage == "GREETING"
    second = model.update(
        "This is Agent Smith from the IRS calling about your account.", 30.0, "far"
    )
    assert second.stage_map == "AUTHORITY"
    assert second.stage == "AUTHORITY"


def test_dwell_floor_of_one_commits_immediately():
    model = _model(min_stage_turns=1)
    model.update("Hello, good morning.", 0.0, "far")
    state = model.update(
        "This is Agent Smith from the IRS calling about your account.", 10.0, "far"
    )
    assert state.stage == "AUTHORITY"


# ---------------------------------------------------------------- tactics


def test_tactics_rise_on_their_evidence_and_persist():
    model = _model()
    authority = model.update(
        "This is Agent Smith from the IRS calling about your account.", 0.0, "far"
    )
    assert authority.tactic_probs["authority"] > 0.5
    assert "authority" in authority.active_tactics
    assert authority.tactic_probs["reward"] < 0.2

    neutral = model.update("Let me check your file for a moment.", 10.0, "far")
    # No new evidence: persistence holds the tactic above its prior.
    assert neutral.tactic_probs["authority"] > 0.10
    assert neutral.tactic_probs["authority"] < authority.tactic_probs["authority"]

    urgent = model.update("You must act immediately, within thirty minutes.", 20.0, "far")
    assert urgent.tactic_probs["urgency"] > 0.5
    fearful = model.update(
        "There is a warrant for your arrest if you do not pay.", 30.0, "far"
    )
    assert fearful.tactic_probs["fear_threat"] > 0.5


def test_tactic_probs_stay_in_range():
    state = _model().update("Hello, you won a prize, trust me, keep it secret.", 0.0, "far")
    for prob in state.tactic_probs.values():
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------- hazard


def test_hazard_arithmetic_and_horizon():
    model = _model(horizon_turns=3)
    state = model.update("Read me the verification code you just received.", 0.0, "far")
    harm = state.harm_probability
    assert harm > 0.5
    assert state.hazard == pytest.approx(1.0 - (1.0 - harm) ** 3)
    assert state.fraud_score == pytest.approx(state.hazard)
    assert state.fraud_score >= harm

    one_step = _model(horizon_turns=1).update(
        "Read me the verification code you just received.", 0.0, "far"
    )
    assert one_step.hazard == pytest.approx(one_step.harm_probability)
    assert one_step.hazard < state.hazard


def test_harmful_probability_sums_the_harmful_actions():
    probs = {a: 0.1 for a in ACTIONS}
    probs["none"] = 0.4
    probs["personal_data"] = 0.1
    # 0.1 each on the four harmful actions = 0.4.
    assert harmful_probability(probs) == pytest.approx(0.4)


def test_fraud_rises_along_a_scam_script():
    model = _model()
    turns = [
        "Hello, good morning, this is a courtesy call.",
        "This is Agent Smith from the IRS calling about your account.",
        "We found suspicious activity and your account is suspended.",
        "You must act immediately, within thirty minutes.",
        "Read me the verification code you just received.",
    ]
    states = _run(model, [DialogueTurn(t, float(i * 10)) for i, t in enumerate(turns)])
    assert states[-1].fraud_score > states[0].fraud_score
    assert states[-1].action == "one_time_code_or_credential"
    assert requested_action_active(states[-1])
    assert states[-1].progression_depth >= 3


# ---------------------------------------------------------------- robustness


def test_empty_turn_keeps_beliefs_and_reports_current_state():
    model = _model()
    first = model.update("Read me the verification code.", 0.0, "far")
    empty = model.update("   ", 10.0, "far")
    assert empty.timestamp_sec == 10.0
    assert empty.action_probs == pytest.approx(first.action_probs)
    assert empty.fraud_score == pytest.approx(first.fraud_score)


def test_unknown_speaker_and_bad_config_raise():
    with pytest.raises(ValueError):
        _model().update("Hello.", 0.0, "robot")
    with pytest.raises(ValueError):
        FactorialConfig(horizon_turns=0)
    with pytest.raises(ValueError):
        FactorialConfig(tactic_hit_active=0.05, tactic_hit_inactive=0.08)
    with pytest.raises(ValueError):
        FactorialConfig(guard_emission_weight=1.5)


def test_reset_restores_the_initial_state():
    model = _model()
    model.update("Read me the verification code.", 0.0, "far")
    model.reset()
    # An empty turn reports the current state without updating
    # any belief, so it exposes the reset priors directly.
    state = model.update("   ", 0.0, "far")
    assert state.stage == "BENIGN"
    assert state.tactic_probs["authority"] == pytest.approx(0.10)
    assert state.action_probs["none"] == pytest.approx(0.70)
    benign = model.update("Hello, just a reminder about your appointment.", 10.0, "far")
    assert benign.fraud_score < 0.3


def test_requested_action_active_threshold():
    state = _model().update("Read me the verification code.", 0.0, "far")
    assert requested_action_active(state, threshold=0.3)
    assert not requested_action_active(state, threshold=0.99)
    with pytest.raises(ValueError):
        requested_action_active(state, threshold=1.5)


# ---------------------------------------------------------------- lead time


def _stub_scorer(score_map):
    def score(turns):
        return [score_map[i] for i in range(len(turns))]

    return score


def test_lead_time_hand_computed_case():
    scam = CallTranscript(
        turns=[_turn("hello", t) for t in (0.0, 10.0, 20.0, 30.0)],
        label=1,
        first_harm_index=3,
    )
    benign = CallTranscript(
        turns=[_turn("hello", t) for t in (0.0, 10.0)], label=0
    )
    calls = [scam, benign]

    def scorer(turns):
        return [0.1, 0.6, 0.2, 0.9][: len(turns)]

    report = evaluate_lead_time(calls, 0.5, scorer)
    assert report.warned_before_harm == 1
    assert report.missed == 0
    assert report.lead_times_sec == (20.0,)
    assert report.median_lead_time_sec == 20.0
    assert report.false_alerts == 1
    assert report.false_alerts_per_100_benign == 100.0


def test_lead_time_missed_and_late_warnings():
    missed_call = CallTranscript(
        turns=[_turn("hello", t) for t in (0.0, 10.0, 20.0)],
        label=1,
        first_harm_index=2,
    )
    report = evaluate_lead_time(
        [missed_call], 0.5, _stub_scorer({0: 0.1, 1: 0.2, 2: 0.9})
    )
    # Warning only at the harmful turn itself: lead time zero,
    # counted as warned, not missed.
    assert report.warned_before_harm == 1
    assert report.missed == 0
    assert report.lead_times_sec == (0.0,)

    late_call = CallTranscript(
        turns=[_turn("hello", t) for t in (0.0, 10.0, 20.0, 30.0)],
        label=1,
        first_harm_index=2,
    )
    report = evaluate_lead_time(
        [late_call], 0.5, _stub_scorer({0: 0.1, 1: 0.2, 2: 0.2, 3: 0.9})
    )
    # The only warning comes after the harmful request.
    assert report.warned_before_harm == 0
    assert report.missed == 1
    assert report.median_lead_time_sec is None


def test_scorer_must_return_one_score_per_turn():
    call = CallTranscript(turns=[_turn("hello", 0.0)], label=0)
    with pytest.raises(ValueError):
        evaluate_lead_time([call], 0.5, lambda turns: [])


def test_threshold_at_false_alert_budget_picks_most_sensitive_passing():
    benign = [[0.9], [0.1], [0.2]]
    # 1 of 3 false alerts is 33.3 per 100: passes a 34 budget at 0.9.
    assert threshold_at_false_alert_budget(benign, 34.0) == pytest.approx(0.9)
    # Zero budget: only the above-all-scores threshold passes.
    assert threshold_at_false_alert_budget(benign, 0.0) > 0.9
    # Generous budget: the lowest observed score passes.
    assert threshold_at_false_alert_budget(benign, 100.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        threshold_at_false_alert_budget([], 10.0)
    with pytest.raises(ValueError):
        threshold_at_false_alert_budget(benign, 101.0)


# ---------------------------------------------------------------- baseline


def test_tfidf_baseline_fits_scores_and_is_deterministic():
    texts = [
        "buy a gift card wire transfer bitcoin send money",
        "verification code password pin card number",
        "appointment reminder no action required have a nice day",
        "thank you have a good day survey reminder",
    ]
    labels = [1, 1, 0, 0]
    baseline = TfidfPrefixBaseline().fit(texts, labels)
    assert baseline.is_fitted
    scam_score = baseline.score_text("buy a gift card send money wire transfer")
    benign_score = baseline.score_text("appointment reminder have a nice day")
    assert scam_score > benign_score
    assert 0.0 <= scam_score <= 1.0

    again = TfidfPrefixBaseline().fit(texts, labels)
    assert again.score_text("buy a gift card") == pytest.approx(
        baseline.score_text("buy a gift card")
    )


def test_tfidf_baseline_rejects_bad_fits_and_unfitted_scoring():
    with pytest.raises(ValueError):
        TfidfPrefixBaseline().score_text("hello")
    with pytest.raises(ValueError):
        TfidfPrefixBaseline().fit(["only one class"], [1])
    with pytest.raises(ValueError):
        TfidfPrefixBaseline().fit([], [])


def test_tfidf_prefix_scores_accumulate_text():
    baseline = TfidfPrefixBaseline().fit(
        [
            "buy a gift card wire transfer",
            "appointment reminder no action required",
        ],
        [1, 0],
    )
    turns = [_turn("hello good morning", 0.0), _turn("buy a gift card now", 10.0)]
    scores = baseline.score_prefixes(turns)
    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_compare_lead_time_at_equal_false_alert_budget():
    scam_turns = [
        _turn("Hello good morning", 0.0),
        _turn("This is Agent Smith from the IRS", 10.0),
        _turn("Suspicious activity, your account is suspended", 20.0),
        _turn("Buy a gift card now and read the numbers", 30.0),
    ]
    benign_turns = [
        _turn("Hello, reminder about your appointment", 0.0),
        _turn("No action is required, have a nice day", 10.0),
    ]
    calls = [
        CallTranscript(turns=scam_turns, label=1, first_harm_index=3),
        CallTranscript(turns=scam_turns, label=1, first_harm_index=3),
        CallTranscript(turns=benign_turns, label=0),
        CallTranscript(turns=benign_turns, label=0),
    ]
    baseline = TfidfPrefixBaseline().fit(
        [" ".join(t.text for t in c.turns) for c in calls],
        [c.label for c in calls],
    )
    factorial_scorer = factorial_scorer_for(FactorialDiscourseModel())
    comparison = compare_lead_time(
        calls, factorial_scorer, baseline.score_prefixes, budget_per_100=50.0
    )
    assert comparison.budget_per_100_benign == 50.0
    assert comparison.factorial.n_scam_with_harm == 2
    assert comparison.baseline.n_scam_with_harm == 2
    # Thresholds were set from the benign calls at the budget.
    assert comparison.factorial.false_alerts_per_100_benign <= 50.0 + 1e-9
    assert comparison.baseline.false_alerts_per_100_benign <= 50.0 + 1e-9
    assert isinstance(comparison.factorial_wins_on_lead_time, bool)


def test_factorial_scorer_resets_between_calls():
    scorer = factorial_scorer_for(FactorialDiscourseModel())
    scam = [_turn("Read me the verification code.", 0.0)]
    benign = [_turn("Hello, reminder about your appointment.", 0.0)]
    assert scorer(scam)[0] > 0.5
    # The same model object scores the benign call from a reset state.
    assert scorer(benign)[0] < 0.3

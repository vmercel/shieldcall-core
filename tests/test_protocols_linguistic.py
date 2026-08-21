"""Linguistic protocol tests that do not need recorded speech."""

from shieldcall.eval.corpora.vishing_scripts import all_scripts, heldout_scripts, train_scripts
from shieldcall.eval.protocols import linguistic_protocol, score_script


def test_corpus_splits_and_size():
    assert len(all_scripts()) >= 80
    assert len(train_scripts()) >= 40
    assert len(heldout_scripts()) >= 30
    ids = [s.script_id for s in all_scripts()]
    assert len(ids) == len(set(ids))
    assert any(s.is_scam for s in heldout_scripts())
    assert any(not s.is_scam for s in heldout_scripts())
    assert any(s.trap == "paraphrase" for s in heldout_scripts())
    assert any(s.trap == "isolated_keyword" for s in heldout_scripts())


def test_canonical_scam_high_benign_low():
    scam = next(s for s in train_scripts() if s.script_id == "s01")
    benign = next(s for s in train_scripts() if s.script_id == "b01")
    assert score_script(scam, True).fraud_prob > 0.5
    assert score_script(benign, True).fraud_prob < 0.35


def test_heldout_sdtg_not_worse_than_keywords():
    ling = linguistic_protocol()
    assert ling["ling_heldout_patterns_sdtg"].auc + 1e-9 >= ling["ling_heldout_patterns"].auc
    traps = ling["ling_heldout_trap_mean"].eer_estimate
    assert traps < 0.5

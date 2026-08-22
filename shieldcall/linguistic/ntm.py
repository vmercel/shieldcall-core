"""Neural/linear trajectory model trained only on the train split.

A logistic regressor over per-stage emission counts plus a forward-transition
count. This is the NTM baseline: learned, still small, no LLM, no test
writer in the training set.

If NTM ≉ wide-lexicon on independent text, the discourse claim is not
supported (checklist kill criterion).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .discourse import STAGE_EMISSIONS, STAGES, ScamDiscourseGraph, wide_lexicon_score


SCAM_STAGES = [s for s in STAGES if s != "BENIGN"]


def turn_features(turns: Sequence[str]) -> np.ndarray:
    graph = ScamDiscourseGraph()
    counts = {s: 0.0 for s in SCAM_STAGES}
    last_stage = "BENIGN"
    forward = 0.0
    chain = ["GREETING", "AUTHORITY", "PROBLEM", "URGENCY", "HARVEST", "PAYMENT", "SECRECY", "THREAT"]
    idx = {s: i for i, s in enumerate(chain)}
    path = 0.0
    for i, text in enumerate(turns):
        st = graph.update(text, float(i))
        if st.stage in counts:
            counts[st.stage] += 1.0
        if last_stage in idx and st.stage in idx and idx[st.stage] >= idx[last_stage]:
            forward += 1.0
        last_stage = st.stage
        path = st.path_score
    wide = max((wide_lexicon_score(t) for t in turns), default=0.0)
    vec = [counts[s] for s in SCAM_STAGES] + [forward, path, wide, float(len(turns))]
    return np.asarray(vec, dtype=np.float64)


class NeuralTrajectoryModel:
    def __init__(self, seed: int = 0):
        self.clf = LogisticRegression(max_iter=400, solver="liblinear", random_state=seed)
        self.fitted = False

    def fit(self, turn_lists: Iterable[Sequence[str]], labels: Iterable[int]) -> "NeuralTrajectoryModel":
        X = np.stack([turn_features(t) for t in turn_lists], axis=0)
        y = np.asarray(list(labels), dtype=int)
        self.clf.fit(X, y)
        self.fitted = True
        return self

    def score(self, turns: Sequence[str]) -> float:
        if not self.fitted:
            return float(turn_features(turns)[-3])  # path_score fallback
        proba = self.clf.predict_proba(turn_features(turns).reshape(1, -1))[0]
        # class 1 is scam if both classes seen
        classes = list(self.clf.classes_)
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[-1])

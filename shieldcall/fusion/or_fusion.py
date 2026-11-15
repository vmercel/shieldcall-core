"""Calibrated OR fusion with a monotone safety envelope (TDD v2.0 section 8.1; ADR-006).

Each stream supplies a calibrated LLR (stage 2a per-cell maps). With
deployment priors ``pi_v`` (synthetic voice) and ``pi_s`` (fraud script)
the per-stream posteriors are TDD eq. 17::

    pV = sigma(l_ac + logit(pi_v))
    pS = sigma(l_lang + logit(pi_s) + delta_att)

where ``delta_att`` is the STIR/SHAKEN attestation term (A: 0, B: 0.07,
C or unsigned: 0.18 nats, bounded by |delta_att| <= 0.2). Attestation
moves the prior odds of a scam, never the verdict.

The threat probability is P(V union S). The joint term is a mixture
between independence and full redundancy, with one dependence parameter
``rho`` in [0, 1] fitted on paired development calls by minimising the
Cllr of the fused output (TDD eq. 18-19)::

    P(V and S) = (1 - rho) * pV * pS + rho * min(pV, pS)
    p_threat   = (1 - rho) * (1 - (1 - pV) * (1 - pS)) + rho * max(pV, pS)

Proposition 1 (monotone safety envelope): for every rho in [0, 1],
(a) max(pV, pS) <= p_threat <= noisy-OR, (b) p_threat is non-decreasing
in each argument, (c) rho = 1 gives max, rho = 0 gives noisy-OR. A
strong single-stream threat is therefore never diluted by a quiet
second stream, and evidence can never hurt.

Severity is not probability. An AI-voiced scam (both V and S) is worse
than either alone, so the joint probability is carried separately as
``sev = P(V and S) / p_threat`` for the agent to choose stronger
guidance; the probability itself stays calibrated.

Temporal smoothing follows TDD section 8.1: per-stream LLRs are
accumulated with leaky integration (beta = 0.9 per fusion tick, cap
|llr_bar| <= 8) before conversion to posteriors.

CSCF stays the production default until this rule passes its release
gate (G2 calibration + G7 shadow, TDD section 11.2). The two run side
by side in shadow mode: ``ShadowFusion`` feeds both engines the same
ticks, records paired rows, and compares them without touching the
CSCF decision path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..acoustic.scorer import AcousticScore
from ..linguistic.scorer import LinguisticScore
from .calibration import cllr
from .engine import FusionEngine

# TDD section 8.1: STIR/SHAKEN attestation term in nats. A spoofed
# number with full attestation is still caught on content.
ATTESTATION_DELTA: Dict[str, float] = {
    "A": 0.0,
    "B": 0.07,
    "C": 0.18,
    "unsigned": 0.18,
}

# Deployment priors. Defaults match the repo's p_target = 0.05
# convention; they are base rates to be set from measured traffic, not
# tuned to fit a benchmark.
DEFAULT_PRIOR_V = 0.05
DEFAULT_PRIOR_S = 0.05

# Leaky-integration constants, TDD section 8.1.
DEFAULT_BETA = 0.9
DEFAULT_LLR_CAP = 8.0

_EPS = 1e-6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _logit(p: float) -> float:
    p = float(np.clip(p, _EPS, 1.0 - _EPS))
    return float(np.log(p / (1.0 - p)))


def attestation_delta(attestation: str) -> float:
    """STIR/SHAKEN prior term in nats; unknown values get the C term."""
    return float(ATTESTATION_DELTA.get(str(attestation), 0.18))


def posterior(llr: float, prior: float, delta_att: float = 0.0) -> float:
    """Per-stream posterior, TDD eq. 17: sigma(llr + logit(prior) + delta)."""
    return float(_sigmoid(np.array(llr + _logit(prior) + delta_att)))


def joint_prob(p_v: float, p_s: float, rho: float) -> float:
    """P(V and S), TDD eq. 18: mixture of independence and redundancy."""
    rho = float(np.clip(rho, 0.0, 1.0))
    return float((1.0 - rho) * p_v * p_s + rho * min(p_v, p_s))


def noisy_or(p_v: float, p_s: float) -> float:
    """1 - (1 - pV)(1 - pS): the rho = 0 endpoint of the envelope."""
    return float(1.0 - (1.0 - p_v) * (1.0 - p_s))


def fused_threat(p_v: float, p_s: float, rho: float) -> float:
    """P(V union S), TDD eq. 19."""
    rho = float(np.clip(rho, 0.0, 1.0))
    return float((1.0 - rho) * noisy_or(p_v, p_s) + rho * max(p_v, p_s))


def severity(p_v: float, p_s: float, rho: float) -> float:
    """P(V and S) / p_threat: the share of the threat that is both streams.

    0 when there is no threat at all. Carried separately from the
    probability so the agent can escalate guidance on dual threats
    without the probability itself meaning something else.
    """
    p_threat = fused_threat(p_v, p_s, rho)
    if p_threat <= 0.0:
        return 0.0
    return float(joint_prob(p_v, p_s, rho) / p_threat)


def _threat_prior(prior_v: float, prior_s: float, rho: float) -> float:
    """Threat prior implied by the stream priors and dependence rho.

    pi_threat = pi_V + pi_S - P(V and S), with the joint term from
    eq. 18 evaluated at the priors themselves.
    """
    joint = (1.0 - rho) * prior_v * prior_s + rho * min(prior_v, prior_s)
    return float(np.clip(prior_v + prior_s - joint, _EPS, 1.0 - _EPS))


def _fused_llrs(
    p_v: Sequence[float],
    p_s: Sequence[float],
    rho: float,
    prior: Optional[float] = None,
    prior_v: Optional[float] = None,
    prior_s: Optional[float] = None,
) -> np.ndarray:
    """Fused threat as LLRs, for Cllr scoring of the OR label.

    The fused output is a posterior of threat (V union S), so its
    log-likelihood ratio is ``logit(p_threat) - logit(pi_threat)``:
    posterior odds are prior odds times the likelihood ratio, and the
    Cllr in ``fusion.calibration`` is defined on true LLRs. Using
    ``logit(p_threat)`` alone would smuggle the threat prior into the
    score and bias the rho fit toward whichever rho implies a prior
    closest to 0.5. The threat prior comes from the stream priors via
    :func:`_threat_prior` when they are given, else from the caller.
    Clipped to keep perfectly confident outputs finite.
    """
    p_v = np.asarray(p_v, dtype=float)
    p_s = np.asarray(p_s, dtype=float)
    rho = float(np.clip(rho, 0.0, 1.0))
    p_threat = np.clip(
        (1.0 - rho) * (1.0 - (1.0 - p_v) * (1.0 - p_s)) + rho * np.maximum(p_v, p_s),
        _EPS,
        1.0 - _EPS,
    )
    if prior_v is not None and prior_s is not None:
        pi = _threat_prior(float(prior_v), float(prior_s), rho)
    elif prior is not None:
        pi = float(np.clip(prior, _EPS, 1.0 - _EPS))
    else:
        pi = 0.5
    return np.log(p_threat / (1.0 - p_threat)) - np.log(pi / (1.0 - pi))


def fused_cllr(
    p_v: Sequence[float],
    p_s: Sequence[float],
    labels: Sequence[int],
    rho: float,
    prior: Optional[float] = None,
    prior_v: Optional[float] = None,
    prior_s: Optional[float] = None,
) -> float:
    """Cllr of the fused output at dependence ``rho`` (TDD eq. 10).

    ``prior`` is the threat prior for the LLR conversion; when neither
    it nor the stream priors are given, the empirical label rate of
    the scored set is used.
    """
    if prior is None and (prior_v is None or prior_s is None):
        labels_arr = np.asarray(labels, dtype=float)
        prior = float(np.clip(np.mean(labels_arr), _EPS, 1.0 - _EPS)) if len(labels_arr) else 0.5
    return cllr(labels, _fused_llrs(p_v, p_s, float(np.clip(rho, 0.0, 1.0)), prior, prior_v, prior_s))


@dataclass
class RhoFit:
    """Result of fitting rho on paired development calls."""

    rho: float
    cllr: float
    cllr_rho0: float
    cllr_rho1: float
    n: int


def fit_rho(
    p_v: Sequence[float],
    p_s: Sequence[float],
    labels: Sequence[int],
    grid: int = 101,
    prior_v: Optional[float] = None,
    prior_s: Optional[float] = None,
) -> RhoFit:
    """Fit rho in [0, 1] by minimising the fused Cllr.

    Cllr as a function of rho is not guaranteed unimodal, so the fit is
    a coarse grid followed by golden-section refinement inside the best
    bracket. The fitted value is therefore never worse than either
    endpoint (rho = 0 noisy-OR, rho = 1 max) on the fit data. When the
    stream priors are given, the threat prior for the LLR conversion
    is the model-implied one at each candidate rho; otherwise the
    empirical label rate of the fit set is used.
    """
    p_v = list(p_v)
    p_s = list(p_s)
    labels = list(labels)
    n = len(labels)
    if n == 0 or len(np.unique(labels)) < 2:
        return RhoFit(rho=0.0, cllr=1.0, cllr_rho0=1.0, cllr_rho1=1.0, n=n)

    def _score(rho: float) -> float:
        return fused_cllr(p_v, p_s, labels, rho, prior_v=prior_v, prior_s=prior_s)

    candidates = np.linspace(0.0, 1.0, max(int(grid), 2))
    scores = [_score(r) for r in candidates]
    best = int(np.argmin(scores))
    lo = float(candidates[max(best - 1, 0)])
    hi = float(candidates[min(best + 1, len(candidates) - 1)])
    # Golden-section refinement inside the best bracket.
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = _score(c)
    fd = _score(d)
    for _ in range(60):
        if abs(b - a) < 1e-6:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = _score(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = _score(d)
    rho = float(np.clip((a + b) / 2.0, 0.0, 1.0))
    fitted = _score(rho)
    if fitted > scores[best]:
        rho = float(candidates[best])
        fitted = float(scores[best])
    return RhoFit(
        rho=rho,
        cllr=float(fitted),
        cllr_rho0=float(scores[0]),
        cllr_rho1=float(scores[-1]),
        n=n,
    )


@dataclass
class OrFusionResult:
    """One tick of calibrated OR fusion."""

    p_v: float  # synthetic-voice posterior
    p_s: float  # fraud-script posterior
    p_threat: float  # P(V union S)
    severity: float  # P(V and S) / p_threat, carried separately
    joint: float  # P(V and S)
    rho: float
    attestation: str = "A"


class OrFusion:
    """Streaming calibrated OR fusion (TDD section 8.1).

    Per-stream LLRs are accumulated with leaky integration (beta per
    tick, capped) before conversion to posteriors, so a short burst of
    evidence is discounted over time rather than forgotten instantly
    or held forever.
    """

    def __init__(
        self,
        prior_v: float = DEFAULT_PRIOR_V,
        prior_s: float = DEFAULT_PRIOR_S,
        rho: float = 0.0,
        beta: float = DEFAULT_BETA,
        llr_cap: float = DEFAULT_LLR_CAP,
    ):
        self.prior_v = float(prior_v)
        self.prior_s = float(prior_s)
        self.rho = float(np.clip(rho, 0.0, 1.0))
        self.beta = float(beta)
        self.llr_cap = float(llr_cap)
        self._llr_v = 0.0
        self._llr_s = 0.0

    def reset(self) -> None:
        """Clear the leaky integrators (new call)."""
        self._llr_v = 0.0
        self._llr_s = 0.0

    def fit(self, p_v: Sequence[float], p_s: Sequence[float], labels: Sequence[int]) -> RhoFit:
        """Fit rho on paired development posteriors; stores and returns it.

        Uses this instance's deployment priors for the threat-prior
        conversion, so the fitted rho is the one the deployed model
        would actually use.
        """
        result = fit_rho(p_v, p_s, labels, prior_v=self.prior_v, prior_s=self.prior_s)
        self.rho = result.rho
        return result

    def update(
        self, llr_ac: float, llr_lang: float, attestation: str = "A"
    ) -> OrFusionResult:
        """One fusion tick from per-stream calibrated LLRs."""
        self._llr_v = float(np.clip(self.beta * self._llr_v + llr_ac, -self.llr_cap, self.llr_cap))
        self._llr_s = float(np.clip(self.beta * self._llr_s + llr_lang, -self.llr_cap, self.llr_cap))
        delta = attestation_delta(attestation)
        p_v = posterior(self._llr_v, self.prior_v)
        p_s = posterior(self._llr_s, self.prior_s, delta)
        p_threat = fused_threat(p_v, p_s, self.rho)
        return OrFusionResult(
            p_v=p_v,
            p_s=p_s,
            p_threat=p_threat,
            severity=severity(p_v, p_s, self.rho),
            joint=joint_prob(p_v, p_s, self.rho),
            rho=self.rho,
            attestation=str(attestation),
        )


@dataclass
class ShadowRow:
    """One paired tick scored by both fusions in shadow mode."""

    cscf_risk: float
    or_threat: float
    or_severity: float
    label: Optional[int] = None  # 1 = threat (V or S), 0 = benign


def shadow_compare(rows: Sequence[ShadowRow]) -> Dict[str, object]:
    """Compare CSCF (production) against OR fusion on recorded shadow rows.

    Returns per-system Cllr / EER / minDCF / actDCF on the labelled
    rows, plus the tier-disagreement rate at the CSCF thresholds
    (0.35 suspicious, 0.62 high risk). Rows without a label count
    toward disagreement only. No decision is taken here; this is the
    evidence gate G7 asks to be reviewed.
    """
    from ..eval.metrics import act_dcf, equal_error_rate, min_dcf

    rows = list(rows)
    labelled = [r for r in rows if r.label is not None]
    out: Dict[str, object] = {"n": len(rows), "n_labelled": len(labelled)}

    def _tier(x: float) -> str:
        if x >= 0.62:
            return "HIGH_RISK"
        if x >= 0.35:
            return "SUSPICIOUS"
        return "SAFE"

    if rows:
        out["disagreement_rate"] = float(
            np.mean([_tier(r.cscf_risk) != _tier(r.or_threat) for r in rows])
        )
    else:
        out["disagreement_rate"] = None

    if len(labelled) >= 2 and len(np.unique([r.label for r in labelled])) == 2:
        labels = [r.label for r in labelled]
        # Same prior-corrected LLR conversion as fused_cllr: posterior
        # odds are prior odds times the likelihood ratio, and the
        # empirical label rate is the prior of the shadow set.
        pi = float(np.clip(np.mean(labels), _EPS, 1.0 - _EPS))
        logit_pi = float(np.log(pi / (1.0 - pi)))
        for name, scores in (
            ("cscf", [r.cscf_risk for r in labelled]),
            ("or_fusion", [r.or_threat for r in labelled]),
        ):
            clipped = np.clip(scores, _EPS, 1.0 - _EPS)
            llrs = np.log(clipped / (1.0 - clipped)) - logit_pi
            out[f"{name}_cllr"] = float(cllr(labels, llrs))
            out[f"{name}_eer"] = float(equal_error_rate(labels, scores))
            out[f"{name}_min_dcf"] = float(min_dcf(labels, llrs))
            out[f"{name}_act_dcf"] = float(act_dcf(labels, llrs))
    else:
        for name in ("cscf", "or_fusion"):
            out[f"{name}_cllr"] = None
            out[f"{name}_eer"] = None
            out[f"{name}_min_dcf"] = None
            out[f"{name}_act_dcf"] = None
    return out


class ShadowFusion:
    """Run OR fusion in shadow beside a CSCF engine.

    Both engines get the same ticks. The CSCF result is returned
    untouched and is the only one the caller may act on; the OR
    result is recorded for later comparison through :func:`shadow_compare`.
    """

    def __init__(self, cscf: Optional[FusionEngine] = None, or_fusion: Optional[OrFusion] = None):
        self.cscf = cscf if cscf is not None else FusionEngine()
        self.or_fusion = or_fusion if or_fusion is not None else OrFusion()
        self.rows: List[ShadowRow] = []

    def reset(self) -> None:
        self.cscf.reset()
        self.or_fusion.reset()
        self.rows.clear()

    def update(
        self,
        acoustic: AcousticScore,
        linguistic: LinguisticScore,
        llr_ac: float,
        llr_lang: float,
        attestation: str = "A",
        label: Optional[int] = None,
    ) -> Tuple[float, OrFusionResult]:
        """Feed one tick to both engines; return (cscf risk, OR result)."""
        self.cscf.update_acoustic(acoustic)
        self.cscf.update_linguistic(linguistic)
        cscf_risk = self.cscf.fuse(timestamp_sec=acoustic.timestamp_sec).risk_score
        or_result = self.or_fusion.update(llr_ac, llr_lang, attestation)
        self.rows.append(
            ShadowRow(
                cscf_risk=float(cscf_risk),
                or_threat=float(or_result.p_threat),
                or_severity=float(or_result.severity),
                label=label,
            )
        )
        return float(cscf_risk), or_result

    def report(self) -> Dict[str, object]:
        """Shadow comparison over all recorded rows (see :func:`shadow_compare`)."""
        return shadow_compare(self.rows)

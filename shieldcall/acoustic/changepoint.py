"""Page CUSUM for an upward shift in a streaming acoustic score.

This is the 1954 cumulative-sum procedure, not a learned detector.
It answers: *when did production start looking more synthetic?*
Utterance-mean scores cannot answer that.

Reference: E. S. Page, Biometrika 41 (1954).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ChangeAlarm:
    timestamp_sec: float
    statistic: float
    pre_mean: float
    post_hint: float


class StreamingCUSUM:
    """One-sided CUSUM for an increase in ``x_t`` relative to a burn-in mean.

    Recursion: ``s_t = max(0, s_{t-1} + x_t - mu0 - k)``.
    Alarm when ``s_t >= h``. After an alarm the statistic is reset so a
    later second shift can still be seen.

    ``k`` is the slack (half the shift we want to detect). ``h`` is the
    threshold in cumulative-excess units. Neither is learned from test
    labels; they are constants. Burn-in frames estimate ``mu0`` only.
    """

    def __init__(
        self,
        k: float = 0.08,
        h: float = 0.40,
        burn_in: int = 8,
        min_alarm_gap_sec: float = 0.40,
        ignore_first_sec: float = 0.12,
    ):
        self.k = k
        self.h = h
        self.burn_in = burn_in
        self.min_alarm_gap_sec = min_alarm_gap_sec
        self.ignore_first_sec = ignore_first_sec
        self._buf: List[float] = []
        self._mu0: Optional[float] = None
        self._s_up = 0.0
        self._s_dn = 0.0
        self._last_alarm_t = -1e9
        self.alarms: List[ChangeAlarm] = []

    def reset(self) -> None:
        self._buf = []
        self._mu0 = None
        self._s_up = 0.0
        self._s_dn = 0.0
        self._last_alarm_t = -1e9
        self.alarms = []

    def update(self, x: float, timestamp_sec: float) -> Optional[ChangeAlarm]:
        x = float(x)
        if self._mu0 is None:
            self._buf.append(x)
            if len(self._buf) < self.burn_in:
                return None
            self._mu0 = float(sum(self._buf) / len(self._buf))
            return None

        self._s_up = max(0.0, self._s_up + x - self._mu0 - self.k)
        self._s_dn = max(0.0, self._s_dn + self._mu0 - self.k - x)
        s = max(self._s_up, self._s_dn)
        if timestamp_sec < self.ignore_first_sec:
            return None
        if s < self.h:
            return None
        if timestamp_sec - self._last_alarm_t < self.min_alarm_gap_sec:
            self._s_up = 0.0
            self._s_dn = 0.0
            return None
        alarm = ChangeAlarm(
            timestamp_sec=float(timestamp_sec),
            statistic=float(s),
            pre_mean=float(self._mu0),
            post_hint=x,
        )
        self.alarms.append(alarm)
        self._last_alarm_t = timestamp_sec
        self._s_up = 0.0
        self._s_dn = 0.0
        return alarm


def estimate_mean_shift_time(
    values: Sequence[float],
    times: Sequence[float],
    win: int = 16,
) -> Optional[Tuple[float, float]]:
    """Offline two-window mean-shift time.

    Returns (timestamp, abs_delta) for the index that maximises
    |mean(after) - mean(before)|. Needs at least 2*win+1 points.
    This is the batch analogue of CUSUM localisation; it does not
    run online.
    """
    n = len(values)
    if n < 2 * win + 1 or n != len(times):
        return None
    x = [float(v) for v in values]
    best_d = -1.0
    best_t: Optional[float] = None
    for i in range(win, n - win):
        before = sum(x[i - win : i]) / win
        after = sum(x[i : i + win]) / win
        d = abs(after - before)
        if d > best_d:
            best_d = d
            best_t = float(times[i])
    if best_t is None:
        return None
    return best_t, float(best_d)

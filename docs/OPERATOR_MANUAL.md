# Operator actions (one page)

The sidecar never drops RTP. These labels are **recommendations** on a
forked copy of the call.

| Action | What the user/operator sees | When the agent chooses it |
|--------|-----------------------------|---------------------------|
| MONITOR | Nothing | Benign mode, low threat mass |
| ABSTAIN | Nothing; interval spans SAFE and threat | ACI/heuristic band too wide |
| WARN | In-band caution to the callee; call continues | Human social-engineering hypothesis dominates. A nonce would pass. |
| CHALLENGE | Short liveness prompt (numbers). At most one per call. | Synthetic or unknown-family; the nonce is informative |
| ESCALATE | Hand-off to a human operator | High threat mass after the budget is spent, or handoff hypothesis |
| ADAPT | No user prompt; enroll a labelled embedding | Coverage gap high; operator/challenge provided a label |

**Do not** treat WARN as a block. **Do not** challenge a live social engineer.
**Do not** log the nonce in the audit export.

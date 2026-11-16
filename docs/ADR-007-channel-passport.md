# ADR-007: Channel Passport (blind channel estimation)

## Status

Accepted (2026-11-16). Implements TDD v2.0 §5 (Channel Passport) and the
stage-2 item "Stage 2c" in the backlog. Third slice of roadmap stage 2
("Baselines, calibration and fusion"), after ADR-005 (per-cell calibration)
and ADR-006 (calibrated OR fusion in shadow). The passport is the missing
input both of those consume: ADR-005 defined calibration keyed on a cell
label but had no way to obtain one from live audio, and the expert gating
of TDD §5.2 item 2 had no gate signal at all.

## Context

Phase 8 is measured and locked (11.78% pooled EER, 0.442 min t-DCF on the
full ASVspoof 2021 LA set), but the same model scores 9.09% EER on
uncompressed audio and 16.35% on PSTN-transmitted audio (TDD v2.0 §5).
A score means different things on different channels, so one global
threshold is wrong. The passport estimates the channel blindly from the
received audio and the frame envelope, then conditions calibration, model
choice and abstention on it. It emits one vector every 500 ms (TDD §5.1).

No part of this existed in the repo before this ADR: the channel twin
(`shieldcall/audio/channel.py`) simulates channels for evaluation but
estimates nothing from received audio, and the calibration service
(ADR-005) takes its cell key from the caller.

## Decision

1. **Module placement follows the TDD module map.** The passport lives
   in `shieldcall/acoustic/passport.py` (TDD §17 table). It uses numpy
   and scipy only; no new dependency. It imports nothing from the
   serving or fusion layers, so the scoring path stays dependency-light.
   The one deliberate import direction is outward: the passport feeds
   `PerCellCalibrator` (ADR-005) through a small helper, it does not
   depend on it.

2. **Passport contents (TDD §5.1).** `estimate_passport` returns a
   `PassportVector` with:
   - Declared codec and rate from the envelope (`declared_codec`,
     `declared_rate`): exact when present, `None` when absent. The
     passport never invents a declared codec.
   - Effective bandwidth `B̂`: the frequency below which 99% of the
     long-term spectral energy lies (TDD eq. 1), from the mean
     periodogram over 20 ms frames. This catches upstream transcoding
     the declared codec hides (a wideband call that crossed a
     narrowband trunk). White line noise raises the measured bandwidth;
     that is real behaviour, documented, not corrected away.
   - Companding signature: the audio is re-encoded through a μ-law and
     an A-law round trip (G.711, 256 levels) and the fraction of active
     samples reproduced exactly (within 1e-6) is reported per law; the
     better law wins and is named. A signal that was already companded
     upstream re-quantises losslessly, so its signature is near 1; a
     clean float signal is moved by up to half a quantisation step and
     scores near 0. Only samples with |x| > 0.02 count, and fewer than
     100 active samples means "no evidence" rather than a guessed law.
   - Loss and concealment rate from the envelope (sequence gaps plus
     concealed flags over packets). Without envelope packet counts the
     rate is 0.0 and is reported as unobserved, not measured.
   - SNR from voice-activity-separated frame energy (TDD eq. 1 form:
     `10 log10((E_speech - E_noise) / E_noise)`). Frame energies are
     split by percentile: noise is the mean of the lowest quintile of
     20 ms frame energies, speech is the mean of frames more than 6 dB
     above the noise floor (falling back to the top quintile when no
     frame clears it). This is an energy-percentile estimate, not a
     true VAD; it is labelled as such.
   - Clipping fraction: the share of samples with |x| >= 0.999.
   - Processing cue `noise_floor_modulation`: the coefficient of
     variation of the pause-frame energies across the window, where a
     pause frame is one within 6 dB of the estimated noise floor (the
     mean of the lowest quintile of frame energies). A steady line
     noise floor scores near 0; a gated noise suppressor makes the
     pause floor bursty and scores high. Echo-canceller residue
     needs a far-end reference signal, which a single-channel passport
     does not have; it is out of scope and stated as such, not
     approximated from the modulation cue.

3. **Cell mapping is a deterministic rule with documented precedence.**
   Cells are the TDD §5 example set: `clean_wideband`,
   `opus_wideband`, `narrowband_g711`, `pstn_degraded`, `amr_nb`,
   `low_snr`, `high_loss`, `unknown`. Precedence:
   1. `high_loss` when loss >= 0.05;
   2. `low_snr` when SNR < 10 dB;
   3. declared codec when present (opus -> `opus_wideband`, g711 /
      ulaw / alaw -> `narrowband_g711`, amr -> `amr_nb`, pstn ->
      `pstn_degraded`);
   4. otherwise bandwidth plus companding evidence: bandlimited
      (B̂ <= 4200 Hz) and companded -> `pstn_degraded`; wideband
      (B̂ > 6000 Hz) -> `clean_wideband`;
   5. anything else -> `unknown`. A bandlimited signal with no
      companding evidence and no declared codec is `unknown`: it could
      be AMR-NB, G.729 or Opus-NB, and the label never claims a codec
      it did not see.
   Loss and SNR take precedence because TDD §5 names them as cells in
   their own right: a call that is both PSTN and lossy is, for
   calibration purposes, a high-loss call first.

4. **Support score ρ is a kernel-density proxy, not a fitted density.**
   ρ is the maximum Gaussian kernel value of the passport's feature
   vector (log10 B̂, SNR, loss rate, companding score) against a small
   set of nominal reference points, one or more per known cell, with
   fixed scales (0.12 dex, 8 dB, 0.05, 1.0). The reference points are
   nominal values from the TDD and the Phase 8 conditions, labelled as
   such in the code; they are not fitted to data. Companding is
   deliberately weak in the kernel (scale 1.0) because post-companding
   line noise erases the exactness the signature relies on. Fitting
   them on real channel traffic is future work and is called out in
   Consequences.
   ρ is in [0, 1] by construction. A passport sitting on a reference
   point scores 1.0; a passport far from every reference scores near 0.

5. **Abstention (TDD §5.2 item 3, invariant I7).** When ρ < ρ_min
   (default 0.25, overridable) the acoustic stream reports "channel
   out of support" and contributes no evidence for synthesis; the
   language stream still runs. The default is a placeholder pending
   calibration on real data and is marked as such. The passport itself
   never suppresses anything: it reports the flag and the caller
   decides.

6. **Expert gating (TDD §5.2 item 2).** `gate_weights` returns a
   softmax over the known cells of the negative squared distance to
   each cell's nearest reference point, summing to 1. The mapped cell
   gets the largest weight when the passport sits near its reference.
   The experts themselves are not implemented here; the passport only
   supplies the gate signal the mixture needs.

7. **Calibration feed (TDD §5.2 item 1).** `calibrated_llr(passport,
   calibrator, score)` delegates to `PerCellCalibrator.llr` with
   `passport.cell`, so a raw acoustic score is mapped through the
   cell-specific affine map ADR-005 fitted. An `unknown` cell falls
   back to the pooled map through the calibrator's existing rule; the
   fallback stays visible via `fallback_cells`.

8. **Streaming shape.** `ChannelPassport` accumulates audio and emits
   one `PassportVector` per 500 ms window, matching the TDD cadence.
   Envelope data (declared codec/rate, packet counts) is fed in
   through `note_envelope` / `note_packets`; the estimator never
   assumes it.

9. **Telemetry (TDD §5.2 item 4).** `PassportVector.to_dict()` gives a
   JSON-safe record for the fleet channel-mix histograms. No audio is
   retained in the vector.

## Alternatives considered

**A learned channel classifier (e.g. a small neural net on LFCCs).**
Rejected for this slice: it needs labelled channel data that does not
exist in the repo, and the TDD's own evidence status says the passport
components are standard signal measurements. A classifier can replace
the rule mapper later without changing the vector shape; the cell set
and the support score are the contract, not the mapper.

**Estimating loss from the audio instead of the envelope.** Rejected:
packet loss with concealment is deliberately hard to see in the
waveform (the twin's PLC repeats the last packet), so an audio-only
loss estimate would be a guess. The envelope is the honest source; the
passport reports loss as unobserved when there is no envelope.

**One global support threshold per cell.** Deferred: ρ_min is a single
default until per-cell support is measured on real traffic.

**Fitting the reference points to ASVspoof 2021 LA codec conditions
now.** Blocked: no licensed ASVspoof data is present locally (the P2-2
data gap, unchanged). Nominal points keep the mechanism testable on
synthetic data; the acceptance test below is the check that must run
once data exists.

## Consequences

- `shieldcall/acoustic/passport.py` is new: `PassportVector`,
  `estimate_passport`, `ChannelPassport`, `gate_weights`,
  `calibrated_llr`, plus the reference points and cell rules.
  `shieldcall/acoustic/__init__.py` exports the new names.
- The acceptance criterion (TDD §5, Table 31) is now checkable in
  principle: feed passport cells into `per_cell_report` on ASVspoof
  2021 LA and require the best-to-worst codec actDCF spread to narrow
  by at least 20% without raising pooled minDCF. It cannot be run
  here yet: no licensed ASVspoof data is present locally.
- The mechanism is validated on synthetic data in
  `tests/test_channel_passport.py`, with known ground truth for every
  measurement (bandwidth of a bandlimited signal, SNR of added noise,
  companding of a μ-law round trip, loss from envelope counts).
- Nothing in this change alters any live decision path: CSCF remains
  the production fusion, no threshold in the sidecar or app changes,
  and abstention is reported, not enforced, by the passport.
- Follow-ups (not this slice): fit the reference points and ρ_min on
  real channel traffic; wire the passport into the sidecar ingest so
  live calls carry a cell; feed passport cells to the stage-2a
  calibrator in the evaluation harness; implement the channel-expert
  mixture the gate weights feed. One measured limit to carry forward:
  post-companding line noise erases the exactness the companding
  signature relies on, so undeclared PSTN is recognised from the
  signature only on clean lines; the declared-codec path is the
  reliable route for cell labelling.

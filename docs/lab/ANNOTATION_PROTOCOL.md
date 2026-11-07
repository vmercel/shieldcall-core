# Annotation protocol (Phase 1)

**Audio labels.** No new human annotation. Labels are the ASVspoof 2019 organisers' protocol keys: `bonafide` (genuine human speech) or `spoof` plus the system id A01-A19. Positive class = spoof.

**Text labels.** Taken from the BothBosu dataset (`label` 1 = scam, 0 = non-scam; `type` = scenario). They were assigned by the generating prompt, not by human review. A 100-item human spot check has **not** been done; do not describe the labels as human-verified.

**Definitions used by the scorers.**
- *Scam call*: the caller tries to obtain money, credentials or personal identifiers under a pretext (impersonation, refund, prize, tech support, urgency).
- *Benign call*: delivery notice, insurance quote, telemarketing offer or wrong number with no such request. Telemarketing is deliberately included as a hard negative.
- *Synthetic voice*: speech produced or converted by one of the 19 ASVspoof LA systems.

**If human annotation is added later.** Two annotators per call label scam / not scam / unsure; disagreements are adjudicated by a third; report Cohen's kappa; annotators never see detector scores. Items changed after a detector has been run on them are logged in `docs/lab/CONTAMINATION.md` and removed from confirmatory tables.

**Splits and locking.** See `data/DATA_CARD.md`. Rule: thresholds, calibration and model fitting use train/dev only; the test split is scored once per configuration. The hash of the test audio is in `docs/results/phase1_lock.json`.

**Reproduce.**
```bash
pip install -r requirements.txt av pandas
curl -L -o LA.zip https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip   # ~7.6 GB, GET (HEAD returns HTML)
python scripts/phase1_build_dataset.py --zip LA.zip
python scripts/phase2_acoustic.py && python scripts/phase2_language.py
python scripts/phase2_repo_acoustic.py
python scripts/phase2_report.py && python scripts/phase2_tables.py
```
Runtime on one CPU: about 10 min (acoustic), 1.5 min (language), about 20 min (repository scorer subset).

# Record the live-call MVP

This is the walkthrough for a screen-recorded video of ShieldCall detection and analysis on a consented live call.

The lab is a sidecar. It does not sit in the telephone path. It never hangs up. If the Python process dies, the call continues (fail-open).

## What the video should show

1. Detector Lab opens at `http://127.0.0.1:8765`.
2. Consent is checked. A live session starts.
3. Microphone listen is on. A phone on speaker (or two people in the room) produces a live call.
4. The **linguistic** meter moves when vishing language is recognized.
5. The **acoustic** meter moves from live PCM (vocoder / synthetic artifacts when present).
6. **Fused risk**, regime, and a recommend-only action (`MONITOR` / `WARN` / `CHALLENGE` / `ESCALATE`) update on screen.
7. Optional A/B: inject `ind_b01` (dentist, safe) then `ind_s01` (grandparent bond, social engineering). Fraud should rise on the second fixture.

Do not claim the system hung up a carrier call. It did not.

## Run

```bash
cd shieldcall-core
pip install -r requirements-serve.txt
python scripts/run_sidecar.py
```

Open Chrome at http://127.0.0.1:8765

Use Chrome for Web Speech. Safari and Firefox may still send PCM but will not auto-transcribe.

## Live call setup

- Place a real, consented call on a phone.
- Speakerphone, laptop a foot away.
- Click **Start live session**, then **Listen on microphone**.
- If speech recognition misses a sentence, paste it into **Push turn**.

## After recording

Host the clip (YouTube unlisted, or a GitHub Release asset) and put the URL in the README "Recorded walkthrough" section.

Keep the recording short (60–90 seconds). Show consent, listen, one safe turn, one social-engineering turn, meters moving, stop.

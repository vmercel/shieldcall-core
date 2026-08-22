# Contamination log

A test utterance is **contaminated** if it influenced a regex, a floor, or a hyperparameter after the lexicon lock. Contaminated IDs are excluded from confirmatory tables.

## Lock

- Date: 2026-08-22
- Object: `STAGE_EMISSIONS` and `PATTERN_GROUPS` as of this commit
- Rule: no new regex because a test line missed. If a regex is added, the triggering ID is listed here and dropped from confirmatory splits.

## Known contamination (pre-lock; not used as confirmatory)

| ID | Split | Why |
|----|-------|-----|
| `h01`–`h10` in `vishing_scripts.py` | author held-out | Written against the same author’s broader lexicon as `STAGE_EMISSIONS` (`tax bureau`, `benefits integrity`, `card services`, `helpdesk`, `public defender`, `enrollment desk`, `cloud account`, `power cooperative`, `parcel claims`, `prosecutor`). These remain a **sanity** split only. |
| Isolated-keyword traps `t01`– in author scripts | author | Written to be easy traps. |

## Independent set

`shieldcall/eval/corpora/independent_scripts.py` was drafted **after** the lock, from public FTC/CISA impersonation tropes, without adding regexes. If a later commit adds a regex because an `ind_*` line missed, that `ind_*` ID is contaminated.

## Current confirmatory exclusions

None of `ind_*` are contaminated at freeze.

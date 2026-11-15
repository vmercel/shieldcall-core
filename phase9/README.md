# Phase 9: frame-level temporal head

Run (needs about 30 GB free disk; resumable; several hours on a laptop):

    git pull && bash phase9/run.sh smoke     # a few minutes, checks the whole pipeline at toy scale
    caffeinate -i bash phase9/run.sh full    # extraction, selection (2019 dev only), final fit, ASVspoof 2021 LA scoring

Upload `phase9_report_full.zip`. See PROTOCOL.md for the pre-registered design.

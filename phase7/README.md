# Phase 7: ASVspoof 2021 LA (telephony codecs), fixed recipe

```bash
git clone https://github.com/vmercel/shieldcall-core && cd shieldcall-core
bash phase7/run.sh smoke     # 2-5 min, checks everything, tiny download
bash phase7/run.sh sample    # ~14k trials; downloads the 7.8 GB eval audio once
bash phase7/run.sh full      # all 148,176 trials; resumable, can take several hours
```
Each run ends by printing the path of `phase7_report_<mode>.zip`. Send that file back.
Needs: python3, git, curl, about 12 GB free disk. If a run stops, run the same command again; it resumes.
See PROTOCOL.md for what is fixed and what is reported.

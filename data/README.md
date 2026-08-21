# Data (not in git)

Mini LibriSpeech `dev-clean-2` (OpenSLR 31):

```bash
python scripts/download_speech.py
# or
curl -L -o data/dev-clean-2.tar.gz http://www.openslr.org/resources/31/dev-clean-2.tar.gz
tar -xzf data/dev-clean-2.tar.gz -C data
```

ASVspoof is **not** redistributed. If you have a licensed copy:

```bash
export SHIELDCALL_ASVSPOOF_ROOT=/path/to/ASVspoof2019
```

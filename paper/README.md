# Journal manuscript

**Target (not printed in the PDF):** Paper B to Information Fusion or IEEE TASLP. Paper A (fast path) Interspeech/ICASSP/Odyssey. Do **not** submit to Computers \& Security (AI/ML moratorium since 2024).

The manuscript states measured claims only. Do not add ASVspoof SOTA language. Confirmatory tables come from `docs/results/upgrade_experiments.json`.

Source: `main.tex`, `references.bib`, `figures/`. Compiled: `main.pdf`.

## Compile

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Needs `natbib`, `booktabs`, `geometry`, `hyperref`, `amsmath`, `setspace`.

## What the paper claims

See Highlights in `main.tex`. Negative results (LPC at chance, SAPC audio failure) are part of the contribution. Removing them would make the paper less true and less journal-grade.

## Elsevier extras (at submission)

- Graphical abstract: `figures/graphical_abstract.pdf` (PNG sibling for web)
- Highlights: five bullets in the manuscript front matter
- CRediT and competing-interest statements are in the `.tex`

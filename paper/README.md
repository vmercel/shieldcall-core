# Journal manuscript

**Intended venue:** Computers \& Security (Elsevier), Q1. Alternative: Information Fusion.

This is **not** an arXiv-first note. The manuscript states measured claims only. Do not add ASVspoof SOTA language.

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

- Graphical abstract: `figures/graphical_abstract.png`
- Highlights: five bullets in the manuscript front matter
- CRediT and competing-interest statements are in the `.tex`

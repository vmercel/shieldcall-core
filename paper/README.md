# arXiv draft

Source: `main.tex`, `references.bib`, `figures/`.  
Compiled PDF: `main.pdf` (4 pages).

## Compile

Needs a TeX install with `natbib`, `booktabs`, `geometry`, `hyperref`, `amsmath`:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Submit

1. Create an arXiv account: https://arxiv.org/user/register
2. Category: **cs.SD** (primary), cross-list **eess.AS** and **cs.CR**
3. Upload `main.tex`, `references.bib`, `main.bbl` (after bibtex), and `figures/`
4. License: arXiv perpetual non-exclusive or CC BY 4.0
5. After announcement, put the arXiv id in the repository README

Do not claim ASVspoof SOTA in the abstract on arXiv. The current abstract already forbids that.

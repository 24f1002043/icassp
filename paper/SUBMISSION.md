# ICASSP 2027 submission checklist

Conference: IEEE ICASSP 2027, Toronto, 16–21 May 2027.
Paper kit: <https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php>

`spconf.sty` and `IEEEbib.bst` in this directory are the official files from
that kit, downloaded unmodified from
`https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/`. The full kit,
including `Template.tex`, `Template.pdf` and the Word template, is kept under
`official_kit/` for reference.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Verified automatically

| Requirement | Status |
|---|---|
| Max 5 pages | 5 pages |
| 5th page: references / acknowledgements / ethics only | page 4 ends the technical content; page 5 holds only the ethics statement and references |
| Letter, 8.5 × 11 in | 612 × 792 pt |
| Two columns, 86 mm each | via `spconf.sty` |
| Times-Roman, ≥ 9 pt | `\ninept`; Nimbus Roman + STIX maths |
| All fonts embedded | every font reports `emb yes` under `pdffonts` |
| File size ≤ 5 MB | ~0.22 MB |
| No page numbers | `spconf.sty` suppresses them |
| Not blind — author list on the PDF | author and affiliation on page 1 |
| Abstract 100–150 words | 149 |
| Discussion of relation to prior work | Section 2, its own numbered section |
| Compliance with Ethical Standards statement | before the references |

| ORCiD on the PDF | `0009-0005-8766-7012`, as a footnote on page 1 |

## Still to do before you upload

1. **Enter the ORCiD on the submission form too.** It is on the PDF, but CMS
   checks the form; a missing one there withdraws the paper automatically. Also
   verify one of your ORCiD email addresses — the record still shows both as
   unverified, which blocks you from editing the record itself.
2. **Repository link.** `main.tex` carries a footnote reading "Repository link
   to be inserted on acceptance." Either push this repository and paste the URL,
   or leave the placeholder — but do not forget it in the camera-ready.
3. **Affiliation.** Check the department line is how you want it to appear.
4. **Author list must match the submission form exactly.**
5. Re-run `python src/09_verify.py` after any change to the pipeline; it checks
   the headline numbers, that every posterior is a valid distribution, that no
   actor spans two cross-validation folds, and that all paper artefacts exist.

## Suggested reviewers / related groups

The work sits closest to Emily Mower Provost (Michigan) on annotator modelling
and label uncertainty, Carlos Busso (MSP-Podcast, MSP-IMPROV) on corpora built
around perceptual consensus, and Shrikanth Narayanan (USC SAIL). All three are
cited.

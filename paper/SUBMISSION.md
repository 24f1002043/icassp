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
| File size ≤ 5 MB | ~0.25 MB |
| No page numbers | `spconf.sty` suppresses them |
| Not blind: author list on the PDF | author and affiliation on page 1 |
| Abstract 100–150 words | 146 |
| Discussion of relation to prior work | Section 2, its own numbered section |
| Compliance with Ethical Standards statement | before the references |

| Authors on the PDF | Abhiram Radhakrishnan, Gengaraj P, Praveen V, Sri Sairam Engineering College |
| Repository link | footnote 2 on page 1: <https://github.com/24f1002043/icassp> |

## Still to do before you upload

1. **ORCiD for all three authors.** Abhiram Radhakrishnan: `0009-0005-8766-7012`. Gengaraj P
   and Praveen V each need their own (free, <https://orcid.org/register>). All
   three go into the CMS form; a missing one withdraws the paper automatically.
   Verify an email on each ORCiD record.
2. **Repository link.** The footnote points at
   <https://github.com/24f1002043/icassp>. Make sure the repository is public and
   pushed before you submit, or reviewers will hit a 404.
3. **Upload `radhakrishnan.pdf`**, not `main.pdf`, since the kit asks for the first
   author's last name as the filename. Paste title, authors, keywords and the
   ASCII abstract from `submission_form.txt`.
4. **LLM policy.** ICASSP 2027 forbids submitting a manuscript that an LLM
   generated in whole or in significant part, and requires authors to have
   verified all LLM output. The authors must rewrite the text in their own words
   and re-check every number (`bash scripts/run_all.sh`, `src/09_verify.py`)
   before submitting, and disclose AI assistance as IEEE's AI-generated-text
   guidelines require.
4. **Author list must match the submission form exactly.**
5. Re-run `python src/09_verify.py` after any change to the pipeline; it checks
   the headline numbers, that every posterior is a valid distribution, that no
   actor spans two cross-validation folds, and that all paper artefacts exist.

## Suggested reviewers / related groups

The work sits closest to Emily Mower Provost (Michigan) on annotator modelling
and label uncertainty, Carlos Busso (MSP-Podcast, MSP-IMPROV) on corpora built
around perceptual consensus, and Shrikanth Narayanan (USC SAIL). All three are
cited.

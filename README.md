# Designing Benchmarks for Knowledge Work

This repository contains the paper, companion atlas, and submission packets for
**Designing Benchmarks for Knowledge Work**.

The paper argues that knowledge-work agent benchmarks should make four choices
explicit: the represented work activity, the tested setting, the required work
product, and the evaluated result. It derives an 18-activity inventory from
O\*NET task statements and applies the representation to GDPval, OfficeQA Pro,
and APEX-SWE.

Paper (arXiv v1; v2 source in this repo):

<https://arxiv.org/abs/2605.23262>

Companion HTML atlas:

- Local: [`index.html`](index.html)
- Public: <https://ningkko.github.io/knowledge-work-position/index.html>

## Repository layout

| Path | Contents |
| --- | --- |
| `arxiv/` | Named arXiv v2 LaTeX source (authors visible; `tmlr` preprint option) |
| `tmlr/` | Anonymous TMLR-formatted LaTeX source |
| `packets/` | Exact upload zips for arXiv replace and TMLR |
| `src/` | Companion HTML figure assets |
| `index.html` | Interactive O\*NET / ESCO atlas |
| `archive/unsubmitted_rebuttal/` | Unique TMLR rebuttal from the older underscore checkout; not submitted |

Do not link the named GitHub or arXiv pages from the TMLR OpenReview form.
TMLR is double-blind; the `tmlr/` source stays anonymous.

## Work-activity reference

The inventory is derived from [O\*NET](https://www.onetcenter.org/database.html)
30.2 task statements in Job Zones 3–5. After screening, the reporting corpus
contains 12,464 task statements. A stricter atlas subset contains 8,372
statements.

[ESCO](https://esco.ec.europa.eu/en) is used as an external legibility check.
The machine-readable 18-activity vocabulary and validation summary are in
`arxiv/supplementary/` and `tmlr/supplementary/`.

## Citation

```bibtex
@article{hua2026designing,
  title={Designing Benchmarks for Knowledge Work},
  author={Hua, Yining and Na, Hongbin and Ayubcha, Cyrus and Lian, Levi},
  journal={arXiv preprint arXiv:2605.23262},
  year={2026}
}
```

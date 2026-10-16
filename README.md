# Designing Benchmarks for Knowledge Work

Companion data and interactive atlas for
[Designing Benchmarks for Knowledge Work](https://arxiv.org/abs/2605.23262).

The work-activity inventory is derived from [O\*NET](https://www.onetcenter.org/database.html)
30.2 task statements in Job Zones 3–5 (12,464 task statements after screening;
8,372 in the stricter atlas subset). [ESCO](https://esco.ec.europa.eu/en) is
used as an external legibility check.

## Contents

| Path | Contents |
| --- | --- |
| `index.html` | Interactive O\*NET / ESCO work-activity atlas ([live](https://ningkko.github.io/knowledge-work-position/index.html)) |
| `data/activity_inventory_18.csv` | The 18-activity vocabulary: corpus counts, definitions, input/output fields, distinct-from contrasts |
| `data/validation_summary.csv` | Summary of the five validation checks |
| `src/` | Figure assets used by the atlas |

## Citation

```bibtex
@article{hua2026designing,
  title={Designing Benchmarks for Knowledge Work},
  author={Hua, Yining and Na, Hongbin and Ayubcha, Cyrus and Lian, Levi},
  journal={arXiv preprint arXiv:2605.23262},
  year={2026}
}
```

## Reproduction code

The [reproduction/](reproduction/README.md) directory contains the Python and shell scripts for O*NET processing, taxonomy construction, ESCO analysis, benchmark coding, and figure generation. It also includes the curated taxonomy and small derived data tables.

The research code was developed across several iterations. Some steps require an OpenAI API key and generated intermediate files; the end-to-end pipeline has not been independently rerun as part of this release. The public atlas and two CSV files above remain unchanged.

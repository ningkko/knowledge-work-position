# Reproduction materials

Companion research scripts and selected data for *Designing Benchmarks for Knowledge Work* ([paper](https://arxiv.org/abs/2605.23262)). These are the original analysis scripts from the research project, organized to make the methods inspectable. They include exploratory and earlier-version entrypoints alongside the finalized 18-activity taxonomy.

## Files

- `scripts/`: O*NET processing, knowledge-work screening, embeddings, clustering, taxonomy construction, benchmark coding, and visualizations.
- `scripts/esco/`: ESCO preparation, embedding, independent clustering, mapping, robustness checks, and figures.
- `data/derived/manual_labeling/operation_taxonomy.json`: curated 18-operation taxonomy; `cluster_cross_job_labels.json`: cluster-level coding.
- `data/`: selected authored benchmark labels, frozen labels, validation summaries, and third-party data attribution.

The public release at repository root also provides `data/activity_inventory_18.csv`, `data/validation_summary.csv`, and the [interactive atlas](../index.html).

## Requirements

Use Python 3 and install the Python dependencies:

```bash
cd reproduction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The original analysis used O*NET Database 30.2 and ESCO v1.1.1. See `data/NOTICE` for attribution. Scripts using OpenAI embeddings or LLM classification need `OPENAI_API_KEY` set in your environment. Depending on the chosen analysis path, additional optional dependencies or external benchmark datasets may be required.

## Entry points

The O*NET source-data download and initial preparation can be started with:

```bash
bash scripts/download_onet_core.sh
python scripts/prepare_onet_tasks.py --job-zones 3 4 5 --write-csv
```

Subsequent analysis is split into scripts because taxonomy development used iterative clustering, manual review, and model-assisted screening. Relevant entrypoints are:

| Stage | Script(s) |
| --- | --- |
| Task-level screening | `round2_knowledge_screen.py` |
| Embeddings and clustering | `embed_onet_tasks.py`, `cluster_onet_tasks.py` |
| Operation taxonomy and task assignment | `build_operation_hierarchy.py`, `finalize_work_patterns.py`, `code_onet_tasks.py` |
| Atlas | `build_operation_atlas.py` |
| Benchmark coding and coverage | `build_benchmark_inventory.py`, `build_comparison.py` |
| ESCO comparison | `scripts/esco/` (see `data/derived/esco/README.md`) |

For the ESCO workflow, follow the command sequence in `data/derived/esco/README.md` after preparing the O*NET embeddings, clusters, and taxonomy. The benchmark download/distillation scripts are `scripts/download_benchmarks.sh` and `scripts/distill_benchmark_corpus.py`.

## Reproducibility and release scope

The repository contains scripts and selected small derived artifacts, including taxonomy specifications. Large embedding arrays, raw O*NET and ESCO downloads, benchmark corpus downloads, generated caches, and some intermediate analysis outputs are omitted. The source scripts may expect intermediate artifacts that must be regenerated or recovered from prior pipeline steps.

The scripts reflect successive development versions, including earlier labels and screening approaches. They are not a single verified, turnkey pipeline for regenerating every published result. The full end-to-end reproduction and agreement with the final manuscript have not been tested in this release. Consult the script documentation and compare outputs with the two public summary CSVs before reporting reproduced results.

The repository's Apache-2.0 code license does not replace the applicable attribution and reuse conditions for O*NET, ESCO, or external benchmark data; see `data/NOTICE`.

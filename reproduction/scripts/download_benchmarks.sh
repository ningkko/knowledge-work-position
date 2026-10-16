#!/usr/bin/env bash
# Download rubrics/tasks/datasheets for benchmarks referenced in the Knowledge
# Work Bench paper. Downloads go to `data/raw/benchmarks/<name>/`, which is
# gitignored (see `.gitignore`). Re-run idempotently.
#
# See `docs/benchmarks_downloads.md` for the full inventory, license notes,
# and gaps. Sources are each benchmark's own README/LICENSE — follow their
# citation and redistribution terms.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO_ROOT}/data/raw/benchmarks"
mkdir -p "${OUT}"

say() { printf "\n=== %s ===\n" "$*"; }

# Optional: rehydrate from the private HF mirror before doing any upstream
# fetching. This is dramatically faster than re-cloning ~13 GB of vendor
# repos and avoids per-user TOS gating for datasets we've already mirrored.
# Usage:
#   HF_MIRROR=kimmich07/kwbench-corpus HF_TOKEN=hf_... scripts/download_benchmarks.sh
# The upstream commands below are still run afterwards — they're idempotent
# (clone_shallow / fetch / hf_snapshot all skip when the target exists), so
# anything the mirror is missing will be pulled fresh from the origin.
if [[ -n "${HF_MIRROR:-}" ]]; then
  say "Rehydrating from HF mirror ${HF_MIRROR}"
  python3 - "$HF_MIRROR" "$OUT" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo_id, dest = sys.argv[1:]
try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dest,
        token=os.environ.get("HF_TOKEN"),
        max_workers=4,
    )
    print(f"[ok] rehydrated {repo_id} into {dest}")
except Exception as e:
    print(f"[warn] HF mirror pull failed ({type(e).__name__}: {e}); will fall back to upstream", file=sys.stderr)
PY
fi

clone_shallow() {
  # clone_shallow <dest> <url> [extra git flags...]
  local dest="$1" url="$2"
  shift 2
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" fetch --depth 1 origin >/dev/null 2>&1 || true
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  git clone --depth 1 "$@" "${url}" "${dest}"
}

fetch() {
  # fetch <dest-file> <url>
  local dest="$1" url="$2"
  mkdir -p "$(dirname "${dest}")"
  if [[ ! -s "${dest}" ]]; then
    curl -fsSL -A "Mozilla/5.0" -o "${dest}" "${url}"
  fi
}

fetch_raw() {
  # fetch_raw <dest-dir> <raw-url-prefix> <file1> [file2...]
  local dest="$1" prefix="$2"
  shift 2
  mkdir -p "${dest}"
  for f in "$@"; do
    if [[ ! -s "${dest}/${f}" ]]; then
      curl -fsSL -o "${dest}/${f}" "${prefix}/${f}" || echo "  [warn] missing ${f}"
    fi
  done
}

hf_snapshot() {
  # hf_snapshot <repo_id> <repo_type=dataset|model> <local_dir> [allow_patterns...]
  local repo_id="$1" repo_type="$2" dest="$3"
  shift 3
  python3 - "$repo_id" "$repo_type" "$dest" "$@" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo_id, repo_type, dest, *patterns = sys.argv[1:]
kwargs = dict(repo_id=repo_id, repo_type=repo_type, local_dir=dest)
if patterns:
    kwargs["allow_patterns"] = list(patterns)
if os.environ.get("HF_TOKEN"):
    kwargs["token"] = os.environ["HF_TOKEN"]
try:
    snapshot_download(**kwargs)
except Exception as e:
    print(f"[warn] hf snapshot_download failed for {repo_id}: {e}", file=sys.stderr)
    sys.exit(0)
PY
}

###############################################################################
# Priority — user-named
###############################################################################
say "LegalBench"
clone_shallow "${OUT}/legalbench/repo"   https://github.com/HazyResearch/legalbench.git
hf_snapshot   nguha/legalbench dataset  "${OUT}/legalbench/hf"
fetch         "${OUT}/legalbench/paper_2308.11462.pdf" https://arxiv.org/pdf/2308.11462

say "FinanceBench"
clone_shallow "${OUT}/financebench/repo" https://github.com/patronus-ai/financebench.git
hf_snapshot   PatronusAI/financebench dataset "${OUT}/financebench/hf"
fetch         "${OUT}/financebench/paper_2311.11944.pdf" https://arxiv.org/pdf/2311.11944

say "GDPval"
# Only pull task metadata + rubrics + deliverable-file URIs. Skip bulk .xlsx/.pptx/.docx
# reference/deliverable binaries (~2.3GB) — they are referenced by URL in the parquet.
hf_snapshot openai/gdpval dataset "${OUT}/gdpval/hf_meta" \
  "README*" "*.md" "*.json" "*.jsonl" "*.csv" "*.parquet" "*.py" "rubrics/**" "data/train-*"
fetch "${OUT}/gdpval/paper_2510.04374.pdf" https://arxiv.org/pdf/2510.04374

say "OfficeQA"
# Clone is ~4GB+ (includes treasury_bulletin_pdfs/ 20GB via history). Fetch only
# the files we care about via raw.githubusercontent. Run the upstream clone
# separately if you need the Treasury Bulletin source PDFs.
fetch_raw "${OUT}/officeqa/repo" \
  https://raw.githubusercontent.com/databricks/officeqa/main \
  README.md officeqa_pro.csv officeqa_full.csv reward.py logo.png
fetch "${OUT}/officeqa/paper_2603.08655.pdf" https://arxiv.org/pdf/2603.08655

say "MADQA"
clone_shallow "${OUT}/madqa/repo" https://github.com/OxRML/MADQA.git
hf_snapshot   OxRML/MADQA dataset "${OUT}/madqa/hf"   # gated — needs HF_TOKEN + TOS
fetch         "${OUT}/madqa/paper_2603.12180.pdf" https://arxiv.org/pdf/2603.12180

say "APEX family (Mercor)"
clone_shallow "${OUT}/apex_agents/apex_swe_repo"    https://github.com/Mercor-Intelligence/apex-swe.git
clone_shallow "${OUT}/apex_agents/archipelago_repo" https://github.com/Mercor-Intelligence/archipelago.git
hf_snapshot   mercor/APEX-v1-extended dataset "${OUT}/apex_agents/apex_v1_extended_hf"
hf_snapshot   mercor/apex-agents dataset "${OUT}/apex_agents/apex_agents_hf"   # gated — needs HF_TOKEN + TOS
fetch "${OUT}/apex_agents/paper_2509.25721.pdf"          https://arxiv.org/pdf/2509.25721
fetch "${OUT}/apex_agents/paper_apex_agents_2601.14242.pdf" https://arxiv.org/pdf/2601.14242
fetch "${OUT}/apex_agents/paper_apex_swe_2601.08806.pdf"    https://arxiv.org/pdf/2601.08806
fetch "${OUT}/apex_agents/apex_overview.html"   https://www.mercor.com/apex/
fetch "${OUT}/apex_agents/apex_blog.html"       https://www.mercor.com/blog/introducing-apex-ai-productivity-index
fetch "${OUT}/apex_agents/apex_agents_blog.html" https://www.mercor.com/blog/introducing-apex-agents/

###############################################################################
# Secondary — in paper's `data/benchmarks.csv`
###############################################################################
say "OfficeBench"
clone_shallow "${OUT}/officebench/repo" https://github.com/zlwang-cs/OfficeBench.git
fetch "${OUT}/officebench/paper_2407.19056.pdf" https://arxiv.org/pdf/2407.19056

say "GAIA"
hf_snapshot gaia-benchmark/GAIA dataset "${OUT}/gaia/hf"   # gated — needs HF_TOKEN + TOS
fetch "${OUT}/gaia/paper_2311.12983.pdf" https://arxiv.org/pdf/2311.12983

say "WebArena"
clone_shallow "${OUT}/webarena/repo" https://github.com/web-arena-x/webarena.git
fetch "${OUT}/webarena/paper_2307.13854.pdf" https://arxiv.org/pdf/2307.13854

say "WorkArena"
clone_shallow "${OUT}/workarena/repo" https://github.com/ServiceNow/WorkArena.git
fetch "${OUT}/workarena/paper_2403.07718.pdf" https://arxiv.org/pdf/2403.07718

say "OSWorld"
clone_shallow "${OUT}/osworld/repo" https://github.com/xlang-ai/OSWorld.git --filter=blob:limit=10m
fetch "${OUT}/osworld/paper_2404.07972.pdf" https://arxiv.org/pdf/2404.07972

say "BrowseComp"
clone_shallow "${OUT}/browsecomp/simple_evals_repo" https://github.com/openai/simple-evals.git
fetch "${OUT}/browsecomp/browse_comp_test_set.csv"  https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv
fetch "${OUT}/browsecomp/paper_2504.12516.pdf"      https://arxiv.org/pdf/2504.12516

say "MedAgentBench"
clone_shallow "${OUT}/medagentbench/repo" https://github.com/stanfordmlgroup/MedAgentBench.git
fetch "${OUT}/medagentbench/paper_2501.14654.pdf" https://arxiv.org/pdf/2501.14654

###############################################################################
# Extended — referenced in workshop / paper adjacent
###############################################################################
say "TheAgentCompany"
clone_shallow "${OUT}/theagentcompany/repo" https://github.com/TheAgentCompany/TheAgentCompany.git
fetch "${OUT}/theagentcompany/paper_2412.14161.pdf" https://arxiv.org/pdf/2412.14161

say "tau-bench / tau2-bench"
clone_shallow "${OUT}/tau_bench/repo"      https://github.com/sierra-research/tau-bench.git
clone_shallow "${OUT}/tau_bench/tau2_repo" https://github.com/sierra-research/tau2-bench.git || true
fetch "${OUT}/tau_bench/paper_2406.12045.pdf" https://arxiv.org/pdf/2406.12045

say "KWBench"
fetch "${OUT}/kwbench/site.html"             https://kwbench.github.io/
fetch "${OUT}/kwbench/paper_2604.15760.pdf"  https://arxiv.org/pdf/2604.15760

say "Spider 2.0"
clone_shallow "${OUT}/spider2/repo" https://github.com/xlang-ai/Spider2.git --filter=blob:limit=10m
fetch "${OUT}/spider2/paper_2411.07763.pdf" https://arxiv.org/pdf/2411.07763

###############################################################################
# Benchmarks cited in frontier-lab model cards (docs/frontier_lab_benchmark_usage.md)
###############################################################################
say "Finance Agent (Vals AI)"
# Cited in OpenAI GPT-5.5 launch as "FinanceAgent" (60.0%). 187 public tasks
# with ground-truth answers + per-criterion rubrics in data/public.csv
# (the full 537-task set is gated behind the Vals platform).
clone_shallow "${OUT}/finance_agent/repo" https://github.com/vals-ai/finance-agent.git
fetch "${OUT}/finance_agent/paper_2508.00828.pdf" https://arxiv.org/pdf/2508.00828

say "Vending-Bench 2 (Andon Labs)"
# Cited in Gemini 3 Pro eval + Claude Opus 4.7 system card. No public task
# dataset / rubric file — scoring = bank balance after simulated year. We
# archive the paper + leaderboard HTML for the manifest.
fetch "${OUT}/vending_bench_2/paper_2502.15840.pdf"   https://arxiv.org/pdf/2502.15840
fetch "${OUT}/vending_bench_2/vending_bench_2.html"   https://andonlabs.com/evals/vending-bench-2
fetch "${OUT}/vending_bench_2/vending_bench_v1.html"  https://andonlabs.com/evals/vending-bench

say "Terminal-Bench 2.0 (Laude Institute / Harbor)"
# Cited in all 6 frontier-lab launches surveyed. 91 containerized shell tasks
# graded by pytest unit tests. v1 "original-tasks" has 80+ older tasks.
clone_shallow "${OUT}/terminal_bench/v1_repo" https://github.com/laude-institute/terminal-bench.git
clone_shallow "${OUT}/terminal_bench/v2_repo" https://github.com/laude-institute/terminal-bench-2.git
fetch "${OUT}/terminal_bench/paper_2503.18809.pdf" https://arxiv.org/pdf/2503.18809
fetch "${OUT}/terminal_bench/leaderboard.html"     https://harborframework.com/leaderboards

say "SWE-Bench Pro (ScaleAI)"
# Cited by OpenAI GPT-5 / Anthropic Claude Opus 4.7 / Google Gemini 3.x.
# 731 long-horizon SWE tasks on 11 private-enterprise repos, MIT licensed.
clone_shallow "${OUT}/swe_bench_pro/repo" https://github.com/scaleapi/SWE-bench_Pro-os.git
hf_snapshot   ScaleAI/SWE-bench_Pro dataset "${OUT}/swe_bench_pro/hf"
fetch "${OUT}/swe_bench_pro/paper_2509.16941.pdf"     https://arxiv.org/pdf/2509.16941
fetch "${OUT}/swe_bench_pro/leaderboard_public.html"  https://scale.com/leaderboard/swe_bench_pro_public
fetch "${OUT}/swe_bench_pro/leaderboard_private.html" https://labs.scale.com/leaderboard/swe_bench_pro_private

say "PaperBench (OpenAI)"
# 23 ICML 2024 papers, each with a hierarchical rubric (~8k leaf requirements)
# co-developed with paper authors. Cited in GPT-5 system card. Sparse checkout
# to avoid pulling the other preparedness evals.
if [[ ! -d "${OUT}/paperbench/repo/.git" ]]; then
  mkdir -p "${OUT}/paperbench"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/openai/preparedness.git "${OUT}/paperbench/repo"
  ( cd "${OUT}/paperbench/repo" && git sparse-checkout set project/paperbench ) || true
fi
fetch "${OUT}/paperbench/paper_2504.01848.pdf" https://arxiv.org/pdf/2504.01848

say "LAB-Bench (FutureHouse)"
# 8 biology research subtasks (~1,967 public questions, ~80% of the full set;
# 20% held out to monitor contamination). Cited by Claude Opus 4.7 (FigQA).
clone_shallow "${OUT}/lab_bench/repo" https://github.com/Future-House/LAB-Bench.git
hf_snapshot   futurehouse/lab-bench dataset "${OUT}/lab_bench/hf"
fetch "${OUT}/lab_bench/paper_2407.10362.pdf" https://arxiv.org/pdf/2407.10362

say "Done. See docs/benchmarks_downloads.md for the manifest and known gaps."

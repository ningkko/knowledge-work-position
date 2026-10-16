"""Distill every benchmark we downloaded into a unified task-level JSONL.

Input:  data/raw/benchmarks/<bench>/...  (populated by scripts/download_benchmarks.sh)
Output: data/derived/benchmark_corpus/<bench>.jsonl, one row per scoreable unit.

Each row has the shape:
    {
      "benchmark_id": "gdpval",
      "task_id": "83d10b06-26d1-4636-a32c-23f92c57f30b",
      "prompt":   "...",         # task / question text
      "answer":   "..." | null,  # gold / reference answer if the bench has one
      "rubric":   "..." | null,  # text of the grading criteria / scoring rules
      "meta":     {...}          # benchmark-specific extras (sector, domain, difficulty, ...)
    }

This is intentionally small (<100MB total) and gets committed under
`data/derived/benchmark_corpus/` so downstream NLP comparison against the 24
work-pattern labels can run without re-downloading anything.

Run:
    python3 scripts/distill_benchmark_corpus.py
Re-runs are idempotent; missing benchmarks are skipped with a warning.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "benchmarks"
OUT = REPO_ROOT / "data" / "derived" / "benchmark_corpus"
OUT.mkdir(parents=True, exist_ok=True)


def _jsonable(v: Any) -> Any:
    """Convert numpy arrays / pandas Timestamps / NaNs to plain JSON-friendly types."""
    import numpy as np
    import pandas as pd

    if v is None:
        return None
    if isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        return None if (v != v) else v  # NaN -> null
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalar
        try:
            return _jsonable(v.item())
        except Exception:
            pass
    return str(v)


def write_jsonl(bench_id: str, rows: Iterator[dict]) -> int:
    dest = OUT / f"{bench_id}.jsonl"
    n = 0
    with dest.open("w") as fh:
        for r in rows:
            r = {k: _jsonable(v) for k, v in r.items()}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n} rows -> {dest.relative_to(REPO_ROOT)}")
    return n


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def emit_gdpval() -> int:
    src = RAW / "gdpval" / "hf_meta" / "data" / "train-00000-of-00001.parquet"
    if not src.exists():
        print("  [skip] gdpval: missing", src)
        return 0
    import pandas as pd

    df = pd.read_parquet(src)

    def _arr(v):
        try:
            return list(v) if v is not None else []
        except TypeError:
            return []

    def rows():
        for _, r in df.iterrows():
            yield {
                "benchmark_id": "gdpval",
                "task_id": r["task_id"],
                "prompt": r["prompt"],
                "answer": None,  # no single gold answer; deliverables are the reference
                "rubric": r.get("rubric_pretty") or r.get("rubric_json"),
                "meta": {
                    "sector": r.get("sector"),
                    "occupation": r.get("occupation"),
                    "reference_files": _arr(r.get("reference_files")),
                    "deliverable_files": _arr(r.get("deliverable_files")),
                    "rubric_json": r.get("rubric_json"),
                },
            }

    return write_jsonl("gdpval", rows())


def emit_apex_agents() -> int:
    src = RAW / "apex_agents" / "apex_agents_hf" / "tasks_and_rubrics.json"
    if not src.exists():
        print("  [skip] apex_agents: missing", src)
        return 0
    tasks = json.loads(src.read_text())

    def rows():
        for t in tasks:
            rubric_text = "\n".join(
                f"- [{c.get('verifier_id')}] {c.get('criteria')}" for c in (t.get("rubric") or [])
            )
            yield {
                "benchmark_id": "apex_agents",
                "task_id": t["task_id"],
                "prompt": t["prompt"],
                "answer": t.get("gold_response"),
                "rubric": rubric_text or None,
                "meta": {
                    "task_name": t.get("task_name"),
                    "world_id": t.get("world_id"),
                    "domain": t.get("domain"),
                    "expected_output": t.get("expected_output"),
                    "gold_response_type": t.get("gold_response_type"),
                    "rubric_criteria": t.get("rubric"),
                    "task_input_files": t.get("task_input_files"),
                },
            }

    return write_jsonl("apex_agents", rows())


def emit_apex_v1_extended() -> int:
    src = RAW / "apex_agents" / "apex_v1_extended_hf" / "data" / "train.csv"
    if not src.exists():
        print("  [skip] apex_v1_extended: missing", src)
        return 0
    import pandas as pd

    df = pd.read_csv(src)

    def rows():
        for _, r in df.iterrows():
            yield {
                "benchmark_id": "apex_v1_extended",
                "task_id": str(r["Task ID"]),
                "prompt": r["Prompt"],
                "answer": None,
                "rubric": r.get("Rubric JSON"),
                "meta": {
                    "domain": r.get("Domain"),
                    "file_attachments": (r.get("File Attachments") or "").splitlines()
                    if isinstance(r.get("File Attachments"), str)
                    else [],
                },
            }

    return write_jsonl("apex_v1_extended", rows())


def emit_officeqa() -> int:
    total = 0
    for split in ("pro", "full"):
        src = RAW / "officeqa" / "repo" / f"officeqa_{split}.csv"
        if not src.exists():
            print(f"  [skip] officeqa_{split}: missing", src)
            continue
        import pandas as pd

        df = pd.read_csv(src)
        bench_id = f"officeqa_{split}"

        def rows(df=df, split=split):
            for _, r in df.iterrows():
                yield {
                    "benchmark_id": f"officeqa_{split}",
                    "task_id": r["uid"],
                    "prompt": r["question"],
                    "answer": r.get("answer"),
                    "rubric": None,  # reward.py is a deterministic numeric grader
                    "meta": {
                        "source_docs": r.get("source_docs"),
                        "source_files": r.get("source_files"),
                        "difficulty": r.get("difficulty"),
                    },
                }

        total += write_jsonl(bench_id, rows())
    return total


def emit_financebench() -> int:
    # Prefer the HF jsonl (may differ in column shape); fall back to GitHub data/.
    candidates = [
        RAW / "financebench" / "hf" / "financebench_merged.jsonl",
        RAW / "financebench" / "repo" / "data" / "financebench_open_source.jsonl",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("  [skip] financebench: no source file found")
        return 0
    import json as J

    def rows():
        for line in src.open():
            line = line.strip()
            if not line:
                continue
            r = J.loads(line)
            yield {
                "benchmark_id": "financebench",
                "task_id": r.get("financebench_id") or r.get("id"),
                "prompt": r.get("question"),
                "answer": r.get("answer"),
                "rubric": r.get("justification"),  # the justification is the reference reasoning
                "meta": {
                    "company": r.get("company"),
                    "doc_name": r.get("doc_name"),
                    "doc_type": r.get("doc_type"),
                    "doc_period": r.get("doc_period"),
                    "doc_link": r.get("doc_link"),
                    "gics_sector": r.get("gics_sector"),
                    "question_type": r.get("question_type"),
                    "question_reasoning": r.get("question_reasoning"),
                    "domain_question_num": r.get("domain_question_num"),
                    "dataset_subset_label": r.get("dataset_subset_label"),
                    "evidence": r.get("evidence"),
                },
            }

    return write_jsonl("financebench", rows())


def emit_legalbench() -> int:
    """Two files:

    - ``legalbench_tasks.jsonl``: one row per sub-task (162), with the full README
      as `rubric` and the first test example as the representative `prompt`.
      This is the primary file for NLP comparison against work-pattern labels.
    - ``legalbench.jsonl``: one row per test example, with `rubric=None` and
      `meta.task` pointing back to the task-level file. Kept for per-example
      analysis without replicating the rubric text ~90k times.
    """
    data_dir = RAW / "legalbench" / "hf" / "data"
    task_docs_dir = RAW / "legalbench" / "repo" / "tasks"
    if not data_dir.exists():
        print("  [skip] legalbench: missing", data_dir)
        return 0

    task_rows: list[dict] = []
    example_rows: list[dict] = []
    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        test_tsv = task_dir / "test.tsv"
        if not test_tsv.exists():
            continue
        readme = task_docs_dir / task_name / "README.md"
        rubric = readme.read_text() if readme.exists() else None
        with test_tsv.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)
        first_prompt = None
        for i, row in enumerate(rows):
            prompt_col = row.get("text") or row.get("question") or row.get("input")
            answer_col = row.get("answer") or row.get("label")
            if first_prompt is None and prompt_col:
                first_prompt = prompt_col
            example_rows.append(
                {
                    "benchmark_id": "legalbench",
                    "task_id": f"{task_name}/{row.get('index', i)}",
                    "prompt": prompt_col,
                    "answer": answer_col,
                    "rubric": None,  # de-duplicated; see legalbench_tasks.jsonl
                    "meta": {"task": task_name, "row_index": row.get("index", i)},
                }
            )
        task_rows.append(
            {
                "benchmark_id": "legalbench_tasks",
                "task_id": task_name,
                "prompt": first_prompt,
                "answer": None,
                "rubric": rubric,
                "meta": {
                    "task": task_name,
                    "num_examples": len(rows),
                    "reasoning_type": _legalbench_reasoning_type(task_name),
                },
            }
        )

    n_tasks = write_jsonl("legalbench_tasks", iter(task_rows))
    n_examples = write_jsonl("legalbench", iter(example_rows))
    return n_tasks + n_examples


_LEGALBENCH_TYPE_PREFIXES = {
    "issue": "issue_spotting",
    "rule": "rule_application",
    "conclusion": "rule_conclusion",
    "application": "rule_application",
    "interpretation": "interpretation",
    "rhetorical": "rhetorical_analysis",
    "cuad": "rule_application",
    "maud": "rule_application",
    "contract": "rule_application",
    "privacy": "interpretation",
}


def _legalbench_reasoning_type(task_name: str) -> str | None:
    for prefix, ty in _LEGALBENCH_TYPE_PREFIXES.items():
        if task_name.startswith(prefix):
            return ty
    return None


def emit_madqa() -> int:
    data_dir = RAW / "madqa" / "hf" / "data"
    if not data_dir.exists():
        print("  [skip] madqa: missing", data_dir)
        return 0
    import pandas as pd

    def rows():
        for split in ("dev", "test", "train"):
            p = data_dir / f"{split}-00000-of-00001.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            for _, r in df.iterrows():
                ans = r.get("answer_variants")
                if ans is not None:
                    try:
                        ans = list(ans)
                    except Exception:
                        pass
                yield {
                    "benchmark_id": "madqa",
                    "task_id": r["id"],
                    "prompt": r["question"],
                    "answer": ans,
                    "rubric": None,
                    "meta": {
                        "split": split,
                        "document_category": r.get("document_category"),
                        "domain": r.get("domain"),
                        "evidence": r.get("evidence"),
                    },
                }

    return write_jsonl("madqa", rows())


def _browsecomp_derive_key(password: str, length: int) -> bytes:
    import hashlib

    h = hashlib.sha256()
    h.update(password.encode())
    key = h.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _browsecomp_decrypt(ciphertext_b64: str, password: str) -> str:
    import base64

    enc = base64.b64decode(ciphertext_b64)
    key = _browsecomp_derive_key(password, len(enc))
    return bytes(a ^ b for a, b in zip(enc, key)).decode("utf-8", errors="replace")


def emit_browsecomp() -> int:
    """Distill BrowseComp.

    The upstream CSV XOR-encrypts ``problem``, ``answer``, and
    ``problem_topic`` with the per-row ``canary`` column so the benchmark does
    not leak into training corpora as plaintext. The row's own ``canary`` text
    literally says:

        BENCHMARK DATA SHOULD NEVER APPEAR AS PLAIN TEXT ONLINE.

    We respect that — ``prompt`` and ``answer`` stay ciphertext so re-hosting
    the corpus doesn't republish plaintext. But we decrypt internally to emit
    *structural* meta fields (plaintext topic, prompt/answer character &
    word counts) because those are what the paper's real-work audit needs
    (topic distribution, answer-shape distribution) and they aren't the
    contamination surface the canary is protecting.

    Analysts with a legitimate need for plaintext can reproduce the
    decryption locally; see ``simple_evals_repo/browsecomp_eval.py``.
    """
    src = RAW / "browsecomp" / "browse_comp_test_set.csv"
    if not src.exists():
        print("  [skip] browsecomp: missing", src)
        return 0
    import pandas as pd

    df = pd.read_csv(src)

    def rows():
        for i, r in df.iterrows():
            ct_prompt = r.get("problem")
            ct_answer = r.get("answer")
            # `problem_topic` ships as plaintext in the CSV (e.g. 'Art', 'Sports').
            # Only `problem` and `answer` are XOR-encrypted; see
            # simple_evals_repo/browsecomp_eval.py:97-98.
            plain_topic = r.get("problem_topic")
            if not isinstance(plain_topic, str):
                plain_topic = None
            canary = r.get("canary") or ""
            prompt_chars = prompt_words = None
            answer_chars = answer_words = None
            try:
                if isinstance(ct_prompt, str) and canary:
                    p = _browsecomp_decrypt(ct_prompt, canary)
                    prompt_chars = len(p)
                    prompt_words = len(p.split())
                if isinstance(ct_answer, str) and canary:
                    a = _browsecomp_decrypt(ct_answer, canary)
                    answer_chars = len(a)
                    answer_words = len(a.split())
            except Exception:
                pass
            yield {
                "benchmark_id": "browsecomp",
                "task_id": str(i),
                "prompt": ct_prompt,  # ciphertext — do not publish plaintext
                "answer": ct_answer,  # ciphertext — do not publish plaintext
                "rubric": (
                    "Exact-match via LM judge over plaintext question/answer. "
                    "`problem` and `answer` are XOR-encrypted with the per-row "
                    "`canary` column as password; see "
                    "simple_evals_repo/browsecomp_eval.py. "
                    "`meta.problem_topic` is plaintext in the upstream CSV. "
                    "`meta.prompt_chars/words` and `answer_chars/words` are "
                    "derived from decrypted plaintext but do not expose the "
                    "question/answer text itself."
                ),
                "meta": {
                    "problem_topic": plain_topic,  # plaintext topic (e.g. 'Art')
                    "canary": canary,
                    "prompt_chars": prompt_chars,
                    "prompt_words": prompt_words,
                    "answer_chars": answer_chars,
                    "answer_words": answer_words,
                },
            }

    return write_jsonl("browsecomp", rows())


def emit_medagentbench() -> int:
    data_dir = RAW / "medagentbench" / "repo" / "data"
    if not data_dir.exists():
        print("  [skip] medagentbench: missing", data_dir)
        return 0
    # Find any task json files.
    candidates = list(data_dir.glob("*.json")) + list(data_dir.glob("**/*.json"))
    candidates = [p for p in candidates if "task" in p.name.lower() or "test" in p.name.lower()]
    if not candidates:
        print("  [skip] medagentbench: no task json found in", data_dir)
        return 0

    def rows():
        for p in sorted(set(candidates)):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "tasks" in data:
                items = data["tasks"]
            else:
                continue
            for i, t in enumerate(items):
                if not isinstance(t, dict):
                    continue
                yield {
                    "benchmark_id": "medagentbench",
                    "task_id": t.get("id") or f"{p.stem}/{i}",
                    "prompt": t.get("instruction") or t.get("prompt") or t.get("question"),
                    "answer": t.get("sol") or t.get("answer") or t.get("gold"),
                    "rubric": t.get("eval") or t.get("evaluator") or t.get("rubric"),
                    "meta": {k: v for k, v in t.items() if k not in {"instruction", "prompt", "question", "answer", "gold", "eval", "evaluator", "rubric"}},
                }

    return write_jsonl("medagentbench", rows())


def emit_webarena() -> int:
    cfg_dir = RAW / "webarena" / "repo" / "config_files"
    if not cfg_dir.exists():
        print("  [skip] webarena: missing", cfg_dir)
        return 0
    files = sorted(cfg_dir.glob("*.json"))
    if not files:
        # some releases use test.raw.json
        files = sorted(cfg_dir.glob("**/*.json"))

    def rows():
        for p in files:
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]
            for i, t in enumerate(items):
                if not isinstance(t, dict):
                    continue
                eval_spec = t.get("eval") or {}
                rubric = (
                    json.dumps(eval_spec, ensure_ascii=False) if eval_spec else None
                )
                yield {
                    "benchmark_id": "webarena",
                    "task_id": str(t.get("task_id") or t.get("id") or f"{p.stem}/{i}"),
                    "prompt": t.get("intent"),
                    "answer": eval_spec.get("reference_answers") if isinstance(eval_spec, dict) else None,
                    "rubric": rubric,
                    "meta": {
                        "sites": t.get("sites"),
                        "start_url": t.get("start_url"),
                        "eval_types": (eval_spec or {}).get("eval_types") if isinstance(eval_spec, dict) else None,
                        "source_file": p.name,
                    },
                }

    return write_jsonl("webarena", rows())


def emit_osworld() -> int:
    root = RAW / "osworld" / "repo" / "evaluation_examples"
    if not root.exists():
        print("  [skip] osworld: missing", root)
        return 0

    def rows():
        for p in sorted(root.rglob("*.json")):
            try:
                t = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(t, dict):
                continue
            eval_spec = t.get("evaluator") or t.get("eval")
            yield {
                "benchmark_id": "osworld",
                "task_id": t.get("id") or str(p.relative_to(root)),
                "prompt": t.get("instruction"),
                "answer": None,
                "rubric": json.dumps(eval_spec, ensure_ascii=False) if eval_spec else None,
                "meta": {
                    "source": t.get("source"),
                    "snapshot": t.get("snapshot"),
                    "config": t.get("config"),
                    "source_file": str(p.relative_to(root)),
                },
            }

    return write_jsonl("osworld", rows())


def emit_tau_bench() -> int:
    """tau-bench stores tasks as python dicts in ``tau_bench/envs/<domain>/tasks*.py``.

    We exec the module in a fake ``Action`` shim environment so we don't need
    the full package installed, then dump the ``tasks`` list.
    """
    root = RAW / "tau_bench" / "repo"
    if not root.exists():
        print("  [skip] tau_bench: missing", root)
        return 0
    envs_dir = root / "tau_bench" / "envs"
    if not envs_dir.exists():
        print("  [skip] tau_bench: missing envs dir")
        return 0

    rows_out: list[dict] = []
    for domain_dir in sorted(envs_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        for tasks_py in sorted(domain_dir.glob("tasks*.py")):
            if tasks_py.name.endswith("test.py"):
                continue
            split = tasks_py.stem.replace("tasks", "").strip("_") or "main"
            # Exec the file with a placeholder `Action` available, since some
            # tasks*.py import it.
            ns: dict = {
                "Action": lambda **kw: kw,  # noqa: E731
                "TASKS_TEST": None,
                "__name__": "tau_tasks",
            }
            try:
                exec(compile(tasks_py.read_text(), str(tasks_py), "exec"), ns)
            except Exception as e:
                print(f"  [warn] tau_bench {tasks_py.relative_to(root)}: {type(e).__name__}: {e}")
                continue
            tasks_list = ns.get("tasks") or ns.get("TASKS") or ns.get("tasks_test") or ns.get("TASKS_TEST")
            if not isinstance(tasks_list, list):
                continue
            for i, t in enumerate(tasks_list):
                if not isinstance(t, dict):
                    continue
                rows_out.append(
                    {
                        "benchmark_id": "tau_bench",
                        "task_id": f"{domain_dir.name}/{split}/{i}",
                        "prompt": t.get("instruction") or t.get("user_request"),
                        "answer": t.get("outputs") or t.get("expected"),
                        "rubric": json.dumps(t.get("actions"), ensure_ascii=False)
                        if t.get("actions")
                        else None,
                        "meta": {
                            "domain": domain_dir.name,
                            "split": split,
                            "annotator": t.get("annotator"),
                            "user_id": t.get("user_id"),
                            "actions": t.get("actions"),
                        },
                    }
                )

    return write_jsonl("tau_bench", iter(rows_out))


def emit_tau2_bench() -> int:
    """tau2-bench ships per-domain ``tasks.json`` at
    ``data/tau2/domains/<domain>/tasks.json``.

    Each task is a dict with:

      - ``id`` — human-readable id like ``[mobile_data_issue]user_abroad_...``
      - ``description.purpose`` — one of ~3 'resolution paths' per domain
      - ``user_scenario.instructions.reason_for_call`` /
        ``.task_instructions`` / ``.known_info`` — simulated-user prompt
      - ``user_scenario.persona`` — optional persona string (41-yo admin, etc.)
      - ``ticket`` — the "ticket text" the agent sees
      - ``initial_state.initialization_actions`` — env setup
      - ``evaluation_criteria`` — ``actions`` + ``env_assertions`` +
        ``communicate_info`` + ``nl_assertions`` + ``reward_basis``

    We emit one file per domain (``tau2_retail.jsonl``, ``tau2_airline.jsonl``,
    ``tau2_telecom.jsonl``, ...) so the paper can point at the domain-specific
    construct — e.g. OpenAI's GPT-5.5 headline reports only the Telecom split.
    Only ``tasks.json`` is emitted; ``tasks_voice.json`` and
    ``tasks_small.json`` are skipped (voice-modality or smoke subsets).
    """
    root = RAW / "tau_bench" / "tau2_repo" / "data" / "tau2" / "domains"
    if not root.exists():
        print("  [skip] tau2_bench: missing", root)
        return 0

    total = 0
    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir():
            continue
        # Skip tutorial / mock domains that don't represent a real benchmark split.
        if domain_dir.name in {"mock"}:
            continue
        src = domain_dir / "tasks.json"
        if not src.exists():
            continue
        try:
            tasks = json.loads(src.read_text())
        except Exception as e:
            print(f"  [warn] tau2_bench {domain_dir.name}: {type(e).__name__}: {e}")
            continue
        if not isinstance(tasks, list):
            continue

        bench_id = f"tau2_{domain_dir.name}"

        def rows_for(tasks=tasks, domain=domain_dir.name, bench_id=bench_id):
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                us = t.get("user_scenario") or {}
                if not isinstance(us, dict):
                    us = {}
                ins = us.get("instructions")
                persona = us.get("persona")
                # Prompt = what the simulated user tells the agent. It is NOT
                # a first-person task instruction to the agent — it's the
                # user-side scenario script. Some domains (banking_knowledge)
                # ship a plain instructions string; others (telecom, retail,
                # airline) ship a dict with reason_for_call / known_info /
                # task_instructions sub-fields.
                prompt_parts: list[str] = []
                if isinstance(ins, dict):
                    for k in ("reason_for_call", "known_info", "task_instructions"):
                        v = ins.get(k)
                        if v:
                            prompt_parts.append(f"[{k}] {v}".strip())
                elif isinstance(ins, str) and ins.strip():
                    prompt_parts.append(ins.strip())
                prompt = "\n\n".join(prompt_parts) or None

                ec = t.get("evaluation_criteria") or {}
                if not isinstance(ec, dict):
                    ec = {}
                init = t.get("initial_state") or {}
                if not isinstance(init, dict):
                    init = {}
                init_actions = init.get("initialization_actions") or []
                eval_actions = ec.get("actions") or []
                env_assertions = ec.get("env_assertions") or []
                nl_assertions = ec.get("nl_assertions")
                communicate_info = ec.get("communicate_info")

                desc = t.get("description")
                purpose = desc.get("purpose") if isinstance(desc, dict) else None
                yield {
                    "benchmark_id": bench_id,
                    "task_id": t.get("id") or f"{domain}/{hash(json.dumps(t, sort_keys=True, default=str))}",
                    "prompt": prompt,
                    "answer": None,  # no gold text; rubric is a spec
                    "rubric": json.dumps(ec, ensure_ascii=False) if ec else None,
                    "meta": {
                        "domain": domain,
                        "purpose": purpose,
                        "persona": persona,
                        "ticket": t.get("ticket"),
                        "reward_basis": ec.get("reward_basis"),
                        "n_initialization_actions": len(init_actions) if isinstance(init_actions, list) else 0,
                        "n_eval_actions": len(eval_actions) if isinstance(eval_actions, list) else 0,
                        "n_env_assertions": len(env_assertions) if isinstance(env_assertions, list) else 0,
                        "has_nl_assertions": bool(nl_assertions),
                        "has_communicate_info": bool(communicate_info),
                    },
                }

        total += write_jsonl(bench_id, rows_for())
    return total


# ---------------------------------------------------------------------------
# GDPval rubric-item level distillation
#
# Classifies each rubric ``criterion`` text as one of:
#   mechanical    — file/format/naming checks ("workbook basename is 'Sample'")
#   institutional — authority / audit / citation / provenance / reviewer
#   substantive   — professional judgment (calculation, methodology, analysis)
#   other         — content-specific predicate on the expected deliverable
#                   ("the first tab contains at least one row where division
#                    is Corporate Banking and country is Luxembourg")
#
# The classifier is keyword-based and deliberately simple. It exists so the
# paper can report a headline number — "X% of rubric points measure
# professional judgment vs. form-filling" — that can be re-derived from the
# committed JSONL without re-running the classifier.
# ---------------------------------------------------------------------------

import re as _re

_GDPVAL_MECHANICAL_RE = _re.compile(
    r"\b(file ?name|basename|file is \.|\.xlsx|\.pdf|\.docx|\.xls|\.pptx|"
    r"extension|workbook file|workbook contains|worksheet named|sheet is named|"
    r"sheet titled|column named|column [A-Z]\b|cell [A-Z]?\d|first worksheet|"
    r"title includes|title contains|header text|page \d|font|size \d+ ?pt|"
    r"\bpt\b|margin|tab \d|tab titled|file format|line spacing|bold|italic|"
    r"ordered list|bullet|color|row \d|header row)\b",
    _re.IGNORECASE,
)

_GDPVAL_INSTITUTIONAL_RE = _re.compile(
    r"\b(sign.?off|approv|reviewer|supervisor|authority|audit trail|audit log|"
    r"stakeholder|disclosur|disclaim|escalat|exception report|versioned|"
    r"change log|reference(?:d|s)?|citation|cite(?:s|d)?|cited|source(?:s|d)?|"
    r"provenance|attribut|traceable|justif|warrant|basis|authoriz|"
    r"chain of custody|retention|compliance note|regulat|disclosure)\b",
    _re.IGNORECASE,
)

_GDPVAL_SUBSTANTIVE_RE = _re.compile(
    r"\b(calculat|variance|formula|depreciat|reconcil|accurate|correct(?:ly)?|"
    r"compliant|regulator|GAAP|IFRS|tax|complete|deduct|net\b|gross|interest|"
    r"amortiz|present value|\bNPV\b|\bIRR\b|sampling|confidence|margin of error|"
    r"analysis|reasoning|factual|explanation|standard|precedent|methodology|"
    r"interpret|argue|conclude|diagnos|differenti|determin|evidence|"
    r"support(?:s|ed)|rationale|weigh|measure|estimate|assumption|scenario|"
    r"sensitivity|forecast)\b",
    _re.IGNORECASE,
)


def _gdpval_classify(criterion: str) -> str:
    if not criterion:
        return "other"
    # Priority: institutional > substantive > mechanical. "other" = no match.
    # Institutional wins because those are the failure modes the paper argues
    # aggregate scoring hides.
    if _GDPVAL_INSTITUTIONAL_RE.search(criterion):
        return "institutional"
    if _GDPVAL_SUBSTANTIVE_RE.search(criterion):
        return "substantive"
    if _GDPVAL_MECHANICAL_RE.search(criterion):
        return "mechanical"
    return "other"


def emit_gdpval_rubric_items() -> int:
    """Explode GDPval's parquet into one row per rubric item.

    ``gdpval.jsonl`` keeps one row per task with the pretty rubric string.
    This file exposes the ~10k individual rubric criteria so the paper can
    audit what the rubric is actually grading. Each row carries a keyword-
    based ``kind`` tag (mechanical / substantive / institutional / other).
    """
    src = RAW / "gdpval" / "hf_meta" / "data" / "train-00000-of-00001.parquet"
    if not src.exists():
        print("  [skip] gdpval_rubric_items: missing", src)
        return 0
    import pandas as pd

    df = pd.read_parquet(src)

    def rows():
        for _, r in df.iterrows():
            rb = r.get("rubric_json")
            if isinstance(rb, str):
                try:
                    items = json.loads(rb)
                except Exception:
                    items = []
            elif isinstance(rb, list):
                items = rb
            else:
                items = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                criterion = it.get("criterion") or ""
                kind = _gdpval_classify(criterion)
                tags = it.get("tags") or []
                if not isinstance(tags, list):
                    try:
                        tags = list(tags)
                    except Exception:
                        tags = []
                yield {
                    "benchmark_id": "gdpval_rubric_items",
                    "task_id": f"{r['task_id']}/{it.get('rubric_item_id')}",
                    "prompt": criterion,  # the rubric criterion IS the scoreable unit
                    "answer": None,
                    "rubric": None,
                    "meta": {
                        "parent_task_id": r["task_id"],
                        "sector": r.get("sector"),
                        "occupation": r.get("occupation"),
                        "rubric_item_id": it.get("rubric_item_id"),
                        "score": it.get("score"),
                        "required": it.get("required"),
                        "author_type": it.get("author_type"),
                        "read_only": it.get("read_only"),
                        "tags": tags,
                        "kind": kind,
                    },
                }

    return write_jsonl("gdpval_rubric_items", rows())


def emit_theagentcompany() -> int:
    root = RAW / "theagentcompany" / "repo"
    if not root.exists():
        print("  [skip] theagentcompany: missing", root)
        return 0
    # Look for per-task JSON/MD/YAML under evaluation/ or workspaces/
    tasks = []
    for sub in ("workspaces", "evaluation", "tasks"):
        d = root / sub
        if d.exists():
            tasks.extend(d.rglob("config.yaml"))
            tasks.extend(d.rglob("task.md"))
            tasks.extend(d.rglob("task.json"))
    if not tasks:
        print("  [skip] theagentcompany: no task config files found")
        return 0

    import yaml  # type: ignore

    def rows():
        for p in tasks:
            try:
                if p.suffix == ".yaml":
                    t = yaml.safe_load(p.read_text())
                elif p.suffix == ".json":
                    t = json.loads(p.read_text())
                else:  # md
                    t = {"instruction": p.read_text()}
            except Exception:
                continue
            if not isinstance(t, dict):
                continue
            task_id = str(p.relative_to(root))
            yield {
                "benchmark_id": "theagentcompany",
                "task_id": task_id,
                "prompt": t.get("instruction") or t.get("prompt") or t.get("description"),
                "answer": None,
                "rubric": json.dumps(t.get("evaluator") or t.get("eval") or t.get("checkpoints"), ensure_ascii=False)
                if (t.get("evaluator") or t.get("eval") or t.get("checkpoints"))
                else None,
                "meta": {"source_file": task_id},
            }

    return write_jsonl("theagentcompany", rows())


def emit_officebench() -> int:
    root = RAW / "officebench" / "repo" / "tasks"
    if not root.exists():
        print("  [skip] officebench: missing", root)
        return 0

    def rows():
        for p in sorted(root.rglob("*.json")):
            try:
                t = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(t, dict):
                continue
            evaluation = t.get("evaluation") or t.get("evaluator") or t.get("rubric")
            yield {
                "benchmark_id": "officebench",
                "task_id": t.get("id") or str(p.relative_to(root)),
                "prompt": t.get("task") or t.get("instruction") or t.get("prompt"),
                "answer": t.get("answer") or t.get("expected_output"),
                "rubric": json.dumps(evaluation, ensure_ascii=False) if evaluation else None,
                "meta": {
                    "source_file": str(p.relative_to(root)),
                    "username": t.get("username"),
                    "date": t.get("date"),
                    "weekday": t.get("weekday"),
                    "time": t.get("time"),
                },
            }

    return write_jsonl("officebench", rows())


def emit_workarena() -> int:
    # WorkArena tasks are python-defined; emit a pointer-only row per task class found.
    root = RAW / "workarena" / "repo" / "src" / "workarena" / "tasks"
    if not root.exists():
        root = RAW / "workarena" / "repo"
    if not root.exists():
        print("  [skip] workarena: missing repo")
        return 0
    # Best-effort: list the discoverable task files as a "task inventory".
    files = list(root.rglob("*.py"))
    if not files:
        return 0

    def rows():
        for p in sorted(files):
            yield {
                "benchmark_id": "workarena",
                "task_id": str(p.relative_to(root)),
                "prompt": None,
                "answer": None,
                "rubric": None,
                "meta": {
                    "note": "WorkArena tasks are python-defined; see source file.",
                    "source_file": str(p.relative_to(root)),
                },
            }

    return write_jsonl("workarena", rows())


def emit_spider2() -> int:
    root = RAW / "spider2" / "repo"
    if not root.exists():
        print("  [skip] spider2: missing", root)
        return 0
    candidates = []
    for sub in ("spider2-lite", "spider2-snow", "spider2-dbt"):
        d = root / sub
        if d.exists():
            candidates.extend(d.rglob("*.jsonl"))
    if not candidates:
        print("  [skip] spider2: no jsonl found")
        return 0

    def rows():
        for p in candidates:
            bucket = p.parent.name
            for i, line in enumerate(p.open()):
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue
                yield {
                    "benchmark_id": "spider2",
                    "task_id": str(t.get("instance_id") or t.get("id") or f"{p.stem}/{i}"),
                    "prompt": t.get("instruction") or t.get("question"),
                    "answer": t.get("gold") or t.get("sql") or t.get("answer"),
                    "rubric": t.get("evaluator") or t.get("rubric"),
                    "meta": {
                        "source_file": str(p.relative_to(root)),
                        "db_id": t.get("db_id"),
                        "schema": t.get("schema"),
                        "bucket": bucket,
                    },
                }

    return write_jsonl("spider2", rows())


def emit_kwbench() -> int:
    # No dataset released yet — emit a pointer row so the corpus index still
    # notes its existence.
    site = RAW / "kwbench" / "site.html"
    if not site.exists():
        return 0
    rows = [
        {
            "benchmark_id": "kwbench",
            "task_id": "kwbench/pending-release",
            "prompt": None,
            "answer": None,
            "rubric": None,
            "meta": {
                "note": "KWBench public dataset not yet released as of 2026-04. See paper arXiv:2604.15760 and site kwbench.github.io.",
            },
        }
    ]
    return write_jsonl("kwbench", iter(rows))


def emit_finance_agent() -> int:
    """Vals AI Finance Agent Benchmark — 187 public tasks from `data/public.csv`.

    Cited in OpenAI's GPT-5.5 launch as "FinanceAgent" (60.0%). Full benchmark
    has 537 tasks behind the Vals platform; 187 are released in the public
    GitHub repo. Each row has a question, gold answer, question type,
    expert-time estimate, and a JSON rubric (list of {operator, criteria}).
    """
    src = RAW / "finance_agent" / "repo" / "data" / "public.csv"
    if not src.exists():
        print("  [skip] finance_agent: missing", src)
        return 0

    import pandas as pd

    df = pd.read_csv(src)

    def rows():
        for i, r in df.iterrows():
            rubric_raw = r.get("Rubric")
            rubric_text = None
            rubric_criteria = None
            if isinstance(rubric_raw, str) and rubric_raw.strip():
                try:
                    rubric_criteria = json.loads(rubric_raw)
                    bullets = []
                    for c in rubric_criteria:
                        op = c.get("operator") if isinstance(c, dict) else None
                        cr = c.get("criteria") if isinstance(c, dict) else None
                        bullets.append(f"- [{op}] {cr}" if op else f"- {cr}")
                    rubric_text = "\n".join(bullets)
                except Exception:
                    rubric_text = rubric_raw
            yield {
                "benchmark_id": "finance_agent",
                "task_id": f"finance_agent/public/{i:04d}",
                "prompt": r.get("Question"),
                "answer": r.get("Answer"),
                "rubric": rubric_text,
                "meta": {
                    "question_type": r.get("Question Type"),
                    "expert_time_mins": r.get("Expert time (mins)"),
                    "rubric_criteria": rubric_criteria,
                    "source": "vals-ai/finance-agent data/public.csv",
                    "full_benchmark_size": 537,
                    "public_subset_size": int(len(df)),
                },
            }

    return write_jsonl("finance_agent", rows())


def emit_vending_bench_2() -> int:
    """Andon Labs Vending-Bench 2 — pointer row.

    Cited in Gemini 3 Pro eval methodology and Claude Opus 4.7 system card.
    The benchmark is a year-long vending-machine simulation scored by final
    bank balance (not by per-task rubrics), and no task/rubric files are
    released. We emit a single pointer row describing the scoring rule and
    referencing the paper + leaderboard.
    """
    paper = RAW / "vending_bench_2" / "paper_2502.15840.pdf"
    html = RAW / "vending_bench_2" / "vending_bench_2.html"
    if not (paper.exists() or html.exists()):
        print("  [skip] vending_bench_2: no artifacts downloaded yet")
        return 0
    rubric_text = (
        "Score = final bank account balance (USD) after simulated year of "
        "running a vending machine business. Agent must balance inventory, "
        "place orders, set prices, handle daily fees, negotiate with suppliers "
        "and handle customer complaints. No per-task gold answer — scoring "
        "is purely outcome-based on capital accumulation over >20M tokens "
        "per run."
    )
    rows = [
        {
            "benchmark_id": "vending_bench_2",
            "task_id": "vending_bench_2/vending-business-simulation",
            "prompt": (
                "You run a vending machine business for one simulated year. "
                "Maintain inventory, set prices, negotiate with suppliers, "
                "handle adversarial inputs, and maximize your end-of-year "
                "bank account balance."
            ),
            "answer": None,
            "rubric": rubric_text,
            "meta": {
                "note": (
                    "Pointer-only row. Vending-Bench 2 is an agent simulation "
                    "environment with no public task/rubric dataset. See paper "
                    "arXiv:2502.15840 and leaderboard andonlabs.com/evals/vending-bench-2."
                ),
                "paper_arxiv": "2502.15840",
                "leaderboard_url": "https://andonlabs.com/evals/vending-bench-2",
                "scoring": "bank_balance_usd_after_simulated_year",
            },
        }
    ]
    return write_jsonl("vending_bench_2", iter(rows))


def emit_terminal_bench() -> int:
    """Terminal-Bench v2 (91 containerized shell tasks) + v1 originals.

    Each task dir under terminal_bench/v2_repo/<slug>/ has:
      - instruction.md   (task prompt to the agent)
      - task.toml        (metadata: difficulty, category, tags, time estimates)
      - tests/           (pytest tests acting as the rubric / grader)
      - solution/        (reference solution; not treated as gold "answer" because
                         the rubric is the passing test suite, not a single string)
    We emit one row per task with instruction.md as prompt and the concatenated
    test file names + truncated contents as rubric text.
    """
    rows: list[dict] = []
    for vlabel, root in [
        ("v2", RAW / "terminal_bench" / "v2_repo"),
        ("v1", RAW / "terminal_bench" / "v1_repo" / "original-tasks"),
    ]:
        if not root.exists():
            print(f"  [skip] terminal_bench {vlabel}: missing {root}")
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            instr = d / "instruction.md"
            if not instr.exists():
                continue
            meta: dict = {"version": vlabel, "slug": d.name}
            toml_path = next((p for p in [d / "task.toml", d / "task.yaml"] if p.exists()), None)
            if toml_path and toml_path.suffix == ".toml":
                try:
                    import tomllib
                    with toml_path.open("rb") as fh:
                        cfg = tomllib.load(fh)
                    md = cfg.get("metadata", {}) or {}
                    meta.update({
                        "difficulty": md.get("difficulty"),
                        "category": md.get("category"),
                        "tags": md.get("tags"),
                        "expert_time_min": md.get("expert_time_estimate_min"),
                        "junior_time_min": md.get("junior_time_estimate_min"),
                        "author": md.get("author_name"),
                    })
                except Exception:
                    pass
            # Rubric = pytest / test files under tests/ (agent is graded by these passing)
            tests_dir = d / "tests"
            test_names: list[str] = []
            rubric_parts: list[str] = []
            if tests_dir.exists():
                for t in sorted(tests_dir.rglob("*")):
                    if t.is_file() and t.suffix in {".py", ".sh", ".bats"}:
                        test_names.append(str(t.relative_to(d)))
                        try:
                            body = t.read_text(errors="replace")[:1500]
                            rubric_parts.append(f"### {t.relative_to(d)}\n{body}")
                        except Exception:
                            pass
            meta["test_files"] = test_names
            rubric = "\n\n".join(rubric_parts)[:10000] if rubric_parts else None
            rows.append({
                "benchmark_id": "terminal_bench",
                "task_id": f"{vlabel}/{d.name}",
                "prompt": instr.read_text(errors="replace")[:12000],
                "answer": None,  # agentic; graded by tests
                "rubric": rubric,
                "meta": meta,
            })
    return write_jsonl("terminal_bench", iter(rows))


def emit_swe_bench_pro() -> int:
    """SWE-Bench Pro (ScaleAI) — 731 long-horizon SWE tasks.

    Each row: problem statement + gold patch + tests (fail_to_pass, pass_to_pass).
    We emit prompt=problem_statement, answer=patch, rubric=summary of test
    expectations + selected tests. Test bodies can be huge (kBs–MBs), so we
    truncate for the JSONL but keep counts / names in meta.
    """
    src = RAW / "swe_bench_pro" / "hf" / "data" / "test-00000-of-00001.parquet"
    if not src.exists():
        print("  [skip] swe_bench_pro: missing", src)
        return 0
    import pandas as pd
    df = pd.read_parquet(src)
    rows: list[dict] = []
    for _, r in df.iterrows():
        f2p = r.get("fail_to_pass") or ""
        p2p = r.get("pass_to_pass") or ""
        rubric = (
            f"After applying the patch these tests must flip from failing "
            f"to passing:\n{str(f2p)[:2500]}\n\n"
            f"These tests must continue passing:\n{str(p2p)[:2500]}"
        )
        rows.append({
            "benchmark_id": "swe_bench_pro",
            "task_id": r["instance_id"],
            "prompt": (
                f"# Issue\n{r.get('problem_statement','')}\n\n"
                f"# Requirements\n{r.get('requirements') or ''}\n\n"
                f"# Interface / API spec\n{r.get('interface') or ''}"
            )[:15000],
            "answer": str(r.get("patch") or "")[:12000],
            "rubric": rubric,
            "meta": {
                "repo": r.get("repo"),
                "base_commit": r.get("base_commit"),
                "repo_language": r.get("repo_language"),
                "issue_specificity": r.get("issue_specificity"),
                "issue_categories": r.get("issue_categories"),
                "selected_test_files_to_run": r.get("selected_test_files_to_run"),
                "test_patch_len_chars": len(str(r.get("test_patch") or "")),
                "patch_len_chars": len(str(r.get("patch") or "")),
            },
        })
    return write_jsonl("swe_bench_pro", iter(rows))


def emit_paperbench() -> int:
    """PaperBench (OpenAI) — 23 ICML 2024 papers each with a hierarchical rubric.

    Leaves of the rubric tree correspond to ~8k individually gradable sub-tasks.
    We emit one row per paper with the rubric flattened as an outline (preserving
    the tree shape) and meta.n_leaves = count of leaf requirements.
    """
    root = RAW / "paperbench" / "repo" / "project" / "paperbench" / "data" / "papers"
    if not root.exists():
        print("  [skip] paperbench: missing", root)
        return 0
    import json as _json

    def _flatten(node: dict, depth: int = 0, lines: list[str] | None = None) -> list[str]:
        lines = lines if lines is not None else []
        req = node.get("requirements", "")
        weight = node.get("weight", "")
        lines.append(f"{'  ' * depth}- (w={weight}) {req}")
        for sub in node.get("sub_tasks", []) or []:
            _flatten(sub, depth + 1, lines)
        return lines

    def _count_leaves(node: dict) -> int:
        subs = node.get("sub_tasks") or []
        if not subs:
            return 1
        return sum(_count_leaves(s) for s in subs)

    rows: list[dict] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rj = d / "rubric.json"
        if not rj.exists():
            continue
        tree = _json.loads(rj.read_text())
        addendum = (d / "addendum.md").read_text(errors="replace") if (d / "addendum.md").exists() else ""
        paper_md = (d / "paper.md").read_text(errors="replace") if (d / "paper.md").exists() else ""
        # Try to pull title from first heading of paper.md
        title = ""
        for line in paper_md.splitlines()[:40]:
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
        outline = "\n".join(_flatten(tree))[:18000]
        rows.append({
            "benchmark_id": "paperbench",
            "task_id": d.name,
            "prompt": (
                f"Replicate the ICML 2024 paper '{title or d.name}' from scratch, "
                f"implementing the codebase and running the experiments that validate "
                f"its contributions.\n\n"
                f"# Addendum / clarifications from authors\n{addendum[:4000]}"
            )[:8000],
            "answer": None,
            "rubric": outline,
            "meta": {
                "slug": d.name,
                "title": title,
                "n_leaf_requirements": _count_leaves(tree),
                "top_level_requirement": tree.get("requirements", ""),
                "rubric_id": tree.get("id"),
                "grader": "paperbench_llm_judge",
            },
        })
    return write_jsonl("paperbench", iter(rows))


def emit_lab_bench() -> int:
    """LAB-Bench (FutureHouse) — 8 biology research subtasks (~1,967 questions).

    Each subtask ships a multiple-choice QA parquet. We emit one row per question
    with prompt=question, answer=ideal, rubric="multiple choice; correct=<ideal>;
    distractors=[...]" and meta.subtask / source.
    """
    root = RAW / "lab_bench" / "hf"
    if not root.exists():
        print("  [skip] lab_bench: missing", root)
        return 0
    import pandas as pd
    subtasks = ["CloningScenarios", "DbQA", "FigQA", "LitQA2", "ProtocolQA",
                "SeqQA", "SuppQA", "TableQA"]
    rows: list[dict] = []
    for sub in subtasks:
        p = root / sub / "train-00000-of-00001.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            distractors = r.get("distractors")
            try:
                distractors_list = list(distractors) if distractors is not None else []
            except Exception:
                distractors_list = []
            rubric = (
                f"Multiple choice. Correct answer: {r.get('ideal')}. "
                f"Distractors: {distractors_list}."
            )
            rd = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            meta: dict = {
                "subtask": sub,
                "source": rd.get("source"),
                "lab_bench_subtask_field": rd.get("subtask"),
            }
            # Subtask-specific scalar fields
            for extra in ("paper-title", "tag", "version", "is_opensource"):
                v = rd.get(extra)
                if v is None:
                    continue
                if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == 0:
                    continue
                meta[extra] = v
            # Drop heavy bytes/bytearray columns (figures, tables) — keep path only.
            # Some subtasks store table-path as an array of strings; take .tolist()
            fp = rd.get("figure-path")
            if fp is not None and not (hasattr(fp, "__len__") and len(fp) == 0):
                meta["figure_path"] = fp.tolist() if hasattr(fp, "tolist") else fp
            tp = rd.get("table-path")
            if tp is not None and not (hasattr(tp, "__len__") and len(tp) == 0):
                meta["table_path"] = tp.tolist() if hasattr(tp, "tolist") else tp
            rows.append({
                "benchmark_id": "lab_bench",
                "task_id": f"{sub}/{r.get('id')}",
                "prompt": str(r.get("question", ""))[:6000],
                "answer": r.get("ideal"),
                "rubric": rubric,
                "meta": meta,
            })
    return write_jsonl("lab_bench", iter(rows))


def emit_gaia() -> int:
    """GAIA (General AI Assistant) — 466 tasks across 3 levels, test + validation.

    Emits one row per question. Answer = 'Final answer'. Rubric is exact-match
    with canonicalization (per the GAIA paper); we capture the annotator's step
    trace in meta['steps'] so the implicit rubric (what should be used) is audit-able.
    """
    root = RAW / "gaia" / "hf" / "2023"
    if not root.exists():
        print("  [skip] gaia: missing", root)
        return 0
    import pandas as pd
    rows: list[dict] = []
    for split in ("test", "validation"):
        p = root / split / "metadata.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            ann = r.get("Annotator Metadata") or {}
            if not isinstance(ann, dict):
                try:
                    ann = dict(ann)
                except Exception:
                    ann = {}
            rows.append({
                "benchmark_id": "gaia",
                "task_id": f"{split}/{r.get('task_id')}",
                "prompt": str(r.get("Question") or "")[:8000],
                "answer": str(r.get("Final answer") or "") or None,
                "rubric": (
                    "GAIA exact-match grading: final answer must match the reference "
                    "after GAIA's canonicalisation (number/list/string). "
                    "Annotator's reference solution steps:\n"
                    f"{ann.get('Steps','')[:3000]}"
                ),
                "meta": {
                    "split": split,
                    "level": int(r.get("Level")) if r.get("Level") is not None else None,
                    "file_name": r.get("file_name") or None,
                    "tools_count": ann.get("Number of tools"),
                    "tools": ann.get("Tools"),
                    "steps_count": ann.get("Number of steps"),
                    "time_taken_human": ann.get("How long did this take?"),
                },
            })
    return write_jsonl("gaia", iter(rows))


ADAPTERS = {
    "apex_agents": emit_apex_agents,
    "apex_v1_extended": emit_apex_v1_extended,
    "browsecomp": emit_browsecomp,
    "finance_agent": emit_finance_agent,
    "financebench": emit_financebench,
    "gaia": emit_gaia,
    "gdpval": emit_gdpval,
    "gdpval_rubric_items": emit_gdpval_rubric_items,
    "kwbench": emit_kwbench,
    "lab_bench": emit_lab_bench,
    "legalbench": emit_legalbench,
    "madqa": emit_madqa,
    "medagentbench": emit_medagentbench,
    "officebench": emit_officebench,
    "officeqa": emit_officeqa,
    "osworld": emit_osworld,
    "paperbench": emit_paperbench,
    "spider2": emit_spider2,
    "swe_bench_pro": emit_swe_bench_pro,
    "tau_bench": emit_tau_bench,
    "tau2_bench": emit_tau2_bench,
    "terminal_bench": emit_terminal_bench,
    "theagentcompany": emit_theagentcompany,
    "vending_bench_2": emit_vending_bench_2,
    "webarena": emit_webarena,
    "workarena": emit_workarena,
}


def main() -> int:
    only = set(sys.argv[1:])
    totals: dict[str, int] = {}
    for bench_id, fn in ADAPTERS.items():
        if only and bench_id not in only:
            continue
        print(f"\n=== {bench_id} ===")
        try:
            totals[bench_id] = fn()
        except Exception as e:
            print(f"  [error] {bench_id}: {type(e).__name__}: {e}")
            totals[bench_id] = 0

    print("\n=== summary ===")
    for k, v in sorted(totals.items()):
        print(f"  {k:20s} {v:>7d}")
    print(f"  {'TOTAL':20s} {sum(totals.values()):>7d}")

    # Emit a concise index CSV so downstream tooling can load counts without
    # parsing every jsonl.
    index_path = OUT / "corpus_index.csv"
    index_rows = []
    for p in sorted(OUT.glob("*.jsonl")):
        n = with_prompt = with_answer = with_rubric = 0
        for line in p.open():
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if r.get("prompt"):
                with_prompt += 1
            if r.get("answer") is not None and r.get("answer") != "":
                with_answer += 1
            if r.get("rubric"):
                with_rubric += 1
        index_rows.append(
            {
                "benchmark_id": p.stem,
                "file": str(p.relative_to(REPO_ROOT)),
                "rows": n,
                "has_prompt": with_prompt,
                "has_answer": with_answer,
                "has_rubric": with_rubric,
                "size_kb": round(p.stat().st_size / 1024),
            }
        )
    with index_path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "benchmark_id",
                "file",
                "rows",
                "has_prompt",
                "has_answer",
                "has_rubric",
                "size_kb",
            ],
        )
        w.writeheader()
        for r in index_rows:
            w.writerow(r)
    print(f"\nwrote index -> {index_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

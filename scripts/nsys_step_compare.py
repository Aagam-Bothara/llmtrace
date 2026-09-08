"""Compare llmtrace's per-step CUDA-event spans with Nsight Systems kernel activity.

Input: an ``nsys export --type sqlite`` database from a run made with
``enable_nvtx=True`` (llmtrace pushes an ``llmtrace step <n>`` NVTX range around
every engine step), plus that run's directory (``gpu_steps_*.jsonl``,
``batches_*.jsonl``).

For every NVTX step range this computes, from ``CUPTI_ACTIVITY_KIND_KERNEL``
and ``CUPTI_ACTIVITY_KIND_GRAPH_TRACE``:

* ``gpu_busy_ms``: the union of kernel execution intervals and CUDA-graph
  execution intervals overlapping the range (GPU busy time, what CUPTI/Nsight
  measures and llmtrace cannot). vLLM runs decode steps as CUDA graphs, and
  with Nsight's default ``--cuda-graph-trace=graph`` a graph execution is one
  record in the graph-trace table and its kernels are absent from the kernel
  table, so both tables are needed (with ``--cuda-graph-trace=node`` the
  kernels appear individually instead; the union handles either);
* ``kernels`` / ``graph_launches``: record counts;
* ``nvtx_range_ms``: the range's own duration (host side).

and joins them with llmtrace's ``gpu_span_ms`` and ``host_step_ms`` by step
index. ``gpu_span_ms`` is defined as an upper bound on busy time (it includes
launch gaps on the stream); the ratio ``gpu_busy / gpu_span`` says how tight
that bound is on this workload. Steps whose ranges contain no GPU work are
reported, not dropped.

    nsys profile -t cuda,nvtx -o run python experiments/mixed_prompts/run.py ... --enable-nvtx
    nsys export --type sqlite -o run.sqlite run.nsys-rep
    python scripts/nsys_step_compare.py run.sqlite <run_dir> --json compare.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

STEP_RE = re.compile(r"llmtrace step (\d+)")


def _columns(con: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _tables(con: sqlite3.Connection) -> List[str]:
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def load_nvtx_steps(con: sqlite3.Connection) -> Dict[int, Tuple[int, int]]:
    """step_index -> (start_ns, end_ns) for llmtrace step ranges."""
    tables = _tables(con)
    if "NVTX_EVENTS" not in tables:
        raise SystemExit("no NVTX_EVENTS table: profile with -t nvtx and run with enable_nvtx=True")
    cols = _columns(con, "NVTX_EVENTS")
    strings: Dict[int, str] = {}
    if "StringIds" in tables:
        strings = {r[0]: r[1] for r in con.execute("SELECT id, value FROM StringIds")}
    text_expr = "text" if "text" in cols else "NULL"
    tid_expr = "textId" if "textId" in cols else "NULL"
    out: Dict[int, Tuple[int, int]] = {}
    for start, end, text, text_id in con.execute(f"SELECT start, end, {text_expr}, {tid_expr} FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        label = text if text else strings.get(text_id) if text_id is not None else None
        if not label:
            continue
        m = STEP_RE.search(str(label))
        if m:
            out[int(m.group(1))] = (int(start), int(end))
    return out


def load_gpu_work(con: sqlite3.Connection) -> List[Tuple[int, int, str]]:
    """(start_ns, end_ns, kind) for kernels and whole-graph executions, sorted by start."""
    tables = _tables(con)
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise SystemExit("no CUPTI_ACTIVITY_KIND_KERNEL table: profile with -t cuda")
    work = [(int(s), int(e), "kernel") for s, e in con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        work += [(int(s), int(e), "graph") for s, e in con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
    work.sort()
    return work


def busy_union_ns(work: List[Tuple[int, int, str]], lo: int, hi: int) -> Tuple[int, int, int]:
    """(union of work intervals clipped to [lo, hi], kernel count, graph count) using a sorted sweep."""
    total, kernels, graphs = 0, 0, 0
    cur_s = cur_e = None
    for s, e, kind in work:
        if e <= lo:
            continue
        if s >= hi:
            break
        s, e = max(s, lo), min(e, hi)
        if kind == "graph":
            graphs += 1
        else:
            kernels += 1
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total, kernels, graphs


def compare(sqlite_path: str, run_dir: str) -> Dict[str, Any]:
    from llmtrace import io

    con = sqlite3.connect(sqlite_path)
    ranges = load_nvtx_steps(con)
    work = load_gpu_work(con)
    spans = {g.step_index: g for g in io.load_gpu_steps([run_dir])}
    batches = {b.step_index: b for b in io.load_batches([run_dir])}
    rows: List[Dict[str, Any]] = []
    for idx in sorted(ranges):
        lo, hi = ranges[idx]
        busy, n_k, n_g = busy_union_ns(work, lo, hi)
        n = n_k + n_g
        g = spans.get(idx)
        b = batches.get(idx)
        row = {
            "step_index": idx,
            "nvtx_range_ms": (hi - lo) / 1e6,
            "gpu_busy_ms": busy / 1e6,
            "kernels": n_k,
            "graph_launches": n_g,
            "gpu_span_ms": g.gpu_span_ms if g else None,
            "host_step_ms": g.host_step_ms if g else None,
            "scheduled_tokens": b.total_scheduled_tokens if b else None,
            "biggest_chunk": max(b.scheduled_tokens.values()) if b and b.scheduled_tokens else None,
        }
        row["busy_over_span"] = (row["gpu_busy_ms"] / g.gpu_span_ms) if g and g.gpu_span_ms and n > 0 else None
        rows.append(row)
    with_both = [r for r in rows if r["busy_over_span"] is not None]
    ratios = [r["busy_over_span"] for r in with_both]
    long_steps = [r for r in with_both if (r["biggest_chunk"] or 0) > 128]
    summary = {
        "nvtx_steps": len(ranges),
        "steps_with_span_and_kernels": len(with_both),
        "steps_without_gpu_work": sum(1 for r in rows if r["kernels"] + r["graph_launches"] == 0),
        "gpu_busy_ms_p50": statistics.median(r["gpu_busy_ms"] for r in with_both) if with_both else None,
        "gpu_span_ms_p50": statistics.median(r["gpu_span_ms"] for r in with_both) if with_both else None,
        "busy_over_span_p50": statistics.median(ratios) if ratios else None,
        "busy_over_span_min": min(ratios) if ratios else None,
        "busy_over_span_max": max(ratios) if ratios else None,
        "span_exceeds_busy_all_steps": all(r["gpu_span_ms"] + 0.05 >= r["gpu_busy_ms"] for r in with_both) if with_both else None,
        "span_minus_busy_ms_p50": statistics.median(r["gpu_span_ms"] - r["gpu_busy_ms"] for r in with_both) if with_both else None,
        "long_chunk_steps": len(long_steps),
        "long_chunk_gpu_busy_ms_p50": statistics.median(r["gpu_busy_ms"] for r in long_steps) if long_steps else None,
        "long_chunk_gpu_span_ms_p50": statistics.median(r["gpu_span_ms"] for r in long_steps) if long_steps else None,
        "kernels_per_step_p50": statistics.median(r["kernels"] for r in with_both) if with_both else None,
        "graph_launches_per_step_p50": statistics.median(r["graph_launches"] for r in with_both) if with_both else None,
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("run_dir")
    ap.add_argument("--json", help="write full comparison here")
    args = ap.parse_args()
    res = compare(args.sqlite, args.run_dir)
    s = res["summary"]
    print(json.dumps(s, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

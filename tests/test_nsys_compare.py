"""nsys sqlite comparison: interval union, NVTX label resolution, join with llmtrace step spans."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nsys_step_compare import busy_union_ns, compare  # noqa: E402

from llmtrace import io  # noqa: E402
from llmtrace.data_plane.cuda_timing import StepGpuTiming  # noqa: E402
from llmtrace.models.trace import BatchMetadata  # noqa: E402


def test_busy_union_merges_overlaps_and_clips():
    k = [(0, 10, "kernel"), (5, 20, "kernel"), (30, 40, "graph"), (50, 60, "kernel")]
    assert busy_union_ns(k, 0, 100) == (40, 3, 1)  # [0,20] + [30,40] + [50,60]
    assert busy_union_ns(k, 15, 35) == (10, 1, 1)  # clipped: [15,20] + [30,35]
    assert busy_union_ns(k, 70, 90) == (0, 0, 0)


def _db(path, ranges, kernels, use_string_ids, graphs=()):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, textId INT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, streamId INT)")
    if graphs:
        con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INT, end INT, graphId INT)")
        for s, e in graphs:
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?, 1)", (s, e))
    if use_string_ids:
        con.execute("CREATE TABLE StringIds (id INT, value TEXT)")
        for i, (label, s, e) in enumerate(ranges):
            con.execute("INSERT INTO StringIds VALUES (?, ?)", (i, label))
            con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, ?)", (s, e, i))
    else:
        for label, s, e in ranges:
            con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)", (s, e, label))
    for s, e in kernels:
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 7)", (s, e))
    con.commit()
    con.close()


@pytest.mark.parametrize("use_string_ids", [False, True])
def test_compare_joins_steps(tmp_path, use_string_ids):
    ms = 1_000_000
    ranges = [("llmtrace step 1", 0, 3 * ms), ("llmtrace step 2", 3 * ms, 6 * ms), ("other range", 6 * ms, 7 * ms), ("llmtrace step 3", 7 * ms, 9 * ms)]
    kernels = [(int(0.2 * ms), int(1.2 * ms)), (int(1.0 * ms), int(2.0 * ms)),  # step 1: union 1.8 ms
               (int(3.5 * ms), int(5.5 * ms))]                                  # step 2: 2.0 ms; step 3: none
    db = tmp_path / "run.sqlite"
    _db(str(db), ranges, kernels, use_string_ids)
    run = tmp_path / "run"
    run.mkdir()
    io.write_jsonl(run / "gpu_steps_x.jsonl", [StepGpuTiming(step_index=i, timestamp=0, host_step_ms=3.0, gpu_span_ms=2.5) for i in (1, 2, 3)])
    io.write_jsonl(run / "batches_x.jsonl", [
        BatchMetadata(batch_id=f"b{i}", step_index=i, timestamp=0, num_requests=1, num_prefill=0, num_decode=1,
                      total_scheduled_tokens=t, request_ids=["r"], scheduled_tokens={"r": t}) for i, t in ((1, 1), (2, 1500), (3, 1))])
    res = compare(str(db), str(run))
    rows = {r["step_index"]: r for r in res["rows"]}
    assert set(rows) == {1, 2, 3}
    assert rows[1]["gpu_busy_ms"] == pytest.approx(1.8) and rows[1]["kernels"] == 2 and rows[1]["busy_over_span"] == pytest.approx(0.72)
    assert rows[2]["gpu_busy_ms"] == pytest.approx(2.0) and rows[2]["biggest_chunk"] == 1500
    assert rows[3]["kernels"] == 0 and rows[3]["busy_over_span"] is None
    s = res["summary"]
    assert s["nvtx_steps"] == 3 and s["steps_with_span_and_kernels"] == 2 and s["steps_without_gpu_work"] == 1
    assert s["span_exceeds_busy_all_steps"] is True and s["long_chunk_steps"] == 1
    assert s["long_chunk_gpu_busy_ms_p50"] == pytest.approx(2.0)
    assert s["span_minus_busy_ms_p50"] == pytest.approx(0.6)  # median of (2.5-1.8, 2.5-2.0)


def test_graph_executions_count_as_busy(tmp_path):
    """vLLM decode steps run as CUDA graphs: Nsight records the whole graph in GRAPH_TRACE, not per kernel."""
    ms = 1_000_000
    ranges = [("llmtrace step 1", 0, 2 * ms)]
    kernels = [(int(0.1 * ms), int(0.2 * ms)), (int(1.5 * ms), int(1.6 * ms))]  # sampler/prep kernels: 0.2 ms
    graphs = [(int(0.3 * ms), int(1.3 * ms))]  # the model forward as one graph launch: 1.0 ms
    db = tmp_path / "run.sqlite"
    _db(str(db), ranges, kernels, False, graphs=graphs)
    run = tmp_path / "run"
    run.mkdir()
    io.write_jsonl(run / "gpu_steps_x.jsonl", [StepGpuTiming(step_index=1, timestamp=0, host_step_ms=2.0, gpu_span_ms=1.5)])
    res = compare(str(db), str(run))
    row = res["rows"][0]
    assert row["gpu_busy_ms"] == pytest.approx(1.2) and row["kernels"] == 2 and row["graph_launches"] == 1
    assert row["busy_over_span"] == pytest.approx(0.8)
    assert res["summary"]["graph_launches_per_step_p50"] == 1

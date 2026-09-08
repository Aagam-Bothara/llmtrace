"""Chrome trace export and HTML report on synthetic runs (CPU-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner
from fakes import FakeClock, FakeLLMEngine, FakeNVMLBackend, SamplingParams, run_to_completion

from llmtrace import LLMTracer, TracerConfig
from llmtrace.cli import main
from llmtrace.visualize import RunData, export_chrome_trace, render_html_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "mixed_prompts"))


def _record(tmp_path, name):
    clock = FakeClock()
    tracer = LLMTracer(TracerConfig(output_dir=str(tmp_path / name), collection_interval_s=0.02, gpu_sampler={"sample_interval_ms": 10}),
                       gpu_backend=FakeNVMLBackend({0: 120.0}))
    tracer.vllm_instrumentation._monotonic = clock.monotonic
    tracer.vllm_instrumentation._wall = clock.time
    engine = FakeLLMEngine(clock=clock, step_seconds=0.002, step_seconds_per_token=5e-6, prefill_chunk=512)
    tracer.instrument_engine(engine)
    engine.add_request("short-0", {"prompt_token_ids": [1, 2, 3]}, SamplingParams(max_tokens=5))
    engine.add_request("long-0", {"prompt_token_ids": list(range(1000))}, SamplingParams(max_tokens=2))
    run_to_completion(engine)
    tracer.stop()
    return tmp_path / name


def test_chrome_trace_structure(tmp_path):
    d = _record(tmp_path, "a")
    run = RunData.load(str(d))
    out = tmp_path / "trace.json"
    counts = export_chrome_trace(run, str(out))
    doc = json.loads(out.read_text())
    ev = doc["traceEvents"]
    assert counts["requests"] == 2 and counts["steps"] >= 5
    steps = [e for e in ev if e.get("cat") == "step"]
    assert all(e["ph"] == "X" and e["dur"] >= 0 and "scheduled_tokens" in e["args"] for e in steps)
    assert any(e["args"]["biggest_chunk"] == 512 for e in steps)  # chunked long prefill visible
    names = {e["args"]["name"] for e in ev if e.get("name") == "thread_name"}
    assert {"short-0", "long-0", "scheduler steps"} <= names
    assert any(e["ph"] == "C" and "power" in e["name"] for e in ev)  # GPU counter track
    spans = [e for e in ev if e.get("cat") == "span"]
    assert {e["name"] for e in spans} >= {"queue", "prefill", "decode"}
    assert doc["metadata"]["llmtrace_clock"] in ("monotonic", "wall")


def test_html_report_and_cli(tmp_path):
    a = _record(tmp_path, "a")
    b = _record(tmp_path, "b")
    out = tmp_path / "report.html"
    render_html_report(RunData.load(str(a), "base"), str(out), RunData.load(str(b), "cand"))
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text and "Request timeline" in text and "Comparison: cand vs base" in text
    assert "short-0" in text and "long" in text
    r = CliRunner().invoke(main, ["visualize", str(a), "--compare", str(b), "--html-out", str(tmp_path / "r.html"),
                                  "--trace-out", str(tmp_path / "t.json")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "r.html").exists() and (tmp_path / "t.json").exists()
    r = CliRunner().invoke(main, ["visualize", str(a)])
    assert r.exit_code == 2

"""Visualization exports for recorded runs (offline; no GPU, vLLM or extra dependencies).

Two outputs:

* ``export_chrome_trace``: Trace Event Format JSON. Open it at https://ui.perfetto.dev
  (or chrome://tracing). Tracks: engine steps (one slice per scheduler step, args
  carry scheduled tokens, prefill/decode counts, KV usage, request ids), one
  thread per request with its lifecycle spans, and counter tracks for GPU power,
  utilization and KV-cache usage.
* ``render_html_report``: a single self-contained HTML file with inline SVG:
  request timeline (Gantt, colored by request class and phase), step durations
  over time, step duration vs scheduled tokens, GPU power, and latency tables.
  Optionally a second run for side-by-side comparison.

Times use the monotonic clock when every record carries one (same tracer
session), otherwise wall clock; the choice is written into the output.
"""

from __future__ import annotations

import html
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llmtrace import io
from llmtrace.control_plane.reporter import percentile
from llmtrace.control_plane.steps import per_request_intervals
from llmtrace.data_plane.vllm_stats import VLLMIterationRecord
from llmtrace.models.trace import BatchMetadata, GPUSample, RequestTrace

PHASE_COLORS = {"queue": "#9aa3ad", "prefill": "#d98500", "time_to_first_token": "#b89b00", "decode": "#2a6fdb"}
CLASS_COLORS = ["#2a6fdb", "#d2452f", "#2f9e63", "#7a4fc9", "#d98500", "#0e8a86"]
LONG_CHUNK = "#d2452f"
BAR = "#2a6fdb"

CSS = """
:root{--bg:#f6f7f9;--surface:#ffffff;--ink:#1c2430;--muted:#5b6675;--rule:#d5dae1;--grid:#e3e7ec;--accent:#2a6fdb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f141a;--surface:#161c24;--ink:#e6ebf1;--muted:#98a3b0;--rule:#2b3542;--grid:#232c37;--accent:#5b93ea}}
:root[data-theme="dark"]{--bg:#0f141a;--surface:#161c24;--ink:#e6ebf1;--muted:#98a3b0;--rule:#2b3542;--grid:#232c37;--accent:#5b93ea}
body{font-family:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;font-size:15px;line-height:1.5;color:var(--ink);background:var(--bg);margin:0}
main{max-width:1040px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:1.75rem;font-weight:600;line-height:1.2;text-wrap:balance;margin:0 0 6px}
h2{font-size:1.25rem;font-weight:600;margin:40px 0 4px;padding-top:16px;border-top:1px solid var(--rule)}
h3{font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);margin:24px 0 6px}
p{max-width:68ch;margin:6px 0;color:var(--muted)}
.lede{color:var(--ink)}
table{border-collapse:collapse;margin:8px 0 4px;font-family:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace;font-size:0.82rem;font-variant-numeric:tabular-nums;background:var(--surface)}
th,td{border:1px solid var(--rule);padding:4px 10px;text-align:right}
th{font-weight:500;color:var(--muted)}
th:first-child,td:first-child{text-align:left;font-family:'IBM Plex Sans',system-ui,sans-serif}
.table-wrap{overflow-x:auto}
svg{display:block;margin:6px 0 2px;background:var(--surface);border:1px solid var(--rule)}
svg text{fill:var(--ink);font-family:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace;font-size:10px}
svg .grid{stroke:var(--grid)}
svg .axis{fill:var(--muted)}
.legend{font-size:0.8rem;color:var(--muted);margin:2px 0 0}
.legend b{font-weight:600}
"""


class RunData:
    """Loaded run with a consistent time base."""

    def __init__(self, traces: List[RequestTrace], batches: List[BatchMetadata], samples: List[GPUSample], label: str = "run",
                 vllm_stats: Optional[List[VLLMIterationRecord]] = None):
        self.traces, self.batches, self.samples, self.label = traces, batches, samples, label
        self.vllm_stats = vllm_stats or []
        mono_ok = (
            traces and all(t.start_monotonic is not None and t.end_monotonic is not None for t in traces)
            and all(b.monotonic is not None for b in batches) and all(s.monotonic is not None for s in samples)
            and len({t.clock_domain for t in traces} | {s.clock_domain for s in samples}) <= 1
        )
        self.clock = "monotonic" if mono_ok else "wall"
        starts = [self.t_start(t) for t in traces] + [self.b_start(b) for b in batches]
        self.t0 = min(starts) if starts else 0.0

    @classmethod
    def load(cls, run_dir: str, label: Optional[str] = None) -> "RunData":
        d = Path(run_dir)
        return cls(io.load_traces([d]), io.load_batches([d]), io.load_gpu_samples([d]), label or d.name, io.load_vllm_stats([d]))

    # --- time accessors (seconds, absolute in the chosen clock)
    def t_start(self, t: RequestTrace) -> float:
        return t.start_monotonic if self.clock == "monotonic" else t.start_time

    def t_end(self, t: RequestTrace) -> float:
        return t.end_monotonic if self.clock == "monotonic" else t.end_time

    def span_bounds(self, s: Any) -> Tuple[float, float]:
        if self.clock == "monotonic" and s.start_monotonic is not None and s.end_monotonic is not None:
            return s.start_monotonic, s.end_monotonic
        return s.start_time, s.end_time

    def b_start(self, b: BatchMetadata) -> float:
        return b.monotonic if self.clock == "monotonic" else b.timestamp

    def b_end(self, b: BatchMetadata) -> Optional[float]:
        if b.step_end_monotonic is None or b.monotonic is None:
            return None
        return b.step_end_monotonic if self.clock == "monotonic" else b.timestamp + (b.step_end_monotonic - b.monotonic)

    def s_time(self, s: GPUSample) -> float:
        return s.monotonic if self.clock == "monotonic" and s.monotonic is not None else s.timestamp

    def v_time(self, r: VLLMIterationRecord) -> float:
        return r.monotonic if self.clock == "monotonic" and r.monotonic is not None else r.timestamp

    def rel(self, t: float) -> float:
        return t - self.t0

    def request_class(self, t: RequestTrace) -> str:
        return t.request_id.split("-")[0] if "-" in t.request_id else "request"


# ----------------------------------------------------------------- chrome trace

def export_chrome_trace(run: RunData, out_path: str) -> Dict[str, int]:
    """Write a Trace Event Format JSON file. Returns event counts."""
    ev: List[Dict[str, Any]] = []
    us = 1e6

    def meta(pid: int, tid: Optional[int], name: str) -> None:
        e: Dict[str, Any] = {"ph": "M", "pid": pid, "args": {"name": name}}
        if tid is None:
            e["name"] = "process_name"
        else:
            e["name"], e["tid"] = "thread_name", tid
        ev.append(e)

    meta(1, None, "engine")
    meta(1, 1, "scheduler steps")
    for b in run.batches:
        end = run.b_end(b)
        if end is None:
            continue
        ev.append({
            "name": f"step {b.step_index}", "cat": "step", "ph": "X", "pid": 1, "tid": 1,
            "ts": run.rel(run.b_start(b)) * us, "dur": max(end - run.b_start(b), 0.0) * us,
            "args": {"scheduled_tokens": b.total_scheduled_tokens, "num_requests": b.num_requests,
                     "num_prefill": b.num_prefill, "num_decode": b.num_decode,
                     "biggest_chunk": max(b.scheduled_tokens.values()) if b.scheduled_tokens else 0,
                     "kv_cache_usage": b.kv_cache_usage_fraction, "request_ids": ",".join(b.request_ids)},
        })
        if b.kv_cache_usage_fraction is not None:
            ev.append({"name": "kv cache usage", "ph": "C", "pid": 1, "ts": run.rel(run.b_start(b)) * us,
                       "args": {"fraction": b.kv_cache_usage_fraction}})
        ev.append({"name": "scheduled tokens", "ph": "C", "pid": 1, "ts": run.rel(run.b_start(b)) * us,
                   "args": {"tokens": b.total_scheduled_tokens}})

    meta(2, None, "requests")
    for i, t in enumerate(sorted(run.traces, key=run.t_start)):
        tid = i + 1
        meta(2, tid, t.request_id)
        base = {"pid": 2, "tid": tid}
        ev.append({**base, "name": t.request_id, "cat": run.request_class(t), "ph": "X",
                   "ts": run.rel(run.t_start(t)) * us, "dur": max(run.t_end(t) - run.t_start(t), 0.0) * us,
                   "args": {"status": t.status.value, "prompt_tokens": t.prompt_length, "output_tokens": t.output_length,
                            "ttft_ms": t.ttft_ms, "tpot_ms": t.tpot_ms, "batches": len(t.batch_ids),
                            "attributed_joules": t.energy.attributed_joules if t.energy else None}})
        for s in t.spans:
            a, b_ = run.span_bounds(s)
            ev.append({**base, "name": s.phase.value, "cat": "span", "ph": "X", "ts": run.rel(a) * us,
                       "dur": max(b_ - a, 0.0) * us, "args": dict(s.metadata)})

    meta(3, None, "gpu")
    for s in sorted(run.samples, key=run.s_time):
        ts = run.rel(run.s_time(s)) * us
        if s.power_draw_watts is not None:
            ev.append({"name": f"gpu{s.gpu_id} power (W)", "ph": "C", "pid": 3, "ts": ts, "args": {"W": s.power_draw_watts}})
        if s.gpu_utilization_pct is not None:
            ev.append({"name": f"gpu{s.gpu_id} utilization (%)", "ph": "C", "pid": 3, "ts": ts, "args": {"pct": s.gpu_utilization_pct}})

    if run.vllm_stats:
        meta(4, None, "vllm stats")
        for r in sorted(run.vllm_stats, key=run.v_time):
            ts = run.rel(run.v_time(r)) * us
            for name, val in (("vllm kv cache usage", r.kv_cache_usage), ("vllm waiting", r.num_waiting_reqs),
                              ("vllm running", r.num_running_reqs), ("vllm preempted", r.num_preempted_reqs)):
                if val is not None:
                    ev.append({"name": name, "ph": "C", "pid": 4, "ts": ts, "args": {"v": val}})

    doc = {"traceEvents": ev, "displayTimeUnit": "ms",
           "metadata": {"llmtrace_clock": run.clock, "run": run.label, "t0": run.t0}}
    Path(out_path).write_text(json.dumps(doc), encoding="utf-8")
    return {"events": len(ev), "steps": sum(1 for e in ev if e.get("cat") == "step"), "requests": len(run.traces)}


# --------------------------------------------------------------------- html

def _fmt(v: Optional[float], d: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def _latency_rows(run: RunData) -> List[Dict[str, Any]]:
    step_ms, itl_ms = per_request_intervals(run.traces, run.batches)
    by: Dict[str, List[RequestTrace]] = {}
    for t in run.traces:
        by.setdefault(run.request_class(t), []).append(t)
    rows = []
    for cls, ts in sorted(by.items()):
        ttft = [t.ttft_ms for t in ts if t.ttft_ms is not None]
        tpot = [t.tpot_ms for t in ts if t.tpot_ms is not None]
        itl = [v for t in ts for v in itl_ms.get(t.request_id, [])]
        steps = [v for t in ts for v in step_ms.get(t.request_id, [])]
        rows.append({"class": cls, "n": len(ts), "ttft_p50": percentile(ttft, 50), "ttft_p95": percentile(ttft, 95),
                     "tpot_p50": percentile(tpot, 50), "tpot_p95": percentile(tpot, 95),
                     "itl_p99": percentile(itl, 99), "itl_max": max(itl) if itl else None,
                     "step_max": max(steps) if steps else None})
    return rows


def _svg_gantt(run: RunData, width: int = 960) -> str:
    traces = sorted(run.traces, key=run.t_start)
    if not traces:
        return "<p>no requests</p>"
    t_max = max(run.rel(run.t_end(t)) for t in traces) or 1.0
    row_h = max(3, min(10, 600 // max(len(traces), 1)))
    left = 130
    h = row_h * len(traces) + 34
    sx = (width - left - 10) / t_max
    classes = sorted({run.request_class(t) for t in traces})
    color = {c: CLASS_COLORS[i % len(CLASS_COLORS)] for i, c in enumerate(classes)}
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="request timeline">']
    for i, t in enumerate(traces):
        y = 20 + i * row_h
        a, b = run.rel(run.t_start(t)), run.rel(run.t_end(t))
        out.append(f'<rect x="{left + a * sx:.1f}" y="{y}" width="{max((b - a) * sx, 1):.1f}" height="{row_h - 1}" '
                   f'fill="{color[run.request_class(t)]}" opacity="0.35"><title>{html.escape(t.request_id)}: '
                   f'TTFT {_fmt(t.ttft_ms)} ms, TPOT {_fmt(t.tpot_ms)} ms, {t.output_length} tokens</title></rect>')
        for s in t.spans:
            sa, sb = (run.rel(x) for x in run.span_bounds(s))
            out.append(f'<rect x="{left + sa * sx:.1f}" y="{y}" width="{max((sb - sa) * sx, 0.5):.1f}" height="{row_h - 1}" '
                       f'fill="{PHASE_COLORS.get(s.phase.value, "#000")}" opacity="0.9"><title>{s.phase.value} {s.duration_ms:.2f} ms</title></rect>')
        if row_h >= 8:
            out.append(f'<text x="2" y="{y + row_h - 2}">{html.escape(t.request_id[:22])}</text>')
    for k in range(0, 11):
        x = left + t_max * k / 10 * sx
        out.append(f'<line class="grid" x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{h - 14}"/><text class="axis" x="{x:.1f}" y="10" text-anchor="middle">{t_max * k / 10:.2f}s</text>')
    out.append("</svg>")
    legend = " ".join(f'<span style="color:{c}">&#9632;</span> {html.escape(k)}' for k, c in color.items())
    phases = " ".join(f'<span style="color:{c}">&#9632;</span> {k}' for k, c in PHASE_COLORS.items())
    out.append(f'<p class="legend"><b>request class</b> {legend} &nbsp; <b>phase</b> {phases}</p>')
    return "".join(out)


def _svg_series(points: Sequence[Tuple[float, float]], width: int, height: int, xlabel: str, ylabel: str,
                colors: Optional[Sequence[str]] = None, kind: str = "bar", titles: Optional[Sequence[str]] = None) -> str:
    if not points:
        return "<p>no data</p>"
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1 = min(xs), max(xs) or 1.0
    y1 = max(ys) or 1.0
    left, bottom = 56, 28
    sx = (width - left - 10) / max(x1 - x0, 1e-9)
    sy = (height - bottom - 10) / y1
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(ylabel)} vs {html.escape(xlabel)}">']
    for k in range(5):
        yv = y1 * k / 4
        yy = height - bottom - yv * sy
        out.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width - 10}" y2="{yy:.1f}"/><text class="axis" x="{left - 4}" y="{yy + 3:.1f}" text-anchor="end">{yv:.3g}</text>')
    for k in range(6):
        xv = x0 + (x1 - x0) * k / 5
        xx = left + (xv - x0) * sx
        out.append(f'<text class="axis" x="{xx:.1f}" y="{height - bottom + 12}" text-anchor="middle">{xv:.3g}</text>')
    if kind == "line":
        pts = " ".join(f"{left + (x - x0) * sx:.1f},{height - bottom - y * sy:.1f}" for x, y in points)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{BAR}" stroke-width="1.2"/>')
    else:
        for i, (x, y) in enumerate(points):
            c = colors[i] if colors else BAR
            title = f"<title>{html.escape(titles[i])}</title>" if titles else ""
            out.append(f'<rect x="{left + (x - x0) * sx - 0.6:.1f}" y="{height - bottom - y * sy:.1f}" width="1.2" height="{y * sy:.1f}" fill="{c}">{title}</rect>')
    out.append(f'<text class="axis" x="{width - 10}" y="{height - 4}" text-anchor="end">{html.escape(xlabel)}</text>')
    out.append(f'<text class="axis" x="{left}" y="10">{html.escape(ylabel)}</text></svg>')
    return "".join(out)


def _steps(run: RunData) -> List[Dict[str, Any]]:
    out = []
    for b in run.batches:
        end = run.b_end(b)
        if end is None:
            continue
        out.append({"t": run.rel(run.b_start(b)), "dur": (end - run.b_start(b)) * 1000, "tokens": b.total_scheduled_tokens,
                    "biggest": max(b.scheduled_tokens.values()) if b.scheduled_tokens else 0, "idx": b.step_index})
    return out


def _run_section(run: RunData, chunk_threshold: int) -> str:
    steps = _steps(run)
    colors = [LONG_CHUNK if s["biggest"] > chunk_threshold else BAR for s in steps]
    titles = [f"step {s['idx']}: {s['dur']:.2f} ms, {s['tokens']} tokens (biggest chunk {s['biggest']})" for s in steps]
    rows = _latency_rows(run)
    table = ["<table><tr><th>class</th><th>n</th><th>TTFT p50</th><th>TTFT p95</th><th>TPOT p50</th><th>TPOT p95</th>"
             "<th>ITL p99</th><th>ITL max</th><th>step max</th></tr>"]
    for r in rows:
        table.append(f"<tr><td>{html.escape(r['class'])}</td><td>{r['n']}</td><td>{_fmt(r['ttft_p50'])}</td><td>{_fmt(r['ttft_p95'])}</td>"
                     f"<td>{_fmt(r['tpot_p50'])}</td><td>{_fmt(r['tpot_p95'])}</td><td>{_fmt(r['itl_p99'])}</td><td>{_fmt(r['itl_max'])}</td>"
                     f"<td>{_fmt(r['step_max'])}</td></tr>")
    table.append("</table>")
    table = ['<div class="table-wrap">'] + table + ["</div>"]
    power = [(run.rel(run.s_time(s)), s.power_draw_watts) for s in sorted(run.samples, key=run.s_time) if s.power_draw_watts is not None]
    med = statistics.median(s["dur"] for s in steps) if steps else None
    parts = [
        f"<h2>{html.escape(run.label)}</h2>",
        f"<p class='lede'>{len(run.traces)} requests, {len(steps)} scheduler steps, {len(run.samples)} GPU samples; "
        f"clock: {run.clock}" + (f"; median step {med:.2f} ms" if med else "") + ". All latencies in milliseconds.</p>",
        "".join(table),
        "<p>ITL: intervals between a request's successive step ends from its first token onward (real inter-token latency). "
        "step: duration of the engine steps the request was scheduled in (compute proxy).</p>",
        "<h3>Request timeline</h3>", _svg_gantt(run),
        "<h3>Step duration over time</h3><p>Red bars: the step carried a single prefill chunk larger than "
        f"{chunk_threshold} tokens. Every short request decoding in that step waits for it.</p>",
        _svg_series([(s["t"], s["dur"]) for s in steps], 960, 200, "time (s)", "step ms", colors, "bar", titles) if steps else "<p>no batch metadata (in-process scheduler required)</p>",
        "<h3>Step duration vs scheduled tokens</h3>",
        _svg_series(sorted((s["tokens"], s["dur"]) for s in steps), 960, 200, "scheduled tokens", "step ms", None, "bar") if steps else "",
        "<h3>GPU power</h3>",
        _svg_series(power, 960, 160, "time (s)", "W", kind="line") if power else "<p>no power samples</p>",
    ]
    if run.vllm_stats:
        from llmtrace.control_plane.reporter import summarize_vllm_stats
        vs = sorted(run.vllm_stats, key=run.v_time)
        summ = summarize_vllm_stats(vs)
        kv = [(run.rel(run.v_time(r)), r.kv_cache_usage) for r in vs if r.kv_cache_usage is not None]
        waiting = [(run.rel(run.v_time(r)), float(r.num_waiting_reqs)) for r in vs if r.num_waiting_reqs is not None]
        parts += ["<h3>vLLM engine stats (stat_loggers hook)</h3>", f"<p class='lede'>{html.escape(summ['text'])}</p>"]
        if kv:
            parts.append(_svg_series(kv, 960, 140, "time (s)", "KV cache usage (fraction)", kind="line"))
        if waiting:
            parts.append(_svg_series(waiting, 960, 140, "time (s)", "requests waiting", kind="line"))
    return "\n".join(parts)


def render_html_report(run: RunData, out_path: str, compare: Optional[RunData] = None, chunk_threshold: int = 128,
                       title: str = "llmtrace run report") -> None:
    body = [f"<h1>{html.escape(title)}</h1>",
            "<p>Generated by llmtrace from recorded trace files. Step-level charts need batch metadata "
            "(in-process scheduler). Latencies are measured at engine-step boundaries.</p>",
            _run_section(run, chunk_threshold)]
    if compare is not None:
        body.append(_run_section(compare, chunk_threshold))
        rows_a = {r["class"]: r for r in _latency_rows(run)}
        rows_b = {r["class"]: r for r in _latency_rows(compare)}
        body.append(f"<h2>Comparison: {html.escape(compare.label)} vs {html.escape(run.label)}</h2>"
                    "<p>Change is relative to the first run; negative means faster.</p><div class='table-wrap'><table>"
                    "<tr><th>class / metric</th><th>" + html.escape(run.label) + "</th><th>" + html.escape(compare.label) + "</th><th>change</th></tr>")
        for cls in sorted(set(rows_a) | set(rows_b)):
            for m in ("ttft_p95", "tpot_p95", "itl_p99", "itl_max"):
                a, b = rows_a.get(cls, {}).get(m), rows_b.get(cls, {}).get(m)
                ch = f"{(b - a) / a * 100:+.1f}%" if a and b is not None else "n/a"
                body.append(f"<tr><td>{html.escape(cls)} {m}</td><td>{_fmt(a)}</td><td>{_fmt(b)}</td><td>{ch}</td></tr>")
        body.append("</table></div>")
    head = ("<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap'>"
            f"<style>{CSS}</style>")
    doc = f"<!doctype html><html><head>{head}</head><body><main>{''.join(body)}</main></body></html>"
    Path(out_path).write_text(doc, encoding="utf-8")

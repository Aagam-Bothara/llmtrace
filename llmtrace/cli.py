"""Command-line interface for llmtrace (offline analysis; no GPU or vLLM needed)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click

from llmtrace import __version__, io
from llmtrace.control_plane.correlator import Correlator, CorrelationResult
from llmtrace.control_plane.reporter import Reporter
from llmtrace.control_plane.rules_engine import RulesEngine
from llmtrace.models.config import AutopsyConfig, EnergyConfig, ReporterConfig, TracerConfig
from llmtrace.models.trace import MetricComparison

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable INFO logging")
def main(verbose: bool) -> None:
    """llmtrace - flight recorder, attribution and autopsy for vLLM inference."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _correlate_dir(
    trace_paths: List[str], gpu_paths: Optional[List[str]], attribution: str
) -> CorrelationResult:
    traces = io.load_traces(trace_paths)
    dirs = io.run_directories_for(trace_paths)
    samples = io.load_gpu_samples(gpu_paths if gpu_paths else dirs)
    batches = io.load_batches(dirs)
    correlator = Correlator(EnergyConfig(attribution_method=attribution))  # type: ignore[arg-type]
    return correlator.correlate(traces, samples, batches)


@main.command()
@click.argument("trace_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--gpu-samples", "gpu_paths", multiple=True, type=click.Path(exists=True),
              help="GPU sample file(s)/dir(s); default: gpu_*.jsonl next to the trace files")
@click.option("--baseline", type=click.Path(exists=True), help="Baseline run directory for comparison")
@click.option("--attribution", default="equal_share",
              type=click.Choice(["equal_share", "proportional_tokens", "window_only"]))
@click.option("--output", type=click.Path(), help="Write report (.json for machine-readable, else text)")
@click.option("--no-rich", is_flag=True, help="Plain text output")
def analyze(trace_paths: Tuple[str, ...], gpu_paths: Tuple[str, ...], baseline: Optional[str],
            attribution: str, output: Optional[str], no_rich: bool) -> None:
    """Analyze trace files or run directories."""
    result = _correlate_dir(list(trace_paths), list(gpu_paths) or None, attribution)
    if not result.traces:
        click.echo("No traces found", err=True)
        sys.exit(EXIT_USAGE)
    rules = RulesEngine(AutopsyConfig())
    for t in result.traces:
        t.diagnosis = rules.diagnose_request(t)

    baseline_traces = baseline_ledger = None
    if baseline:
        b = _correlate_dir([baseline], None, attribution)
        baseline_traces, baseline_ledger = b.traces, b.ledger
        if not baseline_traces:
            click.echo(f"Warning: no baseline traces in {baseline}", err=True)

    vllm_stats = io.load_vllm_stats(io.run_directories_for(list(trace_paths)))
    if vllm_stats:
        from llmtrace.control_plane.reporter import summarize_vllm_stats
        result.ledger.notes.append("vLLM engine stats (stat_loggers hook): " + summarize_vllm_stats(vllm_stats)["text"])
    reporter = Reporter(ReporterConfig(cli_rich_output=not no_rich))
    analysis = reporter.generate_analysis(result.traces, baseline_traces, result.ledger, baseline_ledger)
    reporter.print_analysis(analysis)
    if output:
        reporter.export_to_file(analysis, Path(output))
        click.echo(f"Report written to {output}")


def evaluate_regressions(
    comparisons: dict, thresholds: dict
) -> Tuple[List[str], List[str], List[str]]:
    """Classify comparisons. Returns (regressions, improvements_or_ok, unavailable).

    A metric regresses only when its percent change is *positive* and exceeds
    the threshold (all compared metrics are higher-is-worse). Negative changes
    are improvements and never fail the check.
    """
    regressions, ok, unavailable = [], [], []
    for name, threshold in thresholds.items():
        cmp: Optional[MetricComparison] = comparisons.get(name)
        if cmp is None or cmp.pct_change is None:
            status = cmp.status if cmp is not None else "missing"
            unavailable.append(f"{name}: {status} (baseline={cmp.baseline if cmp else None}, "
                               f"current={cmp.current if cmp else None})")
            continue
        line = f"{name}: {cmp.pct_change:+.2f}% (threshold +{threshold}%)"
        if cmp.pct_change > threshold:
            regressions.append(line)
        else:
            ok.append(line)
    return regressions, ok, unavailable


@main.command()
@click.option("--baseline", required=True, type=click.Path(exists=True), help="Baseline run directory")
@click.option("--current", required=True, type=click.Path(exists=True), help="Current run directory")
@click.option("--ttft-threshold", default=5.0, type=float, help="Max allowed p95 TTFT increase (%)")
@click.option("--tpot-threshold", default=None, type=float, help="Max allowed p95 TPOT increase (%)")
@click.option("--energy-threshold", default=10.0, type=float, help="Max allowed J/output-token increase (%)")
@click.option("--attribution", default="equal_share",
              type=click.Choice(["equal_share", "proportional_tokens", "window_only"]))
@click.option("--fail-on-regression", is_flag=True, help="Exit 1 if any threshold is exceeded")
@click.option("--fail-on-missing", is_flag=True, help="Exit 1 if a thresholded metric is unavailable")
@click.option("--no-rich", is_flag=True, help="Plain text output")
def compare(baseline: str, current: str, ttft_threshold: float, tpot_threshold: Optional[float],
            energy_threshold: float, attribution: str, fail_on_regression: bool, fail_on_missing: bool,
            no_rich: bool) -> None:
    """Compare a current run against a baseline run (positive change = worse)."""
    base = _correlate_dir([baseline], None, attribution)
    cur = _correlate_dir([current], None, attribution)
    if not base.traces or not cur.traces:
        click.echo(f"Error: baseline has {len(base.traces)} traces, current has {len(cur.traces)}", err=True)
        sys.exit(EXIT_USAGE)

    reporter = Reporter(ReporterConfig(cli_rich_output=not no_rich))
    analysis = reporter.generate_analysis(cur.traces, base.traces, cur.ledger, base.ledger)
    reporter.print_analysis(analysis)

    thresholds = {"p95_ttft_ms": ttft_threshold, "joules_per_output_token": energy_threshold}
    if tpot_threshold is not None:
        thresholds["p95_tpot_ms"] = tpot_threshold
    regressions, ok, unavailable = evaluate_regressions(analysis.regressions, thresholds)

    click.echo("\nRegression check (positive change = worse):")
    for line in ok:
        click.echo(f"  ok         {line}")
    for line in unavailable:
        click.echo(f"  unavailable {line}")
    for line in regressions:
        click.echo(f"  REGRESSION {line}")

    code = EXIT_OK
    if regressions and fail_on_regression:
        code = EXIT_REGRESSION
    if unavailable and fail_on_missing:
        code = EXIT_REGRESSION
    if code != EXIT_OK:
        click.echo("FAILED")
    elif regressions:
        click.echo("Regressions found (not failing: --fail-on-regression not set)")
    else:
        click.echo("PASSED")
    sys.exit(code)


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--compare", "compare_dir", type=click.Path(exists=True), help="Second run directory for side-by-side report")
@click.option("--trace-out", type=click.Path(), help="Write a Chrome/Perfetto trace JSON here (open at ui.perfetto.dev)")
@click.option("--html-out", type=click.Path(), help="Write a self-contained HTML report here")
@click.option("--chunk-threshold", type=int, default=128, help="Highlight steps whose largest prefill chunk exceeds this")
@click.option("--title", default="llmtrace run report")
def visualize(run_dir: str, compare_dir: Optional[str], trace_out: Optional[str], html_out: Optional[str],
              chunk_threshold: int, title: str) -> None:
    """Export a Perfetto trace and/or an HTML report from a recorded run directory."""
    from llmtrace.visualize import RunData, export_chrome_trace, render_html_report

    if not trace_out and not html_out:
        click.echo("Nothing to do: pass --trace-out and/or --html-out", err=True)
        sys.exit(EXIT_USAGE)
    run = RunData.load(run_dir)
    if not run.traces:
        click.echo(f"No traces in {run_dir}", err=True)
        sys.exit(EXIT_USAGE)
    if trace_out:
        counts = export_chrome_trace(run, trace_out)
        click.echo(f"Perfetto trace written to {trace_out} ({counts['events']} events; open at https://ui.perfetto.dev)")
    if html_out:
        cmp = RunData.load(compare_dir) if compare_dir else None
        render_html_report(run, html_out, cmp, chunk_threshold, title)
        click.echo(f"HTML report written to {html_out}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--chunk-threshold", type=int, default=128, help="Prefill chunk size (tokens) that counts as 'long'")
@click.option("--queue-threshold-ms", type=float, default=100.0)
@click.option("--kv-threshold", type=float, default=0.9, help="KV-cache usage fraction that counts as pressure")
@click.option("--json", "json_out", type=click.Path(), help="Write findings JSON here")
@click.option("--verbose", "verbose", is_flag=True, help="Also print each check's assumptions, competing explanations and limits")
def findings(run_dir: str, chunk_threshold: int, queue_threshold_ms: float, kv_threshold: float, json_out: Optional[str],
             verbose: bool) -> None:
    """Evaluate the hypotheses on a recorded run: queue overload, long-prompt interference, KV pressure, host overhead, tracer self-effect."""
    from llmtrace.control_plane.findings import evaluate_all, format_findings

    traces = io.load_traces([run_dir])
    if not traces:
        click.echo(f"No traces in {run_dir}", err=True)
        sys.exit(EXIT_USAGE)
    result = evaluate_all(traces, io.load_batches([run_dir]), io.load_vllm_stats([run_dir]), io.load_collector_events([run_dir]),
                          chunk_threshold, queue_threshold_ms, kv_threshold, io.load_gpu_steps([run_dir]))
    click.echo(format_findings(result, verbose=verbose))
    if json_out:
        Path(json_out).write_text(json.dumps([f.model_dump() for f in result], indent=2), encoding="utf-8")
        click.echo(f"Findings written to {json_out}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--chunk-threshold", type=int, default=128, help="Prefill chunk size (tokens) that counts as 'long'")
@click.option("--queue-threshold-ms", type=float, default=100.0)
@click.option("--kv-threshold", type=float, default=0.9)
@click.option("--max-candidates", type=int, default=4, show_default=True)
@click.option("--repeats", type=int, default=3, show_default=True)
@click.option("--json", "json_out", type=click.Path(), help="Write the plan JSON here (input for `llmtrace run --plan`)")
def plan(run_dir: str, chunk_threshold: int, queue_threshold_ms: float, kv_threshold: float, max_candidates: int, repeats: int,
         json_out: Optional[str]) -> None:
    """From a recorded run's findings, propose a bounded set of configuration experiments (plans only; runs nothing)."""
    from llmtrace.control_plane.experiments import plan_experiments
    from llmtrace.control_plane.findings import evaluate_all
    from llmtrace.manifest import RunManifest

    traces = io.load_traces([run_dir])
    if not traces:
        click.echo(f"No traces in {run_dir}", err=True)
        sys.exit(EXIT_USAGE)
    batches = io.load_batches([run_dir])
    result = evaluate_all(traces, batches, io.load_vllm_stats([run_dir]), io.load_collector_events([run_dir]),
                          chunk_threshold, queue_threshold_ms, kv_threshold, io.load_gpu_steps([run_dir]))
    p = plan_experiments(result, RunManifest.read(run_dir), batches, max_candidates=max_candidates, repeats=repeats, source_run=run_dir)
    click.echo(p.format())
    if json_out:
        Path(json_out).write_text(p.model_dump_json(indent=2), encoding="utf-8")
        click.echo(f"Plan written to {json_out}; execute with: llmtrace run --plan {json_out} --workload <spec.json> --engine <fake|vllm> --out <dir>")


@main.command()
@click.option("--target", required=True, help="e.g. 'short ttft_p95 <= 300ms' (class or *; ttft|tpot|e2e; p50/p90/p95/p99/max)")
@click.option("--config", "configs", multiple=True, required=True,
              help="name=run_dir[,run_dir...] (repeats of one configuration); repeatable")
@click.option("--attribution", default="equal_share", type=click.Choice(["equal_share", "proportional_tokens", "window_only"]))
@click.option("--exclude-class", "exclude_classes", multiple=True,
              help="Drop requests of this class (id prefix before '-') before evaluating, e.g. settle or warm; repeatable")
@click.option("--min-metric-coverage", type=float, default=1.0, help="Share of selected requests that must carry the target metric")
@click.option("--slo", "slos", multiple=True,
              help="Per-class request SLOs for goodput, e.g. 'short: ttft <= 50ms, tpot <= 15ms' ('*' for all classes); repeatable")
@click.option("--json", "json_out", type=click.Path(), help="Write the decision JSON here")
def decide(target: str, configs: Tuple[str, ...], attribution: str, exclude_classes: Tuple[str, ...], min_metric_coverage: float,
           slos: Tuple[str, ...], json_out: Optional[str]) -> None:
    """Compare configurations against a latency target, with goodput under SLOs and bootstrap intervals (advisory; changes nothing)."""
    from llmtrace.control_plane.decision import Slo, Target, evaluate, format_decision

    try:
        tgt = Target.parse(target)
        parsed_slos = [Slo.parse(s) for s in slos]
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(EXIT_USAGE)
    parsed: dict = {}
    for c in configs:
        if "=" not in c:
            click.echo(f"--config expects name=dir[,dir...], got {c!r}", err=True)
            sys.exit(EXIT_USAGE)
        name, dirs = c.split("=", 1)
        parsed[name.strip()] = [d.strip() for d in dirs.split(",") if d.strip()]
    dec = evaluate(parsed, tgt, attribution, min_metric_coverage, list(exclude_classes) or None, slos=parsed_slos or None)
    click.echo(format_decision(dec))
    if json_out:
        Path(json_out).write_text(dec.model_dump_json(indent=2), encoding="utf-8")
        click.echo(f"Decision written to {json_out}")


@main.command()
@click.option("--pid", type=int, help="(not implemented)")
def monitor(pid: Optional[int]) -> None:
    """Attach to a running vLLM process. NOT IMPLEMENTED."""
    click.echo(
        "monitor is not implemented: llmtrace cannot attach to an external process. "
        "Use LLMTracer.instrument_engine() inside the vLLM process.",
        err=True,
    )
    sys.exit(EXIT_NOT_IMPLEMENTED)


@main.group()
def workload() -> None:
    """Configuration-driven workloads (deterministic request lists for `llmtrace run`)."""


@workload.command("template")
@click.option("--output", default="workload.json", help="Where to write the template spec")
def workload_template(output: str) -> None:
    """Write a template workload spec (short stream + injected long prompts) to edit."""
    from llmtrace.workload import template

    p = template().save(output)
    click.echo(f"Workload template written to {p}")


@workload.command("preview")
@click.argument("spec_path", type=click.Path(exists=True))
@click.option("--json", "json_out", type=click.Path(), help="Write the summary JSON here")
@click.option("--requests", "requests_out", type=click.Path(), help="Write the generated request list (JSONL) here")
def workload_preview(spec_path: str, json_out: Optional[str], requests_out: Optional[str]) -> None:
    """Generate a spec's request list and print its summary (counts, lengths, arrivals, hash)."""
    from llmtrace.workload import WorkloadSpec

    try:
        spec = WorkloadSpec.load(spec_path)
    except Exception as exc:
        click.echo(f"Invalid workload spec: {exc}", err=True)
        sys.exit(EXIT_USAGE)
    specs = spec.generate()
    summary = spec.summary(specs)
    click.echo(json.dumps(summary, indent=2))
    if json_out:
        Path(json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if requests_out:
        with open(requests_out, "w", encoding="utf-8") as f:
            for r in specs:
                f.write(json.dumps(r.to_dict()) + "\n")


@main.command()
@click.option("--workload", "workload_path", required=True, type=click.Path(exists=True), help="Workload spec JSON (see `llmtrace workload template`)")
@click.option("--plan", "plan_path", type=click.Path(exists=True),
              help="Experiment plan JSON from `llmtrace plan`: runs its baseline and every candidate as <out>/<config>/r<i> (overrides --config-name/--set/--repeat)")
@click.option("--engine", type=click.Choice(["fake", "vllm"]), default=None,
              help="fake: synthetic CPU engine (invented cost model, not evidence); vllm: real vLLM 0.11.0 on a GPU "
                   "[default: the plan's source engine, else fake]")
@click.option("--out", "out_dir", required=True, type=click.Path(), help="Run directory (raw traces + manifest); with --repeat, <out>/r<i>")
@click.option("--model", default=None, help="vllm only [default: the plan's source model, else facebook/opt-125m]")
@click.option("--overwrite", is_flag=True, help="Remove a previous run's files from a run directory instead of refusing it")
@click.option("--config-name", default="default", show_default=True, help="Label for the configuration under test")
@click.option("--set", "changes", multiple=True, metavar="KEY=JSON",
              help="Engine kwarg under test, e.g. --set long_prefill_token_threshold=256 (recorded as scheduling_change)")
@click.option("--engine-kwargs", default="{}", help="JSON of other engine kwargs (vllm: LLM(...); fake: FakeLLMEngine(...))")
@click.option("--repeat", type=int, default=1, show_default=True, help="Independent repeats with identical work")
@click.option("--collection-interval", type=float, default=0.1, show_default=True, help="llmtrace collector drain interval (s)")
@click.option("--enable-nvtx", is_flag=True, help="NVTX range per engine step (vllm; for nsys)")
@click.option("--no-ignore-eos", is_flag=True, help="Let requests stop at EOS (work then differs across configs)")
@click.option("--no-warmup", is_flag=True, help="vllm: skip the untraced warm-up replay")
@click.option("--settle", type=int, default=4, show_default=True, help="vllm: traced settling requests before the measured replay")
def run(workload_path: str, plan_path: Optional[str], engine: Optional[str], out_dir: str, model: Optional[str], overwrite: bool,
        config_name: str, changes: Tuple[str, ...], engine_kwargs: str, repeat: int, collection_interval: float, enable_nvtx: bool,
        no_ignore_eos: bool, no_warmup: bool, settle: int) -> None:
    """Replay a workload spec under llmtrace and write run directories (raw data + manifest)."""
    from llmtrace.runner import RunOptions, run_workload
    from llmtrace.workload import WorkloadSpec

    try:
        spec = WorkloadSpec.load(workload_path)
        extra = json.loads(engine_kwargs)
        change = {}
        for c in changes:
            k, sep, v = c.partition("=")
            if not k or not sep:
                raise ValueError(f"--set expects KEY=JSON, got {c!r}")
            change[k.strip()] = json.loads(v)
        jobs: List[Tuple[str, dict, dict, str]] = []  # (config name, engine kwargs, scheduling change, out dir)
        if plan_path:
            from llmtrace.control_plane.experiments import ExperimentPlan

            p = ExperimentPlan.model_validate_json(Path(plan_path).read_text(encoding="utf-8"))
            if p.workload_hash and p.workload_hash != spec.hash():
                click.echo(f"warning: plan was made from workload {p.workload_hash}, this spec is {spec.hash()}")
            if engine is None:
                engine = p.source_engine or "fake"
            elif p.source_engine and engine != p.source_engine:
                click.echo(f"warning: plan was made from a {p.source_engine} run, running on {engine}")
            if model is None and p.source_model:
                model = p.source_model
            repeat = p.repeats
            for cfg in p.configs():
                # the source run's engine kwargs first, explicit --engine-kwargs on top, then the candidate's change
                kw = {**cfg["engine_kwargs"], **extra}
                for i in range(repeat):
                    jobs.append((cfg["name"], kw, cfg["scheduling_change"], str(Path(out_dir) / cfg["name"] / f"r{i}")))
        else:
            for i in range(repeat):
                jobs.append((config_name, extra, change, out_dir if repeat == 1 else str(Path(out_dir) / f"r{i}")))
        engine = engine or "fake"
        model = model or "facebook/opt-125m"
    except Exception as exc:
        click.echo(f"Invalid arguments: {exc}", err=True)
        sys.exit(EXIT_USAGE)
    if repeat < 1:
        click.echo("--repeat must be >= 1", err=True)
        sys.exit(EXIT_USAGE)
    from llmtrace.runner import existing_run_files

    if not overwrite:
        busy = [out for _, _, _, out in jobs if existing_run_files(out)]
        if busy:
            click.echo(f"{busy[0]} already holds a run ({len(busy)} such director{'y' if len(busy) == 1 else 'ies'}); "
                       "use a new --out or --overwrite", err=True)
            sys.exit(EXIT_USAGE)
    failed = False
    for name, kw, chg, out in jobs:
        opts = RunOptions(engine=engine, out_dir=out, model=model, config_name=name, scheduling_change=chg,
                          engine_kwargs=kw, collection_interval_s=collection_interval, enable_nvtx=enable_nvtx,
                          ignore_eos=not no_ignore_eos, warmup=not no_warmup, settle_requests=settle, overwrite=overwrite)
        m = run_workload(spec, opts)
        if m.status != "ok":
            click.echo(f"[{out}] FAILED: {m.error}", err=True)
            failed = True
            continue
        problems = m.extra.get("problems") or []
        click.echo(f"[{out}] {m.engine}{' (synthetic)' if m.synthetic else ''} config={m.config_name} workload={m.workload_hash} "
                   f"steps={m.steps} finished={m.finished}/{m.expected_requests} wall={m.wall_s:.3f}s "
                   f"arrival delay p50/max {m.arrival_delay_ms_p50} / {m.arrival_delay_ms_max} ms")
        h = m.health or {}
        click.echo(f"    scheduler visible: {h.get('scheduler_visible_during_run')} ({h.get('scheduler_unavailable_reason_during_run')}); "
                   f"executor visible: {h.get('executor_visible_during_run')} ({h.get('executor_unavailable_reason_during_run')})")
        for pr in problems:
            click.echo(f"    PROBLEM: {pr}", err=True)
        for tp in m.extra.get("telemetry_problems") or []:
            click.echo(f"    telemetry: {tp}")
        failed = failed or bool(problems)
    if plan_path:
        names = [cfg["name"] for cfg in p.configs()]
        cfgs = " ".join(f"--config {n}={','.join(str(Path(out_dir) / n / f'r{i}') for i in range(repeat))}" for n in names)
        click.echo(f"compare with: llmtrace decide --target '<class> ttft_p95 <= <ms>' {cfgs}")
    sys.exit(EXIT_REGRESSION if failed else EXIT_OK)


@main.command()
@click.argument("run_dir", required=False, type=click.Path(exists=True))
@click.option("--json", "json_out", type=click.Path(), help="Write the report JSON here")
def doctor(run_dir: Optional[str], json_out: Optional[str]) -> None:
    """Pre-flight checks: which signals this environment can produce (no engine started), or, with a
    run directory, which signals a recorded run has and why the others are missing."""
    from llmtrace.doctor import environment_report, run_report

    report = run_report(run_dir) if run_dir else environment_report()
    click.echo(report.format())
    if json_out:
        Path(json_out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    sys.exit(EXIT_REGRESSION if report.errors else EXIT_OK)


@main.command("init-config")
@click.option("--output", default="llmtrace_config.json", help="Output config file path")
def init_config(output: str) -> None:
    """Write a default configuration file (load with LLMTracer.from_config_file)."""
    with open(output, "w", encoding="utf-8") as f:
        json.dump(TracerConfig().model_dump(), f, indent=2)
    click.echo(f"Default configuration written to {output}")


if __name__ == "__main__":
    main()

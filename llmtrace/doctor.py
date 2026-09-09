"""Pre-flight compatibility checks and per-run signal availability.

``environment_report()`` answers, before an engine is started, which llmtrace
signals this environment can produce and why the others cannot: the vLLM
version against the verified target, the engine-core process mode
(``VLLM_ENABLE_V1_MULTIPROCESSING``), CUDA for step spans, NVML for GPU
telemetry, optional writers. ``run_report(run_dir)`` answers the same question
for a recorded run from its files and manifest: which data types are present,
what the tracer's health said about the missing ones, whether every expected
request finished, and how late the load generator was.

Every probe is injectable (``Probes``) so the logic is unit-tested without vLLM,
torch or a GPU; the defaults import lazily and never raise.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, Field

from llmtrace.data_plane.vllm_instrumentation import TARGET_VLLM_VERSION
from llmtrace.health import assess_health
from llmtrace.io import _PREFIXES as DATA_TYPES
from llmtrace.manifest import RunManifest

Status = str  # ok | warn | missing | error


class Check(BaseModel):
    name: str
    status: Status
    detail: str
    consequence: Optional[str] = None  # what llmtrace cannot do because of this


class Signal(BaseModel):
    name: str
    available: bool
    reason: Optional[str] = None  # why not, when unavailable


class DoctorReport(BaseModel):
    kind: str  # environment | run
    checks: List[Check] = Field(default_factory=list)
    signals: List[Signal] = Field(default_factory=list)

    @property
    def errors(self) -> List[Check]:
        return [c for c in self.checks if c.status == "error"]

    def format(self) -> str:
        lines = [f"llmtrace doctor ({self.kind})", ""]
        for c in self.checks:
            lines.append(f"  [{c.status:>7}] {c.name}: {c.detail}")
            if c.consequence:
                lines.append(f"            -> {c.consequence}")
        lines += ["", "  signals:"]
        for s in self.signals:
            lines.append(f"    {'yes' if s.available else 'no ':>3}  {s.name}" + (f"  ({s.reason})" if s.reason else ""))
        return "\n".join(lines)


# ----------------------------------------------------------------------------- probes


def _default_import(name: str) -> Any:
    return importlib.import_module(name)


def _default_cuda() -> Tuple[bool, str]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the environment
        return False, f"torch not importable ({type(exc).__name__})"
    try:
        ok = bool(torch.cuda.is_available())
        return ok, (f"torch {torch.__version__}, {torch.cuda.device_count()} CUDA device(s)" if ok
                    else f"torch {torch.__version__}, torch.cuda.is_available() is False")
    except Exception as exc:  # pragma: no cover
        return False, f"torch.cuda probe failed ({type(exc).__name__}: {exc})"


def _default_nvml() -> Tuple[bool, str]:
    try:
        import pynvml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the environment
        return False, f"pynvml not importable ({type(exc).__name__}); install llmtrace[nvml]"
    try:
        pynvml.nvmlInit()
        try:
            n = pynvml.nvmlDeviceGetCount()
            drv = pynvml.nvmlSystemGetDriverVersion()
            drv = drv.decode() if isinstance(drv, bytes) else drv
            return n > 0, f"driver {drv}, {n} device(s)"
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # pragma: no cover
        return False, f"NVML init failed ({type(exc).__name__}: {exc})"


@dataclass
class Probes:
    import_module: Callable[[str], Any] = _default_import
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    cuda: Callable[[], Tuple[bool, str]] = _default_cuda
    nvml: Callable[[], Tuple[bool, str]] = _default_nvml
    python_version: Tuple[int, int, int] = field(default_factory=lambda: sys.version_info[:3])


# ------------------------------------------------------------------------ environment


def environment_report(probes: Optional[Probes] = None) -> DoctorReport:
    p = probes or Probes()
    checks: List[Check] = []

    from llmtrace.provenance import git_state, source_fingerprint

    st = git_state()
    prov = f"fingerprint {source_fingerprint()}"
    if st["commit"]:
        prov += f", commit {st['commit_short']}" + (" DIRTY (uncommitted changes; runs will save source.patch)" if st["dirty"] else " clean")
    else:
        prov += ", no git tree (the fingerprint is the only identity of this code)"
    checks.append(Check(name="llmtrace source", status="warn" if st["dirty"] else "ok", detail=prov,
                        consequence="evidence from a dirty tree is reproducible only with its source.patch" if st["dirty"] else None))
    pv = p.python_version
    checks.append(Check(name="python", status="ok" if (3, 9) <= pv[:2] <= (3, 12) else "warn",
                        detail=f"{pv[0]}.{pv[1]}.{pv[2]}",
                        consequence=None if (3, 9) <= pv[:2] <= (3, 12) else "vLLM 0.11.0 wheels target Python 3.9 to 3.12"))

    vllm_version: Optional[str] = None
    try:
        vllm = p.import_module("vllm")
        vllm_version = str(getattr(vllm, "__version__", "unknown"))
    except Exception as exc:
        checks.append(Check(name="vllm", status="missing", detail=f"not importable ({type(exc).__name__})",
                            consequence="only the synthetic engine (llmtrace run --engine fake) and offline analysis work"))
    if vllm_version is not None:
        if vllm_version == TARGET_VLLM_VERSION:
            checks.append(Check(name="vllm", status="ok", detail=f"{vllm_version} (verified target)"))
        else:
            checks.append(Check(name="vllm", status="warn", detail=f"{vllm_version}; llmtrace was verified against {TARGET_VLLM_VERSION}",
                                consequence="instrumentation attaches by attribute path and may silently miss signals; "
                                            "check health()['instrumentation'] after a run"))

    mp = p.env.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    in_process = mp == "0"
    checks.append(Check(name="engine core process", status="ok" if in_process else "warn",
                        detail=("in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0)" if in_process
                                else f"separate process (VLLM_ENABLE_V1_MULTIPROCESSING={mp!r}, vLLM default)"),
                        consequence=None if in_process else
                        "no scheduler access: no batch membership, queue/prefill split or GPU step spans; "
                        "request-level traces and vLLM per-step stats only. Set VLLM_ENABLE_V1_MULTIPROCESSING=0 for the sync engine; "
                        "AsyncLLM always runs the core out of process"))

    cuda_ok, cuda_detail = p.cuda()
    checks.append(Check(name="torch.cuda", status="ok" if cuda_ok else "missing", detail=cuda_detail,
                        consequence=None if cuda_ok else "no CUDA-event GPU step spans (gpu_steps_*), no host_overhead finding"))

    nvml_ok, nvml_detail = p.nvml()
    checks.append(Check(name="nvml", status="ok" if nvml_ok else "missing", detail=nvml_detail,
                        consequence=None if nvml_ok else "no GPU telemetry (gpu_*): no power, energy, utilization, throttle reasons"))

    for mod, extra, use in (("pyarrow", "parquet", "output_format=parquet"), ("rich", "dev", "rich CLI output (plain text otherwise)")):
        try:
            p.import_module(mod)
            checks.append(Check(name=mod, status="ok", detail="importable"))
        except Exception:
            checks.append(Check(name=mod, status="warn", detail=f"not importable; pip install llmtrace[{extra}]", consequence=f"no {use}"))

    have_vllm = vllm_version is not None
    sched_reason = (None if (have_vllm and in_process) else
                    "vLLM not importable" if not have_vllm else "engine core out of process")
    signals = [
        Signal(name="request traces (traces_*)", available=have_vllm, reason=None if have_vllm else "vLLM not importable"),
        Signal(name="batch membership and chunk sizes (batches_*)", available=sched_reason is None, reason=sched_reason),
        Signal(name="queue/prefill spans per request", available=sched_reason is None,
               reason=sched_reason or None if sched_reason else None),
        Signal(name="vLLM per-step stats (vllm_stats_*)", available=have_vllm, reason=None if have_vllm else "vLLM not importable"),
        Signal(name="GPU step spans (gpu_steps_*)", available=sched_reason is None and cuda_ok,
               reason=sched_reason or (None if cuda_ok else "torch.cuda unavailable")),
        Signal(name="GPU telemetry and energy (gpu_*)", available=nvml_ok, reason=None if nvml_ok else "NVML unavailable"),
    ]
    return DoctorReport(kind="environment", checks=checks, signals=signals)


# -------------------------------------------------------------------------------- run


def _present(run_dir: Path, data_type: str) -> List[Path]:
    from llmtrace.io import _matches_prefix

    return sorted(p for p in run_dir.glob(f"{data_type}_*") if _matches_prefix(p, data_type) and p.is_file())


def run_report(run_dir: str) -> DoctorReport:
    d = Path(run_dir)
    checks: List[Check] = []
    signals: List[Signal] = []
    if not d.is_dir():
        return DoctorReport(kind="run", checks=[Check(name="run directory", status="error", detail=f"{run_dir} is not a directory")])
    m = RunManifest.read(str(d))
    if m is None:
        checks.append(Check(name="manifest", status="warn", detail="no manifest.json",
                            consequence="no workload hash, engine config or arrival records; decide/compare cannot check equivalence"))
    else:
        if m.status != "ok":
            checks.append(Check(name="manifest", status="error", detail=f"status {m.status}: {m.error}",
                                consequence="the run did not complete; its traces (if any) are not comparable"))
        else:
            checks.append(Check(name="manifest", status="ok",
                                detail=f"{m.engine}{' (synthetic)' if m.synthetic else ''}, model {m.model}, "
                                       f"config {m.config_name}, workload {m.workload_hash}"))
        if m.expected_requests is not None:
            fin = m.finished if m.finished is not None else 0
            ok = fin == m.expected_requests or (m.finished is not None and m.finished + int(m.extra.get("settle_requests", 0)) == m.expected_requests)
            checks.append(Check(name="requests finished", status="ok" if ok else "error",
                                detail=f"{m.finished} finished, {m.expected_requests} expected (incl. settle requests)",
                                consequence=None if ok else "decide treats this run as ineligible"))
        fp = m.llmtrace_source_fingerprint
        if fp:
            same = fp == __import__("llmtrace.provenance", fromlist=["source_fingerprint"]).source_fingerprint()
            checks.append(Check(name="source", status="ok" if not m.llmtrace_git_dirty else "warn",
                                detail=f"{fp}, commit {m.llmtrace_git_commit or 'n/a'}"
                                       + (" dirty" if m.llmtrace_git_dirty else "") + (", same code as installed now" if same else ", differs from the installed code"),
                                consequence=(f"reproduce from commit {m.llmtrace_git_commit} plus {m.llmtrace_source_patch}" if m.llmtrace_git_dirty else None)))
        else:
            checks.append(Check(name="source", status="warn", detail="no source fingerprint in the manifest (recorded before provenance existed)"))
        if m.arrival_delay_ms_max is not None:
            late = m.arrival_delay_ms_max > 50.0
            checks.append(Check(name="load generator", status="warn" if late else "ok",
                                detail=f"arrival delay p50 {m.arrival_delay_ms_p50:.2f} ms, max {m.arrival_delay_ms_max:.2f} ms",
                                consequence="TTFT from intended arrival (ttft_sched) differs materially from engine TTFT" if late else None))
        problems = m.extra.get("problems") or []
        for pr in problems:
            checks.append(Check(name="run problem", status="error", detail=str(pr)))
        if m.health:
            a = assess_health(m.health)
            for pr in a.problems:
                checks.append(Check(name="tracer health", status="error", detail=pr,
                                    consequence="the request record is not trustworthy; decide treats this run as ineligible"))
            for tp in a.telemetry_problems:
                checks.append(Check(name="telemetry", status="warn", detail=tp,
                                    consequence="metrics that depend on this signal are reported as unavailable"))

    reasons: Dict[str, Optional[str]] = {}
    if m is not None and isinstance(m.health, dict):
        h = m.health
        sched = h.get("scheduler_unavailable_reason_during_run") or h.get("instrumentation", {}).get("scheduler_unavailable_reason")
        execu = h.get("executor_unavailable_reason_during_run") or h.get("instrumentation", {}).get("executor_unavailable_reason")
        cuda = (h.get("instrumentation", {}).get("cuda_timing") or {}).get("unavailable_reason")
        gpu = (h.get("gpu_sampler") or {}).get("unavailable_reason") or (h.get("gpu_sampler") or {}).get("error")
        stats = (h.get("vllm_stats") or {}).get("unavailable_reason")
        reasons = {"batches": sched, "gpu_steps": execu or cuda, "gpu": gpu, "vllm_stats": stats}
    labels = {"traces": "request traces", "batches": "batch membership and chunk sizes", "gpu": "GPU telemetry and energy",
              "gpu_steps": "GPU step spans", "vllm_stats": "vLLM per-step stats", "collector": "collector self-events"}
    for dt in DATA_TYPES:
        files = _present(d, dt)
        signals.append(Signal(name=f"{labels.get(dt, dt)} ({dt}_*)", available=bool(files),
                              reason=None if files else (reasons.get(dt) or "no file in run directory")))
    if not _present(d, "traces"):
        checks.append(Check(name="traces", status="error", detail="no traces_* file", consequence="nothing to analyze"))
    return DoctorReport(kind="run", checks=checks, signals=signals)


__all__ = ["Check", "Signal", "DoctorReport", "Probes", "environment_report", "run_report"]

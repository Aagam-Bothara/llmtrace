"""Run manifests: everything needed to reproduce or compare a recorded run.

A manifest records the workload (with a content hash), seed, model and
revision, engine and llmtrace versions (and git commit when available), the
effective engine/scheduler settings, the host/GPU, the tracer configuration,
and per-request scheduled-versus-actual arrival times so load-generator delay
never disappears from the results. ``status`` is ``ok`` or ``failed`` (with the
error), so a configuration that ran out of memory stays visible in comparisons.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ArrivalRecord(BaseModel):
    request_id: str
    scheduled_s: float  # intended arrival offset from run start
    actual_s: Optional[float] = None  # when add_request() was actually called
    delay_ms: Optional[float] = None


class RunManifest(BaseModel):
    manifest_version: int = 1
    created_at: float = Field(default_factory=time.time)
    status: str = "ok"  # ok | failed
    error: Optional[str] = None
    label: Optional[str] = None
    engine: str = "unknown"  # vllm | fake
    synthetic: bool = False
    model: Optional[str] = None
    model_revision: Optional[str] = None
    engine_version: Optional[str] = None
    llmtrace_version: Optional[str] = None
    llmtrace_git_commit: Optional[str] = None
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    gpu: Optional[Dict[str, Any]] = None
    workload: Dict[str, Any] = Field(default_factory=dict)
    workload_hash: Optional[str] = None
    seed: Optional[int] = None
    config_name: Optional[str] = None
    scheduling_change: Dict[str, Any] = Field(default_factory=dict)
    engine_kwargs: Dict[str, Any] = Field(default_factory=dict)
    effective_engine_config: Dict[str, Any] = Field(default_factory=dict)
    tracer_config: Dict[str, Any] = Field(default_factory=dict)
    arrivals: List[ArrivalRecord] = Field(default_factory=list)
    arrival_delay_ms_p50: Optional[float] = None
    arrival_delay_ms_max: Optional[float] = None
    steps: Optional[int] = None
    wall_s: Optional[float] = None
    finished: Optional[int] = None
    health: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)

    def finalize_arrivals(self) -> None:
        delays = [a.delay_ms for a in self.arrivals if a.delay_ms is not None]
        if delays:
            s = sorted(delays)
            self.arrival_delay_ms_p50 = s[len(s) // 2]
            self.arrival_delay_ms_max = s[-1]

    def write(self, run_dir: str) -> Path:
        self.finalize_arrivals()
        p = Path(run_dir) / "manifest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def read(cls, run_dir: str) -> Optional["RunManifest"]:
        p = Path(run_dir) / "manifest.json"
        if not p.exists():
            return None
        return cls.model_validate_json(p.read_text(encoding="utf-8"))


def workload_hash(specs: List[Any]) -> str:
    """Stable hash of a workload spec list (dataclasses/dicts with sortable fields)."""
    def to_dict(s: Any) -> Any:
        return s if isinstance(s, dict) else {k: getattr(s, k) for k in getattr(s, "__dataclass_fields__", {})}
    blob = json.dumps([to_dict(s) for s in specs], sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def gpu_info() -> Optional[Dict[str, Any]]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        rows = [line.split(", ") for line in out.stdout.strip().splitlines()]
        return {"devices": [{"name": r[0], "driver": r[1], "memory": r[2] if len(r) > 2 else None} for r in rows]}
    except Exception:
        return None


def llmtrace_version() -> Optional[str]:
    try:
        from llmtrace import __version__
        return __version__
    except Exception:
        return None


def engine_effective_config(engine: Any) -> Dict[str, Any]:
    """Read the effective vLLM config from a real LLMEngine (best effort, duck-typed)."""
    out: Dict[str, Any] = {}
    cfg = getattr(engine, "vllm_config", None)
    for section in ("scheduler_config", "cache_config", "model_config", "parallel_config"):
        obj = getattr(cfg, section, None)
        if obj is None:
            continue
        keys = {
            "scheduler_config": ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "enable_chunked_prefill",
                                 "long_prefill_token_threshold", "max_num_partial_prefills", "max_long_partial_prefills", "policy"),
            "cache_config": ("block_size", "gpu_memory_utilization", "enable_prefix_caching", "num_gpu_blocks"),
            "model_config": ("model", "revision", "dtype", "max_model_len", "seed"),
            "parallel_config": ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"),
        }[section]
        vals = {k: getattr(obj, k, None) for k in keys}
        out[section] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in vals.items()}
    return out

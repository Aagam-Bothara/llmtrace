"""Backward-compatible shim: the fakes now live in ``llmtrace.testing.fakes`` so the package can run synthetic workloads."""

from llmtrace.testing import fakes as _fakes

globals().update({k: v for k, v in vars(_fakes).items() if not k.startswith("__")})

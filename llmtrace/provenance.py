"""Which llmtrace code produced a run: a content fingerprint plus git state and the dirty diff.

GPU sessions 2 and 3 ran fixes that were uploaded to the pod before being
committed, and the pods had no ``.git``, so their manifests say ``git n/a``
and the evidence could only be described as "commit X plus fixes". From now
on every run records:

* ``source_fingerprint``: sha256 over the installed ``llmtrace`` package's
  ``.py`` files (relative path and bytes, sorted), which identifies the exact
  code whether or not git is present;
* ``git_commit`` (full hash), ``git_dirty`` and, when the tree is dirty, the
  diff written next to the manifest as ``source.patch`` (``git diff HEAD``
  for tracked files; untracked ``.py`` files are listed in the manifest).

A fingerprint that matches a clean commit's fingerprint proves the evidence
came from that commit; a dirty run is reproducible from the commit plus the
patch.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def package_root() -> Path:
    import llmtrace

    return Path(llmtrace.__file__).resolve().parent


def source_fingerprint(root: Optional[Path] = None) -> str:
    """sha256 over every ``.py`` under ``root`` (default: the installed llmtrace package), path-sorted."""
    root = Path(root) if root is not None else package_root()
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return "sha256:" + h.hexdigest()[:20]


def _git(args: List[str], cwd: Optional[Path]) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return out.stdout if out.returncode == 0 else None


def repo_dir_for(path: Optional[Path] = None) -> Optional[Path]:
    """The git working tree containing ``path`` (default: the installed package), or None."""
    start = Path(path) if path is not None else package_root()
    top = _git(["rev-parse", "--show-toplevel"], start if start.is_dir() else start.parent)
    return Path(top.strip()) if top and top.strip() else None


def git_state(repo_dir: Optional[Path] = None) -> Dict[str, Any]:
    """{'commit', 'commit_short', 'dirty', 'diff', 'untracked'}; all None/empty when not in a git tree."""
    repo = repo_dir if repo_dir is not None else repo_dir_for()
    if repo is None:
        return {"commit": None, "commit_short": None, "dirty": None, "diff": None, "untracked": []}
    commit = (_git(["rev-parse", "HEAD"], repo) or "").strip() or None
    status = _git(["status", "--porcelain", "--untracked-files=all"], repo) or ""
    lines = [ln for ln in status.splitlines() if ln.strip()]
    untracked = [ln[3:] for ln in lines if ln.startswith("??") and ln.endswith(".py")]
    tracked_changes = [ln for ln in lines if not ln.startswith("??")]
    diff = _git(["diff", "HEAD", "--", "."], repo) if tracked_changes else ""
    dirty = bool(tracked_changes or untracked)
    return {"commit": commit, "commit_short": commit[:7] if commit else None, "dirty": dirty, "diff": diff or "", "untracked": untracked}


def record_provenance(run_dir: str, repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """Fingerprint + git state for a manifest; writes ``source.patch`` into ``run_dir`` when the tree is dirty."""
    repo = Path(repo_dir) if repo_dir else repo_dir_for()
    st = git_state(repo)
    patch_name: Optional[str] = None
    if st["dirty"] and st["diff"]:
        p = Path(run_dir) / "source.patch"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(st["diff"], encoding="utf-8")
        patch_name = p.name
    return {"llmtrace_source_fingerprint": source_fingerprint(), "llmtrace_git_commit": st["commit_short"],
            "llmtrace_git_commit_full": st["commit"], "llmtrace_git_dirty": st["dirty"],
            "llmtrace_source_patch": patch_name, "llmtrace_untracked_py": st["untracked"]}


__all__ = ["source_fingerprint", "git_state", "record_provenance", "repo_dir_for", "package_root"]

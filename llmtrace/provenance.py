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
  for tracked files) plus ``source_untracked.tar.gz`` holding the contents of
  untracked, non-ignored files (each at most ``UNTRACKED_MAX_BYTES``);
* ``snapshot_complete``: false when something could not be captured (a
  binary tracked change, an untracked file above the size cap, a failed
  archive), with the reasons listed. A fingerprint identifies content but
  cannot restore it, so an incomplete snapshot is stated, never implied.

A fingerprint that matches a clean commit's fingerprint proves the evidence
came from that commit. A dirty run is reproducible from the commit, the patch
and the untracked archive only when ``snapshot_complete`` is true.
"""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

UNTRACKED_MAX_BYTES = 1_000_000


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
    untracked = [ln[3:] for ln in lines if ln.startswith("??")]  # every untracked, non-ignored file, not only .py
    tracked_changes = [ln for ln in lines if not ln.startswith("??")]
    diff = _git(["diff", "HEAD", "--", "."], repo) if tracked_changes else ""
    dirty = bool(tracked_changes or untracked)
    return {"commit": commit, "commit_short": commit[:7] if commit else None, "dirty": dirty, "diff": diff or "",
            "untracked": untracked, "repo": str(repo)}


def record_provenance(run_dir: str, repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """Fingerprint + git state for a manifest. When the tree is dirty, writes ``source.patch`` (tracked changes) and
    ``source_untracked.tar.gz`` (untracked file contents) into ``run_dir``, and says whether that snapshot is complete."""
    repo = Path(repo_dir) if repo_dir else repo_dir_for()
    st = git_state(repo)
    out = Path(run_dir)
    patch_name: Optional[str] = None
    archive_name: Optional[str] = None
    incomplete: List[str] = []
    if st["dirty"]:
        out.mkdir(parents=True, exist_ok=True)
        if st["diff"]:
            (out / "source.patch").write_text(st["diff"], encoding="utf-8")
            patch_name = "source.patch"
            if "Binary files" in st["diff"] or "GIT binary patch" in st["diff"]:
                incomplete.append("source.patch contains a binary change that a text diff cannot restore")
        if st["untracked"]:
            repo_path = Path(st["repo"])
            try:
                with tarfile.open(out / "source_untracked.tar.gz", "w:gz") as tar:
                    for rel in st["untracked"]:
                        f = repo_path / rel
                        if not f.is_file():
                            incomplete.append(f"untracked {rel}: not a regular file")
                            continue
                        if f.stat().st_size > UNTRACKED_MAX_BYTES:
                            incomplete.append(f"untracked {rel}: {f.stat().st_size} bytes exceeds the {UNTRACKED_MAX_BYTES}-byte cap, not archived")
                            continue
                        tar.add(str(f), arcname=rel)
                archive_name = "source_untracked.tar.gz"
            except Exception as exc:
                incomplete.append(f"untracked files could not be archived ({type(exc).__name__}: {exc})")
    if st["commit"] is None:
        incomplete.append("no git tree: the fingerprint identifies the code but nothing here can restore it")
    return {"llmtrace_source_fingerprint": source_fingerprint(), "llmtrace_git_commit": st["commit_short"],
            "llmtrace_git_commit_full": st["commit"], "llmtrace_git_dirty": st["dirty"],
            "llmtrace_source_patch": patch_name, "llmtrace_untracked_files": st["untracked"],
            "llmtrace_untracked_archive": archive_name, "llmtrace_snapshot_complete": not incomplete,
            "llmtrace_snapshot_gaps": incomplete}


__all__ = ["source_fingerprint", "git_state", "record_provenance", "repo_dir_for", "package_root"]

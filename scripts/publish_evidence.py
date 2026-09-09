"""Publish the reviewed evidence archives and verify public downloads.

Run from the repository root. Uses GH_TOKEN/GITHUB_TOKEN or Git's configured
credential helper; credentials are never printed. Existing assets are checked,
never overwritten. A new release stays a draft until all assets are uploaded.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request


REPO = "Aagam-Bothara/llmtrace"
TAG = "evidence-2026-09"
API = f"https://api.github.com/repos/{REPO}"


def request(url, token=None, data=None, method=None, content_type="application/json"):
    headers = {"User-Agent": "llmtrace-evidence-publisher", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = content_type
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=120) as r:
        return r.read()


def credential():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                            text=True, capture_output=True, timeout=30,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not fields.get("password"):
        raise RuntimeError("No existing GitHub credential available for release publication")
    return fields["password"]


def main():
    checksum_path = Path("docs/gpu_runs/SHA256SUMS")
    hashes = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name:
            raise RuntimeError("Invalid asset filename")
        hashes[name] = digest
    assets = [Path("evidence_raw") / name for name in hashes]
    for p in assets:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.name]:
            raise RuntimeError(f"Checksum mismatch: {p.name}")
    assets.append(checksum_path)
    hashes[checksum_path.name] = hashlib.sha256(checksum_path.read_bytes()).hexdigest()
    token = credential()
    try:
        release = json.loads(request(f"{API}/releases/tags/{TAG}", token))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        drafts = json.loads(request(f"{API}/releases?per_page=100", token))
        release = next((r for r in drafts if r["tag_name"] == TAG), None)
        if release is None:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            release = json.loads(request(f"{API}/releases", token, json.dumps({
                "tag_name": TAG, "target_commitish": commit, "name": "September 2026 GPU raw evidence",
                "draft": True, "body": "Raw JSONL evidence, one archive per GPU session. Download SHA256SUMS "
                "and the archives, run `sha256sum -c SHA256SUMS`, then extract at the repository root. "
                "Archives restore docs/gpu_runs/<session>/**/*.jsonl. Manifests and derived reports are in git."
            }).encode(), "POST"))
    existing = {a["name"]: a for a in release["assets"]}
    upload = release["upload_url"].split("{")[0]
    for p in assets:
        if p.name in existing:
            a = existing[p.name]
            if a.get("digest") != "sha256:" + hashes[p.name]:
                raise RuntimeError(f"Existing asset digest differs or is unavailable: {p.name}; refusing overwrite")
        else:
            request(upload + "?" + urllib.parse.urlencode({"name": p.name}), token, p.read_bytes(), "POST",
                    "application/gzip" if p.name.endswith(".gz") else "text/plain")
        print(f"Uploaded/checked {p.name}", flush=True)
    if release["draft"]:
        request(f"{API}/releases/{release['id']}", token, b'{"draft":false}', "PATCH")
    public = json.loads(request(f"{API}/releases/tags/{TAG}"))
    public_assets = {a["name"]: a for a in public["assets"]}
    for name, digest in hashes.items():
        data = request(public_assets[name]["browser_download_url"])
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Public download checksum mismatch: {name}")
        print(f"Anonymous download verified: {name}", flush=True)
    print(public["html_url"])


if __name__ == "__main__":
    main()

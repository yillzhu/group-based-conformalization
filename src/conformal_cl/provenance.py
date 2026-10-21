"""Small, dependency-light provenance manifests for regenerated artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .paths import PROJECT_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_paths(paths):
    normalized = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        normalized.append(path.resolve())
    hashes = {}
    for path in sorted(normalized):
        try:
            key = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            # Custom model/output paths are supported by the experiment CLIs.
            # Preserve an unambiguous absolute key when such an input is outside
            # the checkout instead of failing while producing its manifest.
            key = str(path)
        hashes[key] = sha256_file(path)
    return hashes


def _git_state():
    def run(*args):
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def write_run_manifest(
    destination,
    *,
    run_kind,
    parameters,
    inputs,
    artifacts,
):
    """Write hashes and runtime metadata for one corrected run."""

    source_paths = [
        *PROJECT_ROOT.glob("src/**/*.py"),
        *PROJECT_ROOT.glob("experiments/**/*.py"),
        PROJECT_ROOT / "pyproject.toml",
    ]
    package_names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "tensorflow",
        "gurobipy",
        "gurobi-machinelearning",
    ]
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    manifest = {
        "schema_version": 1,
        "run_kind": run_kind,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": versions,
        },
        "git": _git_state(),
        "inputs_sha256": _hash_paths(inputs),
        "source_sha256": _hash_paths(source_paths),
        "artifacts_sha256": _hash_paths(artifacts),
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

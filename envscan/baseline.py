"""Baselines: remember today's findings so only new ones are reported.

An entry is (rule, file, hash of the evidence line text). The line number is not
part of the key, so edits above a finding do not bring it back. Whitespace at
the ends of the line and runs of spaces are ignored. Evidence that is not a line
(for example `pyproject.toml:[tool.uv.sources] ...`) is hashed by its own text.
A finding is in the baseline when every one of its evidence entries is.
`scope` separates tasks of a Harbor dataset (the task folder name).
"""
from __future__ import annotations

import hashlib
import json
import re

from . import __version__
from .config import evidence_path
from .package import Package

FORMAT = 1


class BaselineError(Exception):
    pass


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def evidence_key(pkg: Package, e: str) -> tuple[str, str]:
    path, _, rest = e.partition(":")
    if rest.isdigit():
        lines = (pkg.files.get(path) or "").splitlines()
        n = int(rest)
        text = lines[n - 1] if 0 < n <= len(lines) else ""
    else:
        text = rest
    return path, hashlib.sha256(_norm(text).encode("utf-8", "replace")).hexdigest()[:16]


def finding_keys(pkg: Package, f: dict, scope: str = "") -> list[tuple[str, str, str, str]]:
    if not f["evidence"]:
        return [(scope, f["rule"], "", "")]
    return [(scope, f["rule"], *evidence_key(pkg, e)) for e in f["evidence"]]


def collect(pkg: Package, res: dict, scope: str = "") -> set:
    keys = set()
    for f in res["findings"]:
        if f["kind"] != "hygiene":
            keys.update(finding_keys(pkg, f, scope))
    return keys


def write_baseline(path: str, keys) -> int:
    entries = [{"scope": s, "rule": r, "file": fl, "hash": h} for s, r, fl, h in sorted(keys)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"envscan_baseline": FORMAT, "scanner_version": __version__, "findings": entries}, f, indent=2)
        f.write("\n")
    return len(entries)


def load_baseline(path: str) -> set:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("envscan_baseline") != FORMAT:
            raise ValueError("not an envscan baseline (version mismatch)")
        return {(e.get("scope", ""), e["rule"], e["file"], e["hash"]) for e in data["findings"]}
    except OSError as e:
        raise BaselineError(f"cannot read baseline {path}: {e}")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise BaselineError(f"bad baseline {path}: {e}")


def apply_baseline(pkg: Package, res: dict, base: set, scope: str = "") -> dict:
    kept, n = [], 0
    for f in res["findings"]:
        if f["kind"] != "hygiene" and all(k in base for k in finding_keys(pkg, f, scope)):
            n += 1
        else:
            kept.append(f)
    res["findings"] = kept
    res["baselined"] = res.get("baselined", 0) + n
    return res

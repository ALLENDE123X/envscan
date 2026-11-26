"""SARIF 2.1.0 output for GitHub code scanning.

One run, tool `envscan`, every rule in tool.driver.rules, one result per
non hygiene finding. Level is `error` for confidence likely and `warning` for
review. Locations are file and line relative to the scanned root (or to
`base`, so the paths match the repository root when the scan root is a
subfolder). GitHub reads only the first location of a result, so the other
evidence entries go in relatedLocations.
"""
from __future__ import annotations

import json
import posixpath

from . import __version__
from .config import evidence_path

SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
# GitHub shows security-severity bands: >9 critical, >7 high, >4 medium, >0 low.
SEC = {"critical": "9.5", "high": "8.0", "medium": "5.5", "low": "3.0", "info": "1.0"}
LEVEL = {"likely": "error", "review": "warning"}


def _rule_descriptor(r: dict) -> dict:
    d = {
        "id": r["id"],
        "name": "".join(p.capitalize() for p in r["id"].split("-")),
        "shortDescription": {"text": r["description"]},
        "fullDescription": {"text": r["description"]},
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["hack", "reward-hacking"] if r["kind"] == "hack" else [r["kind"]]},
    }
    if r["kind"] != "hygiene":
        d["properties"]["security-severity"] = SEC.get(r["severity"], "5.0")
    return d


def _uri(prefix: str, path: str) -> str:
    p = posixpath.normpath(path.replace("\\", "/"))
    return posixpath.join(prefix, p) if prefix else p


def _loc(uri: str, line: int | None, idx: int | None = None) -> dict:
    loc = {"physicalLocation": {"artifactLocation": {"uri": uri}, "region": {"startLine": max(1, line or 1)}}}
    if idx is not None:
        loc["id"] = idx
    return loc


def _split(e: str):
    path, _, rest = e.partition(":")
    return path, int(rest) if rest.isdigit() else None


def build_sarif(rules: list[dict], runs_in: list[tuple[dict, str, str | None]]) -> dict:
    """runs_in: (result dict, uri prefix, fallback file) per scanned package."""
    index = {r["id"]: i for i, r in enumerate(rules)}
    results = []
    for res, prefix, fallback in runs_in:
        for f in res["findings"]:
            if f["kind"] == "hygiene":
                continue
            evs = [_split(e) for e in f["evidence"] if evidence_path(e)]
            if evs:
                locs = [_loc(_uri(prefix, p), ln) for p, ln in evs[:1]]
                related = [_loc(_uri(prefix, p), ln, i) for i, (p, ln) in enumerate(evs[1:11])]
            else:
                locs = [_loc(_uri(prefix, fallback or "."), 1)]
                related = []
            r = {
                "ruleId": f["rule"],
                "ruleIndex": index.get(f["rule"], -1),
                "level": LEVEL.get(f["confidence"], "warning"),
                "message": {"text": f["message"]},
                "locations": locs,
                "properties": {"kind": f["kind"], "severity": f["severity"], "confidence": f["confidence"]},
            }
            if f.get("variant"):
                r["properties"]["variant"] = f["variant"]
            if r["ruleIndex"] < 0:
                del r["ruleIndex"]
            if related:
                r["relatedLocations"] = related
            results.append(r)
    return {
        "$schema": SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "envscan",
                "version": __version__,
                "semanticVersion": __version__,
                "rules": [_rule_descriptor(r) for r in rules],
            }},
            "results": results,
            "columnKind": "utf16CodeUnits",
        }],
    }


def write_sarif(path: str, doc: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

"""Scan a Package and turn findings into an env level verdict."""
from __future__ import annotations

import time

from . import __version__
from .package import Package
from .rules import RULES, run_rules

SEV_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

# Rules whose "likely" findings are trusted enough to put an env in the "likely" bucket.
# Set from calibration on the 90 hand audited envs (see calibrate.py); a rule left out
# here can still raise an env to "review".
LIKELY_RULES: set[str] | None = None


def verdict(findings) -> tuple[str, str | None]:
    counted = [f for f in findings if f.kind in ("hack", "broken")]
    likely = [f for f in counted if f.confidence == "likely" and (LIKELY_RULES is None or f.rule in LIKELY_RULES)]
    if likely:
        top = max(likely, key=lambda f: SEV_ORDER.get(f.severity, 0))
        return "likely", top.severity
    if counted:
        top = max(counted, key=lambda f: SEV_ORDER.get(f.severity, 0))
        return "review", top.severity
    return "clean", None


def scan_package(pkg: Package, env_id: str | None = None) -> dict:
    t0 = time.time()
    findings, errors = run_rules(pkg)
    v, sev = verdict(findings)
    n_py = len(pkg.py)
    parse_errors = [f"{p}: {pf.error}" for p, pf in pkg.py.items() if pf.error]
    return {
        "env": env_id or pkg.root,
        "scanner_version": __version__,
        "verdict": v,
        "top_severity": sev,
        "kinds": sorted({f.kind for f in findings if f.kind != "hygiene"}),
        "findings": [f.to_dict() for f in findings],
        "stats": {
            "files": len(pkg.sizes),
            "text_files": len(pkg.files),
            "py_files": n_py,
            "bytes": sum(pkg.sizes.values()),
            "vendored_files_skipped": pkg.vendored,
            "scan_s": round(time.time() - t0, 3),
        },
        "verifiers_spec": pkg.verifiers_requirement(),
        "parse_errors": parse_errors[:5],
        "rule_errors": errors,
        "notes": pkg.notes,
        "file_hashes": dict(sorted(pkg.hashes.items())[:400]),
    }


def finalize(res: dict) -> dict:
    """Recompute verdict, top_severity and kinds after findings were filtered
    (config suppression, baseline). Same logic as `verdict`, on result dicts."""
    from types import SimpleNamespace
    fs = [SimpleNamespace(**{k: f[k] for k in ("rule", "kind", "severity", "confidence")}) for f in res["findings"]]
    res["verdict"], res["top_severity"] = verdict(fs)
    res["kinds"] = sorted({f["kind"] for f in res["findings"] if f["kind"] != "hygiene"})
    return res


def rule_table():
    return [{k: v for k, v in m.items() if k != "fn"} for m in RULES.values()]

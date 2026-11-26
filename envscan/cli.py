"""envscan CLI.

  envscan <path>                 scan a local env directory (or .tar.gz)
  envscan <dataset dir>          a folder of 2+ Harbor tasks: one verdict per task
  envscan owner/name[@version]   fetch from the Hub in memory and scan
  envscan --rules                list rules

Exit code: 0 clean, 1 needs review, 2 likely defect, 3 scan error.
With --fail-on review|likely the exit code is 0 unless that level is reached
(useful as a publish gate in CI).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .baseline import BaselineError, apply_baseline, collect, load_baseline, write_baseline
from .config import ConfigError, Config, apply_config, find_config, load_config
from .core import finalize, rule_table, scan_package
from .package import Package
from .rules import ENABLED, OFF_BY_DEFAULT, OFF_VARIANTS, RULES
from .sarif import build_sarif, write_sarif

LEVEL = {"clean": 0, "review": 1, "likely": 2}


def load_target(target: str) -> tuple[Package, str]:
    if os.path.isdir(target):
        return Package.from_dir(target), os.path.abspath(target)
    if os.path.isfile(target) and target.endswith((".tar.gz", ".tgz")):
        with open(target, "rb") as f:
            return Package.from_tar_stream(f, target), target
    if "/" in target and not os.path.exists(target):
        from .hub import fetch
        env, _, ver = target.partition("@")
        return fetch(env, ver or "latest"), env
    raise SystemExit(f"envscan: not a directory, archive or owner/name: {target}")


def find_tasks(path: str) -> list[str]:
    """Task folders (each holding a task.toml) directly under `path`, if it is a
    Harbor dataset: two or more of them and no task.toml of its own."""
    if not os.path.isdir(path) or os.path.isfile(os.path.join(path, "task.toml")):
        return []
    try:
        names = sorted(n for n in os.listdir(path)
                       if os.path.isfile(os.path.join(path, n, "task.toml")) and not os.path.islink(os.path.join(path, n)))
    except OSError:
        return []
    return names if len(names) >= 2 else []


def render(res: dict, verbose=False) -> str:
    out = [f"{res['env']}: {res['verdict'].upper()}" + (f" ({res['top_severity']})" if res["top_severity"] else "")]
    order = {"likely": 0, "review": 1}
    sev = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    fs = sorted(res["findings"], key=lambda f: (f["kind"] == "hygiene", order.get(f["confidence"], 2), sev.get(f["severity"], 5)))
    for f in fs:
        if f["kind"] == "hygiene" and not verbose:
            continue
        out.append(f"  [{f['confidence']:6}] {f['severity']:8} {f['rule']}  ({f['kind']})")
        out.append(f"           {f['message']}")
        for e in f["evidence"][:4]:
            out.append(f"           at {e}")
    hyg = [f for f in fs if f["kind"] == "hygiene"]
    if hyg and not verbose:
        out.append(f"  + {len(hyg)} hygiene note(s): " + ", ".join(sorted({f['rule'] for f in hyg})) + " (use -v)")
    extra = []
    if res.get("suppressed"):
        extra.append(f"{len(res['suppressed'])} suppressed by config")
    if res.get("baselined"):
        extra.append(f"{res['baselined']} in baseline")
    if extra:
        out.append("  (" + ", ".join(extra) + ")")
    if res.get("rule_errors"):
        out.append("  rule errors: " + "; ".join(res["rule_errors"]))
    return "\n".join(out)


def render_task_line(res: dict) -> str:
    rules = sorted({f["rule"] for f in res["findings"] if f["kind"] != "hygiene"})
    line = f"{res['env']}: {res['verdict'].upper()}" + (f" ({res['top_severity']})" if res["top_severity"] else "")
    return line + ("  " + ", ".join(rules) if rules else "")


def summary_line(results: list[dict]) -> str:
    n = {"likely": 0, "review": 0, "clean": 0}
    for r in results:
        n[r["verdict"]] += 1
    return f"{len(results)} tasks: {n['likely']} likely, {n['review']} review, {n['clean']} clean"


# A rule that fires on this share of a dataset's tasks is a property of the dataset, not of a task.
DATASET_WIDE_SHARE = 0.9
DATASET_WIDE_MIN_TASKS = 5


def dataset_notes(results: list[dict]) -> list[str]:
    """Findings that are true of the dataset as a whole, said once."""
    total = len(results)
    per_task = [{f["rule"] for f in r["findings"] if f["kind"] != "hygiene"} for r in results]
    count = {}
    for rules in per_task:
        for x in rules:
            count[x] = count.get(x, 0) + 1
    out, wide = [], set()
    n = count.get("harbor-shared-grading", 0)
    if n >= 2:
        wide.add("harbor-shared-grading")
        out.append(f"dataset: {n} of {total} tasks grade in the agent's container (harbor-shared-grading, "
                   "Harbor's default). One change fixes all of them: a separate verifier.")
    if total < DATASET_WIDE_MIN_TASKS:
        return out
    for x in sorted(count, key=lambda k: (-count[k], k)):
        if x not in wide and count[x] >= DATASET_WIDE_SHARE * total:
            wide.add(x)
            out.append(f"dataset: {count[x]} of {total} tasks share {x}. It comes from the dataset's task template, "
                       "so fix it once there.")
    if wide:
        own = [r["env"] for r, rules in zip(results, per_task) if rules - wide]
        line = f"dataset: {len(own)} of {total} tasks have findings of their own beyond the dataset-wide ones"
        if own:
            line += ": " + ", ".join(own[:8]) + (f" and {len(own) - 8} more" if len(own) > 8 else "")
        out.append(line + ("" if own else "."))
    return out


def _fallback_file(pkg: Package) -> str | None:
    for p in ("pyproject.toml", "task.toml"):
        if p in pkg.files or p in pkg.sizes:
            return p
    return min(pkg.sizes, default=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="envscan", description="Static reward hacking and broken grader checks for verifiers environments.")
    ap.add_argument("target", nargs="*", help="env directory, folder of Harbor tasks, .tar.gz, or Hub id owner/name[@version]")
    ap.add_argument("--json", action="store_true", help="print JSON results")
    ap.add_argument("-v", "--verbose", action="store_true", help="include hygiene notes (and findings per task in dataset mode)")
    ap.add_argument("--fail-on", choices=["review", "likely"], help="exit 1 only at or above this level")
    ap.add_argument("--sarif", metavar="FILE", help="also write a SARIF 2.1.0 log (for GitHub code scanning)")
    ap.add_argument("--sarif-base", metavar="DIR", help="make SARIF paths relative to DIR (for example the repo root) instead of the scanned folder")
    ap.add_argument("--config", metavar="FILE", help="config file (default: .envscan.toml at the root of a scanned local folder)")
    ap.add_argument("--baseline", metavar="FILE", help="only report findings that are not in this baseline")
    ap.add_argument("--write-baseline", metavar="FILE", help="save the current findings as a baseline and exit 0 (unless a scan failed)")
    ap.add_argument("--rules", action="store_true", help="list rules and exit")
    ap.add_argument("--enable", metavar="RULE", action="append", default=[], help="also run a rule, or a rule-id:variant branch, that is off by default (repeatable)")
    ap.add_argument("--version", action="version", version=f"envscan {__version__}")
    a = ap.parse_args(argv)
    if a.rules:
        for r in rule_table():
            off = " (off by default, use --enable)" if r["id"] in OFF_BY_DEFAULT else ""
            print(f"{r['id']:32} {r['kind']:8} {r['severity']:9} {r['description']}{off}")
            for key in sorted(k for k in OFF_VARIANTS if k.split(":", 1)[0] == r["id"]):
                print(f"    variant {key.split(':', 1)[1]} off by default, use --enable {key}")
        return 0
    for rid in a.enable:
        if rid not in RULES and rid not in OFF_VARIANTS:
            print(f"envscan: unknown rule {rid!r} for --enable", file=sys.stderr)
            return 3
        ENABLED.add(rid)
    if not a.target:
        ap.print_help()
        return 3
    explicit_cfg = None
    base = None
    try:
        if a.config:
            explicit_cfg = load_config(a.config, RULES)
        if a.baseline:
            base = load_baseline(a.baseline)
    except (ConfigError, BaselineError) as e:
        print(f"envscan: {e}", file=sys.stderr)
        return 3
    multi = len(a.target) > 1
    sarif_base = os.path.abspath(a.sarif_base) if a.sarif_base else None
    worst = 0
    results = []
    sarif_in = []
    keys = set()
    dataset_used = False
    printed_blocks = []

    def finish(pkg, res, cfg, scope, prefix):
        if cfg is not None and not cfg.empty:
            apply_config(res, cfg, scope)
        res.setdefault("suppressed", [])
        if base is not None:
            apply_baseline(pkg, res, base, scope)
        if a.write_baseline:
            keys.update(collect(pkg, res, scope))
        finalize(res)
        sarif_in.append((res, prefix, _fallback_file(pkg)))
        return res

    for t in a.target:
        try:
            tasks = find_tasks(t)
            cfg = explicit_cfg
            if cfg is None and os.path.isdir(t):
                cp = find_config(t)
                cfg = load_config(cp, RULES) if cp else None
            tname = os.path.basename(os.path.abspath(t)) if os.path.isdir(t) else os.path.basename(t)
            lead = ""
            if sarif_base and os.path.isdir(t):
                rel = os.path.relpath(os.path.abspath(t), sarif_base).replace(os.sep, "/")
                lead = "" if rel == "." else rel
                if lead.startswith(".."):
                    lead = ""
            elif multi:
                lead = tname
            if tasks:
                dataset_used = True
                group = []
                for n in tasks:
                    d = os.path.join(t, n)
                    pkg = Package.from_dir(d)
                    res = scan_package(pkg, n)
                    scope = f"{tname}/{n}" if multi else n
                    finish(pkg, res, cfg, scope, f"{lead}/{n}" if lead else n)
                    res.pop("file_hashes", None) if not a.json else None
                    group.append(res)
                    results.append(res)
                    worst = max(worst, LEVEL[res["verdict"]])
                if not a.json:
                    block = []
                    for res in group:
                        block.append(render(res, a.verbose) if a.verbose else render_task_line(res))
                    block.extend(dataset_notes(group))
                    block.append(summary_line(group))
                    print("\n".join(block))
            else:
                pkg, name = load_target(t)
                res = scan_package(pkg, name)
                finish(pkg, res, cfg, tname if multi else "", lead)
                res.pop("file_hashes", None) if not a.json else None
                results.append(res)
                worst = max(worst, LEVEL[res["verdict"]])
                if not a.json:
                    print(render(res, a.verbose))
        except ConfigError as e:
            print(f"envscan: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            print(f"{t}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
            worst = max(worst, 3)
            continue
    if a.json:
        print(json.dumps(results if (len(results) != 1 or dataset_used) else results[0], indent=2))
    if a.sarif and sarif_in:
        write_sarif(a.sarif, build_sarif(rule_table(), sarif_in))
    if a.write_baseline:
        if worst == 3:
            print("envscan: not writing a baseline because a scan failed", file=sys.stderr)
            return 3
        n = write_baseline(a.write_baseline, keys)
        print(f"envscan: wrote {n} baseline entries to {a.write_baseline}", file=sys.stderr)
        return 0
    if a.fail_on:
        if worst == 3:
            return 3
        return 1 if worst >= LEVEL[a.fail_on] else 0
    return worst

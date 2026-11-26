"""Project config (.envscan.toml): ignore rules, ignore paths, suppress single findings.

Format::

    [[ignore-rule]]
    rule = "keyword-reward"
    reason = "our metric is a keyword check on purpose"

    [[ignore-path]]
    glob = "tests/fixtures/**"
    reason = "fixtures are deliberately broken"

    [[suppress]]
    rule = "exit-code-verdict"
    path = "tests/test.sh"
    reason = "reviewed, verifier also writes reward.txt"

Every entry needs a non empty `reason`. Matching works on the file part of a
finding's evidence (`file:line`). Evidence entries that match are dropped from
the finding; a finding left with no evidence is suppressed, left out of the
verdict, and listed under "suppressed" in JSON output. A finding that never had
evidence is only suppressed by an ignore-rule.
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

CONFIG_NAME = ".envscan.toml"
_KEYS = {"ignore-rule": {"rule", "reason"}, "ignore-path": {"glob", "reason"}, "suppress": {"rule", "path", "reason"}}


class ConfigError(Exception):
    pass


def glob_to_regex(glob: str) -> re.Pattern:
    """`*` stays inside one folder, `**` crosses folders, `?` is one character."""
    g = glob.replace("\\", "/")
    while g.startswith("./"):
        g = g[2:]
    out, i = [], 0
    while i < len(g):
        c = g[i]
        if c == "*":
            if g[i:i + 2] == "**":
                i += 2
                if g[i:i + 1] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z", re.S)


def _path_match(rx: re.Pattern, glob: str, path: str, scope: str) -> bool:
    cands = [path] + ([f"{scope}/{path}"] if scope else [])
    for c in cands:
        if rx.match(c):
            return True
        # a plain folder name or file name matches anywhere below / at that name
        if not any(ch in glob for ch in "*?") and (c == glob.strip("/") or c.startswith(glob.strip("/") + "/")):
            return True
    return False


@dataclass
class Config:
    ignore_rules: dict[str, str] = field(default_factory=dict)          # rule -> reason
    ignore_paths: list[tuple[str, re.Pattern, str]] = field(default_factory=list)   # glob, regex, reason
    suppress: list[tuple[str, str, re.Pattern, str]] = field(default_factory=list)  # rule, glob, regex, reason
    source: str | None = None

    @property
    def empty(self) -> bool:
        return not (self.ignore_rules or self.ignore_paths or self.suppress)


def parse_config(text: str, source: str = CONFIG_NAME, known_rules=None) -> Config:
    if tomllib is None:  # pragma: no cover
        raise ConfigError("config files need Python 3.11+ (tomllib)")
    try:
        data = tomllib.loads(text)
    except Exception as e:
        raise ConfigError(f"{source}: invalid TOML: {e}")
    cfg = Config(source=source)
    for key in data:
        if key not in _KEYS:
            raise ConfigError(f"{source}: unknown section [{key}] (expected {', '.join(sorted(_KEYS))})")
    for key, allowed in _KEYS.items():
        entries = data.get(key, [])
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise ConfigError(f"{source}: [[{key}]] must be an array of tables")
        for i, e in enumerate(entries, 1):
            where = f"{source}: [[{key}]] #{i}"
            extra = set(e) - allowed
            if extra:
                raise ConfigError(f"{where}: unknown key {sorted(extra)[0]!r}")
            for k in allowed:
                v = e.get(k)
                if not isinstance(v, str) or not v.strip():
                    raise ConfigError(f"{where}: {k!r} is required" + (" (say why it is safe to ignore)" if k == "reason" else ""))
            if "rule" in e and known_rules is not None and e["rule"] not in known_rules:
                raise ConfigError(f"{where}: unknown rule id {e['rule']!r}")
            if key == "ignore-rule":
                cfg.ignore_rules[e["rule"]] = e["reason"].strip()
            elif key == "ignore-path":
                cfg.ignore_paths.append((e["glob"], glob_to_regex(e["glob"]), e["reason"].strip()))
            else:
                cfg.suppress.append((e["rule"], e["path"], glob_to_regex(e["path"]), e["reason"].strip()))
    return cfg


def load_config(path: str, known_rules=None) -> Config:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise ConfigError(f"cannot read config {path}: {e}")
    return parse_config(text, path, known_rules)


def find_config(target_dir: str) -> str | None:
    """`.envscan.toml` at the scan root of a local directory (never searched upward)."""
    p = os.path.join(target_dir, CONFIG_NAME)
    return p if os.path.isfile(p) and not os.path.islink(p) else None


def evidence_path(e: str) -> str:
    return e.split(":", 1)[0]


def apply_config(res: dict, cfg: Config, scope: str = "") -> dict:
    """Move suppressed findings out of res["findings"] into res["suppressed"].

    Returns res. The caller recomputes the verdict afterwards."""
    kept, suppressed = [], list(res.get("suppressed", []))
    for f in res["findings"]:
        if f["kind"] == "hygiene":
            kept.append(f)
            continue
        rid = f["rule"]
        if rid in cfg.ignore_rules:
            suppressed.append({**f, "suppressed_by": "ignore-rule", "reason": cfg.ignore_rules[rid]})
            continue
        ev = f["evidence"]
        remaining, reason, by = [], None, None
        for e in ev:
            p = evidence_path(e)
            hit = None
            for glob, rx, why in cfg.ignore_paths:
                if _path_match(rx, glob, p, scope):
                    hit = ("ignore-path", why)
                    break
            if hit is None:
                for srule, glob, rx, why in cfg.suppress:
                    if srule == rid and _path_match(rx, glob, p, scope):
                        hit = ("suppress", why)
                        break
            if hit:
                by, reason = hit
            else:
                remaining.append(e)
        if ev and not remaining:
            suppressed.append({**f, "suppressed_by": by, "reason": reason})
        elif len(remaining) != len(ev):
            kept.append({**f, "evidence": remaining})
        else:
            kept.append(f)
    res["findings"] = kept
    res["suppressed"] = suppressed
    return res

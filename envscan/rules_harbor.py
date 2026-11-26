"""Harbor task rules (envscan v0.5, builder `harbor`).

These rules read Harbor task folders as text: task.toml, instruction.md,
tests/test.sh (and steps/*/tests/test.sh), solution/solve.sh, the agent
Dockerfile and compose files under environment/. Nothing is run.

The task index (task roots, verifier mode, agent user, verifier scripts, agent
Dockerfile) is the one rules_image.py builds, reused by import so every Harbor
rule reads a task the same way.
"""
from __future__ import annotations

import difflib
import os
import re

from .package import Package
from .rules import F, rule, ev
from .rules_image import index, sh_lines, docker_instrs, is_dockerfile, HEREDOC

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _tasks(pkg: Package):
    """Harbor tasks in the package (fast empty answer when there is no task.toml)."""
    if not any(p == "task.toml" or p.endswith("/task.toml") for p in pkg.files):
        return []
    return index(pkg).tasks


def _pre(tk) -> str:
    return (tk.root + "/") if tk.root else ""


def _task_files(pkg: Package, tk, sub: str):
    """(path, text) of text files under <task>/<sub>/."""
    pre = _pre(tk) + sub.rstrip("/") + "/"
    return [(p, t) for p, t in pkg.files.items() if p.startswith(pre)]


def _all_scripts(tk):
    return list(tk.scripts)


PLACEHOLDER_LINE = re.compile(r"^\s*\{\{?\s*[A-Za-z_]\w*\s*\}?\}\s*$", re.M)
REWARD_FILE = r"(?:/logs/verifier/reward\.(?:txt|json)|\$\{?\w*REWARD\w*\}?|\$\{?\w*(?:VERIFIER|LOG)\w*\}?/reward\.(?:txt|json)|\breward\.txt\b)"
REWARD_ONE = re.compile(rf"(?:echo|printf)\s+(?:-n\s+|-e\s+)*['\"]?(?:1|1\.0|1\.00)['\"]?\s*(?:\\n['\"]?)?\s*>>?\s*['\"]?{REWARD_FILE}")
REWARD_ZERO = re.compile(rf"(?:echo|printf)\s+(?:-n\s+|-e\s+)*['\"]?(?:0|0\.0)['\"]?\s*>>?\s*['\"]?{REWARD_FILE}")
REWARD_ANY = re.compile(rf">>?\s*['\"]?{REWARD_FILE}")
EVAL_CMD = re.compile(
    r"(\bpytest\b|\bpy\.test\b|\bpython[\d.]*\s+(?:-\w+\s+)*(?:-m\s+(?:pytest|unittest)|\S+\.py)|\buv\s+run\b|\buvx\b|"
    r"\bbash\s+\S+\.sh|\bsh\s+\S+\.sh|(?:^|[\s;&|(])\./[\w./-]+|\bnpm\s+(?:run\s+)?test|\bnpx\s+(?:jest|vitest|mocha)|"
    r"\byarn\s+test|\bcargo\s+test|\bgo\s+test|\bmake\s+(?:test|check)|\bmvn\b[^\n]*\btest|\bgradlew?\b[^\n]*\btest|"
    r"\bdotnet\s+test|\bctest\b|\bnode\s+\S+\.m?js|\bRscript\b|\bjulia\b)")


def _shell_state(text: str):
    """Per logical line: (line, cmd, errexit, pipefail, depth). depth counts open if/case/function/loop blocks."""
    first = text.split("\n", 1)[0]
    errexit = bool(re.match(r"#!\S*(?:\s+\S+)*\s+-\w*e", first))
    pipefail = False
    depth = 0
    out = []
    for ln, line in sh_lines(text):
        s = line.strip()
        for m in re.finditer(r"(?:^|[;&|]\s*)set\s+([^;&|]+)", s):
            toks = m.group(1).split()
            for k, tok in enumerate(toks):
                if not tok or tok[0] not in "-+" or len(tok) < 2:
                    continue
                on = tok[0] == "-"
                letters = tok[1:]
                nxt = toks[k + 1] if k + 1 < len(toks) else ""
                if "e" in letters:
                    errexit = on
                if "o" in letters and nxt in ("pipefail", "errexit"):
                    if nxt == "pipefail":
                        pipefail = on
                    else:
                        errexit = on
        out.append((ln, s, errexit, pipefail, depth))
        opens = len(re.findall(r"(?:^|[;&|]\s*|\s)(?:if|case|for|while|until)\b", " " + s))
        opens += 1 if re.search(r"(\(\)\s*\{|^\s*function\s+\w+.*\{|\{\s*$)", s) and not re.search(r"\}\s*$", s) else 0
        closes = len(re.findall(r"(?:^|[;&|]\s*|\s)(?:fi|esac|done)\b", " " + s))
        closes += 1 if re.match(r"\}", s) else 0
        depth = max(0, depth + opens - closes)
    return out


def _split_pipe(cmd: str):
    """Top level pipeline segments of a command (|| and && kept inside segments)."""
    return re.split(r"(?<![|>&])\|(?![|&])", cmd)


# ---------------------------------------------------------------------------
# 1. verifier-applies-reference-patch
# canon: verifier-applies-reference-patch, reader items, main paper arxiv:2608.17393
# ---------------------------------------------------------------------------

APPLY = re.compile(r"\b(?:git\s+(?:-C\s+\S+\s+)?(?:apply|am)\b(?P<a>[^;&|\n]*)|patch\b(?P<p>[^;&|\n]*))")
COPY = re.compile(r"\b(?:cp|mv|rsync|install)\s+(?P<args>[^;&|\n<>]+)")
CAT_REDIR = re.compile(r"\bcat\s+(?P<src>[^\s;&|<>]+)\s*>\s*(?P<dst>\S+)")
SOLUTION_PATH = re.compile(r"(?:^|[\s\"'=])(/(?:solution|oracle)(?:/[^\s\"';|&]*)?)")


def _sig_lines(text: str):
    out = set()
    for raw in text.splitlines():
        s = raw.strip()
        if len(s) >= 8 and re.search(r"[A-Za-z]{3}", s) and not s.startswith(("#", "//", "---", "+++", "@@", "diff --git", "index ")):
            out.add(s)
    return out


def _patch_added(text: str):
    added = set()
    for raw in text.splitlines():
        if raw.startswith("+") and not raw.startswith("+++"):
            s = raw[1:].strip()
            if len(s) >= 8 and re.search(r"[A-Za-z]{3}", s):
                added.add(s)
    return added


def _heredocs(text: str):
    """{start line of the opener: body text} for every heredoc in a shell script."""
    out, lines, i = {}, text.splitlines(), 0
    while i < len(lines):
        m = HEREDOC.search(lines[i])
        if m and not lines[i].lstrip().startswith("#"):
            tag, body, j = m.group(2), [], i + 1
            while j < len(lines) and lines[j].strip() != tag:
                body.append(lines[j])
                j += 1
            out[i + 1] = "\n".join(body)
            i = j + 1
            continue
        i += 1
    return out


def _gold_lines(pkg: Package, tk):
    gold = set()
    for p, t in _task_files(pkg, tk, "solution"):
        if p.endswith((".patch", ".diff")) or t.lstrip().startswith(("diff --git", "--- ")):
            gold |= _patch_added(t)
        elif p.endswith(".sh"):
            for body in _heredocs(t).values():
                gold |= _patch_added(body) if re.search(r"^\+\+\+ ", body, re.M) else _sig_lines(body)
        else:
            gold |= _sig_lines(t)
    return gold


def _resolve_tests_file(pkg: Package, tk, path: str):
    """Text of a /tests/... path from the task's tests/ folder (or step tests)."""
    path = path.strip("\"'")
    m = re.match(r"(?:/tests|\$\{?TEST_DIR\}?|\$\{?TESTS_DIR\}?)/(.+)", path)
    if not m:
        return None
    rel = m.group(1)
    pre = _pre(tk)
    for cand in [pre + "tests/" + rel] + [p for p in pkg.files if p.startswith(pre + "steps/") and p.endswith("/tests/" + rel)]:
        if cand in pkg.files:
            return pkg.files[cand]
    return None


@rule("verifier-applies-reference-patch", "hack", "critical",
      "The verifier applies the reference patch or copies a solution file into the workspace before it runs the tests.")
def r_applies_reference(pkg: Package):
    out = []
    for tk in _tasks(pkg):
        gold = None
        for sp in _all_scripts(tk):
            text = pkg.files.get(sp, "")
            if not text or not re.search(r"(apply|patch|cp|mv|rsync|install|cat|solution|oracle)", text):
                continue
            bodies = _heredocs(text)
            written = {}  # path written by a heredoc -> body
            for ln, body in bodies.items():
                opener = text.splitlines()[ln - 1]
                mw = re.search(r">\s*['\"]?([^\s'\"]+)", opener)
                if mw:
                    written[mw.group(1)] = body
            lines = sh_lines(text)
            last_eval = max([ln for ln, l in lines if EVAL_CMD.search(l) or REWARD_ANY.search(l)] or [0])
            for ln, line in lines:
                if ln >= last_eval:
                    break
                # (a) a path under /solution or /oracle: only the oracle run has it
                ms = SOLUTION_PATH.search(line)
                if ms and (APPLY.search(line) or COPY.search(line) or CAT_REDIR.search(line) or re.search(r"\bsource\b|^\.\s", line)):
                    out.append((ev(sp, ln), "broken", f"The verifier script reads {ms.group(1)}, which Harbor uploads only for the oracle run, "
                                "so the oracle can pass while agents are graded on something else"))
                    continue
                srcs = []
                ma = APPLY.search(line)
                if ma:
                    args = (ma.group("a") if ma.group("a") is not None else ma.group("p")) or ""
                    if ma.group("p") is not None and not re.search(r"(-p\s*\d|-i\s|<|\.(patch|diff))", args):
                        continue
                    m_in = re.search(r"(?:-i\s*|<\s*)(\S+)", args)
                    toks = [m_in.group(1)] if m_in else [a for a in args.split() if not a.startswith("-")]
                    if "<<" in line:
                        body = bodies.get(ln)
                        if body is not None:
                            srcs.append(("<heredoc>", body, True))
                    for a in toks:
                        srcs.append((a, None, True))
                for mc in COPY.finditer(line):
                    args = [a for a in mc.group("args").split() if not a.startswith("-")]
                    if len(args) >= 2:
                        dst = args[-1]
                        if re.match(r"['\"]?(/logs/|/tmp/|/dev/)", dst):
                            continue
                        for a in args[:-1]:
                            srcs.append((a, None, False))
                mcat = CAT_REDIR.search(line)
                if mcat and not re.match(r"['\"]?(/logs/|/tmp/|/dev/)", mcat.group("dst")):
                    srcs.append((mcat.group("src"), None, False))
                for src, body, is_patch in srcs:
                    if body is None:
                        s = src.strip("\"'")
                        body = written.get(s)
                        if body is None:
                            body = _resolve_tests_file(pkg, tk, s)
                    if not body:
                        continue
                    if gold is None:
                        gold = _gold_lines(pkg, tk)
                    if not gold:
                        break
                    cand = _patch_added(body) if (is_patch or re.search(r"^\+\+\+ ", body, re.M)) else _sig_lines(body)
                    if len(cand) < 3:
                        continue
                    common = cand & gold
                    if len(common) >= 3 and len(common) >= 0.5 * len(cand):
                        what = "applies a patch" if is_patch else "copies a file"
                        out.append((ev(sp, ln), "hack", f"The verifier script {what} whose lines match the reference solution "
                                    f"({len(common)} of {len(cand)} lines) before the tests run, so the tests pass whatever the agent did"))
    if not out:
        return []
    hack = [o for o in out if o[1] == "hack"]
    first = (hack or out)[0]
    msg = first[2][0].upper() + first[2][1:] if not first[2].startswith("test.sh") else first[2]
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    return [F("verifier-applies-reference-patch", msg + ".", [o[0] for o in out], confidence="review",
              kind=first[1], severity="critical" if first[1] == "hack" else "high")]


# ---------------------------------------------------------------------------
# 2. shell-or-true-fail-open (extends grader-fails-open, which reads Python only)
# canon: shell-or-true-fail-open, reader items
# ---------------------------------------------------------------------------

STATUS_CAPTURE = re.compile(r"^(?:local\s+|export\s+)?(\w+)=\$\?\s*$")
STATUS_TEST = re.compile(r"(\[\[?|\btest\b|\(\()[^\]\n)]*\$\{?\?\}?")
NEG_GREP = re.compile(r"!\s*grep\s+(?:-\w+\s+)*['\"]?[^'\"\n]*(FAIL|Fail|fail|ERROR|Error|error|Traceback|Exception)", re.I)
COUNT_FAIL = re.compile(r"^(?:local\s+)?(\w+)=\$\(\s*grep\s+(?:-\w*c\w*)\s+(?:-\w+\s+)*['\"]?[^'\"\n)]*(FAIL|fail|ERROR|error)")
POSITIVE = re.compile(r"(PASS|pass|OK\b|ok\b|success|SUCCESS|correct|\bgrep\s+-\w*q\w*\s+['\"]?(?!.*(fail|error)))")


def _then_writes_one(rows, i, span=4) -> bool:
    for j in range(i, min(len(rows), i + span)):
        if REWARD_ONE.search(rows[j][1]):
            return True
        if j > i and re.match(r"(else|elif|fi)\b", rows[j][1]):
            return False
    return False


def _fail_open_hits(text: str):
    if PLACEHOLDER_LINE.search(text):
        return []  # adapter template with code spliced in at build time: contents unknown
    rows = _shell_state(text)
    hits = []
    pipestatus = "PIPESTATUS" in text
    flawed = {}  # var -> reason
    for i, (ln, s, errexit, pipefail, depth) in enumerate(rows):
        # status captured right after a pipeline or an `|| true`
        cap = STATUS_CAPTURE.match(s)
        test_here = STATUS_TEST.search(s) and re.match(r"(if|elif|while)\b|\[", s)
        if (cap or test_here) and i > 0:
            prod = rows[i - 1][1]
            reason = None
            if EVAL_CMD.search(prod):
                segs = _split_pipe(prod)
                if re.search(r"\|\|\s*(true|:)\s*$", prod):
                    reason = "the test command ends in || true, so $? is always 0"
                elif len(segs) > 1 and EVAL_CMD.search(segs[0]) and not rows[i - 1][3] and not pipestatus:
                    reason = "the test command is piped into " + segs[-1].split()[0] + " with no pipefail, so $? is the status of the last command in the pipe"
            if reason:
                if cap:
                    flawed[cap.group(1)] = (ln, reason)
                elif _then_writes_one(rows, i) and len(re.findall(r"\$\{?\w+", s)) <= 1:
                    hits.append((ln, reason))
        # a later condition on a flawed status variable that pays 1
        if flawed and re.match(r"(if|elif)\b|\[\[?", s):
            used = [v for v in flawed if re.search(rf"\$\{{?{v}\b", s)]
            others = [v for v in re.findall(r"\$\{?(\w+)", s) if v not in flawed and not v.isdigit()]
            if used and not others and _then_writes_one(rows, i):
                hits.append((flawed[used[0]][0], flawed[used[0]][1]))
        # pay 1 when no failure marker is found in a log
        if re.match(r"(if|elif)\b", s) and NEG_GREP.search(s) and not POSITIVE.search(NEG_GREP.sub("", s)) \
                and _then_writes_one(rows, i):
            prior = [r for r in rows[:i] if EVAL_CMD.search(r[1])]
            if prior and (not prior[-1][2] or re.search(r"\|\|\s*(true|:)", prior[-1][1])):
                hits.append((ln, "the reward is 1 when the log has no failure line, so a test run that never starts or prints nothing passes"))
        mc = COUNT_FAIL.match(s)
        if mc:
            v = mc.group(1)
            for j in range(i + 1, min(len(rows), i + 12)):
                c = rows[j][1]
                if re.search(rf"\$\{{?{v}\}}?\"?\s*(-eq|==|=)\s*\"?0\b|\$\{{?{v}\}}?\"?\s*-lt\s+1\b", c) and \
                        not re.search(r"\$\{?\w*(pass|PASS|total|TOTAL|count|COUNT|exit|EXIT|rc|RC|status|STATUS)\w*", c.replace(v, "")) \
                        and _then_writes_one(rows, j):
                    prior = [r for r in rows[:i] if EVAL_CMD.search(r[1])]
                    if prior and (not prior[-1][2] or re.search(r"\|\|\s*(true|:)", prior[-1][1])):
                        hits.append((rows[j][0], "the reward is 1 when the count of failure lines is 0, so a test run that never starts passes"))
                    break
    # an unconditional 1 at top level after the tests, with errexit off and no other reward branch
    if not REWARD_ZERO.search(text) and "trap" not in text:
        evals = []
        for ln, s, errexit, pipefail, depth in rows:
            if depth == 0 and EVAL_CMD.search(s) and not re.search(r"\|\|\s*(exit|return|\{)", s) and not s.startswith(("if ", "elif ")):
                evals.append((ln, errexit))
            if depth == 0 and REWARD_ONE.search(s) and not re.search(r"(&&|\|\|)", s) and evals and not evals[-1][1]:
                hits.append((ln, "reward 1 is written after the tests with no check of their result"))
                break
    seen, out = set(), []
    for ln, r in hits:
        if ln not in seen:
            seen.add(ln)
            out.append((ln, r))
    return out


@rule("shell-or-true-fail-open", "hack", "high",
      "A Harbor test.sh loses the test result, by || true, a pipe with no pipefail or a check for missing failure lines, and still pays 1.")
def r_shell_fail_open(pkg: Package):
    out, reasons = [], []
    for tk in _tasks(pkg):
        for sp in _all_scripts(tk):
            text = pkg.files.get(sp, "")
            if "reward" not in text:
                continue
            for ln, reason in _fail_open_hits(text):
                out.append(ev(sp, ln))
                reasons.append(reason)
    if not out:
        return []
    msg = "In the verifier script " + reasons[0] + ", so a failing or missing test run still writes reward 1"
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    return [F("shell-or-true-fail-open", msg + ".", out, confidence="review")]


# ---------------------------------------------------------------------------
# 3. task-toml-unknown-keys
# canon: task-toml-unknown-keys, reader items (schema: harbor/src/harbor/models/task/config.py,
# which does not forbid extra keys, so pydantic drops them without a word)
# ---------------------------------------------------------------------------

_ANY = "*"
_NET = {"network_mode": None, "allowed_hosts": None}
_HEALTH = {k: None for k in ("command", "interval_sec", "timeout_sec", "start_period_sec", "start_interval_sec", "retries")}
_MCP = {"name": None, "transport": None, "url": None, "command": None, "args": None, "headers": _ANY}
_ENVC = dict(_NET, build_timeout_sec=None, docker_image=None, os=None, cpus=None, memory_mb=None, storage_mb=None,
             gpus=None, gpu_types=None, tpu={"type": None, "topology": None}, mcp_servers=[_MCP], env=_ANY,
             skills_dir=None, healthcheck=_HEALTH, workdir=None, allow_internet=None, memory=None, storage=None)
_COLLECT = {"command": None, "service": None, "timeout_sec": None, "user": None}
_ART = {"source": None, "destination": None, "exclude": None, "service": None}
_AGENT = dict(_NET, timeout_sec=None, user=None)
_VERIFIER = dict(_NET, timeout_sec=None, env=_ANY, user=None, environment_mode=None, environment=_ENVC, collect=[_COLLECT])
_STEP = {"name": None, "agent": _AGENT, "verifier": _VERIFIER, "min_reward": _ANY, "healthcheck": _HEALTH, "artifacts": [_ART]}
SCHEMA = {
    "schema_version": None, "version": None, "source": None, "multi_step_reward_strategy": None,
    "task": {"name": None, "version": None, "description": None, "keywords": None,
             "authors": [{"name": None, "email": None}]},
    "metadata": _ANY, "verifier": _VERIFIER, "agent": _AGENT, "environment": _ENVC,
    "solution": {"env": _ANY}, "steps": [_STEP], "artifacts": [_ART],
}
# keys whose silent default changes who grades or what the agent can do
RISKY = {"environment_mode", "environment", "user", "network_mode", "allow_internet", "allowed_hosts", "verifier", "agent"}


def _unknown(value, schema, prefix=""):
    if schema is _ANY or schema is None:
        return []
    if isinstance(schema, list):
        if not isinstance(value, list):
            return []
        out = []
        for i, v in enumerate(value):
            if isinstance(v, dict):
                out += _unknown(v, schema[0], f"{prefix}[{i}]")
        return out
    if not isinstance(value, dict):
        return []
    out = []
    for k, v in value.items():
        loc = f"{prefix}.{k}" if prefix else k
        if k in schema:
            out += _unknown(v, schema[k], loc)
        else:
            m = difflib.get_close_matches(k, list(schema), n=1, cutoff=0.75)
            out.append((loc, k, m[0] if m else None))
    return out


def _key_line(text: str, key: str) -> int:
    m = re.search(rf"^\s*\[*\s*(?:[\w.]+\.)?{re.escape(key)}\s*[\]=]", text, re.M)
    if not m:
        m = re.search(rf"\b{re.escape(key)}\b", text)
    return text.count("\n", 0, m.start()) + 1 if m else 1


@rule("task-toml-unknown-keys", "broken", "medium",
      "A Harbor task.toml has a key Harbor does not read, an empty instruction.md or no test script, so a setting or the grader is silently missing.")
def r_toml_unknown(pkg: Package):
    out, msgs, risky, typos, files_bad = [], [], False, 0, 0
    generated_pkg = None
    for tk in _tasks(pkg):
        text = pkg.files.get(tk.toml, "")
        cfg = tk.cfg
        if not cfg:
            continue  # template placeholders or a broken file: Harbor would refuse it loudly
        for loc, key, near in _unknown(cfg, SCHEMA):
            out.append(ev(tk.toml, _key_line(text, key)))
            if near:
                typos += 1
                msgs.insert(typos - 1, f"The task.toml key {loc} is not in the Harbor schema (did you mean {near}?), so Harbor ignores it and keeps the default")
                risky = risky or near in RISKY
            else:
                msgs.append(f"The task.toml key {loc} is not in the Harbor schema, so Harbor ignores it")
        pre = _pre(tk)
        steps = cfg.get("steps") if isinstance(cfg.get("steps"), list) else []
        names = [s.get("name") for s in steps if isinstance(s, dict) and isinstance(s.get("name"), str)] or [None]
        envc = cfg.get("environment") if isinstance(cfg.get("environment"), dict) else {}
        windows = str(envc.get("os", "")).lower() == "windows"
        ver = cfg.get("verifier") if isinstance(cfg.get("verifier"), dict) else {}
        for name in names:
            ins = pre + (f"steps/{name}/instruction.md" if name else "instruction.md")
            if generated_pkg is None and ins not in pkg.files:
                generated_pkg = any("instruction.md" in t for p, t in pkg.files.items() if p.endswith(".py"))
            if (generated_pkg and ins not in pkg.files) or "template" in tk.root.lower():
                pass  # the package's own code writes instruction.md (an adapter template)
            elif ins in pkg.files:
                body = re.sub(r"<!--.*?-->", "", pkg.files[ins], flags=re.S)
                body = "\n".join(l for l in body.splitlines() if "canary" not in l.lower())
                if not body.strip():
                    out.append(ev(ins, 1))
                    files_bad += 1
                    msgs.insert(typos, f"The file {ins[len(pre):]} is empty, so the agent gets no task")
            # the verifier script (a separate verifier with its own environment may bundle tests in its image)
            sv = steps[names.index(name)].get("verifier") if name and isinstance(steps[names.index(name)].get("verifier"), dict) else {}
            if isinstance(sv.get("environment"), dict) or isinstance(ver.get("environment"), dict):
                continue
            ext = ".bat" if windows else ".sh"
            cands = [pre + f"steps/{name}/tests/test{ext}"] if name else []
            cands.append(pre + f"tests/test{ext}")
            if not any(c in pkg.sizes for c in cands):
                if any(p.startswith(pre + "tests/") or p.startswith(pre + "steps/") for p in pkg.sizes):
                    out.append(ev(tk.toml, 1))
                    files_bad += 1
                    msgs.insert(typos, f"The task has no {cands[0][len(pre):]}, so the verifier has nothing to run")
    if not out:
        return []
    msg = msgs[0]
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    # a typo of a known key or a missing file is a defect, a stale key from an older format only hygiene
    kind = "hack" if risky else ("broken" if (typos or files_bad) else "hygiene")
    return [F("task-toml-unknown-keys", msg + ".", out, confidence="review", kind=kind,
              severity="high" if risky else ("medium" if kind == "broken" else "low"))]


# ---------------------------------------------------------------------------
# 4. oracle-solution-echoes-answer
# canon: oracle-solution-echoes-answer, reader items (trustworthy-env T-8, get-bitcoin-nodes)
# ---------------------------------------------------------------------------

WRITE_ONLY = re.compile(r"^(echo|printf|cat|mkdir|cd|set|export|true|:|tee|exit|chmod|touch|sleep|cp)\b")
CODE_EXT = re.compile(r"\.(py|c|cc|cpp|h|hpp|js|ts|jsx|tsx|rs|go|java|kt|rb|pl|php|sh|bash|lua|hs|ml|scala|swift|r|jl|sql|"
                      r"cs|ex|exs|clj|dart|zig|nim|v|sv|vhd|asm|s|f90|tex|lean|coq|html|css|ya?ml|toml|ipynb|mk|cmake|vim|scm|ss|rkt|lisp|"
                      r"el|awk|sed|bat|ps1|m|mm|f|for|pas|ada|adb|cob|erl|fs|fsx|groovy|gd|glsl|hlsl|cu|cuh|wat|wasm|sol|"
                      r"dockerfile|patch|diff)$|(?:^|/)(Makefile|Dockerfile|CMakeLists\.txt)$", re.I)


def _literal_values(solve: str):
    vals = []
    for body in _heredocs(solve).values():
        vals += [l.strip() for l in body.splitlines() if l.strip()]
    for m in re.finditer(r"\b(?:echo|printf)\s+(?:-\w+\s+)*(['\"])(.*?)\1\s*>", solve):
        vals += [x.strip() for x in m.group(2).replace("\\n", "\n").splitlines() if x.strip()]
    for m in re.finditer(r"\becho\s+(?:-\w+\s+)*([^'\"\s>][^>\n]*?)\s*>", solve):
        vals.append(m.group(1).strip())
    return [v for v in vals if len(v) >= 3]


@rule("oracle-solution-echoes-answer", "hygiene", "low",
      "solution/solve.sh only writes literal answers that the tests check for, so the oracle run does not show the task can be solved.")
def r_oracle_echo(pkg: Package):
    out = []
    for tk in _tasks(pkg):
        pre = _pre(tk)
        sp = pre + "solution/solve.sh"
        solve = pkg.files.get(sp)
        if not solve:
            continue
        rows = sh_lines(solve)
        if not rows or PLACEHOLDER_LINE.search(solve):
            continue
        if not all(WRITE_ONLY.match(s.strip()) for _, s in rows):
            continue
        targets = []
        for _, s in rows:
            for m in re.finditer(r">>?\s*['\"]?([^\s'\";|&]+)", s):
                if not m.group(1).startswith(("/dev/", "&")):
                    targets.append(m.group(1))
            mt = re.search(r"\btee\s+(?:-a\s+)?['\"]?([^\s'\"]+)", s)
            if mt:
                targets.append(mt.group(1))
        if not targets or any(CODE_EXT.search(t) for t in targets):
            continue
        tests_text = "\n".join(t for p, t in _task_files(pkg, tk, "tests"))
        if not tests_text:
            continue
        vals = _literal_values(solve)
        hit = {v for v in vals if v in tests_text}
        if len(hit) >= 3:
            out.append(ev(sp, rows[0][0]))
    if not out:
        return []
    return [F("oracle-solution-echoes-answer", "The oracle solve.sh writes literal answers that also appear in the tests and runs no program"
              + (f" ({len(out)} tasks)" if len(out) > 1 else "") + ", so a passing oracle run proves only that the tests accept that text.",
              out, confidence="review")]


# ---------------------------------------------------------------------------
# 5. agent-runs-as-root
# canon: agent-runs-as-root, reader items (benchjack: explicit root HIGH, no USER MEDIUM)
# ---------------------------------------------------------------------------


def _compose_files(pkg: Package, tk):
    pre = _pre(tk) + "environment/"
    return [(p, t) for p, t in pkg.files.items() if p.startswith(pre) and
            re.fullmatch(r"(docker-)?compose(\.[\w-]+)?\.ya?ml", os.path.basename(p))]


def _dockerfile_user(instrs):
    users = [(ln, v.split()[0]) for ln, ins, v in instrs if ins == "USER" and v.split()]
    return users[-1] if users else None


def _step_users(cfg):
    agent = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
    top = agent.get("user")
    steps = cfg.get("steps") if isinstance(cfg.get("steps"), list) else []
    users = []
    for s in steps:
        a = s.get("agent") if isinstance(s, dict) and isinstance(s.get("agent"), dict) else {}
        users.append(a.get("user", top) if a.get("user") is not None else top)
    return users or [top]


@rule("agent-runs-as-root", "hack", "medium",
      "A Harbor task sets the agent user to root and grades in the same container, so the agent can replace the tools and files the verifier uses.")
def r_agent_root(pkg: Package):
    idx = None
    out, explicit = [], 0
    n = implicit = 0
    for tk in _tasks(pkg):
        if not tk.shared:
            continue
        cfg = tk.cfg
        if cfg:
            users = _step_users(cfg)
            root_users = [u for u in users if u is None or str(u).split(":")[0] in ("0", "root")]
            if not root_users:
                continue
            exp = any(u is not None for u in root_users)
        else:
            if not tk.agent_root:
                continue
            exp = bool(re.search(r"^\s*user\s*=\s*['\"]?(root|0)\b", pkg.files.get(tk.toml, ""), re.M))
        line = None
        if not exp:
            # no agent user in task.toml: Harbor uses the image's USER
            if tk.dockerfile:
                if idx is None:
                    idx = index(pkg)
                du = _dockerfile_user(idx.instrs.get(tk.dockerfile, []))
                if du and du[1].split(":")[0] not in ("root", "0"):
                    continue
                if du:
                    exp, line = True, ev(tk.dockerfile, du[0])
            elif re.search(r"^\s*docker_image\s*=", pkg.files.get(tk.toml, ""), re.M):
                continue  # prebuilt image: its USER is not visible here
            for p, t in _compose_files(pkg, tk):
                m = re.search(r"^\s{2,}user:\s*['\"]?(root|0)(:\S*)?['\"]?\s*$", t, re.M)
                if m:
                    exp, line = True, ev(p, t.count("\n", 0, m.start()) + 1)
        if not exp:
            implicit += 1
            continue  # no user anywhere: Harbor's default, true of nearly every task (see notes), not reported
        n += 1
        explicit += exp
        out.append(line or ev(tk.toml, _key_line(pkg.files.get(tk.toml, ""), "user") if exp else 1))
    if not out:
        return []
    msg = (f"{n} Harbor task{'s' if n > 1 else ''} with a shared verifier set{'' if n > 1 else 's'} the agent user to root"
           + ", so the agent can replace python, pytest or other files the verifier runs.")
    return [F("agent-runs-as-root", msg, out, confidence="review")]


# ---------------------------------------------------------------------------
# 6. secrets-in-agent-env
# canon: secrets-in-agent-env, reader items + reader items
# ---------------------------------------------------------------------------

SECRET_NAME = re.compile(r"^(?:\w*_)?(API_?KEY|APIKEY|ACCESS_KEY(_ID)?|SECRET(_ACCESS)?(_KEY)?|AUTH_TOKEN|ACCESS_TOKEN|TOKEN|"
                         r"PASSWORD|PASSWD|CREDENTIALS?)$|^(OPENAI|ANTHROPIC|GEMINI|GOOGLE|HF|HUGGING_?FACE|GROQ|TOGETHER|"
                         r"MISTRAL|COHERE|XAI|DEEPSEEK|OPENROUTER|AWS|AZURE|WANDB|GITHUB|GH|JUDGE)_\w*(KEY|TOKEN|SECRET)\w*$", re.I)
DUMMY = re.compile(r"^(|dummy\w*|test\w*|fake\w*|placeholder|x+|changeme|none|null|empty|sk-(test|dummy|fake|x+)\w*|your[\w-]*|"
                   r"<[^>]*>|\*+|-|EMPTY|0|false|true|password|secret|admin|root|postgres)$", re.I)


def _secret_value(val: str):
    v = val.strip().strip("\"'")
    if DUMMY.match(v):
        return None
    if re.search(r"\$\{?\w*(KEY|TOKEN|SECRET|PASS)\w*", v, re.I):
        return "host"      # passes the host's secret into the container
    if v.startswith("$") or "{" in v:
        return None        # unrelated template variable
    if re.fullmatch(r"[A-Za-z0-9_\-.:/+=]{16,}", v) and re.search(r"\d", v) and re.search(r"[A-Za-z]", v):
        return "literal"
    return None


def _env_pairs_docker(value: str):
    v = value.strip()
    if "=" in v.split()[0] if v.split() else False:
        for m in re.finditer(r"(\w+)=(\"[^\"]*\"|'[^']*'|\S*)", v):
            yield m.group(1), m.group(2)
    else:
        parts = v.split(None, 1)
        if len(parts) == 2:
            yield parts[0], parts[1]


def _compose_env(text: str):
    """(line, service, name, value) for compose `environment:` entries."""
    out, svc, svc_ind, in_env, env_ind = [], None, None, False, None
    in_services = False
    for i, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        ind = len(raw) - len(raw.lstrip())
        s = raw.strip()
        if ind == 0:
            in_services = s.startswith("services:")
            svc, in_env = None, False
            continue
        if not in_services:
            continue
        if svc_ind is None or ind <= svc_ind:
            if s.endswith(":") and re.match(r"[\w.-]+:$", s):
                svc, svc_ind, in_env = s[:-1], ind, False
            continue
        if re.match(r"environment:\s*$", s):
            in_env, env_ind = True, ind
            continue
        if in_env and ind <= env_ind:
            in_env = False
        if in_env:
            m = re.match(r"-\s*['\"]?(\w+)=(.*?)['\"]?$", s) or re.match(r"['\"]?(\w+)['\"]?\s*:\s*(.*)$", s)
            if m:
                out.append((i, svc, m.group(1), m.group(2)))
    return out


SANDBOX_ENV_CALL = re.compile(r"(CreateSandboxRequest|create_sandbox|Sandbox\.create|sandboxes\.create|containers\.run|"
                              r"docker_client\.\w+\.run|AsyncSandbox|modal\.Sandbox)")


@rule("secrets-in-agent-env", "hygiene", "medium",
      "An API key, token or password is put in the environment of the container where the agent runs.")
def r_secrets(pkg: Package):
    out, names = [], []
    tasks = _tasks(pkg)
    for tk in tasks:
        cfg = tk.cfg
        envc = cfg.get("environment") if isinstance(cfg.get("environment"), dict) else {}
        envd = envc.get("env") if isinstance(envc.get("env"), dict) else {}
        text = pkg.files.get(tk.toml, "")
        for k, v in envd.items():
            if SECRET_NAME.search(k) and isinstance(v, str) and _secret_value(v):
                out.append(ev(tk.toml, _key_line(text, k)))
                names.append(k)
        for p, t in _compose_files(pkg, tk):
            for ln, svc, k, v in _compose_env(t):
                if svc == "main" and SECRET_NAME.search(k) and _secret_value(v):
                    out.append(ev(p, ln))
                    names.append(k)
    # agent images (Harbor agent Dockerfiles and the sandbox images of other envs)
    if any(is_dockerfile(p) for p in pkg.files):
        idx = index(pkg)
        for p, role in idx.dockerfiles.items():
            if role != "agent":
                continue
            for ln, ins, val in idx.instrs.get(p, []):
                if ins != "ENV":
                    continue
                for k, v in _env_pairs_docker(val):
                    if SECRET_NAME.search(k) and _secret_value(v):
                        out.append(ev(p, ln))
                        names.append(k)
    # verifiers envs that hand a host secret to the sandbox they create for the policy
    for path, text in pkg.files.items():
        if not path.endswith(".py") or not SANDBOX_ENV_CALL.search(text):
            continue
        for m in re.finditer(r"\b(?:env|environment|env_vars|environment_vars|environment_variables)\s*=\s*\{([^{}]{0,800})\}", text):
            ctx = text[max(0, m.start() - 1500): m.end() + 200]
            if not SANDBOX_ENV_CALL.search(ctx):
                continue
            for k in re.finditer(r"['\"](\w+)['\"]\s*:\s*([^,}]+)", m.group(1)):
                if SECRET_NAME.search(k.group(1)) and re.search(r"(os\.environ|getenv|api_key|token|secret)", k.group(2), re.I):
                    out.append(ev(path, text.count("\n", 0, m.start(1) + k.start()) + 1))
                    names.append(k.group(1))
    if not out:
        return []
    uniq = sorted(set(names))
    return [F("secrets-in-agent-env", f"The agent's container gets {', '.join(uniq[:4])}"
              + (f" and {len(uniq) - 4} more" if len(uniq) > 4 else "")
              + " in its environment, so the agent can read the key and use it or send it out.", out, confidence="review")]


# ---------------------------------------------------------------------------
# 7. compose-writable-bind-mounts
# canon: compose-writable-bind-mounts, reader items (benchjack: read-write volume MEDIUM)
# ---------------------------------------------------------------------------

HARBOR_MOUNT = re.compile(r"HOST_(VERIFIER_LOGS|AGENT_LOGS|ARTIFACTS|LOGS)\w*|ENV_\w*LOGS")
SENSITIVE_NAME = re.compile(r"(?:^|[/_.-])(answers?|gold|golden|expected|solutions?|reference|ground[_-]?truth|tests?|grader|"
                            r"verifier|results?|scores?|grades?|labels?|data)(?:$|[/_.-])", re.I)


def _compose_volumes(text: str):
    """(line, service, source, target, read_only) for service volume entries (short and long syntax)."""
    out, svc, svc_ind, in_vol, vol_ind, long = [], None, None, False, None, None
    in_services = False
    for i, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        ind = len(raw) - len(raw.lstrip())
        s = raw.strip()
        if ind == 0:
            if long:
                out.append(long)
                long = None
            in_services, svc, in_vol = s.startswith("services:"), None, False
            continue
        if not in_services:
            continue
        if svc_ind is None or ind <= svc_ind:
            if long:
                out.append(long)
                long = None
            if re.match(r"[\w.-]+:$", s):
                svc, svc_ind, in_vol = s[:-1], ind, False
            continue
        if re.match(r"volumes:\s*$", s):
            in_vol, vol_ind = True, ind
            continue
        if in_vol and ind <= vol_ind:
            in_vol = False
            if long:
                out.append(long)
                long = None
        if not in_vol:
            continue
        if s.startswith("-"):
            if long:
                out.append(long)
                long = None
            item = s[1:].strip().strip("\"'")
            if re.match(r"\w+\s*:\s", item) and not re.match(r"[./~$]", item):
                long = [i, svc, "", "", False, "volume"]
                k, _, v = item.partition(":")
                s = item
            else:
                ph = re.sub(r"\$\{[^}]*\}", lambda m: m.group(0).replace(":", "\x00"), item)
                parts = [x.replace("\x00", ":") for x in ph.split(":")]
                if len(parts) >= 2:
                    src, dst = parts[0], parts[1]
                    mode = parts[2] if len(parts) > 2 else ""
                    out.append([i, svc, src, dst, "ro" in mode.split(","), "bind" if re.match(r"[./~$]", src) else "volume"])
                continue
        if long is not None:
            m = re.match(r"(type|source|target|read_only)\s*:\s*['\"]?([^'\"]*)['\"]?$", s)
            if m:
                k, v = m.group(1), m.group(2).strip()
                if k == "type":
                    long[5] = v
                elif k == "source":
                    long[2] = v
                elif k == "target":
                    long[3] = v
                elif k == "read_only":
                    long[4] = v.lower() in ("true", "yes", "1")
    if long:
        out.append(long)
    return [tuple(x) for x in out]


@rule("compose-writable-bind-mounts", "hack", "medium",
      "A Harbor compose file mounts a host folder into the agent's container without read only, so the agent can change host files that the grader or later trials read.")
def r_compose_mounts(pkg: Package):
    out, details = [], []
    for tk in _tasks(pkg):
        for p, t in _compose_files(pkg, tk):
            if "volumes" not in t:
                continue
            vols = _compose_volumes(t)
            main_src = {v[2] for v in vols if v[1] == "main"}
            for ln, svc, src, dst, ro, typ in vols:
                if ro or typ != "bind" or not src or HARBOR_MOUNT.search(src + dst):
                    continue
                if svc != "main" and src not in main_src:
                    continue
                if src in ("/var/run/docker.sock", "/run/docker.sock") or "docker.sock" in src:
                    continue  # docker socket: the image rules own it
                outside = src.startswith(("/", "~")) or ".." in src.split("/") or (src.startswith("$") and not re.match(r"\$\{?CONTEXT_DIR\}?/?[^.]", src))
                named = SENSITIVE_NAME.search(src) or SENSITIVE_NAME.search(dst)
                if not (outside or named):
                    continue
                out.append(ev(p, ln))
                details.append(f"{src} is mounted at {dst} read write" + (" from outside the task folder" if outside else ""))
    if not out:
        return []
    msg = "In the task compose file " + details[0]
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    return [F("compose-writable-bind-mounts", msg + ", so the agent can change host files that the grader or later trials read.",
              out, confidence="review")]


# ---------------------------------------------------------------------------
# 8. separate-verifier-reads-undeclared-path
# canon: separate-verifier-reads-undeclared-path, reader items
# Harbor (trial.py _run_separate_verifier) starts a fresh verifier container and uploads only the
# declared artifacts (plus /logs/artifacts). Files the agent wrote anywhere else are not there.
# ---------------------------------------------------------------------------

AGENT_PATH = re.compile(r"(?<![\w$./-])(/(?:app|home/[\w.-]+|workspace|root|output|outputs|submission|srv|data|repo|testbed|tmp|logs/agent)"
                        r"(?:/[\w.@+-]+)*)")


def _artifact_sources(cfg, step=None):
    out = []
    lists = [cfg.get("artifacts")]
    if isinstance(cfg.get("steps"), list):
        for s in cfg["steps"]:
            if isinstance(s, dict) and (step is None or s.get("name") == step):
                lists.append(s.get("artifacts"))
    for lst in lists:
        if isinstance(lst, list):
            for a in lst:
                if isinstance(a, str):
                    out.append(a.rstrip("/"))
                elif isinstance(a, dict) and isinstance(a.get("source"), str):
                    out.append(a["source"].rstrip("/"))
    return out


def _separate_everywhere(cfg) -> bool:
    def mode(v):
        if not isinstance(v, dict):
            return None
        m = v.get("environment_mode")
        if isinstance(m, str):
            return m.lower()
        return "separate" if isinstance(v.get("environment"), dict) else None
    top = mode(cfg.get("verifier")) or "shared"
    steps = cfg.get("steps") if isinstance(cfg.get("steps"), list) else []
    if steps:
        return [(mode(s.get("verifier")) if isinstance(s, dict) else None) or top for s in steps]
    return [top]


@rule("separate-verifier-reads-undeclared-path", "broken", "high",
      "A separate Harbor verifier reads a file the agent writes, but task.toml does not declare it as an artifact, so it is missing at grading time.")
def r_separate_undeclared(pkg: Package):
    out, details = [], []
    for tk in _tasks(pkg):
        cfg = tk.cfg
        if not cfg or tk.shared:
            continue
        modes = _separate_everywhere(cfg)
        if "separate" not in modes:
            continue
        pre = _pre(tk)
        arts = _artifact_sources(cfg)
        instr = "\n".join(t for p, t in pkg.files.items() if p.startswith(pre) and os.path.basename(p) == "instruction.md")
        image_text = "\n".join(t for p, t in pkg.files.items() if p.startswith(pre + "environment/") or
                               (p.startswith(pre) and is_dockerfile(p)))
        files = [(p, t) for p, t in pkg.files.items() if p.startswith(pre) and
                 re.search(r"(^|/)tests/", p[len(pre):]) and not os.path.basename(p).startswith("Dockerfile")]
        seen = set()
        for p, t in files:
            for m in AGENT_PATH.finditer(t):
                path = m.group(1).rstrip(".")
                if path in seen or len(path) < 5:
                    continue
                if any(path == a or path.startswith(a + "/") or a.startswith(path + "/") for a in arts):
                    continue
                ln = t.count("\n", 0, m.start()) + 1
                line = t.splitlines()[ln - 1] if t.splitlines() else ""
                if re.search(r"(mkdir|>\s*$|\bcd\b|\bwrite|open\([^)]*['\"]w)", line) and path not in instr:
                    continue
                nxt = "\n".join(t.splitlines()[ln - 1:ln + 2])
                if re.search(rf"(\[\[?\s+-[efsd]\s+['\"]?{re.escape(path)}|assert\s+not\s+[\w.]*exists\(\s*['\"]{re.escape(path)})", line) \
                        and "!" not in line.split(path)[0] and re.search(r"(fail|exit\s+[1-9]|echo\s+0|assert\s+not)", nxt):
                    continue  # the test checks that the file is absent
                if path.startswith("/logs/agent"):
                    if re.search(r"/logs/agent/?$", path):
                        continue
                    seen.add(path)
                    out.append(ev(p, ln))
                    details.append(f"the tests read {path}, but /logs/agent is not copied to a separate verifier unless it is an artifact")
                    continue
                if path.count("/") < 2 or path in image_text or os.path.basename(path) in image_text:
                    continue  # baked into the image, so the fresh verifier copy has it
                if path not in instr:
                    continue  # only flag files the instruction tells the agent to produce
                seen.add(path)
                out.append(ev(p, ln))
                details.append(f"the tests read {path}, which the instruction asks the agent to write, but it is not a declared artifact")
    if not out:
        return []
    msg = "In separate verifier mode " + details[0]
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    return [F("separate-verifier-reads-undeclared-path", msg + ", so correct work fails.", out, confidence="review")]


# ---------------------------------------------------------------------------
# 9. resource-limit-equals-request: dropped (see the internal notes). task.toml has a single
# cpus / memory_mb by schema, and whether it becomes a request, a limit or both is set by the
# environment provider at run time (docker: mem_limit only, gke: memory_limit_multiplier).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 10. verifier-runs-submission-privileged
# canon: verifier-runs-submission-privileged, reader items (Harbor RFC 0003)
# ---------------------------------------------------------------------------

DROP = re.compile(r"\b(runuser|setpriv|su\s+(?:-\w+\s+)*-c|su\s+-s|sudo\s+-u|gosu|su-exec|chroot\s+--userspec|capsh\s+--user)\b")
BUILD = re.compile(r"\b(pip3?\s+install\s+(?:-\w+\s+)*(?:-r\s+(?!/tests)\S+|-e\s+\S+|\.(?:\s|$|\[))|uv\s+pip\s+install\s+(?:-r\s+(?!/tests)\S+|-e\s+\S+|\.(?:\s|$))|"
                   r"uv\s+sync\b|poetry\s+install\b|npm\s+(?:ci|install|i)\b(?![^\n]*--ignore-scripts)|yarn(?:\s+install)?\s*$|pnpm\s+install\b|"
                   r"make\b(?!\s+-C\s+/tests)|cmake\s+--build|cargo\s+(?:build|run|test)\b|bundle\s+install\b|composer\s+install\b|"
                   r"python3?\s+setup\.py\b|\./configure\b)")


@rule("verifier-runs-submission-privileged", "hack", "high",
      "The verifier drops privileges for some steps but builds or installs the agent's files as root in others, so a build hook runs with grader rights.")
def r_submission_privileged(pkg: Package):
    out = []
    for tk in _tasks(pkg):
        for sp in _all_scripts(tk):
            text = pkg.files.get(sp, "")
            if not text or not DROP.search(text):
                continue
            rows = sh_lines(text)
            dropped = [ln for ln, s in rows if DROP.search(s)]
            if not dropped:
                continue
            reward = [ln for ln, s in rows if REWARD_ANY.search(s)]
            last = max(reward) if reward else 10 ** 9
            for ln, s in rows:
                if ln >= last or DROP.search(s):
                    continue
                m = BUILD.search(s)
                if m and not re.search(r"(/tests\b|/opt/verifier|/verifier\b|--no-deps\s+\S*pytest|pytest[=<>])", s):
                    out.append(ev(sp, ln))
    if not out:
        return []
    return [F("verifier-runs-submission-privileged", "The verifier script runs some steps as a lower user but builds or installs the agent's "
              "files as root" + (f" ({len(out)} places)" if len(out) > 1 else "")
              + ", so an install or build hook the agent planted can change the tests or the reward.", out, confidence="review")]


# ---------------------------------------------------------------------------
# 11. root-ops-on-agent-writable-paths
# canon: root-ops-on-agent-writable-paths, reader items (TB #2086: absolute symlink in an artifact dir read /tests/secret.txt)
# ---------------------------------------------------------------------------

ROOT_OP = re.compile(r"\b(chmod|chown|chgrp)\s+(?P<flags>(?:-\w+\s+)*)(?P<rest>[^;&|\n]+)")
AGENT_DIR = re.compile(r"^['\"]?(/app|/workspace|/testbed|/home/[\w.-]+|/logs/artifacts|/submission\w*|/output\w*|\$\{?(?:APP|WORK)\w*\}?)(/|$|['\"])")


@rule("root-ops-on-agent-writable-paths", "hack", "medium",
      "A shared mode verifier runs chmod or chown as root on a path the agent can write, without -h, so a planted symlink changes a file the agent should not touch.")
def r_root_ops(pkg: Package):
    out = []
    for tk in _tasks(pkg):
        if not tk.shared:
            continue
        cfg = tk.cfg
        ver = cfg.get("verifier") if isinstance(cfg.get("verifier"), dict) else {}
        if ver.get("user") is not None and str(ver.get("user")).split(":")[0] not in ("0", "root"):
            continue
        for sp in _all_scripts(tk):
            text = pkg.files.get(sp, "")
            if not re.search(r"\bch(mod|own|grp)\b", text):
                continue
            tests_after = False
            for ln, s in reversed(sh_lines(text)):
                if EVAL_CMD.search(s):
                    tests_after = True
                m = ROOT_OP.search(s)
                if not m or not tests_after or re.search(r"-\w*h", m.group("flags")) or "--no-dereference" in s:
                    continue
                args = [a for a in m.group("rest").split() if not a.startswith("-")]
                targets = args[1:] if m.group(1) != "chmod" or len(args) > 1 else args
                if m.group(1) == "chmod":
                    targets = args[1:]
                if any(AGENT_DIR.match(a) and not re.search(r"\.(sh|py)['\"]?$", a) for a in targets):
                    if re.search(r"\b(777|666|[ao]\+r|o\+\w*r|go\+\w*r|a\+\w*r)\b", s) or m.group(1) != "chmod":
                        out.append(ev(sp, ln))
    if not out:
        return []
    return [F("root-ops-on-agent-writable-paths", "The verifier script runs chmod or chown as root on a path the agent controls before the tests, "
              "with no -h and no symlink check, so a planted link opens up a test or reward file.", out, confidence="review")]

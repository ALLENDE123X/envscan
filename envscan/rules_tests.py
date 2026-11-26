"""v0.2 rules, family "tests": how a grader starts the test run and how it reads the verdict.

Rules (row numbers refer to section 9.2 of the literature review):
  grader-import-surface        #1   pytest or python is started so files in the agent's tree load into the grader
  verdict-parse-semantics      #6   the pass verdict is read so that a skipped test, an empty run, printed text
                                    or an exit status the submission controls counts as a pass
  existence-or-constant-check  #13  the reward or the graded tests only check that something exists or is non empty
  equality-on-model-objects    #22  graded tests compare objects the submission returned with == and no type check

All checks are static. They read shell scripts (Harbor tests/test.sh and scripts shipped inside verifiers
packages), Python test files under a graded tests/ folder, embedded heredoc Python, and verifiers reward code.
Nothing is imported or run.
"""
from __future__ import annotations

import ast
import os
import re
import shlex

from .package import Package, PyFile, parse_py
from .rules import RULES, F, ev, line_of, rule, _uses_answer

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

# When True, grader-import-surface drops hits that an existing shared-box rule already reports for the same
# package (harbor-shared-grading for shared mode Harbor tasks, grader-in-agent-sandbox / keep-sandbox-for-scoring
# for verifiers code, partial-test-restore and model-named-files-into-test-dir for their own files).
SUPPRESS_SHARED_BOX_OVERLAP = True

_CACHE = "_rules_tests_cache"

# Directories an agent works in (Harbor WORKDIRs, SWE style repos). A task's own Dockerfile WORKDIR is added.
AGENT_DIRS = ("/app", "/testbed", "/workspace", "/repo", "/code", "/project", "/home/agent", "/home/user",
              "/submission", "/work", "/workdir")
GRADED_SH = {"test.sh", "run_tests.sh", "run-tests.sh", "run_test.sh", "eval.sh", "evaluate.sh", "grade.sh",
             "verify.sh", "check.sh", "run_eval.sh", "score.sh"}

PASS_TOKENS = {"passed", "pass", "xpass", "xpassed", "ok", "success"}
SOFT_TOKENS = {"skipped", "skip", "xfail", "xfailed"}
FAIL_TOKENS = {"failed", "fail", "error", "errored", "errors", "failure"}


def _cache(pkg: Package) -> dict:
    c = pkg.__dict__.get(_CACHE)
    if c is None:
        c = {}
        setattr(pkg, _CACHE, c)
    return c


# ---------------------------------------------------------------------------
# Harbor task layout
# ---------------------------------------------------------------------------

class _Task:
    __slots__ = ("root", "separate", "artifacts", "workdir", "toml_path", "nonroot_agent")

    def __init__(self, root, separate, artifacts, workdir, toml_path, nonroot_agent=False):
        self.root, self.separate, self.artifacts, self.workdir, self.toml_path = root, separate, artifacts, workdir, toml_path
        self.nonroot_agent = nonroot_agent

    def pkg_path(self, container_path: str) -> str:
        """/tests/x.py -> <root>/tests/x.py (Harbor uploads tests/ to /tests and solution/ to /solution)."""
        for d in ("tests", "solution"):
            if container_path == f"/{d}" or container_path.startswith(f"/{d}/"):
                rel = container_path[1:]
                return f"{self.root}/{rel}" if self.root else rel
        return ""

    def agent_dirs(self):
        if self.separate:
            return tuple(a.rstrip("/") for a in self.artifacts if a.startswith("/")) + ("/logs/artifacts",)
        dirs = AGENT_DIRS
        if self.workdir and self.workdir.startswith("/") and self.workdir not in ("/", "/tests", "/logs"):
            dirs = dirs + (self.workdir.rstrip("/"),)
        return dirs


def _tasks(pkg: Package) -> dict:
    c = _cache(pkg)
    if "tasks" in c:
        return c["tasks"]
    out = {}
    for p, t in pkg.files.items():
        if os.path.basename(p) != "task.toml":
            continue
        root = os.path.dirname(p)
        data = {}
        if tomllib is not None:
            try:
                data = tomllib.loads(t)
            except Exception:
                data = {}
        ver = data.get("verifier") if isinstance(data.get("verifier"), dict) else {}
        separate = str(ver.get("environment_mode", "")).lower() == "separate" or \
            bool(re.search(r"environment_mode\s*=\s*['\"]separate", t))
        arts = data.get("artifacts") or []
        arts = [a if isinstance(a, str) else str(a.get("source", a.get("path", ""))) if isinstance(a, dict) else ""
                for a in arts] if isinstance(arts, list) else []
        env = data.get("environment") if isinstance(data.get("environment"), dict) else {}
        workdir = env.get("workdir") if isinstance(env.get("workdir"), str) else None
        if not workdir:
            for dp in (f"{root}/environment/Dockerfile" if root else "environment/Dockerfile",):
                df = pkg.files.get(dp)
                if df:
                    ws = re.findall(r"^\s*WORKDIR\s+(\S+)", df, re.M | re.I)
                    if ws:
                        workdir = ws[-1].strip("'\"")
        agent = data.get("agent") if isinstance(data.get("agent"), dict) else {}
        user = str(agent.get("user", "") or "")
        out[root] = _Task(root, separate, [a for a in arts if a], workdir, p, bool(user) and user not in ("root", "0"))
    for p in pkg.files:
        if p.endswith("tests/test.sh") or p == "tests/test.sh":
            root = os.path.dirname(os.path.dirname(p))
            if root not in out and not _vendor_like(p):
                df = pkg.files.get(f"{root}/environment/Dockerfile" if root else "environment/Dockerfile", "")
                ws = re.findall(r"^\s*WORKDIR\s+(\S+)", df, re.M | re.I)
                out[root] = _Task(root, False, [], ws[-1].strip("'\"") if ws else None, None)
    c["tasks"] = out
    return out


def _task_of(pkg: Package, path: str):
    best = None
    for root, task in _tasks(pkg).items():
        if root == "" or path == root or path.startswith(root + "/"):
            if best is None or len(root) > len(best.root):
                best = task
    return best


def _under(path: str | None, dirs) -> bool:
    if not path or not path.startswith("/"):
        return False
    path = os.path.normpath(path)
    return any(path == d or path.startswith(d.rstrip("/") + "/") for d in dirs if d)


def _graded_sh(pkg: Package):
    """Shell scripts that grade: Harbor tests/test.sh and step test.sh, plus scripts with grading names or that
    write a reward file."""
    c = _cache(pkg)
    if "graded_sh" in c:
        return c["graded_sh"]
    out = []
    for p, t in pkg.files.items():
        base = os.path.basename(p)
        if not (p.endswith(".sh") or base in ("test", "run_tests")):
            continue
        if base in GRADED_SH or "/tests/" in "/" + p or "reward.txt" in t or "reward.json" in t:
            if _vendor_like(p):
                continue
            out.append((p, t))
    c["graded_sh"] = out
    return out


def _vendor_like(p: str) -> bool:
    return bool(re.search(r"(^|/)(node_modules|third_party|vendor|site-packages|\.git|git-src|external)/", p))


def _graded_test_files(pkg: Package):
    """(path, text) of Python files in a graded tests/ folder: next to a test.sh or inside a Harbor task's tests/."""
    c = _cache(pkg)
    if "graded_files" in c:
        return c["graded_files"]
    sh_dirs = {os.path.dirname(p) for p, _ in _graded_sh(pkg) if os.path.basename(p) in GRADED_SH}
    task_tests = {(f"{r}/tests" if r else "tests") for r in _tasks(pkg)}
    out = []
    for p, t in pkg.files.items():
        if not p.endswith(".py") or _vendor_like(p):
            continue
        d = os.path.dirname(p)
        if d in sh_dirs or any(d == tt or d.startswith(tt + "/") for tt in task_tests):
            out.append((p, t))
    c["graded_files"] = out
    return out


def _pyf(pkg: Package, path: str) -> PyFile:
    """Parsed PyFile for any package file, cached (pkg.py skips test paths, so graded tests are parsed here)."""
    c = _cache(pkg).setdefault("pyparse", {})
    pf = c.get(path)
    if pf is None:
        pf = pkg.py.get(path) or parse_py(path, pkg.files.get(path, ""))
        c[path] = pf
    return pf


# ---------------------------------------------------------------------------
# a small shell reader: logical lines, heredocs, cd / variable / PYTHONPATH tracking
# ---------------------------------------------------------------------------

_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
_OPS = {"&&", "||", ";", "|", "&", "(", ")", ";;", "|&"}
_SH_KEYWORDS = {"if", "then", "elif", "else", "do", "while", "until", "!", "time", "{", "}"}
_REDIR = {">", ">>", "<", ">&", "&>", "<<", "<<<", ">|", "&>>"}


class _Cmd:
    __slots__ = ("line", "argv", "cwd", "env", "exported", "out", "after_or", "raw", "piped_from")

    def __init__(self, line, argv, cwd, env, exported, out, after_or, raw, piped_from):
        self.line, self.argv, self.cwd, self.env, self.exported = line, argv, cwd, env, exported
        self.out, self.after_or, self.raw, self.piped_from = out, after_or, raw, piped_from


_FAST_TOK = re.compile(r"&&|\|\||;;|\|&|>>|>&|&>|[;&|()<>]|[^\s;&|()<>]+")


def _tokens(s: str):
    if not any(ch in s for ch in "'\"\\`#"):
        return _FAST_TOK.findall(s)
    try:
        lx = shlex.shlex(s, posix=True, punctuation_chars=True)
        lx.whitespace_split = True
        lx.commenters = "#"
        return list(lx)
    except ValueError:
        return s.split()


def _subst(tok: str, vars_: dict) -> str:
    def rep(m):
        name = m.group(1) or m.group(3)
        if name in vars_ and "${" not in vars_[name]:
            return vars_[name]
        if m.group(2) is not None:  # ${NAME:-default} with NAME unknown: the default applies
            return m.group(2)
        return m.group(0)
    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}|\$(\w+)", rep, tok)


def _sh_parse(text: str, start_cwd: str | None = None):
    """-> (commands, heredocs). heredocs: list of (header_line_no, header_text, body, body_first_line_no)."""
    lines = text.splitlines()
    cmds, docs = [], []
    cwd = start_cwd
    vars_: dict = {}
    exported: dict = {}
    i = 0
    n = len(lines)
    while i < n:
        start = i
        ln = lines[i]
        while ln.rstrip().endswith("\\") and i + 1 < n:
            i += 1
            ln = ln.rstrip()[:-1] + " " + lines[i]
        i += 1
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = _HEREDOC.search(ln) if "<<<" not in ln else None
        if m:
            delim = m.group(2)
            body_start = i
            body = []
            while i < n and lines[i].strip() != delim:
                body.append(lines[i])
                i += 1
            i += 1  # skip the delimiter
            docs.append((start + 1, ln, "\n".join(body), body_start + 1))
            ln = ln[:m.start()] + ln[m.end():]
        toks = _tokens(ln)
        simple: list = []
        after_or = False
        prev_op = None
        pipe_src = None
        groups = []
        for t in toks + [";"]:
            if t in _OPS:
                if simple:
                    groups.append((simple, after_or, prev_op))
                after_or = t == "||"
                prev_op = t
                simple = []
            else:
                simple.append(t)
        for argv, aor, pop in groups:
            env = {}
            while argv and re.match(r"^[A-Za-z_]\w*=", argv[0]):
                k, _, v = argv[0].partition("=")
                env[k] = _subst(v, vars_)
                argv = argv[1:]
            if argv and argv[0] in ("export", "declare", "local", "readonly"):
                for a in argv[1:]:
                    if "=" in a and re.match(r"^[A-Za-z_]\w*=", a):
                        k, _, v = a.partition("=")
                        v = _subst(v, vars_)
                        vars_[k] = v
                        exported[k] = v
                continue
            if not argv:
                for k, v in env.items():
                    vars_[k] = v
                continue
            # redirections
            clean, out = [], None
            j = 0
            while j < len(argv):
                a = argv[j]
                if a in _REDIR or re.match(r"^\d?(>|>>|<)$", a):
                    if j + 1 < len(argv) and a in (">", ">>", "&>", ">|", "&>>"):
                        out = _subst(argv[j + 1], vars_)
                    j += 2
                    continue
                if a.isdigit() and j + 1 < len(argv) and argv[j + 1] in (">", ">>", ">&", "<"):
                    j += 1
                    continue
                clean.append(_subst(a, vars_))
                j += 1
            while clean and clean[0] in _SH_KEYWORDS:
                clean = clean[1:]
            if not clean:
                continue
            if clean[0] in ("cd", "pushd"):
                if aor:
                    continue
                tgt = clean[1] if len(clean) > 1 else None
                if tgt is None or tgt.startswith("-"):
                    cwd = None
                elif "$" in tgt or "{" in tgt or "`" in tgt:
                    cwd = None
                elif tgt.startswith("/"):
                    cwd = os.path.normpath(tgt)
                elif cwd:
                    cwd = os.path.normpath(os.path.join(cwd, tgt))
                else:
                    cwd = None
                continue
            if clean[0] == "tee" and len(clean) > 1 and pop == "|":
                if cmds:
                    cmds[-1].out = cmds[-1].out or _subst(clean[-1], vars_)
                continue
            cmds.append(_Cmd(start + 1, clean, cwd, env, dict(exported), out, aor, s,
                             cmds[-1] if pop == "|" and cmds else None))
    return cmds, docs


def _sh_cached(pkg: Package, path: str, text: str, start_cwd=None):
    """Parsed script, cached per package. Harbor runs test.sh from the task's WORKDIR, so that is the start cwd."""
    c = _cache(pkg).setdefault("sh", {})
    key = path
    if key not in c:
        task = _task_of(pkg, path)
        start_cwd = task.workdir if task is not None and task.workdir and not task.separate else None
        try:
            c[key] = _sh_parse(text, start_cwd)
        except RecursionError:
            c[key] = ([], [])
    return c[key]


_PY_EXE = re.compile(r"^(?:.*/)?python(?:\d(?:\.\d+)?)?$")
_PYTEST_VAL_OPTS = {"-k", "-m", "-p", "-c", "-o", "-W", "-n", "-r", "--rootdir", "--confcutdir", "--ctrf", "--junitxml",
                    "--junit-xml", "--json-report-file", "--tb", "--timeout", "--durations", "--deselect", "--ignore",
                    "--ignore-glob", "--cov", "--cov-report", "--maxfail", "--log-file", "--basetemp", "--color",
                    "--import-mode", "--capture", "--html", "--report-log", "--reruns", "--override-ini", "--log-level",
                    "--dist", "--alluredir", "--count", "--log-cli-level", "--splits", "--group", "--json-report-indent",
                    "--benchmark-json", "--resultlog", "--cache-dir", "--inifile", "--config-file", "--log-format",
                    "--junit-prefix", "--max-worker-restart", "--timeout-method", "--randomly-seed"}
_WRAPPERS = {"timeout", "uv", "uvx", "run", "conda", "poetry", "pipenv", "exec", "env", "nice", "stdbuf", "xvfb-run",
             "time", "sudo", "su", "runuser", "bash", "sh"}


def _pytest_args(argv):
    """If argv runs pytest, return (args_after_pytest, via_python_m) else None."""
    if not argv or argv[0] in ("echo", "printf", "grep", "cat", "pip", "pip3", "apt-get", "which", "command", "type"):
        return None
    for i, t in enumerate(argv):
        base = os.path.basename(t)
        if base in ("pytest", "py.test"):
            if any(a in ("install", "add", "--with", "-w", "--from") for a in argv[max(0, i - 1):i]):
                continue
            if any(a in ("install", "add") for a in argv[:i]):
                return None
            return argv[i + 1:], False
        if t == "pytest" or (t == "-m" and i + 1 < len(argv) and argv[i + 1] == "pytest"):
            if t == "-m" and any(_PY_EXE.match(a) for a in argv[:i]):
                return argv[i + 2:], True
    return None


def _pytest_paths(args):
    paths, opts = [], set()
    j = 0
    while j < len(args):
        a = args[j]
        if a.startswith("--") and "=" in a:
            opts.add(a.split("=", 1)[0])
        elif a.startswith("-"):
            opts.add(a)
            if a in _PYTEST_VAL_OPTS:
                j += 1
        else:
            paths.append(a.split("::")[0])
        j += 1
    return paths, opts


_WRAP_VAL = {"-n", "--name", "-p", "--prefix", "-u", "--user", "-s", "--signal", "-k", "--kill-after", "--with",
             "-w", "--python", "--directory", "--project", "-e", "--env"}


def _python_launch(argv):
    """If argv starts a python interpreter, return (flags, mode, target) where mode is 'm', 'c', 'script' or 'stdin'.
    Leading wrappers (timeout, uv run, conda run -n env, env, sudo -u x) are skipped."""
    i = 0
    while i < len(argv):
        t = argv[i]
        if _PY_EXE.match(t):
            flags, j = set(), i + 1
            while j < len(argv) and argv[j].startswith("-") and argv[j] not in ("-m", "-c", "-"):
                flags |= set(argv[j][1:]) if not argv[j].startswith("--") else {argv[j]}
                j += 1
            if j >= len(argv) or argv[j] == "-":
                return flags, "stdin", None
            if argv[j] == "-m":
                return flags, "m", argv[j + 1] if j + 1 < len(argv) else None
            if argv[j] == "-c":
                return flags, "c", None
            return flags, "script", argv[j]
        base = os.path.basename(t)
        if i > 0 and argv[i - 1] == "run" and base.endswith(".py") and os.path.basename(argv[0]) in ("uv", "poetry", "pipenv"):
            return set(), "script", t
        if base in _WRAPPERS or re.match(r"^(-\S+|\d+(\.\d+)?[smhd]?|[A-Za-z_]\w*=.*)$", t):
            i += 1
            continue
        if i > 0 and argv[i - 1] in _WRAP_VAL:
            i += 1
            continue
        return None
    return None


# ---------------------------------------------------------------------------
# Python helpers for graded test files
# ---------------------------------------------------------------------------

_STR = (ast.Constant,)


def _str_of(n):
    return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None


def _status_token(n) -> str | None:
    """'PASSED' / TestStatus.PASSED.value / Status.SKIPPED -> lower token."""
    s = _str_of(n)
    if s is not None:
        return s.strip().lower()
    if isinstance(n, ast.Attribute):
        if n.attr == "value" and isinstance(n.value, ast.Attribute):
            return n.value.attr.lower()
        if n.attr.isupper():
            return n.attr.lower()
    return None


def _parents(tree):
    par = {}
    for p in ast.walk(tree):
        for ch in ast.iter_child_nodes(p):
            par[ch] = p
    return par


def _enclosing_func(node, par):
    cur = par.get(node)
    while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
        cur = par.get(cur)
    return cur


SUBMISSION_IMPORT = re.compile(
    r"(spec_from_file_location\s*\(|import_module\s*\(|runpy\.run_(path|module)\s*\(|SourceFileLoader\s*\(|"
    r"load_source\s*\(|exec\s*\(\s*(open|compile|Path)\b|__import__\s*\()")


def _imports_submission(pf: PyFile, agent_dirs) -> list[int]:
    """Lines where a graded test file loads the agent's code into its own process."""
    if pf.tree is None:
        return []
    text = pf.text
    if not any(d in text for d in agent_dirs) and "sys.path" not in text:
        return []
    out = []
    agent_re = "|".join(re.escape(d) for d in agent_dirs if d)
    # 1. explicit loaders pointed at the agent's tree
    for m in SUBMISSION_IMPORT.finditer(text):
        ln = line_of(text, m.start())
        window = "\n".join(pf.lines[max(0, ln - 6):ln + 2])
        if agent_re and re.search(rf"['\"](?:{agent_re})(/|['\"])", window) or \
                re.search(r"(solution|submission|candidate|agent|answer)\w*(_path|_file|_dir)?\b", window):
            out.append(ln)
    # 2. sys.path gets an agent dir, then a non stdlib module is imported
    ins = [m for m in re.finditer(r"sys\.path\.(insert|append)\(\s*(?:0\s*,\s*)?(?:str\()?\s*(?:Path\()?\s*['\"]([^'\"]+)['\"]", text)
           if _under(m.group(2), agent_dirs)]
    if ins:
        first = line_of(text, ins[0].start())
        for n in ast.walk(pf.tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno > first:
                mod = (n.module if isinstance(n, ast.ImportFrom) else n.names[0].name) or ""
                top = mod.split(".")[0]
                if top and top not in _STDLIB_TOP and top not in ("pytest", "numpy", "np", "pandas", "torch", "scipy"):
                    out.append(n.lineno)
                    break
    return sorted(set(out))


_STDLIB_TOP = {"os", "sys", "re", "json", "math", "subprocess", "pathlib", "importlib", "typing", "collections",
               "itertools", "functools", "shutil", "tempfile", "time", "random", "string", "io", "ast", "inspect",
               "unittest", "glob", "hashlib", "csv", "datetime", "decimal", "fractions", "copy", "textwrap",
               "dataclasses", "enum", "contextlib", "signal", "socket", "struct", "threading", "traceback",
               "types", "uuid", "warnings", "logging", "statistics", "operator", "base64", "pickle", "sqlite3",
               "difflib", "platform", "stat", "errno", "codecs", "heapq", "bisect", "array", "queue", "asyncio",
               "concurrent", "multiprocessing", "urllib", "http", "xml", "zipfile", "tarfile", "gzip", "shlex",
               "argparse", "configparser", "pprint", "numbers", "cmath", "secrets", "select", "selectors", "tomllib",
               "__future__", "test_utils", "conftest"}


# ---------------------------------------------------------------------------
# #1 grader-import-surface
# ---------------------------------------------------------------------------

def _partial_restore_flagged(text: str) -> bool:
    """Same condition partial-test-restore uses for a .sh file (so the two rules do not report one file twice)."""
    if "pytest" not in text:
        return False
    if re.search(r"git\s+checkout\s+\S+\s+--\s+\S", text) or re.search(r"git\s+checkout\s+\$\w+\s+\$", text):
        return not re.search(r"(--noconftest|-p\s+no:|git\s+clean\s+-[a-z]*f|find\s+.*conftest.*-delete|rm\s+-[rf]+\s+.*conftest)", text)
    return False


_CONFTEST_SCRUB = re.compile(r"(find\s[^\n]*conftest[^\n]*(-delete|-exec\s+rm|xargs\s+rm)|rm\s+-[rf]+\s[^\n]*conftest|"
                             r"git\s+clean\s+-[a-z]*f|git\s+stash\s+(-u|--include-untracked))")


def _import_surface_sh(pkg: Package):
    """-> list of (kind, evidence, task_or_None)."""
    hits = []
    for p, t in _graded_sh(pkg):
        if not re.search(r"pytest|python|PYTHONPATH", t):
            continue
        task = _task_of(pkg, p)
        if task is not None and task.separate and not task.artifacts:
            continue  # separate verifier and nothing of the agent's copied in
        agent_dirs = task.agent_dirs() if task else AGENT_DIRS
        start = task.workdir if task and task.workdir and not task.separate else None
        cmds, _docs = _sh_cached(pkg, p, t, start)
        scrub = _CONFTEST_SCRUB.search(t)
        partial = _partial_restore_flagged(t)
        for c in cmds:
            pa = _pytest_args(c.argv)
            env_pp = c.env.get("PYTHONPATH") or c.exported.get("PYTHONPATH") or ""
            addopts = (c.env.get("PYTEST_ADDOPTS") or "") + (c.exported.get("PYTEST_ADDOPTS") or "")
            if pa is not None:
                args, via_m = pa
                paths, opts = _pytest_paths(args)
                if "--noconftest" in opts or "--noconftest" in addopts or scrub or partial:
                    pass
                else:
                    pinned = any(o in opts for o in ("-c", "--confcutdir", "--rootdir", "--config-file", "--inifile"))
                    in_tree = False
                    if paths:
                        for x in paths:
                            if "$" in x or "`" in x:
                                continue  # unresolved variable: location unknown
                            if x.startswith("/"):
                                in_tree |= _under(x, agent_dirs)
                            else:
                                in_tree |= _under(c.cwd, agent_dirs)
                    else:
                        in_tree = _under(c.cwd, agent_dirs)
                    if in_tree and not pinned:
                        hits.append(("pytest", ev(p, c.line), task))
            # python started with the agent's tree first on sys.path
            pl = _python_launch(c.argv)
            if pl is not None:
                flags, mode, target = pl
                safe = "I" in flags or "P" in flags or c.env.get("PYTHONSAFEPATH") or c.exported.get("PYTHONSAFEPATH")
                if not safe:
                    if mode in ("m", "c") or (mode == "stdin" and "<<" in c.raw):
                        if _under(c.cwd, agent_dirs):
                            if mode == "m" and target == "pytest":
                                hits.append(("python-m-pytest", ev(p, c.line), task))
                            elif mode == "m" and target and target.split(".")[0] not in _PRELOADED:
                                hits.append(("cwd-on-path", ev(p, c.line), task))
                            elif mode in ("c", "stdin") and _shadowable_imports(_inline_code(c, _docs, mode)):
                                hits.append(("cwd-on-path", ev(p, c.line), task))
                    elif mode == "script" and target and "$" not in target and "`" not in target:
                        sp = target if target.startswith("/") else (os.path.normpath(os.path.join(c.cwd, target)) if c.cwd else "")
                        if _under(sp, agent_dirs) and not _written_by_script(t, target, c.line):
                            hits.append(("script-in-tree", ev(p, c.line), task))
            if env_pp and (pa is not None or pl is not None):
                parts = [x for x in re.split(r"[:\s]", env_pp) if x]
                if any(_under(x, agent_dirs) for x in parts) and not (pl and ("I" in pl[0] or "E" in pl[0])):
                    hits.append(("pythonpath", ev(p, c.line), task))
    return hits


# Built in or imported at interpreter startup: a file in the working directory cannot shadow these.
_PRELOADED = {"sys", "time", "builtins", "os", "posix", "errno", "itertools", "marshal", "gc", "atexit", "pwd", "io",
              "abc", "codecs", "stat", "encodings", "site", "posixpath", "genericpath", "faulthandler", "_io", "_thread",
              "_signal", "_warnings", "_weakref", "_collections_abc", "_sitebuiltins", "_stat", "_abc", "_codecs",
              "_functools", "_imp", "_locale", "_operator", "_sre", "_string", "_symtable", "_tokenize", "_tracemalloc",
              "_typing", "_ast", "_collections"}


def _inline_code(c, docs, mode) -> str:
    if mode == "c":
        try:
            i = c.argv.index("-c")
            return c.argv[i + 1] if i + 1 < len(c.argv) else ""
        except ValueError:
            return ""
    for hl, _header, body, _bl in docs:
        if hl == c.line:
            return body
    return ""


def _shadowable_imports(code: str) -> set:
    mods = set(re.findall(r"(?:^|[;\n])\s*(?:import|from)\s+([A-Za-z_][\w]*)", code or ""))
    for m in re.finditer(r"(?:^|[;\n])\s*import\s+([\w\s,.]+)", code or ""):
        mods |= {x.strip().split(".")[0].split(" ")[0] for x in m.group(1).split(",") if x.strip()}
    return {m for m in mods if m and m not in _PRELOADED}


def _written_by_script(text: str, target: str, before_line: int) -> bool:
    """True if the script creates the file it then runs (cat > x.py <<EOF, cp /tests/x.py ...)."""
    base = re.escape(os.path.basename(target))
    for m in re.finditer(rf"(cat\s+>+\s*\S*{base}|cp\s+\S+\s+\S*{base}\b|cp\s+\S*/{base}\s|tee\s+\S*{base}|>\s*\S*{base}\b)", text):
        if line_of(text, m.start()) <= before_line:
            return True
    return False


_CMD_KW = ("cwd", "workdir", "working_dir", "work_dir", "cwd_path", "directory", "dir")
_PY_CMD_HINT = re.compile(r"pytest|python\d?(\.\d+)?\s+-[mc]\b|PYTHONPATH")
# a string literal (or f-string) that starts pytest or python -m / -c
_PY_CMD_STR = (("pytest", "python"), re.compile(r"['\"][^'\"\n]*(\bpytest\b|\bpython\d?(\.\d+)?\s+-[mc]\s)"), 0)
_AGENTISH_NAME = re.compile(r"(work_?dir|workspace|repo(_?path|_?dir)?|project_?dir|testbed|cwd|code_?dir|app_?dir)", re.I)


def _resolve_str(node, consts) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                x = v.value
                nm = x.id if isinstance(x, ast.Name) else (x.attr if isinstance(x, ast.Attribute) else None)
                if nm and nm in consts and len(consts[nm]) < 200:
                    parts.append(consts[nm])
                elif nm and _AGENTISH_NAME.search(nm):
                    parts.append("/workdir")  # agent work dir placeholder
                else:
                    parts.append("{}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _resolve_str(node.left, consts), _resolve_str(node.right, consts)
        if a is not None or b is not None:
            return (a or "") + (b or "")
    return None


def _import_surface_py(pkg: Package):
    """pytest / python -m started from the agent's work dir in commands that verifiers env code builds."""
    hits = []
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        lines = _hint_lines(pf.text, _PY_CMD_STR)
        if not lines:
            continue
        agent_consts = {k for k, v in pf.consts.items() if _under(v, AGENT_DIRS) and len(v) < 120}
        calls = []
        for unit in _units(pf, lines):
            calls += [x for x in ast.walk(unit.node) if isinstance(x, ast.Call)]
        for n in calls:
            cwd_hint = None
            for kw in n.keywords:
                if kw.arg in _CMD_KW:
                    s = _resolve_str(kw.value, pf.consts)
                    txt = ast.unparse(kw.value) if s is None else s
                    if _under(s, AGENT_DIRS) or _AGENTISH_NAME.search(txt) or any(a in txt for a in agent_consts):
                        cwd_hint = "/workdir"
            if cwd_hint is None:
                for a in n.args[1:]:
                    try:
                        txt = ast.unparse(a)
                    except Exception:
                        continue
                    if len(txt) < 60 and (_AGENTISH_NAME.search(txt) or any(c_ in txt for c_ in agent_consts)) and \
                            not re.search(r"['\"]", txt):
                        cwd_hint = "/workdir"
            for a in list(n.args) + [k.value for k in n.keywords]:
                s = _resolve_str(a, pf.consts)
                if not s or ("pytest" not in s and "python" not in s) or len(s) > 3000:
                    continue
                cmds, _ = _sh_parse(s, cwd_hint)
                for c in cmds:
                    pa = _pytest_args(c.argv)
                    dirs = AGENT_DIRS + ("/workdir",)
                    if pa is not None:
                        paths, opts = _pytest_paths(pa[0])
                        if "--noconftest" in opts or "--noconftest" in s or _CONFTEST_SCRUB.search(s):
                            continue
                        if any(o in opts for o in ("-c", "--confcutdir", "--rootdir")):
                            continue
                        in_tree = any(_under(x, dirs) if x.startswith("/") else _under(c.cwd, dirs) for x in paths) \
                            if paths else _under(c.cwd, dirs)
                        if in_tree:
                            hits.append(("pytest", ev(pf, a.lineno), None))
                            continue
                    pl = _python_launch(c.argv)
                    if pl is not None and pl[1] in ("m", "c") and _under(c.cwd, dirs) and not ({"I", "P"} & pl[0]):
                        hits.append(("python-m-pytest" if pl[2] == "pytest" else "cwd-on-path", ev(pf, a.lineno), None))
    return hits


def _namespace_copies(pkg: Package):
    """Graded tests that copy the submission's names into their own globals (from solution import *,
    globals().update(vars(mod)), a loop assigning globals()[k] = v), so the submission can redefine any name
    the asserts use, builtins included."""
    hits = []
    for path, text in _graded_test_files(pkg):
        if "import *" not in text and "globals()" not in text:
            continue
        pf = _pyf(pkg, path)
        if pf.tree is None:
            continue
        task = _task_of(pkg, path)
        dirs = task.agent_dirs() if task and not task.separate else AGENT_DIRS
        imp = _imports_submission(pf, dirs)
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) and n.module:
                top = n.module.split(".")[0]
                if top in _STDLIB_TOP or top in ("pytest", "numpy", "pandas", "torch"):
                    continue
                if imp or re.search(r"(solution|submission|candidate|answer|student|agent)", n.module, re.I):
                    hits.append(("namespace", ev(pf, n.lineno), task))
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update" and \
                    ast.unparse(n.func.value) == "globals()" and imp:
                hits.append(("namespace", ev(pf, n.lineno), task))
            elif isinstance(n, ast.For) and imp:
                it = ast.unparse(n.iter)
                body = "\n".join(ast.unparse(b) for b in n.body)
                if re.search(r"(__dict__|vars\(|dir\()", it) and re.search(r"globals\(\)\s*\[", body):
                    hits.append(("namespace", ev(pf, n.lineno), task))
    return hits


_IMPORT_MSG = {
    "namespace": "graded tests copy the submission's names into their own namespace, so the submission can "
                 "redefine any helper or builtin the asserts call",
    "pytest": "pytest collects the agent's working tree without --noconftest or a pinned config, so a conftest.py "
              "or pytest.ini the agent leaves there runs inside the grader and can force every test to pass",
    "python-m-pytest": "python -m pytest runs from the agent's working directory, which goes first on sys.path, "
                       "so a pytest.py or conftest.py the agent leaves there replaces the test runner",
    "cwd-on-path": "the grader starts python with -m, -c or stdin from the agent's working directory, which goes "
                   "first on sys.path, so any module the agent drops there shadows the grader's imports",
    "script-in-tree": "the grader runs a Python script that lives in the agent's working tree, so the agent can "
                      "edit it or shadow what it imports",
    "pythonpath": "PYTHONPATH puts the agent's directory ahead of site-packages for the test run, so a module the "
                  "agent drops there (pytest itself included) replaces the grader's code",
}


# Evidence: row 1, arxiv:2605.12673 (BenchJack, conftest hook 500 of 500 on SWE-bench Verified, App E);
# arxiv:2511.18397 Sec 2 (conftest patching in production coding envs); arxiv:2503.11926 (local stub module).
@rule("grader-import-surface", "hack", "high",
      "The grader starts pytest or python so that files in the agent's tree load first, through conftest.py, "
      "pytest.ini, the working directory on sys.path or PYTHONPATH.")
def r_grader_import_surface(pkg: Package):
    sh_hits = _import_surface_sh(pkg)
    py_hits = _import_surface_py(pkg)
    ns_hits = _namespace_copies(pkg)
    v05 = _import_surface_v05(pkg)  # v0.5 widenings, reported as their own findings
    if not sh_hits and not py_hits and not ns_hits:
        return v05
    if SUPPRESS_SHARED_BOX_OVERLAP:
        keep = []
        for h in sh_hits:
            task = h[2]
            if task is not None and task.separate:
                keep.append(h)
            elif task is not None and task.toml_path:
                # a non-root agent defeats the root-only plants harbor-shared-grading describes, but not this one
                if task.nonroot_agent or not _fires(pkg, "harbor-shared-grading"):
                    keep.append(h)
            elif not _shared_box_flagged(pkg):
                keep.append(h)
        sh_hits = keep
        if py_hits and _shared_box_flagged(pkg):
            py_hits = []
        if py_hits and _fires(pkg, "model-named-files-into-test-dir"):
            py_hits = []
    hits = ns_hits + sh_hits + py_hits  # namespace copies do not depend on where the grader runs
    if not hits:
        return v05
    kinds = []
    for k, _e, _t in hits:
        if k not in kinds:
            kinds.append(k)
    evid = [e for _k, e, _t in hits]
    msg = _IMPORT_MSG[kinds[0]]
    if len(kinds) > 1:
        msg += f" (also: {', '.join(kinds[1:])})"
    return [F("grader-import-surface", msg[0].upper() + msg[1:] + ".", evid, confidence="review")] + v05


def _fires(pkg: Package, rid: str) -> bool:
    c = _cache(pkg).setdefault("fires", {})
    if rid not in c:
        meta = RULES.get(rid)
        try:
            c[rid] = bool(meta and meta["fn"](pkg))
        except Exception:
            c[rid] = False
    return c[rid]


def _shared_box_flagged(pkg: Package) -> bool:
    return any(_fires(pkg, r) for r in ("grader-in-agent-sandbox", "keep-sandbox-for-scoring", "harbor-shared-grading"))


# ---------------------------------------------------------------------------
# #6 verdict-parse-semantics
# ---------------------------------------------------------------------------

def _parser_sources(pkg: Package):
    """(path, PyFile, line_offset) for Python that may parse test results: env code, graded tests, heredoc Python
    embedded in graded shell scripts."""
    c = _cache(pkg)
    if "parsers" in c:
        return c["parsers"]
    graded = {p for p, _t in _graded_test_files(pkg)}
    out = [(pf.path, pf, 0, "graded" if pf.path in graded else "env") for pf in pkg.py_all()]
    seen = {pf.path for pf in pkg.py_all()}
    for p, t in _graded_test_files(pkg):
        if p not in seen and any(_hint_lines(t, h) for h in _ALL_VERDICT_HINTS):
            out.append((p, _pyf(pkg, p), 0, "graded"))
            seen.add(p)
    for p, t in _graded_sh(pkg):
        if "<<" not in t:
            continue
        _cmds, docs = _sh_cached(pkg, p, t)
        for hl, header, body, bl in docs:
            if re.search(r"python|\.py\b", header) and len(body) < 200_000:
                pf = parse_py(p, body)
                if pf.tree is not None:
                    out.append((p, pf, bl - 1, "heredoc"))
    c["parsers"] = out
    return out


def _line_in_used_code(pkg: Package, pf: PyFile, line: int) -> bool:
    """False when the line sits in a function nothing refers to (dead helper code)."""
    name = _enclosing_name(pf, line)
    if name is None or name.startswith("test") or name in ("main", "__init__", "__call__"):
        return True
    if re.search(r"(reward|score|grade|judge|verif|metric|evaluat)", name, re.I):
        return True  # handed to a rubric or called by the framework
    rx = re.compile(rf"\b{re.escape(name)}\b")
    if len(rx.findall(pf.text)) > 1:
        return True
    for other in pkg.py_all():
        if other is not pf and name in other.text and rx.search(other.text):
            return True
    return False


def _reward_path_keys(pkg: Package) -> set:
    c = _cache(pkg)
    if "rpk" not in c:
        from .rules import scoring_funcs
        c["rpk"] = {(pf.path, fi.start) for pf, fi in scoring_funcs(pkg)}
    return c["rpk"]


def _is_passish_target(node) -> bool:
    names = []
    for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
        names += [x.id for x in ast.walk(t) if isinstance(x, ast.Name)]
        names += [x.attr for x in ast.walk(t) if isinstance(x, ast.Attribute)]
    return any(re.search(r"(pass|ok\b|success|good|accept|resolved|green|solved)", nm, re.I) for nm in names)


# Candidate lines are found by locating cheap literal gate words with str.find and running the regex on the
# surrounding line only, so a 14 MB package costs a few milliseconds.
_SOFT_HINT = (("skip", "SKIP", "Skip", "xfail", "XFAIL"), re.compile(
    r"(?i)(['\"](passed|pass|xpass\w*|ok|success)['\"]|\.PASSED\b)[^\n]{0,80}(['\"](skipped|skip|xfail\w*)['\"]|\.(SKIPPED|XFAIL)\b)|"
    r"(['\"](skipped|skip|xfail\w*)['\"]|\.(SKIPPED|XFAIL)\b)[^\n]{0,80}(['\"](passed|pass|xpass\w*|ok|success)['\"]|\.PASSED\b)"), 0)
_NOTFAILED_HINT = (("!=", "not in"), re.compile(r"(?i)!=\s*['\"](failed|fail|failure)['\"]|['\"](failed|fail|failure)['\"]\s*!=|"
                                               r"['\"](failed|fail)['\"]\s+not\s+in\b|!=\s*[\w.]*\.FAILED\b"), 0)
_VACUOUS_HINTS = (
    (("else 1", "else True"), re.compile(r"/[^\n]*\belse\s+(1|1\.0|True)\b"), 0),
    (("return 1",), re.compile(r"if\s+(not\s+[\w.]+|len\([\w.]+\)\s*==\s*0|[\w.]*(total|count|num|n)\w*\s*==\s*0)\s*:\s*(#[^\n]*)?\n?\s*"
                               r"return\s+1(\.0)?\b"), 1),
    (("<=",), re.compile(r"(?i)(required|expected|fail_to_pass|f2p|need|target)\w*\s*<=\s*\w*pass"), 0),
    (("0, 5", "0,5"), re.compile(r"[(\[{]\s*0\s*,\s*5\s*[)\]}]"), 0),
    (("all(",), re.compile(r"(?i)\ball\([^\n]*(pass|status|outcome|result|test|success|ok\b)"), 0),
)
_PRINTED_HINT = (("\" in ", "' in ", "re.search(", "re.findall(", "re.match(", ".count("), re.compile(
    r"['\"](PASSED|PASS|passed|pass|OK|ok|SUCCESS|Success|success|All tests passed|ALL TESTS PASSED|all tests passed|"
    r"Accepted|ACCEPTED|TESTS PASSED|tests passed|\d* ?passed)['\"]\s+in\s|"
    r"re\.(search|findall|match)\([^\n]*(PASSED|passed|PASS\b|All tests passed|SUCCESS)|\.count\(\s*['\"](PASSED|passed|PASS|ok|OK|\.)['\"]"), 0)
_JUDGE_HINT = (("\" in ", "' in "), re.compile(r"['\"](correct|Correct|CORRECT|true|True|TRUE)['\"]\s+in\s"), 0)


def _hint_lines(text: str, hint, cap: int = 3000) -> list[int]:
    """Line numbers where one of the gate words occurs and the regex matches that line (plus `span` lines before)."""
    words, rx, span = hint
    found = set()
    for w in words:
        if w not in text:
            continue
        pos, k = text.find(w), 0
        while pos != -1 and k < cap:
            k += 1
            a = text.rfind("\n", 0, pos) + 1
            for _ in range(span):
                if a > 0:
                    a = text.rfind("\n", 0, a - 1) + 1
            b = text.find("\n", pos)
            b = len(text) if b == -1 else b
            if rx.search(text, a, b):
                found.add(pos)
            pos = text.find(w, b)
    lines = sorted({text.count("\n", 0, p_) + 1 for p_ in sorted(found)[:60]})
    return lines


_ALL_VERDICT_HINTS = (_SOFT_HINT, _NOTFAILED_HINT, _PRINTED_HINT, _JUDGE_HINT) + _VACUOUS_HINTS


class _ModuleBody:
    """Module level statements of a file, treated like one function."""

    def __init__(self, pf: PyFile, only_line: int | None = None):
        body = [s for s in (pf.tree.body if pf.tree is not None else [])
                if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if only_line is not None:
            body = [s for s in body if s.lineno <= only_line <= (getattr(s, "end_lineno", s.lineno) or s.lineno)]
        self.node = ast.Module(body=body, type_ignores=[])
        self.name = "<module>"
        self.params = ["log"]
        self.start = 1
        try:
            self.src = ast.unparse(self.node)
        except Exception:
            self.src = ""


def _units(pf: PyFile, lines, whole_module=False):
    """Smallest function (or the module body) around each candidate line."""
    out, seen = [], set()
    for ln in lines:
        best = None
        for fi in pf.funcs:
            if fi.start <= ln <= fi.end and (best is None or fi.end - fi.start < best.end - best.start):
                best = fi
        key = ("f", best.start) if best is not None else ("m", None if whole_module else ln)
        if key in seen:
            continue
        seen.add(key)
        out.append(best if best is not None else _ModuleBody(pf, None if whole_module else ln))
    return out


def _soft_pass_sets(pf: PyFile):
    """Literal status collections that mix a pass status with SKIPPED / XFAIL and contain no failing status."""
    out = []
    if pf.tree is None:
        return out
    lines = _hint_lines(pf.text, _SOFT_HINT)
    if not lines:
        return out
    for unit in _units(pf, lines):
        par = _parents(unit.node)
        for n in ast.walk(unit.node):
            if not isinstance(n, (ast.Tuple, ast.List, ast.Set)) or not (2 <= len(n.elts) <= 8):
                continue
            toks = [_status_token(e) for e in n.elts]
            if any(t is None for t in toks):
                continue
            toks = set(toks)
            if not (toks & PASS_TOKENS and toks & SOFT_TOKENS) or toks & FAIL_TOKENS:
                continue
            p = par.get(n)
            used = False
            if isinstance(p, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in p.ops) and n in p.comparators:
                used = True
            if isinstance(p, (ast.Assign, ast.AnnAssign)) and _is_passish_target(p):
                used = True
            if used:
                out.append(n.lineno)
    return sorted(set(out))


def _not_failed_checks(pf: PyFile):
    """`status != "failed"` counted as a pass."""
    out = []
    if pf.tree is None:
        return out
    lines = _hint_lines(pf.text, _NOTFAILED_HINT)
    if not lines:
        return out
    for unit in _units(pf, lines):
        par = _parents(unit.node)
        for n in ast.walk(unit.node):
            if not isinstance(n, ast.Compare) or len(n.ops) != 1:
                continue
            op = n.ops[0]
            tok = None
            if isinstance(op, ast.NotEq):
                tok = _status_token(n.comparators[0]) or _status_token(n.left)
            elif isinstance(op, ast.NotIn):
                tok = _status_token(n.left)
            if tok not in ("failed", "fail", "failure"):
                continue
            p, ctx = par.get(n), None
            for _ in range(4):
                if p is None:
                    break
                if isinstance(p, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.comprehension, ast.If, ast.Return, ast.Call)):
                    ctx = p
                    break
                p = par.get(p)
            if ctx is None:
                continue
            seg = ast.unparse(ctx)[:300]
            if isinstance(ctx, ast.If):
                body = "\n".join(ast.unparse(s) for s in ctx.body)[:300]
                if re.search(r"(pass\w*|success\w*|ok\w*|score|reward)\s*(\+=|=)", body, re.I):
                    out.append(n.lineno)
            elif re.search(r"(pass|success|ok\b|score|reward|resolved)", seg, re.I):
                out.append(n.lineno)
    return sorted(set(out))


def _vacuous_pass(pf: PyFile):
    """Empty denominator or empty requirement set treated as full credit."""
    out = []
    if pf.tree is None:
        return out
    lines = sorted({ln for h in _VACUOUS_HINTS for ln in _hint_lines(pf.text, h)})
    if not lines:
        return out
    cand = set(lines)
    for unit in _units(pf, lines):
        for n in ast.walk(unit.node):
            ln = getattr(n, "lineno", None)
            if ln is None or not (cand & set(range(ln, (getattr(n, "end_lineno", ln) or ln) + 2))):
                continue
            # a / b if b else 1.0   |   a / b if b > 0 else 1
            if isinstance(n, ast.IfExp) and isinstance(n.body, ast.BinOp) and isinstance(n.body.op, (ast.Div, ast.FloorDiv)):
                den = ast.unparse(n.body.right)
                if isinstance(n.orelse, ast.Constant) and n.orelse.value in (1, 1.0, True) and n.orelse.value is not False \
                        and den in ast.unparse(n.test) and \
                        re.search(r"(pass|test|case|check|assert|correct|criteri|unit|success)", ast.unparse(n.body), re.I):
                    out.append(n.lineno)
            # if total == 0: return 1.0 / if not tests: return 1.0
            if isinstance(n, ast.If) and n.body and isinstance(n.body[0], ast.Return) and \
                    isinstance(n.body[0].value, ast.Constant) and n.body[0].value.value in (1, 1.0) and \
                    n.body[0].value.value is not True:
                tt = ast.unparse(n.test)
                fname = getattr(unit, "name", "")
                # the function must go on to return a pass ratio (passed / total), so 1.0 is full credit, not a p value
                ratio = any(isinstance(r, ast.Return) and isinstance(r.value, ast.BinOp) and
                            isinstance(r.value.op, ast.Div) and
                            re.search(r"(pass|success|correct|\bok\b|solved|resolved|good)", ast.unparse(r.value.left), re.I)
                            for r in ast.walk(unit.node))
                if ratio and re.fullmatch(r"(not\s+\w+(\.\w+)*|len\(\w+(\.\w+)*\)\s*==\s*0|\w*(total|count|num|n)\w*\s*==\s*0|not\s+len\(\w+\))", tt) \
                        and (re.search(r"(test|case|check|assert|unit|criteri|requirement|spec)", tt, re.I) or
                             re.search(r"(test|pass|verdict|resolv|correct|grade|score_tests|run_tests)", fname, re.I)):
                    out.append(n.lineno)
            # required <= passed  (subset as verdict) with no emptiness guard
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.LtE):
                a, b = ast.unparse(n.left), ast.unparse(n.comparators[0])
                if re.search(r"(required|expected|fail_to_pass|f2p|need|target)", a, re.I) and \
                        re.search(r"pass", b, re.I) and not re.search(rf"(len\(\s*{re.escape(a)}\s*\)|if\s+not\s+{re.escape(a)}\b|{re.escape(a)}\s+and\b)", pf.text):
                    out.append(n.lineno)
            # all(...) over parsed test results with no emptiness guard: an empty parse is a pass
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "all" and len(n.args) == 1 and \
                    isinstance(n.args[0], (ast.GeneratorExp, ast.ListComp)) and _vacuous_all(unit, n):
                out.append(n.lineno)
            # pytest exit code 5 (no tests collected) accepted
            if isinstance(n, ast.Compare) and any(isinstance(o, ast.In) for o in n.ops):
                left = ast.unparse(n.left)
                if re.search(r"(returncode|exit_code|exitcode|rc\b|status)", left) and isinstance(n.comparators[0], (ast.Tuple, ast.List, ast.Set)):
                    vals = {getattr(e, "value", None) for e in n.comparators[0].elts}
                    if 0 in vals and 5 in vals and "pytest" in pf.text:
                        out.append(n.lineno)
    return sorted(set(out))


def _vacuous_all(unit, call) -> bool:
    gen = call.args[0]
    inner = ast.unparse(gen.elt)
    if not re.search(r"(PASS|pass|status|outcome|ok\b|success|ACCEPTED|accepted)", inner):
        return False
    it = gen.generators[0].iter
    base = it.func.value if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and \
        it.func.attr in ("values", "items", "keys") else it
    b = ast.unparse(base)
    if not re.fullmatch(r"[\w.\[\]'\"]+", b) or not re.search(r"(result|test|status|outcome|report|parsed|map|cases)", b, re.I):
        return False
    src = unit.src
    eb = re.escape(b)
    if re.search(rf"(len\(\s*{eb}\s*\)|\bnot\s+{eb}\b|\b{eb}\s+and\s+all\(|bool\(\s*{eb}\s*\)|"
                 rf"if\s+{eb}\s*:|{eb}\s*==\s*\{{\}}|{eb}\s*!=\s*\{{\}})", src):
        return False
    # the collection must come from parsing test output (a parser call, json, regex), not from a fixed task list
    for a in ast.walk(unit.node):
        if isinstance(a, ast.Assign) and isinstance(a.value, ast.Call) and any(ast.unparse(t) == b for t in a.targets):
            if re.search(r"(pars|loads?\b|findall|split|extract|get_logs|log|report|status)", ast.unparse(a.value.func), re.I):
                return True
    return False


_PASS_TOKEN_RE = re.compile(r"^(PASSED|PASS|passed|pass|OK|ok|SUCCESS|Success|success|All tests passed|ALL TESTS PASSED|"
                            r"all tests passed|Accepted|ACCEPTED|TESTS PASSED|tests passed|\d* ?passed)$")
_RUN_CALL = re.compile(r"(\bsubprocess\.(run|check_output|Popen|call)|\.exec_run\(|\.execute\w*\(|\.run_command\w*\(|"
                       r"\bsandbox\w*\.\w+\(|\bruntime\.\w+\(|\.exec\(|\.arun\(|\.aexec\(|run_in_sandbox|execute_code|"
                       r"run_code|run_tests)")
_OUTPUTISH = re.compile(r"(stdout|output|out\b|log\b|logs\b|result|res\b|text|content|line|stderr|report)", re.I)


def _printed_token_checks(pf: PyFile, module_level: bool = False):
    """`"PASSED" in out` / re.search("passed", out) on output of a process that ran the submission.
    -> list of (FuncInfo or _ModuleBody, line)."""
    out = []
    if pf.tree is None:
        return out
    lines = _hint_lines(pf.text, _PRINTED_HINT)
    if not lines:
        return out
    for fn in _units(pf, lines, whole_module=True):
        if isinstance(fn, _ModuleBody) and not module_level:
            continue
        src = fn.src
        name = fn.name.lower()
        runs = bool(_RUN_CALL.search(src))
        parses_log = (bool(re.search(r"(parse|log|status|result|report|summary)", name)) and
                      any(re.search(r"(log|output|stdout|text|content)", p_, re.I) for p_ in fn.params)) or \
            (module_level and bool(re.search(r"(open\(|read_text\(|sys\.stdin|sys\.argv)", src)))
        if not (runs or parses_log):
            continue
        if re.search(r"(uuid|secrets\.token|nonce|sentinel|junit|\.xml\b|ElementTree|ctrf)", src):
            continue
        for n in ast.walk(fn.node):
            tok, target = None, None
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.In):
                s = _str_of(n.left)
                if s and _PASS_TOKEN_RE.match(s.strip()):
                    tok, target = s, n.comparators[0]
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("search", "findall", "match") \
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "re" and len(n.args) >= 2:
                s = _str_of(n.args[0])
                if s and re.search(r"(PASSED|passed|\bPASS\b|All tests passed|SUCCESS)", s) and not re.search(r"FAIL|fail|error|ERROR", s):
                    tok, target = s, n.args[1]
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "count" and n.args:
                s = _str_of(n.args[0])
                if s and s.strip() in ("PASSED", "passed", "PASS", "ok", "OK", "."):
                    if s.strip() == "." and "status_line" not in ast.unparse(n.func.value):
                        continue
                    tok, target = s, n.func.value
            if tok is None:
                continue
            try:
                tsrc = ast.unparse(target)
            except Exception:
                continue
            if not _OUTPUTISH.search(tsrc):
                continue
            out.append((fn, n.lineno))
    return out


def _judge_correct_substring(pf: PyFile):
    """`"correct" in judge_reply` passes when the judge says "incorrect"."""
    out = []
    if pf.tree is None:
        return out
    lines = _hint_lines(pf.text, _JUDGE_HINT)
    if not lines:
        return out
    for fn in _units(pf, lines):
        for n in ast.walk(fn.node):
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.In):
                s = _str_of(n.left)
                if s is None or s.strip().lower() not in ("correct", "true"):
                    continue
                tsrc = ast.unparse(n.comparators[0])
                if not re.search(r"(judge|verdict|grade|grading|eval|response|reply|result|output)", tsrc, re.I):
                    continue
                neg = {"correct": r"incorrect|not correct", "true": r"false|untrue"}[s.strip().lower()]
                if re.search(neg, fn.src, re.I):
                    continue
                out.append(n.lineno)
    return out


def _sh_verdicts(pkg: Package):
    """Shell side: pass token grepped from test output, exit 5 accepted, reward on 'no FAIL line'."""
    tok_hits, vac_hits = [], []
    for p, t in _graded_sh(pkg):
        if not re.search(r"(?i)pass|ok|success|fail", t):
            continue
        cmds, _ = _sh_cached(pkg, p, t, None)
        runner_out = {}   # file or var -> line of a command that ran the submission
        for c in cmds:
            if _runs_submission(c) and c.out:
                runner_out[c.out] = c.line
        assigned = {}
        for m in re.finditer(r"^\s*(\w+)=\$\((.*)\)\s*$", t, re.M):
            inner = _tokens(m.group(2))
            if inner and (_pytest_args(inner) is not None or _is_test_runner(inner)):
                assigned[m.group(1)] = line_of(t, m.start())
        for c in cmds:
            if c.argv[0] != "grep":
                continue
            pats = [a for a in c.argv[1:] if not a.startswith("-")]
            if not pats:
                continue
            pat, files = pats[0], pats[1:]
            if not re.search(r"(PASSED|passed|\bPASS\b|All tests passed|SUCCESS|\bOK\b)", pat) or re.search(r"FAIL|fail|ERROR", pat):
                continue
            src_line = None
            for f_ in files:
                if f_ in runner_out:
                    src_line = runner_out[f_]
            if c.piped_from is not None:
                pf_ = c.piped_from
                if pf_.argv[0] in ("echo", "printf", "cat"):
                    for a in pf_.argv[1:]:
                        for var, ln in assigned.items():
                            if f"${var}" in a or "${" + var + "}" in a or a in runner_out:
                                src_line = ln
                        if a in runner_out:
                            src_line = runner_out[a]
                elif _runs_submission(pf_):
                    src_line = pf_.line
            if src_line is not None and not re.search(r"(-A|-B)\s*\d", c.raw):
                tok_hits.append(ev(p, c.line))
        # reward 1 when no FAILED line is found (nothing ran counts as a pass)
        for m in re.finditer(r"if\s+!\s+grep\s+-q\w*\s+['\"]?(FAILED|FAIL|failed|Error|ERROR)['\"]?[^\n]*;\s*then\s*\n\s*(echo\s+['\"]?1(\.0)?['\"]?\s*>|reward=1)", t):
            vac_hits.append(ev(p, line_of(t, m.start())))
        for m in re.finditer(r"-eq\s+5\b|==\s*5\b", t):
            ctx = t[max(0, m.start() - 200): m.end() + 200]
            if "pytest" in t and re.search(r"(echo\s+['\"]?1(\.0)?['\"]?\s*>\s*\S*reward|reward=1)", ctx):
                vac_hits.append(ev(p, line_of(t, m.start())))
    return tok_hits, vac_hits


_RUNNER = re.compile(r"^(go|cargo|npm|yarn|pnpm|jest|npx|mvn|gradle|\./gradlew|make|ctest|dotnet|node|deno|bun|ruby|"
                     r"rspec|bundle|php|phpunit|swift|dart|mix|lein|sbt|stack|cabal|julia|Rscript|lua|perl)$")


def _is_test_runner(argv) -> bool:
    if not argv:
        return False
    a0 = os.path.basename(argv[0])
    if a0 in ("timeout", "uv", "conda", "poetry") and len(argv) > 1:
        return _is_test_runner(argv[1:] if a0 != "timeout" else argv[2:] if len(argv) > 2 else [])
    if _RUNNER.match(a0):
        return True
    if a0 in ("bash", "sh") and len(argv) > 1 and re.search(r"(run_tests|run_script|test)", argv[1]):
        return True
    if re.match(r"^\./?\S*(run_tests|run_test|test)\S*\.sh$", argv[0]):
        return True
    return False


def _runs_submission(c: _Cmd) -> bool:
    if _pytest_args(c.argv) is not None or _is_test_runner(c.argv):
        return True
    pl = _python_launch(c.argv)
    if pl and pl[1] == "script" and pl[2] and not pl[2].startswith("/tests/") and not re.search(r"(parser|grade|evaluat|verif|score|judge)", pl[2]):
        return True
    return False


# exit status of a process that loaded the submission decides the reward
def _exit_status_verdicts(pkg: Package):
    out = []
    for p, t in _graded_sh(pkg):
        task = _task_of(pkg, p)
        if task is None or not re.search(r"pytest|python", t):
            continue
        cmds, _ = _sh_cached(pkg, p, t, task.workdir if task.workdir and not task.separate else None)
        for c in cmds:
            pa = _pytest_args(c.argv)
            targets = []
            selected = {}
            if pa is not None:
                paths, _o = _pytest_paths(pa[0])
                for a in pa[0]:
                    if "::" in a and not a.startswith("-"):
                        selected.setdefault(a.split("::")[0], set()).add(re.sub(r"\[.*$", "", a.split("::")[-1]))
                for x in paths:
                    if x.startswith("/tests/"):
                        targets.append(task.pkg_path(x))
                    elif c.cwd == "/tests" and not x.startswith("/"):
                        targets.append(task.pkg_path("/tests/" + x))
                    if x in selected and targets:
                        selected[targets[-1]] = selected.pop(x)
            else:
                pl = _python_launch(c.argv)
                if pl and pl[1] == "script" and pl[2]:
                    x = pl[2]
                    if x.startswith("/tests/"):
                        targets.append(task.pkg_path(x))
                    elif c.cwd == "/tests" and not x.startswith("/"):
                        targets.append(task.pkg_path("/tests/" + x))
            if not targets or not _exit_status_rewarded(t, c.line):
                continue
            for tp in targets:
                if tp not in pkg.files:
                    continue
                pf = _pyf(pkg, tp)
                if re.search(r"(reward\.txt|reward\.json|write_reward|REWARD)", pf.text):
                    continue  # the script writes its own reward; exit status is not the verdict
                imp = _imports_submission(pf, task.agent_dirs() if not task.separate else AGENT_DIRS)
                sel = selected.get(tp)
                if sel:
                    sel_src = "\n".join(fi.src for fi in pf.funcs if fi.name in sel)
                    imp = [ln for ln in imp if _enclosing_name(pf, ln) is None or _enclosing_name(pf, ln) in sel or
                           (not _enclosing_name(pf, ln).startswith("test") and
                            re.search(rf"\b{re.escape(_enclosing_name(pf, ln))}\b", sel_src))]
                if imp:
                    out.append((ev(p, c.line), ev(pf, imp[0])))
    return out


def _enclosing_name(pf: PyFile, line: int):
    best = None
    for fi in pf.funcs:
        if fi.start <= line <= fi.end and (best is None or fi.end - fi.start < best.end - best.start):
            best = fi
    return best.name if best else None


def _exit_status_rewarded(t: str, line: int) -> bool:
    """The command at `line` is the condition of an if that writes reward 1, or $? right after it decides the reward."""
    lines = t.splitlines()
    here = lines[line - 1] if 0 < line <= len(lines) else ""
    nxt = "\n".join(lines[line - 1: line + 12])
    if re.match(r"\s*(if|elif)\s", here) or re.search(r"\bif\s+(uvx|uv|pytest|python\S*|timeout)\b", here):
        if re.search(r"then\s*(\n|$)[\s\S]{0,200}?(echo\s+['\"]?1(\.0)?['\"]?\s*>|reward=1|printf\s+['\"]?1)", nxt):
            return True
    if re.search(r"(\$\?|\$\{?\w*(exit|status|rc|code)\w*\}?)", nxt) and \
            re.search(r"(-eq\s+0|==\s*0|=\s*0\s*\]|-ne\s+0)", nxt) and re.search(r"(reward|REWARD)", nxt):
        return True
    return False


# agent controllable skips in graded tests, with an exit status or failed count verdict
_SKIP_CALL = re.compile(r"(pytest\.skip\(|pytest\.importorskip\(|mark\.skipif\(|unittest\.skip(If|Unless)\(|self\.skipTest\(|"
                        r"raise\s+(unittest\.)?SkipTest|pytest\.xfail\()")
_FS_COND = re.compile(r"(exists\(|is_file\(|is_dir\(|isfile\(|isdir\(|which\(|find_spec\(|ImportError|ModuleNotFoundError|"
                      r"FileNotFoundError|returncode|os\.environ|getenv|listdir|glob\(|stat\()")


def _agent_skips(pkg: Package):
    out = []
    for path, text in _graded_test_files(pkg):
        if not _SKIP_CALL.search(text):
            continue
        pf = _pyf(pkg, path)
        task = _task_of(pkg, pf.path)
        verdict_exit = False
        for p, t in _graded_sh(pkg):
            if _task_of(pkg, p) is task and os.path.basename(pf.path) in t and "pytest" in t:
                cmds, _ = _sh_cached(pkg, p, t, task.workdir if task and task.workdir and not task.separate else None)
                for c in cmds:
                    if _pytest_args(c.argv) is not None and _exit_status_rewarded(t, c.line):
                        verdict_exit = True
        if not verdict_exit:
            continue
        for m in _SKIP_CALL.finditer(pf.text):
            ln = line_of(pf.text, m.start())
            call_line = pf.line(ln)
            window = "\n".join(pf.lines[max(0, ln - 5):ln])
            if "importorskip" in call_line or "skipif" in call_line and _FS_COND.search(call_line) or \
                    _FS_COND.search(window) or (m.group(0).startswith(("pytest.skip", "self.skipTest", "raise")) and
                                                re.search(r"except\s+\w*(Import|NotFound|OSError)", window)):
                out.append(ev(pf, ln))
    return out


_VERDICT_MSG = {
    "soft": "A test result parser counts SKIPPED or XFAIL as a pass, so a run that skips the graded tests scores",
    "notfailed": "A test result parser counts every status other than failed as a pass, so skipped, errored and "
                 "uncollected tests score",
    "vacuous": "The verdict treats an empty test run or an empty requirement list as full credit, so a run that "
               "collects nothing scores",
    "printed": "The pass verdict is a token searched in output that the submission's own process printed, so code "
               "that prints the token and exits scores",
    "judge": "A judge reply is scored by substring, so a reply such as incorrect or not true contains the pass word",
    "exit": "The reward is the exit status of a test process that imports the submission, so a solution that calls "
            "os._exit(0) on import scores",
    "skip": "Graded tests skip themselves on conditions the agent controls (a missing file or import) and the "
            "reward is the pytest exit status, so deleting the dependency scores",
}


# Evidence: row 6, arxiv:2610.24206 Sec 3.5 (a PASSED printer scored 1 on SWE-Bench Pro before V2);
# arxiv:2610.28679 Sec 2.3 (SKIPPED and XFAIL counted as pass in 35 Terminal-Bench tasks);
# arxiv:2511.18397 Sec 2 (sys.exit(0) before the asserts); inspect_evals BEST_PRACTICES (unanchored verdict parse).
@rule("verdict-parse-semantics", "hack", "high",
      "The pass verdict is read so that skipped tests, an empty run, text the submission printed or an exit "
      "status the submission controls counts as a pass.")
def r_verdict_parse(pkg: Package):
    found: dict[str, list] = {}

    def add(k, e):
        found.setdefault(k, []).append(e)

    for path, pf, off, kind in _parser_sources(pkg):
        for ln in _soft_pass_sets(pf):
            if _line_in_used_code(pkg, pf, ln):
                add("soft", f"{path}:{ln + off}")
        for ln in _not_failed_checks(pf):
            if _line_in_used_code(pkg, pf, ln):
                add("notfailed", f"{path}:{ln + off}")
        for ln in _vacuous_pass(pf):
            if _line_in_used_code(pkg, pf, ln):
                add("vacuous", f"{path}:{ln + off}")
        for fn, ln in _printed_token_checks(pf, module_level=kind == "heredoc"):
            if kind == "env" and not re.search(r"(parse|log)", fn.name, re.I) and \
                    (pf.path, fn.start) not in _reward_path_keys(pkg):
                continue
            if _line_in_used_code(pkg, pf, ln):
                add("printed", f"{path}:{ln + off}")
        for ln in _judge_correct_substring(pf):
            if _line_in_used_code(pkg, pf, ln):
                add("judge", f"{path}:{ln + off}")
        # v0.5: score parsed from stdout, verdict from absence of FAILED, denominator from the candidate's run
        for key, fnc in (("score-stdout", _stdout_scores), ("nofail", _nofail_verdicts),
                         ("denominator", _candidate_denominators)):
            for fn, ln in fnc(pf, module_level=kind == "heredoc"):
                if kind == "env" and (pf.path, getattr(fn, "start", -1)) not in _reward_path_keys(pkg):
                    continue
                if _line_in_used_code(pkg, pf, ln):
                    add(key, f"{path}:{ln + off}")
    for e in _sh_stdout_scores(pkg):
        add("score-stdout", e)
    for e in _conftest_child_verdicts(pkg):
        add("conftest-child", e)
    tok, vac = _sh_verdicts(pkg)
    for e in tok:
        add("printed", e)
    for e in vac:
        add("vacuous", e)
    for sh_e, imp_e in _exit_status_verdicts(pkg):
        add("exit", sh_e)
        add("exit", imp_e)
    for e in _agent_skips(pkg):
        add("skip", e)
    if not found:
        return []
    res = []
    for k in ("exit", "printed", "soft", "skip", "vacuous", "notfailed", "judge"):
        if k in found:
            res.append(F("verdict-parse-semantics", _VERDICT_MSG[k] + ".", found[k], confidence="review",
                         severity="medium" if k in ("judge", "vacuous", "notfailed") else None))
    for k in ("score-stdout", "conftest-child", "nofail", "denominator"):
        if k in found:
            res.append(F("verdict-parse-semantics", _VERDICT_MSG_V05[k] + ".", found[k], confidence="review",
                         severity="medium" if k in ("nofail", "denominator") else None, variant="v05"))
    return res


# ---------------------------------------------------------------------------
# #13 existence-or-constant-check
# ---------------------------------------------------------------------------

_EXIST_CALLS = {"exists", "is_file", "is_dir", "isfile", "isdir", "lexists", "hasattr", "callable", "which",
                "find_spec", "is_symlink", "access", "isabs"}


def _existence_expr(e) -> bool:
    """An assert expression that only checks presence: path.exists(), hasattr, x is not None, len(x) > 0, bare x."""
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
        return _existence_expr(e.operand)
    if isinstance(e, ast.BoolOp):
        return all(_existence_expr(v) for v in e.values)
    if isinstance(e, ast.Call):
        fn = e.func.attr if isinstance(e.func, ast.Attribute) else (e.func.id if isinstance(e.func, ast.Name) else "")
        return fn in _EXIST_CALLS
    if isinstance(e, (ast.Name, ast.Attribute)):
        return True
    if isinstance(e, ast.Compare) and len(e.ops) == 1:
        op, r = e.ops[0], e.comparators[0]
        if isinstance(op, (ast.IsNot, ast.Is)) and isinstance(r, ast.Constant) and r.value is None:
            return True
        if isinstance(op, (ast.Gt, ast.GtE, ast.NotEq)) and isinstance(r, ast.Constant) and r.value in (0, 1) and \
                re.match(r"(len\(|os\.path\.getsize\(|\w+\.stat\(\)\.st_size|\w*size\w*)", ast.unparse(e.left)):
            return True
    return False


_BENIGN_CALLS = {"Path", "exists", "is_file", "is_dir", "isfile", "isdir", "lexists", "hasattr", "callable", "getattr",
                 "print", "len", "str", "join", "joinpath", "which", "find_spec", "getsize", "stat", "format",
                 "expanduser", "resolve", "absolute", "is_symlink", "access", "PurePath", "fspath", "dirname",
                 "basename", "abspath"}


def _existence_only_tests(pf: PyFile) -> int | None:
    """First line of a graded test file whose every assertion is a presence check and whose tests call nothing
    that could raise on a wrong answer. None if not."""
    if pf.tree is None:
        return None
    tests = [fi for fi in pf.funcs if fi.name.startswith("test") and fi.cls is None or
             fi.name.startswith("test") and fi.cls and fi.cls.startswith("Test")]
    if not tests:
        return None
    n_asserts = 0
    for fi in tests:
        for n in ast.walk(fi.node):
            if isinstance(n, ast.Assert):
                n_asserts += 1
                if not _existence_expr(n.test):
                    return None
            elif isinstance(n, ast.Call):
                fn = n.func.attr if isinstance(n.func, ast.Attribute) else (n.func.id if isinstance(n.func, ast.Name) else "")
                if fn in ("assertTrue", "assertIsNotNone", "assertFalse") and n.args and _existence_expr(n.args[0]):
                    n_asserts += 1
                    continue
                if fn not in _BENIGN_CALLS:
                    return None  # calls into a checker, reads content, runs something: more than presence
            elif isinstance(n, ast.Raise):
                return None
    if n_asserts == 0:
        return None
    return tests[0].start


_EXIST_HINT = re.compile(r"exists\(|is_file\(|isfile\(|isdir\(|is_dir\(|hasattr\(|callable\(|is not None|which\(|find_spec\(")
_SH_FILE_TEST = re.compile(r"\[\[?\s*!?\s*-(?:[efsdrx])\s+[^\]]+\]\]?|\btest\s+-[efsd]\s+\S+")


def _sh_existence_verdict(pkg: Package):
    out = []
    for p, t in _graded_sh(pkg):
        if "reward" not in t:
            continue
        for m in re.finditer(r"^\s*(?:if|elif)\s+(.+?);?\s*then\s*$", t, re.M):
            cond = m.group(1)
            if not _SH_FILE_TEST.search(cond):
                continue
            rest = _SH_FILE_TEST.sub("", cond)
            rest = re.sub(r"(&&|\|\||-a\b|-o\b|\s|;)", "", rest)
            if rest:
                continue  # some other check is part of the condition
            body = t[m.end(): m.end() + 300]
            body = re.split(r"^\s*(else|elif|fi)\b", body, maxsplit=1, flags=re.M)[0]
            if re.search(r"^\s*(if|case|while|grep|cmp|diff|python|\[)\b|&&|\|\|", body, re.M):
                continue
            if re.search(r"(echo|printf)\s+['\"]?1(\.0)?['\"]?\s*>+\s*\S*reward|reward=1\b|\"reward\":\s*1", body) and \
                    "!" not in cond.split("-")[0]:
                out.append(ev(p, line_of(t, m.start())))
    return out


_TRUTHY_RET = re.compile(r"^(float|int|bool)\(\s*(bool\()?")
_TRUTHY_HINT = (("return float(", "return int(", "return 1.0 if", "return 1 if"),
                re.compile(r"return\s+(float|int)\(\s*(bool\(|os\.path\.|Path\(|\w+\.exists\(|\w+\s*\)|len\()|"
                           r"return\s+1(\.0)?\s+if\s+[^\n]*\belse\s+0"), 0)


def _registered(pkg: Package, fi) -> bool:
    """Cheap test that a function is handed to a rubric (funcs=[...], add_reward_func, a @reward decorator)."""
    if any(re.search(r"\breward\b", d) for d in fi.decorators):
        return True
    rx = re.compile(rf"(funcs|add_reward_func|Rubric|reward_functions|rewards)\b[^\n]*\b{re.escape(fi.name)}\b|"
                    rf"^\s*{re.escape(fi.name)}\s*,\s*$")
    for pf in pkg.py_all():
        if fi.name in pf.text and rx.search(pf.text):
            return True
    return False


def _truthiness_rewards(pkg: Package):
    """Reward functions handed to a rubric that pay for existence or non emptiness and never look at the answer."""
    out = []
    seen = set()
    for pf in pkg.py_all():
        lines = _hint_lines(pf.text, _TRUTHY_HINT)
        if not lines:
            continue
        for fi in _units(pf, lines):
            if isinstance(fi, _ModuleBody) or fi.name in seen:
                continue
            if re.search(r"(format|length|len_|tag|think|style|xml|struct|parse|brevity|concise|valid|syntax|compil|"
                         r"tool|call|turn|step|efficien|cost|penalty|lang|json|boxed|wellformed)", fi.name, re.I):
                continue
            if _uses_answer(fi) or re.search(r"(state\[|state\.get|info\[|info\.get|\banswer\b|\bground_truth\b|"
                                             r"\bexpected\w*|\breference\b|\btarget\b|\bgold\w*|judge|verify|compare|"
                                             r"==|!=|\bin\s+\w*(answer|gold|expected))", fi.src):
                continue
            rets = [n for n in ast.walk(fi.node) if isinstance(n, ast.Return) and n.value is not None]
            if not rets:
                continue
            kinds = set()
            for r in rets:
                v = r.value
                s_ = ast.unparse(v)
                if isinstance(v, ast.Constant):
                    kinds.add("const")
                elif isinstance(v, ast.IfExp) and isinstance(v.body, ast.Constant) and isinstance(v.orelse, ast.Constant):
                    kinds.add("exist" if _existence_expr(v.test) else "other")
                elif _TRUTHY_RET.match(s_) and isinstance(v, ast.Call) and v.args and (
                        _existence_expr(v.args[0]) or isinstance(v.args[0], ast.Call) and
                        getattr(v.args[0].func, "id", "") == "bool"):
                    kinds.add("exist")
                else:
                    kinds.add("other")
            if not (kinds <= {"const", "exist"} and "exist" in kinds):
                continue
            # the test must be about the model output (completion, parsed answer, a file the agent wrote)
            tests = " ".join(ast.unparse(r.value) for r in rets)
            if not re.search(r"(completion|response|answer|parsed|output|content|text|exists|isfile|path|file|code)", tests, re.I):
                continue
            if _registered(pkg, fi):
                out.append(ev(pf, fi.start))
                seen.add(fi.name)
    return out


# Evidence: row 13, arxiv:2608.05148 App G (a Reasoning Gym scorer gives full credit to any nonempty answer);
# arxiv:2605.25624 App E.3 (existence checks); url:rdi.berkeley.edu/blog/trustworthy-benchmarks (existence-only tests).
@rule("existence-or-constant-check", "hack", "medium",
      "The reward or the graded tests only check that a file, symbol or answer exists or is non empty, not that "
      "it is right.")
def r_existence_check(pkg: Package):
    res = []
    tests = []
    for path, text in _graded_test_files(pkg):
        if not os.path.basename(path).startswith(("test", "check")) or not _EXIST_HINT.search(text) or \
                re.search(r"^\s*assert\b[^\n]*(==|!=|\bin\b|<|>=?\s*[2-9])", text, re.M):
            continue
        pf = _pyf(pkg, path)
        ln = _existence_only_tests(pf)
        if ln is not None:
            tests.append(ev(pf, ln))
    if tests:
        res.append(F("existence-or-constant-check", "Every assertion in the graded test file checks only that a file "
                     "or symbol exists, so an empty placeholder passes.", tests, confidence="review"))
    sh = _sh_existence_verdict(pkg)
    if sh:
        res.append(F("existence-or-constant-check", "The test script pays reward 1 when the output file exists, "
                     "without looking at its contents.", sh, confidence="review"))
    rw = _truthiness_rewards(pkg)
    if rw:
        res.append(F("existence-or-constant-check", "A reward function pays for a non empty or existing output and "
                     "never compares it with the answer.", rw, confidence="review"))
    # v0.5 widenings
    flagged = {e.rsplit(":", 1)[0] for e in tests}
    whole, single = _vacuous_tests(pkg, flagged)
    if whole:
        res.append(F("existence-or-constant-check", "No test in the graded test file can fail, because each one has no "
                     "real check or its check is always true, swallowed, skipped or only printed.", whole,
                     confidence="review"))
    if single:
        res.append(F("existence-or-constant-check", "A graded test can never fail, because its check is always true, "
                     "swallowed by an except, skipped or only printed, so the requirement it covers is not enforced.",
                     single, confidence="review", severity="low"))
    flagged |= {e.rsplit(":", 1)[0] for e in whole}
    grep = _grep_source_tests(pkg, flagged) + _sh_grep_source(pkg)
    if grep:
        res.append(F("existence-or-constant-check", "The graded tests only search the agent's source text for "
                     "keywords and never run the code, so a file with the right words passes without working.", grep,
                     confidence="review"))
    role = _role_only_rewards(pkg)
    if role:
        res.append(F("existence-or-constant-check", "A reward function checks only that the last message came from the "
                     "assistant, not what it says, so any reply scores.", role, confidence="review"))
    lit = _metric_literal_tests(pkg)
    if lit:
        res.append(F("existence-or-constant-check", "A graded test checks that a metric in a file the agent writes "
                     "equals a number the instruction shows, so writing that number passes without the work.", lit,
                     confidence="review"))
    return res


# ---------------------------------------------------------------------------
# #22 equality-on-model-objects
# ---------------------------------------------------------------------------

_SAFE_WRAP = {"str", "repr", "float", "int", "bool", "round", "json.dumps", "dumps", "hash", "len", "abs", "ord",
              "chr", "bytes", "complex", "format", "type", "id", "math.isclose", "isclose", "np.allclose", "allclose",
              "np.array_equal", "array_equal", "assert_allclose", "assert_array_equal", "approx", "pytest.approx",
              "Decimal", "Fraction", "loads", "json.loads", "normalize", "canonical", "serialize", "to_json", "tolist"}


_MOD_SOURCE = re.compile(r"(module_from_spec|import_module|load_source|run_path|SourceFileLoader|__import__)")


def _submission_names(pf: PyFile, imp_lines) -> set:
    """Module level names bound to the submission module or to things imported from it."""
    names = set()
    first = imp_lines[0] if imp_lines else 0
    for n in ast.walk(pf.tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if _MOD_SOURCE.search(ast.unparse(n.value)):
                names.add(n.targets[0].id)
        elif isinstance(n, ast.ImportFrom) and n.lineno >= first and n.module and \
                n.module.split(".")[0] not in _STDLIB_TOP and n.module.split(".")[0] not in ("pytest", "numpy", "pandas"):
            names |= {a.asname or a.name for a in n.names if a.name != "*"}
        elif isinstance(n, ast.Import) and n.lineno >= first:
            for a in n.names:
                top = a.name.split(".")[0]
                if top not in _STDLIB_TOP and top not in ("pytest", "numpy", "pandas", "torch", "scipy"):
                    names.add((a.asname or a.name).split(".")[0])
    # helpers that return the loaded module (def load(): ... return mod)
    for fi in pf.funcs:
        if _MOD_SOURCE.search(fi.src) and re.search(r"\breturn\s+\w+", fi.src):
            names.add(fi.name + "()")
    return names


def _derives(node, derived: set) -> bool:
    """True if the expression is (a call on / an attribute of / a call of) a submission derived value."""
    if isinstance(node, ast.Name):
        return node.id in derived
    if isinstance(node, ast.Attribute):
        return _derives(node.value, derived)
    if isinstance(node, ast.Subscript):
        return _derives(node.value, derived)
    if isinstance(node, ast.Await):
        return _derives(node.value, derived)
    if isinstance(node, ast.Call):
        fn = ast.unparse(node.func)
        if fn in _SAFE_WRAP or fn.split(".")[-1] in _SAFE_WRAP:
            return False
        if fn + "()" in derived:
            return True
        if fn == "getattr" and node.args:
            return _derives(node.args[0], derived)
        if fn in ("list", "tuple", "sorted", "set", "dict", "next", "iter") and node.args:
            return _derives(node.args[0], derived)
        return _derives(node.func, derived)
    return False


def _type_checked(src: str, v: str) -> bool:
    ev_ = re.escape(v)
    return bool(re.search(rf"(assert\s+isinstance\(\s*{ev_}\b|if\s+not\s+isinstance\(\s*{ev_}\b|"
                          rf"type\(\s*{ev_}\s*\)\s*(is|==|!=|is\s+not)\s|\s(is|==)\s+type\(\s*{ev_}\s*\)|"
                          rf"{ev_}\s*=\s*(json\.loads|str|repr|float|int|bool|round)\()", src))


def _eq_on_submission(pf: PyFile, imp_lines) -> list[int]:
    """sub_value == expected (in an assert, an if, a return or assertEqual) where sub_value comes from calling into
    the imported submission and is never type checked or serialized."""
    if pf.tree is None:
        return []
    base = _submission_names(pf, imp_lines)
    if not base:
        return []
    out = []
    units = list(pf.funcs) + [_ModuleBody(pf)]
    for fi in units:
        derived = set(base)
        assigns = [n for n in ast.walk(fi.node) if isinstance(n, ast.Assign)]
        for _ in range(4):  # propagate module -> instance -> bound method -> result
            grew = False
            for n in assigns:
                if _derives(n.value, derived):
                    for t_ in n.targets:
                        if isinstance(t_, ast.Name) and t_.id not in derived:
                            derived.add(t_.id)
                            grew = True
            if not grew:
                break
        src = fi.src
        for n in ast.walk(fi.node):
            pair = None
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Eq, ast.NotEq)):
                pair = (n.left, n.comparators[0])
                if any(isinstance(x, ast.Constant) and x.value in (None, True, False) for x in pair):
                    continue
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and \
                    n.func.attr in ("assertEqual", "assertEquals", "assertListEqual", "assertDictEqual", "assertTupleEqual") \
                    and len(n.args) >= 2:
                pair = (n.args[0], n.args[1])
            if pair is None:
                continue
            for side in pair:
                if _derives(side, derived) and not isinstance(side, ast.Attribute):
                    v = ast.unparse(side)
                    if _type_checked(src, v):
                        continue
                    out.append(n.lineno)
                    break
    return sorted(set(out))


# Evidence: row 22, arxiv:2511.18397 Sec 2 (AlwaysEqual, one of three systemic hacks in production coding
# environments); arxiv:2606.09711; arxiv:2609.39533 App G.
@rule("equality-on-model-objects", "hack", "high",
      "Graded tests compare a value the submission returned with == and never check its type or serialize it, so "
      "an object whose __eq__ always returns True passes.")
def r_equality_on_model_objects(pkg: Package):
    evid = []
    for path, text in _graded_test_files(pkg):
        if not re.search(r"==|assertEqual", text) or not (SUBMISSION_IMPORT.search(text) or "sys.path" in text):
            continue
        pf = _pyf(pkg, path)
        task = _task_of(pkg, pf.path)
        dirs = task.agent_dirs() if task and not task.separate else AGENT_DIRS
        if task and task.separate:
            dirs = dirs + tuple(task.agent_dirs())
        imp = _imports_submission(pf, dirs)
        if not imp:
            continue
        lines = _eq_on_submission(pf, imp)
        evid += [ev(pf, ln) for ln in lines[:3]]
    if not evid:
        return []
    return [F("equality-on-model-objects", "Graded tests compare what the imported submission returned using == with "
              "no type check, so a returned object whose __eq__ always answers True passes.", evid, confidence="review")]


# ---------------------------------------------------------------------------
# v0.5 additions. Each block below widens one of the four rules above. The rule bodies call them.
# ---------------------------------------------------------------------------

_CODE_EXT = re.compile(r"\.(py|pyx|c|cc|cpp|cxx|h|hpp|js|mjs|ts|tsx|jsx|rs|go|java|kt|rb|php|pl|lua|swift|scala|cs|sol|ml|hs|"
                       r"r|R|jl|ex|exs|clj|dart|zig|nim|v|sv|vhd|asm|s|cob|cbl|f90|f|m)$")
_TOKEN_SAFE = re.compile(r"(uuid|secrets\.token|nonce|sentinel|junit|\.xml\b|ElementTree|ctrf|json_report|report\.json)")


def _task_text(pkg: Package, task, rel: str) -> str:
    if task is None:
        return ""
    return pkg.files.get(f"{task.root}/{rel}" if task.root else rel, "")


def _task_sh_texts(pkg: Package, task) -> str:
    return "\n".join(t for p, t in _graded_sh(pkg) if _task_of(pkg, p) is task)


def _sorted_nodes(node, types):
    xs = [n for n in ast.walk(node) if isinstance(n, types)]
    return sorted(xs, key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))


def _call_name(n) -> str:
    f = n.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _names_in(node) -> set:
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}


# ---- item 1 (harness-fn-in-model-namespace), widens grader-import-surface ---------------------------------
# Canon: harness-fn-in-model-namespace. Main papers: arxiv:2603.11337 (evaluator tampering), rl-rewardhacking-ext
# (overwrite-tests loophole), arxiv:2605.15257.

_HARNESS_NAMES = ("run_tests", "run_test", "verify_solution", "check_solution", "test_solution", "evaluate_solution",
                  "run_checks", "check", "verify", "test")
_HARNESS_CALL = re.compile(r"(?<![\w.])(" + "|".join(_HARNESS_NAMES) + r")\s*\(")
_HARNESS_GATE = re.compile(r"(" + "|".join(_HARNESS_NAMES) + r")\s*\(")
_HARNESS_LIT = re.compile(r"(\\n|['\"])\s*(" + "|".join(_HARNESS_NAMES) + r")\s*\(")
_CODEISH = re.compile(r"(code|completion|response|solution|program|answer|parsed|generated|submission|candidate|snippet)", re.I)
_NOT_CODEISH = re.compile(r"(test|harness|check|assert|verif|prefix|header|setup|import|preamble|footer|suffix|driver|stub)", re.I)


def _concat_parts(n):
    """A string built by +, an f-string or '\\n'.join([...]) -> list of ('lit', text) / ('expr', source), else None."""
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
        out = []
        for side in (n.left, n.right):
            sub = _concat_parts(side)
            if sub is None:
                s = _str_of(side)
                if s is not None:
                    sub = [("lit", s)]
                else:
                    try:
                        sub = [("expr", ast.unparse(side))]
                    except Exception:
                        return None
            out += sub
        return out
    if isinstance(n, ast.JoinedStr):
        out = []
        for v in n.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(("lit", v.value))
            elif isinstance(v, ast.FormattedValue):
                try:
                    out.append(("expr", ast.unparse(v.value)))
                except Exception:
                    return None
        return out
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "join" and \
            _str_of(n.func.value) is not None and len(n.args) == 1 and isinstance(n.args[0], (ast.List, ast.Tuple)):
        out = []
        for e in n.args[0].elts:
            s = _str_of(e)
            if s is not None:
                out.append(("lit", s))
            else:
                try:
                    out.append(("expr", ast.unparse(e)))
                except Exception:
                    return None
        return out
    return None


def _is_code_part(src: str) -> bool:
    last = re.split(r"[.\[\]()'\"]+", src)
    return bool(_CODEISH.search(src)) and not _NOT_CODEISH.search(src) and len(src) < 80 and bool(last)


def _harness_in_completion(pkg: Package):
    """The grader runs the model's code followed by a call to a harness function (run_tests(), check(...)) that the
    grader never defines after the model's code, so the model's own definition of that function is what runs."""
    hits = []
    for pf in pkg.py_all():
        if pf.tree is None or not (_HARNESS_GATE.search(pf.text) or pf.text.count("exec(") >= 2):
            continue
        if not re.search(r"(exec\(|subprocess|sandbox|run_code|execute|\.run\()", pf.text):
            continue
        for fi in pf.funcs:
            src = fi.src
            if not (_HARNESS_LIT.search(src) or src.count("exec(") >= 2) or not re.search(r"(exec\(|subprocess|sandbox|run_code|execute|\.run\()", src):
                continue
            for n in ast.walk(fi.node):
                parts = _concat_parts(n)
                if not parts or len(parts) < 2:
                    continue
                idx = [i for i, (k, s) in enumerate(parts) if k == "expr" and _is_code_part(s)]
                if not idx:
                    continue
                i = idx[-1]
                tail = parts[i + 1:]
                if not tail or any(k == "expr" for k, _ in tail):
                    continue  # something unknown follows the code and may define the harness
                tail_txt = "".join(s for _, s in tail)
                calls = {m.group(1) for m in _HARNESS_CALL.finditer(tail_txt)}
                calls = {c for c in calls if not re.search(rf"\bdef\s+{c}\s*\(", tail_txt)}
                if not calls:
                    continue
                if re.search(r"(assert|==|raise)", tail_txt):
                    continue  # the tail carries its own checks
                hits.append(("harness-in-completion", ev(pf, n.lineno), None))
                break
            # exec(harness, ns) then exec(code, ns) then ns["name"](): the model's code redefines the harness
            execs = [c for c in _sorted_nodes(fi.node, ast.Call) if _call_name(c) == "exec" and len(c.args) >= 2]
            for a_i, a in enumerate(execs):
                try:
                    a0, ns = ast.unparse(a.args[0]), ast.unparse(a.args[1])
                except Exception:
                    continue
                if not _NOT_CODEISH.search(a0) or _is_code_part(a0):
                    continue
                for b in execs[a_i + 1:]:
                    try:
                        b0, ns2 = ast.unparse(b.args[0]), ast.unparse(b.args[1])
                    except Exception:
                        continue
                    if ns2 != ns or not _is_code_part(b0):
                        continue
                    later = src[src.find(b0) + len(b0):] if b0 in src else ""
                    m = re.search(rf"{re.escape(ns)}\s*(\[\s*['\"](\w+)['\"]\s*\]|\.get\(\s*['\"](\w+)['\"]\s*\))\s*\(", later)
                    if m and re.search(r"(test|check|verif|harness|run)", m.group(2) or m.group(3) or "", re.I):
                        hits.append(("harness-in-completion", ev(pf, b.lineno), None))
                    break
    return hits


_EVALUATORISH = re.compile(r"(eval|metric|scor|grad|check|verif|judge|oracle|refer|baseline|bench|truth|expected|"
                           r"harness|reward|test)", re.I)
_EDIT_VERB = re.compile(r"(implement|modify|edit|fix|complete|write|update|optimi|rewrite|improve|change|create|"
                        r"refactor|speed|replace|port|translate|debug|extend|add)", re.I)


def _agent_tree_imports(pf: PyFile, dirs) -> list:
    """(module, line) for modules a graded test loads from the agent's tree."""
    out = []
    text = pf.text
    ins = [m for m in re.finditer(r"sys\.path\.(insert|append)\(\s*(?:0\s*,\s*)?(?:str\()?\s*(?:Path\()?\s*['\"]([^'\"]+)['\"]", text)
           if _under(m.group(2), dirs)]
    if ins:
        first = line_of(text, ins[0].start())
        for n in ast.walk(pf.tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno > first:
                mods = [n.module] if isinstance(n, ast.ImportFrom) else [a.name for a in n.names]
                for mod in mods:
                    top = (mod or "").split(".")[0]
                    if top and top not in _STDLIB_TOP:
                        out.append((top, n.lineno))
            elif isinstance(n, ast.Call) and _call_name(n) == "import_module" and n.args and n.lineno > first:
                s = _str_of(n.args[0])
                if s:
                    out.append((s.split(".")[0], n.lineno))
    for n in ast.walk(pf.tree):
        if isinstance(n, ast.Call) and _call_name(n) == "spec_from_file_location" and len(n.args) >= 2:
            s = _resolve_str(n.args[1], pf.consts)
            if s and _under(s, dirs) and s.endswith(".py"):
                out.append((os.path.basename(s)[:-3], n.lineno))
    return out


def _evaluator_in_tree(pkg: Package):
    hits = []
    for path, text in _graded_test_files(pkg):
        if "sys.path" not in text and "spec_from_file_location" not in text:
            continue
        task = _task_of(pkg, path)
        if task is None:
            continue
        pf = _pyf(pkg, path)
        if pf.tree is None:
            continue
        root = f"{task.root}/" if task.root else ""
        shipped = {os.path.basename(p)[:-3] for p in pkg.files if p.startswith(root + "environment/") and p.endswith(".py")}
        if not shipped:
            continue
        instr = _task_text(pkg, task, "instruction.md")
        sol = "\n".join(t for p, t in pkg.files.items() if p.startswith(root + "solution/"))
        sh = _task_sh_texts(pkg, task)
        for mod, ln in _agent_tree_imports(pf, task.agent_dirs()):
            if mod not in shipped or not _EVALUATORISH.search(mod):
                continue
            esc = re.escape(mod)
            if re.search(rf"(open\([^)\n]*{esc}\.py[^)\n]*['\"]w|cp\s+[^\n]*{esc}\.py|copy\w*\([^)\n]*{esc}\.py|"
                         rf"cat\s*>\s*\S*{esc}\.py|{esc}\.py\S*\s*\)\.write_text)", text + "\n" + sh):
                continue  # the grader restores the file before it imports it
            if re.search(r"(hashlib|sha256|md5|checksum)", text + sh) and mod in (text + sh):
                continue
            if any(_EDIT_VERB.search(s) for s in re.split(r"(?<=[.!?\n])\s", instr) if f"{mod}.py" in s or
                   re.search(rf"\b{esc}\b", s) and "`" in s):
                continue  # the task asks the agent to work on this file: it is the submission
            if re.search(rf"{esc}\.py", sol):
                continue  # the oracle solution edits it: it is the submission
            hits.append(("evaluator-in-tree", ev(pf, ln), task))
            break
    return hits


# ---- item 6 (startup-hook-files + agent-writable-interpreter-or-binaries), widens grader-import-surface -----
# Canon: startup-hook-files, agent-writable-interpreter-or-binaries. Main papers: arxiv:2605.12673 (BenchJack),
# arxiv:2610.12822, arxiv:2609.22978 (overwrite of /bin/bash).

_VENV_PATH = re.compile(r"(\S*/(\.?venv|env|\.env|virtualenv)\w*)/bin/(python\S*|pytest|py\.test|pip\S*)$")
_FRESH_VENV = re.compile(r"(uv\s+venv|python\S*\s+-m\s+venv|virtualenv\s|uvx\s|uv\s+run\s+[^\n]*--(isolated|no-project)|"
                         r"conda\s+create|pip\s+install\s+[^\n]*--target)")
_PY_LIB_DIR = re.compile(r"(/usr/(local/)?lib/python[\d.]*|site-packages|dist-packages|/usr/(local/)?bin\b|/bin\b|"
                         r"/opt/conda|/opt/venv|/usr/local/lib\b|__pycache__)")


def _dockerfile_user(df: str) -> str | None:
    us = re.findall(r"^\s*USER\s+(\S+)", df, re.M | re.I)
    return us[-1] if us else None


def _startup_hooks_sh(pkg: Package):
    hits = []
    for p, t in _graded_sh(pkg):
        if not re.search(r"pytest|python|uv\s+run", t):
            continue
        task = _task_of(pkg, p)
        if task is not None and task.separate and not task.artifacts:
            continue
        dirs = task.agent_dirs() if task else AGENT_DIRS
        start = task.workdir if task and task.workdir and not task.separate else None
        cmds, _docs = _sh_cached(pkg, p, t, start)
        activated = None
        made = set()
        for c in cmds:
            if (c.argv[:2] == ["uv", "venv"] or "venv" in c.argv[:3] and _PY_EXE.match(c.argv[0]) or c.argv[0] == "virtualenv"):
                for a in c.argv[1:]:
                    if not a.startswith("-") and a not in ("venv", "-m"):
                        made.add(a if a.startswith("/") else (os.path.normpath(os.path.join(c.cwd, a)) if c.cwd else a))
                continue
            if c.argv[0] in ("source", ".") and len(c.argv) > 1 and c.argv[1].endswith("/activate"):
                a = c.argv[1] if c.argv[1].startswith("/") else (os.path.normpath(os.path.join(c.cwd, c.argv[1])) if c.cwd else "")
                activated = a if _under(a, dirs) and not any(a.startswith(m.rstrip("/") + "/") for m in made) else None
                continue
            runs_py = _pytest_args(c.argv) is not None or _python_launch(c.argv) is not None
            exe = c.argv[0]
            m = _VENV_PATH.match(exe)
            if m:
                vp = exe if exe.startswith("/") else (os.path.normpath(os.path.join(c.cwd, exe)) if c.cwd else "")
                if _under(vp, dirs) and not any(vp.startswith(m.rstrip("/") + "/") for m in made):
                    hits.append(("venv-in-tree", ev(p, c.line), task))
                    continue
            if activated and runs_py and not _VENV_PATH.match(exe):
                hits.append(("venv-in-tree", ev(p, c.line), task))
                continue
            # uv run from the agent's project: uv syncs and uses that project's .venv
            if exe == "uv" and len(c.argv) > 1 and c.argv[1] == "run" and _under(c.cwd, dirs) and not made and \
                    not re.search(r"--(isolated|no-project|script|project)\b|--directory\b", " ".join(c.argv)):
                root = f"{task.root}/" if task and task.root else ""
                df = pkg.files.get(f"{root}environment/Dockerfile", "")
                if task and (f"{root}environment/pyproject.toml" in pkg.files or re.search(r"\buv\s+(sync|venv)\b", df)):
                    hits.append(("venv-in-tree", ev(p, c.line), task))
        # a non-root agent that the image lets write the interpreter, site-packages or PATH binaries
        if task is not None and not task.separate:
            root = f"{task.root}/" if task.root else ""
            df = pkg.files.get(f"{root}environment/Dockerfile", "")
            user = _dockerfile_user(df)
            nonroot = task.nonroot_agent or (user is not None and user not in ("root", "0", "0:0"))
            if nonroot and not _FRESH_VENV.search(t):
                for m in re.finditer(r"^[^\n#]*\b(chmod\s+(-R\s+)?(777|[0-7]?7[0-7]{2}|a\+w\w*|o\+w\w*|ugo\+w\w*|\+w)|"
                                     r"chown\s+(-R\s+)?\S+)\s+([^\n&;|]+)", df, re.M):
                    args = m.group(5)
                    if _PY_LIB_DIR.search(args):
                        hits.append(("writable-interpreter", ev(f"{root}environment/Dockerfile", line_of(df, m.start())), task))
                        hits.append(("writable-interpreter", ev(p, 1), task))
                        break
    return hits


def _patch_filter_misses_hooks(pkg: Package):
    """A filter that strips conftest.py from a model patch but not sitecustomize.py, usercustomize.py or .pth files."""
    hits = []
    for pf in pkg.py_all():
        t = pf.text
        if "conftest" not in t or not re.search(r"(patch|diff)", t) or pf.tree is None:
            continue
        if re.search(r"(sitecustomize|usercustomize|\.pth\b)", t):
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, (ast.List, ast.Tuple, ast.Set)) and 1 <= len(n.elts) <= 40:
                strs = [_str_of(e) for e in n.elts]
                if any(s and "conftest" in s for s in strs) and sum(1 for s in strs if s) >= len(strs) - 1:
                    fn = _enclosing_name(pf, n.lineno) or ""
                    ctx = "\n".join(pf.lines[max(0, n.lineno - 6):n.lineno + 6])
                    if re.search(r"(exclud|ignor|skip|filter|strip|remov|block|forbid|deny|protect|reject)", fn + ctx, re.I) and \
                            re.search(r"(patch|diff)", fn + ctx, re.I):
                        hits.append(("startup-hook-filter", ev(pf, n.lineno), None))
                        break
    return hits


_IMPORT_MSG_V05 = {
    "harness-in-completion": "The grader appends a call to a test harness function after the model's code but never "
                             "defines that function after it, so the model can define the harness itself and pass",
    "evaluator-in-tree": "Graded tests import an evaluator or reference module that the task ships into the agent's "
                         "working tree, so the agent can edit that module to pass",
    "venv-in-tree": "The grader runs Python from a virtual environment inside the agent's working tree, so a .pth "
                    "file or sitecustomize.py the agent adds there runs inside the grader",
    "writable-interpreter": "The image lets the non-root agent write the Python install or PATH binaries and the "
                            "test script uses that interpreter, so a planted sitecustomize.py, .pth file or binary "
                            "runs inside the grader",
    "startup-hook-filter": "The patch filter strips conftest.py but not sitecustomize.py, usercustomize.py or .pth "
                           "files, so a patch can add a startup hook that runs before the tests",
}


def _import_surface_v05(pkg: Package):
    hits = _harness_in_completion(pkg) + _evaluator_in_tree(pkg) + _startup_hooks_sh(pkg) + _patch_filter_misses_hooks(pkg)
    if not hits:
        return []
    res = []
    for k in ("harness-in-completion", "evaluator-in-tree", "venv-in-tree", "writable-interpreter", "startup-hook-filter"):
        evid = [e for kk, e, _t in hits if kk == k]
        if evid:
            res.append(F("grader-import-surface", _IMPORT_MSG_V05[k] + ".", evid, confidence="review"))
    return res


# ---- items 3, 4 and 8, widen verdict-parse-semantics ------------------------------------------------------

_OUT_ATTR = re.compile(r"(stdout|stderr|output|out\b|text|strip|split|splitlines|decode|lines|\[-1\]|search|match|findall|"
                       r"group|read\(|readlines|getvalue|logs?\b|result)", re.I)
_SUBMISSIONISH = re.compile(r"(code|completion|solution|program|generated|submission|candidate|"
                            r"pytest|python|node|bash|\bsh\b|\./|/app|/workspace|/testbed|/repo|/workdir|run_tests|"
                            r"test_cmd|cmd|command)", re.I)
_GRADER_SCRIPT = re.compile(r"(verif|VERIFY|grade|GRADE|grader|scor|SCOR|judge|JUDGE|eval|EVAL|check|CHECK|reward|REWARD)")
_SCORE_KEY = re.compile(r"(score|reward|metric|accuracy|acc\b|result|f1|value|points|fitness|speedup|perf)", re.I)


def _output_names(unit, params_as_output=False):
    """Names that hold output of a process run in the unit (or of a log parameter), with simple propagation."""
    names = set()
    if params_as_output:
        names |= {p_ for p_ in getattr(unit, "params", []) if re.search(r"(log|output|stdout|text|content|out\b)", p_, re.I)}
    runs = set()
    for n in _sorted_nodes(unit.node, (ast.Assign, ast.AnnAssign)):
        v = n.value
        if v is None:
            continue
        if isinstance(v, ast.Await):
            v = v.value
        try:
            vs = ast.unparse(v)
        except Exception:
            continue
        tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
        tnames = set()
        for t_ in tgts:
            tnames |= {x.id for x in ast.walk(t_) if isinstance(x, ast.Name)}
        if _RUN_CALL.search(vs) and _SUBMISSIONISH.search(vs) and \
                not (_GRADER_SCRIPT.search(vs) and not re.search(r"(/app|/workspace|/testbed|/repo|/workdir)\b", vs)):
            names |= tnames
            runs |= tnames
        elif _names_in(v) & names and _OUT_ATTR.search(vs):
            names |= tnames
    return names, runs


def _stdout_scores(pf: PyFile, module_level=False):
    """float(...) of a number parsed from output of a process that ran the submission, flowing to a return or reward."""
    out = []
    if pf.tree is None or not re.search(r"\b(float|Decimal|loads)\(", pf.text) or not _RUN_CALL.search(pf.text):
        return out
    lines = _hint_lines(pf.text, (("float(", "int(", "loads("), re.compile(r"(float|int|loads)\("), 0))
    for unit in _units(pf, lines, whole_module=True):
        if isinstance(unit, _ModuleBody) and not module_level:
            continue
        src = unit.src
        if not _RUN_CALL.search(src) or _TOKEN_SAFE.search(src):
            continue
        names, runs = _output_names(unit)
        if not runs:
            continue
        returned = set()
        for r in ast.walk(unit.node):
            if isinstance(r, ast.Return) and r.value is not None:
                returned |= _names_in(r.value)
        rewardish_assign = set()
        for a in ast.walk(unit.node):
            if isinstance(a, ast.Assign):
                try:
                    ts = " ".join(ast.unparse(t_) for t_ in a.targets)
                except Exception:
                    continue
                if re.search(r"(reward|score|metric|state\[)", ts, re.I):
                    rewardish_assign |= _names_in(a.value)
        par = _parents(unit.node)
        for n in ast.walk(unit.node):
            if not isinstance(n, ast.Call):
                continue
            nm = _call_name(n)
            target = None
            if nm in ("float", "Decimal", "Fraction") and n.args:
                target = n.args[0]
            elif nm == "loads" and n.args:
                p_ = par.get(n)
                if isinstance(p_, ast.Subscript) and isinstance(p_.slice, ast.Constant) and \
                        isinstance(p_.slice.value, str) and _SCORE_KEY.search(p_.slice.value):
                    target = n.args[0]
                    n = p_
            if target is None:
                continue
            try:
                ts = ast.unparse(target)
            except Exception:
                continue
            if not (_names_in(target) & names):
                continue
            if re.search(r"(pass|PASS|fail|FAIL|returncode|exit_code|time|elapsed|duration|seconds|memory)", ts):
                continue  # pass counts are the printed-token case, timings are kernel-perf-grader's
            # the parsed number must reach the return or a reward variable
            p_, assigned = par.get(n), set()
            for _ in range(6):
                if p_ is None:
                    break
                if isinstance(p_, ast.Return):
                    assigned.add("__ret__")
                    break
                if isinstance(p_, ast.Assign):
                    for t_ in p_.targets:
                        assigned |= {x.id for x in ast.walk(t_) if isinstance(x, ast.Name)}
                        try:
                            if re.search(r"(reward|score|metric|state\[)", ast.unparse(t_), re.I):
                                assigned.add("__ret__")
                        except Exception:
                            pass
                    break
                p_ = par.get(p_)
            if "__ret__" in assigned or assigned & (returned | rewardish_assign):
                out.append((unit, n.lineno))
    return out


def _sh_stdout_scores(pkg: Package):
    """score=$(python /app/run.py | tail -1) written to the reward file, or run ... | tail -1 > reward.txt."""
    out = []
    for p, t in _graded_sh(pkg):
        if "reward" not in t:
            continue
        for m in re.finditer(r"^\s*(?:export\s+|local\s+)?(\w+)=\$\((.*)\)\s*$", t, re.M):
            var, inner = m.group(1), m.group(2)
            first = inner.split("|")[0]
            toks = _tokens(first)
            if not toks:
                continue
            c = _Cmd(0, toks, None, {}, {}, None, False, first, None)
            pl = _python_launch(toks)
            runs = _runs_submission(c) or (pl and pl[1] == "script" and pl[2] and re.search(r"^(/app|/workspace|/testbed|/repo|/code|\./)", pl[2]))
            if not runs or re.search(r"(pass|fail|PASS|FAIL|wc\s+-l|returncode)", inner):
                continue
            rest = t[m.end():]
            if re.search(rf"(echo|printf)\s+[^\n]*\$\{{?{var}\}}?[^\n]*>+\s*\S*reward", rest) or \
                    re.search(rf"['\"]?reward['\"]?\s*:\s*\\?\"?\$\{{?{var}\}}?", rest):
                out.append(ev(p, line_of(t, m.start())))
        for m in re.finditer(r"^[^\n#]*\|\s*tail\s+-(n\s*)?1\s*>+\s*\S*reward\S*", t, re.M):
            seg = m.group(0).split("|")[0]
            toks = _tokens(seg)
            if toks and _runs_submission(_Cmd(0, toks, None, {}, {}, None, False, seg, None)):
                out.append(ev(p, line_of(t, m.start())))
    return out


def _nofail_verdicts(pf: PyFile, module_level=False):
    """The verdict is only that FAILED does not appear in the output of the test run."""
    out = []
    if pf.tree is None or not re.search(r"['\"](FAILED|FAIL|failed)['\"]", pf.text):
        return out
    lines = _hint_lines(pf.text, (("FAIL", "failed"), re.compile(r"['\"](FAILED|FAIL|failed)['\"]\s*(not\s+in|\)\s*==\s*0)|"
                                                               r"not\s+re\.search\(\s*r?['\"](FAILED|FAIL|failed)"), 0))
    for unit in _units(pf, lines, whole_module=True):
        if isinstance(unit, _ModuleBody) and not module_level:
            continue
        src = unit.src
        if not _RUN_CALL.search(src) or _TOKEN_SAFE.search(src):
            continue
        if re.search(r"(?i)(['\"](passed|pass|ok|success)\b|returncode|exit_code|exitcode|\berror\b|collected|"
                     r"no tests ran|expected|fail_to_pass|required|\.passed\b)", src):
            continue  # some positive or error check exists
        names, runs = _output_names(unit)
        if not runs:
            continue
        for n in ast.walk(unit.node):
            hit = False
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.NotIn) and \
                    (_str_of(n.left) or "").strip() in ("FAILED", "FAIL", "failed") and _names_in(n.comparators[0]) & names:
                hit = True
            elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq) and \
                    isinstance(n.left, ast.Call) and _call_name(n.left) == "count" and n.left.args and \
                    (_str_of(n.left.args[0]) or "").strip() in ("FAILED", "FAIL", "failed") and \
                    isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value == 0 and _names_in(n.left) & names:
                hit = True
            if hit:
                out.append((unit, n.lineno))
    return out


def _candidate_denominators(pf: PyFile, module_level=False):
    """passed / (passed + failed) with both counts parsed from the candidate's run, so errored and skipped tests
    drop out of the denominator."""
    out = []
    if pf.tree is None or "/" not in pf.text:
        return out
    lines = _hint_lines(pf.text, (("/",), re.compile(r"(?i)pass\w*\s*/\s*\(?\s*\w*(pass|fail)\w*\s*\+\s*\w*(pass|fail)"), 0))
    for unit in _units(pf, lines, whole_module=True):
        if isinstance(unit, _ModuleBody) and not module_level:
            continue
        names, runs = _output_names(unit, params_as_output=True)
        if not names:
            continue
        derived = set(names)
        for n in _sorted_nodes(unit.node, ast.Assign):
            if _names_in(n.value) & derived:
                for t_ in n.targets:
                    derived |= {x.id for x in ast.walk(t_) if isinstance(x, ast.Name)}
        for n in ast.walk(unit.node):
            if not (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)):
                continue
            try:
                num, den = ast.unparse(n.left), ast.unparse(n.right)
            except Exception:
                continue
            if not re.search(r"(?i)pass", num) or not (re.search(r"(?i)pass", den) and re.search(r"(?i)fail", den)):
                continue
            if re.search(r"(?i)(error|skip|xfail|expected|total|required|len\()", den):
                continue
            if not (_names_in(n.left) & derived and _names_in(n.right) & derived):
                continue
            out.append((unit, n.lineno))
    return out


def _conftest_child_verdicts(pkg: Package):
    out = []
    for p, t in pkg.files.items():
        if os.path.basename(p) != "conftest.py" or "runtestprotocol" not in t or "fork" not in t:
            continue
        if not re.search(r"os\.fork\(|\.fork\(\)|multiprocessing|Process\(", t):
            continue
        if not re.search(r"(os\.pipe|os\.read|os\.write|\.recv\(|\.send\(|Queue\(|Pipe\(|pickle\.loads)", t):
            continue
        task = _task_of(pkg, p)
        if task is None and "/tests/" not in "/" + p and not p.startswith("tests/"):
            continue
        m = re.search(r"runtestprotocol", t)
        out.append(ev(p, line_of(t, m.start())))
    return out


_VERDICT_MSG_V05 = {
    "score-stdout": "The score is a number parsed from the output of a process that also runs the submission, so code "
                    "that prints a high score and exits sets its own reward",
    "nofail": "The pass verdict is only that no FAILED line appears in the test output, so a run that crashes or "
              "collects no tests scores",
    "denominator": "The pass rate divides by the passed and failed counts of the candidate's own run, so tests that "
                   "error or are skipped drop out and raise the score",
    "conftest-child": "A conftest runs each test in a forked child that also imports the submission and trusts the "
                      "result the child sends back, so code in the child can send a pass",
}


# ---- items 2, 5 and 7, widen existence-or-constant-check ---------------------------------------------------
# Canon: vacuous-assert-test-patterns (Terminal-Bench 2 Verified, arxiv:2610.32548), tests-grep-source-or-mock and
# tests-grep-source-not-behavior (arxiv:2602.03012, arxiv:2610.00850), trivial-property-validator (trustworthy-env
# T-4, benchjack role-only check).

_PRINTISH = {"print", "sleep", "info", "debug", "warning", "warn", "log", "pprint", "format", "str", "repr", "len",
             "append", "join", "items", "keys", "values", "write", "flush", "time", "perf_counter", "monotonic"}
_SWALLOW_EXC = {"Exception", "BaseException", "AssertionError", "AssertError"}
_CHECK_CALL = re.compile(r"^(assert\w*|fail|raises|approx|expect\w*|check_\w+|verify\w*)$")


def _test_functions(pf: PyFile):
    return [fi for fi in pf.funcs if fi.name.startswith("test") and
            (fi.cls is None or fi.cls.startswith("Test") or "TestCase" in fi.cls)]


def _helper_raises(pf: PyFile) -> dict:
    """Module level helper name -> True if it can fail a test (assert, raise, pytest.fail, sys.exit, assert* call)."""
    out = {}
    for fi in pf.funcs:
        if fi.name.startswith("test") or fi.cls is not None:
            continue
        can = False
        for n in ast.walk(fi.node):
            if isinstance(n, (ast.Assert, ast.Raise)):
                can = True
            elif isinstance(n, ast.Call):
                cn = _call_name(n)
                if cn.startswith("assert") or cn in ("fail", "exit", "_exit", "skip") or cn not in _PRINTISH and \
                        cn not in out and cn not in {f.name for f in pf.funcs}:
                    can = True
            if can:
                break
        out[fi.name] = can
    return out


def _trivially_true(e) -> bool:
    if isinstance(e, ast.Constant):
        return bool(e.value) and not isinstance(e.value, str) or isinstance(e.value, str) and bool(e.value)
    if isinstance(e, ast.BoolOp) and isinstance(e.op, ast.Or):
        return any(_trivially_true(v) for v in e.values)
    if isinstance(e, ast.Compare) and len(e.ops) == 1:
        try:
            a, b = ast.unparse(e.left), ast.unparse(e.comparators[0])
        except Exception:
            return False
        if isinstance(e.ops[0], (ast.Eq, ast.Is, ast.GtE, ast.LtE)) and a == b and not re.search(r"\(", a):
            return True
        if isinstance(e.ops[0], ast.GtE) and re.match(r"len\(", a) and isinstance(e.comparators[0], ast.Constant) and \
                e.comparators[0].value == 0:
            return True
    if isinstance(e, ast.Call) and _call_name(e) == "isinstance" and len(e.args) == 2 and ast.unparse(e.args[1]) == "object":
        return True
    return False


def _swallowing_try(t: ast.Try) -> bool:
    for h in t.handlers:
        tname = ast.unparse(h.type) if h.type is not None else "BaseException"
        caught = {x.strip() for x in re.split(r"[(),]", tname) if x.strip()}
        if not (caught & _SWALLOW_EXC):
            continue
        if any(isinstance(x, ast.Raise) or isinstance(x, ast.Assert) or
               isinstance(x, ast.Call) and (_call_name(x) in ("fail", "exit", "_exit") or _call_name(x).startswith("assert"))
               for b in h.body for x in ast.walk(b)):
            continue
        return True
    return False


def _vacuous_test(fi, helpers) -> str | None:
    """Kind of vacuity for one test function, or None if it can fail on a wrong answer."""
    for d in fi.decorators:
        if re.match(r"^(pytest\.)?mark\.skip(\(\s*(reason\s*=[^,]*)?\))?$", d) or re.match(r"^(unittest\.)?skip\(", d):
            return "skip"
        if re.match(r"^(pytest\.)?mark\.xfail\b", d) and "strict=True" not in d or d in ("unittest.expectedFailure", "expectedFailure"):
            return "xfail"
    body = [s for s in fi.node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call) and \
            ast.unparse(body[0].value.func) in ("pytest.skip", "self.skipTest", "pytest.xfail"):
        return "skip"
    swallowed_ids = set()
    for t in ast.walk(fi.node):
        if isinstance(t, ast.Try) and _swallowing_try(t):
            for b in t.body:
                swallowed_ids |= {id(x) for x in ast.walk(b)}
    real, trivial, swallowed, helper, other_calls = 0, 0, 0, 0, 0
    for n in ast.walk(fi.node):
        sw = id(n) in swallowed_ids
        if isinstance(n, (ast.Import, ast.ImportFrom)) and not sw:
            other_calls += 1  # an import can fail
        elif isinstance(n, ast.Return) and n.value is not None and not (isinstance(n.value, ast.Constant) and n.value.value is None):
            real += 1  # the test is called as a function and its return value is the verdict
        elif isinstance(n, ast.Assert):
            if _trivially_true(n.test):
                trivial += 1
            elif sw:
                swallowed += 1
            else:
                real += 1
        elif isinstance(n, ast.Raise):
            if sw:
                swallowed += 1
            else:
                real += 1
        elif isinstance(n, ast.Call):
            cn = _call_name(n)
            if cn in ("assertTrue", "assertIsNotNone") and n.args and _trivially_true(n.args[0]):
                trivial += 1
            elif _CHECK_CALL.match(cn) or cn.startswith("assert"):
                if sw:
                    swallowed += 1
                elif cn in helpers and not helpers[cn]:
                    helper += 1
                else:
                    real += 1
            elif cn in helpers:
                if helpers[cn] and not sw:
                    real += 1
                elif not helpers[cn]:
                    helper += 1
                else:
                    swallowed += 1
            elif cn not in _PRINTISH:
                if not sw:
                    other_calls += 1
    if real:
        return None
    if swallowed:
        return "swallowed"
    if trivial:
        return "tautology"
    if helper:
        return "helper"
    if not other_calls:
        return "noassert"
    return None


def _vacuous_tests(pkg: Package, already: set):
    """-> (file_level_evidence, per_test_evidence)."""
    whole, single = [], []
    for path, text in _graded_test_files(pkg):
        base = os.path.basename(path)
        if not base.startswith(("test", "check")) or base == "conftest.py" or path in already:
            continue
        if not re.search(r"(def test|class Test)", text):
            continue
        if re.search(r"__name__\s*==\s*['\"]__main__['\"]", text) and not re.search(r"(pytest|unittest)\.main\(", text):
            continue  # run as a script that decides the verdict itself
        if re.search(r"(?i)(placeholder|will be (overwritten|replaced|generated)|template)", text[:600]):
            continue  # a stub the adapter replaces
        task = _task_of(pkg, path)
        sh = _task_sh_texts(pkg, task)
        if sh and re.search(r"\.py\b", sh) and base not in sh and not re.search(r"pytest\s[^\n]*(/tests/?\s|/tests/?$|tests/?\s*$|\$\w+)", sh, re.M) \
                and "pytest" in sh and re.search(rf"\btest_\w+\.py", sh):
            continue  # test.sh names other test files and not this one
        pf = _pyf(pkg, path)
        if pf.tree is None:
            continue
        tests = _test_functions(pf)
        if not tests:
            continue
        helpers = _helper_raises(pf)
        kinds = [(fi, _vacuous_test(fi, helpers)) for fi in tests]
        if all(k is not None for _fi, k in kinds):
            whole.append(ev(pf, tests[0].start))
            continue
        for fi, k in kinds:
            if k is not None:
                single.append(ev(pf, fi.start))
    return whole, single


_READ_CALLS = {"read", "read_text", "readlines", "read_bytes"}
_RUN_IN_TEST = re.compile(r"(subprocess|os\.system|os\.popen|pexpect|requests\.|urllib|socket\.|exec\(|eval\(|"
                          r"import_module|spec_from_file_location|run_path|ctypes|cdll|docker|sandbox|\.run\(|Popen|"
                          r"check_output|sh\.|plumbum|selenium|playwright|webdriver|httpx|aiohttp)")


def _source_read_paths(pf: PyFile) -> list:
    """(line, path) for reads of a source code file (open(p).read(), Path(p).read_text())."""
    out = []
    for n in ast.walk(pf.tree):
        if isinstance(n, ast.Call) and _call_name(n) in _READ_CALLS:
            base = n.func.value if isinstance(n.func, ast.Attribute) else None
            inner = base
            if isinstance(base, ast.Call) and _call_name(base) in ("open", "Path", "PurePath") and base.args:
                inner = base.args[0]
            elif isinstance(base, ast.Name):
                inner = base
            if inner is None:
                continue
            s = _resolve_str(inner, pf.consts)
            if s is None and isinstance(inner, ast.Name):
                s = pf.consts.get(inner.id)
            if s is None and isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div):
                s = _resolve_str(inner.right, pf.consts)
            if s and _CODE_EXT.search(s.strip()) and not s.startswith("/tests/"):
                out.append((n.lineno, s))
    return out


def _grep_source_tests(pkg: Package, already: set):
    out = []
    for path, text in _graded_test_files(pkg):
        base = os.path.basename(path)
        if not base.startswith(("test", "check")) or path in already:
            continue
        if not re.search(r"(\.read\(|read_text\()", text) or _RUN_IN_TEST.search(text):
            continue
        pf = _pyf(pkg, path)
        if pf.tree is None:
            continue
        task = _task_of(pkg, path)
        dirs = task.agent_dirs() if task and not task.separate else AGENT_DIRS
        if _imports_submission(pf, dirs) or _agent_tree_imports(pf, dirs):
            continue  # the tests also run the code
        reads = _source_read_paths(pf)
        if not reads:
            continue
        tests = _test_functions(pf)
        if not tests:
            continue
        n_src = 0
        ok = True
        for fi in tests:
            s = fi.src
            if re.search(r"(\.read\(|read_text\(|readlines\(|\bsource\b|\bsrc\b|\bcode\b|\bcontent\w*\b)", s) and \
                    re.search(r"(\bin\s+\w+|re\.(search|findall|match|fullmatch)\(|\.count\(|ast\.(parse|walk)|"
                              r"startswith\(|endswith\(|\bnot\s+in\b)", s):
                n_src += 1
            elif all(_existence_expr(a.test) for a in ast.walk(fi.node) if isinstance(a, ast.Assert)):
                continue
            else:
                ok = False
                break
        if ok and n_src:
            out.append(ev(pf, reads[0][0]))
    return out


def _sh_grep_source(pkg: Package):
    out = []
    for p, t in _graded_sh(pkg):
        if "reward" not in t or "grep" not in t:
            continue
        cmds, _ = _sh_cached(pkg, p, t, None)
        if any(_runs_submission(c) or _python_launch(c.argv) is not None or _pytest_args(c.argv) is not None for c in cmds):
            continue
        for c in cmds:
            if c.argv[0] != "grep":
                continue
            files = [a for a in c.argv[1:] if not a.startswith("-")][1:]
            if any(_CODE_EXT.search(f_) and not f_.startswith("/tests/") for f_ in files):
                lines = t.splitlines()
                ctx = "\n".join(lines[c.line - 1:c.line + 6])
                if re.search(r"(echo\s+['\"]?1(\.0)?['\"]?\s*>+\s*\S*reward|reward=1)", ctx) or \
                        re.search(r"^\s*if\b", lines[c.line - 1]):
                    out.append(ev(p, c.line))
                    break
    return out


_ROLE_CMP = re.compile(r"\[\s*['\"]role['\"]\s*\]\s*(==|!=)\s*['\"]assistant['\"]|\.get\(\s*['\"]role['\"]\s*\)\s*(==|!=)\s*"
                       r"['\"]assistant['\"]|\.role\s*(==|!=)\s*['\"]assistant['\"]")


def _role_only_rewards(pkg: Package):
    out = []
    for pf in pkg.py_all():
        if "role" not in pf.text or not _ROLE_CMP.search(pf.text):
            continue
        for fi in pf.funcs:
            src = fi.src
            if not _ROLE_CMP.search(src):
                continue
            if re.search(r"(content|text|answer|parse|judge|state\[|info\[|\.get\(\s*['\"](?!role))", src) or _uses_answer(fi):
                continue
            rets = [n for n in ast.walk(fi.node) if isinstance(n, ast.Return) and n.value is not None]
            if not rets or not all(isinstance(r.value, ast.Constant) or isinstance(r.value, (ast.IfExp, ast.Compare, ast.Call))
                                   for r in rets):
                continue
            if not any(isinstance(r.value, ast.Constant) and r.value.value not in (0, 0.0, False, None) or
                       not isinstance(r.value, ast.Constant) for r in rets):
                continue
            if _registered(pkg, fi):
                out.append(ev(pf, fi.start))
    return out


_METRIC_KEY = re.compile(r"(accuracy|acc\b|loss|f1|auc|bleu|rouge|metric|score|perplexity|ppl|precision|recall|mse|mae|"
                         r"rmse|r2|eval_\w+|map\b|ndcg|iou|dice)", re.I)


def _metric_literal_tests(pkg: Package):
    """assert data["eval_accuracy"] == 0.5271 on a file the agent writes, with the number shown in the instruction."""
    out = []
    for path, text in _graded_test_files(pkg):
        if not re.search(r"==\s*-?\d+\.\d{3,}", text) or not re.search(r"(json\.load|\.read\(|read_text|csv|yaml)", text):
            continue
        task = _task_of(pkg, path)
        instr = _task_text(pkg, task, "instruction.md") if task else ""
        if not instr:
            continue
        pf = _pyf(pkg, path)
        if pf.tree is None:
            continue
        dirs = task.agent_dirs()
        if not any(d in text for d in dirs):
            continue
        for n in ast.walk(pf.tree):
            if not isinstance(n, ast.Assert) or not isinstance(n.test, ast.Compare) or len(n.test.ops) != 1 or \
                    not isinstance(n.test.ops[0], ast.Eq):
                continue
            a, b = n.test.left, n.test.comparators[0]
            lit = b if isinstance(b, ast.Constant) else a if isinstance(a, ast.Constant) else None
            other = a if lit is b else b
            if lit is None or not isinstance(lit.value, float) or isinstance(lit.value, bool):
                continue
            s = repr(lit.value)
            if len(s.split(".")[-1]) < 3:
                continue
            try:
                os_ = ast.unparse(other)
            except Exception:
                continue
            if not _METRIC_KEY.search(os_) or not isinstance(other, ast.Subscript):
                continue
            if s in instr or s.rstrip("0") in instr:
                out.append(ev(pf, n.lineno))
    return out

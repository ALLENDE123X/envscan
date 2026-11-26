"""v0.2 rule family "image": what the agent's image holds and what the grader
trusts at grading time.

Rows of the literature review's section 9.2 table (report_parts/09):
  #2 git-history-or-target-in-image   arxiv:2609.08149, arxiv:2610.00850
  #3 answers-or-grader-readable       arxiv:2610.41014, arxiv:2608.22103
  #4 writable-reward-channel          openreview vxftf9je8s (RewardProof)
  #8 verify-time-fetch                rdi.berkeley.edu/blog/trustworthy-benchmarks-cont
  #9 egress-and-shared-state          arxiv:2610.24206, OpenAI and Hugging Face incident report

The rules read two formats. verifiers packages are Python, and often ship
Dockerfiles, shell scripts or whole Harbor task folders. Harbor tasks are a
folder with task.toml, environment/Dockerfile, tests/test.sh and
solution/solve.sh. In Harbor, tests/ and solution/ only reach the agent's image
if the Dockerfile copies them, the verifier shares the agent's box unless
[verifier] declares a separate environment, and the verifier writes
/logs/verifier/reward.txt or reward.json (reward.json wins when both exist).

Only text is read. Nothing is built, run or imported.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shlex
from dataclasses import dataclass, field

try:
    import tomllib  # py311+
except Exception:  # pragma: no cover
    tomllib = None

from .package import Package, PyFile, _const_text
from .rules import FRESH_BOX, ISOLATION_ANY, SCORING_NAME, F, ev, line_of, referenced_consts, rule

# ---------------------------------------------------------------------------
# shared parsing: Harbor tasks, Dockerfiles, shell scripts
# ---------------------------------------------------------------------------

VERIFIER_DIRS = {"tests", "test", "verifier", "verifiers", "grader", "graders", "judge", "scorer", "scoring",
                 "eval", "evals", "evaluation", "oracle"}
WORKSPACE_DEST = re.compile(r"^(\.|\./?|/app|/testbed|/workspace|/repo|/src|/code|/project|/work|/home/[^/]+(/[^/]+)?|"
                            r"/root/[^/.][^/]*|/app/[^/]+|/testbed/[^/]+|/workspace/[^/]+|\$\{?\w+\}?|\{\w+\})/?$")
DEPENDENCY_DEST = re.compile(r"^(/tmp|/opt|/usr|/var|/etc|/root/\.|/build|/deps|/third_party)|"
                             r"(deps|vendor|third[_-]?party|tools?|build|cache|plugins?|sdk|toolchain)(/|$)", re.I)


@dataclass
class HTask:
    root: str                   # task dir relative to the package root ("" when the package is the task)
    toml: str                   # path of task.toml
    shared: bool                # some verify pass runs in the agent's box
    agent_root: bool            # agent user omitted or root
    internet: bool | None       # agent phase internet, if declared
    scripts: list = field(default_factory=list)      # verifier scripts (tests/test.sh, steps/*/tests/test.sh)
    dockerfile: str | None = None                    # agent image Dockerfile
    cfg: dict = field(default_factory=dict)

    def rel(self, p: str) -> str:
        return p[len(self.root) + 1:] if self.root else p


@dataclass
class Index:
    tasks: list
    dockerfiles: dict           # path -> role: agent | verifier | other
    instrs: dict                # path -> [(line, INSTR, value)]
    pkg: Package = None
    _isolated: bool | None = None
    _grader: list | None = None

    @property
    def grader(self) -> list:
        """grader_funcs(pkg), computed once per package."""
        if self._grader is None:
            self._grader = grader_funcs(self.pkg)
        return self._grader

    @property
    def py_isolated(self) -> bool:
        """The package's own Python declares an isolated verifier (same signal the v0.1 rules use)."""
        if self._isolated is None:
            files = py_with(self.pkg, ISOLATION_LITERALS)
            self._isolated = any(ISOLATION_ANY.search(pf.text) for pf in files)
        return self._isolated

    @property
    def has_py(self) -> bool:
        return bool(self.pkg.py)


ISOLATION_LITERALS = ("fresh_copy", "IsolatedVerifier", "isolated_verifier", "environment_mode", "separate",
                      "stage_verifier", "artifact_max_bytes", "hand-in", "handin", "hand in")


def py_with(pkg: Package, literals) -> list:
    """Parsed package files whose text contains any of the literals (a cheap substring filter)."""
    return [pf for pf in pkg.py_all() if any(x in pf.text for x in literals)]


def funcs_with(pkg: Package, literals, pat: re.Pattern | None = None):
    """(pf, fi) whose own source contains a literal (and matches pat, when given)."""
    for pf in py_with(pkg, literals):
        for fi in pf.funcs:
            if any(x in fi.src for x in literals) and (pat is None or pat.search(fi.src)):
                yield pf, fi


_CACHE: list = [None, None]


def index(pkg: Package) -> Index:
    """Per package parse shared by the image rules (kept for the package being scanned only)."""
    if _CACHE[0] is pkg:
        return _CACHE[1]
    idx = _build_index(pkg)
    _CACHE[0], _CACHE[1] = pkg, idx
    return idx


def _toml(text: str) -> dict:
    if tomllib is None:
        return {}
    try:
        return tomllib.loads(text)
    except Exception:
        return {}


def _verifier_mode(v: dict) -> str | None:
    if not isinstance(v, dict):
        return None
    m = v.get("environment_mode")
    if isinstance(m, str):
        return m.lower()
    if isinstance(v.get("environment"), dict):
        return "separate"
    return None


def _task_from_toml(path: str, text: str, files) -> HTask:
    root = os.path.dirname(path)
    cfg = _toml(text)
    if cfg:
        top = _verifier_mode(cfg.get("verifier", {})) or "shared"
        steps = cfg.get("steps") if isinstance(cfg.get("steps"), list) else []
        modes = [(_verifier_mode(st.get("verifier", {})) if isinstance(st, dict) else None) or top for st in steps]
        shared = any(m == "shared" for m in modes) if modes else top == "shared"
        agent = cfg.get("agent", {}) if isinstance(cfg.get("agent"), dict) else {}
        user = agent.get("user")
        agent_root = user is None or str(user).split(":")[0] in ("0", "root")
        envc = cfg.get("environment", {}) if isinstance(cfg.get("environment"), dict) else {}
        internet = envc.get("allow_internet")
        if internet is None and isinstance(envc.get("network_mode"), str):
            internet = envc["network_mode"] not in ("no-network", "none")
        if isinstance(agent.get("network_mode"), str):  # v0.5: an explicit agent phase policy wins (Harbor)
            internet = agent["network_mode"] not in ("no-network", "none")
    else:  # template placeholders or a broken file: fall back to text
        shared = not re.search(r"environment_mode\s*=\s*['\"]separate|^\s*\[verifier\.environment\]", text, re.M)
        agent_root = not re.search(r"^\s*user\s*=\s*['\"](?!root|0)\w", text, re.M)
        m = re.search(r"allow_internet\s*=\s*(true|false)", text)
        internet = (m.group(1) == "true") if m else None
    pre = root + "/" if root else ""
    scripts = [p for p in files if p.startswith(pre) and
               re.fullmatch(r"(steps/[^/]+/)?tests/test\.sh|run-tests\.sh|run_tests\.sh", p[len(pre):])]
    if any(p[len(pre):].startswith("tests/") for p in scripts):
        scripts = [p for p in scripts if "/" in p[len(pre):]]
    dockerfile = None
    for cand in ("environment/Dockerfile", "Dockerfile", "environment/Containerfile"):
        if pre + cand in files:
            dockerfile = pre + cand
            break
    return HTask(root, path, shared, agent_root, internet if isinstance(internet, bool) else None,
                 sorted(scripts), dockerfile, cfg)


def is_dockerfile(p: str) -> bool:
    b = os.path.basename(p)
    return b in ("Dockerfile", "Containerfile") or b.endswith(".dockerfile") or b.startswith("Dockerfile.")


def _build_index(pkg: Package) -> Index:
    files = pkg.files
    tasks = [_task_from_toml(p, t, files) for p, t in files.items() if os.path.basename(p) == "task.toml"]
    roots = sorted(tasks, key=lambda t: -len(t.root))
    dockerfiles, instrs = {}, {}
    for p, t in files.items():
        if not is_dockerfile(p):
            continue
        role = None
        for tk in roots:
            if tk.root == "" or p.startswith(tk.root + "/"):
                rel = tk.rel(p)
                if p == tk.dockerfile:
                    role = "agent"
                elif re.fullmatch(r"(steps/[^/]+/)?tests/(Dockerfile|Containerfile)", rel):
                    role = "verifier"
                else:
                    role = "other"
                break
        if role is None:
            parts = [x.lower() for x in os.path.dirname(p).split("/") if x]
            role = "verifier" if any(x in VERIFIER_DIRS for x in parts) else "agent"
        dockerfiles[p] = role
        instrs[p] = docker_instrs(t)
    return Index(tasks, dockerfiles, instrs, pkg)


HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")


def docker_instrs(text: str):
    """[(line, INSTRUCTION, value)] with continuations joined, comments dropped and
    heredoc bodies appended to the instruction that opened them."""
    out, pending, start, heredoc, cur = [], "", 1, None, None
    for i, raw in enumerate(text.splitlines(), 1):
        if heredoc:
            if raw.strip() == heredoc:
                heredoc = None
            elif cur is not None:
                cur[2] += "\n" + raw
            continue
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if not pending:
            start = i
        if s.endswith("\\"):
            pending += s[:-1] + " "
            continue
        line, pending = pending + s, ""
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        cur = [start, parts[0].upper(), parts[1]]
        out.append(cur)
        m = HEREDOC.search(parts[1])
        if m and cur[1] in ("RUN", "COPY", "ADD"):
            heredoc = m.group(2)
    if pending:
        parts = pending.split(None, 1)
        if len(parts) == 2:
            out.append([start, parts[0].upper(), parts[1]])
    return [tuple(x) for x in out]


def _strip_comment(s: str) -> str:
    q = None
    for i, ch in enumerate(s):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or s[i - 1] in " \t;"):
            return s[:i]
    return s


def sh_lines(text: str):
    """[(line, logical command line)] for a shell script, without comments or heredoc bodies."""
    out, pending, start, heredoc = [], "", 1, None
    for i, raw in enumerate(text.splitlines(), 1):
        if heredoc:
            if raw.strip() == heredoc:
                heredoc = None
            continue
        s = raw.strip()
        if not pending and (not s or s.startswith("#")):
            continue
        if not pending:
            start = i
        if s.endswith("\\"):
            pending += s[:-1] + " "
            continue
        line, pending = _strip_comment(pending + s), ""
        m = HEREDOC.search(line)
        if m:
            heredoc = m.group(2)
        if line.strip():
            out.append((start, line))
    if pending:
        out.append((start, pending))
    return out


def copy_args(value: str):
    """COPY/ADD value -> (flags dict, sources, dest)."""
    flags = {}
    v = value.strip()
    while v.startswith("--"):
        tok, _, rest = v.partition(" ")
        k, _, val = tok[2:].partition("=")
        flags[k.lower()] = val
        v = rest.strip()
    try:
        args = json.loads(v) if v.startswith("[") else shlex.split(v)
    except Exception:
        args = v.split()
    args = [a for a in args if isinstance(a, str)]
    if len(args) < 2:
        return flags, [], ""
    return flags, args[:-1], args[-1]


def _final_user_nonroot(instrs) -> bool:
    users = [v.split()[0] for _, ins, v in instrs if ins == "USER" and v.split()]
    return bool(users) and users[-1].split(":")[0] not in ("root", "0")


def task_of(idx: Index, path: str):
    for tk in sorted(idx.tasks, key=lambda t: -len(t.root)):
        if tk.root == "" or path.startswith(tk.root + "/"):
            return tk
    return None


def _call_text(node) -> str:
    """Literal text of a call's arguments (strings, f-strings, lists of strings)."""
    parts = []
    for a in list(getattr(node, "args", [])) + [k.value for k in getattr(node, "keywords", [])]:
        t = _const_text(a)
        if t is not None:
            parts.append(t)
        elif isinstance(a, ast.Name):
            parts.append("{" + a.id + "}")
    return " ".join(parts)


BOX_HEAD = re.compile(r"((runtime|sandbox\w*|container\w*|\bbox|session|\bsb|client)\b[\w.\[\]'\"]*\.(run\w*|exec\w*|execute\w*|"
                      r"arun|aexec|read_file|download\w*|get_file|upload\w*|write_file|put_file)\b|"
                      r"(^|[.(])(execute_command\w*|run_command\w*|exec_command\w*|run_in_sandbox|sandbox_exec|"
                      r"run_background_job)\b)")


def _own_nodes(fn):
    """Nodes of a function body, not descending into nested defs (they are visited on their own)."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))


def _box_calls(pf: PyFile, fi):
    """(lineno, text, head) for each call in fi that executes or reads something in the sandbox."""
    out = []
    for n in _own_nodes(fi.node):
        if not isinstance(n, ast.Call):
            continue
        func = n.func
        if isinstance(func, ast.Call):  # with_retry(sandbox.execute_command)(...)
            inner = [a for a in func.args if isinstance(a, (ast.Attribute, ast.Name))]
            func = inner[0] if inner else func.func
        try:
            head = ast.unparse(func)
        except Exception:
            continue
        if not BOX_HEAD.search(head):
            continue
        txt = _call_text(n)
        for a in n.args:
            if isinstance(a, ast.Name) and a.id in pf.consts:
                txt += " " + pf.consts[a.id]
            elif isinstance(a, ast.Attribute) and a.attr in pf.consts:
                txt += " " + pf.consts[a.attr]
        out.append((n.lineno, txt, head))
    return out


def _is_setup_name(name: str) -> bool:
    return bool(re.search(r"(setup|set_up|init|prepare|start|create|build|install|provision|bootstrap|before|pre_|"
                          r"_load|load_|reset|spawn|launch|warm)", name, re.I))


GRADER_NAME = re.compile(r"(reward|score|scoring|grade|grading|judge|verif|check_answer|compute_metric|_metric\b|passed|"
                         r"solved|correctness|finalize|post_rollout|run_tests|evaluate|is_resolved|compute_reward)", re.I)


TOOL_LIST = re.compile(r"\b(?:tools|_tools|TOOLS|tool_list)\s*[=:]\s*[\[(]([^\])]{0,2000})[\])]")
TOOL_ADD = re.compile(r"\b(?:add_tool|register_tool)\(\s*([\w.]+)")


def tool_names(pkg: Package) -> set:
    """Names of functions handed to the agent as tools (tools=[...], self.tools = [...], add_tool(f), @tool)."""
    out = set()
    for pf in py_with(pkg, ("tools", "TOOLS", "add_tool", "register_tool", "tool_list", "@tool", ".tool")):
        for m in TOOL_LIST.finditer(pf.text):
            out |= {x.rsplit(".", 1)[-1] for x in re.findall(r"[A-Za-z_][\w.]*", m.group(1))}
        out |= {x.rsplit(".", 1)[-1] for x in TOOL_ADD.findall(pf.text)}
        for fi in pf.funcs:
            if any(re.search(r"\btool\b", d) for d in fi.decorators):
                out.add(fi.name)
    return out


def _registered(pkg: Package) -> set:
    """Names placed in a rubric (funcs=[...], Rubric([...]), add_reward_func(f)); the same reading as
    rules.registered_reward_names, but only over files that mention a rubric, which keeps big packages fast."""
    out = set()
    for pf in py_with(pkg, ("Rubric", "add_reward_func", "funcs", "reward_funcs")):
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if not isinstance(n, ast.Call):
                continue
            fname = n.func.attr if isinstance(n.func, ast.Attribute) else (n.func.id if isinstance(n.func, ast.Name) else "")
            kws = {k.arg: k.value for k in n.keywords if k.arg}
            lst = kws.get("funcs") or kws.get("reward_funcs")
            if lst is None and (fname.endswith("Rubric") or fname == "RubricGroup") and n.args:
                lst = n.args[0]
            if fname == "add_reward_func" and n.args:
                lst = ast.List(elts=[n.args[0]])
            if isinstance(lst, (ast.List, ast.Tuple)):
                for e in lst.elts:
                    while isinstance(e, ast.Call):
                        e = e.func
                    if isinstance(e, ast.Name):
                        out.add(e.id)
                    elif isinstance(e, ast.Attribute):
                        out.add(e.attr)
    return out


def grader_funcs(pkg: Package) -> list:
    """Functions on the reward path: registered or @reward functions, grader named functions and Harbor style
    finalizers, plus their direct callees. Agent tools are left out (an agent tool named test_x is not the grader)."""
    reg = _registered(pkg)
    tools = tool_names(pkg)
    by_name: dict = {}
    seeds = []
    for pf in pkg.py_all():
        for fi in pf.funcs:
            by_name.setdefault(fi.name, []).append((pf, fi))
            if fi.name in tools:
                continue
            if fi.name in reg or any("reward" in d for d in fi.decorators) or GRADER_NAME.search(fi.name):
                seeds.append((pf, fi))
    out = {(pf.path, fi.start): (pf, fi) for pf, fi in seeds}
    for pf, fi in seeds:
        for n in _own_nodes(fi.node):
            if isinstance(n, ast.Call):
                cn = n.func.id if isinstance(n.func, ast.Name) else (n.func.attr if isinstance(n.func, ast.Attribute) else None)
                if cn and cn not in tools:
                    for pf2, fi2 in by_name.get(cn, []):
                        out.setdefault((pf2.path, fi2.start), (pf2, fi2))
    return list(out.values())


def grades_in_fresh_box(funcs) -> bool:
    return any(FRESH_BOX.search(fi.src) for _, fi in funcs)


# ---------------------------------------------------------------------------
# #2 git history or the target package in the agent's image
# evidence row 2: arxiv:2609.08149 (sealing git history, Table 3), arxiv:2610.00850 (Sec 3.4)
# ---------------------------------------------------------------------------

GIT_CLONE = re.compile(r"\bgit\s+clone\b([^;&|\n]*)")
GIT_PIN = re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?(?:checkout|reset\s+--hard|switch\s+--detach)\s+"
                     r"(?:(?:-q|--quiet|-f|--force|--detach)\s+)*['\"]?(?!-)([^\s;&|)'\"]+)['\"]?(?=[\s;&|)'\",]|$)(?!\s+--(\s|$))")
NON_PIN_REFS = re.compile(r"^(main|master|develop|dev|trunk|HEAD|\.|origin/(main|master)|FETCH_HEAD|-)$")
SCRUB = re.compile(r"(rm\s+-\w*[rf]\w*\s+[^;&|\n]*\.git(?![\w-])|reflog\s+expire|--orphan\b|"
                   r"filter-(branch|repo)|git\s+replace\b|update-ref\s+-d|for-each-ref[^\n]*(update-ref|branch\s+-D)|"
                   r"git\s+gc\b[^;&|\n]*--prune|history\s+(scrub|squash|strip)|squash\w*\s+(the\s+)?history|"
                   r"remove\w*\s+(the\s+)?(future|later)\s+commits)", re.I)
PIP_INSTALL = re.compile(r"\b(?:pip3?|python3?\s+-m\s+pip|uv\s+pip)\s+install\b([^;&|\n]*)")
GIT_PIN_LITERALS = ("git checkout", "git reset", "git switch", "git -C")
SCRUB_LITERALS = (".git", "reflog", "orphan", "filter-", "git replace", "update-ref", "--prune", "history", "commits",
                  "--depth")
SETUP_GIT_PIN = re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?(?:checkout|reset\s+--hard|switch\s+--detach)\s+"
                           r"(?:(?:-q|--quiet|-f|--force|--detach)\s+)*['\"]?"
                           r"(HEAD[~^]\d*|[0-9a-f]{7,40}\b|\{[^}]*(commit|sha|base|instance|ref|rev)[^}]*\})"
                           r"['\"]?(?=[\s;&|)'\",]|$)(?!\s+--(\s|$))", re.I)


# v0.5 gap git-copied-history-refs-hooks (arxiv:2609.08149): a scrub that only expires the reflog or runs
# gc --prune keeps every commit that a branch, a remote ref or a tag still points to, so it does not seal history.
STRONG_SCRUB = re.compile(r"(rm\s+-\w*[rf]\w*\s+[^;&|\n]*\.git(?![\w-])|--orphan\b|filter-(branch|repo)|git\s+replace\b|"
                          r"update-ref\s+-d|for-each-ref[^\n]*(update-ref|branch\s+-D)|history\s+(scrub|squash|strip)|"
                          r"squash\w*\s+(the\s+)?history|remove\w*\s+(the\s+)?(future|later)\s+commits)", re.I)
WEAK_SCRUB = re.compile(r"(reflog\s+expire|git\s+gc\b[^;&|\n]*--prune)", re.I)
LOCAL_DROPPED = re.compile(r"git\s+(-C\s+\S+\s+)?(branch\s+(-\w+\s+)*-[dD]\b|checkout\s+(-\w+\s+)*-B\b|switch\s+-C\b|"
                           r"reset\s+--hard)|update-ref\s+-d\s+refs/heads")
REMOTES_DROPPED = re.compile(r"git\s+(-C\s+\S+\s+)?remote\s+(remove|rm)\b|update-ref\s+-d\s+refs/remotes")
TAGS_DROPPED = re.compile(r"(git\s+(-C\s+\S+\s+)?tag\s+(-\w+\s+)*-d\b|\btag\s+-d\b|--no-tags\b)")
GIT_REMOTE_ADD = re.compile(r"\bgit\s+(?:-C\s+(\S+)\s+)?remote\s+add\s+(?:-\S+\s+)*(\S+)\s+(\S+)")
GIT_FETCH = re.compile(r"\bgit\s+(?:-C\s+(\S+)\s+)?fetch\b([^;&|\n]*)")


def _refs_left(tail: str, single: bool) -> str:
    """Which refs still point past the checked out commit after a reflog or gc only scrub ("" when none)."""
    left = []
    if not LOCAL_DROPPED.search(tail):
        left.append("the local branch")
    if not single and not REMOTES_DROPPED.search(tail):
        left.append("remote refs")
    if not TAGS_DROPPED.search(tail):
        left.append("tags")
    return ", ".join(left[:-1]) + (" and " if len(left) > 1 else "") + left[-1] if left else ""


def _fetch_clones(instrs, workdir_at):
    """git init plus remote add plus a full fetch, read as a clone: [(line, url, dest, shallow, ri, pos)]."""
    out = []
    remotes = {}
    for ri, (ln, ins, v) in enumerate(instrs):
        if ins != "RUN" or "git" not in v:
            continue
        for m in GIT_REMOTE_ADD.finditer(v):
            url = m.group(3).strip("'\"")
            if re.search(r"(://|@[\w.-]+:)", url):
                cd = re.findall(r"\bcd\s+(\S+)\s*&&", v[:m.start()])
                remotes[m.group(2)] = (url, m.group(1) or (cd[-1] if cd else workdir_at[ri]))
        for m in GIT_FETCH.finditer(v):
            args = m.group(2)
            try:
                toks = shlex.split(args)
            except Exception:
                toks = args.split()
            pos_args = [t for t in toks if not t.startswith("-")]
            remote = pos_args[0] if pos_args else "origin"
            if remote not in remotes and not re.search(r"(://|@[\w.-]+:)", remote):
                continue
            url, dest = remotes.get(remote, (remote, workdir_at[ri]))
            refspecs = pos_args[1:]
            shallow = any(t.startswith(("--depth", "--shallow-")) for t in toks)
            everything = any(t in ("--all", "--tags", "-t", "--unshallow") for t in toks)
            if refspecs and not everything and all(re.fullmatch(r"[0-9a-f]{7,40}", r.split(":")[0]) for r in refspecs):
                continue  # fetching one commit brings only its ancestors
            out.append((ln + v.count("\n", 0, m.start()), url, dest, shallow and not everything, ri, m.start()))
    return out


# v0.5 gap residue-and-cache-copies (arxiv:2610.00850, arxiv:2609.06780): the project is installed, built or run at
# the newer state and then moved back, so site-packages, a wheel or .pyc files keep the code the agent must write.
RES_SITE = re.compile(r"(\b(?:pip3?|python3?\s+-m\s+pip|uv\s+pip)\s+install\b(?![^;&|\n]*(?:\s-e\b|--editable))"
                      r"[^;&|\n]*?(?:\s|^)['\"]?(?:\.|\./|\.\[[^\]\s]*\]|/(?:app|testbed|workspace|repo|src)[\w/.-]*)['\"]?(?=\s|$|;|&|\|)|"
                      r"setup\.py\s+(?:install|bdist\w*)\b|python3?\s+-m\s+build\b|pip3?\s+wheel\s+(?:-\S+\s+)*\.)")
RES_PYC = re.compile(r"(\bcompileall\b|(?:^|[\s;&|(])(?:python3?\s+-m\s+)?pytest\b|\btox\b|setup\.py\s+(?:build|build_ext|develop)\b|"
                     r"python3?\s+-c\s+['\"][^'\"]*\bimport\s)")
RES_REVERT = re.compile(r"(git\s+(?:-C\s+\S+\s+)?apply\s+(?:-\S+\s+)*(?:-R|--reverse)\b|\bpatch\s+[^;&|\n]*(?:-R\b|--reverse)|"
                        r"git\s+(?:-C\s+\S+\s+)?stash\b(?!\s+(?:pop|apply))|git\s+(?:-C\s+\S+\s+)?restore\b|"
                        r"git\s+(?:-C\s+\S+\s+)?checkout\s+(?:-\S+\s+)*(?:\S+\s+)?--\s|"
                        r"git\s+(?:-C\s+\S+\s+)?(?:apply|am)\b[^;&|\n]*?(?:setup|remov|strip|scrambl|break|buggy|bug[_.-]|"
                        r"revert|mask|blank|stub|problem)[\w.-]*\.(?:patch|diff)|"
                        r"\bpatch\s+[^;&|\n]*?(?:setup|remov|strip|scrambl|break|buggy|bug[_.-]|revert|mask|blank|stub|problem)"
                        r"[\w.-]*\.(?:patch|diff))", re.I)
RES_CLEAN_SITE = re.compile(r"(\b(?:pip3?|python3?\s+-m\s+pip|uv\s+pip)\s+(?:install|uninstall)\b|setup\.py\s+(?:develop|install)\b)")
RES_CLEAN_PYC = re.compile(r"(\.pyc\b[^;&|\n]*(-delete|-exec\s+rm)|__pycache__[^;&|\n]*(-delete|-exec\s+rm|\brm\b)|"
                           r"\brm\s+[^;&|\n]*(__pycache__|\.pyc\b)|git\s+(?:-C\s+\S+\s+)?clean\s+-\w*x|\bpy3?clean\b)")


def _residue_in_dockerfile(instrs):
    """[(line, detail)]: a build or install step, then a later step that moves the code back, and no cleanup."""
    hits, builds, seen_revert = [], [], None
    for ri, (ln, ins, v) in enumerate(instrs):
        if ins == "FROM":
            builds, seen_revert = [], None
            continue
        if ins != "RUN":
            continue
        for part in re.split(r"&&|;|\|\||\n", v):
            p = part.strip()
            if not p:
                continue
            if seen_revert is not None:
                if RES_CLEAN_SITE.search(p):
                    builds = [b for b in builds if b[1] != "site"]
                if RES_CLEAN_PYC.search(p) or re.search(r"\bcompileall\b|(?:^|\s)(?:python3?\s+-m\s+)?pytest\b", p):
                    builds = [b for b in builds if b[1] != "pyc"]
                continue
            m_site, m_pyc = RES_SITE.search(p), RES_PYC.search(p)
            if m_site:
                builds.append((ln, "site", m_site.group(0).strip()))
            elif m_pyc:
                builds.append((ln, "pyc", m_pyc.group(0).strip()))
            m = RES_REVERT.search(p) or (SETUP_GIT_PIN.search(p) if builds else None)
            if m and re.search(r"checkout\s.*--\s[^;&|]*test", p, re.I):
                m = None  # checking out test files at another commit adds tests, it does not move the code back
            if m and builds and not (m_site or m_pyc):
                seen_revert = (ln + v.count("\n", 0, max(0, v.find(p))), m.group(0).strip())
    if seen_revert is not None and builds:
        kinds = sorted({b[1] for b in builds})
        where = " and ".join({"site": "site-packages or a wheel", "pyc": ".pyc caches"}[k] for k in kinds)
        hits.append((seen_revert[0], f"the project is built with {builds[0][2]!r} before {seen_revert[1]!r} moves the "
                                     f"code back, so {where} keep the newer code"))
    return hits


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.lower()).strip("-")


def _clone_info(args: str):
    """git clone args -> (url, dest, shallow, branch, single_branch)."""
    try:
        toks = shlex.split(args)
    except Exception:
        toks = args.split()
    shallow = any(t.startswith(("--depth", "--shallow-")) for t in toks)
    single = "--single-branch" in toks
    branch, pos, skip = None, [], False
    for i, t in enumerate(toks):
        if skip:
            skip = False
            continue
        if t in ("-b", "--branch", "--origin", "-o", "--config", "-c", "--reference", "--template",
                 "--separate-git-dir", "--depth", "--jobs", "-j", "--filter"):
            if t in ("-b", "--branch") and i + 1 < len(toks):
                branch = toks[i + 1]
            skip = True
            continue
        if t.startswith("-"):
            if t.startswith("--branch="):
                branch = t.split("=", 1)[1]
            continue
        pos.append(t)
    url = pos[0] if pos else ""
    dest = pos[1] if len(pos) > 1 else ""
    return url, dest, shallow, branch, single


def _repo_name(url: str) -> str:
    u = re.sub(r"\.git$", "", url.rstrip("/"))
    return u.rsplit("/", 1)[-1].rsplit(":", 1)[-1] if u else ""


def task_text(pkg: Package, dockerfile: str) -> str:
    """Instruction and config text of the task an image belongs to (lowercased), used to tell the task's
    target repository from a tool cloned to build the image."""
    d = os.path.dirname(dockerfile)
    root = os.path.dirname(d) if os.path.basename(d) in ("environment", "docker", "image", "sandbox") else d
    pre = root + "/" if root else ""
    out = [os.path.basename(root)]
    for name in ("instruction.md", "task.md", "TASK.md", "prompt.md", "README.md", "task.toml", "task.yaml",
                 "instructions.md"):
        t = pkg.files.get(pre + name)
        if t:
            out.append(t[:20000])
    return "\n".join(out).lower()


def _git_in_dockerfile(instrs, ttext: str = ""):
    """[(line, detail)] for one agent image Dockerfile. ttext is the task's instruction text."""
    runs = [(ri, ln, v) for ri, (ln, ins, v) in enumerate(instrs) if ins == "RUN"]
    if not any("git" in v for _, _, v in runs):
        return []
    workdir, clones, workdir_at = "", [], []
    for ri, (ln, ins, v) in enumerate(instrs):
        if ins == "WORKDIR":
            workdir = v.strip()
        workdir_at.append(workdir)
        if ins != "RUN":
            continue
        for m in GIT_CLONE.finditer(v):
            url, dest, shallow, branch, single = _clone_info(m.group(1))
            if not url or not re.search(r"(://|@[\w.-]+:|\$|\{)", url):
                continue  # a local clone copies only what is already there
            clones.append((ln + v.count("\n", 0, m.start()), url, dest, shallow, branch, single, ri, m.start(), workdir))
    fetched = set()
    for line, url, dest, shallow, ri, pos in _fetch_clones(instrs, workdir_at):
        if not any(c[1] == url for c in clones):
            clones.append((line, url, dest, shallow, None, False, ri, pos, workdir_at[ri]))
            fetched.add(line)

    def tail_from(ri, pos):
        return "\n".join(v[pos:] if rj == ri else v for rj, _, v in runs if rj >= ri)

    hits = []
    for line, url, dest, shallow, branch, single, ri, pos, wd in clones:
        tail = tail_from(ri, pos)
        name = _repo_name(url)
        path = dest or name
        later = tail[tail.find("\n") + 1:] if "\n" in tail else ""
        if STRONG_SCRUB.search(tail) or (path and re.search(rf"\brm\s+-\w*r\w*\s+[^;&|\n]*{re.escape(path)}/?(\s|$|;|&)", later)):
            continue
        left = _refs_left(tail, single) if WEAK_SCRUB.search(tail) else None
        if left == "":
            continue  # reflog expired and gc pruned with every ref past the base removed
        if dest:
            target = dest in (".", "./") or (bool(WORKSPACE_DEST.match(dest)) and not DEPENDENCY_DEST.search(dest))
        else:
            target = bool(WORKSPACE_DEST.match(wd or "/")) and not DEPENDENCY_DEST.search(wd or "/tmp")
        nm = name.lower()
        named = len(nm) >= 4 and bool(re.search(rf"(?<![\w-]){re.escape(nm)}(?![\w-])", ttext)) and \
            not re.search(r"(bench|eval|harness|grader|verifier|validat|judge)", nm + " " + (dest or ""))
        if not (target or named):
            continue
        pins = [m for m in GIT_PIN.finditer(tail) if not NON_PIN_REFS.match(m.group(1))]
        deepen = re.search(r"git\s+fetch\b[^;&|\n]*(--unshallow|--tags\b|--all\b)", tail)
        verb = "fetched" if line in fetched else "cloned"
        if (not shallow or deepen) and pins:
            if left:
                hits.append((line, f"{name} is {verb} with full history and checked out at {pins[0].group(1)}, and the "
                                   f"reflog and gc scrub leaves {left} pointing at later commits"))
            else:
                hits.append((line, f"{name} is {verb} with full history and then checked out at {pins[0].group(1)}"))
        elif not shallow and not single and branch and not NON_PIN_REFS.match(branch):
            hits.append((line, f"{name} is cloned at {branch} with the history of every other branch"))
    # the target package also installed from PyPI or from its moving git head
    for line, url, dest, shallow, branch, single, ri, pos, wd in clones:
        name = _norm(_repo_name(url))
        if len(name) < 3:
            continue
        for rj, ln, v in runs:
            for m in PIP_INSTALL.finditer(v):
                toks = [t.strip("'\"") for t in m.group(1).split()]
                prev = ""
                for t in toks:
                    if prev in ("-e", "--editable", "-r", "-c", "--requirement", "--constraint", "-t", "--target"):
                        prev = t
                        continue
                    prev = t
                    if t.startswith("-"):
                        continue
                    if t.startswith(("git+", "https://", "http://")):
                        repo_part = t.split("#", 1)[0]
                        if _norm(_repo_name(re.sub(r"@[^/@]*$", "", repo_part))) == name and \
                                not re.search(r"@[^/@]+$", repo_part.split("://", 1)[-1]):
                            hits.append((line, f"{_repo_name(url)} is also installed from its moving git head"))
                        continue
                    base = re.split(r"[<>=!~\[;]", t)[0]
                    if base and _norm(base) == name and not base.startswith((".", "/", "$", "{")):
                        hits.append((line, f"{_repo_name(url)} is also pip installed from PyPI, so a released copy "
                                           "sits in site-packages"))
    return hits


@rule("git-history-or-target-in-image", "hack", "high",
      "The agent's image keeps git history past the task's base commit or a released copy of the target package, "
      "so the reference fix can be read with git log or from site-packages.")
def r_git_history(pkg: Package):
    idx = index(pkg)
    out, details = [], []
    for p, role in idx.dockerfiles.items():
        if role == "agent":
            for line, detail in _git_in_dockerfile(idx.instrs[p], task_text(pkg, p)):
                out.append(ev(p, line))
                details.append(detail)
            for line, detail in _residue_in_dockerfile(idx.instrs[p]):
                out.append(ev(p, line))
                details.append(detail)
    # Python setup code that moves the sandbox repo to an older commit and keeps the newer history
    cands = list(funcs_with(pkg, GIT_PIN_LITERALS, SETUP_GIT_PIN))
    if cands and not any(SCRUB.search(pf.text) or "--depth" in pf.text for pf in py_with(pkg, SCRUB_LITERALS)):
        for pf, fi in cands:
            if SCORING_NAME.search(fi.name) or not _is_setup_name(fi.name) or \
                    re.search(r"(restore|cleanup|teardown|reset_tests|apply_patch|eval)", fi.name, re.I):
                continue
            for ln, txt, head in _box_calls(pf, fi):
                m = SETUP_GIT_PIN.search(txt)
                if m:
                    out.append(ev(pf, ln))
                    details.append(f"setup checks out {m.group(1)} in the sandbox repo and keeps the newer history")
    if not out:
        return []
    msg = details[0][0].upper() + details[0][1:]
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    return [F("git-history-or-target-in-image", msg + ", so git log --all or the installed package can show the "
              "reference fix.", out, confidence="review")]


# ---------------------------------------------------------------------------
# #3 answers, tests or grader readable from the agent's image
# evidence row 3: arxiv:2610.41014 (visible grader, Sec 4.1), arxiv:2608.22103 (planted keys, App C Table 2)
# ---------------------------------------------------------------------------

DEP_FILES = re.compile(r"(^|/)(requirements[\w.-]*\.txt|pyproject\.toml|uv\.lock|poetry\.lock|package(-lock)?\.json|"
                       r"setup\.(py|cfg)|Pipfile(\.lock)?|environment\.ya?ml|__init__\.py|pytest\.ini|tox\.ini)$")
SECRET_DIR = re.compile(r"(^|/)(hidden[\w-]*|held[-_]?out[\w-]*|private[-_]?tests?|verifier[-_]?data|verifier|grader|"
                        r"graders|answers?|answer[-_]?keys?|gold[\w-]*|ground[-_]?truth[\w-]*|expected[-_]?outputs?|"
                        r"oracle|solutions)(/|$)", re.I)
SECRET_FILE = re.compile(r"(^|/)(solve\.sh|[\w.-]*gold[\w.-]*\.(patch|diff|json|jsonl|txt|py)|[\w.-]*test[_.-]?patch[\w.-]*|"
                         r"answers?\.(json|jsonl|txt|csv)|answer[_-]?key[\w.-]*|expected[_-]?outputs?[\w.-]*|"
                         r"ground[_-]?truth[\w.-]*|grader\.py|grade\.py|compute_reward\.py|"
                         r"hidden[\w.-]*|verifier[\w.-]*\.(tar\.gz|tgz|zip|py|sh|json))$", re.I)
HARBOR_SECRET = re.compile(r"^(\./)?(tests|solution)(/|$)")
RUN_ABS_PATH = re.compile(r"/[\w.${}-]+(?:/[\w.${}-]+)*")
RUN_SECRET_TOKEN = re.compile(r"(?:^|[/_-])(hidden|held[-_]?out|answer[-_]?keys?|answers|gold|ground[-_]?truth|"
                              r"expected[-_]?outputs?|private[-_]?tests?)(?:$|[/_.-])", re.I)
SOLUTION_TOKEN = re.compile(r"(?:^|[/_.-])(solutions?|oracle|reference|gold)(?:$|[/_.-])", re.I)
RUN_WRITE_BEFORE = re.compile(r"(\btar\s+-?\w*c\w*f\s*|\bcp\s+(?:-\w+\s+)*\S+\s+|\bmv\s+(?:-\w+\s+)*\S+\s+|[^2&]>\s*|"
                              r"-O\s*|-o\s+|--output[= ]\s*|\bgit\s+clone\s+(?:-\S+\s+)*\S+\s+|\bunzip\b[^\n]*?-d\s*|"
                              r"\btar\b[^\n]*?-C\s*|--(?:public|hidden|out|output)[-_]?\w*[= ]\s*)$")
NOT_SECRET_PATH = re.compile(r"(site-packages|/usr/(lib|share|include)|node_modules|/\.cache|/proc/|/dev/|/logs/verifier|"
                             r"/etc/|\.whl$)", re.I)
ROOT_ONLY = re.compile(r"chmod\s+(-R\s+)?(0?[0-7]00|[ug]*[+=]\w*,?)*\s*(o-\w*r|go-\w*r|og-\w*r|a-\w*r|0?[4-7]00)")


def _rm_targets(v: str):
    for m in re.finditer(r"\brm\s+(?:-\w+\s+)*([^;&|\n]+)", v):
        for t in m.group(1).split():
            t = t.strip("'\"").rstrip("*").rstrip("/")
            if t and not t.startswith("-"):
                yield m.start(), t


def _removed(instrs, i: int, pos: int, path: str) -> bool:
    """Is path (or a parent) removed later in the same RUN after pos, or in a later RUN?"""
    p = path.rstrip("/")
    for j in range(i, len(instrs)):
        _, ins, v = instrs[j]
        if ins != "RUN":
            continue
        for at, t in _rm_targets(v):
            if j == i and at < pos:
                continue
            if t == p or p.startswith(t + "/"):
                return True
    return False


def _protected(instrs, i: int, path: str, flags: dict, agent_nonroot: bool) -> bool:
    if not agent_nonroot:
        return False
    mode = flags.get("chmod", "")
    if re.fullmatch(r"0?[0-7]{2}0", mode):
        return True
    p = path.rstrip("/")
    for _, ins, v in instrs[i:]:
        if ins == "RUN" and p in v and ROOT_ONLY.search(v):
            return True
    return False


def _readable_in_dockerfile(instrs, harbor_task):
    agent_nonroot = _final_user_nonroot(instrs) or (harbor_task is not None and not harbor_task.agent_root)
    hits = []
    for i, (ln, ins, v) in enumerate(instrs):
        if ins in ("COPY", "ADD"):
            flags, srcs, dest = copy_args(v)
            if "from" in flags or not dest:
                continue
            for s in srcs:
                s0 = s.strip()
                if DEP_FILES.search(s0) or s0.startswith(("http://", "https://")):
                    continue
                secret = SECRET_FILE.search(s0) or SECRET_DIR.search(s0) or re.search(r"\.(patch|diff)$", s0)
                harbor = harbor_task is not None and HARBOR_SECRET.match(s0)
                if not (secret or harbor):
                    continue
                landed = dest.rstrip("/") + "/" + os.path.basename(s0.rstrip("/")) if dest.endswith("/") else dest
                if any(_removed(instrs, i + 1, 0, x) or _protected(instrs, i + 1, x, flags, agent_nonroot)
                       for x in {dest, landed}):
                    continue
                hits.append((ln, f"{ins} {s0} to {dest}"))
        elif ins == "RUN":
            for m in RUN_ABS_PATH.finditer(v):
                pth = m.group(0).rstrip("/.")
                archive = bool(re.search(r"\b(tar\s+-?\w*c\w*f|zip\s+-\w*)\s*$", v[max(0, m.start() - 40): m.start()]))
                if not (RUN_SECRET_TOKEN.search(pth) or (archive and SOLUTION_TOKEN.search(pth))):
                    continue
                if NOT_SECRET_PATH.search(pth) or re.search(r"\$\(|\{\{", pth):
                    continue
                before = v[max(0, m.start() - 200): m.start()].split("\n")[-1]
                if not RUN_WRITE_BEFORE.search(before):
                    continue
                if _removed(instrs, i, m.end(), pth) or _protected(instrs, i, pth, {}, agent_nonroot):
                    continue
                hits.append((ln + v.count("\n", 0, m.start()), f"RUN leaves {pth} in the image"))
                break
    return hits


R2E_TESTS = re.compile(r"/r2e_tests\b")
LINK_LITERALS = ("r2e_tests", "hidden_test", "hidden-test", "private_test", "private-test", "eval_test", "eval-test",
                 "test_patch")
SETUP_LINK_TESTS = re.compile(r"\b(ln\s+-\w*s\w*|cp\s+-\w*r\w*|mv)\s+\S*(r2e_tests|hidden[_-]?tests?|private[_-]?tests?|"
                              r"eval[_-]?tests?|test_patch)\S*\s+\S*(testbed|workspace|/app|repo|/work|\{)", re.I)


@rule("answers-or-grader-readable", "hack", "critical",
      "Tests, reference solutions, expected outputs, grader code or hidden data are placed in the agent's image or "
      "sandbox where the agent can read them.")
def r_answers_readable(pkg: Package):
    idx = index(pkg)
    out, details = [], []
    for p, role in idx.dockerfiles.items():
        if role != "agent":
            continue
        tk = task_of(idx, p)
        for ln, d in _readable_in_dockerfile(idx.instrs[p], tk if tk is not None and tk.dockerfile == p else None):
            out.append(ev(p, ln))
            details.append(d)
    # Harbor tasks: a repository the verifier uses is cloned into the agent's image
    for tk in idx.tasks:
        if tk.dockerfile is None or idx.dockerfiles.get(tk.dockerfile) != "agent":
            continue
        pre = (tk.root + "/") if tk.root else ""
        vtext = "\n".join(t for p, t in pkg.files.items() if p.startswith(pre + "tests/"))
        if not vtext:
            continue
        for ln, ins, v in idx.instrs[tk.dockerfile]:
            if ins != "RUN":
                continue
            for m in GIT_CLONE.finditer(v):
                url, dest, *_ = _clone_info(m.group(1))
                d = dest.rstrip("/")
                if not d.startswith("/") or len(d) < 5 or WORKSPACE_DEST.match(d) or \
                        re.search(rf"(?<![\w./-]){re.escape(d.lower())}(?![\w-])", task_text(pkg, tk.dockerfile)) or \
                        _removed(idx.instrs[tk.dockerfile], 0, 10 ** 9, d):
                    continue  # the agent's own working copy, or gone before the agent starts
                if re.search(rf"{re.escape(d)}(/|\b)", vtext):
                    out.append(ev(tk.dockerfile, ln + v.count("\n", 0, m.start())))
                    details.append(f"the verifier's {_repo_name(url)} checkout at {dest} is cloned into the agent's image")
    # verifiers packages: hidden tests linked or copied into the workspace during sandbox setup
    for pf, fi in funcs_with(pkg, LINK_LITERALS, SETUP_LINK_TESTS):
        if SCORING_NAME.search(fi.name) or not _is_setup_name(fi.name):
            continue
        for ln, txt, head in _box_calls(pf, fi):
            m = SETUP_LINK_TESTS.search(txt)
            if m:
                out.append(ev(pf, ln))
                details.append(f"setup puts {m.group(2)} into the agent's workspace")
    # R2E-Gym images ship the hidden tests at /r2e_tests; they must be removed before the agent runs
    r2e = py_with(pkg, ("r2e_tests",))
    if r2e and not any(re.search(r"rm\s+-\w*\s+[^\n'\"]*/r2e_tests|rmtree\([^)]*r2e_tests", pf.text) for pf in r2e):
        for pf in r2e:
            m = R2E_TESTS.search(pf.text)
            if m:
                out.append(ev(pf, line_of(pf.text, m.start())))
                details.append("R2E-Gym images keep the hidden tests at /r2e_tests and nothing removes them before "
                               "the agent runs")
                break
    # v0.5: labels of the scored split copied into the agent's image
    o2, d2 = _labels_branch(pkg, idx)
    out += o2
    details += d2
    if not out:
        return []
    msg = details[0][0].upper() + details[0][1:]
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    return [F("answers-or-grader-readable", msg + ", so the agent can read the answer key or the grader instead "
              "of solving the task.", out, confidence="review")]


# ---------------------------------------------------------------------------
# #4 reward channel the agent can write
# evidence row 4: RewardProof (openreview vxftf9je8s, Table 3: 123 reward file writes, 91 processes surviving)
# ---------------------------------------------------------------------------

REWARD_LITERALS = ("/logs/verifier", "reward.txt", "reward.json", "grade.json")
KILL_LITERALS = ("kill", "reap", "terminate", "stop_agent")
REWARD_PATH = re.compile(r"(/logs/verifier\b|\breward\.(txt|json)\b|\bgrade\.json\b)")
KILL_AGENT = re.compile(r"(pkill\s+(-\w+\s+)*-u\b|kill\s+-(9|KILL|TERM)\s+-1\b|killall5|kill_agent\w*|kill_all\w*|"
                        r"reap\w*_processes|terminate_agent\w*|stop_agent_processes|pkill\s+-9\s+-v|"
                        r"ps\s+[^\n]*\|\s*xargs\s+(-r\s+)?kill)", re.I)
AGENT_WRITABLE = r"(?:/app|/workspace|/testbed|/home|\$HOME|~|/tmp|/output\w*|/submission\w*|/repo)\b"
REWARD_DEST = r"(?:/logs/verifier/reward\.(?:txt|json)|\$\{?\w*REWARD\w*\}?)"
COPY_TO_REWARD = re.compile(rf"\b(?:cp|mv|cat|ln\s+-\w*s)\s+(?:-\w+\s+)*[\"']?({AGENT_WRITABLE}[^\s\"';|&>]*)[\"']?\s*"
                            rf"(?:>>?)?\s*[\"']?{REWARD_DEST}")
REWARD_WRITE = re.compile(rf">\s*[\"']?{REWARD_DEST}")


NOT_EXISTS = re.compile(r"(\[\[?\s*!\s*-[fse]\s+[^\]]*(reward|REWARD)|test\s+!\s+-[fse]\s+\S*(reward|REWARD)|"
                        r"(\[\[?\s*-[fse]|test\s+-[fse])\s+\S*(reward|REWARD)\S*\s*\]?\]?\s*\|\|)")


def _reward_writes(lines):
    """[(line, guarded)] for each reward file write; guarded means it only runs when no reward file exists."""
    out, stack = [], []
    for ln, line in lines:
        s = line.strip()
        opened = re.match(r"(el)?if\b", s)
        if re.match(r"elif\b|else\b", s) and stack:
            stack[-1] = False
        if opened and not s.startswith("elif"):
            stack.append(bool(NOT_EXISTS.search(s.split("then", 1)[0])))
        elif opened:
            stack[-1] = bool(NOT_EXISTS.search(s.split("then", 1)[0])) if stack else False
        if REWARD_WRITE.search(s):
            body = s.split("then", 1)[1] if opened and "then" in s else s
            guarded = bool(NOT_EXISTS.search(s)) or (bool(stack) and stack[-1] and (not opened or "then" in s))
            out.append((ln, guarded))
            if opened and re.search(r"\bfi\s*$", body):
                stack.pop()
                continue
        if re.search(r"(^|;|\s)fi\s*$", s) and stack and not (opened and not re.search(r"\bfi\s*$", s)):
            stack.pop()
    return out


def _written_before(lines, upto: int, path: str) -> bool:
    pat = re.compile(rf"(>|\btee\b|-o\s|--output\w*[= ]|\bcp\s+\S+\s+|\bmv\s+\S+\s+)\s*[\"']?{re.escape(path)}")
    return any(pat.search(l) for ln, l in lines if ln < upto)


@rule("writable-reward-channel", "hack", "critical",
      "The reward is read from a file the agent can write, either after a verifier that ran in the agent's box "
      "with the agent's processes still alive or from a path the agent owns.")
def r_reward_channel(pkg: Package):
    idx = index(pkg)
    out, details = [], []
    # (a) a verifiers port reads the reward file back out of the agent's sandbox
    cands = list(funcs_with(pkg, REWARD_LITERALS))
    if cands and not idx.py_isolated:
        verifier_sh = "\n".join(t for p, t in pkg.text_files(".sh") if re.search(r"(^|/)tests?/", p))
        killed = any(KILL_AGENT.search(pf.text) for pf in py_with(pkg, KILL_LITERALS)) or \
            (any(x in verifier_sh for x in KILL_LITERALS) and KILL_AGENT.search(verifier_sh))
        gf = idx.grader if not killed else []
        on_path = {(pf.path, fi.start) for pf, fi in gf}
        if gf and not grades_in_fresh_box(gf):
            for pf, fi in cands:
                if (pf.path, fi.start) not in on_path:
                    continue
                ctx = fi.src + "\n" + referenced_consts(pf, fi)
                if not re.search(r"(test\.sh|run_tests|run-tests|verify\.sh|verifier_command|run_verifier|\bpytest\b)", ctx):
                    continue
                for ln, txt, head in _box_calls(pf, fi):
                    if REWARD_PATH.search(txt) and not re.search(r"(upload|write|put)", head, re.I) and \
                            re.search(r"(\bcat\b|\bhead\b|\btail\b|read|download|get_file|\[\s*-s\b|test\s+-f)", txt + " " + head):
                        out.append(ev(pf, ln))
                        details.append("the reward file is read back from the agent's sandbox after the verifier ran "
                                       "there, and nothing stops the agent's processes first")
                        break
    # (b) a verifier script turns an agent-owned file into the reward, or only writes a reward when none exists
    for tk in idx.tasks:
        pre = (tk.root + "/") if tk.root else ""
        memo = []

        def helpers_text():
            """Text of the verifier's helper files (what tests/ writes is not agent owned)."""
            if not memo:
                memo.append("\n".join(t for p, t in pkg.files.items()
                                       if p.startswith(pre + "tests/") and not p.endswith("test.sh")))
            return memo[0]

        for sp in tk.scripts:
            t = pkg.files.get(sp, "")
            if "reward" not in t.lower():
                continue
            lines = sh_lines(t)
            for ln, line in lines:
                m = COPY_TO_REWARD.search(line)
                if not m:
                    continue
                src = m.group(1)
                if _written_before(lines, ln, src) or src in helpers_text():
                    continue
                out.append(ev(sp, ln))
                details.append(f"test.sh copies {src}, a path the agent can write, into the reward file")
            writes = _reward_writes(lines)
            if writes and all(g for _, g in writes) and \
                    not re.search(r"(write_text|json\.dump|open\([^)]*reward[^)]*['\"]w|-o\s+\S*reward|--output\S*\s+\S*reward)",
                                  t + helpers_text()):
                out.append(ev(sp, writes[0][0]))
                details.append("test.sh writes the reward only when no reward file exists yet")
    # v0.5: (c) reward.json planted by code the verifier runs, (d) expected files or validators the agent can
    # edit, (e) a verdict the agent's own tool stores in state
    for branch in (_reward_json_branch, _writable_expected_branch, _state_verdict_branch):
        o2, d2 = branch(pkg, idx)
        out += o2
        details += d2
    if not out:
        return []
    msg = details[0][0].upper() + details[0][1:]
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    return [F("writable-reward-channel", msg + ", so a planted file or a background process can set the score.",
              out, confidence="review")]


# ---------------------------------------------------------------------------
# #8 verify time fetch through binaries the agent controls
# evidence row 8: rdi.berkeley.edu/blog/trustworthy-benchmarks-cont (82 of 89 Terminal-Bench tasks, fake curl won 89 of 89)
# ---------------------------------------------------------------------------

FETCH = [
    ("a curl to shell installer", re.compile(r"\b(curl|wget)\b[^|;&\n]*\|\s*(sudo\s+)?(env\s+\S+\s+)*(ba|z|da)?sh\b")),
    ("pip install", re.compile(r"\b(pip3?|python3?\s+-m\s+pip|uv\s+pip)\s+install\b(?![^;&|\n]*(--no-index|--offline))")),
    ("uvx", re.compile(r"(^|[\s;&|(])uvx\s(?![^;&|\n]*--offline)")),
    ("uv run --with", re.compile(r"\buv\s+run\b[^;&|\n]*--with\b(?![^;&|\n]*--offline)")),
    ("uv tool install", re.compile(r"\buv\s+tool\s+install\b")),
    ("apt-get install", re.compile(r"\b(apt-get|apt|apk|yum|dnf|microdnf)\s+(-\S+\s+)*(install|add)\b")),
    ("npm install", re.compile(r"\b(npm\s+(install|i|ci)\b|npx\s+(-y\s+)?[\w@])")),
    ("download", re.compile(r"\b(curl|wget)\b[^;&|\n]*https?://")),
    ("git clone", re.compile(r"\bgit\s+clone\b[^;&|\n]*(https?://|git@)")),
]
STRONG_FETCH = {"a curl to shell installer", "pip install", "uvx", "uv run --with", "uv tool install"}
FETCH_LITERALS = ("install", "curl", "wget", "uvx", "uv run", "git clone", "_ENSURE_UV", "ensure_uv", "npx ",
                  "command -v")
BOX_LITERALS = ("execute_command", "run_command", "exec_command", "run_background_job", "sandbox", "runtime",
                "container", "session", ".exec", "_exec")
ENSURE_TOOL = re.compile(r"(_ENSURE_UV|ensure_uv|install_uv\b|command\s+-v\s+(uv|uvx|pytest|pip3?)\b[^\n]*(\|\||then)|"
                         r"which\s+(uv|uvx|pytest)\s*(\|\||;)\s*(curl|pip))")


def _fetches(text: str):
    """[(line, kind)] for verify time fetches in a script, strongest kind first (FETCH order)."""
    out = []
    for ln, line in sh_lines(text):
        for rank, (name, pat) in enumerate(FETCH):
            if pat.search(line):
                out.append((rank, ln, name))
                break
    return [(ln, name) for rank, ln, name in sorted(out)]


def _path_dirs_writable(idx: Index, tk: HTask) -> bool:
    for _, ins, v in idx.instrs.get(tk.dockerfile or "", []):
        if ins == "RUN" and re.search(r"chmod\s+(-R\s+)?(0?777|a\+w|o\+w)\s+[^\n]*(/usr/local/bin|/usr/bin|/bin|/usr/local)\b", v):
            return True
    return False


@rule("verify-time-fetch", "hack", "high",
      "The verifier downloads or installs tools at grading time inside the agent's box, so a curl, pip or uv the "
      "agent replaced runs as part of the grader.")
def r_verify_fetch(pkg: Package):
    idx = index(pkg)
    out, kinds, n_tasks, strong = [], [], 0, False
    isolated = idx.has_py and (idx.py_isolated or (any(tk.scripts for tk in idx.tasks) and grades_in_fresh_box(idx.grader)))
    for tk in idx.tasks:
        if not tk.scripts or not tk.shared or isolated:
            continue
        if not tk.agent_root and not _path_dirs_writable(idx, tk):
            continue
        for sp in tk.scripts:
            f = _fetches(pkg.files.get(sp, ""))
            if f:
                n_tasks += 1
                out.append(ev(sp, f[0][0]))
                kinds.append(f[0][1])
                strong = strong or (f[0][1] in STRONG_FETCH and tk.agent_root)
                break
    # verifiers packages that fetch or install tools inside the agent's sandbox on the scoring path
    py = []
    cands = [(pf, fi) for pf, fi in funcs_with(pkg, FETCH_LITERALS) if any(x in fi.src for x in BOX_LITERALS)]
    gf = idx.grader if cands and not idx.py_isolated else []
    on_path = {(pf.path, fi.start) for pf, fi in gf}
    if gf and not grades_in_fresh_box(gf):
        for pf, fi in cands:
            if (pf.path, fi.start) not in on_path or (_is_setup_name(fi.name) and not GRADER_NAME.search(fi.name)):
                continue
            for ln, txt, head in _box_calls(pf, fi):
                if re.search(r"(upload|write|put|download|read)", head, re.I):
                    continue
                if ENSURE_TOOL.search(txt + " " + pf.line(ln)):
                    py.append(ev(pf, ln))
                    kinds.append("a tool install probe")
                    continue
                for name, p2 in FETCH:
                    if p2.search(txt):
                        py.append(ev(pf, ln))
                        kinds.append(name)
                        break
    res = []
    off, off_kinds = _offline_fetch_branch(pkg, idx)
    if off:
        w = ", ".join(sorted(set(off_kinds), key=off_kinds.index)[:3])
        res.append(F("verify-time-fetch", f"{len(off)} task verifier script(s) run {w} while task.toml gives the verify "
                     "phase no network, so the verifier fails and correct work gets no score.", off,
                     confidence="review", kind="broken"))
    if not out and not py:
        return res
    what = ", ".join(sorted(set(kinds), key=kinds.index)[:3])
    if out:
        msg = f"{n_tasks} task verifier script(s) run {what} at grading time in the agent's box"
    else:
        msg = f"Scoring code runs {what} inside the agent's sandbox"
    # likely: a Harbor task verifier, shared with a root agent, that pipes an installer to a shell or installs
    # Python packages (21 hits on the labeled set, Harbor examples and adapters, and the Hub sample, all true
    # positives on hand check, see the internal notes). Everything else stays review.
    return [F("verify-time-fetch", msg + ", so a curl, pip or uv binary the agent planted decides the grade.",
              out + py, confidence="likely" if strong else "review")] + res


# ---------------------------------------------------------------------------
# #9 (stretch) agent phase egress to the answer, or a port that opens the network a task closed
# evidence row 9: arxiv:2610.24206 (237 of 351 solves held with GitHub blocked, Sec 3), OpenAI and HF incident report
# ---------------------------------------------------------------------------

NET_FORCED = re.compile(r"\b(network_access|allow_internet|internet_access|enable_network|network_enabled)\s*[=:]\s*True\b")


@rule("egress-and-shared-state", "hack", "medium",
      "The agent phase has internet while the task's upstream repository is public, or a port turns on network "
      "access that the task declares off.")
def r_egress(pkg: Package):
    idx = index(pkg)
    out, details = [], []
    swe_open, closed = [], []
    for tk in idx.tasks:
        md = tk.cfg.get("metadata", {}) if isinstance(tk.cfg.get("metadata"), dict) else {}
        repo = md.get("repository_url") or md.get("repo_url") or md.get("repo")
        base = md.get("base_commit_hash") or md.get("base_commit")
        if tk.internet is True and isinstance(repo, str) and re.search(r"github\.com|gitlab\.com", repo) and base:
            swe_open.append(tk.toml)
        if tk.internet is False:
            closed.append(tk.toml)
    if swe_open:
        t = pkg.files.get(swe_open[0], "")
        m = re.search(r"allow_internet\s*=\s*true", t)
        out.append(ev(swe_open[0], line_of(t, m.start()) if m else 1))
        details.append(f"{len(swe_open)} task(s) pin a public repository at a base commit and allow internet in the "
                       "agent phase, so the upstream fix can be fetched")
    if closed:
        for pf in py_with(pkg, ("True",)):
            if not any(x in pf.text for x in ("network", "internet")):
                continue
            m = NET_FORCED.search(pf.text)
            if m and not re.search(r"\bif\b|\belse\b|\band\b|\bor\b", pf.line(line_of(pf.text, m.start()))):
                out.append(ev(pf, line_of(pf.text, m.start())))
                details.append(f"the port forces network access on while {len(closed)} task(s) declare "
                               "allow_internet = false")
                break
    # v0.5: the instruction says the agent is offline but the agent phase has internet
    claims = _offline_claim_branch(pkg, idx)
    if claims:
        out.append(ev(*claims[0]))
        details.append(f"{len(claims)} task instruction(s) tell the agent it has no internet while task.toml gives the "
                       "agent phase network access, so only the prompt keeps it from fetching answers or packages")
    if not out:
        return []
    msg = details[0][0].upper() + details[0][1:]
    if len(out) > 1:
        msg += f" (+{len(out) - 1} more)"
    return [F("egress-and-shared-state", msg + ".", out, confidence="review")]


# ---------------------------------------------------------------------------
# v0.5 helpers shared by the widened rules and the new privilege rule
# ---------------------------------------------------------------------------

import fnmatch  # noqa: E402
import posixpath  # noqa: E402

WS_ROOTS = r"(?:/app|/workspace|/testbed|/repo|/code|/project|/src|/work|/home/[\w.-]+|/root(?=/[^.]))"


def image_files(pkg: Package, dockerfile: str, instrs) -> dict:
    """{path in the image: (dockerfile line, package path, COPY flags)} for files the Dockerfile copies from the
    package (build context = the Dockerfile's folder)."""
    ctx = os.path.dirname(dockerfile)
    pre = ctx + "/" if ctx else ""
    names = [p for p in pkg.sizes if p.startswith(pre)] if pre else list(pkg.sizes)
    workdir, out = "/", {}
    for ln, ins, v in instrs:
        if ins == "FROM":
            workdir = "/"
        elif ins == "WORKDIR":
            w = v.strip().strip("'\"")
            workdir = w if w.startswith("/") else posixpath.normpath(posixpath.join(workdir, w))
        elif ins in ("COPY", "ADD"):
            flags, srcs, dest = copy_args(v)
            if "from" in flags or not dest or dest.startswith("$"):
                continue
            d = dest if dest.startswith("/") else posixpath.normpath(posixpath.join(workdir, dest))
            into_dir = dest.endswith("/") or dest in (".", "./") or len(srcs) > 1
            for s in srcs:
                s0 = re.sub(r"^\./", "", s.strip()).rstrip("/")
                if s0.startswith(("http://", "https://", "/", "$")):
                    continue
                base = (pre + s0) if s0 not in ("", ".") else ctx
                if any(ch in s0 for ch in "*?["):
                    for p in names:
                        if fnmatch.fnmatch(p, pre + s0):
                            out[posixpath.join(d, os.path.basename(p))] = (ln, p, flags)
                    continue
                if base in pkg.sizes:
                    landed = posixpath.join(d, os.path.basename(base)) if into_dir else d
                    out[landed] = (ln, base, flags)
                    continue
                for p in names:
                    if not base or p.startswith(base + "/"):
                        out[posixpath.join(d, p[len(base) + 1:] if base else p)] = (ln, p, flags)
    return out


def tests_text(pkg: Package, tk: HTask) -> str:
    """All text under the task's tests/ folders (test.sh, test files, helpers)."""
    pre = (tk.root + "/") if tk.root else ""
    return "\n".join(t for p, t in pkg.files.items()
                     if p.startswith(pre) and re.match(r"(steps/[^/]+/)?tests/", p[len(pre):]))


def instruction_text(pkg: Package, tk: HTask) -> str:
    pre = (tk.root + "/") if tk.root else ""
    return "\n".join(pkg.files.get(pre + n, "") for n in ("instruction.md", "task.md", "TASK.md", "prompt.md"))


def _net_of(table) -> str | None:
    if not isinstance(table, dict):
        return None
    m = table.get("network_mode")
    if isinstance(m, str):
        return m.lower()
    a = table.get("allow_internet")
    if isinstance(a, bool):
        return "public" if a else "no-network"
    return None


def phase_net(tk: HTask, role: str) -> str | None:
    """Network of the agent or verify phase, following Harbor's resolution: an explicit [agent] or [verifier]
    network_mode wins, a separate verifier uses its own environment (public by default), and a shared verifier
    gets the task environment's baseline (public by default). None when task.toml did not parse."""
    cfg = tk.cfg
    if not cfg:
        return None
    env = cfg.get("environment") if isinstance(cfg.get("environment"), dict) else {}
    base = _net_of(env) or "public"
    table = cfg.get(role) if isinstance(cfg.get(role), dict) else {}
    explicit = table.get("network_mode") if isinstance(table.get("network_mode"), str) else None
    if explicit:
        return explicit.lower()
    if role == "verifier" and not tk.shared:
        venv = table.get("environment") if isinstance(table.get("environment"), dict) else {}
        return _net_of(venv) or "public"
    return base


# ---------------------------------------------------------------------------
# v0.5 widening of writable-reward-channel
# ---------------------------------------------------------------------------

# reward-file-semantics-harbor + stale-reward-file-not-cleared (TB #2086 probe, NeMo Gym task folders): Harbor
# reads reward.json before reward.txt, so code the verifier runs can plant reward.json and beat the reward.txt
# that test.sh writes afterwards.
CAND_SH = re.compile(
    rf"(\b(?:python3?|node|bash|sh|ruby|perl|php|Rscript|java\s+-jar|deno\s+run|bun)\s+(?:-\S+\s+)*['\"]?"
    rf"(?:{WS_ROOTS}/|\./)[\w./-]+|"
    rf"(?:^|[;&|(]\s*|\bthen\s+|\bdo\s+)['\"]?{WS_ROOTS}/[\w./-]+['\"]?(?=\s|$|;)|"
    r"\b(?:go\s+test|cargo\s+(?:test|run)|npm\s+(?:run\s+)?test|yarn\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|"
    r"npx\s+(?:jest|vitest|mocha)|mvn\s+(?:\S+\s+)*test|gradlew?\s+(?:\S+\s+)*test|make\s+(?:test|check)\b|ctest|rspec|"
    r"mix\s+test|dotnet\s+test|phpunit|swift\s+test|stack\s+test|cabal\s+test)\b)")
PYTEST_SH = re.compile(r"(?:^|[\s;&|(])(?:python3?\s+-m\s+)?pytest\b([^;&|\n]*)")
CAND_PY_CALL = re.compile(r"(subprocess\.\w+\(|os\.system\(|os\.popen\(|sys\.path\.(?:insert|append)\(|importlib\.\w+|"
                          r"runpy\.\w+\(|spec_from_file_location\(|\bexec\(|pexpect\.\w+\()")
WS_LITERAL = re.compile(rf"['\"]{WS_ROOTS}(/|['\"])")
OTHER_USER = re.compile(r"(^|[\s;&|(])(su\s+-?\s*\w|runuser\b|sudo\s+(-\S+\s+)*-u\b|setpriv\b|chroot\b|gosu\b|setuidgid\b)")


def _runs_candidate(pkg: Package, tk: HTask, lines):
    """Line in test.sh where code the agent wrote or can change runs, or None."""
    pre = (tk.root + "/") if tk.root else ""
    ws_cd = False
    for ln, line in lines:
        if re.search(rf"\bcd\s+['\"]?(?:{WS_ROOTS}|\$\w+|\.\.)", line):
            ws_cd = True
        if CAND_SH.search(line) or (ws_cd and re.search(r"(?:^|[;&|(]\s*|\bthen\s+)\./[\w.-]+", line)):
            return ln
        m = PYTEST_SH.search(line)
        if m:
            targets = [t.strip("'\"") for t in m.group(1).split() if not t.startswith("-")]
            if any(re.match(rf"{WS_ROOTS}/", t) for t in targets) or (ws_cd and not targets):
                return ln
            for t in targets:
                if t.startswith("/tests/"):
                    src = pkg.files.get(pre + "tests/" + t[len("/tests/"):].split("::")[0], "")
                    if src and any(WS_LITERAL.search(src, m.start(), m.end() + 200) for m in CAND_PY_CALL.finditer(src)):
                        return ln
    return None


def _reward_json_branch(pkg: Package, idx: Index):
    out, details = [], []
    for tk in idx.tasks:
        if not tk.scripts:
            continue
        pre = (tk.root + "/") if tk.root else ""
        helpers = "\n".join(t for p, t in pkg.files.items()
                            if p.startswith(pre + "tests/") and p not in tk.scripts)
        for sp in tk.scripts:
            t = pkg.files.get(sp, "")
            if "reward.txt" not in t or "reward.json" in t or "reward.json" in helpers or OTHER_USER.search(t):
                continue
            lines = sh_lines(t)
            if not any(REWARD_WRITE.search(l) for _, l in lines):
                continue
            ln = _runs_candidate(pkg, tk, lines)
            if ln is None:
                continue
            out.append(ev(sp, ln))
            details.append("test.sh runs code the agent controls and then writes only reward.txt, while Harbor reads "
                           "a reward.json that this code can leave in /logs/verifier first")
            break
    return out, details


# state-in-agent-writable-store + writable-expected-or-validator-file (arxiv:2606.09863, consensus w:d4af36d2):
# the verifier compares against an expected file, a baseline or a validator that sits in the agent's box.
EXPECT_DATA = re.compile(r"(expected|reference|(^|[_.-])ref([_.-]|$)|golden|(^|[_.-])gold|answer|baseline|threshold|"
                         r"ground[_-]?truth|(^|[_.-])truth|oracle|correct[_-]?output|target[_-]?output)", re.I)
VALIDATOR = re.compile(r"(validat|checker|(^|[_.-])check([_.-]|$)|verif(y|ier)|grader|(^|[_.-])grade|evaluat|scorer|"
                       r"(^|[_.-])score([_.-]|$)|judge)", re.I)
HASHED = re.compile(r"(hashlib|sha\d+sum|md5sum|\bcksum\b|sha256|md5\(|checksum)", re.I)
DONT_MODIFY = re.compile(r"(do\s+not|don'?t|must\s+not|should\s+not|never|cannot|can'?t|not\s+allowed\s+to)\s+"
                         r"(be\s+)?(modif\w*|chang\w*|edit\w*|touch\w*|alter\w*|overwrit\w*|tamper\w*)|read[- ]only|"
                         r"leave\s+[^.\n]{0,60}\s+unchanged|(is|are)\s+(fixed|frozen|immutable)", re.I)


def _agent_can_write(instrs, tk: HTask, path: str, flags: dict) -> bool:
    if not tk.agent_root or _final_user_nonroot(instrs):
        if flags.get("chown"):
            return True
        p = re.escape(path.rstrip("/"))
        d = re.escape(posixpath.dirname(path))
        return any(ins == "RUN" and re.search(rf"(chmod\s+(-R\s+)?(0?777|a\+w|o\+w|\+w)|chown\s+(-R\s+)?\S+)\s+[^\n;&]*({p}|{d})\b", v)
                   for _, ins, v in instrs)
    return True


def _protected_copy(pkg: Package, tk: HTask, path: str, ttext: str, tsh: str) -> bool:
    name = os.path.basename(path)
    pre = (tk.root + "/") if tk.root else ""
    stem = os.path.splitext(name)[0].lower()
    if any(p.startswith(pre + "tests/") and (os.path.basename(p) == name or
                                              (len(stem) >= 4 and stem in os.path.basename(p).lower()))
           for p in pkg.sizes):
        return True  # the verifier ships its own copy (weights.pt next to /tests/weights_gtruth.pt)
    for m in HASHED.finditer(ttext):
        if name in ttext[max(0, m.start() - 600): m.end() + 600]:
            return True
    return bool(re.search(rf"\b(cp|install|rsync)\b[^\n;&]*/tests/[^\n;&]*{re.escape(path)}", tsh))


def _reads(ttext: str, path: str) -> bool:
    if re.search(rf"{re.escape(path)}(?![\w.-])", ttext):
        return True
    d, stem = posixpath.dirname(path), os.path.splitext(os.path.basename(path))[0]
    if path.endswith(".py") and re.search(rf"sys\.path\.(insert|append)\([^)]*['\"]{re.escape(d)}/?['\"]", ttext):
        return bool(re.search(rf"^\s*(from\s+{re.escape(stem)}\s+import|import\s+{re.escape(stem)}\b)", ttext, re.M))
    return False


def _dont_modify_names(instr: str) -> set:
    out = set()
    for para in re.split(r"\n|(?<=[.!?])\s+", instr):
        if DONT_MODIFY.search(para):
            out |= set(re.findall(r"[`'\"]?(/?[\w./-]+\.\w{1,5})\b", para))
    return out


def _writable_expected_branch(pkg: Package, idx: Index):
    out, details = [], []
    for tk in idx.tasks:
        if not tk.shared or not tk.scripts or not tk.dockerfile or idx.dockerfiles.get(tk.dockerfile) != "agent":
            continue
        instrs = idx.instrs[tk.dockerfile]
        landed = image_files(pkg, tk.dockerfile, instrs)
        if not landed:
            continue
        ttext = tests_text(pkg, tk)
        tsh = "\n".join(pkg.files.get(s, "") for s in tk.scripts)
        readable = {ln for ln, _ in _readable_in_dockerfile(instrs, tk)}
        dont = _dont_modify_names(instruction_text(pkg, tk))
        hit = None
        for path, (ln, src, flags) in sorted(landed.items()):
            if path.startswith(("/tests", "/logs", "/solution", "/oracle")) or ln in readable:
                continue
            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]
            told = any(d == path or (not d.startswith("/") and os.path.basename(d) == name) for d in dont)
            if VALIDATOR.search(stem) and re.search(r"\.(py|sh|js|ts|rb|pl|lua)$", name):
                kind = "validator"
            elif EXPECT_DATA.search(stem):
                kind = "expected"
            else:
                kind = None
            if not (kind or told):
                continue
            if not _reads(ttext, path) or not _agent_can_write(instrs, tk, path, flags) or \
                    _protected_copy(pkg, tk, path, ttext, tsh) or _removed(instrs, 0, 10 ** 9, path):
                continue
            what = {"validator": "the validator", "expected": "the expected or baseline file"}.get(kind, "the file")
            hit = (ln, f"the tests trust {what} {path}, which the Dockerfile puts in the agent's box where the agent can "
                       "change it" + (", and only the instruction says not to" if told else ""))
            break
        if hit:
            out.append(ev(tk.dockerfile, hit[0]))
            details.append(hit[1])
    return out, details


VERDICT_ASSIGN = re.compile(r"state\[\s*['\"](success|succeeded|successful|solved|is_solved|passed|pass|won|win|winner|"
                            r"is_correct|correct|score|reward|task_complete|task_completed|completed|goal_reached|"
                            r"achieved|verdict)['\"]\s*\]\s*=(?!=)", re.I)
VERDICT_KEY = re.compile(r"^(success|succeeded|successful|solved|is_solved|passed|pass|won|win|winner|is_correct|correct|"
                         r"score|reward|task_complete|task_completed|completed|goal_reached|achieved|verdict)$", re.I)


def _state_verdict_branch(pkg: Package, idx: Index):
    """A tool stores its own argument as the verdict in state, and the reward returns that state value."""
    files = [pf for pf in py_with(pkg, ("state[",)) if VERDICT_ASSIGN.search(pf.text)]
    tools = tool_names(pkg) if files else set()
    if not tools:
        return [], []
    set_by_tool = {}
    for pf in files:
        for fi in pf.funcs:
            if fi.name not in tools:
                continue
            params = set(fi.params) - {"self", "state", "kwargs", "args"}
            for n in _own_nodes(fi.node):
                if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript)):
                    continue
                tgt = n.targets[0]
                key = _const_text(tgt.slice)
                if not (isinstance(tgt.value, ast.Name) and tgt.value.id == "state") or not key or not VERDICT_KEY.match(key):
                    continue
                val = n.value
                while isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and \
                        val.func.id in ("bool", "float", "int", "str") and len(val.args) == 1:
                    val = val.args[0]
                if isinstance(val, ast.Name) and val.id in params:
                    set_by_tool.setdefault(key, (pf, n.lineno, fi.name))
    if not set_by_tool:
        return [], []
    for pf, fi in idx.grader:
        if fi.name in tools:
            continue
        for n in _own_nodes(fi.node):
            key = None
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "state" and \
                    isinstance(n.ctx, ast.Load):
                key = _const_text(n.slice)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and \
                    isinstance(n.func.value, ast.Name) and n.func.value.id == "state" and n.args:
                key = _const_text(n.args[0])
            if key in set_by_tool:
                tpf, tln, tname = set_by_tool[key]
                return [ev(pf, n.lineno), ev(tpf, tln)], [
                    f"the reward reads state[{key!r}], which the agent's {tname} tool sets straight from its own argument"]
    return [], []


# ---------------------------------------------------------------------------
# v0.5 widening of answers-or-grader-readable
# labels-or-cached-results-in-workspace (9evfmndm7i): the scored split's labels sit in the agent's image
# ---------------------------------------------------------------------------

SCORED_SPLIT = re.compile(r"(^|[_.-])(test|testing|eval|evaluation|holdout|hold[_-]out|hidden|private|val|valid|"
                          r"validation|dev)([_.-]|\d|$)", re.I)
LABEL_COLS = {"label", "labels", "target", "targets", "y", "class", "category", "answer", "answers", "gold",
              "ground_truth", "truth", "outcome", "survived", "sentiment", "is_fraud", "fraud", "churn",
              "price", "saleprice", "rating", "y_true", "true_label", "gold_label", "expected"}
ID_COLS = re.compile(r"^(id|idx|index|uid|row_?id|\w+_id|name|key|text|question|prompt|input|date|time|timestamp)$", re.I)


def data_header(text: str, path: str) -> list:
    """Column names of a csv, tsv, jsonl or json file from its first row (empty when unknown)."""
    try:
        if path.endswith((".csv", ".tsv")):
            first = text.lstrip("﻿").split("\n", 1)[0]
            sep = "\t" if path.endswith(".tsv") or first.count("\t") > first.count(",") else ","
            return [c.strip().strip('"\'') for c in first.split(sep) if c.strip()]
        if path.endswith(".jsonl"):
            for ln in text.split("\n", 20)[:20]:
                if ln.strip():
                    row = json.loads(ln)
                    return list(row) if isinstance(row, dict) else []
        if path.endswith(".json"):
            m = re.match(r"\s*\[\s*(\{.*?\})\s*[,\]]", text[:20000], re.S)
            if m:
                row = json.loads(m.group(1))
                return list(row) if isinstance(row, dict) else []
    except Exception:
        return []
    return []


def _labels_branch(pkg: Package, idx: Index):
    out, details = [], []
    for tk in idx.tasks:
        if not tk.dockerfile or idx.dockerfiles.get(tk.dockerfile) != "agent":
            continue
        instrs = idx.instrs[tk.dockerfile]
        if not any(ins in ("COPY", "ADD") for _, ins, _ in instrs):
            continue
        landed = image_files(pkg, tk.dockerfile, instrs)
        data = [(p, v) for p, v in landed.items() if v[1].endswith((".csv", ".tsv", ".jsonl", ".json"))
                and v[1] in pkg.files and SCORED_SPLIT.search(os.path.basename(v[1]))
                and not re.search(r"train", os.path.basename(v[1]), re.I)]
        if not data:
            continue
        pre = (tk.root + "/") if tk.root else ""
        gt_cols = set()
        for p, t in pkg.files.items():
            if p.startswith(pre + "tests/") and p.endswith((".csv", ".tsv", ".jsonl", ".json")):
                h = data_header(t, p)
                if 1 <= len(h) <= 4:
                    gt_cols |= {c for c in h if not ID_COLS.match(c)}
        instr = instruction_text(pkg, tk).lower()
        ttext = tests_text(pkg, tk)
        for path, (ln, src, flags) in sorted(data):
            cols = data_header(pkg.files[src], src)
            if len(cols) < 2:
                continue
            leak = [c for c in cols if c in gt_cols]
            if not leak:
                leak = [c for c in cols if c.lower() in LABEL_COLS and
                        re.search(rf"(predict|classif|estimat|forecast)\w*[^.\n]{{0,80}}[`'\"]?{re.escape(c.lower())}\b", instr)
                        and re.search(rf"['\"]{re.escape(c)}['\"]", ttext)]
            if leak and not _removed(instrs, 0, 10 ** 9, path):
                out.append(ev(tk.dockerfile, ln))
                details.append(f"{path} in the agent's image holds the {leak[0]} column of the scored split that the "
                               "tests grade")
                break
    return out, details


# ---------------------------------------------------------------------------
# v0.5 widening of verify-time-fetch
# grading-time-install-and-network (arxiv:2608.17393): the verify phase has no network but test.sh downloads or
# installs, so the verifier cannot run and correct work gets no score.
# ---------------------------------------------------------------------------

def _pkg_names(line: str):
    m = re.search(r"\b(?:install|add)\b(.*)$", line)
    if not m:
        return []
    out = []
    for t in m.group(1).split():
        t = t.strip("'\"")
        if not t or t.startswith("-") or t in ("&&", "||", "|", ";"):
            if t in ("&&", "||", "|", ";"):
                break
            continue
        out.append(re.split(r"[<>=!~\[;]", t)[0].lower())
    return [x for x in out if x]


def _offline_fetch_branch(pkg: Package, idx: Index):
    out, kinds = [], []
    for tk in idx.tasks:
        if not tk.scripts or phase_net(tk, "verifier") != "no-network":
            continue
        built = "\n".join(v for _, ins, v in idx.instrs.get(tk.dockerfile or "", []) if ins == "RUN").lower()
        done = False
        for sp in tk.scripts:
            for ln, line in sh_lines(pkg.files.get(sp, "")):
                for name, pat in FETCH:
                    if not pat.search(line):
                        continue
                    if name in ("pip install", "apt-get install"):
                        names = _pkg_names(line)
                        if names and all(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", built) for n in names):
                            continue  # already in the image, so the install is satisfied offline
                        if not names:
                            continue
                    if name == "npm install":
                        m = re.search(r"\bnpm\s+(?:install|i|ci)\b(.*)|\bnpx\s+(?:-y|--yes)\s+(\S+)", line)
                        names = [x for x in re.split(r"\s+", (m.group(1) or m.group(2) or "").strip())
                                 if x and not x.startswith("-")] if m else []
                        if not names or re.search(r"--offline|--prefer-offline", line) or \
                                all(re.search(rf"(?<![\w-]){re.escape(n.split('@')[0])}(?![\w-])", built) for n in names):
                            continue  # npx and a bare npm install use what the image already holds
                    out.append(ev(sp, ln))
                    kinds.append(name)
                    done = True
                    break
                if done:
                    break
            if done:
                break
    return out, kinds


# ---------------------------------------------------------------------------
# v0.5 widening of egress-and-shared-state
# sandbox-egress-policy-audit (arxiv:2607.21763): the task tells the agent it is offline, but its config gives the
# agent phase internet, so the only fence is the prompt.
# ---------------------------------------------------------------------------

OFFLINE_CLAIM = re.compile(
    r"(you\s+(do\s+not|don'?t|will\s+not|won'?t|cannot|can'?t)\s+have\s+(any\s+)?(internet|network|web)\b|"
    r"you\s+(have|will\s+have)\s+no\s+(internet|network|web)\b|"
    r"\b(internet|network)(\s+access)?\s+is\s+(disabled|unavailable|not\s+available|blocked|turned\s+off)\s+"
    r"(in|for|inside)\s+(this|the|your)\s+(environment|container|sandbox|machine|task)\b|"
    r"do\s+not\s+(use|access)\s+the\s+(internet|web)\b|you\s+are\s+(working\s+)?offline\b|"
    r"\b(this|the|your)\s+(environment|container|sandbox|machine)\s+(is\s+offline|is\s+air[- ]gapped|has\s+no\s+"
    r"(internet|network))\b)", re.I)


def _offline_claim_branch(pkg: Package, idx: Index):
    hits = []
    for tk in idx.tasks:
        if phase_net(tk, "agent") != "public":
            continue
        pre = (tk.root + "/") if tk.root else ""
        for n in ("instruction.md", "task.md", "TASK.md", "prompt.md"):
            t = pkg.files.get(pre + n)
            if not t:
                continue
            m = OFFLINE_CLAIM.search(t)
            if m:
                hits.append((pre + n, line_of(t, m.start())))
                break
    return hits


# ---------------------------------------------------------------------------
# v0.5 new rule: privileged container or Docker socket in the agent's box
# privileged-container-flags + docker-privilege-or-socket (benchjack V8, arxiv:2603.02277, AISI sandbox escape)
# ---------------------------------------------------------------------------

DANGER_CAPS = {"ALL", "SYS_ADMIN", "SYS_MODULE", "SYS_RAWIO", "SYS_PTRACE", "DAC_READ_SEARCH", "BPF", "SYS_BOOT"}
# NET_ADMIN only matters when the egress fence is a firewall inside the agent's own container
NET_ADMIN_WHAT = "holds NET_ADMIN while its egress firewall runs inside the same container"
FIREWALL = re.compile(r"\b(iptables|ip6tables|nft|nftables|ufw)\b")
PRIV_LITERALS = ("privileged", "docker.sock", "network_mode", "--network", "--net", "pid_mode", "pid:", "--pid",
                 "cap_add", "cap-add", "unconfined", "hostNetwork", "hostPID")
RUN_FLAGS = [
    ("runs privileged", re.compile(r"--privileged(?!\s*=\s*false)\b")),
    ("mounts the Docker socket", re.compile(r"(-v|--volume|--mount)[= ]\s*[^\s]*?/(var/)?run/docker\.sock")),
    ("shares the host network", re.compile(r"--(network|net)[= ]\s*['\"]?host\b")),
    ("shares the host PID namespace", re.compile(r"--pid[= ]\s*['\"]?host\b")),
    ("adds powerful capabilities", re.compile(r"--cap-add[= ]\s*['\"]?(CAP_)?(" + "|".join(DANGER_CAPS) + r")\b", re.I)),
    ("turns off seccomp or AppArmor", re.compile(r"--security-opt[= ]\s*['\"]?(seccomp|apparmor)[=:]unconfined")),
]
DOCKER_RUN = re.compile(r"\b(docker|podman|nerdctl)\s+(container\s+)?(run|create)\b")


def _service_flags(block_lines):
    """[(line, what)] for one compose service, from its (line, text) lines."""
    out, in_list = [], None
    for ln, l in block_lines:
        s = l.strip()
        key = re.match(r"^-?\s*['\"]?([\w.-]+)['\"]?\s*:\s*(.*)$", s)
        if key and not s.startswith("-"):
            k, val = key.group(1), key.group(2).strip().strip("'\"").lower()
            in_list = k if k in ("cap_add", "security_opt", "volumes") else None
            if k == "privileged" and val in ("true", "yes"):
                out.append((ln, "runs privileged"))
            elif k == "network_mode" and val == "host":
                out.append((ln, "shares the host network"))
            elif k == "pid" and val == "host":
                out.append((ln, "shares the host PID namespace"))
            if in_list and val.startswith("["):
                for item in re.findall(r"[\w:./=-]+", val):
                    out += _list_item_flag(in_list, item, ln)
            continue
        if in_list and s.startswith("-"):
            out += _list_item_flag(in_list, s[1:].strip().strip("'\""), ln)
            if in_list == "volumes" and "source:" in s:
                out += _list_item_flag(in_list, s, ln)
        elif in_list == "volumes" and re.search(r"(source|target)\s*:", s):
            out += _list_item_flag(in_list, s, ln)
    return out


def _list_item_flag(kind, item, ln):
    cap = re.sub(r"^CAP_", "", item.upper())
    if kind == "cap_add" and cap == "NET_ADMIN":
        return [(ln, NET_ADMIN_WHAT)]
    if kind == "cap_add" and cap in DANGER_CAPS:
        return [(ln, "adds powerful capabilities")]
    if kind == "security_opt" and re.search(r"(seccomp|apparmor)[=:]unconfined", item):
        return [(ln, "turns off seccomp or AppArmor")]
    if kind == "volumes" and re.search(r"/(var/)?run/docker\.sock", item):
        return [(ln, "mounts the Docker socket")]
    return []


def compose_services(text: str) -> dict:
    """{service: [(line, text)]} from a compose file in YAML or JSON form."""
    out = {}
    if text.lstrip().startswith("{"):
        try:
            cfg = json.loads(text)
        except Exception:
            cfg = None
        if isinstance(cfg, dict) and isinstance(cfg.get("services"), dict):
            for name, svc in cfg["services"].items():
                if not isinstance(svc, dict):
                    continue
                m = re.search(rf"['\"]{re.escape(name)}['\"]\s*:", text)
                ln = line_of(text, m.start()) if m else 1
                lines = []
                for k, v in svc.items():
                    if isinstance(v, list):
                        lines.append((ln, f"{k}:"))
                        lines += [(ln, f"- {x}") for x in v if isinstance(x, (str, int))]
                        lines += [(ln, f"- source: {x.get('source', '')}") for x in v if isinstance(x, dict)]
                    elif isinstance(v, (str, bool, int)):
                        lines.append((ln, f"{k}: {str(v).lower() if isinstance(v, bool) else v}"))
                out[name] = lines
        return out
    lines = text.splitlines()
    start = None
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*)services\s*:\s*$", l)
        if m:
            start, ind0 = i, len(m.group(1))
            break
    if start is None:
        return out
    svc_ind, cur = None, None
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        ind = len(l) - len(l.lstrip())
        if ind <= ind0:
            break
        if svc_ind is None:
            svc_ind = ind
        if ind == svc_ind:
            m = re.match(r"^\s*['\"]?([\w.-]+)['\"]?\s*:", l)
            cur = m.group(1) if m else None
            if cur:
                out[cur] = []
        elif cur:
            out[cur].append((i + 1, l))
    return out


K8S_KIND = re.compile(r"^\s*kind\s*:\s*['\"]?(Pod|Deployment|StatefulSet|Job|DaemonSet|CronJob|ReplicaSet|PodTemplate)\b", re.M)
K8S_FLAGS = [
    ("runs privileged", re.compile(r"^\s*privileged\s*:\s*true\b", re.M)),
    ("shares the host network", re.compile(r"^\s*hostNetwork\s*:\s*true\b", re.M)),
    ("shares the host PID namespace", re.compile(r"^\s*hostPID\s*:\s*true\b", re.M)),
    ("mounts the Docker socket", re.compile(r"^\s*path\s*:\s*['\"]?/(var/)?run/docker\.sock", re.M)),
]
SKIP_PRIV_PATH = re.compile(r"(^|/)(tests?|solution|solutions|oracle|docs?|examples?|\.github|outputs?|results?)/", re.I)
PY_PRIV_HINT = re.compile(r"(privileged\s*=\s*True|pid_mode\s*=|network_mode\s*=\s*['\"]host|cap_add\s*=|docker\.sock|"
                          r"(docker|podman)\s+(container\s+)?(run|create)\b[^\n]*--(privileged|net|network|pid|cap-add|"
                          r"security-opt|volume|mount)|-v\s+/var/run/docker\.sock)")
PY_RUN_HEAD = re.compile(r"(\.containers\.(run|create)|\bcreate_container|\b(subprocess\.)?(run|Popen|call|check_call|check_output)|"
                         r"\bos\.system|create_subprocess_(exec|shell)|\bexec\w*)$")


def _py_privilege(pf: PyFile):
    """[(line, what)] for container launches in Python with privilege flags (docker SDK keywords or docker run text)."""
    out = []
    if pf.tree is None:
        return out
    for n in ast.walk(pf.tree):
        if not isinstance(n, ast.Call):
            continue
        try:
            head = ast.unparse(n.func)
        except Exception:
            continue
        if not PY_RUN_HEAD.search(head):
            continue
        kws = {k.arg: k.value for k in n.keywords if k.arg}
        if ".containers." in head or "create_container" in head:
            if isinstance(kws.get("privileged"), ast.Constant) and kws["privileged"].value is True:
                out.append((n.lineno, "runs privileged"))
            for k, what in (("network_mode", "shares the host network"), ("pid_mode", "shares the host PID namespace")):
                if isinstance(kws.get(k), ast.Constant) and kws[k].value == "host":
                    out.append((n.lineno, what))
            caps = kws.get("cap_add")
            if isinstance(caps, (ast.List, ast.Tuple)) and any(
                    isinstance(e, ast.Constant) and str(e.value).upper().replace("CAP_", "") in DANGER_CAPS for e in caps.elts):
                out.append((n.lineno, "adds powerful capabilities"))
            vol = kws.get("volumes")
            try:
                vtxt = ast.unparse(vol) if vol is not None else ""
            except Exception:
                vtxt = ""
            if "docker.sock" in vtxt:
                out.append((n.lineno, "mounts the Docker socket"))
            continue
        txt = _call_text(n)
        if DOCKER_RUN.search(txt):
            for what, pat in RUN_FLAGS:
                if pat.search(txt):
                    out.append((n.lineno, what))
    return out


@rule("docker-privilege-or-socket", "hack", "high",
      "The agent's container runs privileged, mounts the Docker socket, shares the host network or PID namespace, "
      "adds powerful capabilities or turns off seccomp, so the agent can reach the host or the grader.")
def r_docker_privilege(pkg: Package):
    idx = index(pkg)
    found = []  # (evidence, what)
    task_roots = sorted((tk.root for tk in idx.tasks), key=len, reverse=True)
    for p, t in pkg.files.items():
        base = os.path.basename(p)
        if not (p.endswith((".yaml", ".yml", ".json", ".sh", ".bash")) or base == "Makefile") or \
                p.endswith(".json") and "compose" not in base or not any(x in t for x in PRIV_LITERALS):
            continue
        if re.fullmatch(r"(docker-)?compose[\w.-]*\.(ya?ml|json)", base):
            root = next((r for r in task_roots if r == "" or p.startswith(r + "/")), None)
            if root is None:
                continue  # only Harbor task compose files say which service is the agent's (main)
            rel = p[len(root) + 1:] if root else p
            if not rel.startswith("environment/"):
                continue
            svcs = compose_services(t)
            fw = None
            for ln, what in _service_flags(svcs.get("main", [])):
                if what == NET_ADMIN_WHAT:
                    if fw is None:
                        epre = (root + "/" if root else "") + "environment/"
                        fw = any(FIREWALL.search(x) for q, x in pkg.files.items() if q.startswith(epre) and q != p)
                    if not fw:
                        continue
                found.append((ev(p, ln), what))
        elif p.endswith((".yaml", ".yml")) and K8S_KIND.search(t) and not SKIP_PRIV_PATH.search(p):
            for what, pat in K8S_FLAGS:
                m = pat.search(t)
                if m:
                    found.append((ev(p, line_of(t, m.start())), what))
        elif (p.endswith((".sh", ".bash")) or base == "Makefile") and not SKIP_PRIV_PATH.search(p):
            for ln, line in sh_lines(t):
                if DOCKER_RUN.search(line):
                    for what, pat in RUN_FLAGS:
                        if pat.search(line):
                            found.append((ev(p, ln), what))
    box_text = any(any(x in pf.text for x in BOX_LITERALS + ("exec_run", "docker exec")) for pf in pkg.py_all())
    if box_text:
        for pf in py_with(pkg, ("privileged", "pid_mode", "network_mode", "cap_add", "docker.sock", "docker run",
                                "podman run", "docker create", "container run")):
            if SKIP_PRIV_PATH.search(pf.path) or not PY_PRIV_HINT.search(pf.text):
                continue
            for ln, what in _py_privilege(pf):
                found.append((ev(pf, ln), what))
    if not found:
        return []
    order = [w for w, _ in RUN_FLAGS] + [NET_ADMIN_WHAT]
    whats = sorted({w for _, w in found}, key=order.index)
    msg = "The agent's container " + (", ".join(whats[:-1]) + " and " + whats[-1] if len(whats) > 1 else whats[0])
    return [F("docker-privilege-or-socket", msg + ", so the agent can get out of its sandbox to the host, the grader "
              "or the network the task closed.", [e for e, _ in found], confidence="review")]

"""Graded test strength rules (envscan v0.5, builder `testq`).

These rules read the graded tests of code tasks as text and judge how strong
they are: Harbor tests/ folders (test.sh and the Python files it runs), and the
reward code and shipped dataset rows of verifiers code environments. Nothing is
run. Several of these are quality risks, not proven defects, and report as
`hygiene`.
"""
from __future__ import annotations

import ast
import json
import os
import re
import statistics

from .package import Package, PyFile, parse_py
from .rules import F, rule, ev, line_of
from .rules_grader import _cache, _reward_weights, _own_nodes, _num, _callee
from .rules_image import index, sh_lines

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

TEST_PY = re.compile(r"^(test_.*|.*_test)\.py$")


class TaskTests:
    """The graded test side of one Harbor task."""

    def __init__(self, tk, pre, scripts, graded, instruction, solve):
        self.tk = tk
        self.pre = pre
        self.scripts = scripts          # [(path, text)] verifier scripts
        self.graded = graded            # [PyFile] python files the scripts run (and helpers they import)
        self.instruction = instruction  # (path, text) or None
        self.solve = solve              # (path, text) or None

    @property
    def stext(self) -> str:
        return "\n".join(t for _, t in self.scripts)


def _owner(p: str, roots: list[str]) -> str | None:
    for r in roots:  # longest first
        if r == "" or p.startswith(r + "/"):
            return r
    return None


def task_tests(pkg: Package) -> list[TaskTests]:
    """Harbor tasks with their verifier scripts and graded Python files (cached per package)."""
    c = _cache(pkg)
    if "testq_tasks" in c:
        return c["testq_tasks"]
    out: list[TaskTests] = []
    c["testq_tasks"] = out
    if not any(p == "task.toml" or p.endswith("/task.toml") for p in pkg.files):
        return out
    tasks = index(pkg).tasks
    roots = sorted({tk.root for tk in tasks}, key=lambda r: -len(r))
    owned: dict[str, list[str]] = {}
    for p in pkg.files:
        r = _owner(p, roots)
        if r is not None:
            owned.setdefault(r, []).append(p)
    for tk in tasks:
        pre = (tk.root + "/") if tk.root else ""
        scripts = [(p, pkg.files.get(p, "")) for p in tk.scripts if p in pkg.files]
        stext = "\n".join(t for _, t in scripts)
        tests_py = []
        for p in owned.get(tk.root, []):
            rel = p[len(pre):]
            if p.endswith(".py") and re.match(r"(steps/[^/]+/)?tests/", rel):
                tests_py.append(p)
        runs_pytest = bool(re.search(r"\bpytest\b|unittest", stext))
        rewardkit = bool(re.search(r"\brewardkit\b", stext))
        graded = [p for p in tests_py
                  if os.path.basename(p) in stext or (TEST_PY.match(os.path.basename(p)) and runs_pytest) or rewardkit]
        # helpers the graded files import by module name
        mods = {os.path.splitext(os.path.basename(p))[0]: p for p in tests_py}
        for p in list(graded):
            t = pkg.files.get(p, "")
            for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", t, re.M):
                name = (m.group(1) or m.group(2)).split(".")[-1]
                q = mods.get(name)
                if q and q not in graded:
                    graded.append(q)
        pfs = []
        for p in graded:
            pf = pkg.py.get(p) or parse_py(p, pkg.files.get(p, ""))
            pfs.append(pf)
        ins = None
        for cand in ("instruction.md", "instruction.md.j2"):
            if pre + cand in pkg.files:
                ins = (pre + cand, pkg.files[pre + cand])
                break
        solve = (pre + "solution/solve.sh", pkg.files[pre + "solution/solve.sh"]) if pre + "solution/solve.sh" in pkg.files else None
        tt = TaskTests(tk, pre, scripts, pfs, ins, solve)
        tt.owned = owned.get(tk.root, [])
        tt.helpers = [p for p in graded if not TEST_PY.match(os.path.basename(p))]
        out.append(tt)
    return out


def _test_funcs(pf: PyFile):
    return [fi for fi in pf.funcs if fi.name.startswith("test")]


def _names(e) -> set[str]:
    return {n.id for n in ast.walk(e) if isinstance(n, ast.Name)}


def _literal_bindings(pf: PyFile, fnode=None) -> dict[str, ast.AST]:
    """name -> literal list / tuple / set / dict node, from module level and the given function."""
    out: dict[str, ast.AST] = {}
    if pf.tree is None:
        return out
    bodies = [pf.tree.body]
    if fnode is not None:
        bodies.append(list(_own_nodes(fnode)))
    for body in bodies:
        for n in body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and isinstance(n.value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                out[n.targets[0].id] = n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                    and isinstance(n.value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                out[n.target.id] = n.value
    return out


def _iter_len(it, binds) -> int | None:
    """Number of items of a literal iterable (through zip, enumerate, .items(), a bound name)."""
    if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
        return len(it.elts)
    if isinstance(it, ast.Dict):
        return len(it.keys)
    if isinstance(it, ast.Name) and it.id in binds:
        return _iter_len(binds[it.id], {})
    if isinstance(it, ast.Call):
        fn = _callee(it)
        if fn in ("enumerate", "list", "tuple", "sorted") and it.args:
            return _iter_len(it.args[0], binds)
        if fn == "zip" and it.args:
            ns = [_iter_len(a, binds) for a in it.args]
            return min(ns) if all(x is not None for x in ns) else None
        if fn in ("items", "keys", "values") and isinstance(it.func, ast.Attribute):
            return _iter_len(it.func.value, binds)
    return None


def _target_names(gen) -> set[str]:
    return _names(gen.target)


SKIP_CALLS = {"exists", "is_file", "is_dir", "endswith", "startswith", "isfile", "isdir", "match", "search",
              "fullmatch", "lower", "strip", "str", "len", "isinstance", "hasattr", "glob", "Path", "open", "join",
              "read_text", "int", "float", "repr", "format"}


def _is_case_check(elt, loop_vars: set[str]) -> bool:
    """A per case check: a compare or a call on a value computed from the loop item (the loop item is an
    argument of a call), not an 'output equals one of these alternatives' or a file exists test."""
    if isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.Not):
        elt = elt.operand
    calls = [n for n in ast.walk(elt) if isinstance(n, ast.Call)]
    if not calls:
        return False
    if isinstance(elt, ast.Call) and _callee(elt) in SKIP_CALLS:
        return False
    for c in calls:
        fn = _callee(c)
        if fn in SKIP_CALLS:
            continue
        arg_names = set()
        for a in list(c.args) + [k.value for k in c.keywords]:
            arg_names |= _names(a)
        if arg_names & loop_vars:
            if isinstance(elt, ast.Compare):
                if any(isinstance(op, (ast.In, ast.NotIn, ast.Is, ast.IsNot)) for op in elt.ops):
                    return False
            return True
    return False


def _compare_need(test) -> tuple[str, float] | None:
    """assert X >= k / X > k / X != 0 -> (X source, smallest passing count)."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    op, right = test.ops[0], test.comparators[0]
    k = _num(right)
    if k is None:
        return None
    left = test.left
    if isinstance(op, ast.GtE):
        need = k
    elif isinstance(op, ast.Gt):
        need = k + (1 if float(k).is_integer() else 0)
    elif isinstance(op, ast.NotEq) and k == 0:
        need = 1
    else:
        return None
    return ast.unparse(left), need


# ---------------------------------------------------------------------------
# 1. any-of-n-pass-condition
# canon: any-of-n-pass-condition (reader items). Main paper: trustworthy-env T-13 (word2vec-from-scratch:
# 4 analogy pairs, 1 of 4 needed).
# ---------------------------------------------------------------------------

def _any_of_n_hits(pf: PyFile):
    out = []
    for fi in _test_funcs(pf):
        binds = _literal_bindings(pf, fi.node)
        own = list(_own_nodes(fi.node))
        # counters: name -> (N, loop line) when name += 1 under a per case check inside a literal loop
        counters: dict[str, int] = {}
        for n in own:
            if not isinstance(n, ast.For):
                continue
            N = _iter_len(n.iter, binds)
            if N is None or N < 3:
                continue
            lv = _names(n.target)
            for m in ast.walk(n):
                if isinstance(m, ast.If) and _is_case_check(m.test, lv):
                    for st in m.body:
                        for s in ast.walk(st):
                            if isinstance(s, ast.AugAssign) and isinstance(s.op, ast.Add) and isinstance(s.target, ast.Name) \
                                    and _num(s.value) == 1.0:
                                counters[s.target.id] = N
        for n in own:
            if not isinstance(n, ast.Assert):
                continue
            t = n.test
            # assert any(check(x) for x in [a, b, c, d])
            if isinstance(t, ast.Call) and _callee(t) == "any" and t.args and \
                    isinstance(t.args[0], (ast.GeneratorExp, ast.ListComp)) and len(t.args[0].generators) == 1:
                g = t.args[0]
                N = _iter_len(g.generators[0].iter, binds)
                if N is not None and N >= 3 and _is_case_check(g.elt, _target_names(g.generators[0])):
                    out.append((n.lineno, 1, N))
                continue
            cn = _compare_need(t)
            if not cn:
                continue
            src, need = cn
            N = None
            lt = t.left
            if isinstance(lt, ast.Call) and _callee(lt) == "sum" and lt.args and \
                    isinstance(lt.args[0], (ast.GeneratorExp, ast.ListComp)) and len(lt.args[0].generators) == 1:
                g = lt.args[0]
                n_ = _iter_len(g.generators[0].iter, binds)
                if n_ is not None and _is_case_check(g.elt, _target_names(g.generators[0])):
                    N = n_
            elif isinstance(lt, ast.Name) and lt.id in counters:
                N = counters[lt.id]
            elif isinstance(lt, ast.BinOp) and isinstance(lt.op, ast.Div) and isinstance(lt.left, ast.Name) \
                    and lt.left.id in counters:
                N = counters[lt.left.id]
                need = need * N  # a fraction threshold
            if N is None or N < 3:
                continue
            if need <= 1 or need / N <= 0.25:
                out.append((n.lineno, max(1, int(need + 0.999)), N))
    return out


@rule("any-of-n-pass-condition", "hack", "medium",
      "A graded test passes when only one of several case checks passes, so output that gets one easy case right passes.")
def r_any_of_n(pkg: Package):
    tt = task_tests(pkg)
    if not tt:
        return []
    hits, worst = [], None
    for t in tt:
        for pf in t.graded:
            if pf.tree is None:
                continue
            for ln, need, N in _any_of_n_hits(pf):
                hits.append(ev(pf, ln))
                if worst is None or need / N < worst[0] / worst[1]:
                    worst = (need, N)
    if not hits:
        return []
    return [F("any-of-n-pass-condition",
              f"A graded test passes when {worst[0]} of {worst[1]} case checks pass, so a reply that solves one easy "
              "case passes the whole test.", hits, confidence="review")]


# ---------------------------------------------------------------------------
# 2. no-pass-to-pass-regression-tests
# canon: no-pass-to-pass-regression-tests (reader items). Main paper: trustworthy-env S-4 (19.8% of
# solved instances semantically wrong under weak suites), 2509.22237 (P2P regression suite).
# ---------------------------------------------------------------------------

F2P = re.compile(r"FAIL_TO_PASS|fail_to_pass|\bf2p\b|FAIL2PASS|fail2pass")
P2P = re.compile(r"PASS_TO_PASS|pass_to_pass|\bp2p\b|PASS2PASS|pass2pass|regression[_ ]tests?|run_all_tests|full[_ ]suite")
DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
TEST_RUNNER = re.compile(r"\b(pytest|py\.test|python[\d.]*\s+-m\s+(?:pytest|unittest)|go\s+test|cargo\s+test|npx\s+(?:jest|vitest|mocha)|"
                         r"npm\s+(?:run\s+)?test|yarn\s+test|mvn\b|gradlew?|tox|nox|rspec|phpunit|ctest)\b")
TESTISH = re.compile(r"(^|/)(tests?|spec|__tests__)(/|$)|(^|/)test_[^/]*$|_test\.\w+$|\.test\.\w+$|Test\w*\.java$|_spec\.\w+$")


def _f2p(text: str):
    low = ("AIL_TO_PASS" in text or "ail_to_pass" in text or "f2p" in text or "AIL2PASS" in text or "ail2pass" in text)
    return F2P.search(text) if low else None


def _p2p(text: str):
    low = ("ASS_TO_PASS" in text or "ass_to_pass" in text or "p2p" in text or "ASS2PASS" in text or "ass2pass" in text
           or "egression" in text or "run_all_tests" in text or "ull suite" in text or "ull_suite" in text)
    return P2P.search(text) if low else None


def _code_text(pkg: Package, skip_docs=True):
    for p, t in pkg.files.items():
        if skip_docs and p.endswith((".md", ".txt", ".rst")):
            continue
        yield p, t


@rule("no-pass-to-pass-regression-tests", "hack", "medium",
      "A SWE style grader runs only the tests the reference change touches and no pass to pass regression tests.")
def r_no_p2p(pkg: Package):
    hits, msg = [], None
    # verifiers side: scoring code reads a fail to pass list and nothing in the package names a pass to pass list
    pys = [pf for pf in pkg.py_all() if _f2p(pf.text)]
    if pys and not any(_p2p(t) for _, t in pkg.files.items()):
        for pf in pys:
            m = _f2p(pf.text)
            hits.append(ev(pf, line_of(pf.text, m.start())))
        msg = ("The grader reads a fail to pass test list and no pass to pass list, so a patch that fixes the target "
               "tests while breaking other behavior is still paid.")
    for t in task_tests(pkg):
        own = [(p, pkg.files[p]) for p in t.owned if p[len(t.pre):].startswith(("tests/", "solution/", "steps/"))
               and not p.endswith((".md",))]
        if any(_p2p(x) for _, x in own):
            continue
        # a config with a fail to pass key and no pass to pass key
        f2p_at = None
        for p, x in own:
            m = _f2p(x)
            if m and (p.endswith(".json") or p in dict(t.scripts) or p.endswith(".py")) and "/tests/" in "/" + p[len(t.pre):]:
                f2p_at = ev(p, line_of(x, m.start()))
                break
        if f2p_at:
            hits.append(f2p_at)
            msg = msg or ("The grader reads a fail to pass test list and no pass to pass list, so a patch that fixes the "
                          "target tests while breaking other behavior is still paid.")
            continue
        # a SWE shaped task: the oracle is a code patch and test.sh runs only the files of the test patch
        if not t.solve or not re.search(r"diff --git|git\s+apply|\bpatch\s+-p", t.solve[1]):
            continue
        patch_files = set()
        for _, x in t.scripts:
            for m in DIFF_FILE.finditer(x):
                if TESTISH.search(m.group(2)):
                    patch_files.add(m.group(2))
        for p, x in own:
            if p.endswith((".patch", ".diff")) and "/tests/" in "/" + p[len(t.pre):]:
                for m in DIFF_FILE.finditer(x):
                    if TESTISH.search(m.group(2)):
                        patch_files.add(m.group(2))
        if not patch_files:
            continue
        for sp, x in t.scripts:
            for ln, cmd in sh_lines(x):
                if not TEST_RUNNER.search(cmd):
                    continue
                args = [a.strip("'\"") for a in cmd.split()[1:]]
                files = [a.split("::")[0] for a in args if re.search(r"\.\w+(::|$)", a.split("::")[0]) and TESTISH.search(a.split("::")[0])]
                dirs = [a for a in args if not a.startswith("-") and a.rstrip("/").split("/")[-1] in ("tests", "test", "spec")]
                if files and not dirs and all(f.lstrip("./") in patch_files or f in patch_files for f in files):
                    hits.append(ev(sp, ln))
                    msg = msg or ("test.sh runs only the test files from the reference test patch and no pass to pass "
                                  "regression tests, so a patch that breaks other behavior is still paid.")
    if not hits:
        return []
    return [F("no-pass-to-pass-regression-tests", msg, hits, confidence="review")]


# ---------------------------------------------------------------------------
# 3. thin-test-suite
# canon: thin-test-suite (reader items). Main paper: arxiv:2601.22803 (52.9% of
# CodeRM tests run but assert wrong outputs), trustworthy-env terminalbench2_poc (4 games, 118 positions).
# ---------------------------------------------------------------------------

ASSERT_CALL = re.compile(r"^(assert\w*|fail|raises|expect\w*|check\w*)$")
OTHER_CHECK = re.compile(r"(\bdiff\s|\bcmp\s|\bgrep\s+-\w*q|\bpython[\d.]*\s+(?!-m\s+pytest)\S+\.py|\./\S+|"
                         r"\bnode\s|\bcargo\s|\bgo\s+test|\bnpm\s|\bjest|\bmake\s|\bjava\s|\bdotnet\s|rewardkit)")


def _assert_count(pf: PyFile) -> tuple[int, bool]:
    """(assert count, any check runs in a loop or a parametrize) over the whole file."""
    n, many = 0, False
    if pf.tree is None:
        return 0, True
    if re.search(r"parametrize|hypothesis|@given|subTest", pf.text):
        many = True
    loops = [x for x in ast.walk(pf.tree) if isinstance(x, (ast.For, ast.While, ast.comprehension))]
    in_loop = set()
    for lp in loops:
        for x in ast.walk(lp if not isinstance(lp, ast.comprehension) else lp.iter):
            in_loop.add(id(x))
    for x in ast.walk(pf.tree):
        if isinstance(x, ast.Assert):
            n += 1
            if id(x) in in_loop:
                many = True
        elif isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and ASSERT_CALL.match(x.func.attr):
            n += 1
            if id(x) in in_loop:
                many = True
        elif isinstance(x, ast.Call) and _callee(x) in ("all", "any"):
            many = True  # one assert over many items
    return n, many


AGENT_CALL = re.compile(r"(subprocess\.(run|check_output|Popen|call)|importlib|sys\.path\.(insert|append)|"
                        r"spec_from_file_location|^\s*from\s+(solution|main|app|submission|answer)\b|"
                        r"^\s*import\s+(solution|main|submission)\b|os\.system\()", re.M)


def _literal_io(pf: PyFile) -> bool:
    """Some assert checks a call with literal inputs against a literal output, as in f("hello") == "BOFFE"."""
    if pf.tree is None:
        return False
    for n in ast.walk(pf.tree):
        if isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare) and len(n.test.ops) == 1 \
                and isinstance(n.test.ops[0], ast.Eq):
            sides = [n.test.left, n.test.comparators[0]]
            for a, b in (sides, sides[::-1]):
                if isinstance(a, ast.Call) and a.args and all(isinstance(x, ast.Constant) for x in a.args) \
                        and isinstance(b, ast.Constant):
                    return True
    return False


TESTS_FIELDS = ("test_list", "test_cases", "tests", "test", "testcases", "public_tests", "private_tests",
                "hidden_tests", "input_output", "test_inputs", "assertions")


def _row_test_count(row: dict) -> int | None:
    best = None
    for k in TESTS_FIELDS:
        if k not in row:
            continue
        v = row[k]
        if isinstance(v, str):
            s = v.strip()
            if s[:1] in "[{":
                try:
                    v = json.loads(s)
                except Exception:
                    v = None
            else:
                cnt = len(re.findall(r"^\s*assert\b", s, re.M))
                v = None
                if cnt:
                    best = max(best or 0, cnt)
        if isinstance(v, list):
            best = max(best or 0, len(v))
        elif isinstance(v, dict):
            for kk in ("inputs", "input", "stdin"):
                if isinstance(v.get(kk), list):
                    best = max(best or 0, len(v[kk]))
    return best


def _jsonl_rows(text: str, limit=300):
    rows = []
    s = text.lstrip()
    if s.startswith("["):
        try:
            data = json.loads(text)
            return [r for r in data[:limit] if isinstance(r, dict)]
        except Exception:
            return []
    for line in text.splitlines()[:limit]:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def _dataset_files(pkg: Package):
    for p, t in pkg.files.items():
        if p.endswith((".jsonl", ".json")) and not os.path.basename(p).startswith(("package", "tsconfig", "config")) \
                and "/outputs/" not in "/" + p and not p.startswith("outputs/") and len(t) > 200:
            yield p, t


@rule("thin-test-suite", "hygiene", "low",
      "The graded tests have very few checks, so a reply that handles only those few cases is paid as correct.")
def r_thin_suite(pkg: Package):
    hits, msgs = [], []
    for t in task_tests(pkg):
        tests = [pf for pf in t.graded if _test_funcs(pf)]
        if not tests:
            continue
        stext = t.stext
        # another grader step in test.sh (a diff, a checker script, a build) means pytest is not the only check
        other = [c for _, x in t.scripts for _, c in sh_lines(x) if OTHER_CHECK.search(c) and "pytest" not in c]
        if other:
            continue
        if re.search(r"rewardkit", stext):
            continue
        if t.helpers:
            continue  # a helper module may hold the real checks
        gt = "\n".join(pf.text for pf in t.graded)
        if not AGENT_CALL.search(gt):
            continue  # an answer file or a state check: one assert can be the whole task
        if any(isinstance(x, (ast.For, ast.While, ast.comprehension)) for pf in t.graded if pf.tree for x in ast.walk(pf.tree)):
            continue
        if re.search(r"(/tests/\S+\.(sh|py)|\brandom\b|seed)", gt):
            continue  # the checks live in another script, or inputs change per run
        if not any(_literal_io(pf) for pf in t.graded):
            continue  # one goal tasks (a config, a file, an exploit) are not thin, they have one thing to check
        nfun = sum(len(_test_funcs(pf)) for pf in tests)
        counts = [_assert_count(pf) for pf in t.graded]
        nas = sum(c for c, _ in counts)
        many = any(m for _, m in counts)
        if many or nas == 0 or nas > 3 or nfun > 3:
            continue
        pf = tests[0]
        hits.append(ev(pf, _test_funcs(pf)[0].start))
        msgs.append((nfun, nas))
    # verifiers side: shipped rows whose test lists, read by the scorer, are short
    if pkg.py:
        scoring = "\n".join(pf.text for pf in pkg.py_all())
        for p, x in _dataset_files(pkg):
            rows = _jsonl_rows(x)
            if len(rows) < 10:
                continue
            cnts = [c for c in (_row_test_count(r) for r in rows) if c is not None]
            if len(cnts) < 10:
                continue
            fields = [k for k in TESTS_FIELDS if k in rows[0]]
            if not any(re.search(rf"['\"]{re.escape(k)}['\"]", scoring) for k in fields):
                continue
            med = statistics.median(cnts)
            if med <= 3:
                hits.append(ev(p, 1))
                msgs.append((None, med))
    if not hits:
        return []
    if msgs[0][0] is not None:
        m = (f"The graded tests have {msgs[0][0]} test functions with {msgs[0][1]} checks in all, so a reply that only "
             "handles those few cases is paid as correct.")
    else:
        m = (f"Dataset rows carry a median of {msgs[0][1]:g} test cases per problem, so a reply that only handles those "
             "few cases is paid as correct.")
    return [F("thin-test-suite", m, hits, confidence="review")]


# ---------------------------------------------------------------------------
# 4. tests-call-network-or-clock
# canon: tests-call-network-or-clock (reader items). Main paper: trustworthy-env S-6 (14-15 of
# 500 gold patches fail, partly from network and version drift), arxiv:2602.07274.
# ---------------------------------------------------------------------------

LOCAL_HOST = re.compile(r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|\[?::1\]?|host\.docker\.internal|[\w-]+)$")
URL = re.compile(r"https?://([\w.-]+|\[[\w:]+\])")
NET_CALL = {"get", "post", "put", "delete", "head", "patch", "request", "urlopen", "urlretrieve", "Request",
            "create_connection", "fetch"}
NET_MOD = re.compile(r"^(requests|httpx|urllib|urllib\.request|aiohttp|session|client|s|http|urllib3)$")
CLOCK_CALL = re.compile(r"\b(date\.today|datetime\.now|datetime\.today|datetime\.utcnow|datetime\.datetime\.now|"
                        r"datetime\.date\.today)\s*\(")


def _external(url_host: str) -> bool:
    return "." in url_host and not LOCAL_HOST.match(url_host)


def _str_consts(pf: PyFile, fnode) -> dict[str, str]:
    out = dict(pf.consts)
    for n in _own_nodes(fnode):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                out[n.targets[0].id] = n.value.value
            elif isinstance(n.value, ast.JoinedStr):
                out[n.targets[0].id] = "".join(v.value for v in n.value.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return out


def _net_hits_py(pf: PyFile):
    out = []
    if pf.tree is None or not re.search(r"https?://", pf.text):
        return out
    for fi in pf.funcs:
        consts = None
        for n in _own_nodes(fi.node):
            if not isinstance(n, ast.Call) or _callee(n) not in NET_CALL:
                continue
            f = n.func
            if isinstance(f, ast.Attribute):
                base = ast.unparse(f.value)
                if not NET_MOD.match(base) and not re.search(r"(session|client|requests|httpx|urllib)", base, re.I):
                    continue
            elif _callee(n) not in ("urlopen", "urlretrieve", "Request"):
                continue
            if not n.args and not n.keywords:
                continue
            a = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg in ("url", "uri")), None)
            if a is None:
                continue
            txt = None
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                txt = a.value
            elif isinstance(a, ast.JoinedStr):
                txt = "".join(v.value for v in a.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
            elif isinstance(a, ast.Name):
                consts = consts if consts is not None else _str_consts(pf, fi.node)
                txt = consts.get(a.id)
            elif isinstance(a, ast.BinOp):
                consts = consts if consts is not None else _str_consts(pf, fi.node)
                nm = a.left.id if isinstance(a.left, ast.Name) else None
                txt = consts.get(nm) if nm else (a.left.value if isinstance(a.left, ast.Constant) and isinstance(a.left.value, str) else None)
            if not txt:
                continue
            m = URL.match(txt.strip())
            if m and _external(m.group(1)):
                out.append((n.lineno, m.group(1)))
    return out


SH_FETCH = re.compile(r"\b(curl|wget|http)\b[^|;&\n]*?https?://([\w.-]+)")
INSTALLER = re.compile(r"(\|\s*(ba|z)?sh\b|install|get-pip|setup|rustup|nvm|nodesource|\.deb\b|\.tar|\.tgz|\.zip|\.whl|"
                       r"apt|releases/download|-o\s|--output|-O\b|>\s*\S)")


def _net_hits_sh(text: str):
    out = []
    for ln, cmd in sh_lines(text):
        for m in SH_FETCH.finditer(cmd):
            host = m.group(2)
            if not _external(host):
                continue
            if INSTALLER.search(cmd) and not re.search(r"\$\(\s*(curl|wget)|\|\s*(grep|jq)\b|^\s*if\b", cmd):
                continue
            out.append((ln, host))
    return out


def _clock_hits(pf: PyFile):
    out = []
    if pf.tree is None or not CLOCK_CALL.search(pf.text):
        return out
    for fi in _test_funcs(pf):
        own = list(_own_nodes(fi.node))
        clock_vars = set()
        for n in own:
            if isinstance(n, ast.Assign) and CLOCK_CALL.search(ast.unparse(n.value)):
                for tg in n.targets:
                    clock_vars |= _names(tg)
        for n in own:
            if isinstance(n, ast.Assert):
                s = ast.unparse(n.test)
                if CLOCK_CALL.search(s) or (_names(n.test) & clock_vars):
                    out.append(n.lineno)
    return out


@rule("tests-call-network-or-clock", "hygiene", "low",
      "Graded tests call a public web service or read today's date, so a right answer can fail when the service or the date changes.")
def r_net_clock(pkg: Package):
    net, clock, hosts = [], [], set()
    for t in task_tests(pkg):
        for pf in t.graded:
            for ln, host in _net_hits_py(pf):
                net.append(ev(pf, ln))
                hosts.add(host)
            for ln in _clock_hits(pf):
                clock.append(ev(pf, ln))
        for sp, x in t.scripts:
            # only the lines after setup: commands that read a service and use its reply in a check
            for ln, host in _net_hits_sh(x):
                net.append(ev(sp, ln))
                hosts.add(host)
    if not (net or clock):
        return []
    if net:
        h = ", ".join(sorted(hosts)[:3])
        m = (f"Graded tests call a public web service ({h}), so a right answer can fail when that service is slow, "
             "down or changes its reply.")
    else:
        m = "Graded tests compare against today's date, so a right answer can fail when the test runs on another day."
    return [F("tests-call-network-or-clock", m, net + clock, confidence="review")]


# ---------------------------------------------------------------------------
# 5. multi-answer-exact-diff-no-checker
# canon: multi-answer-exact-diff-no-checker (reader items). Main paper: arxiv:2506.05817
# (CodeContests: about 33% invalid tests, over 4,000 problems with false negatives).
# ---------------------------------------------------------------------------

MULTI = re.compile(r"(if there (are|is) (several|multiple|many|more than one)[^.\n]{0,60}(answers?|solutions?|ways|outputs?)"
                   r"[^.\n]{0,40}(print|output|return|give|choose|pick)[^.\n]{0,15}\bany\b|"
                   r"(print|output|return)\s+any\s+(one\s+)?(of\s+them|valid|correct|such|possible|optimal)|"
                   r"any\s+(valid|correct|optimal)\s+(answer|solution|ordering|order|assignment|output|permutation)\s+"
                   r"(is|will be)\s+accepted|"
                   r"you\s+(may|can)\s+(print|output|return)\s+any)", re.I)
CHECKER = re.compile(r"(checker|special[_ ]judge|spj|validat\w*|verify_\w+|is_valid\w*|check_(solution|answer|output)\w*|"
                     r"testlib|interactor|accepted_answers|alternatives|any_of|valid_answers)", re.I)
OUTISH = re.compile(r"(stdout|output|out\b|result|actual|got|answer|res\b)", re.I)
EXPISH = re.compile(r"(expected|gold|reference|ref_|answer_key|want|correct_out|^answer\b)", re.I)


def _exact_compare_lines(pf: PyFile):
    out = []
    if pf.tree is None:
        return out
    for n in ast.walk(pf.tree):
        if isinstance(n, (ast.Assert, ast.If, ast.Return)):
            t = n.test if not isinstance(n, ast.Return) else n.value
            if t is None:
                continue
            for c in ast.walk(t):
                if isinstance(c, ast.Compare) and len(c.ops) == 1 and isinstance(c.ops[0], (ast.Eq, ast.NotEq)):
                    a, b = ast.unparse(c.left), ast.unparse(c.comparators[0])
                    if (OUTISH.search(a) and EXPISH.search(b)) or (OUTISH.search(b) and EXPISH.search(a)):
                        if not re.search(r"(len\(|returncode|exit|\.count|status|type\(|sorted\(|set\(|Counter\()", a + b):
                            out.append(n.lineno)
    return out


def _shell_diff_lines(text: str):
    return [ln for ln, cmd in sh_lines(text) if re.search(r"\b(diff|cmp)\s+(?!--git)", cmd) and re.search(r"(expect|answer|\.out\b|\.ans\b|output)", cmd, re.I)]


@rule("multi-answer-exact-diff-no-checker", "broken", "medium",
      "The problem allows several right answers but the grader compares the output to one expected output with no checker.")
def r_multi_answer(pkg: Package):
    hits = []
    for t in task_tests(pkg):
        if not t.instruction:
            continue
        ip, itext = t.instruction
        m = MULTI.search(itext)
        if not m:
            continue
        gtext = "\n".join(pf.text for pf in t.graded) + "\n" + t.stext
        if CHECKER.search(gtext):
            continue
        lines = [ev(pf, ln) for pf in t.graded for ln in _exact_compare_lines(pf)]
        lines += [ev(sp, ln) for sp, x in t.scripts for ln in _shell_diff_lines(x)]
        if lines:
            hits.append(ev(ip, line_of(itext, m.start())))
            hits += lines[:3]
    # verifiers side: shipped problem rows say any answer is fine, and the paid scorer compares stdout exactly
    if pkg.py and not hits:
        allpy = "\n".join(pf.text for pf in pkg.py_all())
        if not CHECKER.search(allpy):
            exact = [ev(pf, ln) for pf in pkg.py_all() for ln in _exact_compare_lines(pf)
                     if re.search(r"(stdout|expected_output|output)", pf.text)]
            if exact:
                for p, x in _dataset_files(pkg):
                    rows = _jsonl_rows(x)
                    multi = 0
                    for r in rows:
                        q = " ".join(str(v) for k, v in r.items() if isinstance(v, str) and k in
                                     ("question", "prompt", "problem", "description", "statement", "problem_statement", "text"))
                        if MULTI.search(q):
                            multi += 1
                    if multi >= 3:
                        hits.append(ev(p, 1))
                        hits += exact[:2]
                        break
    if not hits:
        return []
    return [F("multi-answer-exact-diff-no-checker",
              "The problem says any of several answers is fine, but the grader compares the output to one expected "
              "output with no checker, so a different right answer is scored wrong.", hits, confidence="review")]


# ---------------------------------------------------------------------------
# 6. test-inputs-below-constraint-range
# canon: test-inputs-below-constraint-range (reader items). Main paper: CodeScaler (tests covered
# n up to 9 of a 3..30 range), arxiv:2605.28751.
# ---------------------------------------------------------------------------

NUM = (r"(\d+(?:\.\d+)?\s*(?:[*×x·]|\\times|\\cdot)\s*10\s*\^\s*\{?\s*\d+\s*\}?|10\s*\^\s*\{?\s*\d+\s*\}?|"
       r"10\s*\*\*\s*\d+|\d+(?:\.\d+)?e\d+|\d[\d,]*(?:\.\d+)?)")
BOUND = re.compile(rf"(?:≤|<=|\\leq?|\\le\b|&le;|up to|at most)\s*\$?\s*{NUM}", re.I)
GEN = re.compile(r"(\brandom\b|np\.random|numpy\.random|default_rng|\bfaker\b|hypothesis|range\(\s*\d{4,}|"
                 r"\*\s*\d{4,}|\d{4,}\s*\*|10\s*\*\*\s*\d|1e\d|generate\w*\(|gen_\w+\()", re.I)


def _num_val(s: str) -> float | None:
    s = re.sub(r"\s+", "", s).replace(",", "").replace("{", "").replace("}", "").replace("\\times", "*").replace("\\cdot", "*")
    try:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)[*×x·]10\^(\d+)", s)
        if m:
            return float(m.group(1)) * 10 ** int(m.group(2))
        m = re.fullmatch(r"10(?:\^|\*\*)(\d+)", s)
        if m:
            return 10.0 ** int(m.group(1))
        return float(s)
    except Exception:
        return None


def _max_bound(text: str) -> float | None:
    best = None
    for m in BOUND.finditer(text):
        v = _num_val(m.group(1))
        if v is not None and (best is None or v > best):
            best = v
    return best


def _max_input_py(pf: PyFile) -> float:
    best = 0.0
    if pf.tree is None:
        return best
    for n in ast.walk(pf.tree):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                best = max(best, abs(float(n.value)))
            elif isinstance(n.value, str):
                best = max(best, float(len(n.value)))
                for x in re.findall(r"\d{1,12}", n.value):
                    best = max(best, float(x))
        elif isinstance(n, (ast.List, ast.Tuple)):
            best = max(best, float(len(n.elts)))
    return best


@rule("test-inputs-below-constraint-range", "hygiene", "low",
      "All graded test inputs are far below the size limits the problem states, so a slow brute force answer passes.")
def r_below_range(pkg: Package):
    hits, info = [], None
    for t in task_tests(pkg):
        if not t.instruction or not t.graded:
            continue
        ip, itext = t.instruction
        B = _max_bound(itext)
        if B is None or B < 10_000:
            continue
        gtext = "\n".join(pf.text for pf in t.graded) + "\n" + t.stext
        if GEN.search(gtext):
            continue
        # data files under tests/ that hold inputs
        data = [(p, pkg.files[p]) for p in t.owned if p.startswith(t.pre + "tests/") and not p.endswith((".py", ".sh"))]
        big_data = any(pkg.sizes.get(p, 0) > 50_000 for p in pkg.sizes if p.startswith(t.pre + "tests/") and not p.endswith((".py", ".sh")))
        if big_data:
            continue
        M = max([_max_input_py(pf) for pf in t.graded] + [0.0])
        for _, x in data:
            for v in re.findall(r"\d{1,12}", x[:200_000]):
                M = max(M, float(v))
            M = max(M, float(len(x.split())))
        if M * 100 <= B:
            m = BOUND.search(itext)
            hits.append(ev(ip, line_of(itext, m.start())))
            hits.append(ev(t.graded[0], 1))
            info = info or (B, M)
    if not hits:
        return []
    return [F("test-inputs-below-constraint-range",
              f"The problem allows inputs up to {info[0]:g} but no graded test input goes above {info[1]:g}, so a slow "
              "brute force answer passes.", hits, confidence="review")]


# ---------------------------------------------------------------------------
# 7. partial-credit-pass-rate-reward
# canon: partial-credit-pass-rate-reward (reader items). Main paper: arxiv:2605.02944
# (pass rate reward reverses the ranking in 36.5% of tasks).
# ---------------------------------------------------------------------------

PASSN = re.compile(r"(pass|correct|succe|solved|\bok\b|right|good|hits?\b|n_ok|num_ok|green)", re.I)
TOTALN = re.compile(r"(total|len\(|count|num_|n_|cases|tests|results|max\()", re.I)
TESTWORD = re.compile(r"(test|case|unit|assert|spec)", re.I)
TESTCTX = re.compile(r"(test|case|unit|assert|pytest|run_code|execute|sandbox|stdin|stdout)", re.I)


def _is_fraction(e) -> bool:
    if not (isinstance(e, ast.BinOp) and isinstance(e.op, ast.Div)):
        return False
    a, b = ast.unparse(e.left), ast.unparse(e.right)
    if _num(e.right) is not None:
        return False
    if not TOTALN.search(b):
        return False
    # a count of passed tests or cases: "pass" in the numerator, or test words on either side
    if not (re.search(r"pass", a, re.I) or (PASSN.search(a) and TESTWORD.search(a + " " + b))):
        return False
    if re.search(r"(time|token|length|len\(\s*(completion|response|text|answer|code)|char|word)", a + b, re.I):
        return False
    return True


def _fraction_names(own) -> dict[str, int]:
    """name -> line for names bound to a pass fraction (one hop through a plain name copy)."""
    out: dict[str, int] = {}
    for n in own:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            val = n.value
            tgs = n.targets if isinstance(n, ast.Assign) else [n.target]
            if val is None:
                continue
            if any(_is_fraction(x) for x in ast.walk(val)):
                for tg in tgs:
                    for nm in _names(tg):
                        out.setdefault(nm, n.lineno)
    for n in own:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id in out:
            for tg in n.targets:
                for nm in _names(tg):
                    out.setdefault(nm, out[n.value.id])
    return out


def _has_all_pass_gate(own, frac_names: set[str]) -> bool:
    """An if that compares the pass count with the total (or tests all(...)) and returns or sets a constant."""
    for n in own:
        if isinstance(n, (ast.If, ast.IfExp)):
            s = ast.unparse(n.test)
            gate = (re.search(r"(==|!=|<|>=)", s) and PASSN.search(s) and TOTALN.search(s)) or \
                re.search(r"\ball\(", s) or re.search(r"(==|>=|<)\s*1(\.0)?\b", s) and (frac_names & _names(n.test))
            if not gate:
                continue
            body = n.body if isinstance(n, ast.If) else [n.body, n.orelse]
            for b in (body if isinstance(body, list) else [body]):
                for x in ast.walk(b if isinstance(b, ast.AST) else ast.Module(body=[b], type_ignores=[])):
                    if isinstance(x, ast.Return) and x.value is not None and _num(x.value) == 0.0:
                        return True
                    if isinstance(x, ast.Assign) and _num(x.value) == 0.0 and (_names(x.targets[0]) & frac_names):
                        return True
            if isinstance(n, ast.If):
                for x in n.orelse:
                    if isinstance(x, ast.Return) and x.value is not None and _num(x.value) == 0.0:
                        return True
    return False


def _returns_fraction(fi) -> int | None:
    own = list(_own_nodes(fi.node))
    names = _fraction_names(own)
    if _has_all_pass_gate(own, set(names)):
        return None
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None:
            v = n.value
            if isinstance(v, ast.IfExp):
                v = v.body
            if any(_is_fraction(x) for x in ast.walk(v)):
                return n.lineno
            if _names(v) & set(names) and not isinstance(v, ast.Compare):
                # a dict return pays only the key named reward or score
                if isinstance(v, ast.Dict):
                    for k, val in zip(v.keys, v.values):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str) and \
                                re.fullmatch(r"(reward|score|total_reward|final_score)", k.value) and (_names(val) & set(names)):
                            return n.lineno
                    continue
                return n.lineno
    return None


WRITES_REWARD = re.compile(r"(reward\.(txt|json)|/logs/verifier|REWARD_PATH|reward_path|REWARD_FILE)")


def _sh_fraction(text: str):
    """[(line)] where a test.sh writes reward.txt from arithmetic over pass and total counts."""
    out, frac_vars = [], set()
    for ln, cmd in sh_lines(text):
        m = re.match(r"^\s*(?:export\s+|local\s+)?(\w+)=(.*)$", cmd)
        if m and re.search(r"(\$\(\(|\bbc\b|\bawk\b|python3?\s+-c)", m.group(2)) and "/" in m.group(2) \
                and PASSN.search(m.group(2)) and re.search(r"(TOTAL|total|COUNT|count|NUM|num)", m.group(2)):
            frac_vars.add(m.group(1))
            continue
        if re.search(r">>?\s*['\"]?\S*reward\.(txt|json)", cmd):
            if any(re.search(rf"\$\{{?{v}\b", cmd) for v in frac_vars):
                out.append(ln)
            elif "/" in cmd.split(">")[0] and PASSN.search(cmd) and re.search(r"(TOTAL|total)", cmd):
                out.append(ln)
    return out


def _paid_py_funcs(pkg: Package):
    W = _reward_weights(pkg)
    if not W:
        return []
    paid = {nm for nm, ws in W.items() if any(w > 0 for w, _ in ws)}
    return [(pf, fi) for pf in pkg.py_all() for fi in pf.funcs if fi.name in paid]


@rule("partial-credit-pass-rate-reward", "hygiene", "low",
      "A code reward pays the fraction of tests passed with no all pass gate, so a reply that games some tests is paid.")
def r_partial_credit(pkg: Package):
    hits = []
    # verifiers: a paid reward function returns passed / total
    for pf, fi in _paid_py_funcs(pkg):
        if not TESTCTX.search(fi.src):
            continue
        ln = _returns_fraction(fi)
        if ln:
            hits.append(ev(pf, ln))
    # Harbor: a graded file writes the fraction to reward.txt or reward.json, a rewardkit criterion returns it,
    # or test.sh does the arithmetic itself
    for t in task_tests(pkg):
        for pf in t.graded:
            if pf.tree is None or "/" not in pf.text:
                continue
            crit = re.search(r"@\w*\.?criterion", pf.text) is not None
            writes = WRITES_REWARD.search(pf.text) is not None
            if not (crit or writes):
                continue
            for fi in pf.funcs:
                if crit and not any("criterion" in d for d in fi.decorators) and not writes:
                    continue
                ln = _returns_fraction(fi)
                if ln is None:
                    continue
                if crit and any("criterion" in d for d in fi.decorators):
                    hits.append(ev(pf, ln))
                elif writes and re.search(rf"\b{re.escape(fi.name)}\(", pf.text.replace(f"def {fi.name}(", "")):
                    hits.append(ev(pf, ln))
        for sp, x in t.scripts:
            for ln in _sh_fraction(x):
                hits.append(ev(sp, ln))
    if not hits:
        return []
    return [F("partial-credit-pass-rate-reward",
              "The code reward pays the fraction of tests passed with no all pass gate, so code that games some "
              "tests is paid more than a near miss and partial work is paid at all.", hits, confidence="review")]

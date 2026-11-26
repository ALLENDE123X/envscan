"""Grader family rules (envscan v0.2).

Rules about the grader itself: what it pays when it fails, what a formal checker
lets through, how a judge prompt fences the response, and how loosely an answer
is matched. Like every envscan rule these read source text and ASTs only.

Evidence ids in the comments refer to the ranked table in section 9.2 of the
literature review (research/lit-review/rollup/report_parts/09-open-problems-product.md).
"""
from __future__ import annotations

import ast
import json
import os
import re

from .package import Package, PyFile, FuncInfo, _const_text, parse_py
from .rules import (F, SCORING_NAME, _callable_name, _iter_jsonl_rows, all_funcs, called_names, ev, line_of,
                    r_judge_undelimited, r_lean, rule)

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_CACHE_ATTR = "_envscan_grader_cache"


def _cache(pkg: Package) -> dict:
    c = getattr(pkg, _CACHE_ATTR, None)
    if c is None:
        c = {}
        try:
            setattr(pkg, _CACHE_ATTR, c)
        except Exception:
            pass
    return c


def _harbor_test_py(pkg: Package) -> list[PyFile]:
    """Python files under a Harbor task's tests/ dir (the verifier). Package.py skips every
    tests/ path because in a verifiers env those are unit tests, so parse them here, but
    only when the tests/ dir belongs to a Harbor task (task.toml above it or test.sh in it)."""
    c = _cache(pkg)
    if "harbor_py" in c:
        return c["harbor_py"]
    files = pkg.files
    tomls = {os.path.dirname(p) for p in files if os.path.basename(p) == "task.toml"}
    tsh = {os.path.dirname(p) for p in files if os.path.basename(p) == "test.sh"}
    out = []
    for p, t in files.items():
        if not p.endswith(".py") or p in pkg.py:
            continue
        parts = p.split("/")
        if "tests" not in parts[:-1]:
            continue
        i = parts.index("tests")
        tdir = "/".join(parts[:i + 1])
        prefixes = {"/".join(parts[:k]) for k in range(0, i + 1)}
        if tdir in tsh or prefixes & tomls:
            out.append(parse_py(p, t))
    c["harbor_py"] = out
    return out


_REWARD_GATE = re.compile(r"(Rubric|funcs|add_reward_func|reward)")


def _registered_reward_names(pkg: Package) -> set[str]:
    """Same result as rules.registered_reward_names (names only), but files that contain none of
    the tokens it keys on are not walked."""
    out: set[str] = set()
    for pf in pkg.py_all():
        if pf.tree is None or not _REWARD_GATE.search(pf.text):
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call):
                fname = n.func.attr if isinstance(n.func, ast.Attribute) else (n.func.id if isinstance(n.func, ast.Name) else "")
                kws = {k.arg: k.value for k in n.keywords if k.arg}
                if fname.endswith("Rubric") or fname in ("Rubric", "RubricGroup") or "funcs" in kws:
                    funcs = kws.get("funcs") or kws.get("reward_funcs")
                    if funcs is None and n.args and isinstance(n.args[0], (ast.List, ast.Tuple)):
                        funcs = n.args[0]
                    if funcs is not None and isinstance(funcs, (ast.List, ast.Tuple)):
                        for e in funcs.elts:
                            nm = _callable_name(e)
                            if nm:
                                out.add(nm)
                if fname == "add_reward_func" and n.args:
                    nm = _callable_name(n.args[0])
                    if nm:
                        out.add(nm)
        for fi in pf.funcs:
            if any(re.search(r"\breward\b", d) for d in fi.decorators):
                out.add(fi.name)
    return out


def _scoring_funcs(pkg: Package):
    """Same set as rules.scoring_funcs: scorer named, reward decorated or rubric registered
    functions plus their direct callees in the package."""
    reg = _registered_reward_names(pkg)
    by_name: dict[str, list] = {}
    for pf, fi in all_funcs(pkg):
        by_name.setdefault(fi.name, []).append((pf, fi))
    seeds = []
    for pf, fi in all_funcs(pkg):
        dec = " ".join(fi.decorators)
        if fi.name in reg or "reward" in dec or "metric" in dec or SCORING_NAME.search(fi.name):
            seeds.append((pf, fi))
    out = {(pf.path, fi.start): (pf, fi) for pf, fi in seeds}
    for pf, fi in seeds:
        for cn in called_names(fi):
            for pf2, fi2 in by_name.get(cn, []):
                out.setdefault((pf2.path, fi2.start), (pf2, fi2))
    return list(out.values())


def _grader_funcs(pkg: Package):
    """(PyFile, FuncInfo) on the grading path: verifiers scoring functions (and their direct
    callees) plus every function of a Harbor verifier under tests/."""
    c = _cache(pkg)
    if "grader_funcs" in c:
        return c["grader_funcs"]
    out = _scoring_funcs(pkg)
    for pf in _harbor_test_py(pkg):
        for fi in pf.funcs:
            out.append((pf, fi))
    c["grader_funcs"] = out
    return out


def _reward_weights(pkg: Package) -> dict[str, list[tuple[float, float | None]]]:
    """name -> [(weight, share of its rubric's total weight or None)] for every place a function is
    put in a rubric: funcs=[...] with weights=[...] (verifiers defaults each weight to 1.0),
    add_reward_func(f, weight=w) and @reward(weight=w). add_metric(f) and @metric count as weight 0
    (logged, not paid). The share is known only for a literal funcs/weights list, where the
    rubric's total is in view."""
    c = _cache(pkg)
    if "weights" in c:
        return c["weights"]
    out: dict[str, list[tuple[float, float | None]]] = {}
    for pf in pkg.py_all():
        if pf.tree is None or not _REWARD_GATE.search(pf.text):
            continue
        for n in ast.walk(pf.tree):
            if not isinstance(n, ast.Call):
                continue
            fname = _callee(n)
            kws = {k.arg: k.value for k in n.keywords if k.arg}
            if fname.endswith("Rubric") or fname in ("Rubric", "RubricGroup") or "funcs" in kws:
                funcs = kws.get("funcs") or kws.get("reward_funcs")
                if funcs is None and n.args and isinstance(n.args[0], (ast.List, ast.Tuple)):
                    funcs = n.args[0]
                if not isinstance(funcs, (ast.List, ast.Tuple)):
                    continue
                weights = kws.get("weights")
                if weights is None and len(n.args) > 1 and isinstance(n.args[1], (ast.List, ast.Tuple)):
                    weights = n.args[1]
                literal, ws = True, []
                if isinstance(weights, (ast.List, ast.Tuple)):
                    for w in weights.elts:
                        x = _num(w)
                        literal = literal and x is not None
                        ws.append(1.0 if x is None else x)
                elif weights is not None:
                    literal = False
                names = [_callable_name(e) for e in funcs.elts]
                full = [ws[i] if i < len(ws) else 1.0 for i in range(len(names))]
                tot = sum(w for w in full if w > 0)
                for nm, w in zip(names, full):
                    if nm:
                        out.setdefault(nm, []).append((w, w / tot if literal and tot > 0 and w > 0 else None))
            elif fname == "add_reward_func" and n.args:
                nm = _callable_name(n.args[0])
                wn = kws.get("weight") or (n.args[1] if len(n.args) > 1 else None)
                w = _num(wn) if wn is not None else 1.0
                if nm:
                    out.setdefault(nm, []).append((1.0 if w is None else w, None))
            elif fname == "add_metric" and n.args:  # a metric is logged, not paid
                nm = _callable_name(n.args[0])
                if nm:
                    out.setdefault(nm, []).append((0.0, None))
        for fi in pf.funcs:
            for d in fi.decorators:
                if re.search(r"\breward\b", d):
                    m = re.search(r"weight\s*=\s*(-?[0-9.]+)", d)
                    out.setdefault(fi.name, []).append((float(m.group(1)) if m else 1.0, None))
                elif re.search(r"\bmetric\b", d):  # @vf.metric: logged, not paid
                    out.setdefault(fi.name, []).append((0.0, None))
    c["weights"] = out
    return out


def _payout_weight(pkg: Package, fi: FuncInfo):
    """-> (registered only at weight 0 or below, largest rubric share or None) for a function, judged by the
    rubric entries of the function itself and of its direct callers. Unknown means keep."""
    W = _reward_weights(pkg)
    if not W:
        return False, None
    c = _cache(pkg)
    if "callers" not in c:
        m: dict[str, set[str]] = {}
        for _, g in all_funcs(pkg):
            for cn in called_names(g):
                m.setdefault(cn, set()).add(g.name)
        c["callers"] = m
    names = {fi.name} | c["callers"].get(fi.name, set())
    entries = [e for nm in names for e in W.get(nm, [])]
    if not entries:
        return False, None
    if all(w <= 0 for w, _ in entries):
        return True, 0.0  # weight 0 (logged only) or negative (a penalty: a high value lowers reward)
    shares = [sh for w, sh in entries if w > 0]
    if not shares or any(sh is None for sh in shares):
        return False, None
    return False, max(shares)


def _own_nodes(fnode):
    """Nodes of a function body, not descending into nested defs or lambdas (they are
    their own FuncInfo)."""
    stack = list(ast.iter_child_nodes(fnode))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def _num(n):
    """Numeric value of a literal (True counts as 1), else None."""
    if isinstance(n, ast.Constant):
        v = n.value
        if v is True:
            return 1.0
        if v is False:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        v = _num(n.operand)
        return -v if v is not None else None
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("float", "int") and len(n.args) == 1:
        return _num(n.args[0])
    return None


def _callee(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _strings(pkg: Package) -> str:
    """All grader side source text: env python, Harbor verifier python and shell scripts."""
    c = _cache(pkg)
    if "text" not in c:
        parts = [pf.text for pf in pkg.py_all()] + [pf.text for pf in _harbor_test_py(pkg)]
        parts += [t for p, t in pkg.files.items() if p.endswith((".sh", ".bash"))]
        c["text"] = "\n".join(parts)
    return c["text"]


# ---------------------------------------------------------------------------
# 5. grader-fails-open
# Evidence row 5 (section 9.2), arxiv:2608.31119 (PaperGym reward code returns 0.5 on
# grader failure), arxiv:2609.02246 (a parser that defaulted every rating to 3),
# inspect-evals-lint IEBP012, FAPO reward_fn (cons:81fcd69e).
# ---------------------------------------------------------------------------

# variables that hold the judge's or checker's verdict (grader side)
JUDGE_VAR = re.compile(r"(verdict|judg|rating|grading|critic|feedback|review|evaluation|eval_out|eval_res|llm_out|"
                       r"llm_resp|grader|^grade$|^grades$|^score_?(str|text|raw|resp\w*|match|m)?$|^scores$)", re.I)
# variables that hold the policy's reply or what was parsed from it (model side)
MODEL_VAR = re.compile(r"(completion|response|reply|output|answer_text|model_answer|^pred|prediction|parsed|extracted|"
                       r"^text$|^content$|generated|^final\w*$|^raw$|^m$|^mo$|match|^res$|result|^answer$|^ans$)", re.I)
# names in MODEL_VAR that say nothing on their own (a regex match, a result): model side only when
# the function assigns them from the reply
WEAK_MODEL_VAR = re.compile(r"^(m|mo|match\w*|\w*_match|res|result\w*|raw|final\w*|ans)$", re.I)
# a check whose True means something bad was found: True is not a pass there
NEGATIVE_PREDICATE = re.compile(r"(escape|touch|violat|leak|unsafe|danger|exploit|vulnerab|crash|detect|suspicious|"
                                r"forbid|protected|block|attack|inject|harm|monitor|cheat|hack|flag|refus|error|fail|"
                                r"invalid|reject|penal|banned|abuse|tamper|overflow|timeout|timed_out)", re.I)
# the reference side: a missing gold is a data problem, not a grader failure
GOLDISH = re.compile(r"(gold|expected|^ref|reference|target|truth|^label|solution|^key|rubric|criteri|keyword|"
                     r"required|^correct|^answers?$|^gt$|oracle|^info$|^task|^example|^question|^prompt|"
                     r"config|^cfg|^args|^kwargs|^state$|^self$|^metadata|^board|^env$)", re.I)
JUDGE_CALL = re.compile(r"(judge|grade|grading|rate|rating|verdict|classif|chat|completion|generate|call_llm|llm|"
                        r"invoke|evaluat|critic|ask_model|query_model)", re.I)
JUDGE_FN = re.compile(r"(judg|verdict|critic|rating|grade|grading|review)", re.I)
PARSE_CALL = re.compile(r"^(search|match|fullmatch|findall|finditer|loads|literal_eval|_?parse\w*|_?extract\w*|"
                        r"get_score|read_score)$")
FORMAT_FN = re.compile(r"(format|xml|tag|structure|style|length|len_|brevity|valid_json|schema|parseable|parsable)", re.I)
SCOREISH = re.compile(r"(score|reward|rating|grade|verdict|correct|passed|pass_rate|judg|points|accuracy|success)", re.I)
FAILOPEN_SKIP_FN = re.compile(r"(^main$|^_?not_|_not_|_not$|absent|missing|^__\w+__$|^test_|setup|teardown|"
                              r"cleanup|download|^load_|_config$|^config|connect|^build|^make_|^gen_|plot)", re.I)
HELPER_PATH = re.compile(r"(^|/)(scripts?|tools|analysis|notebooks?|examples?|data_prep|benchmarks?|experiments?|"
                         r"docs?|outputs)/|(^|/)(build|make|gen|generate|prepare|convert|download|create|plot|"
                         r"analy[sz]e|setup)_\w*\.py$")
SKIP_EXC = {"StopIteration", "StopAsyncIteration", "GeneratorExit", "KeyboardInterrupt"}
MODEL_EXC = re.compile(r"^\(?(SyntaxError|IndentationError|TabError)(\s*,\s*(SyntaxError|IndentationError|TabError))*\)?$")
TIMEOUT_EXC = re.compile(r"(Timeout|TimeoutExpired|TimeoutError|CancelledError)")
POS_LITERAL = re.compile(r"(\breturn\s+\(?(True\b|[0-9.]*[1-9])|=\s*(True\b|[0-9.]*[1-9])|:\s*[0-9.]*[1-9]|"
                         r"\belse\s+[0-9.]*[1-9]|,\s*[0-9.]*[1-9][0-9.]*\s*\)|write\w*\(\s*['\"]1)")


def _assignments(own) -> dict[str, list[tuple[str, set[str]]]]:
    """name -> [(callee, names in the call's arguments)] for x = f(...) / x = await f(...) in the function."""
    out: dict[str, list[tuple[str, set[str]]]] = {}
    for n in own:
        if not isinstance(n, (ast.Assign, ast.AnnAssign)) or n.value is None:
            continue
        tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
        val = n.value.value if isinstance(n.value, ast.Await) else n.value
        calls = []
        for c in ast.walk(val):
            if isinstance(c, ast.Call):
                args = set()
                for a in list(c.args) + [k.value for k in c.keywords]:
                    args |= _names_in(a)
                if isinstance(c.func, ast.Attribute):
                    args |= _names_in(c.func.value)  # judge_response.strip() / resp.json()
                calls.append((_callee(c), args))
        if not calls and isinstance(val, (ast.Name, ast.Attribute, ast.Subscript)):
            calls.append(("", _names_in(val)))  # plain copy: text = judge_response
        for t in tgts:
            for x in ast.walk(t):
                if isinstance(x, ast.Name):
                    out.setdefault(x.id, []).extend(calls)
    return out


def _returned_names(own) -> set[str]:
    out = set()
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None:
            for x in ast.walk(n.value):
                if isinstance(x, ast.Name):
                    out.add(x.id)
    return out


def _positive_payout(stmt, returned: set[str]):
    """If a statement pays a positive constant reward, return that value, else None."""
    if isinstance(stmt, ast.Return) and stmt.value is not None:
        v = stmt.value
        x = _num(v)
        if x is not None:
            return x if x > 0 else None
        if isinstance(v, ast.Tuple) and v.elts:
            x = _num(v.elts[0])
            return x if x is not None and x > 0 else None
        if isinstance(v, ast.Dict):
            for k, val in zip(v.keys, v.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and SCOREISH.search(k.value):
                    x = _num(val)
                    if x is not None and x > 0:
                        return x
        return None
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        val = stmt.value
        x = _num(val) if val is not None else None
        if x is None or x <= 0:
            return None
        tgts = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for t in tgts:
            if isinstance(t, ast.Name) and SCOREISH.search(t.id) and t.id in returned:
                return x
            if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str) \
                    and re.search(r"^(reward|score|passed|correct|success|is_correct|grade)$", t.slice.value):
                return x
            if isinstance(t, ast.Attribute) and re.search(r"^(reward|score)$", t.attr):
                return x
        return None
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        # write_reward(1.0) / Path(".../reward.txt").write_text("1")
        c = stmt.value
        try:
            txt = ast.unparse(c)
        except Exception:
            return None
        if re.search(r"(reward|score)", _callee(c), re.I) or re.search(r"reward\.(txt|json)", txt):
            for a in c.args:
                x = _num(a)
                if x is None and isinstance(a, ast.Constant) and isinstance(a.value, str):
                    try:
                        x = float(a.value.strip())
                    except ValueError:
                        x = None
                if x is not None and x > 0:
                    return x
    return None


def _is_logging(stmt) -> bool:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        c = stmt.value
        f = c.func
        recv = ast.unparse(f.value) if isinstance(f, ast.Attribute) else ""
        return _callee(c) in ("print", "debug", "info", "warning", "warn", "error", "exception", "log") or \
            bool(re.search(r"(log|logger|logging|warnings)", recv))
    return isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))


def _scale(own) -> float:
    vals = []
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None:
            x = _num(n.value)
            if x is not None:
                vals.append(x)
    return max(vals) if vals else 1.0


def _nm(e):
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self":
        return e.attr
    return None


def _cond_var(test):
    """`x is None`, `not x`, `x == ""`, `len(x) == 0`, `x is None or ...` -> 'x' (Name or self.x)."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        for v in test.values:
            r = _cond_var(v)
            if r:
                return r
        return None
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _nm(test.operand)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, right = test.ops[0], test.comparators[0]
        left = test.left
        if isinstance(op, (ast.Is, ast.Eq)) and isinstance(right, ast.Constant) and right.value in (None, ""):
            return _nm(left)
        if isinstance(op, ast.Eq) and isinstance(left, ast.Call) and _callee(left) == "len" and left.args \
                and _num(right) == 0:
            return _nm(left.args[0])
    return None


def _truthy_var(test):
    """`m`, `m is not None`, `len(m) > 0` -> 'm'."""
    if isinstance(test, ast.Name):
        return test.id
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.left, ast.Name):
        op, right = test.ops[0], test.comparators[0]
        if isinstance(op, ast.IsNot) and isinstance(right, ast.Constant) and right.value is None:
            return test.left.id
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.left, ast.Call) \
            and _callee(test.left) == "len" and test.left.args and isinstance(test.left.args[0], ast.Name) \
            and isinstance(test.ops[0], (ast.Gt, ast.GtE, ast.NotEq)):
        return test.left.args[0].id
    return None


def _names_in(e) -> set[str]:
    return {x.id for x in ast.walk(e) if isinstance(x, ast.Name)} | \
        {x.attr for x in ast.walk(e) if isinstance(x, ast.Attribute)}


_MODULE_NAMES = {"re", "json", "math", "ast", "np", "numpy", "regex", "str", "int", "float", "len", "list", "dict"}


def _side(var: str, fi: FuncInfo, asg, depth=0) -> str | None:
    """'judge' when var holds the judge's or checker's reply (or something parsed from it), 'model'
    when it holds the policy's reply or something parsed from it, else None. A value parsed out of
    the gold answer or other task data (and nothing else) is None: its failure is a data issue."""
    if GOLDISH.search(var):
        return None
    if JUDGE_VAR.search(var):
        return "judge"
    calls = asg.get(var, [])
    if any(JUDGE_CALL.search(c) and not PARSE_CALL.match(c) for c, _ in calls):
        return "judge"
    srcs = set()
    for c, args in calls:
        for a in args:
            if a == var or a in _MODULE_NAMES:
                continue
            if GOLDISH.search(a):
                srcs.add("gold")
            elif depth < 2:
                sa = _side(a, fi, asg, depth + 1)
                if sa:
                    srcs.add(sa)
    if "judge" in srcs:
        return "judge"
    named = MODEL_VAR.search(var) and not (WEAK_MODEL_VAR.match(var) and not calls)
    if "model" in srcs or (named and "gold" not in srcs):
        return "judge" if JUDGE_FN.search(fi.name) else "model"
    return None


CONFIG_PARAM = re.compile(r"(threshold|limit|timeout|max_|min_|_max|_min|path|dir|file|url|_id$|^id$|size|^num_|^n_|"
                          r"count|seed|config|cfg|mode|name|key|client|logger|verbose|debug|retries|attempts|"
                          r"duration|tolerance|tol$|weight)", re.I)


def _graded_try(t: ast.Try, fi: FuncInfo, asg) -> bool:
    """The try body handles the judge reply or the model output: it calls a judge, reads a value
    whose side is judge or model, or passes one of the function's inputs (a parameter that is not
    the reference, task data or a setting) to a call. A try that only parses the gold answer or
    other task data is not a grader failure."""
    params = {x for x in fi.params if not GOLDISH.search(x) and not CONFIG_PARAM.search(x)}
    logged = set()
    for s_ in t.body:
        for x in ast.walk(s_):
            if isinstance(x, ast.Call) and _is_logging(ast.Expr(value=x)):
                logged |= {id(y) for y in ast.walk(x)}
    for s_ in t.body:
        for x in ast.walk(s_):
            if id(x) in logged:
                continue
            if isinstance(x, ast.Call):
                if JUDGE_CALL.search(_callee(x)) and not PARSE_CALL.match(_callee(x)):
                    return True
                for a in list(x.args) + [k.value for k in x.keywords] + \
                        ([x.func.value] if isinstance(x.func, ast.Attribute) else []):
                    if any(isinstance(nm, ast.Name) and nm.id in params for nm in ast.walk(a)):
                        return True
            elif isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id not in _MODULE_NAMES \
                    and not GOLDISH.search(x.id) and _side(x.id, fi, asg) in ("judge", "model"):
                return True
    return False


def _bool_true(stmt) -> bool:
    """`return True` or `return True, msg`: a check function's pass value."""
    if not isinstance(stmt, ast.Return) or stmt.value is None:
        return False
    v = stmt.value
    if isinstance(v, ast.Tuple) and v.elts:
        v = v.elts[0]
    return isinstance(v, ast.Constant) and v.value is True


def _const_returns(own):
    """(line, value) of every return of a literal number or bool, both arms of a literal IfExp too."""
    out = []
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None:
            v = n.value.elts[0] if isinstance(n.value, ast.Tuple) and n.value.elts else n.value
            if isinstance(v, ast.Dict):
                v = next((val for k, val in zip(v.keys, v.values) if isinstance(k, ast.Constant)
                          and isinstance(k.value, str) and re.search(r"^(reward|score)$", k.value)), v)
            arms = [v.body, v.orelse] if isinstance(v, ast.IfExp) else [v]
            for a in arms:
                x = _num(a)
                if x is not None:
                    out.append((n.lineno, x))
    return out


def _blocks(fnode, own):
    """Every statement list in the function (its body and the bodies of if/for/while/with/try
    and handlers)."""
    yield fnode.body
    for n in own:
        for fld in ("body", "orelse", "finalbody"):
            b = getattr(n, fld, None)
            if isinstance(b, list) and b and isinstance(b[0], ast.stmt):
                yield b


def _denominator_names(own) -> set[str]:
    out = set()
    for n in own:
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv)):
            for c in ast.walk(n.right):
                if isinstance(c, ast.Call) and _callee(c) == "len" and c.args:
                    out |= _names_in(c.args[0])
    return out


# v0.4: a pairwise or multiple-choice verdict is compared with chance, not with the wrong-answer value
PAIRWISE_CTX = re.compile(r"(pairwise|winner|prefer|chosen|rejected|swap|option|choice|letter|multiple.?choice|mcq|slot)", re.I)


def _chance(pf: PyFile, fi: FuncInfo) -> float | None:
    """1 / number of options when the function grades a pairwise or multiple-choice verdict whose option labels
    (single capital letters) are written in it or in a function it calls in the same file, else None."""
    if not PAIRWISE_CTX.search(fi.name + " " + fi.src):
        return None
    local = {g.name: g for g in pf.funcs}
    nodes = list(ast.walk(fi.node))
    for cn in called_names(fi):
        if cn in local and local[cn] is not fi:
            nodes += list(ast.walk(local[cn].node))
    letters = {n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and len(n.value) == 1 and n.value.isupper()}
    if len(letters) < 2 or len(letters) > 10 or letters != {chr(65 + i) for i in range(len(letters))}:
        return None
    # a verdict pays all or nothing: every return is a literal, a comparison, or a name only ever set to literals
    own = list(_own_nodes(fi.node))
    lit_names = {}
    for n in own:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            lit_names.setdefault(n.targets[0].id, []).append(_num(n.value) is not None)
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None:
            v = n.value.elts[0] if isinstance(n.value, ast.Tuple) and n.value.elts else n.value
            if _num(v) is not None or isinstance(v, ast.Compare) or (
                    isinstance(v, ast.IfExp) and _num(v.body) is not None and _num(v.orelse) is not None):
                continue
            if isinstance(v, ast.Name) and lit_names.get(v.id) and all(lit_names[v.id]):
                continue
            return None
    return 1.0 / len(letters)


METRIC_NAME = re.compile(r"(f1|accuracy|precision|recall|score|fidelity|valid|ratio|similarity|rate|frac|share|prob|"
                         r"coverage|overlap|match)", re.I)
_INF = float("inf")


def _bounds(e, consts: dict, depth: int = 0) -> tuple[float, float]:
    """Interval of a reward expression on the success path: literals, names assigned once to a literal,
    metric-named values (taken as 0 to 1), sums, positive scalings, min, max and if-else."""
    if depth > 12 or e is None:
        return -_INF, _INF
    x = _num(e)
    if x is not None:
        return x, x
    if isinstance(e, ast.Name):
        if e.id in consts:
            return consts[e.id]
        return (0.0, 1.0) if METRIC_NAME.search(e.id) else (-_INF, _INF)
    if isinstance(e, ast.IfExp):
        if isinstance(e.test, ast.Name) and e.test.id in consts:  # flag = True earlier: only one arm is reached
            return _bounds(e.body if consts[e.test.id][0] else e.orelse, consts, depth + 1)
        a, b = _bounds(e.body, consts, depth + 1), _bounds(e.orelse, consts, depth + 1)
        return min(a[0], b[0]), max(a[1], b[1])
    if isinstance(e, ast.BinOp):
        a, b = _bounds(e.left, consts, depth + 1), _bounds(e.right, consts, depth + 1)
        if isinstance(e.op, ast.Add):
            return a[0] + b[0], a[1] + b[1]
        if isinstance(e.op, ast.Sub):
            return a[0] - b[1], a[1] - b[0]
        if isinstance(e.op, ast.Mult) and min(a[0], b[0]) >= 0:
            return a[0] * b[0], a[1] * b[1]
        return -_INF, _INF
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.args:
        bs = [_bounds(a, consts, depth + 1) for a in e.args]
        if e.func.id == "min":
            return min(b[0] for b in bs), min(b[1] for b in bs)
        if e.func.id == "max":
            return max(b[0] for b in bs), max(b[1] for b in bs)
        if e.func.id in ("float", "round") and len(bs) >= 1:
            return bs[0]
    return -_INF, _INF


def _success_floor(fi: FuncInfo, own, line: int) -> float | None:
    """Lowest reward a computed return after `line` can pay (the success path), or None when it is unknown."""
    once: dict[str, list] = {}
    for n in own:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            once.setdefault(n.targets[0].id, []).append(n)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and isinstance(n.target, ast.Name):
            once.setdefault(n.target.id, []).append(n)
    consts = {}
    for nm, asg in once.items():
        if len(asg) == 1 and isinstance(asg[0], ast.Assign) and _num(asg[0].value) is not None:
            consts[nm] = (_num(asg[0].value),) * 2
    lows = []
    for n in own:
        if not isinstance(n, ast.Return) or n.value is None or n.lineno <= line:
            continue
        v = n.value
        if isinstance(v, ast.Dict):
            v = next((val for k, val in zip(v.keys, v.values) if isinstance(k, ast.Constant)
                      and isinstance(k.value, str) and re.search(r"^(reward|score)$", k.value)), None)
            if v is None:
                continue
        if _num(v) is not None or (isinstance(v, ast.IfExp) and _num(v.body) is not None and _num(v.orelse) is not None):
            continue  # literal returns are already in the wrong-answer floor
        if isinstance(v, ast.Name):
            # follow the variable through its assignments after the flagged line, in order
            seq = sorted([a for a in once.get(v.id, []) if line < a.lineno < n.lineno], key=lambda a: a.lineno)
            if not seq:
                return None
            cur = None
            for a in seq:
                if isinstance(a, ast.AugAssign):
                    return None
                c2 = dict(consts)
                if cur is not None:
                    c2[v.id] = cur
                nb = _bounds(a.value, c2)
                # a later assignment may sit in a branch, so the old value can still be returned
                cur = nb if cur is None else (min(cur[0], nb[0]), max(cur[1], nb[1]))
            lo = cur[0] if cur else -_INF
        else:
            lo = _bounds(v, consts)[0]
        if lo == -_INF:
            return None
        lows.append(lo)
    return min(lows) if lows else None


def _failopen_hits(pf: PyFile, fi: FuncInfo):
    """-> list of (line, value, cause) for one grader function."""
    fnode = fi.node
    if FAILOPEN_SKIP_FN.search(fi.name) or not POS_LITERAL.search(fi.src):
        return []
    if re.search(rf"exit\(\s*(await\s+)?{re.escape(fi.name)}\(", pf.text):
        return []  # its return value is a process exit status
    own = list(_own_nodes(fnode))
    returned = _returned_names(own)
    asg = _assignments(own)
    params = set(fi.params)
    scale = _scale(own)
    model_ok = not FORMAT_FN.search(fi.name)
    hits = []

    def model_pays(v):  # a missing policy answer only counts at half credit or more
        return model_ok and v >= 0.5 * max(scale, 1.0)

    parent = {}
    for n in own:
        for ch in ast.iter_child_nodes(n):
            parent[ch] = n
    for ch in ast.iter_child_nodes(fnode):
        parent[ch] = fnode

    def gold_branch(node):  # inside an if whose test reads the reference (`if not expected: ...`)
        p = parent.get(node)
        while p is not None and p is not fnode:
            if isinstance(p, ast.If) and any(GOLDISH.search(x) for x in _names_in(p.test)):
                return True
            p = parent.get(p)
        return False

    negative_check = bool(NEGATIVE_PREDICATE.search(fi.name))

    # (a) an except branch pays a positive constant, around the judge reply or the model output
    for t in own:
        if not isinstance(t, ast.Try):
            continue
        graded = None
        for h in t.handlers:
            tname = ast.unparse(h.type) if h.type is not None else "Exception"
            if any(x in tname for x in SKIP_EXC):
                continue
            if any(isinstance(x, ast.Raise) for x in h.body):
                continue
            flat = []
            for x in h.body:
                flat.append(x)
                if isinstance(x, ast.If):  # `except: if retries_left: continue / else: return 0.5`
                    flat += list(x.orelse)
            for x in flat:
                v = _positive_payout(x, returned)
                if v is None:
                    continue
                if graded is None:
                    graded = _graded_try(t, fi, asg)
                if not graded:
                    break  # only the gold answer or task data can raise here
                if _bool_true(x) and negative_check:
                    break  # True means "found something bad" here, so the branch fails closed
                if MODEL_EXC.match(tname):
                    if model_pays(v):
                        hits.append((x.lineno, v, "the submitted code does not compile"))
                else:
                    hits.append((x.lineno, v, "the grader times out" if TIMEOUT_EXC.search(tname) else "the grader raises"))
                break

    denoms = _denominator_names(own)
    # (b) a missing or empty verdict pays a positive constant (early exit guard)
    for n in own:
        if not isinstance(n, ast.If):
            continue
        var = _cond_var(n.test)
        if not var or var in denoms:  # an empty denominator is verdict-parse-semantics
            continue
        if any(GOLDISH.search(x) for x in _names_in(n.test)):
            continue
        first = next((x for x in n.body if not _is_logging(x)), None)
        v = _positive_payout(first, returned) if first is not None else None
        if v is None or _bool_true(first):  # an early `return True` is a check's pass value
            continue
        if var in params and re.search(r"^(judge|judge_client|judge_fn|client|grader|judge_model|scorer)$", var):
            hits.append((first.lineno, v, "no judge is configured"))
            continue
        if gold_branch(n):
            continue  # a branch for what the reference expects, not a grader failure
        side = _side(var, fi, asg)
        if side == "judge":
            hits.append((first.lineno, v, "the judge returns no verdict"))
        elif side == "model" and model_pays(v) and any(PARSE_CALL.match(c) for c, _ in asg.get(var, [])):
            # a parse of the reply that found nothing (an empty reply guard is a "nothing to grade" exit)
            hits.append((first.lineno, v, "the reply has no answer"))

    # (c) an unparsable reply falls through to a positive constant
    for b in _blocks(fnode, own):
        for i, n in enumerate(b):
            if not isinstance(n, ast.If):
                continue
            var = _truthy_var(n.test)
            if not var or not any(PARSE_CALL.match(c) for c, _ in asg.get(var, [])):
                continue
            def _first(v):
                return v.elts[0] if isinstance(v, ast.Tuple) and v.elts else v
            if not any(isinstance(x, ast.Return) and x.value is not None and _num(_first(x.value)) is None
                       for x in ast.walk(n)):
                continue
            tail = n.orelse[0] if len(n.orelse) == 1 else (b[i + 1] if i + 1 < len(b) else None)
            if not isinstance(tail, ast.Return) or _bool_true(tail):
                continue
            v = _positive_payout(tail, returned)
            if v is None:
                continue
            side = _side(var, fi, asg)
            if side == "judge":
                hits.append((tail.lineno, v, "the judge reply cannot be parsed"))
            elif side == "model" and model_pays(v):
                hits.append((tail.lineno, v, "the reply cannot be parsed"))
    def pays_out(node):  # the value is returned or stored as the score
        p = parent.get(node)
        while isinstance(p, (ast.BinOp, ast.UnaryOp)) or (
                isinstance(p, ast.Call) and _callee(p) in ("float", "int", "min", "max", "round", "abs")):
            p = parent.get(p)
        if isinstance(p, ast.Return):
            return True
        if isinstance(p, (ast.Assign, ast.AnnAssign)):
            tg = p.targets if isinstance(p, ast.Assign) else [p.target]
            return any(isinstance(t, ast.Name) and SCOREISH.search(t.id) for t in tg)
        return False

    for n in own:
        if isinstance(n, ast.IfExp):
            if not pays_out(n):
                continue
            var = _truthy_var(n.test)
            v = _num(n.orelse)
            if not var or v is None or v <= 0 or var not in _names_in(n.body):
                continue
            if not any(PARSE_CALL.match(c) for c, _ in asg.get(var, [])):
                continue
            side = _side(var, fi, asg)
            if side == "judge":
                hits.append((n.lineno, v, "the judge reply cannot be parsed"))
            elif side == "model" and model_pays(v):
                hits.append((n.lineno, v, "the reply cannot be parsed"))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and len(n.args) == 2:
            key, dflt = n.args
            v = _num(dflt)
            if v is None or v <= 0 or not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if not re.search(r"^(score|rating|grade|verdict|reward|correct|passed|judgment|judgement)$", key.value, re.I):
                continue
            recv = n.func.value
            rname = recv.id if isinstance(recv, ast.Name) else ""
            if rname and _side(rname, fi, asg) == "judge" and pays_out(n):
                hits.append((n.lineno, v, "the judge reply has no score field"))
    # (d) a score variable starts at a positive default that only a successful judge call or
    # parse overwrites:  score = 3 / try: score = parse(judge(...)) except: pass / return score
    for b in _blocks(fnode, own):
        for i, n in enumerate(b):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)):
                continue
            name = n.targets[0].id
            v = _num(n.value)
            if v is None or v <= 0 or name not in returned or not SCOREISH.search(name):
                continue
            for m in b[i + 1:i + 6]:
                if isinstance(m, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in m.targets):
                    break  # unconditionally overwritten: the default is never paid
                over = None
                if isinstance(m, ast.Try):
                    sets = any(isinstance(x, (ast.Assign, ast.AugAssign)) and any(
                        isinstance(t, ast.Name) and t.id == name for t in
                        (x.targets if isinstance(x, ast.Assign) else [x.target])) for s_ in m.body for x in ast.walk(s_))
                    calls = {_callee(c) for s_ in m.body for c in ast.walk(s_) if isinstance(c, ast.Call)}
                    quiet = all(not any(isinstance(x, (ast.Raise, ast.Return)) or (
                        isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in x.targets))
                        for s_ in h.body for x in ast.walk(s_)) for h in m.handlers)
                    if sets and quiet and any(JUDGE_CALL.search(c) or PARSE_CALL.match(c) for c in calls):
                        over = "the grader raises"
                elif isinstance(m, ast.If):
                    var = _truthy_var(m.test)
                    sets = any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in x.targets)
                               for s_ in m.body for x in ast.walk(s_))
                    if var and sets and not m.orelse and any(PARSE_CALL.match(c) for c, _ in asg.get(var, [])) \
                            and _side(var, fi, asg) == "judge":
                        over = "the judge reply cannot be parsed"
                if over:
                    hits.append((n.lineno, v, over))
                    break

    # (e) a verdict chain over the judge reply whose last branch pays a positive constant:
    # if "YES" in r: return 1.0 / elif "NO" in r: return 0.0 / else: return 0.5
    for b in _blocks(fnode, own):
        for i, n in enumerate(b):
            if not isinstance(n, ast.If):
                continue
            chain, cur = [], n
            while isinstance(cur, ast.If):
                chain.append(cur)
                cur = cur.orelse[0] if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If) else None
            if len(chain) < 2:
                continue
            last_if = chain[-1]
            tail = last_if.orelse[0] if len(last_if.orelse) == 1 else (b[i + 1] if i + 1 < len(b) else None)
            if not isinstance(tail, ast.Return):
                continue
            tv = _positive_payout(tail, returned)
            if tv is None:
                continue
            vals, var = [], None
            for c in chain:
                lit = [x for x in ast.walk(c.test) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
                if not lit or not isinstance(c.body[-1], ast.Return) or _num(c.body[-1].value) is None:
                    vals = []
                    break
                vals.append(_num(c.body[-1].value))
                names = [x for x in _names_in(c.test) if x not in ("lower", "upper", "strip", "startswith", "endswith")]
                var = var or (names[0] if names else None)
            if not vals or var is None or min(vals) >= tv or _side(var, fi, asg) != "judge":
                continue
            hits.append((tail.lineno, tv, "the judge reply matches no verdict"))

    execish = re.search(r"(run|exec|subprocess|cmd|command|process|shell|spawn)", fi.name, re.I)

    # values above 10 are not rewards (a confidence of 100, exit status 124), and an exec
    # helper's positive integer is an exit status, which means failure
    def exit_status(line):  # `return 1` / `return 1, msg` from a helper that runs a process
        return bool(execish) and bool(re.match(r"\s*return\s+\(?\s*\d+\s*(,|\)|#|$)", pf.line(line)))
    hits = [x for x in hits if x[1] <= 10 and not (execish and x[1] >= 2) and not exit_status(x[0])]
    if not hits:
        return []
    # The fallback must pay more than the function's own wrong-answer return, the value a reply
    # that parses but is wrong gets. That is taken from the literal returns after the flagged line
    # (they are reached once parsing succeeds), else from the earlier ones, leaving out the flagged
    # lines, other except branches, earlier "nothing to grade" guards (if not x: return ...) and
    # returns of the same value (another fallback of the same kind).
    flagged = {x[0] for x in hits}
    excepts, guards = set(), set()
    for t in own:
        if isinstance(t, ast.ExceptHandler):
            excepts |= {x.lineno for s_ in t.body for x in ast.walk(s_) if isinstance(x, ast.Return)}
        elif isinstance(t, ast.If) and _cond_var(t.test):
            first = next((x for x in t.body if not _is_logging(x)), None)
            if isinstance(first, ast.Return):
                guards.add(first.lineno)
    consts = [(ln, v) for ln, v in _const_returns(own) if ln not in flagged and ln not in excepts]
    chance = _chance(pf, fi)
    out = []
    for ln, v, cause in hits:
        # a return of the same value is another fallback of the same kind, not a wrong answer,
        # unless it is a later "found nothing" guard (if not hits: return 0.1), which is an outcome
        after = [x for l2, x in consts if l2 > ln and (x != v or l2 in guards)]
        before = [x for l2, x in consts if l2 < ln and l2 not in guards and x != v]
        floor = min(after) if after else (min(before) if before else 0.0)
        # v0.4: the lowest reward a valid answer gets on the computed success path also bounds the gain
        sf = _success_floor(fi, own, ln)
        if sf is not None:
            floor = min(after + [sf]) if after else sf
        # v0.4: a pairwise or multiple-choice verdict at or below chance pays what a guess already earns
        if chance is not None and v <= chance * max(scale, 1.0) + 1e-9:
            continue
        if v >= 0.1 and v > floor:
            out.append((ln, v, cause, floor))
    return out


# v0.4: a fallback in a function that is only a metric can still reach a paid reward through `state`
STATE_SCORE_KEY = re.compile(r"(score|gate|reward|grade|pass|multiplier|factor|verdict|rating)", re.I)


_STATE_INIT: dict = {}  # (file, key) -> literal the key starts at in a state.update({...})


def _state_keys_written(pf: PyFile) -> dict[str, int]:
    """Score-like keys the file writes into state: state["k"] = ..., state.update({"k": ...}),
    state.setdefault("k", ...)."""
    out: dict[str, int] = {}
    if pf.tree is None:
        return out
    for n in ast.walk(pf.tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str) \
                        and re.search(r"state$", _unparse_safe(t.value)):
                    out.setdefault(t.slice.value, n.lineno)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and re.search(r"state$", _unparse_safe(n.func.value)):
            if n.func.attr == "update" and n.args and isinstance(n.args[0], ast.Dict):
                for k, val in zip(n.args[0].keys, n.args[0].values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        out.setdefault(k.value, k.lineno)
                        if _num(val) is not None:
                            _STATE_INIT.setdefault((pf.path, k.value), _num(val))
            elif n.func.attr == "setdefault" and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                out.setdefault(n.args[0].value, n.lineno)
    return {k: v for k, v in out.items() if STATE_SCORE_KEY.search(k)}


def _unparse_safe(n) -> str:
    try:
        return ast.unparse(n)
    except Exception:
        return ""


def _state_route(pkg: Package, pf: PyFile):
    """(paid function, its file, line, key, used as a multiplier, condition text) when a paid reward reads a
    score-like state key that `pf` writes, else None."""
    keys = _state_keys_written(pf)
    if not keys:
        return None
    W = _reward_weights(pkg)
    paid = {nm for nm, ws in W.items() if any(w > 0 for w, _ in ws)}
    for pf2, g in all_funcs(pkg):
        if g.name not in paid:
            continue
        parent = {}
        for x in ast.walk(g.node):
            for ch in ast.iter_child_nodes(x):
                parent[ch] = x
        for n in ast.walk(g.node):
            key = None
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args \
                    and isinstance(n.args[0], ast.Constant) and n.args[0].value in keys \
                    and re.search(r"state$", _unparse_safe(n.func.value)):
                key = n.args[0].value
                if len(n.args) > 1 and _num(n.args[1]) is not None:
                    _STATE_INIT.setdefault((pf2.path, "get:" + key), _num(n.args[1]))
            elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load) and isinstance(n.slice, ast.Constant) \
                    and n.slice.value in keys and re.search(r"state$", _unparse_safe(n.value)):
                key = n.slice.value
            if key is None:
                continue
            mult, cond, cur = False, None, n
            while cur in parent:
                up = parent[cur]
                if isinstance(up, ast.BinOp) and isinstance(up.op, ast.Mult):
                    mult = True
                if isinstance(up, ast.If) and cur is not up.test and cond is None:
                    t = up.test
                    if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq) \
                            and isinstance(t.left, ast.Name) and isinstance(t.comparators[0], ast.Constant):
                        cond = f"{t.left.id} == {t.comparators[0].value!r}"
                cur = up
            return g, pf2, n.lineno, key, mult, cond
    return None


@rule("grader-fails-open", "hack", "high",
      "A reward function pays a positive constant when the grader raises, times out or returns no "
      "parsable verdict, so breaking the grader earns reward.")
def r_grader_fails_open(pkg: Package):
    out = _failopen_core(pkg)
    # v0.5 widening (denominator-drops-failed-items): a mean over only the items that were judged or answered
    return out + _denominator_drop_finding(pkg)


def _failopen_core(pkg: Package):
    _STATE_INIT.clear()  # per package
    found, routed = [], []
    seen = set()
    for pf, fi in _grader_funcs(pkg):
        if HELPER_PATH.search(pf.path):
            continue
        hits = _failopen_hits(pf, fi)
        if not hits:
            continue
        zero, share = _payout_weight(pkg, fi)
        route = None
        if zero:
            # in the rubric at weight 0 (logged, never paid) or below 0 (a penalty), unless the fallback
            # reaches a paid reward through a score kept in state (v0.4)
            route = _state_route(pkg, pf)
            if route is None:
                continue
        for line, v, cause, floor in hits:
            if route is None and share is not None and share * (v - floor) < 0.05:
                continue  # the gain over a wrong answer is under 0.05 of the total reward
            key = (pf.path, line)
            if key in seen:
                continue
            seen.add(key)
            if route is not None:
                routed.append((pf, fi, line, v, cause, route))
            else:
                found.append((pf, fi, line, v, cause))
    out = []
    if routed and not found:
        pf, fi, line, v, cause, (g, gpf, gln, skey, mult, cond) = routed[0]
        shown = int(v) if float(v).is_integer() else v
        when = f" when {cond}" if cond else ""
        if mult:
            starts = {_STATE_INIT.get((pf.path, skey)), _STATE_INIT.get((gpf.path, "get:" + skey))} - {None}
            held = f" and leaves state['{skey}'] at {shown}" if starts and all(x == v for x in starts) else ""
            msg = (f"{fi.name} returns {shown} when {cause}{held}, and the paid {g.name} multiplies the score by "
                   f"state['{skey}']{when}, so a rollout that breaks the judge keeps its "
                   "full score and skips the judge's penalty. The fallback only removes a penalty: it cannot pay more "
                   "than the score the other terms give.")
        else:
            msg = (f"{fi.name} returns {shown} when {cause}, and the paid {g.name} reads state['{skey}']{when}, so a "
                   "rollout that breaks or confuses the judge still earns reward.")
        out.append(F("grader-fails-open", msg, [ev(p, ln) for p, _, ln, _, _, _ in routed] + [ev(gpf, gln)],
                     confidence="review", severity="low" if mult else "medium"))
        return out
    if not found:
        return []
    full = [x for x in found if x[3] >= 1.0 and x[3] >= _scale(list(_own_nodes(x[1].node)))]
    grader_side = [x for x in found if x[4] not in ("the reply has no answer", "the reply cannot be parsed",
                                                     "the submitted code does not compile")]
    top = (full or grader_side or found)[0]
    pf, fi, line, v, cause = top
    shown = int(v) if float(v).is_integer() else v
    msg = (f"{fi.name} returns {shown} when {cause}, so a rollout that breaks or confuses the grader "
           f"still earns reward" + (f" ({len(found)} such branches)." if len(found) > 1 else "."))
    return [F("grader-fails-open", msg, [ev(p, ln) for p, _, ln, _, _ in found],
              confidence="review", severity="high" if full else "medium")]


# ---------------------------------------------------------------------------
# 7. formal-checker-escape-hatch (extends lean-no-axiom-check)
# Evidence row 7 (section 9.2), arxiv:2603.21065 (999 of 1,024 hacked proofs passed syntax and
# target checks), doi:10.5281/zenodo.24346906 (axiom-only reward paid statement swaps in 74 of
# 96), Vals debug.skipKernelTC incident, OR:g1o5hm271c (local notation), OR:igkknzztsd (Dafny,
# Verus channels), GH:formal-verification-rsi.
# lean-no-axiom-check fires when a Lean grader has no audit at all; this rule covers the case it
# stays silent on (some ban list or axiom check exists) and the other proof languages.
# ---------------------------------------------------------------------------

# how a grader runs Lean: literal markers, plus `lean <flag|path|{var}>` in a command string.
# Literal checks and a literal-prefixed regex keep this fast on large data heavy packages.
LEAN_LITS = ("lake env", "lake build", "lake exe", "LeanTaskset", "lean_check", "kimina", "LeanServer", "lean_server",
             "lean_repl", "leanrepl", "repl.lean", "Mathlib", "'lean'", '"lean"')
LEAN_CMD = re.compile(r"lean\s+(--|-[a-zA-Z]|\{|\$|/)")


def _lean_used(text: str) -> bool:
    return any(x in text for x in LEAN_LITS) or bool(LEAN_CMD.search(text))


LEAN_PROOF = re.compile(r"(theorem|lemma|proof|sorry|tactic)", re.I)
# audit tokens a Lean grader can use (any of them counts as "some audit exists")
LEAN_AUDIT = re.compile(r"(#print\s+axioms|print_axioms|collect_axioms|CollectAxioms|axioms_used|['\"]axiom['\"]|"
                        r"\baxiom\b\s*['\"\\]|\\baxiom|axiom\s+in\b|forbidden|banned|blacklist|blocklist|denylist|"
                        r"sorryAx|['\"]sorry['\"]|\\bsorry|sorry\s+in\b|lean_common|lean4checker|SafeVerify|safe_verify|"
                        r"comparator)")
LEAN_ALLOWLIST = re.compile(r"(propext[^\n]{0,120}Quot\.sound|Quot\.sound[^\n]{0,120}propext|Classical\.choice[^\n]{0,120}propext|"
                            r"STANDARD_AXIOMS|ALLOWED_AXIOMS|allowed_axioms|standard_axioms)")
KERNEL_REPLAY = re.compile(r"(lean4checker|leanchecker|SafeVerify|safe_verify|comparator)", re.I)
# (escape hatch, what covers it)
LEAN_HATCHES = [
    ("set_option debug.skipKernelTC", re.compile(r"(skipKernelTC|set_option|debug\.)")),
    ("native_decide / implemented_by", re.compile(r"(native_decide|ofReduceBool|implemented_by|@\[extern)")),
    ("local notation or macro redefinition", re.compile(r"(notation|macro_rules|\bmacro\b|\bsyntax\b|\belab\b|infix)")),
]
STATEMENT_CHECK = re.compile(
    r"((statement|header|signature|theorem_text|formal_stmt|stmt)\w*\s+(not\s+)?in\s+\w+|"
    r"\.startswith\(\s*[\w.\[\]'\"]*(statement|header|signature|stmt)|"
    r"(statement|signature|stmt)\w*\s*[!=]=|[!=]=\s*\w*(statement|signature|stmt)|"
    r"check_statement|verify_statement|statement_match|same_statement|statement_unchanged|signature[ _]guard|"
    r"compare_statement|theorem_matches|comparator|SafeVerify|safe_verify)", re.I)
FROZEN_STATEMENT = re.compile(
    r"(\{\s*\w*(statement|header|stmt|signature)\w*\s*\}[^\n'\"]{0,40}\{\s*\w*(proof|completion|response|answer|tactic|code|body)\w*"
    r"|\b\w*(statement|stmt|header)\w*\s*\+\s*\w*(proof|completion|response|tactic|body)\w*)", re.I)
SORRY_WORD_RE = re.compile(r"\\bsorry\\b")

# v0.5 (lean-source-cheat-tokens + formal-escape-hatch-additions, arxiv:2603.21065, doi:10.5281/zenodo.24346906,
# paranoia ban list, Vals kernel-check incident): more Lean hatches the source ban list must name.
# (escape hatch, what covers it, what makes it moot: "replay" = a clean kernel replay or statement
# comparator, "frozen" = the grader writes the statement itself so the policy cannot put commands before it,
# "axioms" = an axiom audit with an allowlist or a sorryAx check)
LEAN_HATCHES_V05 = [
    ("#exit (it stops the file before an appended audit runs)", re.compile(r"#exit"), ("replay",)),
    ("local or scoped instance, variable or priority rebinding", re.compile(r"(\binstance\b|\bvariable\b|scoped|priority)"),
     ("replay", "frozen")),
    ("unsafe or partial declarations", re.compile(r"(\bunsafe\b|\bpartial\b)"), ("replay", "axioms")),
    ("admit", re.compile(r"\badmit\b"), ("replay", "axioms")),
    ("#eval, run_cmd, #guard_msgs or addDecl commands (they can print a fake audit line or add unchecked declarations)",
     re.compile(r"(#eval|run_cmd|run_tac|guard_msgs|addDecl|modifyEnv|\belab\b|Lean\.Elab|#\[?\w*\s*command)"), ("replay",)),
]

# every Lean marker above contains one of these
LEAN_GATE = ("lake ", "Mathlib", "LeanTaskset", "lean_check", "kimina", "LeanServer", "lean_server", "lean_repl",
             "leanrepl", "repl.lean", "'lean'", '"lean"', "lean -", "lean {", "lean $", "lean /")
OTHER_CHECKERS = [
    # (language, substring gate, how the grader calls it, escape hatches: (name, covered by))
    ("Dafny", ("dafny", "Dafny", "DAFNY"), re.compile(r"(?i:dafny)(\s+(verify|/compile|build|run|translate)\b|['\"]|_verify)"),
     [("assume", re.compile(r"\bassume\b")), ("{:axiom}", re.compile(r":axiom")),
      ("{:verify false}", re.compile(r"verify\s+false")), ("{:extern}", re.compile(r":extern")),
      ("expect", re.compile(r"\bexpect\b"))]),
    ("Verus", ("verus",), re.compile(r"verus(['\"]|\s+[-\w/$])"),
     [("assume", re.compile(r"\bassume\b")), ("admit", re.compile(r"\badmit\b")),
      ("external_body", re.compile(r"external_body"))]),
    ("Coq", ("coqc", "coqtop", "rocq "), re.compile(r"(coqc|coqtop|rocq\s+(c|compile))\b"),
     [("Admitted", re.compile(r"\bAdmitted\b|Print\s+Assumptions")), ("admit", re.compile(r"\badmit\b|Print\s+Assumptions")),
      ("Axiom", re.compile(r"\bAxiom\b|Print\s+Assumptions"))]),
    ("Frama-C", ("frama-c", "framac", "FRAMAC", "Frama-C"), re.compile(r"((?i:frama-?c)[^\n]{0,80}-wp\b|['\"]-wp['\"])"),
     [("//@ admit (an assertion taken without proof)", re.compile(r"\badmit\b"))]),
    ("Isabelle", ("isabelle",), re.compile(r"isabelle\s+(build|process|jedit)\b"),
     [("sorry", re.compile(r"\bsorry\b")), ("oops", re.compile(r"\boops\b")),
      ("axiomatization", re.compile(r"axiomatization"))]),
]


_FORMAL_GATE = re.compile("|".join(re.escape(t) for t in LEAN_GATE + tuple(g for _, gs, _, _ in OTHER_CHECKERS for g in gs)))


def _formal_files(pkg: Package) -> list[str]:
    """Texts of grader side files that mention any proof checker (one literal-alternation scan per
    file, so large data heavy packages stay cheap)."""
    c = _cache(pkg)
    if "formal" not in c:
        texts = [pf.text for pf in list(pkg.py_all()) + _harbor_test_py(pkg)]
        texts += [t for p, t in pkg.files.items() if p.endswith((".sh", ".bash"))]
        c["formal"] = [t for t in texts if _FORMAL_GATE.search(t)]
    return c["formal"]


def _gated_text(pkg: Package, gate) -> str:
    """Grader side text of only the files that contain one of the gate substrings."""
    return "\n".join(t for t in _formal_files(pkg) if any(g in t for g in gate))


def _first_line(pkg: Package, pat: re.Pattern):
    pfs = list(pkg.py_all()) + _harbor_test_py(pkg)
    pfs.sort(key=lambda pf: bool(HELPER_PATH.search(pf.path)))  # the grader before scripts and tools
    for pf in pfs:
        m = pat.search(pf.text)
        if m:
            return ev(pf, line_of(pf.text, m.start()))
    for p, t in pkg.files.items():
        if p.endswith((".sh", ".bash")):
            m = pat.search(t)
            if m:
                return f"{p}:{line_of(t, m.start())}"
    return None


def _grader_literals(pkg: Package) -> str:
    """String literals of grader side Python plus shell text: where ban lists and audit commands live."""
    c = _cache(pkg)
    if "lits" in c:
        return c["lits"]
    out = []
    for pf in list(pkg.py_all()) + _harbor_test_py(pkg):
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 4000:
                out.append(n.value)
    out += [t for p, t in pkg.files.items() if p.endswith((".sh", ".bash"))]
    c["lits"] = "\n".join(out)
    return c["lits"]


@rule("formal-checker-escape-hatch", "hack", "high",
      "A Lean, Dafny, Verus, Frama-C, Coq or Isabelle grader accepts a proof whose audit misses a known "
      "escape hatch, so a proof of the wrong statement or of False can be paid.")
def r_formal_escape(pkg: Package):
    out = []
    text = _gated_text(pkg, LEAN_GATE)
    # Lean: only where lean-no-axiom-check is silent because some audit exists
    if text and _lean_used(text) and LEAN_PROOF.search(text) and LEAN_AUDIT.search(text) and not r_lean(pkg):
        lits = _grader_literals(pkg)
        audit = lits + "\n" + text
        missing = []
        allow = bool(LEAN_ALLOWLIST.search(audit))
        replay = bool(KERNEL_REPLAY.search(audit))
        for name, cover in LEAN_HATCHES:
            if name.startswith("set_option") and replay:
                continue
            if name.startswith("native_decide") and allow:
                continue  # ofReduceBool is a non-standard axiom: an allowlist audit catches it
            if name.startswith("local notation") and (STATEMENT_CHECK.search(text) or FROZEN_STATEMENT.search(text)):
                continue
            if not cover.search(lits):
                missing.append(name)
        if not (STATEMENT_CHECK.search(text) or FROZEN_STATEMENT.search(text)):
            missing.append("a changed theorem statement (no comparison to the frozen statement)")
        if SORRY_WORD_RE.search(lits) and "sorryAx" not in audit and not allow:
            missing.append("sorryAx (the \\bsorry\\b pattern does not match it)")
        # v0.5: more hatches, each only where nothing else in the grader makes it moot
        moot = set()
        if replay:
            moot.add("replay")
        if FROZEN_STATEMENT.search(text) or re.search(r"\.startswith\(\s*[\w.\[\]'\"]*(statement|header|signature|stmt)", text, re.I):
            moot.add("frozen")  # the statement comes first, so nothing the policy writes can sit before it
        if allow or "sorryAx" in audit:
            moot.add("axioms")
        for name, cover, by in LEAN_HATCHES_V05:
            if moot & set(by):
                continue
            if not cover.search(lits):
                missing.append(name)
        if missing:
            where = [x for x in (_first_line(pkg, LEAN_AUDIT), _first_line(pkg, LEAN_CMD)) if x]
            out.append(("Lean", missing, where))
    for lang, gate, use, hatches in OTHER_CHECKERS:
        text = _gated_text(pkg, gate)
        m = use.search(text) if text else None
        if not m:
            continue
        lits = _grader_literals(pkg)
        missing = [h for h, cover in hatches if not cover.search(lits)]
        if lang == "Dafny" and not re.search(r"(ensures|requires)\w*\s*(not\s+)?(in|==|!=)|spec\w*\s*[!=]=|"
                                             r"compare_spec|spec_unchanged|frozen_spec", text, re.I):
            missing.append("a weakened ensures/requires clause (no comparison to the frozen spec)")
        # v0.5 (dafny-ensures-binding-and-opt-out, arxiv:2611.05590): `decreases *` opts a method out of the
        # termination proof, so a loop that never ends verifies any postcondition
        if lang == "Dafny" and not re.search(r"decreases\s*\*|decreases\\s|['\"]decreases", lits):
            missing.append("decreases * (a method that never ends verifies any postcondition)")
        if missing:
            out.append((lang, missing, [x for x in (_first_line(pkg, use),) if x]))
    if not out:
        return []
    lang, missing, where = out[0]
    shown = ", ".join(missing[:4])
    msg = (f"The {lang} grader's audit does not cover {shown}, so a proof that uses one of them can be accepted"
           + (f" (also {', '.join(o[0] for o in out[1:])})." if len(out) > 1 else "."))
    evid = [e for _, _, w in out for e in w] or ["(package)"]
    return [F("formal-checker-escape-hatch", msg, evid, confidence="review")]



# ---------------------------------------------------------------------------
# 10. judge-forgeable-delimiter (extends judge-prompt-undelimited)
# Evidence row 10 (section 9.2), inspect_evals BEST_PRACTICES.md (a forged [END DATA] line
# flipped the judge about 90% of the time), doi:10.18653/2026.emnlp-main.2496 (policies learned
# fake end markers in 100 of 100 inspected samples).
# Reported as hygiene like judge-prompt-undelimited: the 90 env audit rated every judge
# injection surface it saw as low (plausible but untested), so a forgeable fence alone does not meet the bar for a counted finding.
# ---------------------------------------------------------------------------

RESPONSE_SLOT = re.compile(r"\{(response|completion|model_answer|model_response|output|prediction|predicted_answer|"
                           r"student_answer|generated|generation|submission|candidate|candidate_answer|agent_answer|"
                           r"model_output|agent_response|assistant_response|text|reply|poem|essay|trajectory|"
                           r"final_answer|actions|advice)\}")
RESPONSE_NAMES = re.compile(r"^(response|completion|model_answer|model_response|output|prediction|predicted_answer|"
                            r"student_answer|generated|generation|submission|candidate|candidate_answer|agent_answer|"
                            r"model_output|agent_response|assistant_response|reply|poem|essay|response_text|"
                            r"completion_text|answer_text|final_answer|code_text|code)$")
JUDGE_CONTEXT = re.compile(r"(judge|grader|grade|grading|evaluat|rubric|assess|rate |rating|score|correct|verdict)", re.I)
_TQ = '"' * 3
_SQ = "'" * 3
# (name, opener at the end of the text before the slot, closer at the start of the text after it)
FENCES = [
    ("XML tag", re.compile(r"<([A-Za-z_][\w\-' ]{0,40})>\s*$"), None),
    ("code fence", re.compile(r"(```|~~~)[\w-]*\s*$"), re.compile(r"^\s*(```|~~~)")),
    ("triple quote", re.compile("(" + _TQ + "|" + _SQ + r")\s*$"), re.compile(r"^\s*(" + _TQ + "|" + _SQ + ")")),
    ("BEGIN/END marker", re.compile(r"\[\s*(BEGIN|START)[^\]\n]{0,40}\]\s*$", re.I),
     re.compile(r"^\s*\[\s*(END|STOP)[^\]\n]{0,40}\]", re.I)),
    ("rule line", re.compile(r"(-{3,}|={3,}|#{3,}|\*{3,}|<{3,}|>{3,})[^\n]{0,40}\s*$"),
     re.compile(r"^\s*(-{3,}|={3,}|#{3,}|\*{3,}|<{3,}|>{3,})")),
]
ESCAPING = re.compile(r"(html\.escape\(|saxutils\.escape|xml_escape|escape_xml|quoteattr\(|uuid\.uuid4|uuid4\(\)|"
                      r"secrets\.token|token_hex\(|token_urlsafe\(|\bnonce\s*=|\{nonce\}|\bboundary\s*=|\{boundary\})")


def _fence(before: str, after: str):
    """-> fence name if the slot sits inside a fixed opener and closer pair, else None."""
    for name, opener, closer in FENCES:
        m = opener.search(before)
        if not m:
            continue
        if name == "XML tag":
            raw = m.group(1).strip()
            tag = re.escape(raw)
            base = re.escape(re.sub(r"\s*(begin|start)\s*$", "", raw, flags=re.I))
            if re.match(r"^\s*</\s*" + tag + r"\s*>", after) or re.match(r"^\s*<\s*" + base + r"\s*(end|stop)\s*>", after, re.I):
                return f"<{raw}> tag"
            continue
        if closer.search(after):
            return name
    return None


def _inside_fence(before: str, after: str):
    """The slot sits inside an open code fence or triple quote block (an odd count before it and a
    closer after it), with field labels in between: ```\nQuestion: ..\nPredicted answer: {response}\n```."""
    for name, tok in (("code fence", "```"), ("triple quote", _TQ), ("triple quote", _SQ)):
        if before.count(tok) % 2 == 1 and tok in after:
            return name
    return None


def _templates(pkg: Package):
    """(PyFile, label, line, text, slot match) for judge prompt templates: module level string
    constants and f-strings built inside grader functions."""
    out = []
    for pf in list(pkg.py_all()) + _harbor_test_py(pkg):
        if pf.tree is None:
            continue
        for name, t in pf.consts.items():
            m = RESPONSE_SLOT.search(t)
            if m and JUDGE_CONTEXT.search(name + " " + t[:800]):
                out.append((pf, name, None, t, m))
    for pf, fi in _grader_funcs(pkg):
        if "f\"" not in fi.src and "f'" not in fi.src:
            continue
        for n in _own_nodes(fi.node):
            if not isinstance(n, ast.JoinedStr):
                continue
            parts, slot_at = [], None
            for v in n.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    nm = _nm(v.value) or (v.value.attr if isinstance(v.value, ast.Attribute) else "")
                    if slot_at is None and nm and RESPONSE_NAMES.match(nm):
                        slot_at = sum(len(x) for x in parts)
                        parts.append("{response}")
                    else:
                        parts.append("{}")
            if slot_at is None:
                continue
            t = "".join(parts)
            if len(t) < 80 or not JUDGE_CONTEXT.search(t[:800] + " " + fi.name):
                continue
            m = RESPONSE_SLOT.search(t, slot_at)
            if m:
                out.append((pf, fi.name, n.lineno, t, m))
    return out


@rule("judge-forgeable-delimiter", "hygiene", "medium",
      "A judge prompt fences the response with a fixed marker and no nonce or escaping, so a response "
      "can close the fence itself and write text the judge reads as instructions.")
def r_judge_forgeable(pkg: Package):
    tpls = _templates(pkg)
    if not tpls:
        return []
    already = set()
    for f in r_judge_undelimited(pkg):
        already |= set(f.evidence)
    text = _strings(pkg)
    hits, kinds = [], []
    for pf, label, lineno, t, m in tpls:
        if f"{pf.path}:{label}" in already:
            continue
        before, after = t[:m.start()], t[m.end():]
        fence = _fence(before[-200:], after[:200]) or _inside_fence(before, after)
        if not fence or not after.strip():
            continue  # no fence, or nothing after it for a forged closer to break out into
        closer = after.strip()[:6]
        if ESCAPING.search(pf.text) or (closer and re.search(r"\.replace\(\s*['\"]" + re.escape(closer), text)):
            continue  # the closer is stripped from the response, or a nonce or escaping is in use
        line = lineno or next((i for i, ln in enumerate(pf.lines, 1)
                               if re.match(r"\s*" + re.escape(label) + r"\s*[:=]", ln)), 1)
        hits.append(ev(pf, line))
        kinds.append(fence)
    if not hits:
        return []
    return [F("judge-forgeable-delimiter",
              f"The judge prompt wraps the response in a fixed {kinds[0]} with no nonce or escaping, so a response "
              "that writes the closing marker itself can append text the judge reads as its own instructions.",
              hits, confidence="review")]


# ---------------------------------------------------------------------------
# 16. lenient-answer-match
# Evidence row 16 (section 9.2), CS:4f5068ef (math-verify accepted all 10 "gold value in prose"
# cases), 2611.format-drift (a "reference number anywhere" loophole), 2609.01354 (tolerance).
# Only the substring form ships: the relative tolerance, default Math-Verify and few option MCQ
# forms matched clean envs in the labeled set (see the internal notes).
# ---------------------------------------------------------------------------

GOLD_NAME = re.compile(r"^(?!.*(keyword|kw|term|concept|phrase|word|vocab|token))"
                       r"(answer|answers|gold\w*|ground_truth|gt|expected\w*|target|targets|target_answer|target_text|"
                       r"reference|references|reference_answer|ref|ref_answer|"
                       r"correct_answer|correct_answers|correct_option|correct_letter|correct_choice|solution|label|truth|"
                       r"target_flag|flag|expected_function|answer_key|aliases|answer_aliases|acceptable\w*|valid_answers|"
                       r"possible_answers|accepted_answers)$", re.I)
FULLTEXT_NAME = re.compile(r"^(completion|response|text|content|output|full_text|resp|reply|generated|model_output|"
                           r"completion_text|response_text|last_message|msg|message|final_response|raw|raw_text|"
                           r"full_response|assistant_text|model_response|output_text)$", re.I)
NORMALIZERS = {"lower", "strip", "upper", "casefold", "lstrip", "rstrip", "replace"}


def _plain(e):
    """Name behind light normalisation: str(x).lower().strip(), x["content"], x.content."""
    while True:
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr in NORMALIZERS:
            e = e.func.value
        elif isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "str" and e.args:
            e = e.args[0]
        else:
            break
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        return e.attr
    if isinstance(e, ast.Subscript) and isinstance(e.slice, ast.Constant) and isinstance(e.slice.value, str):
        return e.slice.value
    return None


def _plain_node(e):
    """The expression behind light normalisation (str(x).lower().strip() -> x)."""
    while True:
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr in NORMALIZERS:
            e = e.func.value
        elif isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "str" and e.args:
            e = e.args[0]
        else:
            return e


HTTP_VERB = {"get", "post", "put", "patch", "delete", "head", "request", "urlopen", "fetch", "aget", "apost"}
HTTP_RECV = re.compile(r"(requests|httpx|session|client|http|aiohttp|urllib|api)", re.I)
PROC_CALL = {"run", "check_output", "communicate", "exec_run", "execute", "exec_command", "run_command", "Popen",
             "popen", "arun", "aexec", "run_code", "execute_code", "run_script", "run_tests"}
PROC_RECV = re.compile(r"(subprocess|sandbox|container|runtime|docker|proc|shell|process|session|box)", re.I)
OUTPUT_ATTR = {"text", "content", "body", "stdout", "stderr", "output", "data", "json"}


def _external_value(e, fnode, depth=0) -> bool:
    """e is an HTTP response body or the output of a process (code under test), read straight from
    the call or through a variable assigned from it in the function: r = requests.get(..); r.text."""
    e = _plain_node(e)
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr in OUTPUT_ATTR:
        e = e.func.value  # r.json()
    if isinstance(e, ast.Attribute) and e.attr in OUTPUT_ATTR:
        e = e.value
    if isinstance(e, ast.Await):
        e = e.value
    if isinstance(e, ast.Call):
        c = _callee(e)
        recv = ast.unparse(e.func.value) if isinstance(e.func, ast.Attribute) else ""
        args = " ".join(ast.unparse(a) for a in list(e.args[:1]) + [k.value for k in e.keywords if k.arg == "url"])
        if c in HTTP_VERB and (HTTP_RECV.search(recv) or re.search(r"(http|localhost|url)", args, re.I)):
            return True
        if c in PROC_CALL and (PROC_RECV.search(recv) or c in ("check_output", "communicate", "exec_run", "Popen")):
            return True
        return False
    if isinstance(e, ast.Name) and depth < 2:
        for n in _own_nodes(fnode):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, ast.Name) and t.id == e.id for t in tgts) and _external_value(n.value, fnode, depth + 1):
                    return True
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for it in n.items:
                    if isinstance(it.optional_vars, ast.Name) and it.optional_vars.id == e.id and \
                            _external_value(it.context_expr, fnode, depth + 1):
                        return True
    return False


def _literal_source(name: str, pf: PyFile, fi: FuncInfo, loops: dict) -> bool:
    """The "gold" is fixed text written in the code (a module constant, a literal list it loops
    over, a literal assigned in the function), not a per task answer."""
    if name in loops:
        it = loops[name]
        if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
            return True
        nm = _plain(it)
        return bool(nm and (nm.isupper() or _literal_source(nm, pf, fi, {})))
    if name.isupper() and len(name) > 1:
        return True
    for n in _own_nodes(fi.node):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return isinstance(n.value, (ast.Constant, ast.List, ast.Tuple, ast.Set))
    if pf.tree is not None and name not in fi.params:
        for st in pf.tree.body:
            if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in st.targets):
                return isinstance(st.value, (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict))
    return False


IDENTIFIER = re.compile(r"^(?=[A-Za-z_]\w*$)(?=.*(_|[a-z][A-Z]|[A-Za-z]\d))")
ROW_GOLD_NAME = re.compile(r"^(answer|answers|gold\w*|expected|expected_answer|target|solution|label|ground_truth|"
                           r"reference|reference_answer)$")
ROW_GOLD_KEYS = {"answer", "gold", "gold_answer", "expected", "expected_answer", "target", "solution", "label",
                 "ground_truth", "reference"}


def _gold_shown_in_task(pkg: Package) -> bool:
    """Most task rows the package ships (dict literals in the code, or .jsonl rows) have a gold
    answer that is a code identifier also written in another field of the same row, such as the
    name of the function the task shows. Then the answer is given, not hidden."""
    c = _cache(pkg)
    if "gold_shown" in c:
        return c["gold_shown"]
    rows = []
    for pf in pkg.py_all():
        if pf.tree is None or not re.search(r"['\"](answer|gold|expected|target|solution|label)", pf.text):
            continue
        for n in ast.walk(pf.tree):
            if not isinstance(n, ast.Dict):
                continue
            gold, other = None, []
            for k, v in zip(n.keys, n.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                if k.value in ROW_GOLD_KEYS and isinstance(v, ast.Constant) and isinstance(v.value, str):
                    gold = v.value
                else:
                    t = v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else _const_text(v)
                    if t:
                        other.append(t)
            if gold is not None and other:
                rows.append((gold, other))
    if len(rows) < 3:
        for p, t in pkg.text_files(".jsonl"):
            for r in _iter_jsonl_rows(t, 200):
                if not isinstance(r, dict):
                    continue
                gold = next((r[k] for k in ROW_GOLD_KEYS if isinstance(r.get(k), str)), None)
                other = [json.dumps(v) if not isinstance(v, str) else v for k, v in r.items() if k not in ROW_GOLD_KEYS]
                if gold is not None and other:
                    rows.append((gold, other))
            if len(rows) >= 200:
                break
    shown = sum(1 for g, other in rows if IDENTIFIER.match(g) and any(g in t for t in other))
    c["gold_shown"] = len(rows) >= 3 and shown >= 0.5 * len(rows)
    return c["gold_shown"]


def _flow(node, parent, fi, depth=0):
    """Where a containment test leads: 'pos' when a match raises the reward, 'neg' when it lowers
    it, fails the check or raises, else None."""
    sign, cur = True, node
    p = parent.get(cur)
    while p is not None:
        if isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not):
            sign = not sign
        elif isinstance(p, (ast.BoolOp, ast.Compare, ast.BinOp, ast.Tuple, ast.Dict, ast.FormattedValue)) or (
                isinstance(p, ast.Call) and _callee(p) in ("any", "all", "sum", "len", "float", "int", "bool", "min", "max")):
            pass
        elif isinstance(p, ast.comprehension) or (isinstance(p, (ast.GeneratorExp, ast.ListComp, ast.SetComp))):
            pass  # a filter: matches are what the comprehension collects
        elif isinstance(p, ast.IfExp):
            if cur is p.test:
                a, b = _num(p.body), _num(p.orelse)
                if a is not None and b is not None and a != b:
                    return "pos" if (a > b) == sign else "neg"
                return None
        elif isinstance(p, ast.If):
            if cur is not p.test:
                return None
            return _branch_effect(p.body if sign else p.orelse)
        elif isinstance(p, ast.Return):
            return "neg" if NEGATIVE_PREDICATE.search(fi.name) else ("pos" if sign else None)
        elif isinstance(p, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tg = p.targets if isinstance(p, ast.Assign) else [p.target]
            names = [t.id for t in tg if isinstance(t, ast.Name)]
            if not names:
                return None
            if NEGATIVE_PREDICATE.search(names[0]):
                return "neg"
            if isinstance(p, ast.AugAssign):
                return "pos" if sign and isinstance(p.op, ast.Add) else None
            if depth >= 2:
                return None
            # follow the variable to where it is used
            effects = set()
            for n in _own_nodes(fi.node):
                if isinstance(n, ast.Name) and n.id == names[0] and isinstance(n.ctx, ast.Load):
                    e = _flow(n, parent, fi, depth + 1)
                    if e:
                        effects.add(e if sign else ("neg" if e == "pos" else "pos"))
            return effects.pop() if len(effects) == 1 else None
        else:
            return None
        cur, p = p, parent.get(p)
    return None


def _branch_effect(stmts):
    """'pos' when the branch pays or counts a hit, 'neg' when it fails, penalises or raises."""
    pos = neg = False
    for s_ in stmts:
        if _is_logging(s_):
            continue
        for x in ast.walk(s_):
            if isinstance(x, ast.Raise):
                neg = True
            elif isinstance(x, ast.Return) and x.value is not None:
                v = x.value.elts[0] if isinstance(x.value, ast.Tuple) and x.value.elts else x.value
                k = _num(v)
                if k is None and isinstance(v, ast.Constant) and v.value is None:
                    k = 0.0
                if k is None:
                    pos = True  # a computed or named value: retrieval_reward, score
                elif k > 0:
                    pos = True
                else:
                    neg = True
            elif isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name):
                if NEGATIVE_PREDICATE.search(x.target.id) or isinstance(x.op, ast.Sub):
                    neg = True
                elif isinstance(x.op, ast.Add):
                    pos = True
            elif isinstance(x, ast.Assign):
                k = _num(x.value)
                for t in x.targets:
                    nm = t.id if isinstance(t, ast.Name) else (t.slice.value if isinstance(t, ast.Subscript) and
                                                               isinstance(t.slice, ast.Constant) else "")
                    if not isinstance(nm, str) or not nm:
                        continue
                    if NEGATIVE_PREDICATE.search(nm):
                        neg = neg or (k is None or k > 0)
                    elif k is not None:
                        if k > 0:
                            pos = True
                        else:
                            neg = True  # passed = False, score = 0
            elif isinstance(x, ast.Call) and _callee(x) in ("append", "add", "extend"):
                pos = True
        if isinstance(s_, ast.Continue) or isinstance(s_, ast.Break):
            pass
    if pos and not neg:
        return "pos"
    if neg and not pos:
        return "neg"
    return None


def _raw_message_list(pkg: Package, fi: FuncInfo, name: str) -> bool:
    """`name` is the reward's own chat completion parameter, used as is: never reassigned (no [-1]["content"],
    str(...) or join first), never checked for str, and the package does not use plain-text completions."""
    if name not in ("completion", "messages") or name not in fi.params:
        return False
    for n in _own_nodes(fi.node):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.NamedExpr)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(x, ast.Name) and x.id == name for t in tgts for x in ast.walk(t)):
                return False
        if isinstance(n, ast.Call) and _callee(n) == "isinstance" and n.args and isinstance(n.args[0], ast.Name) \
                and n.args[0].id == name:
            return False
    c = _cache(pkg)
    if "text_completions" not in c:
        c["text_completions"] = any(re.search(r"message_type\s*=\s*['\"]completion['\"]", pf.text) for pf in pkg.py_all())
    return not c["text_completions"]


@rule("lenient-answer-match", "hack", "medium",
      "A reward pays when the gold answer appears anywhere in the full response, so a response that "
      "lists every candidate answer is paid like a correct one.")
def r_lenient_match(pkg: Package):
    hits = []
    for pf, fi in _grader_funcs(pkg):
        if HELPER_PATH.search(pf.path) or (" in " not in fi.src and ".find(" not in fi.src and
                                            ".count(" not in fi.src and "escape(" not in fi.src):
            continue
        asg = parent = None
        loop_gold, loop_iter = {}, {}
        for n in _own_nodes(fi.node):
            gens = []
            if isinstance(n, (ast.For, ast.AsyncFor)):
                gens = [(n.target, n.iter)]
            elif isinstance(n, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                gens = [(g.target, g.iter) for g in n.generators]
            for tgt, it in gens:
                if isinstance(tgt, ast.Name):
                    loop_iter[tgt.id] = it
                    src = _plain(it)
                    if src and GOLD_NAME.match(src):
                        loop_gold[tgt.id] = src
        for n in _own_nodes(fi.node):
            pair = None
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.In):
                pair = (n.left, n.comparators[0])
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("find", "count", "index") \
                    and len(n.args) == 1:
                pair = (n.args[0], n.func.value)
            elif isinstance(n, ast.Call) and _callee(n) in ("search", "findall") and len(n.args) >= 2:
                esc = [c for c in ast.walk(n.args[0]) if isinstance(c, ast.Call) and _callee(c) == "escape" and c.args]
                if esc:  # re.search(re.escape(gold), text) / re.search(rf"\b{re.escape(gold)}\b", text)
                    pair = (esc[0].args[0], n.args[1])
            if not pair:
                continue
            gold, hay = _plain(pair[0]), _plain(pair[1])
            if not gold or not hay:
                continue
            gname = gold
            if gold in loop_gold:  # for a in answers: if a in text
                gold = loop_gold[gold]
            if gold == hay or not GOLD_NAME.match(gold) or not FULLTEXT_NAME.match(hay):
                continue
            # v0.4: `x in completion` on the raw message list tests list membership and is never true
            if isinstance(pair[1], ast.Name) and _raw_message_list(pkg, fi, pair[1].id):
                continue
            if asg is None:
                own = list(_own_nodes(fi.node))
                asg = _assignments(own)
                parent = {}
                for x in own:
                    for ch in ast.iter_child_nodes(x):
                        parent[ch] = x
                for ch in ast.iter_child_nodes(fi.node):
                    parent[ch] = fi.node
            # a haystack that is an extracted final answer is anchored enough
            if any(PARSE_CALL.match(c) or re.search(r"(parse|extract|answer|boxed|final)", c, re.I)
                   for c, _ in asg.get(hay, [])):
                continue
            # a "gold" that was parsed out of the reply is the model's answer, not the reference
            if any(PARSE_CALL.match(c) or re.search(r"(parse|extract)", c, re.I) for c, _ in asg.get(gold, [])):
                continue
            # (3) a fixed marker written in the code is a keyword probe, not a hidden answer
            if _literal_source(gname, pf, fi, loop_iter) or (gname != gold and _literal_source(gold, pf, fi, {})):
                continue
            # (4) an HTTP response body or the output of the code under test is a functional test
            if _external_value(pair[1], fi.node):
                continue
            # (1) the match must raise the reward (not fail the check, penalise or stop the run)
            if _flow(n, parent, fi) != "pos":
                continue
            hits.append((pf, fi, n.lineno, gold))
    if not hits:
        return []
    kept = []
    zero = {}
    for pf, fi, ln, gold in hits:
        # (2) a function in the rubric only at weight 0 (logged) or below 0 (a penalty) pays nothing
        key = (pf.path, fi.start)
        if key not in zero:
            zero[key] = _payout_weight(pkg, fi)[0]
        if zero[key]:
            continue
        # (3) the row's gold is an identifier the task itself shows, so there is nothing to guess
        if ROW_GOLD_NAME.match(gold) and _gold_shown_in_task(pkg):
            continue
        kept.append((pf, fi, ln))
    if not kept:
        return []
    pf, fi, line = kept[0]
    return [F("lenient-answer-match",
              f"{fi.name} checks whether the gold answer occurs anywhere in the response, so listing several "
              "candidate answers is paid like answering.", [ev(p, ln) for p, _, ln in kept], confidence="review")]


# ---------------------------------------------------------------------------
# v0.5 shared helpers
# ---------------------------------------------------------------------------

def _is_paid(pkg: Package, fi: FuncInfo) -> bool:
    """False when the function is in a rubric only at weight 0 or below (logged, or a penalty)."""
    return not _payout_weight(pkg, fi)[0]


def _parents(fnode) -> dict:
    parent = {}
    for x in ast.walk(fnode):
        for ch in ast.iter_child_nodes(x):
            parent[ch] = x
    return parent


def _returned_value(node, parent, returned: set[str]) -> bool:
    """The expression is returned, or assigned to a name that is returned (through light wrapping)."""
    p = parent.get(node)
    while isinstance(p, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.BoolOp)) or (
            isinstance(p, ast.Call) and _callee(p) in ("float", "int", "min", "max", "round", "abs", "bool")):
        p = parent.get(p)
    if isinstance(p, ast.Return):
        return True
    if isinstance(p, (ast.Assign, ast.AnnAssign)):
        tg = p.targets if isinstance(p, ast.Assign) else [p.target]
        return any(isinstance(t, ast.Name) and t.id in returned for t in tg) or any(
            isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)
            and re.search(r"^(reward|score)$", t.slice.value) for t in tg)
    return False


# ---------------------------------------------------------------------------
# v0.5 grader-failure-scored-zero (+ infra-error-scored-as-zero)
# arxiv:2611.24321 (56 of 88 zero runs carried no validity flag), arxiv:2606.30931 (parser failure up to
# 33% on small judges), arxiv:2512.16144 (INTELLECT-3 masks rollouts on sandbox failure).
# The other half of grader-fails-open: a judge call that raises, or a judge reply with no readable
# verdict, scores 0 for the policy. Correct work is then scored as wrong whenever the judge fails.
# Built as its own id because it is a "broken" defect, not a "hack": reporting it under
# grader-fails-open would say the opposite of that rule's description.
# Model side failures (no answer, code that does not compile) are left out: 0 is the right score there.
# ---------------------------------------------------------------------------

LLM_NAME = re.compile(r"(judge|grade_with|grading|verdict|critic|call_llm|ask_model|query_model|query_llm|llm_\w*call|"
                      r"call_model|_llm$|^llm)", re.I)
LLM_VERB = {"create", "acreate", "chat", "achat", "generate", "agenerate", "invoke", "ainvoke", "complete", "acomplete",
            "completion", "acompletion", "parse", "aparse", "send_message", "generate_content", "generate_content_async"}
LLM_RECV = re.compile(r"(client|judge|openai|anthropic|llm|completions|messages|grader|gemini|genai|chat)", re.I)
INFRA_EXC = re.compile(r"(APIError|APIConnection|APIStatus|APITimeout|RateLimit|ConnectionError|ConnectError|HTTPError|"
                       r"HTTPStatus|ServiceUnavailable|Sandbox\w*|ServerError|InternalServerError|openai\.|httpx\.|"
                       r"aiohttp\.|requests\.|anthropic\.)")
MASK_MARK = re.compile(r"(state\[\s*['\"]\w*(error|mask|invalid|skip|fail|unscored|exclude|valid)\w*['\"]\s*\]\s*=|"
                       r"\bmask\w*\s*=|is_valid\s*=|['\"](judge_error|grader_error|error)['\"]\s*:)", re.I)


def _llm_call(c: ast.Call) -> bool:
    name = _callee(c)
    if not name or PARSE_CALL.match(name) or name in ("format", "join", "append", "get", "replace"):
        return False
    if LLM_NAME.search(name):
        return True
    if name in LLM_VERB and isinstance(c.func, ast.Attribute):
        return bool(LLM_RECV.search(_unparse_safe(c.func.value)))
    return False


def _zero_return(stmt):
    """Value of `return 0` / `return 0.0, msg` / `return False` / `return {"score": 0}` (0 or below), else None."""
    if not isinstance(stmt, ast.Return) or stmt.value is None:
        return None
    v = stmt.value
    if isinstance(v, ast.Tuple) and v.elts:
        v = v.elts[0]
    if isinstance(v, ast.Dict):
        v = next((val for k, val in zip(v.keys, v.values) if isinstance(k, ast.Constant) and isinstance(k.value, str)
                  and re.search(r"^(reward|score)$", k.value)), None)
        if v is None:
            return None
    x = _num(v)
    return x if x is not None and x <= 0 else None


def _zero_on_failure_hits(pf: PyFile, fi: FuncInfo):
    """-> [(line, value, cause)] where a judge or infra failure scores 0 or below."""
    if FAILOPEN_SKIP_FN.search(fi.name) or FORMAT_FN.search(fi.name):
        return []
    if re.search(rf"exit\(\s*(await\s+)?{re.escape(fi.name)}\(", pf.text):
        return []
    if MASK_MARK.search(fi.src):
        return []  # the function flags the failure next to the score
    own = list(_own_nodes(fi.node))
    # the success path can pay something above 0
    if not any(isinstance(n, ast.Return) and n.value is not None and (_num(n.value) is None or _num(n.value) > 0)
               for n in own):
        return []
    hits = []
    for t in own:
        if not isinstance(t, ast.Try):
            continue
        llm = any(isinstance(x, ast.Call) and _llm_call(x) for s_ in t.body for x in ast.walk(s_))
        for h in t.handlers:
            tname = _unparse_safe(h.type) if h.type is not None else "Exception"
            if any(x in tname for x in SKIP_EXC) or MODEL_EXC.match(tname):
                continue
            if any(isinstance(x, ast.Raise) for s_ in h.body for x in ast.walk(s_)):
                continue
            infra = bool(INFRA_EXC.search(tname))
            if not (llm or infra):
                continue
            first = next((x for x in h.body if not _is_logging(x)), None)
            v = _zero_return(first) if first is not None else None
            if v is None:
                continue
            cause = f"the {_fault_word(t)} call fails" if llm else f"the grader hits {tname}"
            hits.append((first.lineno, v, cause))
    # a judge reply with no readable verdict: m = re.search(..., judge_reply) / if not m: return 0
    asg = None
    for n in own:
        if not isinstance(n, ast.If):
            continue
        var = _cond_var(n.test)
        if not var:
            continue
        if asg is None:
            asg = _assignments(own)
        if not any(PARSE_CALL.match(c) for c, _ in asg.get(var, [])):
            continue
        first = next((x for x in n.body if not _is_logging(x)), None)
        v = _zero_return(first) if first is not None else None
        if v is None or not _judge_reply_var(var, own, asg):
            continue
        hits.append((first.lineno, v, "the judge reply has no readable verdict"))
    return hits


def _judge_reply_var(var: str, own, asg, depth=0) -> bool:
    """Stricter than _side: the value comes from an LLM call in this function (or is named like a verdict),
    not from the policy's completion text."""
    if GOLDISH.search(var):
        return False
    assigned = False
    for n in own:
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            if not any(isinstance(x, ast.Name) and x.id == var for t in tg for x in ast.walk(t)):
                continue
            assigned = True
            if any(isinstance(c, ast.Call) and _llm_call(c) for c in ast.walk(n.value)):
                return True
            if _names_in(n.value) & {"completion", "completions", "messages", "response", "parser"}:
                return False  # parsed out of the policy's reply, whatever it is called
    if not assigned and JUDGE_VAR.search(var) and not re.search(r"(completion|response)", var, re.I):
        return True
    if depth < 2:
        for _, args in asg.get(var, []):
            for a in args:
                if a != var and a not in _MODULE_NAMES and a not in ("completion", "prompt", "messages", "answer") \
                        and _judge_reply_var(a, own, asg, depth + 1):
                    return True
    return False


def _fault_word(t: ast.Try) -> str:
    """judge, sandbox or grader: what the failing call in a try body talks to (0.5.0 fix wave)."""
    src = "\n".join(_unparse_safe(x) for x in t.body)
    judge = re.search(r"(judge|grader_model|chat|completions|messages\.create|openai|anthropic|llm|\bmodel\b)", src, re.I)
    box = re.search(r"(sandbox|container|docker|subprocess|pip install|\bmodal\b|e2b|daytona|\bexec)", src, re.I)
    if box and not judge:
        return "sandbox"
    return "judge" if judge else "grader"


def _paid_reach(pkg: Package):
    """Names of functions that a rubric term with nonzero weight reaches through calls or references, or
    None when no rubric weight is in view (0.5.0 fix wave, check_1.md)."""
    c = _cache(pkg)
    if "paid_reach" in c:
        return c["paid_reach"]
    W = _reward_weights(pkg)
    seeds = {nm for nm, entries in W.items() if any(w > 0 for w, _ in entries)} if W else set()
    if not seeds:
        c["paid_reach"] = None
        return None
    by_name: dict[str, list] = {}
    for pf, fi in all_funcs(pkg):
        if _module_imported(pkg, pf):
            by_name.setdefault(fi.name, []).append(fi)
    # a rubric term held in a variable (judge_reward_func = make_judge_reward(judge)) reaches what builds it
    extra = set()
    for q in pkg.py_all():
        if q.tree is None:
            continue
        for n in ast.walk(q.tree):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in seeds and t.id not in by_name
                                                 for t in n.targets):
                extra |= {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
                extra |= {x.attr for x in ast.walk(n.value) if isinstance(x, ast.Attribute)}
    seeds = seeds | {x for x in extra if x in by_name}
    seen, todo = set(seeds), list(seeds)
    while todo:
        nm = todo.pop()
        for fi in by_name.get(nm, []):
            refs = set(called_names(fi))
            for n in ast.walk(fi.node):
                if isinstance(n, ast.Name):
                    refs.add(n.id)
                elif isinstance(n, ast.Attribute):
                    refs.add(n.attr)
            for r in refs:
                if r not in seen and r in by_name:
                    seen.add(r)
                    todo.append(r)
    c["paid_reach"] = seen
    return seen


def _module_imported(pkg: Package, pf: PyFile) -> bool:
    """False for a .py file that no other file imports and that builds no env or rubric itself."""
    c = _cache(pkg)
    key = ("imported", pf.path)
    if key in c:
        return c[key]
    stem = os.path.splitext(os.path.basename(pf.path))[0]
    ok = stem == "__init__" or len(pkg.py_all()) <= 1 or pf.path in {q.path for q in _harbor_test_py(pkg)} or \
        bool(re.search(r"(def load_environment|Rubric\(|add_reward_func|@vf\.reward|funcs\s*=)", pf.text))
    if not ok:
        pat = re.compile(rf"^\s*(from|import)\s[^\n]*\b{re.escape(stem)}\b", re.M)
        ok = any(q.path != pf.path and pat.search(q.text) for q in pkg.py_all())
    c[key] = ok
    return ok


@rule("grader-failure-scored-zero", "broken", "low",
      "A paid reward scores 0 when the judge call fails or its reply cannot be read, so grader faults score "
      "correct work as wrong instead of leaving the rollout unscored.")
def r_grader_failure_scored_zero(pkg: Package):
    found = []
    for pf, fi in _grader_funcs(pkg):
        if HELPER_PATH.search(pf.path) or ("except" not in fi.src and "if not" not in fi.src and "is None" not in fi.src):
            continue
        hits = _zero_on_failure_hits(pf, fi)
        if not hits or not _is_paid(pkg, fi):
            continue
        # fire only when a rubric term with nonzero weight reaches the function (0.5.0 fix wave)
        reach = _paid_reach(pkg)
        if not _module_imported(pkg, pf) or (reach is not None and fi.name not in reach):
            continue
        for line, v, cause in hits:
            found.append((pf, fi, line, v, cause))
    if not found:
        return []
    pf, fi, line, v, cause = found[0]
    shown = int(v) if float(v).is_integer() else v
    word = "judge" if "judge" in cause else ("sandbox" if "sandbox" in cause else "grader")
    return [F("grader-failure-scored-zero",
              f"{fi.name} returns {shown} when {cause}, so a {word} fault scores correct work as wrong instead of "
              "marking the rollout unscored" + (f" ({len(found)} such branches)." if len(found) > 1 else "."),
              [ev(p, ln) for p, _, ln, _, _ in found], confidence="review", severity="low")]


# ---------------------------------------------------------------------------
# v0.5 widening of grader-fails-open: denominator-drops-failed-items
# arxiv:2606.17454 (a parse bug cost 20 points in one harness), etb-harness (reader items).
# Inside one reward call, a loop over sub-items (parts of a question, test cases, claims) skips an item
# whose prediction is missing or whose check raised, and the score divides by the items kept. Leaving the
# hard items unanswered or broken then raises the score.
# ---------------------------------------------------------------------------

MEAN_CALL = {"mean", "fmean", "average", "nanmean"}


def _collector_adds(stmts):
    """Names a loop body adds to: L.append(..) / L.extend(..) -> ("list", L), n += .. -> ("count", n)."""
    out = set()
    for s_ in stmts:
        for x in ast.walk(s_):
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr in ("append", "extend") \
                    and isinstance(x.func.value, ast.Name):
                out.add(x.func.value.id)
            elif isinstance(x, ast.AugAssign) and isinstance(x.op, ast.Add) and isinstance(x.target, ast.Name):
                out.add(x.target.id)
    return out


def _reads_model_loop_item(t: ast.Try, loop, fi: FuncInfo) -> bool:
    """The try body reads the loop variable of a loop over the policy's output (for part in split(completion))."""
    tnames = _names_in(loop.target)
    params = {x for x in fi.params if not GOLDISH.search(x) and not CONFIG_PARAM.search(x)}
    iter_names = _names_in(loop.iter)
    if not (iter_names & params or any(MODEL_VAR.search(x) and not WEAK_MODEL_VAR.match(x) for x in iter_names)):
        return False
    return any(isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id in tnames
               for s_ in t.body for x in ast.walk(s_))


def _denominator_drop_hits(pf: PyFile, fi: FuncInfo):
    """-> [(line of the skip, cause, collector)]."""
    if "continue" not in fi.src or FAILOPEN_SKIP_FN.search(fi.name) or FORMAT_FN.search(fi.name):
        return []
    own = list(_own_nodes(fi.node))
    asg = _assignments(own)
    parent = _parents(fi.node)
    returned = _returned_names(own)
    # what the function divides by or averages, when that value is paid
    denoms = {}
    for n in own:
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            r = n.right
            if isinstance(r, ast.Call) and _callee(r) == "max" and r.args:
                r = r.args[0]
            nm = None
            if isinstance(r, ast.Call) and _callee(r) == "len" and r.args and isinstance(r.args[0], ast.Name):
                nm = r.args[0].id
            elif isinstance(r, ast.Name):
                nm = r.id
            if nm and _returned_value(n, parent, returned):
                denoms[nm] = n
        elif isinstance(n, ast.Call) and _callee(n) in MEAN_CALL and n.args and isinstance(n.args[0], ast.Name):
            if _returned_value(n, parent, returned):
                denoms[n.args[0].id] = n
    if not denoms:
        return []
    hits = []
    for loop in own:
        if not isinstance(loop, (ast.For, ast.AsyncFor)):
            continue
        it_names = _names_in(loop.iter)
        body_nodes = [x for s_ in loop.body for x in ast.walk(s_)]
        for n in body_nodes:
            skip, cause, guard_body = None, None, None
            if isinstance(n, ast.Try):
                for h in n.handlers:
                    tname = _unparse_safe(h.type) if h.type is not None else "Exception"
                    if any(x in tname for x in SKIP_EXC):
                        continue
                    if any(isinstance(x, ast.Continue) for s_ in h.body for x in ast.walk(s_)) and \
                            not any(isinstance(x, ast.Raise) for s_ in h.body for x in ast.walk(s_)):
                        if _graded_try(n, fi, asg) or _reads_model_loop_item(n, loop, fi):
                            llm = any(isinstance(x, ast.Call) and _llm_call(x) for s_ in n.body for x in ast.walk(s_))
                            skip, cause, guard_body = h, ("the judge call fails" if llm else "the check raises"), h.body
                            break
            elif isinstance(n, ast.If):
                var = _cond_var(n.test)
                first = next((x for x in n.body if not _is_logging(x)), None)
                if var and isinstance(first, ast.Continue) and not any(GOLDISH.search(x) for x in _names_in(n.test)):
                    side = _side(var, fi, asg)
                    if side in ("model", "judge"):
                        skip, cause, guard_body = n, ("the item has no prediction" if side == "model"
                                                      else "the judge gives no verdict"), n.body
            if skip is None:
                continue
            added_in_skip = _collector_adds(guard_body)
            line = getattr(skip, "lineno", n.lineno)
            later = [s_ for s_ in ast.walk(loop) if isinstance(s_, ast.stmt) and getattr(s_, "lineno", 0) > max(
                line, max((getattr(x, "end_lineno", 0) or 0) for x in [n]))]
            adds = _collector_adds(later) - added_in_skip
            for nm in adds:
                if nm in denoms and nm not in it_names and not GOLDISH.search(nm):
                    # the full count is not used anywhere: no len(items) / len(gold) next to it
                    if any(isinstance(x, ast.Call) and _callee(x) == "len" and x.args and
                           _names_in(x.args[0]) & it_names for x in ast.walk(denoms[nm])):
                        continue
                    hits.append((line, cause, nm))
                    break
    return hits


def _denominator_drop_finding(pkg: Package):
    found = []
    for pf, fi in _grader_funcs(pkg):
        if HELPER_PATH.search(pf.path) or "continue" not in fi.src:
            continue
        hits = _denominator_drop_hits(pf, fi)
        if hits and _is_paid(pkg, fi):
            found += [(pf, fi, ln, cause, nm) for ln, cause, nm in hits]
    if not found:
        return []
    pf, fi, line, cause, nm = found[0]
    return [F("grader-fails-open",
              f"{fi.name} skips an item when {cause} and divides by the items kept in {nm}, so leaving hard items "
              "unanswered or broken raises the score.", [ev(p, ln) for p, _, ln, _, _ in found],
              confidence="review", severity="medium", variant="v05")]


# ---------------------------------------------------------------------------
# v0.5 set-answer-recall-no-precision (extends lenient-answer-match)
# arxiv:2504.07615 (a length reward on OVDEval rose 21.68 to 31.01), arxiv:2611.19871 (reader items).
# The reward is |predicted & gold| / |gold| (or a count of gold items found in the predicted list over
# the gold count) with no precision term and no cap on how many items are listed. Listing every
# candidate then scores 1.0. lenient-answer-match covers the full-text substring form, so a haystack
# that is the whole response text is left to it.
# ---------------------------------------------------------------------------

SET_GOLD = re.compile(r"(gold|expected|target|truth|reference|^refs?$|^ref_|label|solution|correct|^gt$|^gt_|_gt$|"
                      r"^answers?$|^answer_(set|list|items)|ground)", re.I)
PRECISION_MARK = re.compile(r"(precision|\bf1\b|f1_|_f1|fbeta|f_score|jaccard|\biou\b|false_pos|\bfp\b|n_fp|\bextras?\b|extra_(items|preds|count)|spurious|"
                            r"penal|over_predict|hallucinat|wrong_items)", re.I)


def _base_name(e):
    """x / set(x) / list(x) / set(x.lower() ...) / info["answers"] -> name."""
    while isinstance(e, ast.Call) and _callee(e) in ("set", "list", "frozenset", "sorted", "tuple") and e.args:
        e = e.args[0]
    return _plain(e)


def _set_recall_hits(pkg: Package, pf: PyFile, fi: FuncInfo):
    own = list(_own_nodes(fi.node))
    parent = _parents(fi.node)
    returned = _returned_names(own)
    hits = []
    loop_over = {}  # counter -> (gold iter name, pred name) from: for g in gold: if g in pred: n += 1
    for n in own:
        if isinstance(n, (ast.For, ast.AsyncFor)) and isinstance(n.target, ast.Name):
            g = _base_name(n.iter)
            for x in ast.walk(n):
                if isinstance(x, ast.If) and isinstance(x.test, ast.Compare) and len(x.test.ops) == 1 \
                        and isinstance(x.test.ops[0], ast.In) and isinstance(x.test.left, ast.Name) \
                        and x.test.left.id == n.target.id:
                    p = _base_name(x.test.comparators[0])
                    for c in _collector_adds(x.body):
                        loop_over[c] = (g, p)
    for n in own:
        if not (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)):
            continue
        r = n.right
        if isinstance(r, ast.Call) and _callee(r) == "max" and r.args:
            r = r.args[0]
        if not (isinstance(r, ast.Call) and _callee(r) == "len" and r.args):
            continue
        gden = _base_name(r.args[0])
        if not gden or not SET_GOLD.search(gden):
            continue
        num, pred = n.left, None
        if isinstance(num, ast.Call) and _callee(num) == "len" and num.args:
            a = num.args[0]
            if isinstance(a, ast.BinOp) and isinstance(a.op, ast.BitAnd):
                sides = [_base_name(a.left), _base_name(a.right)]
            elif isinstance(a, ast.Call) and _callee(a) == "intersection" and isinstance(a.func, ast.Attribute) and a.args:
                sides = [_base_name(a.func.value), _base_name(a.args[0])]
            else:
                sides = []
            if gden in sides:
                rest = [x for x in sides if x != gden]
                pred = rest[0] if rest else None
        elif isinstance(num, ast.Call) and _callee(num) == "sum" and num.args and isinstance(num.args[0], ast.GeneratorExp):
            gen = num.args[0]
            comp = gen.generators[0]
            tests = [gen.elt] + list(comp.ifs)
            for t in tests:
                if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.In) \
                        and _base_name(comp.iter) == gden:
                    pred = _base_name(t.comparators[0])
        elif isinstance(num, ast.Name) and num.id in loop_over and loop_over[num.id][0] == gden:
            pred = loop_over[num.id][1]
        if not pred or pred == gden or SET_GOLD.search(pred) or FULLTEXT_NAME.match(pred):
            continue
        if not _returned_value(n, parent, returned):
            continue
        hits.append((n.lineno, pred, gden))
    if not hits:
        return []
    # a precision term, a union denominator or a cap on how many items count
    for ln, pred, gden in list(hits):
        bad = False
        if PRECISION_MARK.search(fi.src):
            bad = True
        for x in own:
            if bad:
                break
            if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Div):
                r = x.right.args[0] if isinstance(x.right, ast.Call) and _callee(x.right) == "max" and x.right.args else x.right
                if isinstance(r, ast.Call) and _callee(r) == "len" and r.args:
                    a = r.args[0]
                    if _base_name(a) == pred or (isinstance(a, ast.BinOp) and isinstance(a.op, ast.BitOr)) or \
                            (isinstance(a, ast.Call) and _callee(a) == "union"):
                        bad = True
            elif isinstance(x, ast.Compare) and isinstance(x.left, ast.Call) and _callee(x.left) == "len" \
                    and x.left.args and _base_name(x.left.args[0]) == pred:
                bad = True  # len(pred) > k: a size cap or a size check
            elif isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Slice) and _plain(x.value) == pred:
                bad = True  # pred[:k]
        if bad:
            hits.remove((ln, pred, gden))
    return hits


@rule("set-answer-recall-no-precision", "hack", "medium",
      "A reward pays the share of gold items found in the predicted list with no precision term or size cap, "
      "so listing every candidate scores like a correct answer.")
def r_set_recall(pkg: Package):
    found = []
    for pf, fi in _grader_funcs(pkg):
        if HELPER_PATH.search(pf.path) or "len(" not in fi.src or "/" not in fi.src:
            continue
        hits = _set_recall_hits(pkg, pf, fi)
        if hits and _is_paid(pkg, fi):
            found += [(pf, fi, ln, p, g) for ln, p, g in hits]
    if not found:
        return []
    pf, fi, line, pred, gold = found[0]
    wt = _weight_text(pkg, fi)
    return [F("set-answer-recall-no-precision",
              f"{fi.name} pays the share of {gold} found in {pred} with no precision term or cap, so listing every "
              "candidate answer earns full reward" + (f" on this term, which has {wt}." if wt else "."),
              [ev(p, ln) for p, _, ln, _, _ in found], confidence="review")]


def _weight_text(pkg: Package, fi: FuncInfo) -> str:
    """'weight 0.45 (30% of the reward)' for the rubric entry of a function or its direct caller, else ''
    (0.5.0 fix wave, check_4.md: state the weight of the flagged term when it can be read)."""
    W = _reward_weights(pkg)
    if not W:
        return ""
    _payout_weight(pkg, fi)  # fills the callers map
    names = [fi.name] + sorted(_cache(pkg).get("callers", {}).get(fi.name, set()))
    for nm in names:
        entries = [(w, sh) for w, sh in W.get(nm, []) if w > 0]
        if entries:
            w, sh = max(entries, key=lambda e: e[0])
            who = "" if nm == fi.name else f" through {nm}"
            return f"weight {w:g}{who}" + (f" ({sh:.0%} of the reward)" if sh is not None else "")
    return ""


# ---------------------------------------------------------------------------
# v0.5 numeric-tolerance-too-loose (extends lenient-answer-match)
# kd4ujtfxzm (60.1% of wrong answers absorbed by a tolerant matcher), arxiv:2609.01354 (a tolerance
# of 10^4 accepted wrong answers), reader items.
# Forms: a relative tolerance of 5% or more (1% or more when answers are integers), an absolute
# tolerance of 1 or more on integer answers, a sign blind compare (abs(pred) == abs(gold)), and a
# percent blind compare that also accepts pred * 100 or pred / 100.
# ---------------------------------------------------------------------------

NUM_GOLD = re.compile(r"(gold|expected|target|truth|^refs?$|^ref_|reference|label|solution|correct|^gt$|^gt_|_gt$|"
                      r"^answer$|^answers$|ground|true_val|true_ans)", re.I)
INTEGER_HINT = re.compile(r"(gsm8k|\baime\b|\bamc\b|countdown|integer answer|answer is an integer|whole number|"
                          r"int\(\s*\w*(answer|gold|expected|target)|\\boxed\{\d)", re.I)


def _sides_gold_model(a, b):
    """(gold side, model side) when one operand reads a gold-named value and the other does not."""
    na, nb = _names_in(a) - _MODULE_NAMES, _names_in(b) - _MODULE_NAMES
    ga = any(NUM_GOLD.search(x) for x in na)
    gb = any(NUM_GOLD.search(x) for x in nb)
    if ga and not gb and nb:
        return a, b
    if gb and not ga and na:
        return b, a
    return None


def _abs_diff(e):
    """abs(a - b) -> (a, b)."""
    if isinstance(e, ast.Call) and _callee(e) in ("abs", "fabs") and e.args and isinstance(e.args[0], ast.BinOp) \
            and isinstance(e.args[0].op, ast.Sub):
        return e.args[0].left, e.args[0].right
    return None


def _is_abs(e):
    return isinstance(e, ast.Call) and _callee(e) in ("abs", "fabs") and len(e.args) == 1


def _scaled_100(e):
    return isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Mult, ast.Div)) and (
        _num(e.right) in (100.0, 0.01) or _num(e.left) in (100.0, 0.01))


def _tolerance_hits(pf: PyFile, fi: FuncInfo, integer: bool):
    own = list(_own_nodes(fi.node))
    parent = _parents(fi.node)
    rel_bound = 0.01 if integer else 0.05
    hits = []

    def add(node, what):
        hits.append((node.lineno, what))

    for n in own:
        if isinstance(n, ast.Call) and _callee(n) in ("isclose", "allclose", "assert_allclose") and len(n.args) >= 2:
            pair = _sides_gold_model(n.args[0], n.args[1])
            if not pair:
                continue
            kw = {k.arg: _num(k.value) for k in n.keywords if k.arg}
            rel = kw.get("rel_tol", kw.get("rtol"))
            ab = kw.get("abs_tol", kw.get("atol"))
            if rel is not None and rel >= rel_bound and rel > 0.01:
                add(n, f"a relative tolerance of {rel:g}")
            elif ab is not None and ab >= 1 and integer:
                add(n, f"an absolute tolerance of {ab:g} on integer answers")
            elif _is_abs(n.args[0]) and _is_abs(n.args[1]):
                add(n, "a compare of absolute values, so -x matches x")
            elif _scaled_100(n.args[0]) or _scaled_100(n.args[1]):
                add(n, "a compare that also accepts the answer times 100 or divided by 100")
        elif isinstance(n, ast.Compare) and len(n.ops) == 1:
            op, left, right = n.ops[0], n.left, n.comparators[0]
            d = _abs_diff(left)
            if d and isinstance(op, (ast.Lt, ast.LtE)):
                if not _sides_gold_model(d[0], d[1]):
                    continue
                t = _num(right)
                if t is not None:
                    if integer and (t > 1 or (t == 1 and isinstance(op, ast.LtE))):
                        add(n, f"an absolute tolerance of {t:g} on integer answers")
                    continue
                if isinstance(right, ast.BinOp) and isinstance(right.op, ast.Mult):
                    r = _num(right.left) if _num(right.left) is not None else _num(right.right)
                    if r is not None and r >= rel_bound and r > 0.01:
                        add(n, f"a relative tolerance of {r:g}")
                continue
            if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div) and _abs_diff(left.left) \
                    and isinstance(op, (ast.Lt, ast.LtE)):
                dd = _abs_diff(left.left)
                r = _num(right)
                if r is not None and r >= rel_bound and r > 0.01 and _sides_gold_model(dd[0], dd[1]):
                    add(n, f"a relative tolerance of {r:g}")
                continue
            if isinstance(op, ast.Eq) and _is_abs(left) and _is_abs(right) and _sides_gold_model(left, right):
                add(n, "a compare of absolute values, so -x matches x")
                continue
            if isinstance(op, ast.Eq) and (_scaled_100(left) or _scaled_100(right)) and _sides_gold_model(left, right):
                p = parent.get(n)
                if isinstance(p, ast.BoolOp) and isinstance(p.op, ast.Or):
                    add(n, "a compare that also accepts the answer times 100 or divided by 100")
            if isinstance(op, ast.Eq) and _sides_gold_model(left, right):
                p = parent.get(n)
                neg = [x for x in (left, right) if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub)]
                if neg and isinstance(p, ast.BoolOp) and isinstance(p.op, ast.Or):
                    add(n, "a compare that also accepts the negated answer")
    # the matched value must reach the reward: the function pays on the compare (not a metric or log)
    if hits and not any(isinstance(x, ast.Return) and x.value is not None for x in own):
        return []
    # 0.5.0 fix wave (check_1.md): a partial credit branch that pays less than the exact match branch
    if hits and "==" in fi.src:
        top = _max_pay(own)
        keep = []
        for ln, what in hits:
            node = next((x for x in own if getattr(x, "lineno", None) == ln and isinstance(x, (ast.Compare, ast.Call))), None)
            pay = _branch_pay(node, parent) if node is not None else None
            if pay is not None and top is not None and pay < top:
                continue
            keep.append((ln, what))
        hits = keep
    return hits


def _pay_value(stmt):
    if isinstance(stmt, ast.Return) and stmt.value is not None:
        return _num(stmt.value)
    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and stmt.value is not None:
        return _num(stmt.value)
    return None


def _max_pay(own):
    vals = []
    for x in own:
        v = _pay_value(x)
        if v is not None:
            vals.append(v)
        elif isinstance(x, ast.IfExp):
            vals += [v for v in (_num(x.body), _num(x.orelse)) if v is not None]
    return max(vals) if vals else None


def _branch_pay(node, parent):
    """Largest constant the branch guarded by this compare pays, or None when it is not a constant."""
    p, child = parent.get(node), node
    while p is not None:
        if isinstance(p, ast.IfExp) and child is p.test:
            return _num(p.body)
        if isinstance(p, ast.If) and child is p.test:
            vals = [_pay_value(x) for x in p.body]
            vals = [v for v in vals if v is not None]
            return max(vals) if vals else None
        if isinstance(p, ast.stmt) and not isinstance(p, ast.If):
            return None
        p, child = parent.get(p), p
    return None


EXACT_TERM = re.compile(r"(exact|correct|accura|match)", re.I)


def _below_exact_term(pkg: Package, fi: FuncInfo) -> bool:
    """The function's largest pay times its rubric weight is below the weight of a separate exact match term."""
    W = _reward_weights(pkg)
    mine = [w for w, _ in W.get(fi.name, []) if w > 0]
    if not mine:
        return False
    top = _max_pay(list(_own_nodes(fi.node)))
    if top is None or top <= 0:
        return False
    others = [w for nm, entries in W.items() if nm != fi.name and EXACT_TERM.search(nm)
              for w, _ in entries if w > 0]
    return bool(others) and max(mine) * top < max(others)


@rule("numeric-tolerance-too-loose", "hack", "medium",
      "A numeric answer matcher accepts values far from the gold answer, such as a wide relative tolerance, "
      "a sign blind compare or a times 100 compare, so wrong numbers are paid as correct.")
def r_numeric_tolerance(pkg: Package):
    c = _cache(pkg)
    if "integer_hint" not in c:
        txt = " ".join([pkg.root, str(pkg.meta.get("name", "")) if isinstance(pkg.meta, dict) else ""])
        c["integer_hint"] = bool(INTEGER_HINT.search(txt) or INTEGER_HINT.search(_strings(pkg)[:2_000_000]))
    integer = c["integer_hint"]
    found = []
    for pf, fi in _grader_funcs(pkg):
        if HELPER_PATH.search(pf.path) or not re.search(r"(isclose|allclose|abs\(|fabs\(|\*\s*100|/\s*100|==\s*-)", fi.src):
            continue
        hits = _tolerance_hits(pf, fi, integer)
        if hits and _is_paid(pkg, fi) and not _below_exact_term(pkg, fi):
            found += [(pf, fi, ln, w) for ln, w in hits]
    if not found:
        return []
    pf, fi, line, what = found[0]
    return [F("numeric-tolerance-too-loose",
              f"{fi.name} matches numbers with {what}, so some wrong numbers are paid as correct.", [ev(p, ln) for p, _, ln, _ in found], confidence="review")]


# ---------------------------------------------------------------------------
# v0.5 formal-statement-vacuity (+ lean-dataset-vacuous-statements, extends formal-checker-escape-hatch)
# Statement Gaming (Haiku gaming 68% to 53% with a fidelity contract), arxiv:2606.29493 (4,833 defects
# in 398 problems), arxiv:2610.10604, arxiv:2611.05590 (Dafny verified but vacuous 89-95%).
# A shipped theorem the grader asks the policy to prove is vacuous: its conclusion is True, a
# hypothesis is False, the conclusion is also a hypothesis, or two hypotheses contradict (P and not P).
# A shipped Dafny spec with `ensures true` is the same defect. The proof then needs no work. Only the
# forms that plain text shows are checked: unused or jointly unsatisfiable hypotheses need Lean.
# Kept apart from formal-checker-escape-hatch because it is a dataset defect, not a gap in the audit.
# ---------------------------------------------------------------------------

LEAN_DECL = re.compile(r"\b(theorem|lemma)\s+([^\s(:{\[⦃❴]+)")
TRIVIAL_TYPES = {"ℕ", "ℤ", "ℝ", "ℚ", "ℂ", "Nat", "Int", "Real", "Rat", "Complex", "Prop", "Type", "Bool", "Fin n",
                 "Type*", "Sort*", "Type u"}
OPEN, CLOSE = "([{⦃", ")]}⦄"


def _lean_statement(text: str, start: int):
    """-> (binder types, conclusion) of the declaration whose name ends at `start`, or None."""
    i, n, binders = start, len(text), []
    limit = min(n, start + 4000)
    while i < limit:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in OPEN:
            depth, j = 0, i
            while j < limit:
                if text[j] in OPEN:
                    depth += 1
                elif text[j] in CLOSE:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = text[i + 1:j]
            if ":" in inner and not inner.lstrip().startswith(":"):
                binders.append(" ".join(inner.split(":", 1)[1].split()))
            i = j + 1
            continue
        if ch == ":" and not text.startswith(":=", i):
            depth, j = 0, i + 1
            while j < limit:
                c = text[j]
                if c in OPEN:
                    depth += 1
                elif c in CLOSE:
                    depth -= 1
                elif depth == 0 and (text.startswith(":=", j) or text.startswith("\n\n", j) or
                                     re.match(r"\s(by|where)\b", text[j:j + 7])):
                    break
                j += 1
            return binders, " ".join(text[i + 1:j].split())
        return None
    return None


def _vacuity(binders, concl) -> str | None:
    c = concl.strip("() ")
    if c == "True":
        return "its conclusion is True"
    hyps = [b.strip("() ") for b in binders]
    if any(h == "False" for h in hyps):
        return "a hypothesis is False"
    props = [h for h in hyps if h not in TRIVIAL_TYPES and not re.match(r"^(Type|Sort|Finset|Set|Fin|List)\b", h)
             and re.search(r"[=<>≤≥≠∣∧∨¬→↔]", h)]
    if c and c in props:
        return "its conclusion is also a hypothesis"
    for h in props:
        m = re.match(r"^¬\s*\(?(.*?)\)?$", h)
        if m and m.group(1).strip() in props:
            return "two hypotheses contradict"
    return None


def _statement_texts(pkg: Package):
    """(where, text) of shipped problem statements: .lean files and string fields of .jsonl / .json rows."""
    for p, t in pkg.files.items():
        if p.endswith(".lean") and ("theorem" in t or "lemma" in t):
            yield p, 1, t
    for p, t in pkg.text_files(".jsonl", ".json"):
        if "theorem" not in t and "ensures" not in t and "lemma" not in t:
            continue
        rows = []
        if p.endswith(".jsonl"):
            ln = 0
            for raw in t.splitlines()[:3000]:
                ln += 1
                if raw.lstrip().startswith("{"):
                    try:
                        rows.append((ln, json.loads(raw)))
                    except Exception:
                        continue
        else:
            try:
                d = json.loads(t)
            except Exception:
                continue
            items = d if isinstance(d, list) else (list(d.values()) if isinstance(d, dict) else [])
            rows = [(1, r) for r in items[:3000]]
        for ln, r in rows:
            vals = r.values() if isinstance(r, dict) else [r]
            for v in vals:
                if isinstance(v, str) and ("theorem" in v or "lemma" in v or "ensures" in v):
                    yield p, ln, v


@rule("formal-statement-vacuity", "hack", "medium",
      "A shipped Lean or Dafny problem statement is vacuous, such as a True conclusion, a False hypothesis or a "
      "conclusion repeated as a hypothesis, so it can be proved without solving the problem.")
def r_formal_vacuity(pkg: Package):
    if not _formal_files(pkg):
        return []
    text = "\n".join(_formal_files(pkg))
    lean = _lean_used(text)
    dafny = bool(re.search(r"(?i:dafny)", text))
    if not (lean or dafny):
        return []
    bad, total = [], 0
    for p, ln, t in _statement_texts(pkg):
        if lean:
            for m in LEAN_DECL.finditer(t):
                st = _lean_statement(t, m.end())
                if st is None:
                    continue
                total += 1
                why = _vacuity(*st)
                if why:
                    where = ln if ln > 1 or not p.endswith(".lean") else line_of(t, m.start())
                    bad.append((p, where, m.group(2), why))
        if dafny:
            for m in re.finditer(r"\bensures\s+(true\b|\(\s*true\s*\))", t):
                total += 1
                bad.append((p, ln if not p.endswith(".lean") else line_of(t, m.start()), "a method", "it has ensures true"))
        if total > 5000:
            break
    if not bad:
        return []
    p, ln, name, why = bad[0]
    return [F("formal-statement-vacuity",
              f"Shipped statement {name} is vacuous because {why}, so a proof needs no work"
              + (f" ({len(bad)} of {total} statements)." if len(bad) > 1 else "."),
              [f"{p}:{l}" for p, l, _, _ in bad], confidence="review")]

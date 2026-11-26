"""Reward family rules (envscan v0.3, batch A).

Rules about what the reward is computed from: tests the policy already sees, claims the policy makes
about its own work, auxiliary terms that pay for doing less, and graders that change the live
environment they grade. Like every envscan rule these read source text and ASTs only.

Each rule reports only when the flagged function is paid (a rubric weight above 0 on it or on a
direct caller) and the flagged value flows to that function's return. All findings are `review`.

Evidence ids in the comments refer to the ranked table in section 9.2 of the literature review
(research/lit-review/rollup/report_parts/09-open-problems-product.md).
"""
from __future__ import annotations

import ast
import re

from .package import Package, PyFile, FuncInfo, _const_text
from .rules import F, _classify_reward, all_funcs, called_names, ev, rule
from .rules_grader import _cache, _callee, _grader_funcs, _num, _own_nodes, _reward_weights

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _wheel_roots(pkg: Package) -> list[str] | None:
    """Paths the wheel ships, from pyproject.toml, or None when the build config does not limit them."""
    def calc():
        text = pkg.files.get("pyproject.toml")
        if not text:
            return None
        try:
            import tomllib
            cfg = tomllib.loads(text)
        except Exception:
            return None
        tool = cfg.get("tool", {}) if isinstance(cfg, dict) else {}
        hatch = tool.get("hatch", {}).get("build", {}) if isinstance(tool.get("hatch"), dict) else {}
        wheel = hatch.get("targets", {}).get("wheel", {}) if isinstance(hatch.get("targets"), dict) else {}
        roots = []
        for key in ("packages", "only-include", "include"):
            v = wheel.get(key) if isinstance(wheel, dict) else None
            if isinstance(v, list):
                roots += [str(x) for x in v]
        st = tool.get("setuptools", {}) if isinstance(tool.get("setuptools"), dict) else {}
        if isinstance(st.get("packages"), list):
            roots += [str(x).replace(".", "/") for x in st["packages"]]
        if isinstance(st.get("py-modules"), list):
            roots += [str(x) + ".py" for x in st["py-modules"]]
        roots = [r.strip("/").removeprefix("./") for r in roots if r and "*" not in r]
        return roots or None
    c = _cache(pkg)
    if "reward_wheel_roots" not in c:
        c["reward_wheel_roots"] = calc()
    return c["reward_wheel_roots"]


def _shipped(pkg: Package, path: str) -> bool:
    """False for a .py file that the wheel's build config leaves out (a stale copy beside the package)."""
    roots = _wheel_roots(pkg)
    if not roots:
        return True
    return any(path == r or path.startswith(r + "/") or path == r + ".py" for r in roots)


# calls that only parse or reshape text keep it the policy's text. Any other call computes something.
PASS_CALL = re.compile(r"^(str|int|float|bool|repr|list|dict|tuple|set|next|reversed|sorted|enumerate|filter|map|zip|"
                       r"getattr|any|all|min|max|abs|round|len|isinstance|hasattr|"
                       r"strip|lstrip|rstrip|split|rsplit|splitlines|lower|upper|casefold|title|replace|join|partition|"
                       r"rpartition|format|format_map|encode|decode|removeprefix|removesuffix|startswith|endswith|find|"
                       r"rfind|index|count|get|items|values|keys|pop|copy|setdefault|group|groups|groupdict|search|"
                       r"match|fullmatch|findall|finditer|sub|compile|dumps)$")


def _parse_only(pkg: Package, name: str, depth: int = 2, seen: frozenset = frozenset()) -> bool:
    """name is a parse step: a builtin text or regex call, a json or parse helper, or a function defined in the
    package whose own calls are all parse steps. An HTTP call, a model, a linter or an unknown import is not.
    A helper that calls itself (a recursive text flattener) is judged by its other calls."""
    if not name or PASS_CALL.match(name) or PARSE_FN.match(name) or name in seen:
        return True
    if depth <= 0:
        return False
    c = _cache(pkg)
    if "reward_local_funcs" not in c:
        c["reward_local_funcs"] = {fi.name: fi for _, fi in all_funcs(pkg)}
    local = c["reward_local_funcs"]
    fi = local.get(name)
    if fi is None:
        return False
    return all(_parse_only(pkg, _callee(c), depth - 1, seen | {name}) for c in ast.walk(fi.node)
               if isinstance(c, ast.Call) and c is not fi.node)


def _callers(pkg: Package) -> dict[str, set[str]]:
    c = _cache(pkg)
    if "rw_callers" not in c:
        m: dict[str, set[str]] = {}
        for _, g in all_funcs(pkg):
            for cn in called_names(g):
                m.setdefault(cn, set()).add(g.name)
        c["rw_callers"] = m
    return c["rw_callers"]


def _paid_weight(pkg: Package, name: str) -> float:
    """Largest positive rubric weight on a function or on one of its direct callers, else 0.
    Unlike rules_grader._payout_weight, a function that is in no rubric counts as not paid."""
    W = _reward_weights(pkg)
    if not W:
        return 0.0
    names = {name} | _callers(pkg).get(name, set())
    ws = [w for nm in names for w, _ in W.get(nm, [])]
    return max([w for w in ws if w > 0], default=0.0)


def _parents(fi: FuncInfo) -> dict:
    out = {}
    for x in [fi.node] + list(_own_nodes(fi.node)):
        for ch in ast.iter_child_nodes(x):
            out[ch] = x
    return out


def _loads(e) -> set[str]:
    return {x.id for x in ast.walk(e) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def _str_key(e):
    """'k' for x['k'] or x.get('k', ...)."""
    if isinstance(e, ast.Subscript) and isinstance(e.slice, ast.Constant) and isinstance(e.slice.value, str):
        return e.slice.value
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr == "get" and e.args \
            and isinstance(e.args[0], ast.Constant) and isinstance(e.args[0].value, str):
        return e.args[0].value
    return None


def _keys_read(node) -> set[str]:
    out = set()
    for n in ast.walk(node):
        k = _str_key(n)
        if k is not None and not (isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)):
            out.add(k)
    return out


def _assigned(own) -> list[tuple[list[str], ast.AST]]:
    """[(target names, value)] for simple assignments, augmented assignments and for loops."""
    out = []
    for n in own:
        if isinstance(n, ast.Assign):
            names = [x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)]
            out.append((names, n.value))
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None and isinstance(n.target, ast.Name):
            out.append(([n.target.id], n.value))
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            out.append(([x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)], n.iter))
        elif isinstance(n, ast.NamedExpr):
            out.append(([n.target.id], n.value))
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for it in n.items:
                if isinstance(it.optional_vars, ast.Name):
                    out.append(([it.optional_vars.id], it.context_expr))
    return out


def _positive_return(stmts) -> bool:
    for s_ in stmts:
        for x in ast.walk(s_):
            if isinstance(x, ast.Return) and x.value is not None:
                v = _num(x.value)
                if v is not None and v > 0:
                    return True
    return False


# ---------------------------------------------------------------------------
# 11. no-held-out-tests
# Evidence row 11 (section 9.2), arxiv:2601.06953 (56 of 700 X-Coder RL rollouts hard-coded the
# sample I/O shown in the prompt, and 59 of 60 hacked rollouts failed hidden tests).
# ---------------------------------------------------------------------------

TEST_KEY = re.compile(r"(^|_)(tests?|test_list|test_cases?|asserts?|assertions|unit_tests?|test_code|test_imports|"
                      r"examples?|sample_tests?|public_tests?|io|input_output|test_inputs?|test_outputs?)($|_)", re.I)
HIDDEN_KEY = re.compile(r"(hidden|private|held|challenge|extra|generated|full|all_tests|more_tests|plus|secret|eval_tests)",
                        re.I)
PROMPT_KEY = {"prompt", "question", "content", "instruction", "problem", "user_prompt", "task"}
RUN_CALL = re.compile(r"(exec|eval|subprocess|run|sandbox|execute|check_output|Popen|test|assert)", re.I)


def _prompt_shown_keys(pkg: Package, wanted: set[str]) -> dict[str, str]:
    """Test-like dataset keys (of `wanted`) whose value is written into the prompt text: key -> evidence."""
    out: dict[str, str] = {}
    quoted = [f'"{k}"' for k in wanted] + [f"'{k}'" for k in wanted]
    for pf in pkg.py_all():
        if pf.tree is None or not any(q in pf.text for q in quoted):
            continue
        # names assigned from a test-like key anywhere in a function: tests = row["test_list"]
        alias: dict[tuple[int, str], str] = {}
        for fi in pf.funcs:
            for names, val in _assigned(_own_nodes(fi.node)):
                k = _str_key(val)
                if isinstance(val, ast.Call) and val.args and k is None:  # "\n".join(row["test_list"])
                    k = next((_str_key(a) for a in val.args if _str_key(a)), None)
                if k and TEST_KEY.search(k) and not HIDDEN_KEY.search(k):
                    for nm in names:
                        alias[(fi.start, nm)] = k
        for fi in pf.funcs:
            for n in _own_nodes(fi.node):
                texts = []
                if isinstance(n, ast.Dict):
                    for k, v in zip(n.keys, n.values):
                        if isinstance(k, ast.Constant) and k.value in PROMPT_KEY:
                            texts.append(v)
                elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and re.search(
                        r"(prompt|question|instruction|user_msg|content)", t.id, re.I) for t in n.targets):
                    texts.append(n.value)
                for t in texts:
                    if not isinstance(t, (ast.JoinedStr, ast.BinOp, ast.Call)):
                        continue  # the whole field copied as is is not text the tests are added to
                    for x in ast.walk(t):
                        k = _str_key(x)
                        if k is None and isinstance(x, ast.Name):
                            k = alias.get((fi.start, x.id))
                        if k in wanted and k not in out:
                            out[k] = ev(pf, getattr(x, "lineno", fi.start))
    return out


@rule("no-held-out-tests", "hack", "high",
      "The tests that pay the reward are the same tests the prompt shows the policy, so code that "
      "hard codes the shown cases is paid like a real solution.")
def r_no_held_out(pkg: Package):
    cands = []
    for pf, fi in _grader_funcs(pkg):
        if not _shipped(pkg, pf.path):
            continue
        if not re.search(r"(test|example|sample|assert|\bio\b|input_output)", fi.src, re.I):
            continue
        keys = _keys_read(fi.node)
        test_keys = {k for k in keys if TEST_KEY.search(k) or HIDDEN_KEY.search(k)}
        if not test_keys or any(HIDDEN_KEY.search(k) for k in test_keys):
            continue  # the scorer also runs a test set the prompt is not built from
        if not RUN_CALL.search(" ".join(called_names(fi))) or not _paid_weight(pkg, fi.name):
            continue
        cands.append((pf, fi, test_keys))
    if not cands:
        return []
    shown = _prompt_shown_keys(pkg, set().union(*(k for _, _, k in cands)))
    hits = []
    for pf, fi, test_keys in cands:
        used = test_keys & set(shown)
        if used and not (test_keys - set(shown)):
            hits.append((pf, fi, sorted(used)))
    if not hits:
        return []
    pf, fi, used = hits[0]
    k = used[0]
    return [F("no-held-out-tests",
              f"The prompt shows the `{k}` tests and {fi.name} scores the reply only on those same tests, so "
              "hard coding the shown cases is paid as a solution.",
              [shown[k]] + [ev(p, f.start) for p, f, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# 12. reward-trusts-agent-claim
# Evidence row 12 (section 9.2), or:rwjv2zg0wa (a self-report verifier would accept 60% of
# untrained episodes vs 28.6% true success, Table 2), arxiv:2609.09218 (CONTRACT).
# Two forms: (a) the reward is a number or flag the policy writes in its own reply, (b) the reward
# is a state flag that an agent tool sets from its own argument or without any check.
# A success string printed by a process that ran model code is verdict-parse-semantics, not this.
# ---------------------------------------------------------------------------

CLAIM_WORD = re.compile(r"(\bfs\b|score|points|reward|success|succeed|complete|\bdone\b|solved|passed|pass_rate|resolved|"
                        r"achieved|accomplished|\bstatus\b|task_?finished|\bwon\b|\bwin\b)", re.I)
CLAIM_KEY = re.compile(r"^(fs|score|final_score|total_score|self_score|points|reward|success|succeeded|successful|"
                       r"complete|completed|is_complete|task_complete|task_completed|done|is_done|solved|is_solved|"
                       r"passed|all_passed|pass_rate|resolved|achieved|accomplished|status|task_status|finished|won)$",
                       re.I)
MODEL_PARAM = {"completion", "completions", "response", "messages", "model_output", "reply", "output"}
# calls whose result is no longer the policy's claim: a judge, a checker, running code, a reference
CHECK_CALL = re.compile(r"(judge|chat|create|generate|complet|invoke|run|exec|sandbox|subprocess|verify|grade|evaluat|"
                        r"compare|check|simulat|step|play|request|post|fetch|score|reward|test)", re.I)
REF_NAME = re.compile(r"(answer|gold|expected|target|reference|^ref|truth|^gt$|solution|label|info|state|task|example|"
                      r"oracle|correct|metadata|game|env|board|world)", re.I)


def _model_tainted(fi: FuncInfo, own, pkg: Package | None = None) -> set[str]:
    """Names in the function that hold the policy's reply or something parsed out of it. Only parse steps
    keep the taint: a value that went through any other call (an HTTP classifier, the env's own linter, a
    simulator) was computed, not claimed."""
    taint = {p for p in fi.params if p in MODEL_PARAM}
    if not taint:
        return set()
    asg = _assigned(own)
    for _ in range(4):
        grew = False
        for names, val in asg:
            if not (_loads(val) & taint):
                continue
            calls = [_callee(c) for c in ast.walk(val) if isinstance(c, ast.Call)]
            if any(CHECK_CALL.search(c) for c in calls):
                continue
            if pkg is not None and not all(_parse_only(pkg, c) for c in calls):
                continue
            if any(REF_NAME.search(x) for x in _loads(val) - taint):
                continue
            for nm in names:
                if nm not in taint:
                    taint.add(nm)
                    grew = True
        if not grew:
            break
    return taint


def _claim_expr(e, taint: set[str], regex_of: dict[str, str]) -> bool:
    """e reads a claim out of the policy's reply: d.get('success'), d['score'], m.group(1) where m is a
    regex search for a score or success field."""
    k = _str_key(e)
    if k is not None:
        base = e.value if isinstance(e, ast.Subscript) else e.func.value
        if isinstance(base, ast.Call) and PARSE_FN.match(_callee(base)):  # json.loads(text)["score"]
            base = base.args[0] if base.args else base
        return bool(CLAIM_KEY.match(k)) and isinstance(base, ast.Name) and base.id in taint
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr in ("group", "groups", "groupdict") \
            and isinstance(e.func.value, ast.Name):
        pat = regex_of.get(e.func.value.id, "")
        if not pat:
            return False
        if e.args and isinstance(e.args[0], ast.Constant) and isinstance(e.args[0].value, str):
            return bool(CLAIM_KEY.match(e.args[0].value) or CLAIM_WORD.search(e.args[0].value))
        return bool(CLAIM_WORD.search(pat))
    return False


PARSE_FN = re.compile(r"^(loads|literal_eval|_?parse\w*|_?extract\w*|from_json|model_validate_json)$")
# text the environment wrote (game state, tool results, the user simulator) is not the policy's claim
ENV_TEXT = re.compile(r"(get_user_messages|get_tool_messages|role[\"']?\]?\s*==\s*[\"'](user|tool)|"
                      r"role\s*==\s*[\"'](user|tool)|[\"']role[\"']\)\s*==\s*[\"'](user|tool))")


def _self_report_hits(pf: PyFile, fi: FuncInfo, pkg: Package | None = None):
    if ENV_TEXT.search(fi.src):
        return []
    own = list(_own_nodes(fi.node))
    taint = _model_tainted(fi, own, pkg)
    if not taint:
        return []
    regex_of: dict[str, str] = {}
    for names, val in _assigned(own):
        v = val.value if isinstance(val, ast.Await) else val
        if isinstance(v, ast.Call) and _callee(v) in ("search", "match", "fullmatch") and len(v.args) >= 2 \
                and _loads(v.args[1]) & taint:
            pat = _const_text(v.args[0]) or ""
            for nm in names:
                regex_of[nm] = pat
    # names that carry the claim
    claim: dict[str, int] = {}
    claim_at: set[tuple[str, int]] = set()  # (name, line) of assignments that carry the claim
    assigned_at: dict[str, list[int]] = {}
    asg = _assigned(own)
    for names, val in asg:
        for nm in names:
            assigned_at.setdefault(nm, []).append(val.lineno)
    def live_claim(e, line):
        """e at `line` reads a claim: the latest assignment before `line` of a name it loads carries one."""
        if any(_claim_expr(x, taint, regex_of) for x in ast.walk(e)):
            return True
        for nm in _loads(e) & set(claim):
            before = [ln for ln in assigned_at.get(nm, []) if ln < line]
            if before and (nm, max(before)) in claim_at:
                return True
        return False

    for _ in range(4):
        n0 = len(claim_at)
        for names, val in asg:
            if live_claim(val, val.lineno):
                if any(isinstance(c, ast.Call) and CHECK_CALL.search(_callee(c)) for c in ast.walk(val)):
                    continue
                if pkg is not None and not all(_parse_only(pkg, _callee(c)) for c in ast.walk(val)
                                               if isinstance(c, ast.Call)):
                    continue
                for nm in names:
                    claim.setdefault(nm, val.lineno)
                    claim_at.add((nm, val.lineno))
        if len(claim_at) == n0:
            break
    def carries(e):
        return bool(_loads(e) & set(claim)) or any(_claim_expr(x, taint, regex_of) for x in ast.walk(e))
    # a claim checked against a reference or the real state is not trusted
    for n in own:
        if isinstance(n, ast.Compare) and carries(n):
            others = [n.left] + n.comparators
            if any(not carries(o) and not isinstance(o, ast.Constant) and _num(o) is None and
                   any(REF_NAME.search(x) for x in _loads(o) | {a.attr for a in ast.walk(o) if isinstance(a, ast.Attribute)})
                   for o in others):
                return []
    hits = []
    parent = _parents(fi)
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None and _num(n.value) is None and live_claim(n.value, n.lineno):
            hits.append(n.lineno)
        elif isinstance(n, (ast.If, ast.IfExp)) and live_claim(n.test, n.lineno):
            body = n.body if isinstance(n, ast.If) else [ast.Return(value=n.body)]
            if isinstance(n, ast.IfExp):
                a, b = _num(n.body), _num(n.orelse)
                if a is not None and b is not None and a > b and isinstance(parent.get(n), ast.Return):
                    hits.append(n.lineno)
            elif _positive_return(body):
                hits.append(n.lineno)
    return sorted(set(hits))


TOOL_DECOR = re.compile(r"(\btool\b|function_tool)")


def _tool_names(pkg: Package) -> set[str]:
    out = set()
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call):
                for k in n.keywords:
                    if k.arg == "tools" and isinstance(k.value, (ast.List, ast.Tuple)):
                        for e in k.value.elts:
                            if isinstance(e, ast.Name):
                                out.add(e.id)
                            elif isinstance(e, ast.Attribute):
                                out.add(e.attr)
                if _callee(n) in ("add_tool", "register_tool") and n.args:
                    a = n.args[0]
                    out.add(a.id if isinstance(a, ast.Name) else a.attr if isinstance(a, ast.Attribute) else "")
        for fi in pf.funcs:
            if any(TOOL_DECOR.search(d) for d in fi.decorators):
                out.add(fi.name)
    out.discard("")
    return out


STATE_RECV = re.compile(r"^(state|self\.state|self|self\._state|env_state)$")


def _tool_flag_writes(pkg: Package) -> dict[str, tuple[PyFile, FuncInfo, int]]:
    """state key -> (tool, line) for a tool that sets a claim key to one of its own arguments, or to
    True with no condition above it."""
    out = {}
    if not any("tool" in pf.text for pf in pkg.py_all()):
        return out
    tools = _tool_names(pkg)
    if not tools:
        return out
    for pf, fi in all_funcs(pkg):
        if fi.name not in tools:
            continue
        args = {p for p in fi.params if p not in ("self", "state", "kwargs", "args")}
        parent = _parents(fi)
        for n in _own_nodes(fi.node):
            if not isinstance(n, ast.Assign):
                continue
            for t in n.targets:
                key = recv = None
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                    key, recv = t.slice.value, ast.unparse(t.value)
                elif isinstance(t, ast.Attribute):
                    key, recv = t.attr, ast.unparse(t.value)
                if not key or not CLAIM_KEY.match(key) or not STATE_RECV.match(recv or ""):
                    continue
                from_arg = bool(_loads(n.value) & args) and not any(isinstance(c, ast.Call) for c in ast.walk(n.value))
                p, guarded = parent.get(n), False
                while p is not None and p is not fi.node:
                    if isinstance(p, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
                        guarded = True
                    p = parent.get(p)
                const_true = isinstance(n.value, ast.Constant) and n.value.value is True and not guarded
                if from_arg or const_true:
                    out.setdefault(key, (pf, fi, n.lineno))
    return out


def _flag_payout(fi: FuncInfo, key: str) -> int | None:
    """Line where a reward pays on state[key] alone (returns it, or returns a positive constant when it
    is set), when the function reads nothing that could be the real outcome."""
    reads = _keys_read(fi.node)
    if key not in reads:
        return None
    if any(REF_NAME.search(k) and k != key for k in reads if k not in ("info", "state")):
        return None
    for n in _own_nodes(fi.node):
        if isinstance(n, ast.Return) and n.value is not None:
            if key in _keys_read(n.value) and _num(n.value) is None:
                return n.lineno
        if isinstance(n, ast.If) and key in _keys_read(n.test) and _positive_return(n.body):
            return n.lineno
    return None


@rule("reward-trusts-agent-claim", "hack", "high",
      "The reward is read from what the policy says about its own work, a score or success field in its "
      "reply or a flag its own tool call sets, so claiming success is paid like succeeding.")
def r_trusts_claim(pkg: Package):
    out = []
    hits = []
    for pf, fi in _grader_funcs(pkg):
        if not set(fi.params) & MODEL_PARAM or not CLAIM_WORD.search(fi.src) or not _shipped(pkg, pf.path):
            continue
        lines = _self_report_hits(pf, fi, pkg)
        if lines and _paid_weight(pkg, fi.name):
            hits.append((pf, fi, lines))
    if hits:
        pf, fi, lines = hits[0]
        out.append(F("reward-trusts-agent-claim",
                     f"{fi.name} pays a score or success value that it parses out of the policy's own reply and never "
                     "checks, so a reply that just claims success is paid.",
                     [ev(p, ln) for p, _, ls in hits for ln in ls], confidence="review"))
    flags = _tool_flag_writes(pkg)
    fhits = []
    for key, (tpf, tfi, tln) in flags.items():
        if not _shipped(pkg, tpf.path):
            continue
        for pf, fi in _grader_funcs(pkg):
            if not _shipped(pkg, pf.path):
                continue
            ln = _flag_payout(fi, key)
            if ln and _paid_weight(pkg, fi.name):
                fhits.append((key, tpf, tfi, tln, pf, fi, ln))
    if fhits:
        key, tpf, tfi, tln, pf, fi, ln = fhits[0]
        out.append(F("reward-trusts-agent-claim",
                     f"The tool {tfi.name} sets `{key}` from what the policy passes it and {fi.name} pays on that flag, "
                     "so calling the tool is paid like doing the task.",
                     [ev(tpf, tln), ev(pf, ln)] + [ev(h[4], h[6]) for h in fhits[1:]], confidence="review"))
    # v0.5 branches: a judge that reads the policy's own report, and a score block the policy can forge
    done = {e.rsplit(":", 1)[0] for f in out for e in f.evidence}
    out += _judge_self_report(pkg, done)
    out += _forged_score_token(pkg, done)
    return out


# ---------------------------------------------------------------------------
# 14. aggregation-without-gate (extends format-heavy-reward)
# Evidence row 14 (section 9.2), arxiv:2610.21404 (a brevity term gave pass@4 0.138 vs 0.481,
# Table 1), arxiv:2609.38847 (additive scoring dropped appropriateness from 54.1 to 26.0, Table 3).
# Only the part no existing rule covers: an ungated term that pays more the less work is done
# (fewer turns, tokens, characters or dollars), added to the task terms with a weight at least as
# large. Format and keyword shares are format-heavy-reward and keyword-reward.
# ---------------------------------------------------------------------------

AUX_NAME = re.compile(r"(length|brev|concis|short|token|cost|effic|speed|fast|turn|step_count|latency|budget|verbos|"
                      r"word_count|len_|num_steps|steps_used)", re.I)
WORK_COUNT = re.compile(r"(len\(|turn|token|step|cost|word|char|time|latency|elapsed|duration|num_calls|tool_calls)", re.I)
GATE_REF = re.compile(r"(answer|gold|expected|target|reference|truth|correct|solved|success|passed|is_right|accuracy|"
                      r"verify|judge|score|reward|info\b|result)", re.I)


def _decreasing_term(fi: FuncInfo):
    """-> scale (largest value) when every return is a literal or an arithmetic expression that falls as a
    count of work (turns, tokens, length, cost) rises, for example 1 - turns / max_turns. A plain
    threshold (return 1.0 if tokens <= 2048) is a budget, not a trade off, and does not count."""
    arith, scale = False, []
    for n in _own_nodes(fi.node):
        if not isinstance(n, ast.Return) or n.value is None:
            continue
        v = n.value
        x = _num(v)
        if x is not None:
            scale.append(x)
            continue
        if isinstance(v, ast.IfExp):
            return None
        src = ast.unparse(v)
        mult = 1.0
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult):
            a, b = _num(v.left), _num(v.right)
            mult = a if a is not None else (b if b is not None else 1.0)
            v = v.right if a is not None else v.left
            src = ast.unparse(v)
        names = _loads(v)
        local = {}
        for names_, val in _assigned(_own_nodes(fi.node)):
            for nm in names_:
                local.setdefault(nm, []).append(val)
        exprs = [v] + [e for nm in names for e in local.get(nm, [])]
        text = " ".join(ast.unparse(e) for e in exprs) + " " + " ".join(names)
        has_sub = any(isinstance(b, ast.BinOp) and isinstance(b.op, (ast.Sub, ast.Div)) for e in exprs for b in ast.walk(e))
        if not (WORK_COUNT.search(text) and has_sub):
            return None
        arith = True
        scale.append(mult)
    if not arith:
        return None
    return max(scale) if scale else 1.0


@rule("aggregation-without-gate", "hack", "medium",
      "An ungated term that pays more for fewer turns, tokens or characters is added to the task reward "
      "with at least the same weight, so a fast wrong reply can outscore a slow correct one.")
def r_aggregation(pkg: Package):
    old = _aggregation_v04(pkg)
    if old:
        return old
    return _aggregation_v05(pkg)


def _aggregation_v04(pkg: Package):
    W = _reward_weights(pkg)
    if not W:
        return []
    funcs = {}
    for pf, fi in all_funcs(pkg):
        if _shipped(pkg, pf.path):
            funcs.setdefault(fi.name, (pf, fi))
    aux, task = [], 0.0
    for nm, entries in W.items():
        w = sum(x for x, _ in entries if x > 0)
        if w <= 0:
            continue
        if nm in funcs and AUX_NAME.search(nm):
            pf, fi = funcs[nm]
            names = _loads(fi.node) | _keys_read(fi.node) | {p for p in fi.params}
            gated = any(GATE_REF.search(x) for x in names - {"kwargs", "state", "completion", "parser"})
            sc = None if gated else _decreasing_term(fi)
            if sc:
                aux.append((pf, fi, w * sc))
                continue
        sc = 1.0
        if nm in funcs:
            fi = funcs[nm][1]
            vals = [_num(r.value) for r in _own_nodes(fi.node) if isinstance(r, ast.Return) and r.value is not None]
            vals = [x for x in vals if x is not None]
            if vals and all(x <= 1.0 for x in vals) and max(vals) > 0:
                sc = max(vals)
        task += w * sc
    if not aux or task <= 0:
        return []
    aux_w = sum(w for _, _, w in aux)
    if aux_w < task:
        return []
    names = ", ".join(fi.name for _, fi, _ in aux)
    # weights given by a variable are read as verifiers' default 1.0, so say so instead of stating them as fact
    unread = any(share is None and w > 0 for entries in W.values() for w, share in entries)
    wtxt = (f"weight {aux_w:g} against {task:g} for the task terms"
            + (" (a weight set by a variable is read as verifiers' default 1.0, check the real values)" if unread else ""))
    return [F("aggregation-without-gate",
              f"{names} has no correctness gate and divides by a length, turn, token or cost count, so its value "
              f"rises as the reply gets shorter or cheaper. It is added to the task reward at {wtxt}, so a short "
              "wrong reply can outscore a correct one.",
              [ev(pf, fi.start) for pf, fi, _ in aux], confidence="review")]


# ---------------------------------------------------------------------------
# 17. grader-mutates-live-env
# Evidence row 17 (section 9.2), url:openreview.net/forum?id=gwfsbzapnd (a Hub tau-bench env that
# computes the reward twice reported 127 passes of 159 where the benchmark's grade gives 22,
# Sec 3.1, App C.1). tau-bench's Env.calculate_reward reloads the initial data and replays the
# gold actions on the live env, so a second scoring of the same rollout reads the gold state.
# ---------------------------------------------------------------------------

RELOAD_CALL = re.compile(r"^(reset|reload\w*|load_state|set_state|restore_state|load_db|load_data|reinit\w*|"
                         r"init_state|replay\w*|data_load_func)$")
RELOAD_TEXT = re.compile(r"\.(reset|reload|load_state|set_state|restore_state|load_db|load_data|reinit|init_state|replay)|"
                         r"data_load_func|\.data\s*=")
STEP_CALL = re.compile(r"^(step|invoke|invoke_tool|execute_action|apply_action|call_tool|run_action)$")
FRESH_CALL = re.compile(r"(deepcopy|copy|clone|fork|snapshot|from_dict|model_copy|Env$|Environment$|"
                        r"get_env|make_env|new_env|create)", re.I)
GOLD_ITER = re.compile(r"(gold|expected|reference|ground_truth|task\.actions|\.actions\b|oracle|solution)", re.I)


def _live_mutations(fi: FuncInfo):
    """[(line, receiver text)] where the function reloads or resets an env object it did not build itself
    and then replays reference actions on that same object."""
    own = list(_own_nodes(fi.node))
    built = set()
    for names, val in _assigned(own):
        if any(isinstance(c, ast.Call) and (FRESH_CALL.search(_callee(c)) or _callee(c)[:1].isupper())
               for c in ast.walk(val)):
            built.update(names)
    reloads = []
    for n in own:
        recv = None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and RELOAD_CALL.match(n.func.attr):
            recv = n.func.value
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute) \
                and n.targets[0].attr == "data" and isinstance(n.value, ast.Call) and \
                re.search(r"(load|init)", _callee(n.value), re.I):
            recv = n.targets[0].value  # self.data = self.data_load_func()
        if recv is None:
            continue
        root = next((x.id for x in ast.walk(recv) if isinstance(x, ast.Name)), None)
        if root in built:
            continue
        reloads.append((n.lineno, ast.unparse(recv)))
    if not reloads:
        return []
    out = []
    for ln, rtext in reloads:
        for n in own:
            if isinstance(n, (ast.For, ast.AsyncFor)) and n.lineno > ln and GOLD_ITER.search(ast.unparse(n.iter)):
                for c in ast.walk(n):
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and STEP_CALL.match(c.func.attr) \
                            and ast.unparse(c.func.value) == rtext:
                        out.append((ln, rtext))
    return out


@rule("grader-mutates-live-env", "broken", "high",
      "The reward reloads the live environment and replays the reference actions on it, so any later "
      "scoring of the same rollout reads the reference state instead of the agent's.")
def r_grader_mutates(pkg: Package):
    grader = [(pf, fi) for pf, fi in _grader_funcs(pkg) if _shipped(pkg, pf.path)]
    W = _reward_weights(pkg)
    if not W:
        return []
    scorers = set(W)  # paid rewards and logged metrics both run on every rollout
    mut = []
    for pf, fi in grader:
        if not RELOAD_TEXT.search(fi.src):
            continue
        for ln, recv in _live_mutations(fi):
            mut.append((pf, fi, ln, recv))
    # tau-bench's own Env.calculate_reward does the reload and replay inside the library
    if re.search(r"\btau_bench\b", "\n".join(pf.text for pf in pkg.py_all())):
        for pf, fi in grader:
            for n in _own_nodes(fi.node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "calculate_reward":
                    root = next((x.id for x in ast.walk(n.func.value) if isinstance(x, ast.Name)), "")
                    if root not in {nm for names, v in _assigned(_own_nodes(fi.node)) for nm in names
                                    if any(isinstance(c, ast.Call) and FRESH_CALL.search(_callee(c)) for c in ast.walk(v))}:
                        mut.append((pf, fi, n.lineno, "tau:" + ast.unparse(n.func.value)))
    if not mut:
        return []
    # the defect bites only when a rollout is scored more than once on that env: two scorers reach the
    # mutating function, or one scorer reaches it twice
    callers = _callers(pkg)
    hits = []
    for pf, fi, ln, recv in mut:
        reach = set()
        frontier = {fi.name}
        for _ in range(3):
            nxt = set()
            for nm in frontier:
                if nm in scorers:
                    reach.add(nm)
                nxt |= callers.get(nm, set())
            frontier = nxt - reach
        target = "calculate_reward" if recv.startswith("tau:") else fi.name
        memo: dict[str, bool] = {}

        def leads(nm, depth=0):
            if nm not in memo:
                memo[nm] = False
                memo[nm] = nm in scorers or (depth < 3 and any(leads(c, depth + 1) for c in callers.get(nm, ())))
            return memo[nm]
        sites = sum(1 for _, g in grader if leads(g.name) for c in _own_nodes(g.node)
                    if isinstance(c, ast.Call) and _callee(c) == target)
        recv = recv[4:] if recv.startswith("tau:") else recv
        if fi.name in scorers:
            reach.add(fi.name)
        if len(reach) >= 2 or sites >= 2:
            hits.append((pf, fi, ln, recv, sorted(reach)))
    if not hits:
        return []
    pf, fi, ln, recv, reach = hits[0]
    return [F("grader-mutates-live-env",
              f"{fi.name} reloads `{recv}` and replays the reference actions on it, and {len(reach) or 2} scoring "
              "calls read that env, so a later score sees the reference state.",
              [ev(p, l) for p, _, l, _, _ in hits], confidence="review")]


# ===========================================================================
# v0.5 additions (builder `reward`). Reader item ids (ST....) refer to research/mine/merge/static_rule.md.
# ===========================================================================

GOLDISH_NAME = re.compile(r"^(answer|answers|gold\w*|ground_truth|gt|expected\w*|target|targets|reference\w*|ref|solution|"
                          r"solutions|label|labels|truth|correct_answer|final_answer|true_answer|oracle\w*)$", re.I)
VERIFY_NAME = re.compile(r"(math_verify|verify|is_equiv|equiv|grade_answer|exact_match|run_tests|execute|sandbox|"
                         r"simulat|compare|check_answer|score_answer|evaluate_answer|isclose|allclose)", re.I)
CORRECT_NAME = re.compile(r"(correct|accura|solved|success|pass|win|match|exact|answer|solution|result|outcome|"
                          r"task_score|verif|reward_func$|^reward$|f1|recall|precision|em$|judge)", re.I)
FORMATISH = re.compile(r"(format|xml|tag|json|schema|parse|structure|style|length|brev|concis)", re.I)
AUXISH_NAME = re.compile(r"(format|style|valid|reason|think|explain|quality|cite|citation|tool|progress|step|length|"
                         r"coheren|fluen|clarity|polite|tone|structure|engag|effort|safety|creativ|helpful|readab|"
                         r"brev|concis|speed|efficien|turn|usage|partial|strateg|methodolog|complete(ness)?$|"
                         r"thorough|detail|depth|insight|xml|tag|json|schema|parse)", re.I)


def _reads_gold(fi: FuncInfo) -> bool:
    """The function reads a gold answer: an answer like parameter, key or attribute, or an info/state field
    that holds one."""
    for x in _own_nodes(fi.node):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and GOLDISH_NAME.match(x.id):
            return True
        if isinstance(x, ast.Attribute) and GOLDISH_NAME.match(x.attr):
            return True
        k = _str_key(x)
        if k and GOLDISH_NAME.match(k):
            return True
    return False


def _reads_outcome(fi: FuncInfo) -> bool:
    """The function reads an outcome the env computed (a state field like success or score, a test run,
    a simulator), which makes it a correctness term even with no gold answer in view."""
    for x in _own_nodes(fi.node):
        k = _str_key(x)
        if k and re.search(r"(success|solved|score|reward|passed|pass_rate|win|won|result|outcome|correct|"
                           r"completed|done|resolved|accuracy|test)", k, re.I):
            return True
        if isinstance(x, ast.Call) and VERIFY_NAME.search(_callee(x)):
            return True
        if isinstance(x, ast.Attribute) and re.search(r"(success|solved|score|passed|won|winner|result|outcome)",
                                                       x.attr, re.I):
            return True
    return False


def _max_return(fi: FuncInfo) -> float:
    """Largest literal return, read as the term's scale when every return is a literal or bounded (else 1)."""
    vals = [_num(r.value) for r in _own_nodes(fi.node) if isinstance(r, ast.Return) and r.value is not None]
    lits = [v for v in vals if v is not None]
    if vals and len(lits) == len(vals) and max(lits) > 0:
        return max(lits)
    return 1.0


# ---------------------------------------------------------------------------
# aggregation-without-gate, v0.5 widening
# Item 1: additive-rubric-without-hard-gate + ungated-non-correctness-term. Reader items reader items,
# reader items,
# reader items. Main papers arxiv:2610.21404 (brevity pass@4 0.581 gated vs 0.220 additive),
# d33ce8re31 (dropping the multiplicative gate reverses 28 large gap pairs).
# Item 5: process-or-tool-credit-ungated. Reader items reader items. Main paper og5skvifue
# (mean aggregation reward 0.95 with accuracy 0.9 pp below base).
# Three forms: (a) rubric terms that never read the answer or an env outcome carry at least the weight of
# the correctness terms, (b) one reward function adds such terms to its correctness term with at least
# the same coefficient and no gate, (c) a term pays per tool call or step with no cap and no gate.
# Rubrics where format-heavy-reward or keyword-reward already fires are left to those rules.
# ---------------------------------------------------------------------------

TOOL_COUNT_TEXT = re.compile(r"(tool_calls|['\"]tool['\"]|num_tools|tool_count|n_tools|num_tool_calls|tool_use|"
                             r"calls_made|num_calls|step_scores|prm|process_reward|step_rewards)", re.I)
CAP_CALL = re.compile(r"^(min|clip|clamp|tanh|sigmoid)$")


def _uncapped_count(fi: FuncInfo) -> int | None:
    """Line of a return that pays a raw count of tool calls or steps (len(...) or sum(...) over them), not
    divided and not capped."""
    for r in _own_nodes(fi.node):
        if not isinstance(r, ast.Return) or r.value is None:
            continue
        v = r.value
        if isinstance(v, ast.Call) and _callee(v) in ("float", "int") and v.args:
            v = v.args[0]
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult):
            v = v.right if _num(v.left) is not None else v.left if _num(v.right) is not None else v
        if any(isinstance(c, ast.Call) and CAP_CALL.match(_callee(c)) for c in ast.walk(r.value)):
            continue
        if any(isinstance(b, ast.BinOp) and isinstance(b.op, (ast.Div, ast.FloorDiv)) for b in ast.walk(r.value)):
            continue
        text = ast.unparse(v)
        defs = {}
        for names, val in _assigned(_own_nodes(fi.node)):
            for nm in names:
                defs.setdefault(nm, []).append(ast.unparse(val))
        more = " ".join(t for nm in _loads(v) for t in defs.get(nm, []))
        if isinstance(v, ast.Call) and _callee(v) in ("len", "sum") and TOOL_COUNT_TEXT.search(text + " " + more):
            return r.lineno
        if isinstance(v, ast.Name) and TOOL_COUNT_TEXT.search(v.id + " " + more) and \
                re.search(r"(len\(|sum\(|\+=|count)", more + " " + fi.src):
            return r.lineno
    return None


def _never_positive(fi: FuncInfo) -> bool:
    """Every return is a literal at or below 0 or a negated value: a penalty, which cannot add reward."""
    rets = [r.value for r in _own_nodes(fi.node) if isinstance(r, ast.Return) and r.value is not None]
    if not rets:
        return True
    for v in rets:
        x = _num(v)
        if x is not None:
            if x > 0:
                return False
            continue
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub):
            continue
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult) and ((_num(v.left) or 0) < 0 or (_num(v.right) or 0) < 0):
            continue
        if isinstance(v, ast.Call) and _callee(v) == "min" and any(_num(a) == 0 for a in v.args):
            continue
        return False
    return True


def _term_kind(pf: PyFile, fi: FuncInfo) -> str:
    """correctness | aux | penalty | unknown for a rubric term. An aux term is computed from the policy's
    text alone: a term that reads the env state or the task info may be an earned outcome."""
    if _reads_gold(fi) or _reads_outcome(fi):
        return "correctness"
    if _never_positive(fi):
        return "penalty"
    if "state" in _loads(fi.node) or "info" in _loads(fi.node) or "state" in fi.params or "info" in fi.params \
            or _keys_read(fi.node) & {"info", "state", "task", "example", "metadata"}:
        return "unknown"
    if CORRECT_NAME.search(fi.name) and not AUXISH_NAME.search(fi.name):
        return "unknown"
    return "aux"


def _rubric_level(pkg: Package, W, funcs):
    corr, aux, unknown = 0.0, [], 0.0
    for nm, entries in W.items():
        w = max([x for x, _ in entries if x > 0], default=0.0)  # a name in several rubrics counts once
        if w <= 0:
            continue
        if nm not in funcs:
            if re.search(r"format", nm, re.I):
                aux.append((None, None, w, nm))
            else:
                unknown += w
            continue
        pf, fi = funcs[nm]
        k = _term_kind(pf, fi)
        if k == "correctness":
            corr += w * _max_return(fi)
        elif k == "aux":
            aux.append((pf, fi, w * _max_return(fi), nm))
        elif k == "unknown":
            unknown += w
    if corr <= 0 or not aux or unknown > 0:
        return None
    aux_w = sum(w for _, _, w, _ in aux)
    if aux_w < corr:
        return None
    # format only aux terms are format-heavy-reward's case
    if all(fi is None or _classify_reward(pf, fi) in ("format", "bag") or FORMATISH.search(fi.name)
           for pf, fi, _, _ in aux):
        return None
    return corr, aux, aux_w


def _additive_terms(e):
    """[(coef, expr)] for e = c1 * x1 + c2 * x2 + ... (a bare x counts as coefficient 1), else None."""
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
        a, b = _additive_terms(e.left), _additive_terms(e.right)
        return a + b if a is not None and b is not None else None
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
        a, b = _num(e.left), _num(e.right)
        if a is not None and b is None:
            return [(a, e.right)]
        if b is not None and a is None:
            return [(b, e.left)]
        return None  # a product of two values is a gate
    if isinstance(e, ast.Call) and _callee(e) in ("float", "int") and len(e.args) == 1:
        return _additive_terms(e.args[0])
    if isinstance(e, ast.IfExp):
        a, b = _num(e.body), _num(e.orelse)
        if a is not None and b == 0:
            return [(a, e.test)]
        return None
    if _num(e) is not None:
        return []
    if isinstance(e, (ast.Name, ast.Attribute, ast.Subscript, ast.Call, ast.Compare, ast.BoolOp)):
        return [(1.0, e)]
    return None


def _expr_kind(pkg: Package, e, defs, fi: FuncInfo) -> str:
    chain = [e]
    frontier = _loads(e)
    seen = set()
    for _ in range(4):
        nxt = set()
        for nm in frontier - seen:
            seen.add(nm)
            for v in defs.get(nm, []):
                chain.append(v)
                nxt |= _loads(v)
        frontier = nxt
    local = _callers(pkg)  # touch the cache so helper lookups below are cheap
    del local
    lf = _cache(pkg).setdefault("reward_local_funcs", {f.name: f for _, f in all_funcs(pkg)})
    for x in chain:
        for n in ast.walk(x):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and GOLDISH_NAME.match(n.id):
                return "correctness"
            k = _str_key(n)
            if k and (GOLDISH_NAME.match(k) or re.search(r"(success|solved|passed|correct|score|result)", k, re.I)):
                return "correctness"
            if isinstance(n, ast.Call):
                cn = _callee(n)
                if VERIFY_NAME.search(cn) or re.search(r"(judge|grade|correct|accura|solved|score)", cn, re.I):
                    return "correctness"
                g = lf.get(cn)
                if g is not None and g is not fi and (_reads_gold(g) or _reads_outcome(g)):
                    return "correctness"
    return "aux"


def _in_function(pkg: Package, pf: PyFile, fi: FuncInfo):
    """-> (line, corr coef, aux coef, aux text) when the function returns a weighted sum whose non
    correctness terms carry at least the correctness weight, with no gate."""
    own = list(_own_nodes(fi.node))
    defs: dict[str, list] = {}
    for names, val in _assigned(own):
        for nm in names:
            defs.setdefault(nm, []).append(val)
    rets = [r for r in own if isinstance(r, ast.Return) and r.value is not None]
    final = [r for r in rets if _num(r.value) is None]
    if len(final) != 1:
        return None
    r = final[0]
    terms = _additive_terms(r.value)
    # reward = 0, then reward += c under conditions, then return reward
    if (terms is None or len(terms) < 2) and isinstance(r.value, ast.Name):
        acc = r.value.id
        terms = []
        parent = _parents(fi)
        for n in own:
            if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add) and isinstance(n.target, ast.Name) \
                    and n.target.id == acc:
                c = _num(n.value)
                p = parent.get(n)
                if c is not None and isinstance(p, ast.If) and n in p.body:
                    terms.append((c, p.test))
                elif c is None:
                    t = _additive_terms(n.value)
                    if t is None:
                        return None
                    terms += t
                else:
                    terms.append((c, n.value))
            elif isinstance(n, ast.AugAssign) and n.target is not None and isinstance(n.target, ast.Name) \
                    and n.target.id == acc and not isinstance(n.op, ast.Add):
                return None  # a multiply or other op on the total is a gate
    if not terms or len(terms) < 2:
        return None
    corr, aux, aux_txt = 0.0, 0.0, []
    for c, e in terms:
        if c <= 0:
            continue
        if _expr_kind(pkg, e, defs, fi) == "correctness":
            corr += c
        else:
            aux += c
            aux_txt.append(ast.unparse(e)[:40])
    if corr <= 0 or aux < corr:
        return None
    # a gate: an early zero return under a test of the correctness value
    for n in own:
        if isinstance(n, ast.If) and any(isinstance(x, ast.Return) and x.value is not None and _num(x.value) == 0
                                         for s_ in n.body for x in ast.walk(s_)):
            if _expr_kind(pkg, n.test, defs, fi) == "correctness" and n.lineno > fi.start + 1:
                return None
    return r.lineno, corr, aux, aux_txt


def _aggregation_v05(pkg: Package):
    W = _reward_weights(pkg)
    if not W:
        return []
    funcs = {}
    for pf, fi in all_funcs(pkg):
        if _shipped(pkg, pf.path):
            funcs.setdefault(fi.name, (pf, fi))
    from .rules import r_format_heavy, r_keyword
    out = []
    for nm, entries in W.items():
        if nm not in funcs or max([x for x, _ in entries], default=0.0) <= 0:
            continue
        pf, fi = funcs[nm]
        if not _reads_gold(fi) and not _reads_outcome(fi):
            ln = _uncapped_count(fi)
            if ln:
                out.append(F("aggregation-without-gate",
                             f"{fi.name} pays for each tool call or step with no cap and no check of the answer, so "
                             "making more calls raises the reward without solving the task.",
                             [ev(pf, ln)], confidence="review", variant="v05"))
    lvl = None if out else _rubric_level(pkg, W, funcs)
    if lvl and (r_format_heavy(pkg) or r_keyword(pkg)):
        lvl = None
    if lvl:
        corr, aux, aux_w = lvl
        names = ", ".join(nm for _, _, _, nm in aux)
        unread = any(share is None and w > 0 for entries in W.values() for w, share in entries)
        out.append(F("aggregation-without-gate",
                     f"The rubric terms {names} never read the answer or an env outcome and carry weight {aux_w:g} "
                     f"against {corr:g} for the correctness terms, with no gate, so a reply that meets them but is "
                     "wrong earns at least half of the full reward"
                     + (" (a weight set by a variable is read as 1.0, check the real values)." if unread else "."),
                     [ev(pf, fi.start) for pf, fi, _, _ in aux if fi is not None] or ["rubric"], confidence="review", variant="v05"))
    # forms (b) and (c) look inside the paid functions themselves
    for nm, entries in W.items():
        w = sum(x for x, _ in entries if x > 0)
        if w <= 0 or nm not in funcs:
            continue
        pf, fi = funcs[nm]
        if out and any(e.startswith(pf.path + ":") for f in out for e in f.evidence):
            continue
        if set(fi.params) & MODEL_PARAM or "kwargs" in fi.params:
            res = _in_function(pkg, pf, fi)
            if res:
                ln, c, a, txt = res
                out.append(F("aggregation-without-gate",
                             f"{fi.name} adds {', '.join(txt[:3])} to its correctness term with coefficient {a:g} "
                             f"against {c:g} and no gate, so a reply that meets those terms but is wrong earns at "
                             "least half of the full reward.",
                             [ev(pf, ln)], confidence="review", variant="v05"))
    return out[:3]


# ---------------------------------------------------------------------------
# reward-trusts-agent-claim, v0.5 widening (a): the judge reads the policy's own report
# Item 2: judge-reads-agent-self-report + judge-sees-policy-self-report. Reader items reader items,
# reader items. Main paper arxiv:2610.29609 (withholding the report raised judge recall
# +43.4 pp), arxiv:2610.40578 (online FPR 7.7 vs 15.4 with attested evidence).
# A paid judge prompt that asks whether the agent did the task fills a summary, report, explanation or
# claim slot with text the policy wrote, so the judge reads the policy's claim as evidence.
# ---------------------------------------------------------------------------

SELF_SLOT = re.compile(r"^(summary|final_summary|agent_summary|work_summary|report|final_report|agent_report|self_report|"
                       r"explanation|agent_explanation|self_critique|critique|reflection|claim|claims|agent_claim|"
                       r"completion_message|final_message|agent_message|last_message|final_response|agent_response|"
                       r"status_update|what_was_done|actions_summary|agent_output|agent_final_answer)$", re.I)
TASK_JUDGE = re.compile(r"(complet|accomplish|succe|achiev|resolved|fulfil|did the agent|has the agent|whether the "
                        r"agent|assistant (has|was able))", re.I)
JUDGE_CALLISH = re.compile(r"(judge|create|chat|complet|generate|invoke|grade)", re.I)
JUDGEY = re.compile(r"(judge|grade|grading|evaluat|assess|verdict|rate|rating|score)", re.I)


def _template_text(pf: PyFile, e) -> str | None:
    t = _const_text(e)
    if t:
        return t
    if isinstance(e, ast.Name) and e.id in pf.consts:
        return pf.consts[e.id]
    if isinstance(e, ast.Attribute) and e.attr in pf.consts:
        return pf.consts[e.attr]
    return None


def _policy_text(e, taint: set[str], fi: FuncInfo) -> bool:
    if _loads(e) & taint:
        return True
    for n in ast.walk(e):
        k = _str_key(n)
        if k and SELF_SLOT.match(k):
            return True
    return False


def _judge_self_report(pkg: Package, done: set[str]):
    hits = []
    for pf, fi in _grader_funcs(pkg):
        if pf.path in done or not _shipped(pkg, pf.path) or not SELF_SLOT.pattern or "{" not in pf.text:
            continue
        if not re.search(r"(summary|report|explanation|critique|reflection|claim|final_message|agent_message|"
                         r"last_message|final_response|agent_response|status_update|agent_output)", fi.src, re.I):
            continue
        own = list(_own_nodes(fi.node))
        taint = {p for p in fi.params if p in MODEL_PARAM}
        asg = _assigned(own)
        for _ in range(4):
            grew = False
            for names, val in asg:
                if (_loads(val) & taint or any(SELF_SLOT.match(k) for k in _keys_read(val))) and \
                        not any(isinstance(c, ast.Call) and JUDGE_CALLISH.search(_callee(c)) for c in ast.walk(val)):
                    for nm in names:
                        if nm not in taint:
                            taint.add(nm)
                            grew = True
            if not grew:
                break
        for n in own:
            slots = []
            tpl = None
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "format":
                tpl = _template_text(pf, n.func.value)
                slots = [(k.arg, k.value) for k in n.keywords if k.arg and SELF_SLOT.match(k.arg)]
            elif isinstance(n, ast.JoinedStr):
                tpl = "".join(v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
                slots = [(ast.unparse(v.value), v.value) for v in n.values if isinstance(v, ast.FormattedValue)
                         and isinstance(v.value, ast.Name) and SELF_SLOT.match(v.value.id)]
            if not slots or not tpl or not TASK_JUDGE.search(tpl) or not JUDGEY.search(tpl + " " + fi.name):
                continue
            for name, val in slots:
                if _policy_text(val, taint, fi) or (isinstance(val, ast.Name) and val.id in taint):
                    hits.append((pf, fi, n.lineno, name))
                    break
        if hits and hits[-1][1] is fi and not _paid_weight(pkg, fi.name):
            hits.pop()
    if not hits:
        return []
    pf, fi, ln, name = hits[0]
    return [F("reward-trusts-agent-claim",
              f"The judge in {fi.name} decides whether the task was done from a prompt that includes `{name}`, text the "
              "policy wrote about its own work, so a confident claim of success can sway the judge.",
              [ev(p, l) for p, _, l, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# reward-trusts-agent-claim, v0.5 widening (b): a score block the policy can forge
# Item 4: score-token-forged-in-policy-text. Reader item reader items. Main paper arxiv:2608.24231 (fake score
# block injection observed, a reserved token gate rejects them).
# The env writes a score or verdict line into its own messages, and the reward finds that line with a
# regex over text built from every message, the policy's included, so a reply that writes the same
# line first is paid.
# ---------------------------------------------------------------------------

MSG_LIST_KEY = re.compile(r"^(completion|messages|trajectory|responses|history|turns|transcript|conversation|prompt)$")
SCORE_PAT = re.compile(r"(score|reward|verdict|points|final|result|grade|winner|won|game over|total|status|passed)", re.I)


def _literal_prefix(pat: str) -> str:
    out = []
    i = 0
    while i < len(pat):
        ch = pat[i]
        if ch == "\\" and i + 1 < len(pat):
            nx = pat[i + 1]
            if nx in "sdwbSDWB":
                break
            out.append(nx)
            i += 2
            continue
        if ch in "([{.*+?^$|":
            break
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _all_messages_text(e, defs, fi: FuncInfo) -> bool:
    """e is text built from every message of the completion with no role filter."""
    chain = [e]
    frontier, seen = _loads(e), set()
    for _ in range(4):
        nxt = set()
        for nm in frontier - seen:
            seen.add(nm)
            for v in defs.get(nm, []):
                chain.append(v)
                nxt |= _loads(v)
        frontier = nxt
    text = " ".join(ast.unparse(x) for x in chain)
    if re.search(r"role", text):
        return False

    def whole(it):
        if isinstance(it, ast.Name) and it.id in MODEL_PARAM and it.id in fi.params:
            return True
        k = _str_key(it)
        if k and MSG_LIST_KEY.match(k):
            return True
        if isinstance(it, ast.BinOp) and isinstance(it.op, ast.Add):
            return whole(it.left) or whole(it.right)
        return False
    for x in chain:
        if isinstance(x, ast.Call) and _callee(x) in ("str", "dumps", "repr") and x.args and whole(x.args[0]):
            return True
        for n in ast.walk(x):
            if isinstance(n, (ast.GeneratorExp, ast.ListComp)) and any(whole(g.iter) for g in n.generators):
                return True
    return False


def _forged_score_token(pkg: Package, done: set[str]):
    hits = []
    env_strings = None
    for pf, fi in _grader_funcs(pkg):
        if pf.path in done or not _shipped(pkg, pf.path):
            continue
        if not re.search(r"\b(search|findall|finditer|match)\(", fi.src):
            continue
        own = list(_own_nodes(fi.node))
        defs: dict[str, list] = {}
        for names, val in _assigned(own):
            for nm in names:
                defs.setdefault(nm, []).append(val)
        for n in own:
            if not (isinstance(n, ast.Call) and _callee(n) in ("search", "findall", "finditer", "match")
                    and len(n.args) >= 2):
                continue
            pat = _const_text(n.args[0]) or ""
            if not pat or not SCORE_PAT.search(pat):
                continue
            pre = _literal_prefix(pat)
            if len(pre) < 4 or not SCORE_PAT.search(pre):
                continue
            if not _all_messages_text(n.args[1], defs, fi):
                continue
            if env_strings is None:
                env_strings = []
                for pf2, g in all_funcs(pkg):
                    if (pf2, g) in [(pf, fi)]:
                        continue
                    for c in _own_nodes(g.node):
                        if isinstance(c, ast.Constant) and isinstance(c.value, str):
                            env_strings.append(c.value)
                        elif isinstance(c, ast.JoinedStr):
                            env_strings.append("".join(v.value for v in c.values if isinstance(v, ast.Constant)
                                                       and isinstance(v.value, str)))
            if not any(pre.lower() in s.lower() and s.strip() != pat for s in env_strings):
                continue
            if _paid_weight(pkg, fi.name):
                hits.append((pf, fi, n.lineno, pre))
                break
    if not hits:
        return []
    pf, fi, ln, pre = hits[0]
    return [F("reward-trusts-agent-claim",
              f"{fi.name} reads the `{pre}` line the env writes with a regex over every message, the policy's own "
              "included, so a reply that writes its own such line is paid as if the env had written it.",
              [ev(p, l) for p, _, l, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# grader-or-chain
# Item 9. Reader item reader items. Main paper 2611.17955 (no number). A rubric then judge chain where any
# grader passing counts as a pass. Fires when a paid reward passes on an `or` (or any(), or max()) of
# two or more different grader calls on the answer and at least one of them is loose: an LLM judge,
# a fuzzy or similarity score, or a substring test. The OR pays every false pass of the loosest one.
# ---------------------------------------------------------------------------

GRADER_CALL = re.compile(r"(verify|judge|grade|match|equal|equiv|check|compare|correct|score|similar|fuzz|ratio|"
                         r"contains|evaluate|is_\w+)", re.I)
LOOSE_CALL = re.compile(r"(judge|llm|gpt|chat|completions|fuzz|similar|ratio|partial|contains|semantic|embed|"
                        r"rouge|bleu|levenshtein|startswith|endswith)", re.I)


def _or_operands(e):
    if isinstance(e, ast.BoolOp) and isinstance(e.op, ast.Or):
        return list(e.values)
    if isinstance(e, ast.Call) and _callee(e) in ("any", "max") and len(e.args) == 1 and \
            isinstance(e.args[0], (ast.List, ast.Tuple)):
        return list(e.args[0].elts)
    if isinstance(e, ast.Call) and _callee(e) == "max" and len(e.args) >= 2:
        return list(e.args)
    return None


def _grader_operand(e):
    """callee name when e is (await) grader_call(...), possibly compared to a threshold, else None.
    A `x in text` substring test counts as the loose grader `in`."""
    if isinstance(e, ast.Await):
        e = e.value
    if isinstance(e, ast.Compare) and len(e.ops) == 1 and isinstance(e.ops[0], ast.In):
        return "in"
    if isinstance(e, ast.Compare) and len(e.ops) == 1 and _num(e.comparators[0]) is not None:
        e = e.left
        if isinstance(e, ast.Await):
            e = e.value
    if isinstance(e, ast.Call):
        cn = _callee(e)
        if GRADER_CALL.search(cn) or LOOSE_CALL.search(cn):
            return cn
    return None


@rule("grader-or-chain", "hack", "medium",
      "A reward passes when any one of several graders passes, one of them loose, so every false pass of the "
      "loosest grader is paid.")
def r_grader_or_chain(pkg: Package):
    if not _reward_weights(pkg):
        return []
    hits = []
    for pf, fi in _grader_funcs(pkg):
        if not (" or " in fi.src or "any(" in fi.src or "max(" in fi.src) or not _shipped(pkg, pf.path):
            continue
        parent = None
        for n in _own_nodes(fi.node):
            ops = _or_operands(n)
            if not ops or len(ops) < 2:
                continue
            names = [_grader_operand(o) for o in ops]
            gs = [x for x in names if x]
            if len(set(gs)) < 2 or not any(LOOSE_CALL.search(x) or x == "in" for x in gs):
                continue
            if any(x is None for x in names):
                continue  # a plain flag or value mixed in is not a chain of graders
            if parent is None:
                parent = _parents(fi)
            p, ok = parent.get(n), False
            while p is not None and p is not fi.node:
                if isinstance(p, ast.Return):
                    ok = True
                    break
                if isinstance(p, (ast.If, ast.IfExp)):
                    body = p.body if isinstance(p, ast.If) else [ast.Return(value=p.body)]
                    ok = _positive_return(body)
                    break
                if isinstance(p, ast.Assign):
                    tg = {x.id for t in p.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
                    ok = any(isinstance(r, ast.Return) and r.value is not None and _loads(r.value) & tg
                             for r in _own_nodes(fi.node))
                    break
                p = parent.get(p)
            if ok and _paid_weight(pkg, fi.name):
                hits.append((pf, fi, n.lineno, gs))
                break
    if not hits:
        return []
    pf, fi, ln, gs = hits[0]
    return [F("grader-or-chain",
              f"{fi.name} pays when any of {', '.join(sorted(set(gs))[:4])} passes, so every false pass of the "
              "loosest of them is paid as correct.",
              [ev(p, l) for p, _, l, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# count-reward-no-dedup
# Item 10. Reader items reader items. Main paper arxiv:2611.15346 (citations per
# response 12.46 to 4.27 after the fix), catalog case sqlite-db-truncate (trustworthy-env T-9).
# A paid reward (or a Harbor test) counts items the policy produced with no set() or other dedup, and
# pays on that count: a len() threshold, a raw count, or matches over the policy's items divided by
# the gold count. Repeating one good item raises the reward.
# ---------------------------------------------------------------------------

DEDUP = re.compile(r"(\bset\(|frozenset|dict\.fromkeys|unique|dedup|distinct|drop_duplicates|Counter\(|seen\b|"
                   r"\bseen\s*=|np\.unique|OrderedDict\.fromkeys|\bdupl)", re.I)
ITEM_SRC = re.compile(r"(findall|finditer|split|splitlines|readlines|fetchall|json\.load|loads|DictReader|reader\(|"
                      r"extract\w*|parse\w*)")


def _count_hits(pf: PyFile, fi: FuncInfo, harbor: bool):
    own = list(_own_nodes(fi.node))
    defs: dict[str, list] = {}
    for names, val in _assigned(own):
        for nm in names:
            defs.setdefault(nm, []).append(val)
    if harbor:
        taint = set()
        for names, val in _assigned(own):
            if any(isinstance(c, ast.Call) and re.search(r"(load|read|fetchall|findall|split|DictReader)", _callee(c))
                   for c in ast.walk(val)) and re.search(r"(open|read|load|fetch|query|execute|stdout|output|cursor)",
                                                         ast.unparse(val)):
                taint.update(names)
    else:
        taint = _model_tainted(fi, own) | {p for p in fi.params if p in MODEL_PARAM}
    if not taint:
        return None
    if DEDUP.search(fi.src) or (harbor and DEDUP.search(pf.text)):
        return None  # a Harbor task passes only when every test passes, so a sibling dedup test counts

    def items(e):
        """e is a list of items parsed out of the policy's output (a split, findall or json list)."""
        if not isinstance(e, ast.Name) or e.id not in taint or e.id in MODEL_PARAM:
            return False
        vals = defs.get(e.id, [])
        return bool(vals) and all(ITEM_SRC.search(ast.unparse(v)) or isinstance(v, (ast.ListComp,)) for v in vals)

    parent = _parents(fi)
    for n in own:
        # len(items) >= N on the pass path
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.GtE, ast.Gt)) and \
                isinstance(n.left, ast.Call) and _callee(n.left) == "len" and n.left.args and items(n.left.args[0]):
            th = _num(n.comparators[0])
            if th is None or th < 2:
                continue
            p = parent.get(n)
            if harbor and isinstance(p, ast.Assert) and fi.name.startswith("test"):
                return n.lineno, ast.unparse(n.left.args[0]), "a length threshold"
            if isinstance(p, ast.If) and _positive_return(p.body):
                return n.lineno, ast.unparse(n.left.args[0]), "a length threshold"
        # sum(1 for x in items if x in gold) / len(gold), or a raw count of the policy's items
        if isinstance(n, ast.Return) and n.value is not None and not harbor:
            v = n.value
            num = v.left if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Div) else v
            if isinstance(v, ast.Call) and _callee(v) == "min" and v.args:
                inner = [a for a in v.args if _num(a) is None]
                if inner and isinstance(inner[0], ast.BinOp) and isinstance(inner[0].op, ast.Div):
                    num = inner[0].left
                    v = inner[0]
            den = v.right if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Div) else None
            for g in ast.walk(num):
                if isinstance(g, (ast.GeneratorExp, ast.ListComp)) and len(g.generators) == 1:
                    it = g.generators[0].iter
                    if items(it) and any(isinstance(c, ast.Compare) and any(isinstance(o, ast.In) for o in c.ops)
                                         for c in ast.walk(g)):
                        if den is not None and isinstance(den, ast.Call) and _callee(den) == "len" and den.args and \
                                ast.unparse(den.args[0]) == ast.unparse(it):
                            continue  # a precision over the same list does not grow with repeats
                        return n.lineno, ast.unparse(it), "matches over the policy's items"
    return None


@rule("count-reward-no-dedup", "hack", "medium",
      "A reward counts items the policy listed with no dedup, so repeating one good item raises the reward.")
def r_count_no_dedup(pkg: Package):
    hits = []
    if _reward_weights(pkg):
        for pf, fi in _scoring_funcs_shipped(pkg):
            if "len(" not in fi.src and "sum(" not in fi.src:
                continue
            r = _count_hits(pf, fi, False)
            if r and _paid_weight(pkg, fi.name):
                hits.append((pf, r))
    from .rules_grader import _harbor_test_py
    for pf in _harbor_test_py(pkg):
        if "len(" not in pf.text:
            continue
        for fi in pf.funcs:
            if fi.name.startswith("test") and "assert" in fi.src:
                r = _count_hits(pf, fi, True)
                if r:
                    hits.append((pf, r))
    if not hits:
        return []
    pf, (ln, what, how) = hits[0]
    return [F("count-reward-no-dedup",
              f"The reward pays on {how} over `{what}` from the policy's output with no dedup, so listing the "
              "same item many times raises the score.",
              [ev(p, r[0]) for p, r in hits], confidence="review")]


def _scoring_funcs_shipped(pkg: Package):
    return [(pf, fi) for pf, fi in _grader_funcs(pkg) if _shipped(pkg, pf.path) and pf.path in pkg.py]

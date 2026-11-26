"""Environment side rules for tool using and multi-turn environments (envscan v0.5, builder `env`).

Rules about the world the policy acts in rather than the answer check: rewards paid for calling a
tool, per-episode state kept where concurrent rollouts share it, state-diff graders that skip fields,
tasks that name things the shipped start state does not hold, tool lists longer than API caps, tools
that claim success without changing state, tool output cut short with no notice, read-only SQL
guards that reject valid queries, and prompt wording that pushes a policy to pass tests by any means.

Like every envscan rule these read source text and ASTs only. All findings are `review`.
Reader item ids (ST....) refer to research/mine/merge/static_rule.md.
"""
from __future__ import annotations

import ast
import json
import os
import re

from .package import Package, PyFile, FuncInfo, _const_text
from .rules import F, all_funcs, called_names, ev, rule
from .rules_grader import _cache, _callee, _num, _own_nodes, _reward_weights
from .rules_reward import _assigned, _keys_read, _loads, _parents, _positive_return, _shipped, _str_key, _tool_names

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

ENV_METHODS = {"env_response", "step", "setup_state", "is_completed", "rollout", "call_tool", "update_tool_args",
               "reset", "get_observation", "observe", "post_rollout", "add_trajectory_step", "render_obs"}
GOLD_RE = re.compile(r"^(answer|answers|gold\w*|ground_truth|gt|expected\w*|target|targets|reference\w*|ref|solution|"
                     r"solutions|label|labels|truth|correct_\w+|final_answer|true_answer|oracle\w*)$", re.I)
OUTCOME_KEY = re.compile(r"(success|solved|score|reward|passed|pass_rate|\bwin|won|result|outcome|correct|completed|"
                         r"resolved|accuracy|test|answer|db\b|database|hash|final_state|world|expected|target)", re.I)
VERIFY_CALL = re.compile(r"(verify|equiv|grade|judge|exact_match|run_tests|execute|sandbox|simulat|compare|check|"
                         r"evaluate|score|match|assert)", re.I)


def _funcs_by_name(pkg: Package) -> dict[str, tuple[PyFile, FuncInfo]]:
    c = _cache(pkg)
    if "env_funcs" not in c:
        m = {}
        for pf, fi in all_funcs(pkg):
            if _shipped(pkg, pf.path):
                m.setdefault(fi.name, (pf, fi))
        c["env_funcs"] = m
    return c["env_funcs"]


def _tools(pkg: Package) -> set[str]:
    """rules_reward._tool_names, computed once per package."""
    c = _cache(pkg)
    if "env_tools" not in c:
        c["env_tools"] = _tool_names(pkg)
    return c["env_tools"]


def _paid(pkg: Package) -> dict[str, float]:
    """name -> largest positive rubric weight, for functions that are paid."""
    c = _cache(pkg)
    if "env_paid" not in c:
        out = {}
        for nm, entries in _reward_weights(pkg).items():
            w = max([x for x, _ in entries if x > 0], default=0.0)
            if w > 0:
                out[nm] = w
        c["env_paid"] = out
    return c["env_paid"]


def _paid_funcs(pkg: Package):
    """(pf, fi, weight) for paid rubric functions and their direct local callees."""
    c = _cache(pkg)
    if "env_paid_funcs" in c:
        return c["env_paid_funcs"]
    funcs = _funcs_by_name(pkg)
    out, seen = [], set()
    for nm, w in _paid(pkg).items():
        if nm in funcs and nm not in seen:
            seen.add(nm)
            pf, fi = funcs[nm]
            out.append((pf, fi, w))
            for cn in called_names(fi):
                if cn in funcs and cn not in seen:
                    seen.add(cn)
                    out.append((funcs[cn][0], funcs[cn][1], w))
    c["env_paid_funcs"] = out
    return out


def _reads_gold_or_outcome(fi: FuncInfo, skip: set | None = None) -> bool:
    skip = skip or set()
    for x in _own_nodes(fi.node):
        if id(x) in skip:
            continue
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and GOLD_RE.match(x.id):
            return True
        if isinstance(x, ast.Attribute) and GOLD_RE.match(x.attr):
            return True
        k = _str_key(x)
        if k and (GOLD_RE.match(k) or OUTCOME_KEY.search(k)):
            return True
        if isinstance(x, ast.Call) and VERIFY_CALL.search(_callee(x)):
            return True
    return False


def _unparse(n) -> str:
    try:
        return ast.unparse(n)
    except Exception:
        return ""


def _rollout_funcs(pkg: Package):
    """(pf, fi) for code that runs during a rollout: env methods, tools and paid reward functions."""
    c = _cache(pkg)
    if "env_rollout" in c:
        return c["env_rollout"]
    tools = _tools(pkg)
    paid = _paid(pkg)
    out = []
    for pf, fi in all_funcs(pkg):
        if not _shipped(pkg, pf.path):
            continue
        if fi.name in ENV_METHODS or fi.name in tools or fi.name in paid:
            out.append((pf, fi))
    c["env_rollout"] = out
    return out


# ---------------------------------------------------------------------------
# 1. tool-call-count-or-name-reward
# Reader items reader items. Main papers arxiv:2603.16839 (a no-op status tool loop found
# within 300 steps), arxiv:2609.32691 (a grader that keys on tool identity was off by 18x).
# A paid reward whose value is the number of tool calls, whether any tool was called, or whether a
# tool of a given name was called, with no check of the call's arguments, the state or the answer.
# Uncapped raw counts with no answer check are left to aggregation-without-gate.
# ---------------------------------------------------------------------------

TOOLISH = re.compile(r"(tool_calls?\b|tool_call_\w+|tool_name|tool_use\w*|tools_used|tool_history|num_tools|n_tools|"
                     r"tool_count|num_tool_calls|n_tool_calls|calls_made|function_call|['\"]tool['\"]|toolCall|"
                     r"\.function\.name)", re.I)
ARG_READ = re.compile(r"(\.arguments\b|\[['\"](arguments|args|params|parameters|input)['\"]\]|"
                      r"\.get\(['\"](arguments|args|params|parameters|input)['\"]|tool_args|\.args\b|\.input\b)")


def _tool_taint(own) -> set[str]:
    defs = _assigned(own)
    taint: set[str] = set()
    for _ in range(4):
        grew = False
        for names, val in defs:
            if any(n in taint for n in names):
                continue
            if TOOLISH.search(_unparse(val)) or (_loads(val) & taint):
                taint |= set(names)
                grew = True
        if not grew:
            break
    return taint


def _toolish(e, taint) -> bool:
    return bool(TOOLISH.search(_unparse(e)) or (_loads(e) & taint))


def _name_compare(e, taint) -> ast.Compare | None:
    """x.name == 'search', name in ('a', 'b') or similar, where one side is tool data."""
    for n in ast.walk(e):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Eq, ast.In)):
            left, right = n.left, n.comparators[0]
            sides = [_unparse(left), _unparse(right)]
            strs = [isinstance(s, ast.Constant) and isinstance(s.value, str) or
                    (isinstance(s, (ast.Set, ast.List, ast.Tuple)) and s.elts and
                     all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in s.elts))
                    for s in (left, right)]
            if any(strs) and any(re.search(r"(name|tool)", t, re.I) for t in sides) and \
                    (_toolish(left, taint) or _toolish(right, taint)):
                return n
    return None


def _tool_credit(fi: FuncInfo):
    """-> (line, form) when the function's positive value comes from tool call data alone."""
    own = list(_own_nodes(fi.node))
    if not TOOLISH.search(fi.src):
        return None
    if ARG_READ.search(fi.src):
        return None
    taint = _tool_taint(own)
    parent = _parents(fi)
    name_nodes: set[int] = set()
    hit = None
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None and _num(n.value) is None:
            v = n.value
            if not _toolish(v, taint):
                continue
            if any(isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub) for x in ast.walk(v)):
                continue
            if any(isinstance(x, ast.Compare) and any(isinstance(o, (ast.Lt, ast.LtE)) for o in x.ops)
                   for x in ast.walk(v)):
                continue  # pays for fewer calls: an efficiency term
            nc = _name_compare(v, taint)
            if nc is not None:
                name_nodes |= {id(x) for x in ast.walk(nc)}
                hit = hit or (n.lineno, "name")
            elif any(isinstance(x, ast.Call) and _callee(x) in ("len", "sum", "count") for x in ast.walk(v)) or \
                    isinstance(v, ast.Name) or re.search(r"(len\(|sum\(|count)", " ".join(
                        _unparse(val) for names, val in _assigned(own) if set(names) & _loads(v))):
                hit = hit or (n.lineno, "count")
            else:
                hit = hit or (n.lineno, "any")
        elif isinstance(n, ast.If) and _toolish(n.test, taint) and _positive_return(n.body):
            if any(isinstance(o, (ast.Lt, ast.LtE, ast.NotEq, ast.NotIn)) for x in ast.walk(n.test)
                   if isinstance(x, ast.Compare) for o in x.ops):
                continue
            if any(isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.Not) for x in ast.walk(n.test)):
                continue
            nc = _name_compare(n.test, taint)
            if nc is not None:
                name_nodes |= {id(x) for x in ast.walk(nc)}
            hit = hit or (n.lineno, "name" if nc is not None else "any")
        elif isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add) and (_num(n.value) or 0) > 0:
            p = parent.get(n)
            while p is not None and p is not fi.node and not isinstance(p, (ast.For, ast.AsyncFor)):
                p = parent.get(p)
            if isinstance(p, (ast.For, ast.AsyncFor)) and _toolish(p.iter, taint):
                hit = hit or (n.lineno, "count")
    if hit is None:
        return None
    if _reads_gold_or_outcome(fi, name_nodes):
        return None
    return hit


def _rubric_line(pkg: Package, name: str) -> tuple[str, bool] | None:
    """(file:line, weight is a literal) where a function is put in a rubric."""
    for pf in pkg.py_all():
        if pf.tree is None or name not in pf.text:
            continue
        for n in ast.walk(pf.tree):
            if not isinstance(n, ast.Call):
                continue
            fname = _callee(n)
            kws = {k.arg: k.value for k in n.keywords if k.arg}
            funcs = kws.get("funcs") or kws.get("reward_funcs") or \
                (n.args[0] if n.args and isinstance(n.args[0], (ast.List, ast.Tuple)) else None)
            if (fname.endswith("Rubric") or "funcs" in kws) and isinstance(funcs, (ast.List, ast.Tuple)):
                for i, e in enumerate(funcs.elts):
                    if _unparse(e).split(".")[-1].split("(")[0] == name:
                        w = kws.get("weights")
                        lit = w is None or (isinstance(w, (ast.List, ast.Tuple)) and i < len(w.elts) and
                                            _num(w.elts[i]) is not None)
                        return ev(pf, e.lineno), lit
            if fname == "add_reward_func" and n.args and _unparse(n.args[0]).split(".")[-1] == name:
                w = kws.get("weight") or (n.args[1] if len(n.args) > 1 else None)
                return ev(pf, n.lineno), w is None or _num(w) is not None
    return None


def _tool_bonus_writes(pkg: Package):
    """state key -> (pf, fi, line) for a tool that adds a positive constant to a state field."""
    out = {}
    tools = _tools(pkg)
    for pf, fi in all_funcs(pkg):
        if fi.name not in tools:
            continue
        for n in _own_nodes(fi.node):
            if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add) and (_num(n.value) or 0) > 0:
                k = _str_key(n.target) if isinstance(n.target, ast.Subscript) else None
                if k and re.search(r"(reward|bonus|score|points|credit)", k, re.I):
                    out.setdefault(k, (pf, fi, n.lineno))
    return out


@rule("tool-call-count-or-name-reward", "hack", "medium",
      "A paid reward pays for calling a tool, by count or by tool name, with no check of the call's arguments, the state or the answer.")
def r_tool_call_reward(pkg: Package):
    paid = _paid(pkg)
    if not paid or not any(TOOLISH.search(pf.text) or re.search(r"(tools\s*=|add_tool|@\w*tool)", pf.text)
                           for pf in pkg.py_all()):
        return []
    funcs = _funcs_by_name(pkg)
    cands = []
    for nm, w in paid.items():
        if nm not in funcs:
            continue
        pf, fi = funcs[nm]
        res = _tool_credit(fi)
        if res:
            cands.append((pf, fi, w, res))
    bonus = []
    bonus_keys = {k for nm in paid if nm in funcs for k in _keys_read(funcs[nm][1].node)
                  if re.search(r"(reward|bonus|score|points|credit)", k, re.I)}
    for k, (pf, fi, ln) in (_tool_bonus_writes(pkg).items() if bonus_keys else []):
        for nm in paid:
            if nm in funcs and k in _keys_read(funcs[nm][1].node):
                bonus.append((pf, fi, ln, k, nm))
                break
    if not cands and not bonus:
        return []
    covered: list[str] = []
    if cands:
        try:
            from .rules_reward import r_aggregation
            covered = [e for f in r_aggregation(pkg) for e in f.evidence]
        except Exception:
            covered = []
    out = []
    for pf, fi, w, (ln, form) in cands:
        if any(e.rsplit(":", 1)[0] == pf.path and fi.start <= int(e.rsplit(":", 1)[1] or 0) <= fi.end
               for e in covered if e.rsplit(":", 1)[-1].isdigit()):
            continue
        what = {"name": "whether a tool with a given name was called",
                "count": "the number of tool calls",
                "any": "whether any tool was called"}[form]
        rl = _rubric_line(pkg, fi.name)
        wtxt = f"weight {w:g}" + ("" if rl is None or rl[1] else " (set by a variable, read as 1.0, check it)")
        out.append(F("tool-call-count-or-name-reward",
                     f"{fi.name} is paid at {wtxt} and its value is {what}, with no check of the call's "
                     "arguments, the state or the answer, so a call that does nothing useful earns it.",
                     [ev(pf, ln)] + ([rl[0]] if rl else []), confidence="review"))
    for pf, fi, ln, k, nm in bonus:
        out.append(F("tool-call-count-or-name-reward",
                     f"The tool {fi.name} adds a fixed bonus to state['{k}'] on every call and the paid reward {nm} "
                     "reads it, so calling the tool in a loop raises the reward.",
                     [ev(pf, ln)], confidence="review"))
    return out[:4]


# ---------------------------------------------------------------------------
# 2. module-state-shared-across-rollouts
# Reader items reader items. Main paper i7fgehiana (WebArena resets the world only after
# all 812 tasks, so residue from earlier episodes can satisfy later goals).
# verifiers runs the rollouts of a group concurrently on one env object, so per-episode state must
# live in `state`. Three forms: a module level dict, list or set that rollout code mutates and
# reads, an attribute of the env object that rollout code writes and then reads, and a fixed /tmp
# path that rollout or scoring code writes on the host.
# ---------------------------------------------------------------------------

MUTATORS = {"append", "extend", "update", "add", "insert", "setdefault", "pop", "popitem", "clear", "remove",
            "discard", "appendleft", "__setitem__"}
MUTABLE_CALL = {"dict", "list", "set", "defaultdict", "deque", "OrderedDict", "Counter"}
CACHEISH = re.compile(r"(cache|memo|registry|_lock|lock$|client|pool|tokeni|model|session|semaphore|executor|logger|"
                      r"loaded|_seen_warn|warned|stats|metric|counter|usage|cost|timing|_log$|^log|history_log|"
                      r"sandbox|container|_ids?$|pending|tasks$|futures|handles|conns?$|connections|"
                      r"dataset|config|spec|schema|tools?$|tool_map|tool_defs|oai_tools|parser|rubric)", re.I)
ENV_BASE = re.compile(r"^((vf|verifiers)\.\w+|(Multi|Single)TurnEnv|(Stateful)?ToolEnv|SandboxEnv|PythonEnv|"
                      r"CliAgentEnv|TextArenaEnv|HarborEnv|ComposableEnv|RLMEnv|MCPEnv)$")
TMP_LIT = re.compile(r"^/tmp/[\w./-]+$")


def _module_mutables(pf: PyFile) -> dict[str, int]:
    out = {}
    if pf.tree is None:
        return out
    for n in pf.tree.body:
        tgt = val = None
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            tgt, val = n.targets[0].id, n.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            tgt, val = n.target.id, n.value
        if tgt is None:
            continue
        if isinstance(val, (ast.Dict, ast.List, ast.Set)) and not (getattr(val, "keys", None) or getattr(val, "elts", None)):
            out[tgt] = n.lineno
        elif isinstance(val, ast.Call) and _callee(val) in MUTABLE_CALL and not val.args[1:] and \
                not (val.args and isinstance(val.args[0], (ast.List, ast.Tuple, ast.Dict)) and
                     (getattr(val.args[0], "elts", None) or getattr(val.args[0], "keys", None))):
            out[tgt] = n.lineno
    return out


def _mutations(fi: FuncInfo, recv_ok) -> list[tuple[str, int, ast.AST]]:
    """[(receiver text, line, key node or None)] for x[k] = v, x.append(...), x.update(...) and the like."""
    out = []
    for n in _own_nodes(fi.node):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tgts:
                if isinstance(t, ast.Subscript):
                    r = _unparse(t.value)
                    if recv_ok(r):
                        out.append((r, n.lineno, t.slice))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS:
            r = _unparse(n.func.value)
            if recv_ok(r):
                out.append((r, n.lineno, n.args[0] if n.args else None))
    return out


def _memo_guarded(fi: FuncInfo, recv: str) -> bool:
    """A memo cache: the function tests `k in X` or calls X.get(k) before it stores into X."""
    for n in _own_nodes(fi.node):
        if isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops) and \
                any(_unparse(c) == recv for c in n.comparators):
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and \
                _unparse(n.func.value) == recv:
            return True
    return False


def _reads(fi: FuncInfo, recv: str) -> int | None:
    """Line where the function reads recv other than as the receiver of a store or a mutator call."""
    parent = _parents(fi)
    for n in _own_nodes(fi.node):
        if _unparse(n) != recv or not isinstance(n, (ast.Name, ast.Attribute)) or \
                not isinstance(getattr(n, "ctx", None), ast.Load):
            continue
        p = parent.get(n)
        if isinstance(p, ast.Subscript) and isinstance(p.ctx, ast.Store):
            continue
        if isinstance(p, ast.Attribute) and p.attr in MUTATORS:
            continue
        if isinstance(p, ast.AugAssign):
            continue
        return n.lineno
    return None


def _env_classes(pf: PyFile) -> set[str]:
    out = set()
    if pf.tree is None:
        return out
    for n in ast.walk(pf.tree):
        if isinstance(n, ast.ClassDef) and any(ENV_BASE.search(_unparse(b)) for b in n.bases):
            out.add(n.name)
    return out


def _self_attr_writes(fi: FuncInfo) -> list[tuple[str, int]]:
    out = []
    for n in _own_nodes(fi.node):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tgts:
                base = t.value if isinstance(t, ast.Subscript) else t
                if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "self":
                    out.append((base.attr, n.lineno))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS:
            b = n.func.value
            if isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name) and b.value.id == "self":
                out.append((b.attr, n.lineno))
    return out


def _self_attr_reads(fi: FuncInfo, attr: str) -> int | None:
    parent = _parents(fi)
    for n in _own_nodes(fi.node):
        if isinstance(n, ast.Attribute) and n.attr == attr and isinstance(n.value, ast.Name) and \
                n.value.id == "self" and isinstance(n.ctx, ast.Load):
            p = parent.get(n)
            if isinstance(p, ast.Subscript) and isinstance(p.ctx, ast.Store):
                continue
            if isinstance(p, ast.Attribute) and p.attr in MUTATORS:
                continue
            if isinstance(p, ast.AugAssign):
                continue
            return n.lineno
    return None


def _fixed_tmp_writes(pf: PyFile, fi: FuncInfo) -> list[int]:
    out = []
    for n in _own_nodes(fi.node):
        if not isinstance(n, ast.Call):
            continue
        cn = _callee(n)
        path = mode = None
        if cn == "open" and n.args:
            path = n.args[0]
            mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
            mode = mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else ""
            if not re.search(r"[wa+x]", mode):
                continue
        elif cn in ("write_text", "write_bytes") and isinstance(n.func, ast.Attribute):
            b = n.func.value
            if isinstance(b, ast.Call) and _callee(b) in ("Path", "PurePath") and b.args:
                path = b.args[0]
        if path is None:
            continue
        lit = path.value if isinstance(path, ast.Constant) and isinstance(path.value, str) else \
            pf.consts.get(path.id) if isinstance(path, ast.Name) else None
        if lit and TMP_LIT.match(lit):
            out.append(n.lineno)
    return out


# 0.5.0 fix wave (check_2.md): a dict on self keyed by a per-rollout key, and an append only /tmp
# log that nothing reads back, are not shared rollout state.
PER_ROLLOUT_KEY = re.compile(r"(trajectory_id|rollout_id|episode_id|request_id|uuid|(?<![\w.])id\(\s*state\b)")


def _per_rollout_key(fi: FuncInfo, key) -> bool:
    if key is None:
        return False
    if PER_ROLLOUT_KEY.search(_unparse(key)):
        return True
    if isinstance(key, ast.Name):
        for n in _own_nodes(fi.node):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, ast.Name) and t.id == key.id for t in tgts) and \
                        PER_ROLLOUT_KEY.search(_unparse(n.value)):
                    return True
    return False


def _keyed_by_rollout(meths, attr: str) -> bool:
    """True when every write to self.<attr> in these methods is an item keyed by a per-rollout id."""
    seen = False
    for _, fi in meths:
        for n in _own_nodes(fi.node):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tgts:
                    base = t.value if isinstance(t, ast.Subscript) else t
                    if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and \
                            base.value.id == "self" and base.attr == attr:
                        if not isinstance(t, ast.Subscript) or not _per_rollout_key(fi, t.slice):
                            return False
                        seen = True
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS:
                b = n.func.value
                if isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name) and b.value.id == "self" \
                        and b.attr == attr:
                    if n.func.attr not in ("pop", "setdefault") or not n.args or not _per_rollout_key(fi, n.args[0]):
                        return False
                    seen = True
    return seen


def _append_log_never_read(pkg: Package, pf: PyFile, fi: FuncInfo, ln: int) -> bool:
    """True when the /tmp write on line ln is an append only open and no code reads that path back."""
    for n in _own_nodes(fi.node):
        if not (isinstance(n, ast.Call) and n.lineno == ln and _callee(n) == "open" and n.args):
            continue
        mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
        mode = mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else ""
        if "a" not in mode or "+" in mode:
            return False
        path = n.args[0]
        names = []
        if isinstance(path, ast.Constant) and isinstance(path.value, str):
            names.append(path.value)
        elif isinstance(path, ast.Name):
            names.append(path.id)
            if isinstance(pf.consts.get(path.id), str):
                names.append(pf.consts[path.id])
        if not names:
            return False
        pat = re.compile("|".join(re.escape(x) if x.startswith("/") else rf"\b{re.escape(x)}\b" for x in names))
        for q in pkg.py_all():
            for i, line in enumerate(q.text.splitlines(), 1):
                if not pat.search(line) or (q.path == pf.path and i == ln):
                    continue
                if re.match(rf"\s*{re.escape(names[0])}\s*(:[^=]*)?=", line):
                    continue  # the constant that names the path
                if re.search(r"open\(.*['\"]a['\"]", line):
                    continue  # another append
                if re.search(r"(open\(|read|load|Path\(|exists|cat |tail |shutil|copy|unlink|remove)", line):
                    return False
        return True
    return False


@rule("module-state-shared-across-rollouts", "broken", "medium",
      "Rollout or scoring code keeps per-episode state in a module global, on the env object or at a fixed /tmp path, so concurrent rollouts read each other's state.")
def r_shared_state(pkg: Package):
    if not any(re.search(r"(verifiers|vf\.)", pf.text) for pf in pkg.py_all()):
        return []
    rollout = _rollout_funcs(pkg)
    if not rollout:
        return []
    out = []
    # (a) module level mutables
    by_file: dict[str, list] = {}
    for pf, fi in rollout:
        by_file.setdefault(pf.path, []).append((pf, fi))
    for path, items in by_file.items():
        pf = items[0][0]
        mods = {k: v for k, v in _module_mutables(pf).items() if not CACHEISH.search(k)}
        if not mods:
            continue
        for pf_, fi in items:
            for recv, ln, key in _mutations(fi, lambda r: r in mods):
                if _memo_guarded(fi, recv):
                    continue
                if key is not None and re.search(r"(uuid|id\(|rollout_id|trajectory_id|example_id|state\[)",
                                                 _unparse(key)) and "uuid" in _unparse(key) + fi.src:
                    continue
                rd = next(((g, l) for _, g in items for l in [_reads(g, recv)] if l), None)
                if rd:
                    out.append(("global", recv, pf, fi, ln, rd))
                    break
    # (b) attributes of the env object written during a rollout and read during one
    for pf in {pf.path: pf for pf, _ in rollout}.values():
        envc = _env_classes(pf)
        if not envc:
            continue
        meths = [(p, f) for p, f in rollout if p.path == pf.path and f.cls in envc and f.name != "__init__"]
        done = set()
        for _, fi in meths:
            if fi.name in ("reset",):
                continue
            for attr, ln in _self_attr_writes(fi):
                if attr in done or CACHEISH.search(attr) or attr.startswith("__"):
                    continue
                rd = None
                for _, g in meths:
                    if g is fi and fi.name != "setup_state":
                        l = _self_attr_reads(g, attr)
                        if l and l > ln:
                            rd = (g, l)
                    elif g is not fi:
                        l = _self_attr_reads(g, attr)
                        if l:
                            rd = (g, l)
                    if rd:
                        break
                if rd and _keyed_by_rollout(meths, attr):
                    done.add(attr)
                    rd = None
                if rd:
                    done.add(attr)
                    out.append(("attr", "self." + attr, pf, fi, ln, rd))
    # (c) fixed /tmp paths written on the host by rollout or scoring code
    for pf, fi in rollout:
        for ln in _fixed_tmp_writes(pf, fi):
            if _append_log_never_read(pkg, pf, fi, ln):
                continue
            out.append(("tmp", "", pf, fi, ln, None))
            break
    if not out:
        return []
    res = []
    for form, recv, pf, fi, ln, rd in out[:3]:
        if form == "global":
            g, l = rd
            msg = (f"{fi.name} changes the module level {recv} and {g.name} reads it back, so the rollouts that "
                   "run at the same time on this env share one copy and see each other's state.")
            evid = [ev(pf, ln), ev(pf, l)]
        elif form == "attr":
            g, l = rd
            msg = (f"{fi.name} stores {recv} on the env object and {g.name} reads it, but one env object serves "
                   "every concurrent rollout, so one rollout can read another's value.")
            evid = [ev(pf, ln), ev(pf, l)]
        else:
            msg = (f"{fi.name} writes a fixed /tmp path on the host, so two rollouts scored at the same time can "
                   "overwrite each other's file.")
            evid = [ev(pf, ln)]
        res.append(F("module-state-shared-across-rollouts", msg, evid, confidence="review"))
    return res


# ---------------------------------------------------------------------------
# 3. state-diff-ignored-fields
# Reader items reader items. Main papers consensus w:78b010e3 (Agent-Diff), arxiv:2602.06820
# (per-column ignore or fuzzy policies let a wrong value pass).
# A state compare on the reward path that skips a literal list of fields. When a task's text or gold
# actions name a skipped field, the task changes that field and the grader never checks it (hack).
# Otherwise the skip list is reported as hygiene.
# ---------------------------------------------------------------------------

IGNORE_NAME = re.compile(r"(ignor|exclud|skip|exempt|fuzzy|volatile|untracked)\w*_?(field|key|column|col|attr|path)s?|"
                         r"(field|key|column|col|attr)s?_?(to_)?(ignor|exclud|skip|exempt)", re.I)
STATEISH = re.compile(r"(state|db|database|world|snapshot|record|row|table|diff)", re.I)


def _ignore_lists(pf: PyFile) -> list[tuple[str, int, list[str]]]:
    out = []
    if pf.tree is None or not IGNORE_NAME.search(pf.text):
        return out
    for n in ast.walk(pf.tree):
        name = val = None
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            name = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
            val = n.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            name, val = n.target.id, n.value
        elif isinstance(n, ast.keyword) and n.arg:
            name, val = n.arg, n.value
        if not name or val is None or not IGNORE_NAME.search(name):
            continue
        if isinstance(val, ast.Call) and _callee(val) in ("set", "frozenset", "list", "tuple") and val.args:
            val = val.args[0]
        if isinstance(val, (ast.List, ast.Tuple, ast.Set)):
            fields = [e.value for e in val.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if fields:
                out.append((name, getattr(n, "lineno", getattr(val, "lineno", 1)), fields))
    return out


TRIVIAL_FIELD = re.compile(r"^(id|_id|uuid|created_at|updated_at|modified_at|timestamp|ts|time|date_created|"
                           r"last_modified|etag|version|_version|_rev|rev|row_id|rowid|seq|nonce)$", re.I)


@rule("state-diff-ignored-fields", "hack", "medium",
      "A state-diff grader skips a list of fields, and a task changes one of them, so a wrong value in that field still passes.")
def r_state_diff_ignored(pkg: Package):
    out = []
    graders = {(pf.path, fi.start): (pf, fi) for pf, fi, _ in _paid_funcs(pkg)}
    for pf, fi in _rollout_funcs(pkg):
        if fi.name in _paid(pkg) or re.search(r"(diff|compare|grade|verify|check|evaluat|score|reward)", fi.name, re.I):
            graders.setdefault((pf.path, fi.start), (pf, fi))
    if not graders:
        return []
    data = None
    gfiles = {g_pf.path: g_pf for g_pf, g in graders.values() if STATEISH.search(g.src)}
    for pf in gfiles.values():
        for name, ln, fields in _ignore_lists(pf):
            users = [(g_pf, g) for g_pf, g in graders.values()
                     if g_pf.path == pf.path and (name in names_in(g) or g.start <= ln <= g.end)]
            if not users:
                continue
            g_pf, g = users[0]
            if not STATEISH.search(g.src):
                continue
            real = [f for f in fields if not TRIVIAL_FIELD.match(f)]
            if not real:
                continue
            if data is None:
                data = "\n".join(t for p, t in pkg.files.items() if p.endswith((".json", ".jsonl", ".md", ".yaml",
                                                                                  ".yml", ".csv")))
            named = [f for f in real if len(f) > 2 and re.search(r"['\"]" + re.escape(f) + r"['\"]\s*:", data)]
            if named:
                out.append(F("state-diff-ignored-fields",
                             f"{g.name} skips the fields {', '.join(real[:4])} when it compares states, and the "
                             f"task data sets {', '.join(named[:3])}, so a wrong value there still passes.",
                             [ev(pf, ln), ev(g_pf, g.start)], confidence="review"))
            else:
                out.append(F("state-diff-ignored-fields",
                             f"{g.name} skips the fields {', '.join(real[:4])} when it compares states, so a "
                             "change to them is never checked. Confirm no task needs them.",
                             [ev(pf, ln), ev(g_pf, g.start)], confidence="review", kind="hygiene", severity="low"))
    return out[:3]


def names_in(fi: FuncInfo) -> set[str]:
    out = set()
    for n in _own_nodes(fi.node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


# ---------------------------------------------------------------------------
# 4. env-init-state-mismatch
# Reader item reader items. Main paper arxiv:2610.27781 (worked trace in App A.7: the task names a record
# the start state does not hold, so no policy can do it).
# For a dataset row that carries both an instruction and its own start state, and for tasks next to a
# shipped database file, an id in the instruction must appear in the start state when that start
# state holds ids of the same shape.
# ---------------------------------------------------------------------------

ID_RE = re.compile(r"(?<![\w#/.-])#?([A-Z]{1,5}[-_]?\d{4,}|[a-z]+(?:_[a-z]+)+_\d{3,})(?![\w-])")
INSTR_KEYS = ("instruction", "instructions", "task_instruction", "user_instruction", "goal", "task", "query",
              "question", "prompt", "user_prompt", "scenario", "user_scenario", "description")
STATE_KEYS = ("initial_state", "initial_db", "init_state", "start_state", "db", "database", "world", "world_state",
              "fixtures", "environment_state", "initial_data", "seed_data", "state")
DB_FILE = re.compile(r"(^|/)(db|database|initial_state|init_state|initial_db|fixtures?|seed|world|data_?base|"
                     r"[\w-]*_db)\.json$", re.I)


def _shape(tok: str) -> str:
    tok = tok.lstrip("#")
    m = re.match(r"([A-Za-z_]*?)[-_]?(\d+)$", tok)
    if not m:
        return ""
    pre = m.group(1)
    if "_" in pre and pre.islower():
        pre = "lower_name"
    return f"{pre}:{len(m.group(2))}"


def _instr_text(row) -> str:
    parts = []
    if isinstance(row, dict):
        for k in INSTR_KEYS:
            v = row.get(k)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                parts += [x for x in v.values() if isinstance(x, str)]
            elif isinstance(v, list):
                for m in v:
                    if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                        parts.append(m["content"])
    return "\n".join(parts)


def _state_blob(row):
    if not isinstance(row, dict):
        return None
    for k in STATE_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v[:1] in "{[":
            try:
                v = json.loads(v)
            except Exception:
                continue
        if isinstance(v, (dict, list)) and v:
            return v
    info = row.get("info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except Exception:
            info = None
    if isinstance(info, dict):
        return _state_blob(info)
    return None


def _rows(text: str, path: str, limit: int = 400):
    if path.endswith(".jsonl"):
        for i, ln in enumerate(text.splitlines()):
            if i >= limit:
                break
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                yield i + 1, json.loads(ln)
            except Exception:
                continue
    else:
        try:
            obj = json.loads(text)
        except Exception:
            return
        if isinstance(obj, dict):
            for k in ("tasks", "data", "examples", "rows", "items"):
                if isinstance(obj.get(k), list):
                    obj = obj[k]
                    break
        if isinstance(obj, list):
            for i, r in enumerate(obj[:limit]):
                yield 1, r


def _missing_ids(instr: str, blob_text: str) -> list[str]:
    toks = {m.group(1) for m in ID_RE.finditer(instr)}
    if not toks:
        return []
    shapes: dict[str, set[str]] = {}
    for m in ID_RE.finditer(blob_text):
        shapes.setdefault(_shape(m.group(1)), set()).add(m.group(1))
    out = []
    for t in sorted(toks):
        s = _shape(t)
        if not s or len(shapes.get(s, ())) < 3:
            continue
        if t not in blob_text:
            # the instruction asks to create it, or says it does not exist
            ctx = instr[max(0, instr.find(t) - 80): instr.find(t) + 80].lower()
            if re.search(r"(creat|new |add |register|open a|does ?n.t exist|not exist|no such|invalid|fake|unknown|"
                         r"non-?existent|made[- ]up|isn.t (in|on)|cannot find|can.t find)", ctx):
                continue
            out.append(t)
    return out


@rule("env-init-state-mismatch", "broken", "high",
      "A task's instruction names a record id that its shipped start state does not contain, so no policy can do the task.")
def r_init_state_mismatch(pkg: Package):
    data = [(p, t) for p, t in pkg.files.items() if p.endswith((".json", ".jsonl")) and p not in pkg.truncated]
    if not data:
        return []
    out = []
    # (a) rows that carry their own start state
    for p, t in data:
        if not re.search(r"(initial_state|initial_db|init_state|start_state|world_state|\"db\"|fixtures|seed_data)", t):
            continue
        for ln, row in _rows(t, p):
            blob = _state_blob(row)
            if blob is None:
                continue
            instr = _instr_text(row) or _instr_text(row.get("info") if isinstance(row, dict) and
                                                    isinstance(row.get("info"), dict) else None)
            if not instr:
                continue
            miss = _missing_ids(instr, json.dumps(blob))
            if miss:
                out.append((p, ln, miss))
            if len(out) >= 3:
                break
    # (b) task files next to a shipped database file
    dbs = [(p, t) for p, t in data if DB_FILE.search(p)]
    if dbs and len(out) < 3:
        dbtext = "\n".join(t for _, t in dbs)
        for p, t in data:
            if DB_FILE.search(p) or not re.search(r"(task|instruction|scenario)", p + t[:2000], re.I):
                continue
            for ln, row in _rows(t, p):
                if _state_blob(row) is not None:
                    continue
                instr = _instr_text(row)
                if not instr:
                    continue
                miss = _missing_ids(instr, dbtext)
                if miss:
                    out.append((p, ln, miss))
                if len(out) >= 3:
                    break
    if not out:
        return []
    ids = sorted({i for _, _, m in out for i in m})[:4]
    return [F("env-init-state-mismatch",
              f"A task names {', '.join(ids)}, which its shipped start state does not contain although that state "
              "lists other ids of the same kind, so the task cannot be done as written.",
              [ev(p, ln) for p, ln, _ in out], confidence="review")]


# ---------------------------------------------------------------------------
# 5. tool-count-above-api-cap
# Reader item reader items. Main paper or:5bp4vclfl4 (a model hard zeros at 468 actions against a 128 tool
# cap). Some chat APIs reject a request with more than 128 tools, so every rollout of that model fails.
# ---------------------------------------------------------------------------

TOOL_CAP = 128


@rule("tool-count-above-api-cap", "hygiene", "medium",
      "The env gives the model more than 128 tools, which some chat APIs reject, so those models score zero.")
def r_tool_cap(pkg: Package):
    if not any("tool" in pf.text for pf in pkg.py_all()):
        return []
    text = pkg.all_py_text()
    if re.search(r"state\[\s*['\"](tool_defs|oai_tools|tools)['\"]\s*\]\s*=", text):
        return []  # tools are filtered per task
    out = []
    lists: dict[str, tuple[PyFile, int, int]] = {}
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and \
                    isinstance(n.value, (ast.List, ast.Tuple)) and len(n.value.elts) > TOOL_CAP:
                if sum(isinstance(e, (ast.Name, ast.Attribute)) for e in n.value.elts) > TOOL_CAP:
                    lists[n.targets[0].id] = (pf, n.lineno, len(n.value.elts))
            elif isinstance(n, ast.keyword) and n.arg == "tools" and isinstance(n.value, (ast.List, ast.Tuple)) and \
                    len(n.value.elts) > TOOL_CAP:
                out.append((pf, n.value.lineno, len(n.value.elts)))
    for nm, (pf, ln, k) in lists.items():
        if re.search(r"(tools\s*=\s*" + re.escape(nm) + r"\b|for \w+ in " + re.escape(nm) + r"\b[^\n]*:\s*\n[^\n]*add_tool|"
                     r"add_tool\([^)]*" + re.escape(nm) + r")", text):
            out.append((pf, ln, k))
    tools = _tools(pkg)
    if not out and len(tools) > TOOL_CAP:
        decorated = [(pf, fi) for pf, fi in all_funcs(pkg) if fi.name in tools]
        if len(decorated) > TOOL_CAP:
            out.append((decorated[0][0], decorated[0][1].start, len(decorated)))
    # JSON tool specs shipped with the env
    if not out:
        for p, t in pkg.files.items():
            if not p.endswith(".json") or "tool" not in p.lower() or p in pkg.truncated:
                continue
            try:
                obj = json.loads(t)
            except Exception:
                continue
            if isinstance(obj, dict):
                obj = obj.get("tools") if isinstance(obj.get("tools"), list) else None
            if isinstance(obj, list) and len(obj) > TOOL_CAP and \
                    sum(isinstance(x, dict) and ("name" in x or "function" in x) for x in obj) > TOOL_CAP:
                out.append((p, 1, len(obj)))
                break
    if not out:
        return []
    pf, ln, k = max(out, key=lambda x: x[2])
    return [F("tool-count-above-api-cap",
              f"The env gives the model {k} tools, more than the 128 that some chat APIs accept, so those models "
              "fail every rollout.", [ev(pf, ln)], confidence="review")]


# ---------------------------------------------------------------------------
# 6. tool-contract-conformance
# Reader item reader items. Main paper consensus w:ffe70396 (tau2 refuel_data repair: 1,120 passes to 0).
# Three forms: (a) a write tool that returns a success message but changes no state, (b) a paid
# reward that pays on a success word in a tool message instead of the stored state, (c) a tool with
# a commented out guard that raises or returns an error.
# ---------------------------------------------------------------------------

WRITE_TOOL = re.compile(r"^_?(update|create|delete|remove|set|add|cancel|book|modify|write|send|insert|transfer|"
                        r"refund|change|edit|assign|move|post|submit_order|place|reserve|exchange|return_)\w*", re.I)
SUCCESS_WORD = re.compile(r"\b(success(fully)?|updated|created|deleted|removed|cancel+ed|booked|sent|saved|"
                          r"done|completed|added|changed|modified|transferred|refunded|reserved)\b", re.I)
SIDE_EFFECT_CALL = re.compile(r"^(append|extend|update|add|insert|pop|remove|discard|clear|setdefault|write\w*|"
                              r"execute\w*|commit|save\w*|dump\w*|put|post|patch|delete|send\w*|request|run\w*|"
                              r"call\w*|create\w*|set\w*|apply\w*|modify\w*|exec\w*|upsert|replace|move|rename|"
                              r"\w*_tool|invoke|submit\w*|bulk\w*|__setitem__|sleep)$", re.I)


def _phantom(fi: FuncInfo) -> int | None:
    own = list(_own_nodes(fi.node))
    rets = [n for n in own if isinstance(n, ast.Return) and n.value is not None]
    if not rets:
        return None
    succ = None
    for r in rets:
        t = _const_text(r.value) if not isinstance(r.value, ast.Dict) else _unparse(r.value)
        if t and SUCCESS_WORD.search(t) and not re.search(r"(error|fail|not |invalid)", t, re.I):
            succ = r.lineno
            break
    if succ is None:
        return None
    for n in own:
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, (ast.Subscript, ast.Attribute)) for t in tgts):
                return None
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.Delete, ast.Await, ast.Yield)):
            return None
        if isinstance(n, ast.Call):
            cn = _callee(n)
            if SIDE_EFFECT_CALL.match(cn) or not (cn in ("str", "int", "float", "len", "format", "join", "get",
                                                         "dumps", "isinstance", "lower", "upper", "strip")):
                return None
    return succ


TOOL_ROLE = re.compile(r"""(\[\s*['"]role['"]\s*\]|\.get\(\s*['"]role['"]\s*\)|\.role)\s*==\s*['"]tool['"]""")


def _pays_on_tool_message(fi: FuncInfo) -> int | None:
    if not TOOL_ROLE.search(fi.src):
        return None
    for n in _own_nodes(fi.node):
        test = n.test if isinstance(n, (ast.If, ast.IfExp)) else None
        if test is None:
            continue
        for c in ast.walk(test):
            if isinstance(c, ast.Compare) and any(isinstance(o, ast.In) for o in c.ops) and \
                    isinstance(c.left, ast.Constant) and isinstance(c.left.value, str) and \
                    SUCCESS_WORD.search(c.left.value):
                body = n.body if isinstance(n, ast.If) else [ast.Return(value=n.body)]
                if _positive_return(body) or isinstance(n, ast.IfExp):
                    return n.lineno
    return None


GUARD_COMMENT = re.compile(r"^\s*#\s*(if\s+[^#\n]+:\s*$|raise\s+\w+|return\s+[\"'f]?[^\n]*(error|invalid|not allowed|"
                           r"denied|fail)|assert\s+)", re.I)


def _commented_guard(pf: PyFile, fi: FuncInfo) -> int | None:
    lines = pf.lines[fi.start - 1: fi.end]
    for i, ln in enumerate(lines):
        m = GUARD_COMMENT.match(ln)
        if not m:
            continue
        if ln.lstrip("# \t").startswith("if"):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not re.match(r"^\s*#\s*(raise|return)\b", nxt):
                continue
        return fi.start + i
    return None


@rule("tool-contract-conformance", "hygiene", "medium",
      "A tool says it changed state when it did not, a guard in a tool is commented out, or the reward pays on a tool's success message instead of the stored state.")
def r_tool_contract(pkg: Package):
    tools = _tools(pkg)
    paid = _paid(pkg)
    if not tools and not paid:
        return []
    out = []
    for pf, fi in all_funcs(pkg):
        if fi.name not in tools or not _shipped(pkg, pf.path):
            continue
        if WRITE_TOOL.match(fi.name):
            ln = _phantom(fi)
            if ln:
                out.append(F("tool-contract-conformance",
                             f"The tool {fi.name} returns a success message but changes no state, so a grader "
                             "that reads the state scores correct calls as failures.",
                             [ev(pf, ln)], confidence="review"))
                continue
        ln = _commented_guard(pf, fi)
        if ln:
            out.append(F("tool-contract-conformance",
                         f"The tool {fi.name} has a commented out check that would reject a bad call, so calls "
                         "the task rules forbid still go through.", [ev(pf, ln)], confidence="review"))
    funcs = _funcs_by_name(pkg)
    for nm in paid:
        if nm not in funcs:
            continue
        pf, fi = funcs[nm]
        ln = _pays_on_tool_message(fi)
        if ln and not _reads_gold_or_outcome(fi):
            out.append(F("tool-contract-conformance",
                         f"{nm} pays when a tool message contains a success word, not when the stored state is "
                         "right, so a tool reply that only claims success earns the reward.",
                         [ev(pf, ln)], confidence="review", kind="hack"))
    return out[:4]


# ---------------------------------------------------------------------------
# 7. observation-truncation-silent
# Reader items reader items. Main paper exfwxwpz1p (a verifier that saw only tool output approved
# wrong answers 90 of 90 times). A tool or env response cuts its result list or text to a fixed
# length and says nothing, so the policy takes a partial result for the whole one.
# ---------------------------------------------------------------------------

NOTICE = re.compile(r"(truncat|omitted|more (result|item|row|line|match|file)|\bof \{|out of|showing|remaining|"
                    r"\btotal\b|\.\.\.|…|first \{|top \{|limit(ed)? to|cut off|only the first|additional|"
                    r"more\b.{0,20}(not shown|available)|len\(|has_more|next_page|page\b|cursor|offset)", re.I)


def _slice_return(fi: FuncInfo) -> int | None:
    own = list(_own_nodes(fi.node))
    sliced: dict[str, int] = {}
    def has_slice(e):
        for x in ast.walk(e):
            if isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Slice) and x.slice.upper is not None and \
                    (x.slice.lower is None or _num(x.slice.lower) == 0) and x.slice.step is None:
                up = x.slice.upper
                if isinstance(up, ast.UnaryOp):
                    continue  # x[:-1] drops a tail marker, not a cap
                if _num(up) is not None and _num(up) < 3:
                    continue
                if re.match(r"^(str|repr)\(\w*(e|err|exc|error|ex)\)$", _unparse(x.value)):
                    continue  # a cut exception text inside an error message
                return x.lineno
        return None
    for names, val in _assigned(own):
        ln = has_slice(val)
        if ln:
            for nm in names:
                sliced[nm] = ln
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None:
            ln = has_slice(n.value)
            if ln:
                return ln
            for nm in _loads(n.value):
                if nm in sliced:
                    return sliced[nm]
    return None


@rule("observation-truncation-silent", "hygiene", "low",
      "A tool cuts its output to a fixed length and does not say so, so the policy reads a partial result as the whole one.")
def r_trunc_silent(pkg: Package):
    tools = _tools(pkg)
    if not tools and not any("env_response" in pf.text for pf in pkg.py_all()):
        return []
    out = []
    for pf, fi in all_funcs(pkg):
        if not (fi.name in tools or fi.name == "env_response") or not _shipped(pkg, pf.path):
            continue
        ln = _slice_return(fi)
        if not ln:
            continue
        body = fi.src + "\n" + "\n".join(pf.consts.get(n, "") for n in _loads(fi.node))
        if NOTICE.search(body):
            continue
        out.append(ev(pf, ln))
    if not out:
        return []
    return [F("observation-truncation-silent",
              "A tool returns a sliced list or text with no note that items were cut, so the policy can treat a "
              "partial result as complete.", out, confidence="review")]


# ---------------------------------------------------------------------------
# 8. readonly-sql-guard-false-reject
# Main paper arxiv:2610.12274 (3,230 of 3,989 failed executions came from
# comment prefixed SQL). A guard that a query must start with SELECT, with no comment strip, rejects
# valid read queries that begin with a comment (and with WITH, when that is not allowed too).
# ---------------------------------------------------------------------------

SQL_PARAM = re.compile(r"^(query|sql|sql_query|statement|stmt|q|command|code)$", re.I)
SQL_FN = re.compile(r"(sql|query|db|database|select|execute|run)", re.I)
GUARD = re.compile(
    r"(\.(upper|lower)\(\)\s*\.\s*(l?strip)\(\)\s*\.\s*startswith\(|\.(l?strip)\(\)\s*\.\s*(upper|lower)\(\)\s*\."
    r"\s*startswith\(|\.(upper|lower)\(\)\s*\.\s*startswith\(|startswith\(\s*\(?\s*['\"](?i:select)|"
    r"re\.(match|search)\(\s*r?['\"]\^?(\\s\*)?\(?(\?i\))?(?i:select)|split\(\)\s*\[\s*0\s*\]\s*\.\s*(upper|lower)\(\))")
COMMENT_STRIP = re.compile(r"(--|/\*|sqlparse|sqlglot|strip_comments|comment|mode=ro|read_only\s*=\s*True|"
                           r"PRAGMA\s+query_only|readonly)", re.I)


@rule("readonly-sql-guard-false-reject", "broken", "medium",
      "A SQL tool only accepts queries that start with SELECT and does not strip comments, so valid read queries that begin with a comment are rejected.")
def r_sql_guard(pkg: Package):
    if not re.search(r"(?i:select)", pkg.all_py_text()):
        return []
    tools = _tools(pkg)
    out = []
    for pf, fi in all_funcs(pkg):
        if not any(SQL_PARAM.match(p) for p in fi.params):
            continue
        if fi.name not in tools and not (SQL_FN.search(fi.name) and re.search(r"(execute|cursor|sqlite|duckdb|"
                                                                            r"psycopg|connect)", fi.src)):
            continue
        if fi.name not in tools and not re.search(r"(tool|Tool)", pf.text):
            continue
        m = GUARD.search(fi.src)
        if not m or not re.search(r"(?i:['\"][^'\"\n]{0,8}select)", fi.src):
            continue
        if COMMENT_STRIP.search(fi.src):
            continue
        tail = fi.src[m.start(): m.start() + 400]
        if not re.search(r"(raise|return)", tail):
            continue
        allows_with = bool(re.search(r"(?i:['\"]with\b)", fi.src))
        out.append((pf, fi, fi.start + fi.src.count("\n", 0, m.start()), allows_with))
    if not out:
        return []
    pf, fi, ln, w = out[0]
    extra = "" if w else " and queries that start with WITH"
    return [F("readonly-sql-guard-false-reject",
              f"{fi.name} rejects any query that does not start with SELECT and strips no comments, so a valid "
              f"read query that begins with a comment{extra} gets an error.",
              [ev(p, l) for p, _, l, _ in out], confidence="review")]


# ---------------------------------------------------------------------------
# 9. prompt-wording-raises-cheating
# Reader items reader items. Main papers consensus w:892040ce (exploit rate 0% to 38-80% from
# one prompt sentence), arxiv:2611.23305 (Repo2Run cheating 13 of 16 to 0 of 16 when the emphasis was
# removed). The prompt stresses making the tests pass or succeeding at any cost, or calls the task
# impossible, and has no sentence that the tests must not be changed or that the spec wins.
# Puzzle and CTF words are left out: in puzzle and CTF envs they describe the task itself.
# ---------------------------------------------------------------------------

PRESSURE = re.compile(
    r"(make (sure (that )?)?(all )?(of )?(the )?(unit |failing )?tests? (all )?pass|get (all )?(of )?(the )?(failing )?"
    r"tests? to pass|ensure (that )?(all )?(of )?(the )?tests? (all )?pass|all (of )?(the )?tests? must pass|"
    r"must pass all (of )?(the )?tests|until (all )?(of )?(the )?tests? pass|you must succeed|at all costs|"
    r"by any means( necessary)?|whatever it takes|failure is not an option|"
    r"(this|the) task (is|may be|might be|seems) (nearly |almost |practically )?impossible|"
    r"(may|might) (seem|appear) impossible)", re.I)
SPEC_WINS = re.compile(
    r"((do not|don't|never|must not|should not|shouldn't|cannot|can't|may not|not allowed to|not permitted to)\s+"
    r"(\w+\s+){0,3}(modify|edit|change|alter|delete|remove|touch|tamper|rewrite|overwrite|skip|disable)\b[^.\n]{0,40}"
    r"\btest|tests? (may|might|could) be (wrong|incorrect|buggy|broken)|if (a|the|any) tests? (is|are|seems?|looks?) "
    r"(wrong|incorrect|buggy|broken)|spec\w* (takes? precedence|wins|over)|(do not|don't|never) (special[- ]case|"
    r"hard[- ]?code)|tests? (are|is) (read-only|hidden|held out)|without (modifying|changing|editing) (the )?tests)", re.I)
PRESSURE_KEYS = ("tests pass", "test pass", "tests to pass", "tests all pass", "all tests", "all the tests",
                 "all of the tests", "must succeed", "all costs", "any means", "whatever it takes", "not an option",
                 "impossible")


def _pressure_maybe(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in PRESSURE_KEYS)


PROMPT_NAME = re.compile(r"(prompt|system|instruction|template|preamble|task_desc|PROMPT|SYSTEM)", re.I)


def _prompt_texts(pkg: Package):
    """(path, line, text) for agent facing prompt text: Harbor instruction.md and prompt constants."""
    for p, t in pkg.files.items():
        if os.path.basename(p) == "instruction.md":
            yield p, 1, t
    for pf in pkg.py_all():
        if pf.tree is None or not _pressure_maybe(pf.text) or not PRESSURE.search(pf.text):
            continue
        for n in ast.walk(pf.tree):
            name = val = None
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                name, val = n.targets[0].id, n.value
            elif isinstance(n, ast.keyword) and n.arg:
                name, val = n.arg, n.value
            if name and PROMPT_NAME.search(name) and val is not None:
                t = _const_text(val)
                if t:
                    yield pf.path, getattr(val, "lineno", 1), t


@rule("prompt-wording-raises-cheating", "hygiene", "low",
      "The prompt tells the policy to make the tests pass or succeed at any cost and never says the tests must not be changed.")
def r_prompt_wording(pkg: Package):
    hits = []
    for p, ln, t in _prompt_texts(pkg):
        m = PRESSURE.search(t) if _pressure_maybe(t) else None
        if not m:
            continue
        if SPEC_WINS.search(t):
            continue
        hits.append((p, ln + t.count("\n", 0, m.start()) if p.endswith(".md") else ln, m.group(0)))
    if not hits:
        return []
    # one mitigation sentence anywhere in the same task folder or package prompts counts
    roots = {}
    for p, ln, w in hits:
        root = os.path.dirname(p)
        sib = "\n".join(t for q, t in pkg.files.items() if os.path.dirname(q) == root and q.endswith((".md", ".txt")))
        if SPEC_WINS.search(sib):
            continue
        roots.setdefault(root, (p, ln, w))
    if not roots:
        return []
    vals = list(roots.values())
    phrase = vals[0][2]
    return [F("prompt-wording-raises-cheating",
              f"The prompt says \"{phrase}\" and never says the tests must not be changed or that the spec wins, "
              "and wording like this raises test tampering and other cheating.",
              [ev(p, ln) for p, ln, _ in vals], confidence="review")]

"""Judge family rules (envscan v0.4).

Rules about what a judge or reward model sees and how it is set up (the policy's hidden reasoning,
token budgets, unclamped scores, label-free rewards, a judge from the policy's own family), about
Harbor tests that need names the instruction never gives, and about kernel speed graders. Like every
envscan rule these read source text and ASTs only. Every finding is `review`.

Evidence ids in the comments refer to the ranked table in section 9.2 of the literature review
(research/lit-review/rollup/report_parts/09-open-problems-product.md).
"""
from __future__ import annotations

import ast
import os
import re
import sys

from .package import Package, PyFile, FuncInfo, _const_text, parse_py
from .rules import F, all_funcs, called_names, ev, rule
from .rules_grader import _cache, _callee, _grader_funcs, _num, _own_nodes, _reward_weights

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _callers(pkg: Package) -> dict[str, set[str]]:
    c = _cache(pkg)
    if "judge_callers" not in c:
        m: dict[str, set[str]] = {}
        for _, g in all_funcs(pkg):
            for cn in called_names(g):
                m.setdefault(cn, set()).add(g.name)
        c["judge_callers"] = m
    return c["judge_callers"]


def _paid(pkg: Package, name: str, hops: int = 2) -> float:
    """Largest positive rubric weight on a function or on a caller up to `hops` calls up, else 0.
    A function in no rubric counts as not paid."""
    W = _reward_weights(pkg)
    if not W:
        return 0.0
    names, frontier = {name}, {name}
    for _ in range(hops):
        frontier = {c for n in frontier for c in _callers(pkg).get(n, set())} - names
        names |= frontier
    ws = [w for nm in names for w, _ in W.get(nm, [])]
    return max([w for w in ws if w > 0], default=0.0)


def _unparse(n) -> str:
    try:
        return ast.unparse(n)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# judge call sites, shared by rows 21 and 18
# ---------------------------------------------------------------------------

POLICY_PARAM = {"completion", "response", "model_output", "reply", "output", "completion_text", "response_text",
                "model_response", "generated"}
API_CALL = {"create", "acreate", "completion", "acompletion", "generate_content", "agenerate_content", "invoke",
            "ainvoke", "chat", "achat", "generate", "parse"}
API_RECV = re.compile(r"(chat\.completions|completions|responses|messages|client|llm|judge|openai|anthropic|litellm|"
                      r"genai|model|beta\.chat)", re.I)


def _judge_rubrics(pkg: Package) -> list[tuple[PyFile, ast.Call, str | None]]:
    """(file, JudgeRubric(...) call, name it is assigned to or None)."""
    c = _cache(pkg)
    if "judge_rubrics" in c:
        return c["judge_rubrics"]
    out = []
    for pf in pkg.py_all():
        if pf.tree is None or "JudgeRubric" not in pf.text:
            continue
        assigned = {}
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        assigned[id(n.value)] = t.id
                    elif isinstance(t, ast.Attribute):
                        assigned[id(n.value)] = t.attr
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call) and _callee(n) == "JudgeRubric":
                out.append((pf, n, assigned.get(id(n))))
    c["judge_rubrics"] = out
    return out


def _kw(call: ast.Call, name: str):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _value_of(pf: PyFile, e, fi: FuncInfo | None = None, depth: int = 0):
    """Follow a Name to the value assigned to it in the enclosing function or at module level."""
    if not isinstance(e, ast.Name) or depth > 2 or pf.tree is None:
        return e
    scopes = ([fi.node] if fi is not None else []) + [pf.tree]
    for sc in scopes:
        for n in ast.walk(sc):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == e.id for t in n.targets):
                return _value_of(pf, n.value, fi, depth + 1)
    for f2 in pf.funcs:  # a parameter default: def load_environment(judge_model="gpt-5-nano")
        a = f2.node.args
        pos = a.posonlyargs + a.args
        for arg, d in list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + \
                [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]:
            if arg.arg == e.id:
                return d
    return e


BUDGET_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "max_new_tokens", "num_predict",
               "maxOutputTokens")


def _dict_get(d, key):
    if isinstance(d, ast.Dict):
        for k, v in zip(d.keys, d.values):
            if isinstance(k, ast.Constant) and k.value == key:
                return v
    if isinstance(d, ast.Call) and _callee(d) == "dict":
        return _kw(d, key)
    return None


def _budget(pf: PyFile, call: ast.Call, fi: FuncInfo | None, keys_holder=None):
    """(key, value, line) of a literal token budget on a call or in its sampling dict, else None."""
    holders = [call] + ([keys_holder] if keys_holder is not None else [])
    for h in holders:
        for k in BUDGET_KEYS:
            v = _kw(h, k) if isinstance(h, ast.Call) else _dict_get(h, k)
            if v is None and isinstance(h, ast.Call):
                for kk in h.keywords:  # **{"max_tokens": 5} / extra_body={...} / generation_config={...}
                    v = v or _dict_get(_value_of(pf, kk.value, fi), k)
            if v is not None:
                x = _num(_value_of(pf, v, fi))
                if x is not None:
                    return k, int(x), getattr(v, "lineno", call.lineno)
    return None


def _prompt_text(pf: PyFile, fi: FuncInfo | None, call: ast.Call) -> str:
    """Literal text that goes into a judge call: string constants in its arguments, in the function's
    string assignments and in module constants the function refers to."""
    parts = []
    nodes = list(_own_nodes(fi.node)) if fi is not None else list(ast.walk(call))
    for n in nodes:
        if isinstance(n, (ast.Constant, ast.JoinedStr)):
            t = _const_text(n)
            if t and len(t) > 20:
                parts.append(t)
    if fi is not None:
        used = {x.id for x in ast.walk(fi.node) if isinstance(x, ast.Name)} | \
            {x.attr for x in ast.walk(fi.node) if isinstance(x, ast.Attribute)}
        parts += [t for nm, t in pf.consts.items() if nm in used]
    return "\n".join(parts)


def _judge_sites(pkg: Package):
    """Judge calls on the paid reward path: dicts with file, line, func, model, budget, prompt, effort."""
    c = _cache(pkg)
    if "judge_sites" in c:
        return c["judge_sites"]
    out = []
    for pf, call, nm in _judge_rubrics(pkg):
        mdl = _kw(call, "judge_model")
        mv = _value_of(pf, mdl) if mdl is not None else ast.Constant("gpt-4.1-nano")
        model = mv.value if isinstance(mv, ast.Constant) and isinstance(mv.value, str) else None
        sa = _kw(call, "judge_sampling_args")
        sa = _value_of(pf, sa) if sa is not None else None
        jp = _kw(call, "judge_prompt")
        prompt = (_const_text(_value_of(pf, jp)) or "") if jp is not None else ""
        b = None
        for k in BUDGET_KEYS:
            v = _dict_get(sa, k) if sa is not None else None
            x = _num(_value_of(pf, v)) if v is not None else None
            if x is not None:
                b = (k, int(x), v.lineno)
                break
        effort = _dict_get(sa, "reasoning_effort") if sa is not None else None
        out.append(dict(pf=pf, line=call.lineno, func=None, model=model, budget=b, prompt=prompt, kind="JudgeRubric",
                        effort=_const_text(effort) if effort is not None else None, call=call))
    for pf, fi in _grader_funcs(pkg):
        if not re.search(r"(create|completion|generate|invoke|chat)\(", fi.src):
            continue
        for n in _own_nodes(fi.node):
            if not isinstance(n, ast.Call):
                continue
            nm = _callee(n)
            recv = _unparse(n.func.value) if isinstance(n.func, ast.Attribute) else ""
            if nm not in API_CALL or not API_RECV.search(recv + "." + nm):
                continue
            if _kw(n, "messages") is None and _kw(n, "input") is None and _kw(n, "prompt") is None \
                    and _kw(n, "contents") is None and not n.args:
                continue
            mdl = _kw(n, "model")
            mv = _value_of(pf, mdl, fi) if mdl is not None else None
            model = mv.value if isinstance(mv, ast.Constant) and isinstance(mv.value, str) else None
            eff = _kw(n, "reasoning_effort") or _kw(n, "reasoning")
            out.append(dict(pf=pf, line=n.lineno, func=fi, model=model, budget=_budget(pf, n, fi),
                            prompt=_prompt_text(pf, fi, n), kind="call",
                            effort=_unparse(eff) if eff is not None else None, call=n))
    c["judge_sites"] = out
    return out


REASONING_MODEL = re.compile(r"((^|/)o[134](-mini|-pro|-preview)?(-\d{4}-\d\d-\d\d)?$|(^|/)o[34]-mini|gpt-5(?!-chat)(\.\d)?(-mini|-nano|-pro)?|"
                             r"gpt-oss|deepseek-r1|deepseek-reasoner|qwq|r1-distill|magistral|phi-4-reasoning|"
                             r"kimi-k2-thinking|-thinking\b|qwen3-(?!.*instruct)[\w.-]*\d+b(?!.*instruct)|grok-3-mini|grok-4)", re.I)
ASKS_REASON = re.compile(r"(explain|reasoning|justif|rationale|step[ -]by[ -]step|analy[sz]e|critique|think through|"
                         r"before (giving|providing|stating)|first,? (describe|discuss|assess|evaluate|explain))", re.I)
ONLY_VERDICT = re.compile(r"(respond|reply|answer|output|return)\s+(with\s+)?(only|just|nothing but|exactly)|"
                          r"\bonly\s+(the|a|one|with)\s+(single\s+)?(word|letter|number|score|digit|label)|single (word|token|letter)|"
                          r"no (other|additional) (text|words|explanation)|nothing else|do not explain", re.I)
CLAMP_CALL = re.compile(r"^(min|max|clip|clamp\w*|_clamp\w*|bound\w*|normali[sz]e\w*|sigmoid|tanh|round)$")
UNBOUNDED_NUM = re.compile(r"(\\d\+|\\d\*|\[0-9\]\+|\[0-9\]\*|\[\\d\.\]\+|\[0-9\.\]\+|\[\.\\d\]\+|\[\\d,\.\]\+|\\d\{1,\})")


def _reply_names(fi: FuncInfo, starts: set[str]) -> set[str]:
    """Names that hold the judge's reply or something read out of it (regex match, split, json)."""
    taint = set(starts)
    for _ in range(4):
        grew = False
        for n in _own_nodes(fi.node):
            if isinstance(n, ast.Assign):
                v = n.value.value if isinstance(n.value, ast.Await) else n.value
                if any(isinstance(x, ast.Name) and x.id in taint for x in ast.walk(v)):
                    for t in n.targets:
                        for x in ast.walk(t):
                            if isinstance(x, ast.Name) and x.id not in taint:
                                taint.add(x.id)
                                grew = True
        if not grew:
            break
    return taint


def _unclamped_score(fi: FuncInfo):
    """Line of a return that pays a number parsed out of the judge's reply with no upper bound: the
    regex accepts any number of digits (or the score is a JSON field), and no min, max or clip and
    no comparison against a constant limits it. None if not found."""
    own = list(_own_nodes(fi.node))
    starts = set()
    for n in own:
        if isinstance(n, ast.Assign):
            v = n.value.value if isinstance(n.value, ast.Await) else n.value
            if isinstance(v, ast.Call):
                nm = _callee(v)
                recv = _unparse(v.func.value) if isinstance(v.func, ast.Attribute) else ""
                if nm == "judge" or (nm in API_CALL and API_RECV.search(recv + "." + nm)):
                    starts |= {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    if not starts:
        return None
    reply = _reply_names(fi, starts)
    # the regex patterns that read numbers out of the reply
    unbounded_src = False
    for n in own:
        if isinstance(n, ast.Call) and _callee(n) in ("search", "match", "findall", "fullmatch", "finditer") \
                and len(n.args) >= 2 and any(isinstance(x, ast.Name) and x.id in reply for x in ast.walk(n.args[1])):
            pat = _const_text(n.args[0]) or ""
            if UNBOUNDED_NUM.search(pat):
                unbounded_src = True
        if isinstance(n, ast.Call) and _callee(n) in ("loads", "literal_eval") and n.args and \
                any(isinstance(x, ast.Name) and x.id in reply for x in ast.walk(n.args[0])):
            unbounded_src = True
    if not unbounded_src:
        return None
    nums = set()  # names holding float(...)/int(...) of the reply
    for n in own:
        if isinstance(n, ast.Assign):
            v = n.value
            if any(isinstance(c, ast.Call) and _callee(c) in ("float", "int") and
                   any(isinstance(x, ast.Name) and x.id in reply for x in ast.walk(c)) for c in ast.walk(v)):
                if not any(isinstance(c, ast.Call) and CLAMP_CALL.match(_callee(c)) for c in ast.walk(v)):
                    nums |= {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    # any clamp or bound check on those names
    for n in own:
        if isinstance(n, ast.Call) and CLAMP_CALL.match(_callee(n)) and \
                any(isinstance(x, ast.Name) and x.id in nums for a in n.args for x in ast.walk(a)):
            return None
        if isinstance(n, ast.Compare) and any(isinstance(x, ast.Name) and x.id in nums for x in ast.walk(n)) and \
                any(_num(c) is not None for c in [n.left] + n.comparators):
            return None
    for n in own:
        if isinstance(n, ast.Return) and n.value is not None:
            v = n.value
            if any(isinstance(c, ast.Call) and CLAMP_CALL.match(_callee(c)) for c in ast.walk(v)):
                continue
            direct = any(isinstance(c, ast.Call) and _callee(c) in ("float", "int") and
                         any(isinstance(x, ast.Name) and x.id in reply for x in ast.walk(c)) for c in ast.walk(v))
            if direct or any(isinstance(x, ast.Name) and x.id in nums for x in ast.walk(v)):
                return n.lineno
    return None


# ---------------------------------------------------------------------------
# 18. judge-config-hygiene
# Evidence row 18 (section 9.2), arxiv:2607.13707 (a shared max_tokens=5 budget manufactured a
# 32-point judge effect, Sec 3.3, 4.1), inspect_evals BEST_PRACTICES.md (verdict parsing).
# Two forms with a nameable failure: (a) the judge's token budget is too small for the reply the
# judge has to write (a reasoning model, or a prompt that asks for an explanation before the
# verdict), so the verdict is cut off and correct work scores as a failure, and (b) a paid reward
# returns the judge's number with no upper bound, so a judge reply above the scale pays more than
# any correct answer. Unpinned aliases and many criteria per call have no nameable failing rollout
# and are not reported.
# ---------------------------------------------------------------------------


def _budget_hits(pkg: Package):
    out = []
    for d in _judge_sites(pkg):
        b = d["budget"]
        if b is None:
            continue
        key, val, bline = b
        fi = d["func"]
        if fi is not None and not _paid(pkg, fi.name):
            continue
        model = d["model"] or ""
        effort = (d["effort"] or "").lower()
        if REASONING_MODEL.search(model) and val < 1024 and not re.search(r"(minimal|none|low)", effort):
            out.append((d["pf"], bline, f"{key}={val} for the reasoning model {model}, whose hidden reasoning "
                                        "uses up the budget before the verdict"))
        elif val < 64 and ASKS_REASON.search(d["prompt"] or "") and not ONLY_VERDICT.search(d["prompt"] or ""):
            out.append((d["pf"], bline, f"{key}={val} while the judge prompt asks for an explanation before the verdict"))
    return out


@rule("judge-config-hygiene", "broken", "medium",
      "A judge call is set up so its verdict cannot be read or is not bounded: a token budget too small for "
      "the reply it must write, or a paid score taken from the judge's reply with no upper limit.")
def r_judge_config(pkg: Package):
    out = []
    bh = _budget_hits(pkg)
    if bh:
        pf, line, why = bh[0]
        out.append(F("judge-config-hygiene",
                     f"The judge call has {why}, so the reply is cut off before the verdict and a correct answer is "
                     "scored as if the judge said no.", [ev(p, ln) for p, ln, _ in bh], confidence="review"))
    uh = []
    for pf, fi in _grader_funcs(pkg):
        if "judge" not in fi.src and "create(" not in fi.src:
            continue
        ln = _unclamped_score(fi)
        if ln and _paid(pkg, fi.name):
            uh.append((pf, fi, ln))
    if uh:
        pf, fi, ln = uh[0]
        out.append(F("judge-config-hygiene",
                     f"{fi.name} pays the number it reads out of the judge's reply with no upper limit, and the policy's "
                     "reply is part of the judge prompt, so a reply that gets the judge to write a score above the scale "
                     "(for example 100 on a 0 to 10 scale) is paid more than any correct answer.",
                     [ev(p, l) for p, _, l in uh], confidence="review", kind="hack"))
    return out


# ---------------------------------------------------------------------------
# 21. cot-reaches-reward
# Evidence row 21 (section 9.2), url:blog.redwoodresearch.org/p/anthropic-repeatedly-accidentally
# (about 8% of a frontier model's training episodes exposed the CoT to the reward), arxiv:2610.28819
# (the LIFT README warns that an inline think block is scored as part of the answer).
# Fires only when the package itself expects think text in the reply (a prompt that asks for <think>
# tags, or a think parser), a paid reward sends the reply to an LLM judge, and nothing on that path
# strips the think block.
# ---------------------------------------------------------------------------

THINK_PARSER = re.compile(r"^(ThinkParser|MaybeThinkParser)$")
REGEXISH = re.compile(r"(\.\*|\\s|\(\?|\[\^|\\n\)|\.\+)")
STRIP_HELPER = re.compile(r"(think|reason|parse|extract|strip|clean|remove|answer|final|boxed)", re.I)


def _think_expected(pkg: Package):
    """(file, line) where the package asks for <think> text in the reply, else None."""
    c = _cache(pkg)
    if "think_expected" in c:
        return c["think_expected"]
    hit = None
    for pf in pkg.py_all():
        if pf.tree is None or "think" not in pf.text.lower():
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call):
                nm = _callee(n)
                if THINK_PARSER.match(nm) or (nm == "XMLParser" and any(
                        isinstance(x, ast.Constant) and x.value == "think" for a in n.args + [k.value for k in n.keywords]
                        for x in ast.walk(a))):
                    hit = (pf, n.lineno)
                    break
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and "<think>" in n.value \
                    and len(n.value.split()) >= 5 and not REGEXISH.search(n.value):
                hit = (pf, n.lineno)
                break
        if hit:
            break
    c["think_expected"] = hit
    return hit


def _raw_parser(pf: PyFile, call: ast.Call) -> bool:
    """The JudgeRubric hands the judge the raw reply: no parser (verifiers then uses a plain Parser,
    whose parse_answer returns the last message as is) or a plain Parser()."""
    p = _kw(call, "parser")
    if p is None:
        return True
    v = _value_of(pf, p)
    return isinstance(v, ast.Call) and _callee(v) == "Parser" and not v.args and not v.keywords


def _response_slot(pf: PyFile, call: ast.Call) -> bool:
    """The judge prompt has a {response} slot (the default JudgeRubric prompt does)."""
    jp = _kw(call, "judge_prompt")
    if jp is None:
        return True
    t = _const_text(_value_of(pf, jp)) or ""
    return not t or "{response}" in t


def _stripping(e, pkg: Package) -> bool:
    """The expression takes the think block out (or pulls out only the final answer)."""
    for c in ast.walk(e):
        if not isinstance(c, ast.Call):
            continue
        nm = _callee(c)
        lits = " ".join(x.value for x in ast.walk(c) if isinstance(x, ast.Constant) and isinstance(x.value, str))
        if nm in ("split", "rsplit", "partition", "rpartition", "sub", "replace", "find", "rfind", "index") and \
                re.search(r"(think|reason)", lits + " " + _unparse(c), re.I):
            return True
        if nm in ("parse_answer", "parse") or STRIP_HELPER.search(nm):
            return True
    return False


def _raw_reply_names(fi: FuncInfo, pkg: Package) -> set[str]:
    """Names that hold the policy's reply with any think block still in it."""
    taint = {p for p in fi.params if p in POLICY_PARAM}
    if not taint:
        return set()
    for _ in range(4):
        grew = False
        for n in _own_nodes(fi.node):
            tgts, val = [], None
            if isinstance(n, ast.Assign):
                tgts, val = n.targets, n.value
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                tgts, val = [n.target], n.value
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                tgts, val = [n.target], n.iter
            if val is None or not any(isinstance(x, ast.Name) and x.id in taint for x in ast.walk(val)):
                continue
            if _stripping(val, pkg):
                continue
            for t in tgts:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name) and x.id not in taint:
                        taint.add(x.id)
                        grew = True
        if not grew:
            break
    # a stripped copy that is reassigned to the same name (`completion = completion.split("</think>")[-1]`)
    for n in _own_nodes(fi.node):
        if isinstance(n, ast.Assign) and _stripping(n.value, pkg):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    taint.discard(t.id)
    return taint


def _cot_hits(pkg: Package):
    """(file, func, line, how) for paid rewards that send the raw reply to a judge."""
    hits = []
    rubrics = _judge_rubrics(pkg)
    raw_rubric_names = {nm for pf, call, nm in rubrics if nm and _raw_parser(pf, call) and _response_slot(pf, call)}
    any_raw = any(_raw_parser(pf, call) and _response_slot(pf, call) for pf, call, _ in rubrics)
    only_one = len(rubrics) == 1
    # functions placed on a raw JudgeRubric: X.add_reward_func(f) or JudgeRubric(funcs=[f])
    on_raw = set()
    for pf, call, nm in rubrics:
        if _raw_parser(pf, call) and _response_slot(pf, call):
            fl = _kw(call, "funcs")
            if isinstance(fl, (ast.List, ast.Tuple)):
                on_raw |= {x.id for x in fl.elts if isinstance(x, ast.Name)}
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call) and _callee(n) == "add_reward_func" and isinstance(n.func, ast.Attribute) \
                    and n.args and isinstance(n.args[0], ast.Name):
                recv = n.func.value
                rn = recv.id if isinstance(recv, ast.Name) else recv.attr if isinstance(recv, ast.Attribute) else ""
                if rn in raw_rubric_names:
                    on_raw.add(n.args[0].id)
    for pf, fi in _grader_funcs(pkg):
        if "judge" not in fi.src and not any(a in fi.src for a in ("create(", "completion(", "generate")):
            continue
        raw = None
        for n in _own_nodes(fi.node):
            if not isinstance(n, ast.Call):
                continue
            nm = _callee(n)
            f = n.func
            recv = _unparse(f.value) if isinstance(f, ast.Attribute) else ""
            if nm == "judge" and (isinstance(f, ast.Name) or recv in ("self",) or recv in raw_rubric_names):
                # JudgeRubric.judge(prompt, completion, answer, state): the rubric's parser reads the reply
                rubric_raw = fi.name in on_raw or recv in raw_rubric_names or (only_one and any_raw) or \
                    (recv == "self" and any_raw)
                if not rubric_raw:
                    continue
                arg = n.args[1] if len(n.args) > 1 else _kw(n, "completion")
                if arg is None:
                    continue
                names = _raw_reply_names(fi, pkg)
                if isinstance(arg, ast.Name) and arg.id in names:
                    raw = (n.lineno, "JudgeRubric with no think-stripping parser")
                    break
            elif nm in API_CALL and API_RECV.search(recv + "." + nm):
                names = _raw_reply_names(fi, pkg)
                if not names:
                    continue
                # follow prompt and message variables built from the reply
                built = set(names)
                for _ in range(3):
                    for m in _own_nodes(fi.node):
                        if isinstance(m, ast.Assign) and any(isinstance(x, ast.Name) and x.id in built
                                                             for x in ast.walk(m.value)) and not _stripping(m.value, pkg):
                            built |= {x.id for t in m.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
                args = list(n.args) + [k.value for k in n.keywords]
                if any(isinstance(x, ast.Name) and x.id in built for a in args for x in ast.walk(a)):
                    raw = (n.lineno, "a judge call")
                    break
        if raw and _paid(pkg, fi.name):
            hits.append((pf, fi, raw[0], raw[1]))
    return hits


@rule("cot-reaches-reward", "hack", "medium",
      "The package asks the model for a think block in its reply and a paid reward sends the raw reply to an "
      "LLM judge, so the judge reads and scores the policy's reasoning.")
def r_cot_reaches_reward(pkg: Package):
    if not _think_expected(pkg):
        return []
    hits = _cot_hits(pkg)
    if not hits:
        return []
    tpf, tln = _think_expected(pkg)
    pf, fi, line, how = hits[0]
    msg = (f"{fi.name} sends the policy's raw reply to the judge ({how}), and the package asks for a <think> "
           f"block in that reply ({tpf.path}:{tln}), so the judge reads the reasoning: a reply whose reasoning "
           "reaches the right answer, or argues to the judge, can be paid even when its final answer is wrong.")
    return [F("cot-reaches-reward", msg, [ev(p, ln) for p, _, ln, _ in hits] + [ev(tpf, tln)], confidence="review")]


# ---------------------------------------------------------------------------
# 20. label-free-reward
# Evidence row 20 (section 9.2), arxiv:2603.13045 (a MetricX-only reward gave 92.43% wrong-language
# output, Table 2), SRT (majority-vote self-reward collapsed to the same \boxed{1} answer, Sec 4.2),
# arxiv:2508.00410 (self-reward baselines fell to 2.8% on MATH500, Table 11), WeakSup (self-certainty
# collapsed in every setting).
# A paid reward that is computed from the policy's own samples or confidence (majority vote over the
# group, the policy's own logprobs or entropy) or from a reference-free quality metric, while no rubric
# function reads a gold answer and no language, copy or length guard is in the package.
# ---------------------------------------------------------------------------

GOLD_PARAM = {"answer", "answers", "gold", "reference", "references", "target", "targets", "ground_truth", "label"}
GOLD_KEY = re.compile(r"^(answer|gold\w*|reference\w*|target\w*|ground_truth|label|solution|expected\w*)$")
MAJORITY = re.compile(r"(most_common\(|statistics\.mode\(|\bmode\(|max\(\s*set\(\s*\w+\s*\)\s*,\s*key\s*=\s*\w+\.count|"
                      r"majority|plurality|self_consistency|consensus)")
SELF_CONF = re.compile(r"(self_certainty|self-certainty|selfcertainty|token_entropy|mean_entropy|neg_entropy|"
                       r"logprobs?\b|log_prob|avg_logprob|mean_logprob|perplexity)", re.I)
QE_METRIC = re.compile(r"(cometkiwi|comet-kiwi|wmt2[0-9]-cometkiwi|wmt22-comet-qe|comet_qe|cometqe|metricx[\w.-]*qe|"
                       r"xcomet|unbabel/|metricx|kiwi)", re.I)
GUARD = re.compile(r"(langdetect|detect_lang|langid|fasttext|lid\.176|lingua|language_id|lang_id|cld3|\bdetect\(|"
                   r"copy_penalty|is_copy|copied|identical|== *src|== *source|src *==|source *==|overlap|edit_distance|"
                   r"levenshtein|change_rate|length_ratio|len_ratio|min_length|max_length_ratio)", re.I)


def _gold_logged(pkg: Package) -> bool:
    """Some rubric function (paid or a metric) reads a gold answer or a gold field."""
    W = _reward_weights(pkg)
    names = set(W)
    for _, fi in all_funcs(pkg):
        if fi.name not in names:
            continue
        if set(fi.params) & GOLD_PARAM:
            # a gold parameter that the body actually reads
            if any(isinstance(x, ast.Name) and x.id in GOLD_PARAM for x in ast.walk(fi.node)):
                return True
        for n in ast.walk(fi.node):
            k = None
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                k = n.slice.value
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args \
                    and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                k = n.args[0].value
            if k and GOLD_KEY.match(k):
                return True
    return False


def _calls_model(pkg: Package, fi: FuncInfo) -> bool:
    """The function, or a package function it calls (two hops), sends a request to a model API."""
    local = {}
    for _, g in all_funcs(pkg):
        local.setdefault(g.name, g)
    todo, seen = [fi], set()
    for _ in range(3):
        nxt = []
        for f in todo:
            if f.name in seen:
                continue
            seen.add(f.name)
            for n in ast.walk(f.node):
                if isinstance(n, ast.Call):
                    nm = _callee(n)
                    recv = _unparse(n.func.value) if isinstance(n.func, ast.Attribute) else ""
                    if nm in API_CALL and API_RECV.search(recv + "." + nm):
                        return True
                    if nm in local:
                        nxt.append(local[nm])
        todo = nxt
    return False


LF_WORDS = ("most_common", "mode(", "majority", "plurality", "consensus", "self_consistency", "certainty", "entropy",
            "logprob", "log_prob", "perplexity")
QE_WORDS = ("comet", "kiwi", "metricx", "unbabel")
LF_GATE = re.compile(r"(most_common|mode\(|majority|plurality|consensus|self_consistency|certainty|entropy|logprob|"
                     r"log_prob|perplexity|comet|kiwi|metricx|unbabel)", re.I)


def _label_free_hits(pkg: Package):
    out = []
    gate: dict[str, tuple[bool, bool]] = {}  # file -> (any trigger word, a QE metric named in the file)
    for pf in pkg.py_all():
        low = pf.text.lower()  # literal checks: a case-insensitive alternation is slow on large files
        qe_file = any(w in low for w in QE_WORDS)
        gate[pf.path] = (qe_file or any(w in low for w in LF_WORDS), qe_file and bool(QE_METRIC.search(pf.text)))
    if not any(g for g, _ in gate.values()):
        return out
    for pf, fi in _grader_funcs(pkg):
        g, qe_file = gate.get(pf.path, (True, False))
        if not g:
            continue
        src = fi.src
        # a QE model loaded at module level and called here: model.predict(...), scorer.score(...)
        qe = bool(QE_METRIC.search(src) or (qe_file and re.search(r"\.(predict|score|compute)\(", src)))
        if not (MAJORITY.search(src) or SELF_CONF.search(src) or qe):
            continue
        w = _paid(pkg, fi.name, hops=1)
        if not w:
            continue
        if set(fi.params) & GOLD_PARAM and any(isinstance(x, ast.Name) and x.id in GOLD_PARAM for x in ast.walk(fi.node)):
            continue  # compares with the gold itself
        group = any(p in ("completions", "responses", "parsed_answers", "answers_list") for p in fi.params)
        if MAJORITY.search(src) and group:
            out.append((pf, fi, "majority", "rewards agreement with the group's most common answer"))
        elif SELF_CONF.search(src) and re.search(r"(logprob|entropy|certainty|perplexity)", src, re.I) and \
                not re.search(r"(reference|gold|target)\w*\s*[,)=]", src) and \
                set(fi.params) & {"state", "completion", "completions", "responses"} and not _calls_model(pkg, fi):
            # the rollout's own logprobs, not a separate scoring model run over some other text
            out.append((pf, fi, "confidence", "pays the policy's own token confidence (logprobs or entropy)"))
        elif qe and not re.search(r"[\"']ref[\"']\s*:|reference\w*\s*=|\bref\s*=|[\"']reference", src):
            out.append((pf, fi, "qe", "pays a reference-free quality metric"))
    return out


@rule("label-free-reward", "hack", "medium",
      "The paid reward is the policy's agreement with itself, its own confidence or a reference-free quality "
      "metric, with no gold answer checked anywhere and no language, copy or length guard.")
def r_label_free(pkg: Package):
    hits = _label_free_hits(pkg)
    if not hits or _gold_logged(pkg):
        return []
    text = "\n".join(pf.text for pf in pkg.py_all())
    if GUARD.search(text):
        return []
    pf, fi, kind, what = hits[0]
    how = {"majority": "every rollout in a group that gives the same fixed answer (for example \\boxed{1}) is paid "
                       "in full without solving anything",
           "confidence": "a short, repeated or fixed reply that the policy is sure of scores highest whatever it says",
           "qe": "a reply in the wrong language, or a copy of the source text, can score high"}[kind]
    msg = (f"{fi.name} {what}, and no rubric function checks a gold answer and no language, copy or length guard "
           f"exists, so {how}.")
    return [F("label-free-reward", msg, [ev(p, f.start) for p, f, _, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# 19. judge-same-family-or-policy
# Evidence row 19 (section 9.2), arxiv:2604.06996 (GPT-5 about 20 times more likely to wrongly pass its
# own failed output, HSPP 20.15, Table 1), url:openreview.net/forum?id=d33ce8re31 (a 26.6-point
# own-family leniency, Sec 2.5). Counter-evidence: only 3 of 14 same-family contrasts were significant
# in arxiv:2609.34198 (Sec 5.1).
# The policy model is usually chosen at train time, so this fires only when the package itself names
# it: a shipped training config gives the policy model, and either the judge is served by that same
# inference server (the policy grades itself) or the judge model is the same model or the same family.
# ---------------------------------------------------------------------------

FAMILIES = [("OpenAI", r"^(gpt-|o[1-9]($|-)|chatgpt)"), ("Anthropic", r"claude"), ("Google", r"(gemini|gemma)"),
            ("Qwen", r"(qwen|qwq)"), ("Meta", r"llama"), ("DeepSeek", r"deepseek"),
            ("Mistral", r"(mistral|mixtral|magistral|ministral|codestral)"), ("Moonshot", r"(kimi|moonshot)"),
            ("Zhipu", r"glm"), ("xAI", r"grok"), ("Microsoft", r"^phi-"), ("AllenAI", r"(olmo|tulu)"),
            ("NVIDIA", r"nemotron"), ("IBM", r"granite")]


def _family(name: str) -> str | None:
    n = name.lower().strip().split("/")[-1]
    for fam, pat in FAMILIES:
        if re.search(pat, n):
            return fam
    return None


def _norm_url(u: str) -> str:
    u = u.strip().rstrip("/").lower().replace("localhost", "127.0.0.1").replace("0.0.0.0", "127.0.0.1")
    return re.sub(r"/v1$", "", u)


def _train_configs(pkg: Package):
    """(path, policy model, policy urls, env args dict) from shipped TOML training configs."""
    c = _cache(pkg)
    if "train_cfgs" in c:
        return c["train_cfgs"]
    out = []
    try:
        import tomllib
    except Exception:  # pragma: no cover
        tomllib = None
    for p, t in pkg.files.items():
        if not p.endswith(".toml") or os.path.basename(p) == "pyproject.toml" or tomllib is None:
            continue
        if "model" not in t:
            continue
        try:
            cfg = tomllib.loads(t)
        except Exception:
            continue
        roots = [cfg] + [cfg[k] for k in ("orchestrator", "inference", "trainer") if isinstance(cfg.get(k), dict)]
        policy, urls, args = None, [], {}
        for r in roots:
            m = r.get("model")
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                policy = policy or m["name"]
            cl = r.get("client")
            if isinstance(cl, dict):
                b = cl.get("base_url")
                urls += [b] if isinstance(b, str) else [x for x in (b or []) if isinstance(x, str)]
            envs = r.get("env") or r.get("environment")
            for e in (envs if isinstance(envs, list) else [envs] if isinstance(envs, dict) else []):
                if isinstance(e, dict) and isinstance(e.get("args"), dict):
                    args.update(e["args"])
        if policy:
            out.append((p, policy, urls, args))
    c["train_cfgs"] = out
    return out


def _toml_line(pkg: Package, path: str, needle: str) -> int:
    t = pkg.files.get(path, "")
    i = t.find(needle)
    return t.count("\n", 0, i) + 1 if i >= 0 else 1


def _code_judge_models(pkg: Package) -> list[tuple[str, PyFile, int]]:
    out = []
    for d in _judge_sites(pkg):
        if d["model"]:
            out.append((d["model"], d["pf"], d["line"]))
    for pf, fi in all_funcs(pkg):
        a = fi.node.args
        pos = a.posonlyargs + a.args
        pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + \
            [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
        for arg, dflt in pairs:
            if re.search(r"judge\w*model|grader\w*model|critic\w*model", arg.arg) and isinstance(dflt, ast.Constant) \
                    and isinstance(dflt.value, str):
                out.append((dflt.value, pf, dflt.lineno))
    return out


@rule("judge-same-family-or-policy", "hack", "medium",
      "The package names the policy model and makes the judge the policy itself, or a model of the same family, "
      "so the reward comes from a grader that shares the policy's blind spots and own-family leniency.")
def r_judge_same_family(pkg: Package):
    cfgs = _train_configs(pkg)
    if not cfgs:
        return []
    code_models = None
    for path, policy, urls, args in cfgs:
        purls = {_norm_url(u) for u in urls}
        jurl = next((v for k, v in args.items() if re.search(r"judge\w*(base_)?url", k) and isinstance(v, str)), None)
        jmodel = next((v for k, v in args.items() if re.search(r"judge\w*model", k) and isinstance(v, str)), None)
        if jurl and _norm_url(jurl) in purls:
            return [F("judge-same-family-or-policy",
                      f"{path} trains {policy} and sends the judge to the policy's own inference server ({jurl}"
                      f"{', model ' + repr(jmodel) if jmodel else ''}), so the model being trained, or its frozen base, "
                      "grades its own replies: text that model rates highly is paid whether or not it does the task.",
                      [f"{path}:{_toml_line(pkg, path, jurl)}"], confidence="review")]
        fam = _family(policy)
        if not fam:
            continue
        if jmodel and jmodel.lower() not in ("auto", "rollout"):
            cands = [(jmodel, None, None)]
        else:
            if code_models is None:
                code_models = _code_judge_models(pkg)
            cands = code_models
        for m, mpf, mln in cands:
            if _family(m) == fam:
                same = m.lower().split("/")[-1] == policy.lower().split("/")[-1]
                evid = [f"{path}:{_toml_line(pkg, path, policy)}"] + ([ev(mpf, mln)] if mpf is not None else [])
                return [F("judge-same-family-or-policy",
                          f"{path} trains {policy} and the judge is {m}, {'the same model' if same else 'the same family (' + fam + ')'}, "
                          "so the judge is more likely to pass the policy's own failed replies than an outside judge would.",
                          evid, confidence="review")]
    return []


# ---------------------------------------------------------------------------
# 23. spec-oracle-gap
# Evidence row 23 (section 9.2), arxiv:2610.30635 (a model-free static audit found hidden name
# dependencies in 25 of 98 milestones, and blocked units failed in 80.6% of runs vs 48.4%, Sec 6.2),
# arxiv:2607.10891 (no-op and oracle checks missed unstated test literals, App A.3).
# Harbor tasks only. A name counts when a hidden test needs it, the oracle solution defines or
# writes it (so the agent has to produce it, it is not already in the image), and neither
# instruction.md nor any file under environment/ mentions it.
# ---------------------------------------------------------------------------

SYS_ROOTS = ("/tests", "/logs", "/tmp", "/usr", "/bin", "/sbin", "/etc", "/proc", "/dev", "/sys", "/var", "/opt",
             "/lib", "/lib64", "/run", "/oracle", "/solution", "/installed-agent", "/root/.", "/home/agent/.",
             "/nix", "/snap", "/boot", "/mnt", "/srv")
PATH_LIT = re.compile(r"^/[\w.\-]+(/[\w.\-]+)+/?$")
WRITE_HINT = re.compile(r"(>|\btee\b|\s-o\s|--output|\bopen\([^)]*['\"][wax]b?['\"]|\.write|\bsave|to_csv|to_json|"
                        r"to_parquet|\bdump|\bcp\s|\bmv\s|\btouch\s|mkdir|\bln\s|writeFile|write_text|write_bytes|"
                        r"imwrite|savefig|\bexport\b|\bcreate|cargo new|\binit\b|<<)", re.I)
LOADERS = re.compile(r"^(load|loads|safe_load|read_csv|read_json|read_parquet|read_excel|DictReader|reader|"
                     r"json|literal_eval|load_json|read_jsonl|load_yaml)$")
PIP_LINE = re.compile(r"(pip3?|uv\s+pip|uv\s+add|uvx|pipx)\s+(install\s+)?(.*)")
DEFINES = r"(\b(def|class|fn|func|function|struct|enum|trait|interface|type|const|let|var|sub|module)\s+{n}\b|^\s*{n}\s*=[^=])"
TEMPLATE = re.compile(r"(\{\{?\s*[a-z_]+\s*\}\}?|template placeholder)")


def _harbor_tasks(pkg: Package) -> list[str]:
    """Prefixes ("" or "a/b/") of Harbor task folders: a task.toml next to an instruction.md."""
    c = _cache(pkg)
    if "harbor_tasks" not in c:
        out = []
        for p in pkg.files:
            if os.path.basename(p) == "task.toml":
                d = os.path.dirname(p)
                pre = d + "/" if d else ""
                if pre + "instruction.md" in pkg.files:
                    out.append(pre)
        c["harbor_tasks"] = sorted(out)
    return c["harbor_tasks"]


def _const_path(e) -> str | None:
    """Literal path text of "/a/b", os.path.join("/a", "b") or Path("/a") / "b"."""
    if isinstance(e, ast.Constant) and isinstance(e.value, str):
        return e.value
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Div):
        a, b = _const_path(e.left), _const_path(e.right)
        return f"{a.rstrip('/')}/{b.lstrip('/')}" if a and b else None
    if isinstance(e, ast.Call):
        nm = _callee(e)
        if nm in ("Path", "PurePath", "PosixPath") and len(e.args) == 1:
            return _const_path(e.args[0])
        if nm == "join" and e.args and all(_const_path(a) for a in e.args):
            return "/".join(x.strip("/") if i else x.rstrip("/") for i, x in enumerate(_const_path(a) for a in e.args))
    return None


def _word_in(name: str, text: str) -> bool:
    if name not in text:
        return False
    return re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w])", text) is not None


def _in_spec(kind: str, name: str, spec: str) -> bool:
    if kind == "path":
        base = name.rstrip("/").rsplit("/", 1)[-1]
        return name.rstrip("/") in spec or _word_in(base, spec)
    if kind == "key":
        return name in spec
    return _word_in(name, spec)


def _oracle_has(kind: str, name: str, sol: str) -> bool:
    """The oracle solution produces the name: writes the path, defines the identifier, emits the key."""
    if kind == "path":
        base = name.rstrip("/").rsplit("/", 1)[-1]
        if base not in sol:
            return False
        for line in sol.splitlines():
            if (name.rstrip("/") in line or _word_in(base, line)) and WRITE_HINT.search(line):
                return True
        return False
    if name not in sol:
        return False
    if kind in ("function", "class", "attribute"):
        return re.search(DEFINES.format(n=re.escape(name)), sol, re.M) is not None
    if kind == "module":
        return any(_word_in(name + ext, ln) and WRITE_HINT.search(ln) for ln in sol.splitlines()
                   for ext in (".py", "/"))
    return _word_in(name, sol) or (kind == "key" and (f'"{name}"' in sol or f"'{name}'" in sol))


def _derived(v, loaded: set[str]) -> bool:
    """v reads straight out of loaded data: x, x["k"], x.k, x.get("k"), x.items(), list(x)."""
    while True:
        if isinstance(v, ast.Await):
            v = v.value
        elif isinstance(v, (ast.Subscript, ast.Attribute)):
            v = v.value
        elif isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr in ("get", "items", "values",
                                                                                                "keys", "copy"):
            v = v.func.value
        elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("list", "dict", "sorted",
                                                                                         "enumerate", "iter") and v.args:
            v = v.args[0]
        else:
            return isinstance(v, ast.Name) and v.id in loaded


def _test_needs(pf: PyFile, installed: set[str], skip_mods: set[str], agent_ok,
                shipped: set[str] = frozenset()) -> list[tuple[str, str, int]]:
    """(kind, name, line) for names a Python test file needs from the agent's work. Modules in skip_mods
    (the test's own helpers, or code the image already has) are not the agent's."""
    out = []
    tree = pf.tree
    if tree is None:
        return out
    local = {fi.name for fi in pf.funcs}
    local |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    std = set(getattr(sys, "stdlib_module_names", ())) | {"pytest", "_pytest"} | skip_mods
    agent_mods, agent_names = set(), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            top = n.module.split(".")[0]
            if top in std or top in installed or not agent_ok(top):
                continue
            agent_mods.add(top)
            for a in n.names:
                if a.name != "*":
                    agent_names[a.asname or a.name] = a.name
                    out.append(("function" if a.name[:1].islower() else "class", a.name, n.lineno))
        elif isinstance(n, ast.Import):
            for a in n.names:
                top = a.name.split(".")[0]
                if top not in std and top not in installed and agent_ok(top):
                    agent_mods.add(a.asname or top)
    for m in agent_mods:
        out.append(("module", m, 1))
    # tainted: names holding data loaded from a file the agent wrote (json, csv, yaml) at a fixed path
    # outside the system and test folders. Data from the test's own fixtures or from logs is not.
    path_names: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            pv = _const_path(n.value)
            if pv and pv.startswith("/"):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name):
                        path_names[t.id] = pv

    def agent_source(v) -> bool:
        paths = [path_names.get(x.id) for x in ast.walk(v) if isinstance(x, ast.Name)]
        paths += [_const_path(x) for x in ast.walk(v) if isinstance(x, (ast.Constant, ast.BinOp))]
        paths = [x for x in paths if x and x.startswith("/")]
        return bool(paths) and all(PATH_LIT.match(x) and not x.startswith(SYS_ROOTS) and
                                   x.rstrip("/").rsplit("/", 1)[-1] not in shipped for x in paths)

    for n in ast.walk(tree):
        if isinstance(n, (ast.With, ast.AsyncWith)):
            for it in n.items:
                if isinstance(it.optional_vars, ast.Name):
                    pv = [_const_path(x) for x in ast.walk(it.context_expr) if isinstance(x, (ast.Constant, ast.BinOp))]
                    pv += [path_names.get(x.id) for x in ast.walk(it.context_expr) if isinstance(x, ast.Name)]
                    pv = [x for x in pv if x and x.startswith("/")]
                    if pv:
                        path_names[it.optional_vars.id] = pv[0]
    loaded: set[str] = set()
    for _ in range(2):
        for n in ast.walk(tree):
            tgts, val = [], None
            if isinstance(n, ast.Assign):
                tgts, val = n.targets, n.value
            elif isinstance(n, (ast.For, ast.comprehension)):
                tgts, val = [n.target], n.iter
            if val is None:
                continue
            v = val.value if isinstance(val, ast.Await) else val
            src = (any(isinstance(c, ast.Call) and LOADERS.match(_callee(c)) for c in ast.walk(v)) and agent_source(v)) \
                or _derived(v, loaded)
            if src:
                for t in tgts:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            loaded.add(x.id)
        for fi in pf.funcs:  # a pytest fixture that returns the loaded data: tests take it as a parameter
            if fi.name not in loaded and any(isinstance(r, ast.Return) and r.value is not None and
                                             _derived(r.value, loaded) for r in _own_nodes(fi.node)):
                loaded.add(fi.name)
    for n in ast.walk(tree):
        # attribute of an agent module: mod.func(...)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in agent_mods \
                and isinstance(n.ctx, ast.Load):
            out.append(("attribute", n.attr, n.lineno))
        # keyword names on a call to an agent function
        if isinstance(n, ast.Call):
            f = n.func
            fname = f.id if isinstance(f, ast.Name) else None
            if (fname and fname in agent_names) or (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                                                     and f.value.id in agent_mods):
                for k in n.keywords:
                    if k.arg:
                        out.append(("argument", k.arg, n.lineno))
        # keys read out of data loaded from the agent's output
        key, base = None, None
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            key, base = n.slice.value, n.value
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args \
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            key, base = n.args[0].value, n.func.value
        if key and len(key) > 1 and any(isinstance(x, ast.Name) and x.id in loaded for x in ast.walk(base)) \
                and not isinstance(getattr(n, "ctx", None), ast.Store):
            out.append(("key", key, n.lineno))
        # paths the test reads
        p = _const_path(n) if isinstance(n, (ast.Constant, ast.BinOp, ast.Call)) else None
        if p and PATH_LIT.match(p) and not p.startswith(SYS_ROOTS):
            out.append(("path", p, n.lineno))
        # CLI flags passed to a program of the agent's
        if isinstance(n, ast.Call) and _callee(n) in ("run", "check_output", "check_call", "call", "Popen") and n.args:
            a0 = n.args[0]
            parts = [x.value for x in ast.walk(a0) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
            joined = " ".join(parts)
            if re.search(r"(\.py\b|\.sh\b|\./|/app/|/workspace/)", joined):
                for tok in re.findall(r"(?<![\w-])(--[a-z][\w-]+)", joined):
                    out.append(("flag", tok, n.lineno))
    return [(k, nm, ln) for k, nm, ln in out if nm not in local]


def _spec_gaps(pkg: Package, pre: str):
    files = pkg.files
    spec_parts = [files.get(pre + "instruction.md", "")]
    sol_parts, test_py, test_sh = [], [], []
    for p, t in files.items():
        if not p.startswith(pre):
            continue
        rel = p[len(pre):]
        if rel.startswith("environment/"):
            spec_parts.append(t)
        elif rel.startswith("solution/"):
            sol_parts.append(t)
        elif rel.startswith("tests/"):
            if p.endswith(".py"):
                test_py.append(p)
            elif p.endswith((".sh", ".bash")):
                test_sh.append(t)
    instr = spec_parts[0]
    if not sol_parts or not test_py or not instr.strip() or TEMPLATE.search(instr):
        return []  # no oracle, no Python tests, or an adapter template whose instruction is written later
    env_text = "\n".join(spec_parts[1:])
    names = [p[len(pre):] for p in pkg.sizes if p.startswith(pre + "environment/")]
    spec_parts += names
    spec = "\n".join(spec_parts)
    # the test's own helper modules, and modules the image already has (named in the shipped files)
    skip_mods = {os.path.splitext(os.path.basename(p))[0] for p in pkg.sizes if p.startswith(pre + "tests/")}
    skip_mods |= {p.split("/")[-2] for p in pkg.sizes if p.startswith(pre + "tests/") and p.endswith("/__init__.py")}
    sol = "\n".join(sol_parts)
    installed = set()
    for t in test_sh + [x for x in spec_parts if "pip" in x]:
        for m in PIP_LINE.finditer(t):
            for tok in re.split(r"[\s,]+", m.group(3)):
                tok = re.split(r"[<>=\[;&|]", tok.strip("'\""))[0].lower().replace("-", "_")
                if tok and not tok.startswith("-"):
                    installed.add(tok)
    gaps, seen = [], set()
    from .rules_grader import _harbor_test_py
    parsed = {x.path: x for x in _harbor_test_py(pkg)}
    for p in sorted(test_py):
        pf = parsed.get(p) or parse_py(p, files[p])
        mods = {n.module.split(".")[0] for n in ast.walk(pf.tree) if isinstance(n, ast.ImportFrom) and n.module} \
            if pf.tree is not None else set()
        base = skip_mods | {m for m in mods if _word_in(m, env_text)}
        def agent_ok(m, instr=instr):  # the agent's own module: the oracle writes it or the instruction names it
            return _oracle_has("module", m, sol) or re.search(r"(?<![\w-])" + re.escape(m) + r"(\.py\b|/)", instr) is not None
        shipped = {os.path.basename(x) for x in pkg.sizes if x.startswith(pre + "environment/")}
        for kind, name, line in _test_needs(pf, installed, base, agent_ok, shipped):
            if (kind, name) in seen:
                continue
            seen.add((kind, name))
            if _in_spec(kind, name, spec) or not _oracle_has(kind, name, sol):
                continue
            if kind == "module" and any(g[1] == name for g in gaps):
                continue
            gaps.append((kind, name, p, line))
    return gaps


# ---------------------------------------------------------------------------
# v0.5 widenings of spec-oracle-gap
# - test-literal-assertions-absent-from-instruction + tests-assert-gold-specific-strings (arxiv:2607.28587,
#   8aiqtxj5ex): a test asserts a message or output string that the oracle emits but the instruction never
#   states. Also a test that compares an unsorted directory listing to an exact list.
# - tests-import-private-helpers (Datacurve, arxiv:2607.07946): a test imports an underscore name that the
#   oracle defines from a module that is already in the image.
# - spec-gap-other-languages-and-layouts (arxiv:2610.30635, 2504.08703): SWE layout tasks with the hidden
#   tests as a patch and the oracle as a gold patch, in any language. A test uses a symbol that the gold
#   patch newly defines and the issue text never names.
# ---------------------------------------------------------------------------

LIT_MIN = 12
ASSERT_LINE = re.compile(r"(\bassert|\bexpect\(|Equal|\.toBe|\.toEqual|toContain|toThrow|toMatch|raises|match=|"
                         r"\brequire\.|\bcheck\.|should|Contains\(|ErrorContains|EqualError|assert_eq!|panic)")
QUOTED = re.compile(r"\"((?:[^\"\\\n]|\\.){%d,300})\"|'((?:[^'\\\n]|\\.){%d,300})'|`([^`\n]{%d,300})`" %
                    (LIT_MIN, LIT_MIN, LIT_MIN))
NEW_DEF = [re.compile(p) for p in (
    r"^\s*(?:async\s+)?def\s+(\w+)\s*\(",                                         # python
    r"^\s*class\s+(\w+)\b",                                                       # python, ts, js, java
    r"^\s*func\s+(\w+)\s*[\[(]",                                                  # go, not methods
    r"^\s*type\s+(\w+)\s+(?:struct|interface|func)\b",                            # go
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*[<(]",  # ts, js
    r"^\s*export\s+(?:const|let|var)\s+(\w+)\s*[:=]",                             # ts, js
    r"^\s*(?:export\s+)?enum\s+(\w+)\b",                                         # ts
    r"^\s*(?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]",              # rust
    r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)\b",                            # rust
)]
STRIP_STR = re.compile(r"\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|`[^`\n]*`")
COMMON_WORDS = {"test", "tests", "setup", "teardown", "main", "init", "self", "this", "true", "false", "none",
                "null", "string", "error", "value", "result", "data", "name", "type", "func", "self", "args",
                "kwargs", "props", "state", "config", "default", "render", "options", "context", "client"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _lit_ok(lit: str) -> bool:
    s = _norm(lit)
    if len(s) < LIT_MIN or " " not in s or s.startswith(("/", "http", "<", "{")) or "{}" in s or "%s" in s:
        return False
    if "${" in s or re.search(r"\{\w*\}", s):
        return False  # a format template, the test fills it in
    return sum(ch.isalpha() for ch in s) >= 0.6 * len(s.replace(" ", ""))


def _unregex(s: str) -> str:
    s = s.strip("^$")
    return re.sub(r"\\(.)", r"\1", s)


def _patch_sides(text: str) -> tuple[list[tuple[int, str]], str]:
    """(added lines, other text) of a unified diff. Other text is removed lines, context lines and hunk
    headers: what was there before the patch."""
    added, other = [], []
    for i, ln in enumerate(text.splitlines()):
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        if ln.startswith("+"):
            added.append((i + 1, ln[1:]))
        elif ln.startswith(("-", " ")) or ln.startswith("@@"):
            other.append(ln[1:] if not ln.startswith("@@") else ln)
    return added, "\n".join(other)


def _is_patch(text: str) -> bool:
    return "\n@@ " in text or text.startswith(("diff --git", "--- ")) and "\n+++ " in text


def _py_assert_literals(pf: PyFile) -> list[tuple[str, int]]:
    """String constants a Python test asserts output against: assert lit in x, assert x == lit,
    assertEqual, assertIn, assertRaisesRegex and pytest.raises(match=lit)."""
    out = []
    tree = pf.tree
    if tree is None:
        return out

    def lit(e):
        return e.value if isinstance(e, ast.Constant) and isinstance(e.value, str) else None

    for n in ast.walk(tree):
        if isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare):
            c = n.test
            sides = [c.left] + list(c.comparators)
            if all(isinstance(op, (ast.Eq, ast.In)) for op in c.ops):
                for s in sides:
                    v = lit(s)
                    if v is not None and not all(lit(x) is not None for x in sides):
                        out.append((v, n.lineno))
        elif isinstance(n, ast.Call):
            nm = _callee(n)
            if nm in ("assertEqual", "assertIn", "assertEquals", "assertMultiLineEqual"):
                for a in n.args[:2]:
                    v = lit(a)
                    if v is not None:
                        out.append((v, n.lineno))
            elif nm in ("assertRaisesRegex", "assertRegex", "assertRaisesRegexp") and len(n.args) >= 2:
                v = lit(n.args[1])
                if v is not None:
                    out.append((_unregex(v), n.lineno))
            elif nm == "raises":
                m = _kw(n, "match")
                v = lit(m) if m is not None else None
                if v is not None:
                    out.append((_unregex(v), n.lineno))
    return out


def _listdir_order(pf: PyFile) -> list[int]:
    """Lines where a test compares an unsorted os.listdir, glob or iterdir result to a list literal."""
    out = []
    if pf.tree is None or ("listdir" not in pf.text and "glob" not in pf.text and "iterdir" not in pf.text):
        return out
    for n in ast.walk(pf.tree):
        if not (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)):
            continue
        a, b = n.left, n.comparators[0]
        for x, y in ((a, b), (b, a)):
            if isinstance(y, ast.List) and len(y.elts) >= 2 and isinstance(x, ast.Call):
                inner = x
                if _callee(inner) == "list" and inner.args and isinstance(inner.args[0], ast.Call):
                    inner = inner.args[0]
                if _callee(inner) in ("listdir", "glob", "iterdir"):
                    p = next((_const_path(z) for z in ast.walk(inner) if _const_path(z)), None)
                    if not (p and p.startswith(("/tests", "/tmp"))):
                        out.append(n.lineno)
    return out


def _spec_extra(pkg: Package, pre: str):
    """v0.5 gaps for one Harbor task: (kind, name, path, line). Kinds: literal, private, new symbol and
    order (an unsorted listing compared to an exact list)."""
    files = pkg.files
    instr = files.get(pre + "instruction.md", "")
    if not instr.strip() or TEMPLATE.search(instr):
        return []
    env_parts, sol_files, tests = [], [], []
    for p, t in files.items():
        if not p.startswith(pre):
            continue
        rel = p[len(pre):]
        if rel.startswith("environment/"):
            env_parts.append(t)
        elif rel.startswith("solution/"):
            sol_files.append((p, t))
        elif rel.startswith("tests/"):
            tests.append((p, t))
    if not sol_files or not tests:
        return []
    env_parts += [p[len(pre):] for p in pkg.sizes if p.startswith(pre + "environment/")]
    spec = instr + "\n" + "\n".join(env_parts)
    lazy: dict[str, str] = {}

    def normed(key, text):  # whitespace-collapsed text, built only when a literal needs it
        if key not in lazy:
            lazy[key] = _norm(text)
        return lazy[key]
    sol_patches = [t for _, t in sol_files if _is_patch(t)]
    if sol_patches:
        sol_added, sol_before = [], []
        for t in sol_patches:
            a, o = _patch_sides(t)
            sol_added += [x for _, x in a]
            sol_before.append(o)
        sol_new = "\n".join(sol_added)
        sol_old = "\n".join(sol_before)
    else:
        sol_new, sol_old = "\n".join(t for _, t in sol_files), ""
    test_text_all = {p: t for p, t in tests}
    out, seen = [], set()

    def lit_gap(v, path, line, test_text):
        s = _norm(v)
        key = ("literal", s)
        if key in seen or not _lit_ok(s):
            return
        seen.add(key)
        spec_low = normed("spec_low", spec.lower())
        if s.lower() in spec_low or s.rstrip(".!:").lower() in spec_low:
            return
        toks = re.findall(r"[A-Za-z_]\w*", s)
        idents = [x for x in toks if not x.islower() or "_" in x or any(ch.isdigit() for ch in x)]
        if len(toks) <= 3 and idents and all(_word_in(x, spec) for x in idents):
            return  # code made of names the instruction gives, like "service KVStore"
        if s not in normed("new", sol_new) or (sol_old and s in normed("old", sol_old)):
            return  # the oracle does not emit it, or it was already in the code before the gold patch
        if test_text.count(v) > 1 or any(v in t for q, t in test_text_all.items() if q != path):
            return  # also an input the test feeds in, or a fixture
        out.append(("literal", s if len(s) <= 60 else s[:57] + "...", path, line))

    from .rules_grader import _harbor_test_py
    parsed = {x.path: x for x in _harbor_test_py(pkg)}
    for p, t in sorted(tests):
        if p.endswith(".py"):
            pf = parsed.get(p) or parse_py(p, t)
            for v, ln in _py_assert_literals(pf):
                lit_gap(v, p, ln, t)
            for ln in _listdir_order(pf):
                out.append(("order", "listing", p, ln))
            # private helpers from a module the image already has (SWE style), which the oracle defines
            if pf.tree is not None and sol_patches:  # only a gold patch shows that the name is new
                for n in ast.walk(pf.tree):
                    if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                        for a in n.names:
                            nm = a.name
                            if nm.startswith("_") and not nm.startswith("__") and ("private", nm) not in seen:
                                seen.add(("private", nm))
                                if _word_in(nm, spec) or not re.search(DEFINES.format(n=re.escape(nm)), sol_new, re.M):
                                    continue
                                if sol_old and re.search(DEFINES.format(n=re.escape(nm)), sol_old, re.M):
                                    continue
                                out.append(("private", nm, p, n.lineno))
        elif _is_patch(t) and sol_patches:
            added_n, before = _patch_sides(t)
            added = [x for _, x in added_n]
            add_text = "\n".join(added)
            test_defs = {m.group(1) for ln in added for r in NEW_DEF for m in [r.match(ln)] if m}
            new_syms = {}
            for ln in sol_new.splitlines():
                for r in NEW_DEF:
                    m = r.match(ln)
                    if m:
                        new_syms.setdefault(m.group(1), ln)
            for i, ln in enumerate(added):
                s = ln.strip()
                if s.startswith(("#", "//", "*", "/*")):
                    continue
                if ASSERT_LINE.search(ln):
                    for m in QUOTED.finditer(ln):
                        v = next(g for g in m.groups() if g is not None)
                        lit_gap(v.replace('\\"', '"').replace("\\'", "'"), p, added_n[i][0], add_text)
            for nm in sorted(new_syms):
                if len(nm) < 4 or nm.lower() in COMMON_WORDS or nm.startswith("__") or nm in test_defs:
                    continue
                if ("new symbol", nm) in seen:
                    continue
                if not re.search(r"(?<![\w$])" + re.escape(nm) + r"(?![\w$])", add_text):
                    continue
                if re.search(r"(?<![\w$])" + re.escape(nm) + r"(?![\w$])", sol_old + "\n" + before):
                    continue  # it existed before the gold patch
                if re.search(r"(?<![\w$])" + re.escape(nm) + r"(?![\w$])", spec, re.I):
                    continue
                code_lines = [j for j, ln in enumerate(added) if re.search(r"(?<![\w$])" + re.escape(nm) +
                                                                           r"(?![\w$])", STRIP_STR.sub('""', ln))
                              and not ln.strip().startswith(("#", "//", "*"))]
                if not code_lines:
                    continue
                seen.add(("new symbol", nm))
                out.append(("new symbol", nm, p, added_n[code_lines[0]][0]))
    return out


@rule("spec-oracle-gap", "broken", "medium",
      "A Harbor test needs a name (a file path, function, argument, key or flag) that the oracle solution "
      "uses but that the instruction and the shipped files never give, so a solution that follows the "
      "instruction can fail.")
def r_spec_oracle_gap(pkg: Package):
    tasks = _harbor_tasks(pkg)
    if not tasks:
        return []
    hits, order = [], []
    for pre in tasks:
        gaps = _spec_gaps(pkg, pre)
        have = {g[1] for g in gaps}
        extra = [g for g in _spec_extra(pkg, pre) if g[1] not in have]
        gaps += [g for g in extra if g[0] != "order"]
        if gaps:
            hits.append((pre, gaps))
        o = [g for g in extra if g[0] == "order"]
        if o:
            order.append((pre, o))
    out = []
    if hits:
        pre, gaps = hits[0]
        names = ", ".join(f"`{n}` ({k})" for k, n, _, _ in gaps[:4]) + (f" and {len(gaps) - 4} more" if len(gaps) > 4 else "")
        where = f"{len(hits)} tasks, first {pre.rstrip('/') or '.'}" if len(hits) > 1 else (pre.rstrip("/") or "the task")
        msg = (f"Hidden tests need {names}, which the oracle solution uses but instruction.md and the shipped files never "
               f"mention ({where}), so a solution that follows the instruction with other names fails.")
        evid = [ev(p, ln) for _, gs in hits for _, _, p, ln in gs[:3]]
        out.append(F("spec-oracle-gap", msg, evid, confidence="review"))
    if order:
        pre, o = order[0]
        where = f"{len(order)} tasks, first {pre.rstrip('/') or '.'}" if len(order) > 1 else (pre.rstrip("/") or "the task")
        out.append(F("spec-oracle-gap", "A hidden test compares an unsorted directory listing to an exact list, and "
                                        f"the order of a listing is not fixed ({where}), so correct output can fail.",
                     [ev(p, ln) for _, gs in order for _, _, p, ln in gs[:3]], confidence="review"))
    return out


# ---------------------------------------------------------------------------
# 15. kernel-perf-grader
# Evidence row 15 (section 9.2), arxiv:2507.14111 (a CUDA stream trick escaped timing on 82 of 250
# tasks, Sec 3.1), arxiv:2609.22220 (KernelBench's atol of 1e-2 is 4,000x the mean softmax output, so
# an all-zero kernel passes, Sec 2), arxiv:2607.16241 (speedup falls from 1.43x to 0.88x under verified
# evaluation). Two forms with a nameable exploit: (a) a GPU speed timer whose window has no device-wide
# synchronize before the clock stops (a host timer, or CUDA events recorded on one stream), and (b) a
# multi-problem kernel grader whose correctness check allows an absolute error of 0.01 or more. Inputs
# drawn only from [0,1) and one fixed shape depend on the problem files, which are data, and are not
# reported.
# ---------------------------------------------------------------------------

GPU_CTX = re.compile(r"(torch\.cuda|\.cuda\(|['\"]cuda['\"]|\bcupy\b|\btriton\b|load_inline|nvcc|\bjax\b|cudaEvent)")
GPU_LITS = ("cuda", "cupy", "triton", "load_inline", "nvcc", "jax")
HOST_TIMER = {"perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns", "time", "time_ns", "default_timer"}
SYNC_CALL = re.compile(r"^(synchronize|cuda_synchronize|device_synchronize|cudaDeviceSynchronize|block_until_ready|"
                       r"cpu|item|tolist|numpy|wait|query|_sync\w*|sync\w*)$")
SPEED_WORD = re.compile(r"(speedup|speed_up|baseline_(time|ms|runtime|median|mean)|ref(erence)?_(time|ms|runtime)|"
                        r"runtime_ratio|fast_[012]|geomean)", re.I)
MULTI_PROBLEM = re.compile(r"(ModelNew|get_inputs|get_init_inputs|ref_model|reference_model|original_model)")
IPC_CALL = re.compile(r"^(request|send\w*|recv\w*|post|get|put|communicate|run|check_output|check_call|call|Popen|wait|"
                      r"read\w*|urlopen|invoke|ainvoke|create|acreate|json|loads|dumps|sleep)$")
CMP_CALL = {"allclose", "assert_close", "isclose", "assert_allclose", "testing_assert_close"}


def _timer_windows(fi: FuncInfo):
    """(start line, end line, has sync, has work, kind) for each timed window in a function."""
    calls = [n for n in _own_nodes(fi.node) if isinstance(n, ast.Call)]
    marks = []
    for c in calls:
        nm = _callee(c)
        recv = _unparse(c.func.value) if isinstance(c.func, ast.Attribute) else ""
        if nm in HOST_TIMER and (recv in ("time", "timeit", "") or recv.endswith("time")) and \
                (nm != "time" or recv == "time"):
            marks.append((c.lineno, "host", c))
        elif nm == "record" and isinstance(c.func, ast.Attribute):
            marks.append((c.lineno, "event", c))
    marks.sort(key=lambda m: (m[0], m[2].col_offset))
    out = []
    for (l1, k1, c1), (l2, k2, c2) in zip(marks, marks[1:]):
        if k1 != k2 or l2 <= l1:
            continue
        inside = [c for c in calls if l1 <= c.lineno <= l2 and c is not c1 and c is not c2
                  and not (c.lineno == l1 and c.col_offset < c1.col_offset)]
        has_sync = any(SYNC_CALL.match(_callee(c)) for c in inside)
        work = [c for c in inside if _callee(c) not in HOST_TIMER and _callee(c) not in ("print", "collect", "append",
                                                                                        "info", "debug", "len", "range")]
        # a blocking round trip to another process or server waits for its answer: not queued GPU work
        work = [c for c in work if not IPC_CALL.match(_callee(c))]
        out.append((l1, l2, has_sync, bool(work), k1))
    return out


def _paid_path_names(pkg: Package) -> set[str]:
    """Names of paid reward functions and everything they call in the package (three hops)."""
    c = _cache(pkg)
    if "paid_path" in c:
        return c["paid_path"]
    W = _reward_weights(pkg)
    names = {n for n, ws in W.items() if any(w > 0 for w, _ in ws)}
    local: dict[str, list[FuncInfo]] = {}
    for _, g in all_funcs(pkg):
        local.setdefault(g.name, []).append(g)
    frontier = set(names)
    for _ in range(3):
        nxt = set()
        for n in frontier:
            for g in local.get(n, []):
                nxt |= {x for x in called_names(g) if x in local}
        frontier = nxt - names
        names |= frontier
    c["paid_path"] = names
    return names


def _harbor_reward_file(pf: PyFile) -> bool:
    return bool(re.search(r"(reward\.(txt|json)|emit_reward|write_reward|/logs/verifier)", pf.text))


def _kernel_files(pkg: Package):
    """(file, paid function names or None for a Harbor reward script) for grader files about GPU speed."""
    from .rules_grader import _harbor_test_py
    out = []
    def gpu(t):
        return any(x in t for x in GPU_LITS) and GPU_CTX.search(t)
    for pf in pkg.py_all():
        if pf.tree is not None and ("allclose" in pf.text or "speed" in pf.text.lower() or "baseline_" in pf.text) \
                and gpu(pf.text) and (SPEED_WORD.search(pf.text) or "allclose" in pf.text):
            out.append((pf, "verifiers"))
    for pf in _harbor_test_py(pkg):
        if pf.tree is not None and gpu(pf.text) and _harbor_reward_file(pf):
            out.append((pf, "harbor"))
    return out


# ---------------------------------------------------------------------------
# v0.5 widenings of kernel-perf-grader. Each one is written from the item text and the cited notes, and
# each stays narrow where the notes do not say more.
# - kernel-timing-isolation + measurement-in-candidate-process (arxiv:2610.15924, Harbor RFC 0003): the
#   timer or memory counter runs in the process that imported or exec'd the submission.
# - kernel-launch-and-train-branch + kernel-delegates-to-reference-op (arxiv:2603.02637, arxiv:2602.05885,
#   trustworthy-env K-15): nothing checks the submission for a custom kernel, or only a text search does.
# - kernel-input-exposure-and-cache (arxiv:2606.08960, arxiv:2609.28614): every timed call gets the same
#   input objects, so an output cache keyed on the input passes.
# - kernel-tolerance-sub-threshold (arxiv:2609.22220): atol above 1e-3 (the paper's tighter value) now counts.
# - kernel-baseline-precision + kernel-precision-or-dtype-asymmetry (arxiv:2607.16241, trustworthy-env
#   an internal item): a timed FP32 baseline with TF32 never set and a tolerance of 1e-2 or more, or a baseline cast
#   to float32 while the candidate is cast to half.
# - unclipped-speedup-reward (arxiv:2602.05885): a paid speedup with no upper clip. Hygiene.
# - kernel-output-buffer-and-input-aliasing (trustworthy-env K-7): the candidate runs before the reference
#   on the same input objects with no clone.
# ---------------------------------------------------------------------------

INPROC_LOAD = re.compile(r"^(exec|load_inline|spec_from_file_location|module_from_spec|exec_module|load_module|"
                         r"import_module|load_source|run_path)$")
MEM_CALL = {"get_traced_memory", "getrusage", "memory_info", "max_memory_allocated", "max_memory_reserved"}
SPEED_ASSERT = re.compile(r"(time|\bdt\b|elapsed|speed|latency|duration|_ms\b|faster|slower|throughput|runtime|"
                          r"memory|\bmem|peak|rss)", re.I)
AGENT_DIR = re.compile(r"^/(app|workspace|workdir|home|code|src|submission|solution|output|project|repo|testbed|"
                       r"root)(/|$)")
THIRD_PARTY = {"numpy", "np", "torch", "pandas", "scipy", "pytest", "requests", "triton", "jax", "cupy", "sklearn",
               "transformers", "PIL", "cv2", "yaml", "toml", "tomli", "psutil", "mujoco", "numba", "tqdm", "rich",
               "pydantic", "fastapi", "flask", "httpx", "aiohttp", "sympy", "networkx", "matplotlib", "seaborn",
               "polars", "pyarrow", "datasets", "huggingface_hub", "safetensors", "einops", "verifiers", "openai",
               "anthropic", "boto3", "redis", "sqlalchemy", "duckdb", "psycopg2", "pymongo", "grpc", "google"}
KERNEL_PROOF = ("__global__", "triton.jit", "@triton", "load_inline", "cuda_sources", "<<<", "tl.load", "cpp_extension",
                "cuda_source", "@cuda.jit", "tl.store")
CAND_NAME = re.compile(r"(model_new|new_model|modelnew|custom|candidate|submission|kernel_fn|solution|optimized|"
                       r"student|policy_model|generated)", re.I)
REF_NAME = re.compile(r"(^|_)(ref|reference|original|baseline|orig|gold|torch_impl)(_|$|model)|^model$|^Model$", re.I)
FRESH_CALL = re.compile(r"^(clone|detach|copy|deepcopy|get_inputs\w*|randn\w*|rand\w*|randint\w*|empty\w*|zeros\w*|"
                        r"ones\w*|normal_?|uniform_?|tensor|from_numpy|copy_|contiguous|to|cuda|fill_|full\w*|"
                        r"make_inputs|gen\w*|create\w*|sample\w*)$")
CLIP_CALL = re.compile(r"^(min|max|clip|clamp\w*|_clamp\w*|bound\w*|tanh|sigmoid|log\w*|sqrt|minimum|maximum|cap\w*)$")


def _windows_with_calls(fi: FuncInfo):
    """(start line, end line, kind, calls inside) for each timed window, like _timer_windows."""
    calls = [n for n in _own_nodes(fi.node) if isinstance(n, ast.Call)]
    marks = []
    for c in calls:
        nm = _callee(c)
        recv = _unparse(c.func.value) if isinstance(c.func, ast.Attribute) else ""
        if nm in HOST_TIMER and (recv in ("time", "timeit", "") or recv.endswith("time")) and \
                (nm != "time" or recv == "time"):
            marks.append((c.lineno, "host", c))
        elif nm == "record" and isinstance(c.func, ast.Attribute):
            marks.append((c.lineno, "event", c))
    marks.sort(key=lambda m: (m[0], m[2].col_offset))
    out = []
    for (l1, k1, c1), (l2, k2, c2) in zip(marks, marks[1:]):
        if k1 != k2 or l2 <= l1:
            continue
        inside = [c for c in calls if l1 <= c.lineno <= l2 and c is not c1 and c is not c2
                  and not (c.lineno == l1 and c.col_offset < c1.col_offset)]
        out.append((l1, l2, k1, inside))
    return out


def _root_name(e):
    while isinstance(e, (ast.Attribute, ast.Subscript, ast.Call)):
        e = e.func if isinstance(e, ast.Call) else e.value
    return e.id if isinstance(e, ast.Name) else None


def _agent_imports(pkg: Package, pf: PyFile):
    """(names bound to the agent's code, module aliases) for a Harbor test file that imports the submission
    into its own process. A module is the agent's when the oracle writes it, the instruction names it as a
    .py file, or the test puts an agent folder on sys.path and the module is not stdlib, a test helper or a
    common library. importlib loads from an agent path count too."""
    tree = pf.tree
    if tree is None:
        return set(), set()
    parts = pf.path.split("/")
    pre = "/".join(parts[:parts.index("tests")]) + "/" if "tests" in parts and parts.index("tests") > 0 else ""
    instr = pkg.files.get(pre + "instruction.md", "")
    c = _cache(pkg).setdefault("judge_task_ctx", {})
    if pre not in c:
        c[pre] = ("\n".join(t for p, t in pkg.files.items() if p.startswith(pre + "solution/")),
                  {os.path.splitext(os.path.basename(p))[0] for p in pkg.sizes if p.startswith(pre + "tests/")})
    sol, helpers = c[pre]
    std = set(getattr(sys, "stdlib_module_names", ())) | {"pytest", "_pytest", "__future__"}
    nodes = list(ast.walk(tree))
    path_hint = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                    n.func.attr in ("insert", "append", "chdir") and
                    any(isinstance(a, ast.Constant) and isinstance(a.value, str) and AGENT_DIR.match(a.value)
                        for a in n.args) for n in nodes)

    def agent(m):
        top = m.split(".")[0]
        if top in std or top in helpers or top in THIRD_PARTY:
            return False
        if _oracle_has("module", top, sol) or re.search(r"(?<![\w-])" + re.escape(top) + r"\.(py|so)\b", instr):
            return True
        return path_hint
    names, mods = set(), set()
    for n in nodes:
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and agent(n.module):
            names |= {a.asname or a.name for a in n.names if a.name != "*"}
        elif isinstance(n, ast.Import):
            for a in n.names:
                if agent(a.name):
                    mods.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and \
                _callee(n.value) in ("module_from_spec", "import_module", "load_source", "run_path"):
            txt = _unparse(n.value) + " " + " ".join(_unparse(x) for x in ast.walk(tree) if isinstance(x, ast.Call)
                                                      and _callee(x) == "spec_from_file_location")
            if re.search(r"['\"]/(app|workspace|workdir|home|submission|solution|code)/", txt) or \
                    _callee(n.value) == "import_module":
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        mods.add(t.id)
    return names, mods


def _in_process_hits(pkg: Package, files, paid):
    """Timer or memory windows that measure the submission inside the process that loaded it."""
    from .rules_grader import _harbor_test_py
    hits = []
    seen = set()
    # Harbor verifiers: pytest files and reward scripts that import the submission
    for pf in _harbor_test_py(pkg):
        t = pf.text
        if pf.tree is None or not (re.search(r"(perf_counter|monotonic|time\.time|default_timer|\.record\(|"
                                             r"get_traced_memory|getrusage|memory_info)", t)):
            continue
        reward_file = _harbor_reward_file(pf)
        if not ((reward_file and SPEED_WORD.search(t)) or "assert" in t):
            continue
        gated = (reward_file and SPEED_WORD.search(t)) or \
            any(isinstance(n, ast.Assert) and SPEED_ASSERT.search(_unparse(n.test)) for n in ast.walk(pf.tree))
        if not gated:
            continue
        names, mods = _agent_imports(pkg, pf)
        if not names and not mods:
            continue
        if reward_file and _fail_mode_only(pkg, pf):
            continue
        passed = {a.id for n in ast.walk(pf.tree) if isinstance(n, ast.Call)
                  for a in list(n.args) + [k.value for k in n.keywords] if isinstance(a, ast.Name) and a.id in names}
        for fi in pf.funcs:
            params = {a.arg for a in fi.node.args.args + fi.node.args.kwonlyargs}
            def agent_call(c):
                r = _root_name(c.func)
                return r in names or r in mods or (r in params and bool(passed))
            for l1, l2, k, inside in _windows_with_calls(fi):
                if any(IPC_CALL.match(_callee(c)) and _callee(c) not in ("get", "read", "json") for c in inside):
                    continue
                if any(agent_call(c) for c in inside) and (pf.path, fi.name) not in seen:
                    seen.add((pf.path, fi.name))
                    hits.append((pf, fi, l1, "timer"))
            calls = [c for c in _own_nodes(fi.node) if isinstance(c, ast.Call)]
            if any(_callee(c) in MEM_CALL for c in calls) and any(agent_call(c) for c in calls) and \
                    (pf.path, fi.name) not in seen:
                seen.add((pf.path, fi.name))
                hits.append((pf, fi, min(c.lineno for c in calls if _callee(c) in MEM_CALL), "memory"))
    # verifiers kernel graders that exec or load the submission's source into the grader process
    for pf, kind in files:
        if kind != "verifiers" or not SPEED_WORD.search(pf.text):
            continue
        loads = any(isinstance(n, ast.Call) and INPROC_LOAD.match(_callee(n)) and
                    not (n.args and isinstance(n.args[0], ast.Constant)) for n in ast.walk(pf.tree))
        if not loads:
            continue
        for fi in pf.funcs:
            if fi.name not in paid and not any(fi.name in called_names(g) for _, g in all_funcs(pkg)
                                               if g.name in paid):
                continue
            for l1, l2, k, inside in _windows_with_calls(fi):
                work = [c for c in inside if not IPC_CALL.match(_callee(c)) and _callee(c) not in HOST_TIMER]
                if work and (pf.path, fi.name) not in seen:
                    seen.add((pf.path, fi.name))
                    hits.append((pf, fi, l1, "exec"))
    return hits


def _fail_mode_only(pkg: Package, pf: PyFile) -> bool:
    """A Harbor reward script that test.sh only ever runs with a flag the script handles by writing a
    failing reward and returning (frontier-swe calls compute_reward.py only with --fail). Its timing and
    reward code below that branch is never reached."""
    d = os.path.dirname(pf.path)
    sh = pkg.files.get((d + "/" if d else "") + "test.sh", "")
    base = os.path.basename(pf.path)
    if not sh or base not in sh:
        return False
    joined = sh.replace("\\\n", " ")
    calls = [ln for ln in joined.splitlines() if base in ln and not ln.strip().startswith("#")
             and re.search(r"(python|uv run|\$\{?\w*PY)", ln)]
    if not calls:
        return False
    flagsets = [set(re.findall(r"(?<![\w-])--([a-z][\w-]*)", ln)) for ln in calls]
    common = set.intersection(*flagsets) if flagsets else set()
    for f in common:
        attr = f.replace("-", "_")
        if "args." + attr not in pf.text:
            continue
        for n in ast.walk(pf.tree) if pf.tree is not None else []:
            if isinstance(n, ast.If) and re.search(r"\bargs\." + re.escape(attr) + r"\b", _unparse(n.test)) and \
                    any(isinstance(x, (ast.Return, ast.Raise)) or (isinstance(x, ast.Call) and _callee(x) == "exit")
                        for b in n.body for x in ast.walk(b)):
                return True
    return False


def _kernel_check_kind(pkg: Package) -> str | None:
    """"ast" when some grader code parses the submission and looks for kernel code or banned torch calls,
    "text" when the only check is a substring or regex search for a kernel marker, None when there is none."""
    c = _cache(pkg)
    if "kernel_check" in c:
        return c["kernel_check"]
    from .rules_grader import _harbor_test_py
    kind = None
    for pf in list(pkg.py_all()) + list(_harbor_test_py(pkg)):
        if pf.tree is None or not any(k in pf.text for k in KERNEL_PROOF + ("torch.nn.functional", "banned", "forbid")):
            continue
        for fi in pf.funcs:
            src = fi.src
            has_marker = any(k in src for k in KERNEL_PROOF)
            if "ast.parse" in src or "ast.walk" in src or "NodeVisitor" in src:
                if has_marker or re.search(r"(torch\.nn|banned|forbid|disallow|not allowed)", src, re.I):
                    kind = "ast"
                    break
            if not has_marker:
                continue
            for n in _own_nodes(fi.node):
                if isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops) and \
                        isinstance(n.left, ast.Constant) and isinstance(n.left.value, str) and \
                        any(k in n.left.value for k in KERNEL_PROOF):
                    kind = kind or "text"
                if isinstance(n, ast.Call) and _callee(n) in ("search", "match", "findall", "fullmatch", "count",
                                                              "find", "finditer") and \
                        any(isinstance(a, ast.Constant) and isinstance(a.value, str) and
                            any(k in a.value for k in KERNEL_PROOF) for a in n.args):
                    kind = kind or "text"
        if kind == "ast":
            break
    c["kernel_check"] = kind
    return kind


def _same_input_trials(fi: FuncInfo, windows):
    """A timed window inside a loop whose work call takes inputs that the loop never makes fresh."""
    for loop in _own_nodes(fi.node):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        lo, hi = loop.lineno, getattr(loop, "end_lineno", loop.lineno)
        assigned = {x.id for st in loop.body for x in ast.walk(st)
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
        fresh = any(isinstance(x, ast.Call) and FRESH_CALL.match(_callee(x)) for st in loop.body for x in ast.walk(st))
        if fresh:
            continue
        for l1, l2, k, inside in windows:
            if not (lo <= l1 <= hi):
                continue
            for c in inside:
                if IPC_CALL.match(_callee(c)) or _callee(c) in HOST_TIMER or SYNC_CALL.match(_callee(c)):
                    continue
                args = [a.value if isinstance(a, ast.Starred) else a for a in c.args]
                if args and all(isinstance(a, ast.Name) and a.id not in assigned for a in args):
                    return l1
    return None


def _num_kw(pf, fi, call, name):
    a = _kw(call, name)
    return _num(_value_of(pf, a, fi)) if a is not None else None


def _precision_asym(fi: FuncInfo):
    """A reference call on float32 data next to a candidate call on half data in one function."""
    ref_lo = cand_lo = None
    for n in _own_nodes(fi.node):
        if not isinstance(n, ast.Call):
            continue
        r = _root_name(n.func) or ""
        txt = " ".join(_unparse(a) for a in n.args)
        if REF_NAME.search(r) and re.search(r"(\.float\(\)|float32)", txt) and not re.search(r"(half|float16|bfloat16)", txt):
            ref_lo = ref_lo or n.lineno
        if CAND_NAME.search(r) and re.search(r"(\.half\(\)|float16|bfloat16|\.bfloat16\(\))", txt):
            cand_lo = cand_lo or n.lineno
    return ref_lo if ref_lo and cand_lo else None


# 0.5.0 fix wave (check_4.md): a reward that is the geometric mean of trimmed, paired speedups already damps one
# timing artifact, so an uncapped ratio there is the task's stated metric. Read from the file text only.
GEO_MEAN = re.compile(r"(geo_?mean|geometric_?mean|\bgmean\b|exp\(\s*(np\.|numpy\.|math\.|statistics\.|torch\.)?mean\(\s*"
                      r"(np\.|numpy\.|math\.|torch\.)?log)", re.I)
TRIMMED = re.compile(r"(trim|winsor|interquartile|\biqr\b)", re.I)
PAIRED = re.compile(r"(paired|pairwise|interleav)", re.I)


def _trimmed_paired_geomean(pf) -> bool:
    t = pf.text
    return bool(GEO_MEAN.search(t) and TRIMMED.search(t) and PAIRED.search(t))


def _unclipped_speedup(fi: FuncInfo):
    """Line of a return or reward write that pays a speedup ratio with no upper clip."""
    src = fi.src
    if not re.search(r"speed_?up", src, re.I):
        return None
    nodes = list(_own_nodes(fi.node))
    clipped = set()
    for n in nodes:
        if isinstance(n, ast.Call) and CLIP_CALL.match(_callee(n)):
            clipped |= {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare):
            clipped |= {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)}
    sp_names = {x.id for x in ast.walk(fi.node) if isinstance(x, ast.Name) and re.search(r"speed_?up", x.id, re.I)}
    sp_names -= clipped
    sp_names -= {n.func.id for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}  # a callee
    if not sp_names:
        return None
    for n in nodes:
        val = None
        if isinstance(n, ast.Return) and n.value is not None:
            val = n.value
        elif isinstance(n, ast.Call) and _callee(n) in ("write", "write_text", "dump") and n.args and \
                re.search(r"reward", _unparse(n), re.I):
            val = n.args[0]
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and re.match(r"^(reward|score)$", t.id)
                                               for t in n.targets):
            val = n.value
        if val is None or isinstance(val, (ast.Compare, ast.BoolOp, ast.Dict, ast.Tuple, ast.List)):
            continue
        if any(isinstance(x, ast.Call) and CLIP_CALL.match(_callee(x)) for x in ast.walk(val)):
            continue
        if any(isinstance(x, (ast.Compare, ast.IfExp)) for x in ast.walk(val)):
            continue
        names = {x.id for x in ast.walk(val) if isinstance(x, ast.Name)}
        if names & sp_names:
            return n.lineno
    return None


def _alias_order(fi: FuncInfo):
    """Line where the candidate runs on input names that the reference then reads, with no clone between."""
    calls = sorted([n for n in _own_nodes(fi.node) if isinstance(n, ast.Call)], key=lambda c: (c.lineno, c.col_offset))
    stores = [(x.lineno, x.id) for x in _own_nodes(fi.node) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)]
    for i, c in enumerate(calls):
        r = _root_name(c.func) or ""
        if not CAND_NAME.search(r):
            continue
        args = {(a.value if isinstance(a, ast.Starred) else a).id for a in c.args
                if isinstance(a.value if isinstance(a, ast.Starred) else a, ast.Name)}
        if not args:
            continue
        for d in calls[i + 1:]:
            rd = _root_name(d.func) or ""
            if not REF_NAME.search(rd) or CAND_NAME.search(rd):
                continue
            dargs = {(a.value if isinstance(a, ast.Starred) else a).id for a in d.args
                     if isinstance(a.value if isinstance(a, ast.Starred) else a, ast.Name)}
            common = args & dargs
            if not common:
                continue
            between = [x for x in calls if c.lineno <= x.lineno <= d.lineno and x is not c and x is not d]
            if any(_callee(x) in ("clone", "deepcopy", "copy", "copy_") for x in between):
                continue
            if any(c.lineno < ln < d.lineno and nm in common for ln, nm in stores):
                continue
            return c.lineno
    return None


@rule("kernel-perf-grader", "hack", "high",
      "A GPU kernel grader times or checks the submission in a way a kernel can game: the clock stops with no "
      "device-wide sync, or the correctness check allows a large absolute error.")
def r_kernel_perf(pkg: Package):
    files = _kernel_files(pkg)
    from .rules_grader import _harbor_test_py
    if not files and not any(re.search(r"(perf_counter|monotonic|time\.time|default_timer|get_traced_memory|getrusage)",
                                       pf.text) for pf in _harbor_test_py(pkg)):
        return []
    paid = None
    timing, tol, cmp_sites, same_in, tf32, asym, unclipped, alias = [], [], [], [], [], [], [], []
    for pf, kind in files:
        if kind == "verifiers":
            if paid is None:
                paid = _paid_path_names(pkg)
            if not paid:
                continue
        if kind == "harbor" and not SPEED_WORD.search(pf.text) and "allclose" not in pf.text:
            continue
        speed_file = bool(SPEED_WORD.search(pf.text))
        half_file = re.search(r"(\.half\(|float16|bfloat16)", pf.text) is not None
        tf32_set = "allow_tf32" in pf.text or "float32_matmul_precision" in pf.text or "fp32_precision" in pf.text
        for fi in pf.funcs:
            if kind == "verifiers" and fi.name not in paid:
                continue
            windows = []
            if "record" in fi.src or re.search(r"(perf_counter|monotonic|time\.time|default_timer)", fi.src):
                for l1, l2, has_sync, work, wk in _timer_windows(fi):
                    if work and not has_sync and speed_file:
                        timing.append((pf, fi, l1, wk))
                windows = _windows_with_calls(fi)
                if speed_file and windows:
                    ln = _same_input_trials(fi, windows)
                    if ln:
                        same_in.append((pf, fi, ln))
            if MULTI_PROBLEM.search(pf.text):
                for n in _own_nodes(fi.node):
                    if isinstance(n, ast.Call) and _callee(n) in CMP_CALL:
                        cmp_sites.append((pf, fi, n.lineno))
                        x = _num_kw(pf, fi, n, "atol")
                        r = _num_kw(pf, fi, n, "rtol")
                        if x is not None and x > 1e-3:
                            tol.append((pf, fi, n.lineno, x))
                        big = max([v for v in (x, r) if v is not None], default=None)
                        if big is not None and big >= 1e-2 and speed_file and not tf32_set:
                            tf32.append((pf, fi, n.lineno, big))
                ln = _alias_order(fi)
                if ln:
                    alias.append((pf, fi, ln))
            ln = _precision_asym(fi) if half_file else None
            if ln and speed_file:
                asym.append((pf, fi, ln))
            ln = _unclipped_speedup(fi)
            if ln and not (kind == "harbor" and _fail_mode_only(pkg, pf)) and not _trimmed_paired_geomean(pf):
                unclipped.append((pf, fi, ln))
    if paid is None:
        paid = _paid_path_names(pkg) if any(k == "verifiers" for _, k in files) else set()
    inproc = _in_process_hits(pkg, files, paid)
    out = []
    if timing:
        pf, fi, ln, wk = timing[0]
        why = ("the end event is recorded on the current stream with no device-wide sync before it, so a kernel that "
               "runs its work on a side stream is timed only for the launch") if wk == "event" else \
              ("the clock stops as soon as the call returns, with no device-wide sync, so a submission whose GPU work "
               "is still queued when it returns is timed only for the launch")
        out.append(F("kernel-perf-grader", f"{fi.name} times GPU work but {why}, and the measured time sets a paid speed "
                                           "score, so that submission is paid a large fake speedup.",
                     [ev(p, l) for p, _, l, _ in timing], confidence="review"))
    if tol:
        pf, fi, ln, x = tol[0]
        out.append(F("kernel-perf-grader", f"{fi.name} accepts any output within atol={x:g} of the reference, so on a "
                                           "problem whose outputs are all smaller than that (KernelBench's softmax "
                                           "problem averages about 2.5e-6) a kernel that returns zeros passes as correct.",
                     [ev(p, l) for p, _, l, _ in tol], confidence="review"))
    if inproc:
        pf, fi, ln, how = inproc[0]
        what = "memory use" if how == "memory" else "time"
        load = "execs or imports the submission's code" if how == "exec" else "imports the submission"
        out.append(F("kernel-perf-grader", f"{fi.name} measures the submission's {what} inside the same process that "
                                           f"{load}, so the submission can replace the timer or the functions the "
                                           f"grader calls when it loads and report any {what} it likes, which passes "
                                           "the speed check.",
                     [ev(p, l) for p, _, l, _ in inproc], confidence="review"))
    if cmp_sites:
        ck = _kernel_check_kind(pkg)
        if ck != "ast":
            pf, fi, ln = cmp_sites[0]
            why = ("the only proof of a custom kernel is a text search for a marker such as __global__, so a ModelNew "
                   "that puts the marker in a comment and calls the same torch ops as the reference passes") \
                if ck == "text" else ("nothing checks that ModelNew runs a custom kernel, so a ModelNew that calls the "
                                      "same torch ops as the reference, or wraps the reference, passes")
            out.append(F("kernel-perf-grader", f"{fi.name} pays a kernel that matches the reference but {why}.",
                         [ev(p, l) for p, _, l in cmp_sites], confidence="review"))
    if same_in:
        pf, fi, ln = same_in[0]
        out.append(F("kernel-perf-grader", f"{fi.name} times every trial on the same input objects, so a kernel that "
                                           "saves its output on the first call and returns it again for the same input "
                                           "is timed only for that lookup and paid a fake speedup.",
                     [ev(p, l) for p, _, l in same_in], confidence="review"))
    if asym:
        pf, fi, ln = asym[0]
        out.append(F("kernel-perf-grader", f"{fi.name} runs the reference on float32 data and the candidate on half "
                                           "precision data, so the paid speedup comes from lower precision and not "
                                           "from a better kernel.",
                     [ev(p, l) for p, _, l in asym], confidence="review"))
    elif tf32:
        pf, fi, ln, x = tf32[0]
        out.append(F("kernel-perf-grader", f"{fi.name} compares against an FP32 reference with TF32 never set and a "
                                           f"tolerance of {x:g}, so a submission that computes matmuls in TF32 passes and "
                                           "part of its paid speedup comes from lower precision, not a better kernel.",
                     [ev(p, l) for p, _, l, _ in tf32], confidence="review"))
    if alias:
        pf, fi, ln = alias[0]
        out.append(F("kernel-perf-grader", f"{fi.name} runs the candidate before the reference on the same input "
                                           "tensors with no clone, so a candidate that zeros its inputs in place and "
                                           "returns zeros matches the reference on any problem that maps zeros to zeros.",
                     [ev(p, l) for p, _, l in alias], confidence="review"))
    if unclipped:
        pf, fi, ln = unclipped[0]
        out.append(F("kernel-perf-grader", f"{fi.name} pays the speedup ratio with no upper clip, so one timing "
                                           "artifact on one rollout can pay many times the normal reward.",
                     [ev(p, l) for p, _, l in unclipped], confidence="review", kind="hygiene", severity="low"))
    return out

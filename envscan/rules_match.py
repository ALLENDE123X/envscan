"""Answer matching and parsing rules (envscan v0.5, builder `reward`).

Rules about how the policy's answer is pulled out of its reply and matched to the gold answer:
an exact compare with no normalization, an extractor that needs a marker the prompt never asks for,
a tool call parser that accepts malformed calls, and sampling settings that cut the answer off.
Like every envscan rule these read source text and ASTs only. All findings are `review`.

Each rule reports only when the flagged function is paid (a rubric weight above 0 on it or on a
direct caller). Reader item ids (ST....) refer to research/mine/merge/static_rule.md.
"""
from __future__ import annotations

import ast
import json
import re

from .package import Package, PyFile, FuncInfo, _const_text
from .rules import F, all_funcs, called_names, ev, rule
from .rules_grader import _cache, _callee, _grader_funcs, _num, _own_nodes, _reward_weights
from .rules_reward import MODEL_PARAM, _assigned, _loads, _paid_weight, _shipped, _str_key

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _local_funcs(pkg: Package) -> dict[str, FuncInfo]:
    c = _cache(pkg)
    if "match_local_funcs" not in c:
        m = {}
        for _, fi in all_funcs(pkg):
            m.setdefault(fi.name, fi)
        c["match_local_funcs"] = m
    return c["match_local_funcs"]


def _paid_graders(pkg: Package):
    """(pf, fi) on the grading path that a positive rubric weight pays (itself or a direct caller)."""
    c = _cache(pkg)
    if "match_paid" not in c:
        if not _reward_weights(pkg):
            c["match_paid"] = []
        else:
            c["match_paid"] = [(pf, fi) for pf, fi in _grader_funcs(pkg)
                               if _shipped(pkg, pf.path) and _paid_weight(pkg, fi.name) > 0]
    return c["match_paid"]


def _defs(own) -> dict[str, list[ast.AST]]:
    out: dict[str, list[ast.AST]] = {}
    for names, val in _assigned(own):
        for nm in names:
            out.setdefault(nm, []).append(val)
    return out


def _chain(e, defs, depth=4) -> list[ast.AST]:
    """e plus the values assigned to every name it loads, followed back `depth` steps."""
    out, seen, frontier = [e], set(), _loads(e)
    for _ in range(depth):
        nxt = set()
        for nm in frontier - seen:
            seen.add(nm)
            for v in defs.get(nm, []):
                out.append(v)
                nxt |= _loads(v)
        frontier = nxt
    return out


# ---------------------------------------------------------------------------
# exact-compare-no-normalization
# Reader items reader items. Main paper arxiv:2611.14765 (56 to 61% of failures were
# format only, agreement 90 to 94% after normalization).
# A paid reward compares the parsed answer to gold with == and no strip, case fold, number parse,
# comma or unit removal or math check on either side. Only fires when the gold answers look numeric
# or symbolic (a boxed or #### extractor, a math dataset, or numeric answers in shipped rows), since a
# fixed label set is usually matched exactly on purpose.
# ---------------------------------------------------------------------------

NORM_CALL = re.compile(r"(norm|strip|lower|upper|casefold|clean|canon|standard|^float$|^int$|decimal|fraction|sympy|"
                       r"simplif|verify|parse_expr|latex|^sub$|replace|round|isclose|allclose|literal_eval|^eval$|numer|"
                       r"to_num|unit|math_equal|is_equiv|equiv|grade|compare|match_answer|fuzzy|ratio|translate|"
                       r"remove|unicodedata|digits|parse_number|parse_float|tonum)", re.I)
GOLD_NAME = re.compile(r"^(answer|answers|gold\w*|ground_truth|gt|expected\w*|target|reference|ref|solution|label|truth|"
                       r"correct_answer|final_answer|true_answer)$", re.I)
MATH_HINT = re.compile(r"(gsm8k|\bmath\b|math[-_]|aime|amc\b|numina|countdown|arith|deepscaler|olymp|minerva|svamp|asdiv|"
                       r"mathqa|hendrycks|competition_math|polymath|omni[-_]?math|theoremqa)", re.I)
BOXED_PARSE = re.compile(r"(extract_boxed_answer|last_boxed_only|boxed|####)")
MCQ_HINT = re.compile(r"(multiple[- ]choice|\bchoices\b|\boptions\b|answer letter|letter of|\bmcq\b|gpqa|mmlu|"
                      r"\bA\)|\(A\)|\bA-[D-J]\b|ABCD)", re.I)
NUMERIC = re.compile(r"^\s*[-+]?\$?\s*[\d,]*\.?\d+\s*(%|[a-z]{1,6})?\s*$|^\s*\\?frac|^\s*[-+]?\d+\s*/\s*\d+\s*$")


def _helper_normalizes(pkg: Package, name: str, depth: int = 2) -> bool:
    fi = _local_funcs(pkg).get(name)
    if fi is None or depth <= 0:
        return False
    for c in ast.walk(fi.node):
        if isinstance(c, ast.Call) and c is not fi.node:
            cn = _callee(c)
            if NORM_CALL.search(cn) or _helper_normalizes(pkg, cn, depth - 1):
                return True
    return False


def _normalized(pkg: Package, exprs) -> bool:
    for e in exprs:
        for c in ast.walk(e):
            if isinstance(c, ast.Call):
                cn = _callee(c)
                if NORM_CALL.search(cn) or _helper_normalizes(pkg, cn):
                    return True
    return False


def _gold_side(e, chain, fi: FuncInfo) -> bool:
    for x in chain:
        for n in ast.walk(x):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and GOLD_NAME.match(n.id) and n.id in fi.params:
                return True
            k = _str_key(n)
            if k and GOLD_NAME.match(k):
                return True
            if isinstance(n, ast.Attribute) and GOLD_NAME.match(n.attr):
                return True
    return False


def _model_side(chain, fi: FuncInfo) -> bool:
    params = set(fi.params) & MODEL_PARAM
    if not params:
        return False
    return any(_loads(x) & params for x in chain)


def _numeric_gold(pkg: Package) -> str | None:
    """Why the gold answers look numeric or symbolic, else None."""
    c = _cache(pkg)
    if "match_numeric" in c:
        return c["match_numeric"]
    why = None
    text = "\n".join(pf.text for pf in pkg.py_all())
    if MCQ_HINT.search(text):
        c["match_numeric"] = None
        return None  # a letter label is matched exactly on purpose
    if re.search(r"extract_boxed_answer|last_boxed_only|\\\\boxed|['\"]####", text):
        why = "a boxed or #### extractor"
    else:
        m = re.search(r"load_dataset\(\s*['\"]([^'\"]+)['\"]", text)
        if m and MATH_HINT.search(m.group(1)):
            why = f"the math dataset {m.group(1)}"
    if why is None:
        vals = []
        for p, t in pkg.files.items():
            if not p.endswith((".jsonl", ".json")) or len(vals) > 200:
                continue
            for ln in t.splitlines()[:100]:
                ln = ln.strip()
                if not ln.startswith("{"):
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                if isinstance(row, dict):
                    for k in ("answer", "gold", "solution", "target", "expected_answer", "final_answer"):
                        v = row.get(k)
                        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                            vals.append(str(v))
                            break
        if len(vals) >= 5 and sum(1 for v in vals if NUMERIC.match(v)) >= 0.8 * len(vals):
            why = "numeric answers in the shipped rows"
    c["match_numeric"] = why
    return why


def _flows_to_return(n, fi: FuncInfo, parent) -> bool:
    p = parent.get(n)
    while p is not None and p is not fi.node:
        if isinstance(p, ast.Return):
            return True
        if isinstance(p, ast.If) and n in ast.walk(p.test):
            return any(isinstance(x, ast.Return) and x.value is not None and (_num(x.value) or 0) > 0
                       for s in p.body for x in ast.walk(s))
        if isinstance(p, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tg = p.targets if isinstance(p, ast.Assign) else [p.target]
            names = {x.id for t in tg for x in ast.walk(t) if isinstance(x, ast.Name)}
            return any(isinstance(r, ast.Return) and r.value is not None and _loads(r.value) & names
                       for r in _own_nodes(fi.node))
        p = parent.get(p)
    return False


# 0.5.0 fix wave (check_3.md): an exact compare on a small closed label set (0/1, 1/2/3) is correct.
LABEL_KEY = re.compile(r"['\"](answer|gold|target|label)['\"]\s*[:=]\s*(str\(|int\()?\s*\w+\[\s*['\"]"
                       r"(bucket|label|class|category|choice|option|verdict)\w*['\"]\s*\]")


def _short_label(v) -> bool:
    return isinstance(v, ast.Constant) and isinstance(v.value, (str, int)) and not isinstance(v.value, bool) \
        and len(str(v.value)) <= 12


def _closed_label_gold(pkg: Package, fi: FuncInfo) -> bool:
    """The gold answers are a small fixed set of labels: the reward checks membership in a literal list of
    2 to 6 short labels, the dataset builds the gold from a label or bucket field, or the shipped rows
    hold at most 6 distinct short values."""
    for n in _own_nodes(fi.node):
        if isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
            for c in n.comparators:
                if isinstance(c, (ast.List, ast.Tuple, ast.Set)) and 2 <= len(c.elts) <= 6 and \
                        all(_short_label(e) for e in c.elts):
                    return True
    c = _cache(pkg)
    if "match_labels" not in c:
        text = "\n".join(pf.text for pf in pkg.py_all())
        closed = bool(LABEL_KEY.search(text))
        if not closed:
            vals = []
            for p, t in pkg.files.items():
                if not p.endswith((".jsonl", ".json")) or len(vals) > 400:
                    continue
                for ln in t.splitlines()[:200]:
                    ln = ln.strip()
                    if not ln.startswith("{"):
                        continue
                    try:
                        row = json.loads(ln)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        for k in ("answer", "gold", "solution", "target", "expected_answer", "final_answer"):
                            v = row.get(k)
                            if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                                vals.append(str(v).strip())
                                break
            closed = len(vals) >= 20 and len(set(vals)) <= 6 and all(len(v) <= 12 for v in vals)
        c["match_labels"] = closed
    return c["match_labels"]


@rule("exact-compare-no-normalization", "broken", "medium",
      "A paid reward compares the parsed answer to a numeric or symbolic gold answer with plain equality "
      "and no normalizing step, so a correct answer written another way is scored as wrong.")
def r_exact_compare(pkg: Package):
    paid = _paid_graders(pkg)
    if not paid:
        return []
    cands = [(pf, fi) for pf, fi in paid if "==" in fi.src and set(fi.params) & MODEL_PARAM]
    if not cands:
        return []
    why = _numeric_gold(pkg)
    if why is None:
        return []
    hits = []
    for pf, fi in cands:
        own = list(_own_nodes(fi.node))
        defs = None
        parent = None
        for n in own:
            if not (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)):
                continue
            a, b = n.left, n.comparators[0]
            if _num(a) is not None or _num(b) is not None or isinstance(a, ast.Constant) or isinstance(b, ast.Constant):
                continue
            if defs is None:
                defs = _defs(own)
                parent = {}
                for x in [fi.node] + own:
                    for ch in ast.iter_child_nodes(x):
                        parent[ch] = x
            ca, cb = _chain(a, defs), _chain(b, defs)
            pair = None
            if _model_side(ca, fi) and _gold_side(b, cb, fi) and not _model_side(cb, fi):
                pair = (ca, cb)
            elif _model_side(cb, fi) and _gold_side(a, ca, fi) and not _model_side(ca, fi):
                pair = (cb, ca)
            if pair is None or _normalized(pkg, pair[0]) or _normalized(pkg, pair[1]):
                continue
            if not _flows_to_return(n, fi, parent):
                continue
            if _closed_label_gold(pkg, fi):
                break
            hits.append((pf, fi, n.lineno))
            break
    if not hits:
        return []
    pf, fi, ln = hits[0]
    return [F("exact-compare-no-normalization",
              f"{fi.name} pays only when the parsed answer equals the gold answer exactly, with no strip, number "
              f"parse or math check on either side, and the task has {why}, so an answer like 72.0 or 1,000 for "
              "72 or 1000 is scored 0.",
              [ev(p, l) for p, _, l in hits], confidence="review")]


# ---------------------------------------------------------------------------
# extractor-prompt-format-mismatch
# Reader items reader items. Main paper arxiv:2609.33411 (GPT-5.4 0.258 to 0.591 after the
# prompt asked for the marker), arxiv:2610.11012 (a strict parser inflated a gain about 15%).
# The answer extractor needs a marker (\boxed, ####, an <answer> tag, a "Final Answer:" line) that no
# prompt text in the package asks for. Fires only when the package writes its own prompt (a system
# prompt or a prompt template) so the instruction would have to be there.
# ---------------------------------------------------------------------------

PROMPT_KW = re.compile(r"(system_prompt|SYSTEM_PROMPT|system_message|SYSTEM_MSG|prompt_template|PROMPT_TEMPLATE|"
                       r"instruction|INSTRUCTION)")


def _prompt_strings(pkg: Package) -> tuple[bool, list[str]]:
    """(package writes its own prompt, string literals and data text a prompt could come from)."""
    own_prompt = False
    parts = []
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        if PROMPT_KW.search(pf.text):
            own_prompt = True
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > 3:
                parts.append(n.value)
    for p, t in pkg.files.items():
        if p.endswith((".jsonl", ".json", ".txt", ".md", ".yaml", ".yml", ".j2", ".jinja", ".jinja2", ".toml")):
            parts.extend(t[:200_000].splitlines())
    return own_prompt, parts


REGEXY = re.compile(r"(\(\.\*|\(\?|\\s|\\d|\[\^|\.\*\?)")


def _marker_instructed(mk, parts: list[str]) -> bool:
    """A string that names the marker inside prose (three words in a row), not the extractor's own regex."""
    for t in parts:
        if not mk.search(t) or REGEXY.search(t):
            continue
        if re.search(r"[a-z]{2,}\s+[a-z]{2,}\s+[a-z]{2,}", t, re.I):
            return True
    return False


# a prompt constant imported from a library (verifiers' BOXED_SYSTEM_PROMPT, an AnswerFormat option) holds
# the instruction where this scan cannot read it
LIB_PROMPT = re.compile(r"(\w*(BOXED|XML|ANSWER|FORMAT)\w*_PROMPT\b|\bAnswerFormat\b|\w*_PROMPT_(BOXED|XML)\w*)")
FORMAT_HELPER = re.compile(r"(get_format_str|format_prompt|format_str\(|format_instructions)")


EXTRACT_SITE = [
    ("\\boxed{}", re.compile(r"\\boxed\\?\{?\s*\(|\\\\boxed\\\{\(")),
    ("####", re.compile(r"^#{4}$|#{4}\s*(\\s\*)?\s*\(")),
    ("<answer>", re.compile(r"<answer>\s*(\\s\*)?\(")),
    ("Final Answer:", re.compile(r"^\(?\??(final answer|answer)\s*:\s*(\\s\*)?\s*\(", re.I)),
]
MARKER_WORD = {"\\boxed{}": re.compile(r"boxed", re.I), "####": re.compile(r"#{4}"),
               "<answer>": re.compile(r"<answer>|answer>|\banswer tags?\b", re.I),
               "Final Answer:": re.compile(r"answer\s*:", re.I)}


def _fallback(fi: FuncInfo) -> bool:
    """The extractor tries something else when the marker is missing: a second pattern, or a second
    return of an extracted value."""
    regex_calls = [c for c in _own_nodes(fi.node) if isinstance(c, ast.Call) and
                   _callee(c) in ("search", "findall", "finditer", "match", "fullmatch", "split", "rsplit", "rfind", "find")]
    lits = {(_const_text(c.args[0]) or ast.unparse(c.args[0])) for c in regex_calls if c.args}
    if len(lits) >= 2:
        return True
    rets = [r for r in _own_nodes(fi.node) if isinstance(r, ast.Return) and r.value is not None
            and not isinstance(r.value, ast.Constant)]
    loops = any(isinstance(n, (ast.For, ast.While)) for n in _own_nodes(fi.node))
    return loops or len(rets) >= 2 and not _reads_gold_in(fi)


def _reads_gold_in(fi: FuncInfo) -> bool:
    from .rules_reward import _reads_gold
    return _reads_gold(fi)


@rule("extractor-prompt-format-mismatch", "broken", "medium",
      "The answer extractor needs a marker such as a boxed answer or an answer tag that the prompt never asks "
      "for, so correct answers written without it are scored as wrong.")
def r_extractor_mismatch(pkg: Package):
    paid = _paid_graders(pkg)
    if not paid:
        return []
    from .rules_reward import _callers, _reads_gold
    local = _local_funcs(pkg)
    paid_names = {fi.name for _, fi in paid}
    sites = []
    for pf, fi in _grader_funcs(pkg):
        if not _shipped(pkg, pf.path):
            continue
        for n in _own_nodes(fi.node):
            if not (isinstance(n, ast.Constant) and isinstance(n.value, str)) or len(n.value) > 200:
                continue
            for lab, site in EXTRACT_SITE:
                if site.search(n.value.strip()):
                    sites.append((pf, fi, n.lineno, lab))
    if not sites:
        return []
    own_prompt, text = _prompt_strings(pkg)
    if not own_prompt or FORMAT_HELPER.search(pkg.all_py_text()) or LIB_PROMPT.search(pkg.all_py_text()):
        return []
    for pf, fi, ln, lab in sites:
        # the extracted value is the answer the reward checks: this function or a paid caller reads gold
        callers = [local[c] for c in _callers(pkg).get(fi.name, ()) if c in local]
        if not ((fi.name in paid_names and _reads_gold(fi)) or
                any(c.name in paid_names and _reads_gold(c) for c in callers) or
                (fi.name in paid_names and any(_reads_gold(c) for c in callers))):
            continue
        if _fallback(fi) or _marker_instructed(MARKER_WORD[lab], text):
            continue
        return [F("extractor-prompt-format-mismatch",
                  f"The reward pulls the answer out of a {lab} marker with no fallback, but no prompt or data text in "
                  f"the package asks the model to write {lab}, so a correct answer without it scores 0.",
                  [ev(pf, ln)], confidence="review")]
    return []


# ---------------------------------------------------------------------------
# tool-call-parser-lenient
# Reader items reader items. Main paper url:consensus.app w:25f2f0ce (0/24 vs 24/24 for the same
# generations by parser), arxiv:2610.24914 (0/13 vs 6/13 malformed turns rewarded).
# A parser on the paid path takes a tool call with an optional closing tag or repairs broken JSON
# before the reward compares it, so a malformed call is paid like a well formed one.
# ---------------------------------------------------------------------------

TOOL_OPEN = r"(<tool_call>|<function_call>|<functioncall>|<tool_use>|<\|tool_call\|>|\[TOOL_CALLS\]|<function=)"
OPTIONAL_CLOSE = re.compile(TOOL_OPEN + r".*(\|\s*\$|\$\s*\)|</\w+>\)\?|</\w+>\?|\(\?:</\w+>\)\?)")
NO_CLOSE = re.compile(TOOL_OPEN + r"\s*(\\s\*)?\((\?P<\w+>)?\.\*\)\s*$")
REPAIR_CALL = re.compile(r"^(repair_json|json_repair|loads_lenient|demjson\w*|dirtyjson|partial_json\w*|parse_partial\w*|"
                         r"fix_json|repair)$")
REPAIR_MOD = re.compile(r"\b(json_repair|demjson3?|dirtyjson|partial_json_parser)\b")


def _callees_deep(pkg: Package, fi: FuncInfo, depth=2) -> list[FuncInfo]:
    out, frontier, seen = [fi], [fi], {fi.name}
    local = _local_funcs(pkg)
    for _ in range(depth):
        nxt = []
        for f in frontier:
            for cn in called_names(f):
                g = local.get(cn)
                if g is not None and g.name not in seen:
                    seen.add(g.name)
                    nxt.append(g)
        out += nxt
        frontier = nxt
    return out


@rule("tool-call-parser-lenient", "hack", "medium",
      "A tool call parser on the reward path accepts a call with no closing tag or repairs broken JSON, so a "
      "malformed tool call is paid like a well formed one.")
def r_toolcall_lenient(pkg: Package):
    if not re.search(r"tool_call|function_call|functioncall|tool_use|TOOL_CALLS|function=|json_repair|repair_json|"
                     r"demjson|dirtyjson|partial_json", pkg.all_py_text()):
        return []
    paid = _paid_graders(pkg)
    if not paid:
        return []
    paths = {pf.path: pf for pf in pkg.py_all()}
    fpath = {}
    for pf, fi in all_funcs(pkg):
        fpath.setdefault((fi.name, fi.start), pf)
    hits, seen = [], set()
    for pf, fi in paid:
        for g in _callees_deep(pkg, fi):
            if (g.name, g.start) in seen:
                continue
            seen.add((g.name, g.start))
            gpf = fpath.get((g.name, g.start), pf)
            for n in _own_nodes(g.node):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 400:
                    s = n.value
                    if OPTIONAL_CLOSE.search(s) or NO_CLOSE.search(s):
                        hits.append((gpf, n.lineno, "takes a tool call whose closing tag may be missing"))
                elif isinstance(n, ast.Call):
                    cn = _callee(n)
                    recv = ast.unparse(n.func.value) if isinstance(n.func, ast.Attribute) else ""
                    if REPAIR_CALL.match(cn) or REPAIR_MOD.search(recv):
                        hits.append((gpf, n.lineno, "repairs broken tool call JSON"))
    if not hits:
        return []
    pf, ln, what = hits[0]
    return [F("tool-call-parser-lenient",
              f"The parser on the reward path {what} before the reward checks the call, so a malformed tool call "
              "that a strict parser would reject is paid.",
              [ev(p, l) for p, l, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# empty-answer-from-truncation
# Reader items reader items. Main paper arxiv:2609.08765 (fixing stop and token settings moved
# RedSage 0.0 to 85.9).
# The env sets the policy's sampling args itself, with a stop string the answer needs (the closing
# tag the extractor waits for, or the end of the think block before the answer), or a max_tokens
# cap far below a reasoning reply while the prompt asks for step by step reasoning.
# ---------------------------------------------------------------------------

REASON_ASK = re.compile(r"(step by step|step-by-step|<think>|think through|show your (work|reasoning)|reason carefully|"
                        r"chain of thought|explain your reasoning)", re.I)


def _sampling_dicts(pf: PyFile):
    """(line, {key: ast value}) for literal sampling args dicts the env passes for the policy."""
    out = []
    if pf.tree is None or "sampling_args" not in pf.text:
        return out
    for n in ast.walk(pf.tree):
        val = None
        if isinstance(n, ast.keyword) and n.arg == "sampling_args":
            val = n.value
        elif isinstance(n, ast.Assign) and any(re.search(r"sampling_args$", ast.unparse(t)) for t in n.targets):
            val = n.value
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript) and \
                re.search(r"sampling_args", ast.unparse(n.targets[0].value)):
            k = n.targets[0].slice
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                out.append((n.lineno, {k.value: n.value}))
            continue
        if isinstance(val, ast.Dict):
            d = {k.value: v for k, v in zip(val.keys, val.values) if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            out.append((val.lineno, d))
    return out


@rule("empty-answer-from-truncation", "broken", "medium",
      "The env's own sampling settings stop or cut the reply before the answer the extractor needs, so correct "
      "work ends with an empty or unparsable answer.")
def r_truncation(pkg: Package):
    paid = _paid_graders(pkg)
    if not paid:
        return []
    sd = [(pf, ln, d) for pf in pkg.py_all() for ln, d in _sampling_dicts(pf)
          if set(d) & {"stop", "max_tokens", "max_completion_tokens", "max_new_tokens"}]
    if not sd:
        return []
    ptext = "\n".join(fi.src for _, fi in paid) + "\n" + "\n".join(
        m.group(0) for pf in pkg.py_all() for m in re.finditer(r"Parser\([^\n]*\)", pf.text))
    strings = None
    for pf, ln, d in sd:
        stop = d.get("stop")
        stops = []
        if isinstance(stop, (ast.List, ast.Tuple)):
            stops = [e.value for e in stop.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        elif isinstance(stop, ast.Constant) and isinstance(stop.value, str):
            stops = [stop.value]
        for s in stops:
            if not s.strip():
                continue
            closing = re.fullmatch(r"</(\w+)>", s.strip())
            if closing and re.search(re.escape(s.strip()), ptext):
                return [F("empty-answer-from-truncation",
                          f"The env stops sampling at `{s.strip()}`, but the answer extractor needs that closing tag, "
                          "and a stop string is cut from the reply, so the tag is never there and the answer is lost.",
                          [ev(pf, ln)], confidence="review")]
            if s.strip() == "</think>":
                return [F("empty-answer-from-truncation",
                          "The env stops sampling at `</think>`, so the reply ends before any answer that would follow "
                          "the think block and the extractor finds nothing.",
                          [ev(pf, ln)], confidence="review")]
        mt = d.get("max_tokens") or d.get("max_completion_tokens") or d.get("max_new_tokens")
        x = _num(mt) if mt is not None else None
        if x is not None and 0 < x < 512 and strings is None:
            strings = "\n".join(n.value for f in pkg.py_all() if f.tree is not None for n in ast.walk(f.tree)
                                if isinstance(n, ast.Constant) and isinstance(n.value, str))
        if x is not None and 0 < x < 512 and REASON_ASK.search(strings):
            return [F("empty-answer-from-truncation",
                      f"The env caps the policy at {x:g} tokens while its prompt asks for step by step reasoning "
                      "before the answer, so many correct replies are cut before the answer.",
                      [ev(pf, ln)], confidence="review")]
    return []

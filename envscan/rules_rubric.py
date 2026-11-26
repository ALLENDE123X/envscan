"""Rubric family rules (envscan v0.5).

Static reads of LLM judge prompt templates, judge configuration and rubric criteria lists that a
package ships (Python literals, JSON, JSONL or YAML files, or weighted lists written into a judge
prompt). No rule here calls a model or runs env code. Every finding is `review`. Most of these are
quality risks, so they report as `hygiene`. The ones that can pay a rollout for work it did not do
report as `hack`.

Shared pieces:
  _tpls(pkg)     judge prompt templates on the paid reward path, with what fills each slot
  _rubrics(pkg)  rubric criteria lists the package ships and scores with

Builder notes: the internal notes. Reader item ids are in the internal notes.
"""
from __future__ import annotations

import ast
import json
import os
import re

from .package import Package, PyFile, FuncInfo, _const_text
from .rules import F, all_funcs, ev, line_of, referenced_consts, rule, r_judge_undelimited
from .rules_grader import (_cache, _callee, _grader_funcs, _harbor_test_py, _num, _own_nodes, _reward_weights,
                           JUDGE_CONTEXT)
from .rules_judge import _judge_rubrics, _kw, _paid, _value_of, _code_judge_models, _train_configs

# ---------------------------------------------------------------------------
# judge templates
# ---------------------------------------------------------------------------

# verifiers JudgeRubric default prompt (verifiers/rubrics/judge_rubric.py). It fills {question},
# {answer} (the dataset's answer column) and {response} with str.format.
JR_DEFAULT = ("Given a ground truth answer and a response, determine if the response is correct.\n\n"
              "Question:\n```\n{question}\n```\n\nGround truth answer:\n```\n{answer}\n```\n\n"
              "Response:\n```\n{response}\n```\n\nRespond either \"yes\" or \"no\" only.")
JR_DEFAULT_MODEL = "gpt-4.1-nano"

SLOT = re.compile(r"(?<!\{)\{\s*([A-Za-z_][\w.\[\]'\"()]*?)\s*(?:![rsa])?(?::[^{}]*)?\}(?!\})")
RESP_SRC = re.compile(r"(response|completion|output|prediction|predicted|reply|generat|parsed|candidate|submission|"
                      r"model_answer|student|extracted|final_answer|agent_answer|\bcode\b|code_text|code_blocks|attempt|"
                      r"essay|poem|rollout|trajectory|transcript|\bpred\b|assistant|model_text|answer_text|\btext\b)", re.I)
GOLD_SRC = re.compile(r"(gold|reference|\bref\b|ref_|_ref\b|target|expected|ground_?truth|\bgt\b|truth|solution|"
                      r"correct_answer|correct_option|\blabel\b|ideal|answer_key|\bkey\b|\banswer\b|\banswers\b|"
                      r"info\[|info\.get|rubric|criteria|criterion|checklist|requirement|key_?points|must_?have|"
                      r"grading_notes|scoring_guide|guideline|exemplar|\bspec\b)", re.I)
GOLD_NAME = re.compile(r"^(answer|answers|gold\w*|reference\w*|ref|target\w*|expected\w*|ground_truth|groundtruth|gt|"
                       r"truth|solution\w*|correct_answer|label|ideal\w*|answer_key|rubric\w*|criteria|criterion|"
                       r"checklist|requirements?|key_points|keypoints)$", re.I)
BIAS_SRC = re.compile(r"(model_name|policy_model|model_id\b|provider|^model$|previous_(score|verdict|grade|rating|judg)|"
                      r"prior_(score|verdict|grade|rating|judg)|prev_(score|verdict|grade)|old_score|last_score|"
                      r"initial_score|pre_?screen|reviewer_?notes?|review_notes|annotator_notes?|first_pass|"
                      r"other_judge|judge_\d_|human_score|human_rating)", re.I)
TPL_NAME = re.compile(r"(judge|grader|grade|grading|evaluat|rubric|critic|rater|rating|scor|assess|reward|verif)", re.I)
NO_JUDGE_NAME = re.compile(r"(system_prompt|^SYSTEM|instruction|user_prompt|question_template|task_prompt|"
                           r"few_?shot|example)", re.I)


LLM_CALL = re.compile(r"(chat\.completions|\.create\(|acompletion\(|\bcompletion\(|generate_content|\.invoke\(|"
                      r"\.ainvoke\(|\.judge\(|\bjudge\w*\(|\.evaluate\(|messages\.create|responses\.create|call_llm|"
                      r"llm\w*\(|query_model|ask_model|_chat\()")


class Tpl:
    __slots__ = ("kind", "pf", "line", "label", "text", "roles", "paid", "users", "call", "model")

    def __init__(self, kind, pf, line, label, text, roles, paid, users, call=None, model=None):
        self.kind, self.pf, self.line, self.label, self.text = kind, pf, line, label, text
        self.roles, self.paid, self.users, self.call, self.model = roles, paid, users, call, model

    def has(self, role):
        return any(r == role for r, _ in self.roles.values())

    @property
    def evid(self):
        return ev(self.pf, self.line)


def _role(src: str, slot: str = "", jr: bool = False) -> str:
    if jr:
        return {"answer": "gold", "response": "response", "question": "question"}.get(slot, "other")
    s = src or slot
    if BIAS_SRC.search(slot) or BIAS_SRC.search(s.split(".")[-1]):
        return "bias"
    if RESP_SRC.search(s) and not re.search(r"(expected|gold|reference|ground_?truth|ideal)", s, re.I):
        return "response"
    if GOLD_SRC.search(s):
        return "gold"
    if re.search(r"(question|prompt|task|problem|query|instruction)", s, re.I):
        return "question"
    return "other"


def _render(n: ast.JoinedStr) -> str:
    parts = []
    for v in n.values:
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            parts.append(v.value.replace("{", "{{").replace("}", "}}"))
        elif isinstance(v, ast.FormattedValue):
            try:
                parts.append("{" + ast.unparse(v.value) + "}")
            except Exception:
                parts.append("{?}")
    return "".join(parts)


def _allf(pkg: Package):
    c = _cache(pkg)
    if "rubric_allf" not in c:
        c["rubric_allf"] = list(all_funcs(pkg)) + [(pf, fi) for pf in _harbor_test_py(pkg) for fi in pf.funcs]
    return c["rubric_allf"]


PROMPT_VAR = re.compile(r"(prompt|message|content|query|judge|instruction|template|user|text|request|input)", re.I)
PROMPT_KW = {"prompt", "content", "messages", "input", "contents", "text", "question", "query", "user_prompt",
             "system_prompt", "judge_prompt"}


def _prompt_strings(fi: FuncInfo) -> set[int]:
    """ids of f-strings in a function that sit where a prompt goes: a message "content" value, a variable
    named like a prompt, or a prompt keyword argument. print() and logging calls do not count."""
    out = set()
    for n in _own_nodes(fi.node):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and k.value in ("content", "text", "prompt", "input") and isinstance(v, ast.JoinedStr):
                    out.add(id(v))
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and isinstance(n.value, ast.JoinedStr):
            tg = n.targets[0] if isinstance(n, ast.Assign) else n.target
            nm = tg.id if isinstance(tg, ast.Name) else (tg.attr if isinstance(tg, ast.Attribute) else "")
            if PROMPT_VAR.search(nm):
                out.add(id(n.value))
        elif isinstance(n, ast.Call):
            cn = _callee(n)
            if cn in ("print", "debug", "info", "warning", "error", "exception", "log", "write", "echo"):
                continue
            for k in n.keywords:
                if k.arg in PROMPT_KW and isinstance(k.value, ast.JoinedStr):
                    out.add(id(k.value))
    return out


def _users(pkg: Package, name: str):
    """-> ([(pf, fi)] of functions whose own body loads `name`, {slot: source} from NAME.format(...) calls
    in them). Found by a text search first, then confirmed on the AST, and cached per name."""
    c = _cache(pkg).setdefault("rubric_users", {})
    if name in c:
        return c[name]
    word = re.compile(r"\b" + re.escape(name) + r"\b")
    users, kws = [], {}
    for pf, fi in _allf(pkg):
        if name not in fi.src or not word.search(fi.src):
            continue
        hit = False
        for n in _own_nodes(fi.node):
            if (isinstance(n, ast.Name) and n.id == name) or (isinstance(n, ast.Attribute) and n.attr == name):
                hit = True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "format":
                b = n.func.value
                bn = b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else None)
                if bn == name:
                    for k in n.keywords:
                        if k.arg and k.arg not in kws:
                            try:
                                kws[k.arg] = ast.unparse(k.value)
                            except Exception:
                                kws[k.arg] = k.arg
        if hit:
            users.append((pf, fi))
    c[name] = (users, kws)
    return c[name]


def _jr_paid(pkg: Package, pf: PyFile, call: ast.Call, nm: str | None) -> float:
    """Weight a JudgeRubric pays at: funcs given to it, add_reward_func on its name, or a paid function
    that calls .judge( on it."""
    f = _kw(call, "funcs")
    if isinstance(f, (ast.List, ast.Tuple)) and f.elts:
        return 1.0
    if nm and pf.tree is not None:
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_reward_func":
                v = n.func.value
                if (isinstance(v, ast.Name) and v.id == nm) or (isinstance(v, ast.Attribute) and v.attr == nm):
                    wn = _kw(n, "weight") or (n.args[1] if len(n.args) > 1 else None)
                    w = _num(wn) if wn is not None else 1.0
                    if w is None or w > 0:
                        return 1.0 if w is None else w
    for pf2, fi in all_funcs(pkg):
        if ".judge(" in fi.src and (nm is None or nm in fi.src):
            w = _paid(pkg, fi.name)
            if w > 0:
                return w
    return 0.0


def _users_paid(pkg: Package, users) -> float:
    hp = {p.path for p in _harbor_test_py(pkg)}
    best = 0.0
    for pf, fi in users:
        if pf.path in hp:
            return 1.0
        best = max(best, _paid(pkg, fi.name))
    return best


def _tpls(pkg: Package) -> list[Tpl]:
    """Judge prompt templates. Each carries the role of every slot (response, gold, question, bias,
    other) and the largest rubric weight it is paid at (0 means not on the paid path)."""
    c = _cache(pkg)
    if "rubric_tpls" in c:
        return c["rubric_tpls"]
    out: list[Tpl] = []
    used_consts = set()
    for pf, call, nm in _judge_rubrics(pkg):
        jp = _kw(call, "judge_prompt")
        label = "JudgeRubric default prompt"
        if jp is None:
            text = JR_DEFAULT
        else:
            v = _value_of(pf, jp)
            text = _const_text(v) if v is not None else None
            if isinstance(jp, ast.Name):
                used_consts.add(jp.id)
                label = jp.id
            elif isinstance(jp, ast.Attribute):
                used_consts.add(jp.attr)
                label = jp.attr
            else:
                label = "judge_prompt"
            if not text:
                continue
        roles = {s: (_role(s, s, jr=True), s) for s in SLOT.findall(text)}
        mdl = _kw(call, "judge_model")
        mv = _value_of(pf, mdl) if mdl is not None else ast.Constant(JR_DEFAULT_MODEL)
        model = mv.value if isinstance(mv, ast.Constant) and isinstance(mv.value, str) else None
        out.append(Tpl("JudgeRubric", pf, call.lineno, label, text, roles, _jr_paid(pkg, pf, call, nm), [],
                       call=call, model=model))
    # verifiers v1: class MyJudge(vf.Judge / vf.ReferenceJudge): prompt = TEMPLATE
    for pf in pkg.py_all():
        if pf.tree is None or "Judge" not in pf.text:
            continue
        for cl in ast.walk(pf.tree):
            if not isinstance(cl, ast.ClassDef):
                continue
            bases = [b.attr if isinstance(b, ast.Attribute) else (b.id if isinstance(b, ast.Name) else "") for b in cl.bases]
            bases = [b.split("[")[0] for b in bases] + [_callee(b) if isinstance(b, ast.Call) else
                                                         (b.value.attr if isinstance(b, ast.Subscript) and isinstance(b.value, ast.Attribute)
                                                          else (b.value.id if isinstance(b, ast.Subscript) and isinstance(b.value, ast.Name) else ""))
                                                         for b in cl.bases]
            if not any(b.endswith("Judge") for b in bases):
                continue
            for st in cl.body:
                tg = st.targets[0] if isinstance(st, ast.Assign) and len(st.targets) == 1 else (st.target if isinstance(st, ast.AnnAssign) else None)
                if not (isinstance(tg, ast.Name) and tg.id in ("prompt", "template", "judge_prompt", "prompt_template")) or st.value is None:
                    continue
                v = _value_of(pf, st.value)
                text = _const_text(v) if v is not None else None
                if not text:
                    continue
                label = st.value.id if isinstance(st.value, ast.Name) else f"{cl.name}.prompt"
                used_consts.add(label)
                ref = any(b.startswith("Reference") for b in bases)
                roles = {x: (_role(x, x, jr=True) if (ref or x != "answer") else "other", x) for x in SLOT.findall(text)}
                roles = {k: (v if isinstance(v, str) else v[0], k) for k, v in roles.items()}
                used = sum(1 for p2 in pkg.py_all() if cl.name in p2.text) > 1 or len(re.findall(r"\b" + re.escape(cl.name) + r"\b", pf.text)) > 1
                out.append(Tpl("v1Judge", pf, st.lineno, label, text, roles, 1.0 if used else 0.0, []))
    files = list(pkg.py_all()) + _harbor_test_py(pkg)
    for pf in files:
        if pf.tree is None:
            continue
        for name, t in pf.consts.items():
            if name in used_consts or len(t) < 80 or "{" not in t or NO_JUDGE_NAME.search(name):
                continue
            if not (TPL_NAME.search(name) or JUDGE_CONTEXT.search(t[:600])):
                continue
            slots = SLOT.findall(t)
            if not slots:
                continue
            users, kws = _users(pkg, name)
            roles = {}
            for s in slots:
                src = kws.get(s, s)
                roles[s] = (_role(src, s), src)
            if not any(r == "response" for r, _ in roles.values()):
                continue
            line = next((i for i, ln in enumerate(pf.lines, 1) if re.match(r"\s*" + re.escape(name) + r"\s*[:=]", ln)), 1)
            out.append(Tpl("const", pf, line, name, t, roles, _users_paid(pkg, users), users))
    seen = set()
    for pf, fi in _grader_funcs(pkg):
        if (pf.path, fi.start) in seen or ("f\"" not in fi.src and "f'" not in fi.src) or not LLM_CALL.search(fi.src):
            continue
        seen.add((pf.path, fi.start))
        spots = _prompt_strings(fi)
        if not spots:
            continue
        for n in _own_nodes(fi.node):
            if not isinstance(n, ast.JoinedStr) or id(n) not in spots:
                continue
            t = _render(n)
            if len(t) < 40 or not JUDGE_CONTEXT.search(t[:800] + " " + fi.name):
                continue
            roles = {}
            for v in n.values:
                if isinstance(v, ast.FormattedValue):
                    try:
                        src = ast.unparse(v.value)
                    except Exception:
                        continue
                    roles[src] = (_role(src, src.split(".")[-1].split("[")[0]), src)
            if not any(r == "response" for r, _ in roles.values()):
                continue
            out.append(Tpl("fstring", pf, n.lineno, fi.name, t, roles, _users_paid(pkg, [(pf, fi)]), [(pf, fi)]))
    c["rubric_tpls"] = out
    return out


def _paid_tpls(pkg: Package) -> list[Tpl]:
    return [t for t in _tpls(pkg) if t.paid > 0]


def _undelimited(pkg: Package) -> set[str]:
    c = _cache(pkg)
    if "rubric_undelim" not in c:
        s = set()
        for f in r_judge_undelimited(pkg):
            s |= set(f.evidence)
        c["rubric_undelim"] = s
    return c["rubric_undelim"]


# ---------------------------------------------------------------------------
# dataset gold column
# ---------------------------------------------------------------------------

GOLD_KEYS = {"answer", "answers", "gold", "gold_answer", "reference", "reference_answer", "target", "ground_truth",
             "expected", "expected_answer", "expected_output", "solution", "correct_answer", "label", "final_answer"}
ROW_KEYS = {"question", "prompt", "problem", "query", "input", "task", "instruction"}


def _empty(v) -> bool:
    return isinstance(v, ast.Constant) and (v.value is None or v.value == "" or v.value == [])


def _gold_column(pkg: Package):
    """file:line where the package builds dataset rows with a non empty gold answer column, else None."""
    c = _cache(pkg)
    if "rubric_gold" in c:
        return c["rubric_gold"]
    hit = None
    for pf in pkg.py_all():
        if pf.tree is None or hit:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Dict):
                ks = {k.value: v for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                g = [k for k in ks if k in GOLD_KEYS and not _empty(ks[k])]
                if g and ROW_KEYS & set(ks):
                    hit = ev(pf, n.lineno)
                    break
            elif isinstance(n, ast.ClassDef) and re.search(r"(TaskData|Task|Row|Example|Sample|Item|Record|Data)",
                                                           n.name + " " + " ".join(ast.unparse(b) for b in n.bases)):
                for st in n.body:
                    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id in GOLD_KEYS \
                            and not _empty(st.value):
                        hit = ev(pf, st.lineno)
                        break
                if hit:
                    break
            elif isinstance(n, ast.Call) and _callee(n) in ("rename_column", "rename_columns", "add_column"):
                txt = ast.unparse(n)
                if re.search(r"['\"](answer|gold|reference|ground_truth|target)['\"]", txt):
                    hit = ev(pf, n.lineno)
                    break
    if not hit:
        W = _reward_weights(pkg)
        for pf, fi in all_funcs(pkg):
            if "answer" in fi.params and fi.name in W and any(w > 0 for w, _ in W[fi.name]):
                if any(isinstance(n, ast.Name) and n.id == "answer" and isinstance(n.ctx, ast.Load)
                       for n in _own_nodes(fi.node)):
                    hit = ev(pf, fi.start)
                    break
    if not hit:
        for p, t in pkg.files.items():
            if not p.endswith((".jsonl", ".json")) or "/tests/" in "/" + p:
                continue
            head = t[:20000]
            m = re.search(r"\"(answer|gold_answer|reference_answer|ground_truth|expected_answer|correct_answer)\"\s*:\s*"
                          r"(\"[^\"]|\d|\[\s*\S)", head)
            if m and re.search(r"\"(question|prompt|problem|query)\"\s*:", head):
                hit = f"{p}:{head.count(chr(10), 0, m.start()) + 1}"
                break
    c["rubric_gold"] = hit
    return hit


CORRECTNESS = re.compile(r"(\bcorrect|\bincorrect|accura|right answer|\bwrong\b|factual|true or false|valid answer|"
                         r"\bsolv|\bmatch(es)? the|equivalent)", re.I)


BEHAVIOR = re.compile(r"\b(refus\w*|jailbreak\w*|harmful|toxic\w*|unsafe|privacy|persona|in character|funny|humou?r)\b", re.I)


def _user_loads_gold(t: Tpl) -> bool:
    for pf, fi in t.users:
        for n in _own_nodes(fi.node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and GOLD_NAME.match(n.id):
                return True
            if isinstance(n, ast.Attribute) and GOLD_NAME.match(n.attr):
                return True
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and GOLD_NAME.match(n.value):
                return True
    return False


# ---------------------------------------------------------------------------
# 1. judge-prompt-lacks-reference
# Reader items reader items. Main paper arxiv:2306.05685 (math grading
# failures 14/20 to 3/20 with a reference guided judge), also arxiv:2505.15801.
# Fires when a paid judge template has a slot for the response, no slot for a gold answer, rubric or
# reference, the function that fills it reads no gold value, and the package builds dataset rows
# with a non empty gold answer column. For JudgeRubric the {answer} slot is the dataset's answer
# column, so a custom judge_prompt without {answer} never shows the judge the gold.
# ---------------------------------------------------------------------------

@rule("judge-prompt-lacks-reference", "hack", "medium",
      "A paid LLM judge prompt shows the judge the response but not the gold answer that the dataset carries, "
      "so the judge grades from its own knowledge and can pass a confident wrong answer.")
def r_judge_lacks_reference(pkg: Package):
    tp = _paid_tpls(pkg)
    if not tp:
        return []
    hits = []
    for t in tp:
        if not t.has("response") or t.has("gold"):
            continue
        if any(r == "other" for r, _ in t.roles.values()) and t.kind != "JudgeRubric":
            continue  # a slot we cannot name could carry the reference
        if t.kind != "JudgeRubric" and _user_loads_gold(t):
            continue
        if re.search(r"(reference|ground[ -]truth|\bgold\b|correct answer\s*(:|below|provided|given|is\b)|expected answer|"
                     r"answer key|rubric|criteria)", t.text, re.I) and t.kind != "JudgeRubric":
            # the text talks about a reference that must come from somewhere we did not trace
            continue
        ctx = t.text + "\n" + "\n".join(referenced_consts(pf, fi) for pf, fi in t.users)
        if not CORRECTNESS.search(ctx) or BEHAVIOR.search(ctx):
            continue  # a behavior or style judge (refusal, tone), not a correctness judge
        hits.append(t)
    if not hits:
        return []
    gold = _gold_column(pkg)
    if not gold:
        return []
    t = hits[0]
    why = ("its custom judge_prompt has no {answer} slot, so JudgeRubric never puts the answer column in it"
           if t.kind == "JudgeRubric" else "the template has no slot for the gold answer and the code that fills it reads none")
    return [F("judge-prompt-lacks-reference",
              f"The judge prompt {t.label} is paid at weight {t.paid:g} and {why}, while the dataset rows carry a gold "
              "answer, so the judge decides correctness from its own knowledge and pays a fluent wrong answer.",
              [x.evid for x in hits] + [gold], confidence="review")]


# ---------------------------------------------------------------------------
# rubric criteria lists
# ---------------------------------------------------------------------------

TEXT_KEYS = ("criterion", "criteria", "description", "requirement", "text", "title", "item", "check", "statement",
             "rubric", "question", "name", "desc", "content")
WEIGHT_KEYS = ("weight", "points", "point", "score", "max_score", "max_points", "pts", "value", "marks")
CRIT_KEYS = re.compile(r"^(critical|is_critical|required|is_required|must|must_pass|mandatory|hard|is_hard|gate|"
                       r"veto|blocking|essential|fatal|hard_constraint)$", re.I)
CRIT_VALS = re.compile(r"^(critical|must|must[_ -]?pass|required|mandatory|hard|essential|fatal|blocking|veto)$", re.I)
KEY_HINTS = tuple(q + k + q for k in ("criterion", "criteria", "requirement", "points", "weight", "rubric", "rubrics")
                  for q in "'\"")


def _stmts(body, depth=0):
    """Statements in nested statement lists (module, class, function, if, with, try bodies), no expressions."""
    for st in body:
        yield st
        if depth < 6:
            for f in ("body", "orelse", "finalbody", "handlers"):
                sub = getattr(st, f, None)
                if isinstance(sub, list) and sub and isinstance(sub[0], (ast.stmt, ast.excepthandler)):
                    yield from _stmts(sub, depth + 1)
RUBRIC_PARENT = re.compile(r"(rubric|criteria|criterion|checklist|requirements|rules|items|points|scoring|grading)", re.I)


class Crit:
    __slots__ = ("text", "w", "crit")

    def __init__(self, text, w, crit):
        self.text, self.w, self.crit = text, w, crit


class RGroup:
    __slots__ = ("src", "where", "items", "kind", "crit_key", "nrows", "span")

    def __init__(self, src, where, items, kind, crit_key=None, nrows=1, span=None):
        self.src, self.where, self.items, self.kind, self.crit_key, self.nrows = src, where, items, kind, crit_key, nrows
        self.span = span  # (path, first line, last line) of a Python literal


def _crit_from_obj(d: dict):
    txt = next((d[k] for k in TEXT_KEYS if isinstance(d.get(k), str) and d[k].strip()), None)
    if txt is None:
        return None
    w = None
    for k in WEIGHT_KEYS:
        v = d.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            w = float(v)
            break
    ck = None
    for k, v in d.items():
        if isinstance(k, str) and CRIT_KEYS.match(k) and v not in (False, 0, None, "", "false", "no"):
            ck = k
        elif isinstance(k, str) and re.match(r"^(type|kind|level|priority|severity|category|tag|tags)$", k, re.I):
            vals = v if isinstance(v, list) else [v]
            if any(isinstance(x, str) and CRIT_VALS.match(x.strip()) for x in vals):
                ck = k
    return Crit(txt, w, ck)


def _groups_from_json(obj, parent="", depth=0, acc=None):
    if acc is None:
        acc = []
    if depth > 6 or len(acc) > 400:
        return acc
    if isinstance(obj, list):
        dicts = [x for x in obj if isinstance(x, dict)]
        if len(dicts) >= 2 and len(dicts) >= 0.8 * len(obj):
            items = [_crit_from_obj(x) for x in dicts]
            items = [x for x in items if x is not None]
            weighted = sum(1 for x in items if x.w is not None)
            if len(items) >= 2 and len(items) >= 0.8 * len(dicts) and (RUBRIC_PARENT.search(parent) or weighted >= 0.8 * len(items)) \
                    and not re.search(r"(messages|turns|history|conversation|examples|choices|options|tools)", parent, re.I):
                acc.append((parent, items))
                return acc
        for x in obj[:500]:
            _groups_from_json(x, parent, depth + 1, acc)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (list, dict)):
                _groups_from_json(v, str(k), depth + 1, acc)
    return acc


def _ast_obj(n, depth=0):
    """Literal value of a dict/list/constant node, with unknown parts as None."""
    if depth > 8:
        return None
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        v = _ast_obj(n.operand, depth + 1)
        return -v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    if isinstance(n, (ast.List, ast.Tuple)):
        return [_ast_obj(e, depth + 1) for e in n.elts]
    if isinstance(n, ast.Dict):
        out = {}
        for k, v in zip(n.keys, n.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                out[k.value] = _ast_obj(v, depth + 1)
        return out
    if isinstance(n, ast.Call) and _callee(n) == "dict":
        return {k.arg: _ast_obj(k.value, depth + 1) for k in n.keywords if k.arg}
    if isinstance(n, (ast.JoinedStr,)):
        return _const_text(n)
    return None


WEIGHTED_LINE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+(.{6,200}?)\s*[(\[]\s*(?:weight[: ]*)?(-?\d+(?:\.\d+)?)\s*"
                           r"(%|pts?|points?|marks?)?\s*[)\]]\s*:?\s*$", re.I | re.M)


def _yaml_groups(text: str):
    """Very small YAML reader for '- criterion: ...' blocks with an optional points/weight line."""
    items, cur = [], None
    for ln in text.splitlines():
        m = re.match(r"^\s*-\s+(" + "|".join(TEXT_KEYS) + r")\s*:\s*['\"]?(.+?)['\"]?\s*$", ln)
        if m:
            if cur:
                items.append(cur)
            cur = Crit(m.group(2), None, None)
            continue
        if cur is None:
            continue
        m = re.match(r"^\s+(" + "|".join(WEIGHT_KEYS) + r")\s*:\s*(-?\d+(?:\.\d+)?)\s*$", ln)
        if m:
            cur.w = float(m.group(2))
            continue
        m = re.match(r"^\s+(\w+)\s*:\s*['\"]?([\w -]+?)['\"]?\s*$", ln)
        if m and (CRIT_KEYS.match(m.group(1)) and m.group(2).lower() not in ("false", "no", "0")
                  or re.match(r"^(type|kind|level|priority|severity)$", m.group(1)) and CRIT_VALS.match(m.group(2))):
            cur.crit = m.group(1)
    if cur:
        items.append(cur)
    return items


def _referenced(pkg: Package, path: str) -> bool:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    txt = _all_code(pkg)
    return base in txt or (len(stem) > 4 and any(q + stem + e in txt for q in "'\"/" for e in "'\"."))


def _all_code(pkg: Package) -> str:
    c = _cache(pkg)
    if "rubric_code" not in c:
        parts = [pf.text for pf in pkg.py_all()] + [pf.text for pf in _harbor_test_py(pkg)]
        parts += [t for p, t in pkg.files.items() if p.endswith((".sh", ".toml"))]
        c["rubric_code"] = "\n".join(parts)
    return c["rubric_code"]


def _scores_anything(pkg: Package) -> bool:
    return bool(_reward_weights(pkg)) or bool(_harbor_test_py(pkg)) or bool(_paid_tpls(pkg))


def _rubrics(pkg: Package) -> list[RGroup]:
    """Rubric criteria lists that reach scoring: Python literals read by a paid function or a paid judge
    template, data files the code loads, and weighted criteria lines inside a paid judge prompt."""
    c = _cache(pkg)
    if "rubric_groups" in c:
        return c["rubric_groups"]
    out: list[RGroup] = []
    if not _scores_anything(pkg):
        c["rubric_groups"] = out
        return out
    # python literals
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in _stmts(pf.tree.body):
            if not isinstance(n, (ast.Assign, ast.AnnAssign)) or n.value is None:
                continue
            if not isinstance(n.value, (ast.List, ast.Tuple, ast.Dict)):
                continue
            if (getattr(n, "end_lineno", n.lineno) or n.lineno) - n.lineno > 4000:
                continue
            tg = n.targets[0] if isinstance(n, ast.Assign) else n.target
            name = tg.id if isinstance(tg, ast.Name) else (tg.attr if isinstance(tg, ast.Attribute) else "")
            if not RUBRIC_PARENT.search(name):
                el = n.value.elts if isinstance(n.value, (ast.List, ast.Tuple)) else []
                if not el or not isinstance(el[0], ast.Dict):
                    continue
                ks = {k.value for k in el[0].keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                if not (ks & set(TEXT_KEYS)) or not (ks & set(WEIGHT_KEYS)):
                    continue
            obj = _ast_obj(n.value)
            groups = _groups_from_json(obj, name)
            if not groups and isinstance(obj, list) and RUBRIC_PARENT.search(name) and len(obj) >= 2 and \
                    all(isinstance(x, list) and len(x) >= 2 and isinstance(x[0], str) and isinstance(x[1], (int, float))
                        and not isinstance(x[1], bool) for x in obj):
                groups = [(name, [Crit(x[0], float(x[1]), None) for x in obj])]
            if not groups and isinstance(obj, dict) and RUBRIC_PARENT.search(name) and len(obj) >= 2 and \
                    all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj.values()) and \
                    all(isinstance(k, str) and len(k.split()) >= 3 for k in obj):
                groups = [(name, [Crit(k, float(v), None) for k, v in obj.items()])]
            if not groups:
                continue
            users = _users(pkg, name)[0] if name else []
            paid = _users_paid(pkg, users)
            if not paid:
                for t in _paid_tpls(pkg):
                    if any(name and name in fi.src for _, fi in t.users):
                        paid = t.paid
                        break
            if not paid:
                continue
            for parent, items in groups:
                out.append(RGroup(ev(pf, n.lineno), f"{name}", items, "literal",
                                  crit_key=next((x.crit for x in items if x.crit), None),
                                  span=(pf.path, n.lineno, getattr(n, "end_lineno", n.lineno) or n.lineno)))
    # data files
    for p, t in pkg.files.items():
        low = p.lower()
        if not low.endswith((".json", ".jsonl", ".yaml", ".yml")) or os.path.basename(low) in ("package.json", "tsconfig.json"):
            continue
        head = t[:200000].lower()
        if "rubric" not in head and "criteri" not in head and "checklist" not in head:
            continue
        if not (re.search(r"(rubric|criteri|checklist|judge|grad)", low) or _referenced(pkg, p)):
            continue
        if "/tests/" not in "/" + p and not _referenced(pkg, p):
            continue
        items_all, rows, line = [], 0, 1
        crit_key = None
        if low.endswith(".jsonl"):
            for i, ln in enumerate(t.splitlines()[:300]):
                ln = ln.strip()
                if not ln.startswith("{"):
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                gs = _groups_from_json(obj)
                if gs:
                    rows += 1
                    for _, its in gs:
                        items_all.append(its)
        elif low.endswith(".json"):
            try:
                obj = json.loads(t)
            except Exception:
                continue
            for _, its in _groups_from_json(obj):
                items_all.append(its)
            rows = 1
        else:
            its = _yaml_groups(t)
            if len(its) >= 2:
                items_all.append(its)
                rows = 1
        if not items_all:
            continue
        for its in items_all:
            crit_key = crit_key or next((x.crit for x in its if x.crit), None)
        flat = [x for its in items_all for x in its]
        out.append(RGroup(f"{p}:{line}", p, flat, "data", crit_key=crit_key, nrows=max(1, len(items_all))))
    # weighted lines in a paid judge prompt
    for tp in _paid_tpls(pkg):
        its = [Crit(m.group(1), float(m.group(2)), None) for m in WEIGHTED_LINE.finditer(tp.text)]
        if len(its) >= 3:
            out.append(RGroup(tp.evid, tp.label, its, "prompt"))
    c["rubric_groups"] = out
    return out


def _per_row(g: RGroup):
    """Split a data group back into rows of roughly equal size for per row checks (we keep a flat list)."""
    return g.items


PENALTY_TEXT = re.compile(r"(\bnot\b|\bno\b|n't|\bnever\b|\bavoid|\bwithout\b|penal|deduct|incorrect|\bwrong|\berror|"
                          r"hallucinat|fabricat|false|mislead|unsupported|irrelevant|excess|verbose|padding|filler|"
                          r"flatter|sycophan|repetit|unsafe|harm|\bfails?\b|missing|omit|contradict|refus|overclaim|"
                          r"invent|made[- ]up|inaccura|\bonly\b|concise|brief|length|\bextra\b)", re.I)
PRESENCE_TEXT = re.compile(r"^(?:the )?(?:response|answer|model|assistant|output|report|summary|essay|submission|it|"
                           r"agent)?\s*(?:should |must |correctly |clearly |explicitly )?"
                           r"(mention|include|state|explain|list|identif|describ|discuss|provid|note|cite|reference|"
                           r"address|cover|highlight|give|present|recommend|suggest|outline|define|specif|acknowledg|"
                           r"summari|propos|offer|detail|name|point|show|emphasi|consider|contain|has |have |uses? |"
                           r"demonstrat|elaborat|compar|analy[sz]|answers? |says? |notes?|asks? )", re.I)
SAFETY_TEXT = re.compile(r"(harm|unsafe|danger|safety|toxic|illegal|fabricat|hallucin|misinform|incorrect|\bfalse\b|"
                         r"\bwrong\b|contraindicat|lethal|overdose|self-harm|privacy|\bpii\b|discriminat|bias)", re.I)


# ---------------------------------------------------------------------------
# 2. rubric-no-veto-on-critical-criteria
# Reader items reader items. Main paper arxiv:2609.00892
# (veto reward gave Arena-Hard 10.7 vs 2.0 for a static rubric), also arxiv:2602.06486.
# Two shapes, both on rubric data the package scores with:
#   a) items are marked critical, required or must pass, but no scoring code reads that mark, so a
#      failed critical item only loses its own points (hack)
#   b) a safety or factual item is a negative weight in a sum where the positive items are worth at
#      least three times as much, and nothing in the code gates on it (hygiene)
# How reward code combines terms in Python is builder `reward`'s additive-rubric-without-hard-gate.
# ---------------------------------------------------------------------------

GATE_CODE = re.compile(r"(veto|must_pass|hard_fail|hard_gate|is_critical|critical_fail|any_critical|fatal|zero_out|"
                       r"gate\w*\s*=|if\s+[^\n]{0,60}(critical|required|must)[^\n]{0,60}:\s*\n?\s*return\s+0)", re.I)


def _code_without(pkg: Package, spans) -> str:
    """All scoring code with the given literal spans blanked, so a key inside the rubric literal itself does
    not count as code that reads it."""
    by = {}
    for p, a, b in spans:
        by.setdefault(p, []).append((a, b))
    parts = []
    for pf in list(pkg.py_all()) + _harbor_test_py(pkg):
        if pf.path not in by:
            parts.append(pf.text)
            continue
        keep = [ln for i, ln in enumerate(pf.lines, 1) if not any(a <= i <= b for a, b in by[pf.path])]
        parts.append("\n".join(keep))
    return "\n".join(parts)


@rule("rubric-no-veto-on-critical-criteria", "hack", "medium",
      "A rubric marks some criteria as critical or safety items but scores them as plain addends, so a reply that "
      "fails a critical item can still get most of the reward.")
def r_rubric_no_veto(pkg: Package):
    gs = _rubrics(pkg)
    if not gs:
        return []
    out = []
    marked = []
    for g in gs:
        if g.crit_key and sum(1 for x in g.items if x.crit) < len(g.items):
            code = _code_without(pkg, [x.span for x in gs if x.span])
            if not re.search(r"['\"]" + re.escape(g.crit_key) + r"['\"]|\." + re.escape(g.crit_key) + r"\b", code) \
                    and not GATE_CODE.search(code):
                marked.append(g)
    if marked:
        g = marked[0]
        n = sum(1 for x in g.items if x.crit)
        out.append(F("rubric-no-veto-on-critical-criteria",
                     f"{g.where} marks {n} of {len(g.items)} criteria with the field {g.crit_key!r}, but no scoring code "
                     "reads that field, so failing a critical item only costs its own points and the rest still pay.",
                     [x.src for x in marked], confidence="review"))
    neg = []
    for g in gs:
        pos = sum(x.w for x in g.items if x.w is not None and x.w > 0)
        bad = [x for x in g.items if x.w is not None and x.w < 0 and SAFETY_TEXT.search(x.text)]
        if not bad or pos <= 0:
            continue
        rows = max(1, g.nrows)
        worst = min(x.w for x in bad)
        if pos / rows >= 3 * abs(worst) and not GATE_CODE.search(_all_code(pkg)):
            neg.append((g, bad[0], pos / rows))
    if neg:
        g, x, pos = neg[0]
        out.append(F("rubric-no-veto-on-critical-criteria",
                     f"{g.where} scores the safety or factual item {x.text[:70]!r} as a negative weight {x.w:g} in a sum "
                     f"whose positive items are worth about {pos:g}, with no gate, so a reply that breaks it keeps most "
                     "of its score.", [g.src for g, _, _ in neg], confidence="review", kind="hygiene", severity="low"))
    return out


# ---------------------------------------------------------------------------
# 3. rubric-presence-only-no-penalty
# Reader items reader items. Main paper arxiv:2605.12474
# (90% of rubric weight was presence based, and 65 to 75% of newly credited criteria were wrong by
# the end of training). Fires on a scored rubric with at least 5 criteria, all weights positive, no
# criterion that penalizes wrong, extra or filler content, and at least 80% of criteria worded as
# "mentions / includes / explains X".
# ---------------------------------------------------------------------------

@rule("rubric-presence-only-no-penalty", "hygiene", "low",
      "A scored rubric only pays for mentioning things and has no criterion against wrong, extra or filler "
      "content, so stuffing the reply with claims raises the score.")
def r_rubric_presence(pkg: Package):
    gs = _rubrics(pkg)
    if not gs:
        return []
    hits = []
    for g in gs:
        its = g.items
        if len(its) < 5:
            continue
        if any(x.w is not None and x.w < 0 for x in its) or any(PENALTY_TEXT.search(x.text) for x in its):
            continue
        pres = sum(1 for x in its if PRESENCE_TEXT.search(x.text.strip()))
        if pres >= 0.8 * len(its):
            hits.append((g, pres))
    if not hits:
        return []
    g, pres = hits[0]
    return [F("rubric-presence-only-no-penalty",
              f"{g.where} has {len(g.items)} criteria, {pres} of them pay for mentioning something and none takes points "
              "off for wrong, extra or padded content, so a reply that lists every plausible claim scores high.",
              [x.src for x, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# 4. judge-many-criteria-one-call
# Reader items reader items. Main paper arxiv:2603.25133 (+7 to +12 BAcc
# with one call per rubric item), also arxiv:2608.06422 and arxiv:2606.29920 (JU0466).
# Fires on a paid judge template that lists at least 8 criteria and asks for all scores in one
# reply, or that takes a whole per row rubric through one slot and asks to score each item, when
# the shipped rubric rows have at least 8 items.
# ---------------------------------------------------------------------------

LIST_ITEM = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•]|\([a-z0-9]\))\s+\S.{8,}$", re.M)
EACH = re.compile(r"(for each (criterion|criteria|item|requirement|rubric|point|dimension|aspect|check)|"
                  r"each (of the )?(criterion|criteria|item|requirement|rubric item|dimension|aspect)|"
                  r"(score|rate|grade|evaluate|assess|judge) (each|every|all)\b|all (the )?(criteria|items|requirements)|"
                  r"per[- ](criterion|item|requirement))", re.I)
OUT_FMT = re.compile(r"(json|format|respond|return|output|reply)", re.I)


def _criteria_count(text: str) -> int:
    """Largest run of list items in the template text that reads like scored criteria."""
    best, run, last = 0, 0, -10
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if LIST_ITEM.match(ln) and not re.search(r"(respond|output|format|return|json|example|\{)", ln, re.I):
            run = run + 1 if i - last <= 2 else 1
            last = i
            best = max(best, run)
    return best


@rule("judge-many-criteria-one-call", "hygiene", "low",
      "One paid judge call scores many rubric criteria at once, which is less accurate than one call per criterion.")
def r_many_criteria(pkg: Package):
    tp = _paid_tpls(pkg)
    if not tp:
        return []
    hits = []
    rubric_rows = [g for g in _rubrics(pkg) if g.kind == "data"]
    per_row = max((len(g.items) / max(1, g.nrows) for g in rubric_rows), default=0)
    for t in tp:
        n = _criteria_count(t.text)
        asks_all = EACH.search(t.text) or len(re.findall(r"\"[\w ]{2,40}\"\s*:\s*(<|\d|\"?(score|int|float|number))",
                                                         t.text, re.I)) >= 5
        if n >= 8 and asks_all:
            hits.append((t, f"lists {n} criteria and asks for all of their scores in one reply"))
            continue
        if t.has("gold") and EACH.search(t.text) and per_row >= 8 and \
                any(re.search(r"(rubric|criteria|checklist|requirements)", s, re.I) for _, s in t.roles.values()):
            hits.append((t, f"takes the whole rubric through one slot (about {per_row:.0f} items per row) and asks the "
                            "judge to score each item"))
    if not hits:
        return []
    t, why = hits[0]
    return [F("judge-many-criteria-one-call",
              f"The paid judge prompt {t.label} {why}, so later items get less attention and the scores skew toward yes.",
              [x.evid for x, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# 5. judge-no-length-control
# Reader items reader items. Main paper arxiv:2512.23707 (a 750 word cap stopped
# verbosity growth). Fires on a paid judge that scores on a scale (not a yes or no match against a
# gold answer), when the judge prompt has no length or concision wording, the package's reward code
# has no len(), word or token count, and the rubric (if any) has no length criterion. A max_tokens
# limit on the policy also bounds length and is invisible here, so this is hygiene.
# ---------------------------------------------------------------------------

LEN_WORDS = re.compile(r"(length|concise|concision|verbos|brev|succinct|word (limit|count)|\bwords?\b|padding|"
                       r"too long|longer|shorter|filler|rambl|to the point|redundan|repetit)", re.I)
LEN_CODE = re.compile(r"(\blen\(|word_count|num_words|count_words|\.split\(\)\)|token_count|num_tokens|count_tokens|"
                      r"tokenize|max_words|max_length|max_chars|length_penalty|len_penalty|brevity)")
SCALE = re.compile(r"(\b(1|0)\s*(-|to)\s*(5|10|100)\b|out of (5|10|100)|\bscore\b|\brating\b|\brate\b|likert|"
                   r"0\.0 (to|and) 1\.0|scale)", re.I)


@rule("judge-no-length-control", "hygiene", "low",
      "A paid judge scores replies on a scale with no length cap, length penalty or length criterion anywhere on "
      "the reward path, so longer replies tend to score higher.")
def r_no_length(pkg: Package):
    tp = [t for t in _paid_tpls(pkg) if SCALE.search(t.text) and not LEN_WORDS.search(t.text)]
    if not tp:
        return []
    tp = [t for t in tp if not (t.has("gold") and re.search(r"\b(yes|no|correct|incorrect)\b", t.text[-400:], re.I)
                                and not re.search(r"(1|0)\s*(-|to)\s*(5|10)", t.text))]
    if not tp:
        return []
    W = _reward_weights(pkg)
    for pf, fi in _grader_funcs(pkg):
        if LEN_CODE.search(fi.src) and (fi.name in W or "judge" in fi.src.lower() or "reward" in fi.name):
            return []
    for g in _rubrics(pkg):
        if any(LEN_WORDS.search(x.text) for x in g.items):
            return []
    t = tp[0]
    return [F("judge-no-length-control",
              f"The paid judge prompt {t.label} scores the reply on a scale and nothing on the reward path caps or "
              "penalizes length, so padding the reply is a cheap way to raise the judge's score.",
              [x.evid for x in tp], confidence="review")]


# ---------------------------------------------------------------------------
# 6. judge-prompt-missing-guard-instructions (+ judge-template-missing-clauses)
# Reader items reader items and reader items.
# Main paper arxiv:2506.09443 (attack success 15.42% to 3.61% with a hardened judge prompt), also
# arxiv:2610.24320. Extends judge-prompt-undelimited, which checks delimiters only. Fires on a paid
# judge template with a response slot that has none of: a clause that the response is data and its
# instructions are ignored, a clause that the response's own claims are not evidence, or a clause to
# ignore length, style or confidence. Templates judge-prompt-undelimited already reports are skipped.
# ---------------------------------------------------------------------------

GUARD = re.compile(r"(ignore (any |all )?(instructions|commands|directives|requests|attempts)|do not follow|don't follow|"
                   r"not (to )?follow|(as|be treated as|is) (data|content to (be )?(evaluate|grade|judge))|untrusted|"
                   r"injection|disregard (any|all)|manipulat|(claims?|assertions?|statements?) (in|by|of|made) the "
                   r"(response|answer|output|submission|model|agent)|self[- ]report|do not (trust|take)|not (be )?"
                   r"(influenced|swayed)|regardless of (its |the )?(length|style|tone|formatting|confidence)|"
                   r"(length|style|tone|confidence|formatting|verbosity) (alone )?(should|does|must) not|"
                   r"ignore (the )?(length|style|tone|formatting)|do not reward (length|verbosity|style)|"
                   r"(addressed|addresses|speaks) (to )?(the )?(judge|grader|evaluator)|persuas)", re.I)


@rule("judge-prompt-missing-guard-instructions", "hygiene", "low",
      "A paid judge prompt has no instruction that the response is data to grade, that its own claims are not "
      "evidence, or that length and style do not count.")
def r_missing_guard(pkg: Package):
    tp = [t for t in _paid_tpls(pkg) if t.has("response") and not GUARD.search(t.text)]
    if not tp:
        return []
    skip = _undelimited(pkg)
    tp = [t for t in tp if f"{t.pf.path}:{t.label}" not in skip]
    if not tp:
        return []
    t = tp[0]
    return [F("judge-prompt-missing-guard-instructions",
              f"The paid judge prompt {t.label} puts the response in front of the judge with no rule to ignore "
              "instructions or claims inside it and no rule that length or style do not count.",
              [x.evid for x in tp], confidence="review")]


# ---------------------------------------------------------------------------
# 7. unpinned-judge-model-alias
# Reader item reader items. Source: LLM Judges as Raters (version updates shifted scores by up to 13% of
# the scale). Fires when a package with a paid judge names a hosted proprietary judge model by a
# floating alias: a name with "latest" or a vendor model id with no dated snapshot suffix. Open
# weight models are fixed by name and are skipped.
# ---------------------------------------------------------------------------

VENDOR = re.compile(r"^(openai/|anthropic/|google/|x-ai/|xai/|gemini/|vertex_ai/|azure/)?"
                    r"(gpt-|chatgpt|o[1-9](-|$)|claude|gemini|grok)", re.I)
SNAPSHOT = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8}|-\d{4}$|-0\d{2}$|-\d{3}$|@\d{8}|-exp-\d{4})")


def _judge_models(pkg: Package):
    out = []
    for m, pf, ln in _code_judge_models(pkg):
        out.append((m, ev(pf, ln)))
    for t in _paid_tpls(pkg):
        if t.kind == "JudgeRubric" and t.model and _kw(t.call, "judge_model") is None:
            out.append((t.model, t.evid + " (JudgeRubric default)"))
    for pf in pkg.py_all():
        if pf.tree is None or "JudgeConfig" not in pf.text:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call) and _callee(n).endswith("JudgeConfig"):
                v = _kw(n, "model")
                v = _value_of(pf, v) if v is not None else None
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.append((v.value, ev(pf, n.lineno)))
    for path, _, _, args in _train_configs(pkg):
        for k, v in args.items():
            if re.search(r"judge\w*model", k) and isinstance(v, str):
                out.append((v, f"{path}:1"))
    return out


@rule("unpinned-judge-model-alias", "hygiene", "low",
      "The judge model is named by a floating alias instead of a dated snapshot, so a silent provider update can "
      "shift reward in the middle of training.")
def r_unpinned_judge(pkg: Package):
    if not _paid_tpls(pkg) and not any(d for d in _judge_models_quick(pkg)):
        return []
    hits = []
    for m, where in _judge_models(pkg):
        base = m.strip().split("/")[-1]
        if not base:
            continue
        if re.search(r"latest", base, re.I) or (VENDOR.search(m.strip()) and not SNAPSHOT.search(base)):
            hits.append((m, where))
    if not hits:
        return []
    m, where = hits[0]
    names = sorted({x for x, _ in hits})
    return [F("unpinned-judge-model-alias",
              f"The judge model {m!r} is a moving alias with no dated snapshot, so the provider can swap the model "
              f"behind it and change the reward scale during a run (names: {', '.join(names[:4])}).",
              [w.split(" ")[0] for _, w in hits], confidence="review")]


def _judge_models_quick(pkg: Package):
    """Cheap check that the package has a judge on the paid path at all, through the existing judge sites."""
    from .rules_judge import _judge_sites
    out = []
    for d in _judge_sites(pkg):
        fi = d["func"]
        if fi is not None and _paid(pkg, fi.name) > 0:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# 8. judge-grades-code-without-execution
# Reader items reader items. Source: ocybrcgtpe (single seed RL result, judges paid plausible code
# that did not build). Fires when a paid judge template is fed code (the prompt names code, a
# program, a component or a script next to the response slot, or the slot is filled from extracted
# code blocks), the judge share of the reward is at least half, and nothing on the grading path runs,
# builds or renders anything (no subprocess, sandbox, exec, compile, test run, browser or renderer).
# ---------------------------------------------------------------------------

CODE_TPL = re.compile(r"(\bcode\b|\bprogram\b|implementation|\bscript\b|\bfunction\b|component|html|css|javascript|"
                      r"typescript|react|python|\bsql\b|snippet|source)", re.I)
RUNS = re.compile(r"(subprocess|\bexec\(|\beval\(|compile\(|sandbox|docker|container|run_tests|pytest|unittest|"
                  r"playwright|puppeteer|selenium|render|remotion|headless|\bnode\b|npm |npx |tsc\b|esbuild|"
                  r"execute|run_code|run_python|code_interpreter|e2b|modal\.|exit_code|returncode|"
                  r"\.run\(|Popen|os\.system|tmp_path|tempfile)", re.I)


def _judge_share(pkg: Package, t: Tpl):
    W = _reward_weights(pkg)
    names = {fi.name for _, fi in t.users}
    tot = sum(w for es in W.values() for w, _ in es if w > 0)
    if t.kind == "JudgeRubric" or not names or tot <= 0:
        return None
    mine = sum(w for nm in names for w, _ in W.get(nm, []) if w > 0)
    return mine / tot if mine else None


@rule("judge-grades-code-without-execution", "hack", "medium",
      "An LLM judge scores code it never runs and no build, test or render step is on the reward path, so code "
      "that looks right but does not work is paid.")
def r_code_no_exec(pkg: Package):
    tp = []
    for t in _paid_tpls(pkg):
        resp = [s for r, s in t.roles.values() if r == "response"]
        codeish = any(re.search(r"code", s, re.I) for s in resp)
        if not codeish:
            near = t.text[:1200]
            codeish = bool(re.search(r"(evaluate|grade|review|judge|assess|score|rate)[^.\n]{0,60}(\bcode\b|\bprogram\b|"
                                     r"implementation|\bscript\b|component|html|snippet)", near, re.I))
        if codeish:
            tp.append(t)
    if not tp:
        return []
    for pf, fi in _grader_funcs(pkg):
        if RUNS.search(fi.src):
            return []
    for pf in pkg.py_all():
        if pf.tree is not None and re.search(r"(subprocess|sandbox|SandboxEnv|PythonEnv|docker|playwright|exec\()", pf.text):
            return []
    hits = []
    for t in tp:
        sh = _judge_share(pkg, t)
        if sh is None or sh >= 0.5:
            hits.append((t, sh))
    if not hits:
        return []
    t, sh = hits[0]
    share = f"{sh:.0%} of the reward" if sh is not None else f"weight {t.paid:g}"
    return [F("judge-grades-code-without-execution",
              f"The judge prompt {t.label} grades code text and pays {share}, but nothing in the package runs, builds "
              "or renders the code, so plausible code that would not run gets the judge's score.",
              [x.evid for x, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# 9. judge-prompt-carries-bias-cues
# Reader items reader items. Main paper arxiv:2611.05662 (cues next to the response flipped up to
# about 40% of correct verdicts). Fires on a paid judge template with a slot filled from the policy
# model's name or provider, a previous score or verdict, or reviewer or pre screen notes.
# ---------------------------------------------------------------------------

@rule("judge-prompt-carries-bias-cues", "hygiene", "low",
      "A paid judge prompt shows the judge the policy model's name, an earlier score or reviewer notes next to the "
      "response, which sways the verdict.")
def r_bias_cues(pkg: Package):
    hits = []
    for t in _paid_tpls(pkg):
        cues = [s for r, s in t.roles.values() if r == "bias"]
        if cues:
            hits.append((t, cues[0]))
    if not hits:
        return []
    t, cue = hits[0]
    return [F("judge-prompt-carries-bias-cues",
              f"The paid judge prompt {t.label} fills a slot with {cue!r} next to the response, so the judge sees who "
              "wrote it or what an earlier grader said and leans toward that.",
              [x.evid for x, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# 10. judge-template-static-review (extends judge-config-hygiene)
# Reader items JU0486 (rationale before score, EM 42.6% to 51.9%, arxiv:2609.02246), JU0097 (a text
# only judge told to run the test cases hallucinates the check, arxiv:2501.18099), JU0090, JU0431,
# JU0466, JU0476, JU0523, JU0530, JU0663. Static reads of the judge prompt only:
#   a) the output format asks for the score or verdict before the reasoning
#   b) the judge is told to run or execute code or tests, and the judge call gives it no tools
# Batching is judge-many-criteria-one-call. Token budgets are judge-config-hygiene. The other reader
# items need a judge run or are covered elsewhere (see the notes).
# ---------------------------------------------------------------------------

SCORE_KEY = re.compile(r"(\"(score|rating|verdict|grade|label|decision|judgement|judgment|result)\"\s*:|"
                       r"^\s*(score|rating|verdict|grade|decision|judgement|judgment)\s*:|"
                       r"<(score|rating|verdict|grade)>|\b(SCORE|VERDICT|RATING|GRADE)\s*:)", re.I | re.M)
REASON_KEY = re.compile(r"(\"(reasoning|explanation|rationale|justification|analysis|reason|thoughts?|critique|"
                        r"feedback|comment)\"\s*:|^\s*(reasoning|explanation|rationale|justification|analysis|reason|"
                        r"thoughts?|critique)\s*:|<(reasoning|explanation|rationale|analysis|thinking|think)>|"
                        r"\b(REASONING|EXPLANATION|RATIONALE|JUSTIFICATION)\s*:)", re.I | re.M)
SCORE_THEN = re.compile(r"(number|score|rating|verdict|grade|yes or no|\"yes\" or \"no\")[^.\n]{0,40}"
                        r"(first|,)?\s*(then|followed by|and then|after that)[^.\n]{0,30}"
                        r"(explanation|reason|justification|rationale|critique)", re.I)
RUN_TESTS = re.compile(r"\b(run|execute|compile)\s+(the\s+|all\s+|each\s+)?(provided\s+|given\s+|following\s+|hidden\s+)?"
                       r"(test cases?|tests|unit tests|code|program|script|snippet)\b", re.I)


def _score_first(text: str):
    tail = text[-1500:]
    s, r = SCORE_KEY.search(tail), REASON_KEY.search(tail)
    if s and r and s.start() < r.start():
        return "asks for the score before the reasoning"
    if SCORE_THEN.search(tail):
        return "asks for the score first and the explanation after it"
    return None


# 0.5.0 fix wave (check_4.md): a judge told to give only a yes or no answer writes no reasoning, so there is
# nothing for the score to come before.
YES_NO_ONLY = re.compile(r"(answer|respond|reply|output|return|say)\w*[^.\n]{0,40}?[\"'`*]*(yes|no|true|false)[\"'`*]*"
                         r"\s*(or|/)\s*[\"'`*]*(no|yes|false|true)\b", re.I)
ASKS_REASON = re.compile(r"(explain|explanation|give (your |a )?reason|provide (your |a |an )?(reason|rationale|justification|"
                         r"analysis)|reasoning\b|rationale|justif|step by step|think through|chain of thought|<think)", re.I)


def _yes_no_only(t) -> bool:
    if ASKS_REASON.search(t.text):
        return False
    if YES_NO_ONLY.search(t.text):
        return True
    for _, fi in t.users:
        m = re.search(r"max_(completion_)?tokens\s*=\s*(\d+)", fi.src)
        if m and int(m.group(2)) <= 32:
            return True
    return False


@rule("judge-template-static-review", "hygiene", "low",
      "A paid judge prompt asks for its score before its reasoning, or tells a judge with no tools to run the code "
      "or tests.")
def r_judge_template_review(pkg: Package):
    tp = _paid_tpls(pkg)
    if not tp:
        return []
    hits = []
    for t in tp:
        why = _score_first(t.text)
        if why and _yes_no_only(t):
            why = None
        if not why:
            m = RUN_TESTS.search(t.text)
            if m and not re.search(r"(tool|function call|code interpreter|execution result|output of running|"
                                   r"test results?|stdout|stderr)", t.text, re.I):
                calls_tools = any("tools=" in fi.src or "tool_choice" in fi.src for _, fi in t.users)
                if not calls_tools:
                    why = f"tells the judge to {m.group(0).lower()!s} but the judge has no tool to do it, so it can only " \
                          "guess the result"
        if why:
            hits.append((t, why))
    if not hits:
        return []
    t, why = hits[0]
    return [F("judge-template-static-review",
              f"The paid judge prompt {t.label} {why}.", [x.evid for x, _ in hits], confidence="review")]


# ---------------------------------------------------------------------------
# 11. rubric-static-review (extends aggregation-without-gate)
# Reader items JU0272 (rubric reliability alpha 0.19 to 0.31 without lexical anchors vs 0.53 to 0.71
# with, arxiv:2610.24807), JU0181 (one high weight item locks in a wrong reference, arxiv:2605.03871),
# JU0004, JU0196. Static reads of a scored rubric:
#   a) one criterion holds more than half of the positive weight in a rubric of at least 3 items
#   b) at least half of 4 or more criteria are vague: short, no number, no quoted or named term, and
#      built from words like good, clear, appropriate, quality
# Missing negative criteria is rubric-presence-only-no-penalty. A rubric that states an answer
# different from gold needs the gold per row and is not built (see the notes).
# ---------------------------------------------------------------------------

VAGUE = re.compile(r"\b(good|clear|clarity|quality|appropriate|well|overall|helpful|coherent|relevant|reasonable|"
                   r"adequate|sufficient|proper|nice|engaging|effective|strong|thorough|accurate|correct|complete|"
                   r"comprehensive|professional|logical|fluent|natural|useful|meaningful|sensible)\b", re.I)
ANCHOR = re.compile(r"(\d|['\"`]|\b[A-Z][a-z]+[A-Z]|\b[A-Z]{2,}\b|\(|:|(?i:\bat least\b|\bexactly\b|\bno more than\b|"
                    r"\bnamed?\b|\bspecific))")


def _vague(text: str) -> bool:
    t = text.strip()
    words = t.split()
    if len(words) > 12 or not VAGUE.search(t):
        return False
    body = re.sub(r"^\s*(is|are|the|response|answer|output|has|have|uses?|shows?|it)\b", "", t, flags=re.I)
    return not ANCHOR.search(body[1:] if body[:1].isupper() else body)


@rule("rubric-static-review", "hygiene", "low",
      "A scored rubric puts most of its weight on one criterion, or most of its criteria are vague words with no "
      "concrete term or threshold.")
def r_rubric_review(pkg: Package):
    gs = _rubrics(pkg)
    if not gs:
        return []
    hits = []
    for g in gs:
        its = g.items
        if g.nrows > 1:
            continue  # per row rubrics are checked by the presence rule; a dominant item per row needs row bounds
        ws = [x.w for x in its if x.w is not None and x.w > 0]
        if len(its) >= 3 and len(ws) == len(its):
            tot = sum(ws)
            top = max(its, key=lambda x: x.w)
            if tot > 0 and top.w / tot > 0.5:
                hits.append((g, f"gives {top.w / tot:.0%} of its weight to the single criterion {top.text[:60]!r}, so "
                                "that one call decides the score"))
                continue
        if len(its) >= 4:
            v = [x for x in its if _vague(x.text)]
            if len(v) >= 0.5 * len(its):
                hits.append((g, f"has {len(v)} of {len(its)} criteria with no concrete term or threshold "
                                f"(for example {v[0].text[:50]!r}), so the judge's verdict on them is close to a coin flip"))
    if not hits:
        return []
    g, why = hits[0]
    return [F("rubric-static-review", f"The rubric {g.where} {why}.", [x.src for x, _ in hits], confidence="review")]

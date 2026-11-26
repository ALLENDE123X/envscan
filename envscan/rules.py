"""Static rules.

Every rule is a function `rule(pkg) -> list[Finding]` registered with @rule.
Rules only read source text and ASTs; they never import or run env code.

Each finding carries:
  rule        stable id
  kind        "hack" (unearned reward), "broken" (correct work scored wrong or
              env cannot load/score), or "hygiene" (risk worth knowing, never
              counted as a defect on the scoreboard)
  severity    hint: critical / high / medium / low / info
  confidence  "likely" (pattern that matched hand verified defects with high
              precision) or "review" (needs a human to confirm)
  evidence    list of "file:line" strings
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import asdict, dataclass, field

from .package import Package, PyFile, FuncInfo

RULES: dict[str, dict] = {}

# Rules that run only when asked for, by `--enable RULE` or ENVSCAN_ENABLE=rule1,rule2.
# A rule is here when its measured precision is too low to report by default.
OFF_BY_DEFAULT: set[str] = {
    # 0.4.0: failed their tests (see CHANGELOG).
    "spec-oracle-gap", "judge-config-hygiene",
    # 0.5.0: the independent read of the new Hub hits found these wrong, harmless or low value
    # too often (the internal notes).
    "unpinned-judge-model-alias", "judge-prompt-missing-guard-instructions",
    "tool-call-count-or-name-reward", "observation-truncation-silent", "secrets-in-agent-env",
    "empty-answer-from-truncation", "judge-prompt-lacks-reference", "eval-harness-template-bug",
    "judge-many-criteria-one-call", "mcq-gold-position-skew", "solution-in-task-text",
    "thin-test-suite", "tool-call-parser-lenient", "count-reward-no-dedup",
    "env-init-state-mismatch", "task-toml-unknown-keys", "prompt-wording-raises-cheating",
    "grader-or-chain", "agent-runs-as-root",
    # 0.5.0: no Hub hit to judge, and the rule's author asked for it to ship off.
    "oracle-solution-echoes-answer", "root-ops-on-agent-writable-paths",
    "rubric-presence-only-no-penalty", "rubric-static-review", "tool-contract-conformance",
    "multi-answer-exact-diff-no-checker", "test-inputs-below-constraint-range",
}
# Branches of a rule that run only when asked for, as "rule-id:variant". A finding made with
# F(..., variant="x") is dropped when "rule-id:x" is here, unless "rule-id:x" or "rule-id" is in
# ENABLED or the rule was asked for by name. 0.5.0 fix wave (the internal notes).
OFF_VARIANTS: set[str] = {
    "keyword-reward:similarity", "aggregation-without-gate:v05", "grader-fails-open:v05",
    "format-heavy-reward:v05", "verdict-parse-semantics:v05", "first-match-label-regex:first-fenced-block",
}
ENABLED: set[str] = {r.strip() for r in os.environ.get("ENVSCAN_ENABLE", "").split(",") if r.strip()}


@dataclass
class Finding:
    rule: str
    kind: str
    severity: str
    confidence: str
    message: str
    evidence: list[str] = field(default_factory=list)
    variant: str = ""

    def to_dict(self):
        d = asdict(self)
        if not d["variant"]:
            del d["variant"]
        return d


def rule(rid: str, kind: str, severity: str, description: str):
    def deco(fn):
        RULES[rid] = {"id": rid, "kind": kind, "severity": severity, "description": description, "fn": fn}
        fn.rid = rid
        return fn
    return deco


def F(rid, message, evidence, confidence="likely", severity=None, kind=None, variant=""):
    meta = RULES[rid]
    return Finding(rid, kind or meta["kind"], severity or meta["severity"], confidence, message,
                   sorted(set(evidence))[:12], variant)


def variant_on(rid: str, variant: str, only=None) -> bool:
    """True when a finding of this rule branch should be reported."""
    if not variant:
        return True
    key = f"{rid}:{variant}"
    if key not in OFF_VARIANTS:
        return True
    return bool(only and (rid in only or key in only)) or key in ENABLED or rid in ENABLED


def ev(pf_or_path, line):
    p = pf_or_path.path if isinstance(pf_or_path, PyFile) else pf_or_path
    return f"{p}:{line}"


def line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

SCORING_NAME = re.compile(
    r"(reward|score|scoring|grade|grading|evaluat|verif|judge|run_tests|_tests?\b|tests?_|"
    r"post_rollout|finalize|is_resolved|resolve|check_|correct|passed|solved|success|accuracy)", re.I)
TEST_TOKEN = re.compile(
    r"(test\.sh|run_tests|run-tests|\bpytest\b|score\.py|/tests\b|\btests/|verify\.py|grader\.py|"
    r"eval\.sh|test_patch|run_script\.sh|parser\.py|unittest|cargo test|go test|npm test|"
    r"jest|mvn test|gradle test|lake build|lake env lean)", re.I)
BOX_EXEC = re.compile(
    r"\b(?:runtime|sandbox\w*|container\w*|box|session|sb|self\.sandbox\w*|self\.runtime|"
    r"self\.container|env\.sandbox|sandbox_client|self\.sandbox_client|self\._sandbox\w*|"
    r"state\[\s*['\"]sandbox['\"]\s*\])\s*\.\s*"
    r"(?:run|run_command\w*|run_uv_script|run_script|exec|exec_run|exec_command|execute|"
    r"execute_command\w*|run_background_job|send_keys|process\.exec|commands\.run|arun|aexec)\s*\(")
SANDBOX_ID_EXEC = re.compile(r"execute_command\w*\(\s*(?:sandbox_id|state\[|self\.sandbox_id|sid\b)")
FRESH_BOX = re.compile(
    r"(create_sandbox|Sandbox\.create|CreateSandboxRequest|sandboxes\.create|containers\.run\(|"
    r"docker\.from_env|fresh_copy\s*=\s*True|IsolatedVerifier|new_sandbox|spawn_sandbox|"
    r"modal\.Sandbox|stage_verifier|fresh sandbox|fresh container)")
AGENT_TOOLS = re.compile(
    r"(SandboxEnv|ToolEnv|StatefulToolEnv|PythonEnv|CliAgentEnv|HarborEnv|HarborTaskset|ComposableEnv|"
    r"RLMEnv|rlm_harness|harness\s*=|Harness\(|tools\s*=\s*\[|add_tool\(|def bash\(|def execute_bash|"
    r"def run_command|def shell\(|def execute\(|def run_python|tmux|send_keys|str_replace_editor|"
    r"file_editor|def edit_file|def write_file|@vf\.tool|Toolset|workdir\s*[=:]|NEEDS_CONTAINER|docker_image|"
    r"/testbed|capture_patch)")
ISOLATION_ANY = re.compile(
    r"(fresh_copy\s*=\s*True|IsolatedVerifierConfig|isolated_verifier|environment_mode\s*=\s*['\"]separate|"
    r"mode\s*=\s*['\"]separate['\"]|stage_verifier|artifact_max_bytes|hand-?in travels)")

ANSWERISH = {"answer", "answers", "gold", "golden", "reference", "references", "expected", "target", "targets",
             "solution", "solutions", "ground_truth", "groundtruth", "gt", "label", "labels", "truth",
             "correct", "correct_answer", "expected_output", "expected_answer", "ref", "gold_answer",
             "test_cases", "tests", "verifier", "oracle"}
CORRECTNESS_CALLS = re.compile(
    r"(math_verify|verify\(|parse\(|\.judge\(|judge|compare|is_equiv|equivalent|==|grade_answer|"
    r"exact_match|f1_score|evaluate\(|run_tests|execute|subprocess|sandbox|sympy|"
    r"check_answer|score_answer|assert|levenshtein|similarity|rouge|bleu)")


def all_funcs(pkg: Package):
    for pf in pkg.py_all():
        for fi in pf.funcs:
            yield pf, fi


def names_loaded(node) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 40:
            out.add(n.value)
    return out


def referenced_consts(pf: PyFile, fi: FuncInfo) -> str:
    """Text of module level string constants a function refers to (one hop)."""
    out = []
    used = {n.id for n in ast.walk(fi.node) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(fi.node) if isinstance(n, ast.Attribute)}
    for name in used:
        if name in pf.consts:
            out.append(pf.consts[name])
    return "\n".join(out)


def called_names(fi: FuncInfo) -> set[str]:
    out = set()
    for n in ast.walk(fi.node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                out.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                out.add(n.func.attr)
    return out


def scoring_funcs(pkg: Package):
    """Functions on the reward path: named like a scorer, decorated as a reward,
    registered in a rubric, plus anything those call in the same package (one hop)."""
    reg = registered_reward_names(pkg)
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


REWARDISH_NAME = re.compile(r"(reward|score|scoring|grade|grading|judge|verif|check_answer|compute_metric|_metric\b|passed|solved|correctness)", re.I)


def reward_path_funcs(pkg: Package):
    """Tighter than scoring_funcs: registered rewards, @reward methods and reward named functions,
    plus their direct callees."""
    reg = registered_reward_names(pkg)
    by_name: dict[str, list] = {}
    for pf, fi in all_funcs(pkg):
        by_name.setdefault(fi.name, []).append((pf, fi))
    seeds = [(pf, fi) for pf, fi in all_funcs(pkg)
             if fi.name in reg or any("reward" in d for d in fi.decorators) or REWARDISH_NAME.search(fi.name)]
    out = {(pf.path, fi.start): (pf, fi) for pf, fi in seeds}
    for pf, fi in seeds:
        for cn in called_names(fi):
            for pf2, fi2 in by_name.get(cn, []):
                out.setdefault((pf2.path, fi2.start), (pf2, fi2))
    return list(out.values())


def registered_reward_names(pkg: Package) -> dict[str, float]:
    """name -> weight for functions placed in a rubric (funcs=[...], weights=[...],
    add_reward_func(f, weight=w), @vf.reward(weight=w))."""
    out: dict[str, float] = {}
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call):
                fname = n.func.attr if isinstance(n.func, ast.Attribute) else (n.func.id if isinstance(n.func, ast.Name) else "")
                kws = {k.arg: k.value for k in n.keywords if k.arg}
                if fname.endswith("Rubric") or fname in ("Rubric", "RubricGroup") or "funcs" in kws:
                    funcs = kws.get("funcs") or kws.get("reward_funcs")
                    if funcs is None and n.args and isinstance(n.args[0], (ast.List, ast.Tuple)):
                        funcs = n.args[0]
                    weights = kws.get("weights")
                    if funcs is not None and isinstance(funcs, (ast.List, ast.Tuple)):
                        ws = []
                        if isinstance(weights, (ast.List, ast.Tuple)):
                            for w in weights.elts:
                                try:
                                    ws.append(float(ast.literal_eval(w)))
                                except Exception:
                                    ws.append(1.0)
                        for i, e in enumerate(funcs.elts):
                            nm = _callable_name(e)
                            if nm:
                                out[nm] = ws[i] if i < len(ws) else 1.0
                if fname == "add_reward_func" and n.args:
                    nm = _callable_name(n.args[0])
                    w = 1.0
                    wnode = kws.get("weight") or (n.args[1] if len(n.args) > 1 else None)
                    if wnode is not None:
                        try:
                            w = float(ast.literal_eval(wnode))
                        except Exception:
                            pass
                    if nm:
                        out[nm] = w
        for fi in pf.funcs:
            for d in fi.decorators:
                if re.search(r"\breward\b", d):
                    m = re.search(r"weight\s*=\s*([0-9.]+)", d)
                    out.setdefault(fi.name, float(m.group(1)) if m else 1.0)
    return out


def _callable_name(e):
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        return e.attr
    if isinstance(e, ast.Call):  # factory: rf_x()
        return _callable_name(e.func)
    return None


# ---------------------------------------------------------------------------
# 1. grader runs inside the agent's box
# ---------------------------------------------------------------------------

@rule("grader-in-agent-sandbox", "hack", "critical",
      "Reward path runs tests or a scorer inside the same sandbox/container the agent controlled, "
      "with no fresh box or isolated verifier.")
def r_grader_in_box(pkg: Package):
    text = pkg.all_py_text()
    if not AGENT_TOOLS.search(text):
        return []
    if ISOLATION_ANY.search(text):
        return []
    hits = []
    weak = True  # only a math style verify.py with the answer passed as args
    for pf, fi in scoring_funcs(pkg):
        src = fi.src
        if FRESH_BOX.search(src):
            continue
        if not (BOX_EXEC.search(src) or SANDBOX_ID_EXEC.search(src)):
            continue
        ctx = src + "\n" + referenced_consts(pf, fi)
        toks = {m.group(0).lower() for m in TEST_TOKEN.finditer(ctx)}
        if toks and toks != {"verify.py"}:
            weak = False
        if toks:
            m = BOX_EXEC.search(src) or SANDBOX_ID_EXEC.search(src)
            hits.append(ev(pf, fi.start + line_of(src, m.start()) - 1))
    if not hits:
        return []
    mitig = re.search(r"(sha256|checksum|manifest|integrity|tamper|--noconftest|PYTHONNOUSERSITE|python3?\s+-I\b|rm\s+-rf\s+/tests)", text)
    return [F("grader-in-agent-sandbox",
              "Tests/scorer are executed in the agent's own sandbox from a scoring function; a planted "
              "conftest.py, shadowed binary or edited test file can force a pass" +
              (" (some integrity hardening is present; confirm it covers the runner)." if mitig else "."), hits,
              confidence="review" if (mitig or weak) else "likely")]


def _harbor_task_shared(text: str) -> bool:
    """True when a Harbor task.toml leaves a verifier in the agent's container.
    Harbor runs a verifier separately when environment_mode = "separate" or when a
    [verifier.environment] table is present, at the top level or per step."""
    def mode(v):
        if not isinstance(v, dict):
            return None
        m = v.get("environment_mode")
        if isinstance(m, str):
            return m.lower()
        return "separate" if isinstance(v.get("environment"), dict) else None
    try:
        import tomllib
        cfg = tomllib.loads(text)
    except Exception:
        cfg = None
    if isinstance(cfg, dict) and cfg:
        top = mode(cfg.get("verifier", {})) or "shared"
        steps = cfg.get("steps") if isinstance(cfg.get("steps"), list) else []
        modes = [(mode(s.get("verifier", {})) if isinstance(s, dict) else None) or top for s in steps]
        return any(m == "shared" for m in modes) if modes else top == "shared"
    # template placeholders or a broken file: fall back to text
    return not re.search(r"environment_mode\s*=\s*['\"]separate|^\s*\[verifier\.environment\]", text, re.M)


@rule("harbor-shared-grading", "hack", "critical",
      "Uses the verifiers Harbor taskset / Harbor task format without a separate verifier "
      "(ScoringConfig.mode defaults to shared: test.sh runs in the agent's box).")
def r_harbor_shared(pkg: Package):
    text = pkg.all_py_text()
    evid = []
    for pf in pkg.py_all():
        for i, ln in enumerate(pf.lines, 1):
            if re.search(r"\b(HarborTaskset|HarborDatasetTaskSet|HarborEnv)\b", ln) and ("import" in ln or "class" in ln or "(" in ln):
                evid.append(ev(pf, i))
                break
    tomls = [p for p in pkg.files if os.path.basename(p) == "task.toml"]
    sep = [p for p in tomls if not _harbor_task_shared(pkg.files[p])]
    if not evid and not tomls:
        return []
    if ISOLATION_ANY.search(text) or (tomls and len(sep) == len(tomls)):
        return []
    # A plain Harbor task (no verifiers environment code around it): the shared verifier is
    # Harbor's default, true of every task in a dataset, so it is reported at review.
    native = not evid and not re.search(r"\bdef\s+load_environment\b|^\s*(?:import|from)\s+verifiers\b", text, re.M)
    if not evid:
        # task.toml tasks graded by some in-env code: require a test.sh next to them
        tsh = [p for p in pkg.files if p.endswith("tests/test.sh")]
        if not tsh:
            return []
        evid = [f"{tsh[0]}:1"]
    strong = bool(tomls) or bool(re.search(r"^(from\s+verifiers[\w.]*\s+import\s+[^\n]*Harbor|class\s+\w+\([^)]*Harbor)", text, re.M))
    msg = "Harbor style grading in the agent's box"
    if tomls:
        msg += f": {len(tomls) - len(sep)} of {len(tomls)} shipped task.toml declare no separate verifier"
    if native:
        return [F("harbor-shared-grading", msg + ". This is Harbor's default (shared verifier): test.sh runs after the "
                  "agent with the agent's PATH, files and startup hooks. One change fixes a whole dataset: a separate "
                  "verifier ([verifier] environment_mode = \"separate\").",
                  evid + ([f"{tomls[0]}:1"] if tomls else []), confidence="review")]
    return [F("harbor-shared-grading", msg + "; test.sh runs with the agent's PATH, /tests and startup files.",
              evid + ([f"{tomls[0]}:1"] if tomls else []), confidence="likely" if strong else "review")]


@rule("keep-sandbox-for-scoring", "hack", "critical",
      "Scoring explicitly reuses the agent's sandbox (keep_sandbox_for_scoring=True) or uses "
      "RemoteHybridMathRubric, which runs score.py inside the agent's sandbox.")
def r_keep_sandbox(pkg: Package):
    out = []
    keep = []
    for pf in pkg.py_all():
        for i, ln in enumerate(pf.lines, 1):
            if re.search(r"keep_sandbox_for_scoring\s*=\s*True", ln):
                keep.append(ev(pf, i))
            if re.search(r"\bRemoteHybridMathRubric\b", ln) and not ln.lstrip().startswith("#"):
                out.append(ev(pf, i))
    if keep:
        # only counts if scoring then executes something in that box (reading an answer file is fine)
        text = pkg.all_py_text()
        execs = [1 for pf, fi in scoring_funcs(pkg) if BOX_EXEC.search(fi.src) and
                 re.search(r"\.(run|exec\w*|execute\w*|run_command\w*)\s*\(", fi.src) and
                 (TEST_TOKEN.search(fi.src + referenced_consts(pf, fi)) or
                  re.search(r"(python3?\s|bash\s|sh\s+-c|uv\s+run|node\s|\bmake\b|\blake\b|cargo|go\s+run)", fi.src))]
        lib = re.search(r"composable\.tasksets\.math|\bMathTaskSet\b", text)
        if execs or lib or re.search(r"HarborDatasetTaskSet|HarborTaskset|test\.sh|run_tests", text):
            out += keep
    if not out:
        return []
    return [F("keep-sandbox-for-scoring",
              "The scorer runs in the sandbox the agent had root in; agent written modules, binaries or "
              "files can shadow the grader.", out)]


@rule("partial-test-restore", "hack", "high",
      "Grading script restores only listed test files (or only one test dir) and then runs pytest in "
      "the agent's tree, so a planted conftest.py/plugin survives.")
def r_partial_restore(pkg: Package):
    out = []
    n_sh = 0
    for p, t in pkg.text_files(".sh"):
        if "pytest" not in t:
            continue
        if re.search(r"git\s+checkout\s+\S+\s+--\s+\S", t) or re.search(r"git\s+checkout\s+\$\w+\s+\$", t):
            if not re.search(r"(--noconftest|-p\s+no:|git\s+clean\s+-[a-z]*f|find\s+.*conftest.*-delete|rm\s+-[rf]+\s+.*conftest)", t):
                n_sh += 1
                if len(out) < 3:
                    m = re.search(r"git\s+checkout", t)
                    out.append(ev(p, line_of(t, m.start())))
    if not out:
        return []
    text = pkg.all_py_text()
    if ISOLATION_ANY.search(text) and not re.search(r"HarborTaskset|keep_sandbox_for_scoring", text):
        conf = "review"
    else:
        conf = "likely"
    return [F("partial-test-restore",
              f"{n_sh} grading script(s) reset only the listed test files before pytest; conftest.py, "
              "pytest.ini and plugins the agent planted are left in place.", out, confidence=conf)]


@rule("answer-key-staged-in-runtime", "hack", "critical",
      "Scoring writes test cases / expected outputs into the rollout runtime and runs the checker "
      "there, next to the agent's solution.")
def r_answer_staged(pkg: Package):
    text = pkg.all_py_text()
    if ISOLATION_ANY.search(text):
        return []
    out = []
    for pf, fi in scoring_funcs(pkg):
        if "runtime" not in fi.params and "sandbox" not in " ".join(fi.params):
            continue
        src = fi.src
        if FRESH_BOX.search(src):
            continue
        w = re.search(r"\b(?:runtime|sandbox\w*)\s*\.\s*(?:write\w*|upload\w*|put\w*)\s*\(([^)]*)", src)
        if not w:
            continue
        window = src[w.start(): w.start() + 600]
        if re.search(r"(test_cases|tests_path|tests\.json|expected_out\w*|private_test|hidden_test|test_\w+\.py|/tests/)", window, re.I) \
                and BOX_EXEC.search(src):
            out.append(ev(pf, fi.start + line_of(src, w.start()) - 1))
    if not out:
        return []
    conf = "likely" if AGENT_TOOLS.search(text) or "vf.Runtime" in text else "review"
    return [F("answer-key-staged-in-runtime",
              "Hidden tests or expected outputs are written into the runtime that ran the policy and "
              "checked there; an agentic policy can read them or rewrite the checker.", out, confidence=conf)]


# ---------------------------------------------------------------------------
# 2. exit code as verdict
# ---------------------------------------------------------------------------

MODEL_CODE = {"code", "solution", "completion", "program", "submission", "generated_code", "student_code",
              "src", "source", "answer_code", "candidate", "user_code", "model_code", "parsed_code",
              "extracted_code", "solution_code", "code_str", "submitted_code", "response", "student"}


@rule("exit-code-verdict", "hack", "critical",
      "Pass/fail is the exit status of a process that runs model code spliced into the test "
      "harness, so `sys.exit(0)` / `_exit(0)` before the asserts passes.")
def r_exit_code(pkg: Package):
    out_likely, out_review = [], []
    for pf, fi in all_funcs(pkg):
        src = fi.src
        # the exit status must be the pass condition, not just a guard before parsing output
        verdict_form = re.search(
            r"return\s+[^\n]*(returncode|exit_code|exitcode)\s*==\s*0|"
            r"(returncode|exit_code|exitcode)\s*==\s*0\s*:\s*\n\s*(return\s+(True|1|1\.0)\b|\w*(pass|score|reward|success)\w*\s*(\+=|=))|"
            r"(pass\w*|success\w*|ok|score|reward)\s*=\s*[^\n]*(returncode|exit_code|exitcode)\s*==\s*0|"
            r"float\(\s*\w+\.(returncode|exit_code)\s*==\s*0", src)
        if not verdict_form:
            continue
        if re.search(r"(stdout|output)\w*[^\n]*(==|compare\w*\(|in\s+\w+)|json\.loads\(\s*\w*\.?stdout|PASSED|FAILED", src):
            continue
        spliced = False
        for n in ast.walk(fi.node):
            if isinstance(n, ast.JoinedStr):
                vals = [v for v in n.values if isinstance(v, ast.FormattedValue)]
                lits = "".join(v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
                for v in vals:
                    if isinstance(v.value, ast.Name) and v.value.id in MODEL_CODE and \
                            re.search(r"(assert|expected|test|==|check|result)", lits + src[:0]):
                        spliced = True
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                parts = [n.left, n.right]
                nm = [p.id for p in parts if isinstance(p, ast.Name)]
                if any(x in MODEL_CODE for x in nm) and any(re.search(r"test|assert|harness|check", x, re.I) for x in nm + [str(_const_text_safe(p)) for p in parts]):
                    spliced = True
        m = re.search(r"(returncode|exit_code|exitcode)", src)
        line = fi.start + line_of(src, m.start()) - 1 if m else fi.start
        if spliced:
            out_likely.append(ev(pf, line))
        elif SCORING_NAME.search(fi.name) and TEST_TOKEN.search(src) and not re.search(r"(PASSED|FAILED|passed|failed|junit|xml|json\.loads|parse)", src):
            out_review.append(ev(pf, line))
    res = []
    if out_likely:
        res.append(F("exit-code-verdict", "Model code is pasted into the test harness and the process exit "
                     "code decides the pass; exiting 0 early passes every test.", out_likely))
    elif out_review:
        res.append(F("exit-code-verdict", "Reward is a test process exit code with no parsed per test results.",
                     out_review, confidence="review", severity="medium"))
    return res


def _const_text_safe(n):
    try:
        from .package import _const_text
        return _const_text(n) or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 3. keyword / format rewards
# ---------------------------------------------------------------------------

ANSWER_KEY_RE = re.compile(r"^(answer|answers|solution|solutions|gold\w*|ground_truth|expected\w*|correct\w*|"
                           r"target(?!_?(keyword|term|concept|word))\w*|reference\w*|ref|label|labels|optimal\w*|best_\w+|flag|"
                           r"gt|truth|oracle|key_value|final_answer|required_\w+|secret\w*|target_word)$", re.I)
VERIFY_CALL = re.compile(r"(math_verify|\bverify\w*\(|\.judge\(|judge\w*\(|grade_answer|is_equiv|exact_match|run_tests|"
                         r"\bexecute\w*\(|subprocess|sandbox|simulat\w*\(|sympy|evaluate_\w+\(|compare\w*\(|similarity|"
                         r"rouge|bleu|\bf1\w*\(|levenshtein|allclose|isclose|score_answer|check_answer|run_simulation|\.grade\w*\(|"
                         r"ServiceClient|\.post\(|httpx|requests\.|aiohttp|"
                         r"\.step\(|env\.\w+\(|state\[|get_state|game\.|board\.|json\.loads|literal_eval|get_user_messages|state\.get\(|episode_return|trajectory|playwright|"
                         r"to_thread|grader|_grade\w*\(|verifier|"
                         r"final_score|game_state|\.score\b)")


def _literal_probes(node) -> int:
    """Count keyword style probes of the response: re.search/findall/match/count with a literal
    pattern, `"literal" in text`, and comprehension style keyword loops."""
    n = 0
    for x in ast.walk(node):
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute):
            if x.func.attr in ("search", "findall", "match", "finditer", "fullmatch") and x.args:
                a0 = x.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str) and "|" in a0.value:
                    n += 1 + a0.value.count("|") // 3
                elif isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    n += 0.5
            if x.func.attr == "count" and x.args and isinstance(x.args[0], ast.Constant):
                n += 1
        if isinstance(x, ast.Compare) and any(isinstance(o, ast.In) for o in x.ops):
            if isinstance(x.left, ast.Constant) and isinstance(x.left.value, str):
                n += 1
            elif isinstance(x.left, ast.Name) and isinstance(x.comparators[0], (ast.Name, ast.Attribute, ast.Call)):
                n += 0.5  # kw in text inside a loop over a keyword list
        if isinstance(x, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            tnames = {t.id for g in x.generators for t in ast.walk(g.target) if isinstance(t, ast.Name)}
            for c in ast.walk(x):
                if isinstance(c, ast.Compare) and any(isinstance(o, ast.In) for o in c.ops):
                    lft = {t.id for t in ast.walk(c.left) if isinstance(t, ast.Name)}
                    if lft & tnames:
                        n += 2
                        break
        if isinstance(x, ast.For):
            tnames = {t.id for t in ast.walk(x.target) if isinstance(t, ast.Name)}
            for c in ast.walk(x):
                if isinstance(c, ast.Compare) and any(isinstance(o, ast.In) for o in c.ops):
                    lft = {t.id for t in ast.walk(c.left) if isinstance(t, ast.Name)}
                    if lft & tnames:
                        n += 2
                        break
    return n


def _uses_answer(fi: FuncInfo) -> bool:
    for stmt in fi.node.body:
        for x in ast.walk(stmt):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and ANSWER_KEY_RE.match(x.id):
                return True
            if isinstance(x, ast.Attribute) and ANSWER_KEY_RE.match(x.attr):
                return True
            key = None
            if isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Constant) and isinstance(x.slice.value, str):
                key = x.slice.value
            elif isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "get" and x.args \
                    and isinstance(x.args[0], ast.Constant) and isinstance(x.args[0].value, str):
                key = x.args[0].value
            if key and ANSWER_KEY_RE.match(key):
                return True
    return False


def _classify_reward(pf: PyFile, fi: FuncInfo) -> str:
    """correctness | bag | format | other"""
    probes = _literal_probes(fi.node)
    src = fi.src
    body = src.split(":", 1)[1] if ":" in src else src
    uses_answer = _uses_answer(fi)
    verify = bool(VERIFY_CALL.search(body))
    if probes >= 2 and not uses_answer and not verify:
        return "bag"
    if probes >= 3 and re.search(r"(keyword|term|concept|vocab|phrase|indicator|buzz)", src, re.I) and not verify:
        return "bag"
    if re.search(r"(format|length|structure|xml|tag|think|style|len_|brevity|concise|boxed)", fi.name, re.I) and not uses_answer:
        return "format"
    if uses_answer or verify:
        return "correctness"
    if probes >= 1.5 and re.search(r"(keyword|term|concept|vocab|phrase|indicator|buzz|words?\b)", src, re.I):
        return "bag"
    return "other"


# overlap-metric-reward + similarity-metric-main-reward: reader items, main papers arxiv:2601.18533, arxiv:2603.13224
SIM_CALL = re.compile(r"(rouge_scorer|RougeScorer|\brouge\w*\s*\(|\bRouge\(|sacrebleu|corpus_bleu|sentence_bleu|\bbleu\w*\s*\(|"
                      r"\bBLEU\(|\bchrf\w*\(|CHRF\(|meteor_score|bert_score|BERTScorer|rapidfuzz|\bfuzz\.\w+\(|thefuzz|"
                      r"fuzzywuzzy|Levenshtein\.\w+\(|\blevenshtein\w*\(|editdistance\.eval|\bedit_distance\w*\(|"
                      r"SequenceMatcher\(|\bjaccard\w*\(|cosine_similarity\(|util\.cos_sim\(|\bcos_sim\(|textstat\.\w+\()")
SIM_GATE = re.compile(r"(math_verify|\.judge\(|judge\w*\(|exact_match|is_equiv|grade_answer|run_tests|subprocess|sandbox|"
                      r"verify\w*\()")
COPY_SIDE = re.compile(r"(prompt|question|source|input|context|original|passage|document|article|query|instruction|src_text)", re.I)
# citation-count-reward + citation-not-checked-against-retrieval: reader items, main papers arxiv:2611.01313, arxiv:2608.06012
CITE_PAT = re.compile(r"(https\?://|https?://|\\\[\\d|\[\\d|10\\\.\\d|\bdoi\b|<cite|\[source|\[\^|\bcitation|\bcite\b|arxiv)", re.I)
CITE_LOOKUP = re.compile(r"(retriev|search_result|tool_result|tool_output|visited|fetched|evidence|documents|corpus|"
                         r"state\[|state\.get|info\[|requests\.|httpx|aiohttp|crossref|resolve|exists|valid_url|"
                         r"\bin\s+\w*(urls|seen|retrieved|sources|docs|allowed|known)\b|issubset|intersection|&)", re.I)


def _threshold_gate(node) -> bool:
    """`sim >= 0.7` style compare that turns the metric into a pass or fail."""
    for x in ast.walk(node):
        if isinstance(x, ast.Compare) and any(isinstance(o, (ast.Gt, ast.GtE)) for o in x.ops):
            for c in [x.left] + list(x.comparators):
                if isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and 0.7 <= c.value < 1.0:
                    return True
    return False


def _exact_answer_compare(fi: FuncInfo) -> bool:
    for x in ast.walk(fi.node):
        if isinstance(x, ast.Compare) and any(isinstance(o, (ast.Eq, ast.NotEq)) for o in x.ops):
            side = {n.id for n in ast.walk(x) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(x) if isinstance(n, ast.Attribute)}
            if any(ANSWER_KEY_RE.match(s) for s in side):
                return True
    return False


def _module_assign_text(pf: PyFile, fi: FuncInfo) -> str:
    """Source of module level assignments whose names the function uses (scorer objects built once)."""
    if pf.tree is None:
        return ""
    used = {x.id for x in ast.walk(fi.node) if isinstance(x, ast.Name)}
    out = []
    for st in getattr(pf.tree, "body", []):
        if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id in used for t in st.targets):
            out.append(_unparse(st.value))
    return "\n".join(out)


def _similarity_class(pf: PyFile, fi: FuncInfo) -> str | None:
    """copy | blind | ref for a reward that pays a similarity metric with no gate, else None."""
    body = fi.src.split(":", 1)[1] if ":" in fi.src else fi.src
    if not SIM_CALL.search(body + "\n" + _module_assign_text(pf, fi)) or SIM_GATE.search(body) or _threshold_gate(fi.node) or _exact_answer_compare(fi):
        return None
    if re.search(r"textstat\.\w+\(", body) and not re.search(r"(rouge|bleu|fuzz|levenshtein|SequenceMatcher|jaccard|cos)", body, re.I):
        return "blind"
    for x in ast.walk(fi.node):
        if isinstance(x, ast.Call) and SIM_CALL.search(_unparse(x.func) + "("):
            names = {n.id for a in list(x.args) + [k.value for k in x.keywords] for n in ast.walk(a) if isinstance(n, ast.Name)}
            names |= {n.value for a in x.args for n in ast.walk(a) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            if any(COPY_SIDE.search(nm) for nm in names) and not any(ANSWER_KEY_RE.match(nm) for nm in names):
                return "copy"
    return "ref"


def _citation_count(pf: PyFile, fi: FuncInfo) -> bool:
    """Reward counts URLs, DOIs or bracket citations and pays by the count with no lookup."""
    lits = [x.value for x in ast.walk(fi.node) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
    lits += [pf.consts[n] for n in {x.id for x in ast.walk(fi.node) if isinstance(x, ast.Name)} if n in pf.consts]
    if not any(CITE_PAT.search(s) for s in lits):
        return False
    counted = re.search(r"(len\(\s*(re\.)?findall|len\(\s*set\(\s*(re\.)?findall|\.count\(|sum\(|len\(\s*\w*(cit|url|ref|link|source|doi)\w*)",
                        fi.src, re.I)
    if not counted:
        return False
    body = fi.src.split(":", 1)[1] if ":" in fi.src else fi.src
    if CITE_LOOKUP.search(body) or _uses_answer(fi) or VERIFY_CALL.search(re.sub(r"state\[|state\.get\(", "", body)):
        return False
    return True


@rule("keyword-reward", "hack", "high",
      "Most of the reward weight is keyword / substring / regex counting that does not check the answer, "
      "so a fixed task blind blob of domain words scores like a real answer.")
def r_keyword(pkg: Package):
    reg = _reg_cached(pkg)
    if not reg:
        return []
    funcs = {}
    for pf, fi in all_funcs(pkg):
        if fi.name in reg and fi.name not in funcs:
            funcs[fi.name] = (pf, fi)
    if not funcs:
        return []
    total = sum(reg[n] for n in funcs if reg[n] > 0)
    if total <= 0:
        return []
    cls = {n: _classify_reward(pf, fi) for n, (pf, fi) in funcs.items()}
    bag_w = sum(reg[n] for n, c in cls.items() if c == "bag" and reg[n] > 0)
    fmt_w = sum(reg[n] for n, c in cls.items() if c == "format" and reg[n] > 0)
    corr_w = sum(reg[n] for n, c in cls.items() if c == "correctness" and reg[n] > 0)
    share = (bag_w) / total
    blind = (bag_w + fmt_w) / total
    evid = [ev(funcs[n][0], funcs[n][1].start) for n, c in cls.items() if c == "bag"]
    if bag_w > 0:
        detail = f"keyword {bag_w / total:.0%}, format {fmt_w / total:.0%}, correctness {corr_w / total:.0%} of weight"
        if blind >= 0.55 and share >= 0.25:
            return [F("keyword-reward", f"Answer blind reward dominates ({detail}).", evid)]
        if blind >= 0.4 and share >= 0.2:
            return [F("keyword-reward", f"Large answer blind share ({detail}).", evid, confidence="review", severity="medium")]
    # v0.5: citation counts paid with no lookup
    cite = [n for n, (pf, fi) in funcs.items() if reg[n] > 0 and _citation_count(pf, fi)]
    cite_w = sum(reg[n] for n in cite)
    if cite and cite_w / total >= 0.2:
        return [F("keyword-reward", f"A reward pays by the count of citations or links with no check that they exist or were "
                  f"retrieved ({cite_w / total:.0%} of weight), so made up references raise the score.",
                  [ev(funcs[n][0], funcs[n][1].start) for n in cite], confidence="review", severity="medium")]
    # v0.5: main reward is a similarity metric with no answer gate
    sim = {n: _similarity_class(pf, fi) for n, (pf, fi) in funcs.items() if reg[n] > 0}
    sim = {n: c for n, c in sim.items() if c}
    sim_w = sum(reg[n] for n in sim)
    if sim and sim_w / total >= 0.5:
        sevid = [ev(funcs[n][0], funcs[n][1].start) for n in sim]
        worst = "copy" if "copy" in sim.values() else ("blind" if "blind" in sim.values() else "ref")
        if worst == "copy":
            return [F("keyword-reward", f"The main reward is a similarity score against the prompt text ({sim_w / total:.0%} "
                      "of weight), so copying the input scores high.", sevid, confidence="review", severity="medium",
                      variant="similarity")]
        if worst == "blind":
            return [F("keyword-reward", f"The main reward is a readability score that never reads the answer "
                      f"({sim_w / total:.0%} of weight).", sevid, confidence="review", severity="medium",
                      variant="similarity")]
        return [F("keyword-reward", f"The main reward is a similarity score to the reference with no correctness gate "
                  f"({sim_w / total:.0%} of weight), so near copies and padded text are paid.", sevid,
                  confidence="review", severity="low", kind="hygiene", variant="similarity")]
    return []


# reward-has-no-correctness-term: reader items, main paper arxiv:2610.38977 (quality only RL collapsed to empty code)
QUALITY_CALL = re.compile(r"(pylint|(?<![A-Za-z])(ruff|bandit|radon|mypy)(?![A-Za-z])|flake8|pycodestyle|pyflakes|semgrep|eslint|"
                          r"cyclomatic|maintainability|complexity_score|textstat|readability|detoxify|toxicity|perspective|"
                          r"moderation|llama_?guard|guard_model|safety_score|is_safe\b|code_quality|quality_score|"
                          r"lint_score|style_score|security_score)", re.I)
QUALITY_NAME = re.compile(r"(lint|quality|style|security|safety|safe|toxic|guard|readab|complexity|maintainab|clean)", re.I)


def _quality_term(pf: PyFile, fi: FuncInfo) -> bool:
    body = fi.src.split(":", 1)[1] if ":" in fi.src else fi.src
    if not (QUALITY_CALL.search(body) or (QUALITY_NAME.search(fi.name) and QUALITY_CALL.search(body + fi.name))):
        return False
    if _uses_answer(fi) or re.search(r"(run_tests|pytest|unittest|passed|pass_rate|test_cases|tests\b|math_verify|judge\w*\(|"
                                     r"exact_match|==)", body):
        return False
    return True


@rule("format-heavy-reward", "hack", "medium",
      "Format/length/tag rewards carry half or more of the weight.")
def r_format_heavy(pkg: Package):
    reg = _reg_cached(pkg)
    funcs = {}
    for pf, fi in all_funcs(pkg):
        if fi.name in reg and fi.name not in funcs:
            funcs[fi.name] = (pf, fi)
    for n in reg:
        if n not in funcs and re.search(r"format", n, re.I):
            funcs[n] = None
    total = sum(w for n, w in reg.items() if n in funcs and w > 0)
    if total <= 0:
        return []
    fmt = []
    for n, v in funcs.items():
        if v is None:
            if reg[n] > 0:
                fmt.append(n)
        elif _classify_reward(*v) == "format" and reg[n] > 0:
            fmt.append(n)
    w = sum(reg[n] for n in fmt)
    if w / total >= 0.5 and len(funcs) >= 2:
        evid = [ev(funcs[n][0], funcs[n][1].start) for n in fmt if funcs[n]]
        return [F("format-heavy-reward", f"Format only terms carry {w / total:.0%} of the reward weight.",
                  evid or ["rubric"], confidence="review")]
    # v0.5: every paid term is a quality, lint, guard or format check and none reads the answer or a pass flag
    paid = [n for n in funcs if reg.get(n, 0) > 0]
    if not paid or any(funcs[n] is None for n in paid):
        return []
    qual = [n for n in paid if _quality_term(*funcs[n])]
    if qual and all(n in qual or n in fmt for n in paid):
        return [F("format-heavy-reward", "Every paid reward term is a quality, lint, safety or format check and none checks "
                  "that the answer is correct, so empty or trivial output can score well.",
                  [ev(funcs[n][0], funcs[n][1].start) for n in paid], confidence="review", variant="v05")]
    return []


# ---------------------------------------------------------------------------
# 4. answer key reachable
# ---------------------------------------------------------------------------

@rule("model-named-files-into-test-dir", "hack", "critical",
      "Model chosen filenames are written into a copy of the task directory that also holds tests or a "
      "reference solution (.meta), so the model can overwrite tests or read the reference.")
def r_model_files(pkg: Package):
    out = []
    for pf, fi in all_funcs(pkg):
        src = fi.src
        if "copytree" not in src and "copy_tree" not in src:
            continue
        if re.search(r"ignore\s*=", src):
            continue
        for n in ast.walk(fi.node):
            if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute) \
                    and n.iter.func.attr == "items" and isinstance(n.target, ast.Tuple):
                kname = n.target.elts[0].id if isinstance(n.target.elts[0], ast.Name) else None
                body = "\n".join(ast.unparse(s) for s in n.body)
                if kname and re.search(rf"/\s*{kname}\b|join\([^)]*\b{kname}\b", body) and re.search(r"write", body) \
                        and not re.search(rf"if\s+.*\b{kname}\b\s+(not\s+)?in\b|\bassert\b", body):
                    out.append(ev(pf, n.lineno))
    meta = [p for p in pkg.files if "/.meta/" in p]
    if not out:
        return []
    msg = "Model named files are written into a copied task folder before tests run"
    if meta or re.search(r"\.meta\b", pkg.all_py_text()):
        msg += " (the folder layout includes .meta reference solutions)"
    return [F("model-named-files-into-test-dir", msg + "; overwriting the test file passes.", out)]


ANSWER_KEYS = re.compile(r"^(answer|answers|solution|solutions|gold|gold_answer|ground_truth|expected|expected_output|"
                         r"expected_answer|correct_answer|label|target|reference|ref_answer|final_answer)$")


# rubric-shown-to-policy: reader items, main paper arxiv:2505.05410
RUBRIC_KEYS = re.compile(r"^(rubric|rubrics|grading_rubric|rubric_items|criteria|grading_criteria|evaluation_criteria|"
                         r"scoring_criteria|scoring_rubric|grading_notes|grading_guidelines|checklist|answer_hint|"
                         r"hint|hints|solution_hint|hidden_tests|private_tests|judge_notes|grader_notes)$")
JUDGE_TEXT = re.compile(r"(judge|grader|grade the|evaluate the|assess the|score the|you are (an|a) (expert )?(evaluator|grader|judge)|"
                        r"rate the|evaluation)", re.I)


@rule("answer-in-prompt", "hack", "high",
      "The prompt/question text is built from the answer / solution field of the same row.")
def r_answer_in_prompt(pkg: Package):
    out = []
    rub = []
    judge_funcs = None
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if not isinstance(n, ast.Dict):
                continue
            for k, v in zip(n.keys, n.values):
                if not (isinstance(k, ast.Constant) and k.value in ("prompt", "question", "content")):
                    continue
                if not isinstance(v, (ast.JoinedStr, ast.BinOp, ast.Call)):
                    continue
                for sub in ast.walk(v):
                    key = None
                    if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
                        key = sub.slice.value
                    elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get" \
                            and sub.args and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str):
                        key = sub.args[0].value
                    if key and ANSWER_KEYS.match(key):
                        # role message dict with assistant content is a few shot example, not a leak
                        roles = [vv.value for kk, vv in zip(n.keys, n.values)
                                 if isinstance(kk, ast.Constant) and kk.value == "role" and isinstance(vv, ast.Constant)]
                        if "assistant" in roles:
                            continue
                        out.append(ev(pf, sub.lineno))
                        break
                    if key and RUBRIC_KEYS.match(key) and not out:
                        roles = [vv.value for kk, vv in zip(n.keys, n.values)
                                 if isinstance(kk, ast.Constant) and kk.value == "role" and isinstance(vv, ast.Constant)]
                        if "assistant" in roles or JUDGE_TEXT.search(_const_text_safe(v)):
                            continue
                        if judge_funcs is None:
                            judge_funcs = [(p.path, f.start, f.end) for p, f in _reward_path_cached(pkg)]
                            judge_funcs += [(p.path, f.start, f.end) for p, f in all_funcs(pkg)
                                            if re.search(r"(judge|grade|grader|evaluat|critic|score)", f.name, re.I)]
                        if any(pth == pf.path and s <= sub.lineno <= e for pth, s, e in judge_funcs):
                            continue
                        rub.append((ev(pf, sub.lineno), key))
    if out:
        return [F("answer-in-prompt", "A prompt or question string interpolates the row's answer field.", out,
                  confidence="review")]
    if rub:
        keys = sorted({k for _, k in rub})
        return [F("answer-in-prompt", f"The policy prompt includes the row's {', '.join(keys[:3])} field, which tells the "
                  "model what the grader looks for.", [e for e, _ in rub], confidence="review", severity="medium")]
    return []


# answer-bearing-files-by-name: reader items, main papers arxiv:2605.15229, arxiv:2604.11806
BACKUP_EXT = re.compile(r"\.(orig|bak|rej)$|~$")
CONTEXT_FILE = re.compile(r"(^|/)(AGENTS\.md|CLAUDE\.md|GEMINI\.md|\.cursorrules|README(\.md)?|NOTES\.md|HINTS?\.md)$", re.I)


def _dockerfile_text(pkg: Package, root: str) -> str:
    return "\n".join(t for p, rel, t in _task_files(pkg, root, "environment/")
                     if re.search(r"(^|/)(Dockerfile|Containerfile)[^/]*$", rel) or rel.endswith(".dockerfile"))


def _in_image(df: str, rel_env: str) -> bool:
    """Is environment/<rel_env> copied into the image by the Dockerfile?"""
    base = os.path.basename(rel_env)
    for m in re.finditer(r"^\s*(?:COPY|ADD)\s+(?:--\S+\s+)*(.+)$", df, re.M | re.I):
        parts = m.group(1).split()
        if len(parts) < 2:
            continue
        for src in parts[:-1]:
            s = src.strip("\"'[],").rstrip("/")
            if s in (".", "./", "*") or s == rel_env or os.path.basename(s) == base or \
                    (s and rel_env.startswith(s.lstrip("./") + "/")):
                return True
    return False


def _removed_later(df: str, rel_env: str) -> bool:
    base = os.path.basename(rel_env)
    ext = os.path.splitext(base)[1]
    return bool(re.search(rf"(\brm\b[^\n]*({re.escape(base)}|\*{re.escape(ext)})|-name\s+['\"]?\*{re.escape(ext)}['\"]?[^\n]*-delete)",
                          df)) if ext else False


def _harbor_backup_hits(pkg: Package):
    out = []
    for root in _harbor_roots(pkg):
        df = _dockerfile_text(pkg, root)
        sol = _solution_added_lines(pkg, root)
        if not df:
            continue
        for p in list(pkg.sizes):
            rel = _task_rel(root, p)
            if rel is None or not rel.startswith("environment/"):
                continue
            rel_env = rel[len("environment/"):]
            if not _in_image(df, rel_env) or _removed_later(df, rel_env):
                continue
            if BACKUP_EXT.search(rel_env):
                out.append((ev(p, 1), "backup file " + os.path.basename(rel_env)))
                continue
            t = pkg.files.get(p, "")
            if not sol or not t:
                continue
            if rel_env.endswith((".patch", ".diff")) and _looks_diff(t):
                a, r = _diff_lines(t)
                hit = (a | r) & sol
            elif CONTEXT_FILE.search(rel_env):
                hit = {c for c, _ in _candidate_lines(t)} & sol
            else:
                continue
            hit = {x for x in hit if _statement_code(x)}
            if len(hit) >= 3:
                out.append((ev(p, 1), f"{os.path.basename(rel_env)} holds {len(hit)} lines of the reference fix"))
        for m in re.finditer(r"^\s*RUN\b[^\n]*\bpatch\b[^\n]*\s(-b|--backup|-V\s*\w+)\b", df, re.M):
            out.append((f"{root + '/' if root else ''}environment/Dockerfile:{line_of(df, m.start())}",
                        "patch run with backups"))
    return out


@rule("reference-shipped-in-workspace", "hack", "high",
      "Reference solutions / expected outputs ship inside the directory the agent works in "
      "(.meta dirs, solution files uploaded to the sandbox).")
def r_reference_shipped(pkg: Package):
    out = []
    for pf, fi in all_funcs(pkg):
        if SCORING_NAME.search(fi.name) or re.search(r"(validat|gold|oracle|dummy|sanity|smoke|solve|reference_run)", fi.name, re.I):
            continue  # scoring time uploads are covered by other rules; gold paths are validation only
        src = fi.src
        for m in re.finditer(r"\b(?:sandbox\w*|runtime|container|sb)\s*\.\s*(?:upload\w*|write\w*|put\w*|copy\w*)\s*\(([^\n]*)", src):
            arg = m.group(1)
            if re.search(r"(solution|reference|gold|expected_output|answer_key|oracle|\.meta)", arg, re.I) and \
                    not re.search(r"(model|agent|completion|response)_?(solution|answer)", arg, re.I):
                out.append(ev(pf, fi.start + line_of(src, m.start()) - 1))
    if out:
        return [F("reference-shipped-in-workspace",
                  "A reference/solution/expected file is uploaded into the agent's sandbox during setup.", out,
                  confidence="review")]
    hb = _harbor_backup_hits(pkg)
    if hb:
        what = sorted({w for _, w in hb})
        return [F("reference-shipped-in-workspace", "The agent image gets a file that can hold the reference ("
                  + "; ".join(what[:3]) + ").", [e for e, _ in hb], confidence="review")]
    return []


# fetch-tool-blocklist-gaps: reader items, main paper arxiv:2610.16998
SEARCH_API = re.compile(r"(exa_py|\bExa\(|exa\.search|tavily|TavilyClient|serper|serpapi|GoogleSearch|duckduckgo|\bDDGS\b|ddgs|"
                        r"brave|bing\.microsoft|api\.bing|googleapis\.com/customsearch|search_engine|searxng|you\.com|jina\.ai|"
                        r"r\.jina|s\.jina|perplexity|firecrawl|google_search|web_search\()", re.I)
HF_BLOCK = re.compile(r"(huggingface|hf\.co)", re.I)
ALLOW_WORD = re.compile(r"(allow\w*|whitelist|permitted|only_domains|include_domains|includeDomains|domain_filter)", re.I)
BLOCK_WORD = re.compile(r"(block\w*|deny\w*|blacklist|banned|forbid\w*|exclude_domains|excludeDomains|exclude_sites|"
                        r"excluded_domains|-site:|disallow\w*)", re.I)


def _fetch_filter_text(pkg: Package, pf: PyFile, fi: FuncInfo) -> str:
    txt = fi.src + "\n" + referenced_consts(pf, fi)
    # one hop into helpers the tool calls
    names = called_names(fi)
    for pf2, fi2 in all_funcs(pkg):
        if fi2.name in names and fi2 is not fi:
            txt += "\n" + fi2.src + "\n" + referenced_consts(pf2, fi2)
    return txt


def _switch_of(test):
    """(variable name, literal) for a test like `x == "exa"`, `self.x == "exa"` or `x.lower() == "exa"`."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        a, b = test.left, test.comparators[0]
        if isinstance(a, ast.Constant):
            a, b = b, a
        if isinstance(b, ast.Constant) and isinstance(b.value, str):
            while isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute):
                a = a.func.value
            if isinstance(a, ast.Name):
                return a.id, b.value
            if isinstance(a, ast.Attribute):
                return a.attr, b.value
    return None


def _param_defaults(pkg: Package) -> dict[str, set]:
    c = _v05_cache(pkg)
    if "param_defaults" in c:
        return c["param_defaults"]
    out: dict[str, set] = {}
    for pf, fi in all_funcs(pkg):
        a = fi.node.args
        pos = a.posonlyargs + a.args
        for arg, d in list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + \
                [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]:
            if isinstance(d, ast.Constant):
                out.setdefault(arg.arg, set()).add(d.value)
    c["param_defaults"] = out
    return out


def _opt_in_web_tool(pkg: Package, pf: PyFile, fi: FuncInfo) -> bool:
    """True when the open web tool or its web backend is picked only by a non-default argument value, such as
    retriever="exa" next to a local default, and no shipped config sets that value (0.5.0 fix wave, check_3.md)."""
    switches = []
    for q in pkg.py_all():
        if q.tree is None:
            continue
        for n in ast.walk(q.tree):
            if not isinstance(n, (ast.If, ast.IfExp)):
                continue
            sw = _switch_of(n.test)
            if not sw:
                continue
            body = n.body if isinstance(n, ast.If) else [n.body]
            btxt = "\n".join(ast.unparse(x) for x in body)
            uses_tool = re.search(rf"\b{re.escape(fi.name)}\b", btxt)
            other = n.orelse if isinstance(n, ast.If) else [n.orelse]
            if isinstance(n, ast.If) and not other:
                # no else: the statements after the if are the default path
                for blk in ast.walk(q.tree):
                    for field in ("body", "orelse", "finalbody"):
                        seq = getattr(blk, field, None)
                        if isinstance(seq, list) and n in seq:
                            other = seq[seq.index(n) + 1:]
            otxt = "\n".join(ast.unparse(x) for x in other)
            uses_api = q.path == pf.path and fi.start <= n.lineno <= fi.start + fi.src.count("\n") and \
                SEARCH_API.search(btxt) and not SEARCH_API.search(otxt)
            if uses_tool or uses_api:
                switches.append(sw)
    if not switches:
        return False
    defaults = _param_defaults(pkg)
    cfg = "\n".join(t for p, t in pkg.files.items() if p.endswith((".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"))
                    and not p.endswith(("pyproject.toml", "package.json")))
    for var, lit in switches:
        d = defaults.get(var)
        if not d or lit in d:
            return False  # on the default path, or no default to compare with
        if re.search(rf"\b{re.escape(var)}\b['\"]?\s*[=:]\s*['\"]{re.escape(lit)}['\"]", cfg):
            return False  # a shipped config turns it on
    return True


@rule("open-fetch-tool", "hack", "medium",
      "An agent tool fetches arbitrary URLs with no domain filter while the dataset (answer key) is "
      "public on the HF hub.")
def r_fetch_tool(pkg: Package):
    text = pkg.all_py_text()
    alltext = "\n".join(t for p, t in pkg.files.items() if not p.endswith((".jsonl", ".json")))
    ds = re.findall(r"load_dataset\(\s*['\"]([\w.-]+/[\w.-]+)['\"]", text) + \
        re.findall(r"huggingface\.co/datasets/([\w.-]+/[\w.-]+)", alltext) + \
        re.findall(r"(?:DATASET\w*|dataset_name|dataset_id|repo_id)\s*[:=][^=\n]*?['\"]([\w.-]+/[\w.-]+)['\"]", text)
    if not ds:
        return []
    out = []
    for pf, fi in all_funcs(pkg):
        if not re.search(r"(fetch|browse|open_url|visit|get_url|read_url|web|http|download|scrape|open_page|read_page)", fi.name, re.I):
            continue
        if "url" not in " ".join(fi.params).lower():
            continue
        if re.search(r"(requests\.get|httpx\.\w*get|urlopen|aiohttp|session\.get|client\.get|AsyncClient|to_thread\(\s*fetch)", fi.src) and \
                not re.search(r"(allow\w*|block\w*|deny|domain|netloc|huggingface|hf\.co|blacklist|whitelist|banned|forbid)", fi.src, re.I):
            out.append(ev(pf, fi.start))
    if out:
        return [F("open-fetch-tool", f"Unfiltered URL fetch tool; the dataset {ds[0]} is public, so the policy "
                  "can download the answer key.", out, confidence="review")]
    # v0.5: model tools whose filter is a blocklist that leaves the HF hub open, and search tools with no filter
    tools = _tool_names(pkg)
    ds_names = {d.split("/")[-1].lower() for d in ds}
    gaps, kinds = [], set()
    for pf, fi in all_funcs(pkg):
        if fi.name not in tools:
            continue
        is_fetch = re.search(r"(fetch|browse|open_url|visit|get_url|read_url|web|http|download|scrape|open_page|read_page|crawl)",
                             fi.name, re.I) and re.search(r"url", " ".join(fi.params), re.I)
        is_search = re.search(r"search", fi.name, re.I) and re.search(r"(query|q\b|queries|search|term|keyword)", " ".join(fi.params), re.I)
        if not (is_fetch or is_search):
            continue
        ftxt = _fetch_filter_text(pkg, pf, fi)
        if is_fetch and not re.search(r"(requests\.|httpx|urlopen|aiohttp|session\.get|client\.get|AsyncClient|"
                                      r"playwright|page\.goto|crawl|jina|firecrawl|trafilatura)", ftxt):
            continue
        if is_search and not is_fetch and not SEARCH_API.search(ftxt):
            continue
        if ALLOW_WORD.search(ftxt):
            continue
        if HF_BLOCK.search(ftxt) or any(len(d) > 4 and d in ftxt.lower() for d in ds_names):
            continue
        if _opt_in_web_tool(pkg, pf, fi):
            continue
        gaps.append(ev(pf, fi.start))
        kinds.add("a blocklist that leaves huggingface.co open" if BLOCK_WORD.search(ftxt) else
                  ("no URL filter on search results" if is_search and not is_fetch else "no URL filter"))
    if not gaps:
        return []
    prompt_only = bool(re.search(r"(do not|don't|never|must not)\s+(visit|access|use|search|browse)[^\n]{0,80}(huggingface|dataset|benchmark)",
                                 alltext, re.I))
    return [F("open-fetch-tool", f"A model web tool has {' and '.join(sorted(kinds))}"
              + (" and the ban is only in the prompt" if prompt_only else "")
              + f", while the dataset {ds[0]} is public, so search can surface the answer key.", gaps, confidence="review")]


# feedback-reveals-internals: reader items, main paper arxiv:2609.28614
GOLD_NAMES = {"answer", "answers", "gold", "golden", "gold_answer", "ground_truth", "correct_answer", "expected_answer",
              "expected_output", "solution", "reference", "target", "hidden_tests", "test_cases", "oracle", "label",
              "correct", "right_answer", "true_answer", "expected", "expected_value", "gold_value"}
REVEAL_WORDS = re.compile(r"(correct answer|right answer|the answer (is|was)|answer was|answer is|solution (is|was)|"
                          r"should (be|have been)|expected|the target|was looking for|correct (value|label|output) (is|was))")
DONE_MARK = re.compile(r"(\[['\"](done|completed|is_completed|finished|terminated|game_over)['\"]\]\s*=\s*True|"
                       r"\b(done|is_done|finished|game_over|terminated)\s*=\s*True)")


def _feedback_reveal_hits(pf: PyFile, fi: FuncInfo):
    parents = {}
    for p in ast.walk(fi.node):
        for ch in ast.iter_child_nodes(p):
            parents[ch] = p
    hits = []
    for n in ast.walk(fi.node):
        if not isinstance(n, ast.JoinedStr):
            continue
        lits = "".join(v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str)).lower()
        if not REVEAL_WORDS.search(lits):
            continue
        names = {x.id for v in n.values if isinstance(v, ast.FormattedValue) for x in ast.walk(v) if isinstance(x, ast.Name)}
        names |= {x.attr for v in n.values if isinstance(v, ast.FormattedValue) for x in ast.walk(v) if isinstance(x, ast.Attribute)}
        names |= {x.value for v in n.values if isinstance(v, ast.FormattedValue) for x in ast.walk(v)
                  if isinstance(x, ast.Constant) and isinstance(x.value, str)}
        if not (names & GOLD_NAMES):
            continue
        # the episode ends right here: the model gets no turn to use what it was told
        blk = parents.get(n)
        while blk is not None and not isinstance(blk, (ast.If, ast.For, ast.While, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef)):
            blk = parents.get(blk)
        if blk is not None and DONE_MARK.search(_unparse(blk) if not isinstance(blk, (ast.FunctionDef, ast.AsyncFunctionDef))
                                                else ""):
            continue
        hits.append(ev(pf, n.lineno))
    return hits


@rule("verifier-feedback-leak", "hack", "high",
      "Environment feedback shown to the model echoes the expected value (\"expected X, got Y\").")
def r_feedback_leak(pkg: Package):
    out = []
    wide = []
    for pf, fi in all_funcs(pkg):
        if not re.search(r"(env_response|step|tool|feedback|observation|respond|call_tool|execute)", fi.name, re.I):
            continue
        if SCORING_NAME.search(fi.name) and not re.search(r"env_response|feedback", fi.name):
            continue
        for n in ast.walk(fi.node):
            if isinstance(n, ast.JoinedStr):
                lits = "".join(v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str)).lower()
                if "expected" in lits and re.search(r"(got|found|but|actual|received|instead)", lits):
                    names = {x.id for v in n.values if isinstance(v, ast.FormattedValue) for x in ast.walk(v) if isinstance(x, ast.Name)}
                    names |= {x.attr for v in n.values if isinstance(v, ast.FormattedValue) for x in ast.walk(v) if isinstance(x, ast.Attribute)}
                    if names & (ANSWERISH | {"expected_value", "exp", "want", "gold_value"}):
                        out.append(ev(pf, n.lineno))
        if not out and re.search(r"(env_response|feedback|step)", fi.name, re.I):
            wide += _feedback_reveal_hits(pf, fi)
    if out:
        return [F("verifier-feedback-leak", "A message returned to the model contains the expected value; pasting it "
                  "back can reach full reward.", out, confidence="review")]
    if wide:
        return [F("verifier-feedback-leak", "Feedback sent to the model during the episode states the gold answer, so the "
                  "next turn can repeat it.", wide, confidence="review")]
    return []


# ---------------------------------------------------------------------------
# 5. code execution of model text in the grader process
# ---------------------------------------------------------------------------

MODEL_TEXTISH = re.compile(r"(completion|response|answer|parsed|code|text|content|output|func_call|call|expr|"
                           r"expression|action|tool_call|args|arguments|program|solution|submission|strategy|"
                           r"generated|model|kernel|query|cmd|command)", re.I)


@rule("eval-model-text", "hack", "critical",
      "eval()/exec() of model controlled text in the trainer/grader process (not a sandbox).")
def r_eval(pkg: Package):
    likely, review = [], []
    scoring = {(pf.path, fi.start) for pf, fi in reward_path_funcs(pkg)}
    for pf, fi in all_funcs(pkg):
        if re.search(r"(filter|config|cfg|parse_args|resolve_\w*fn|_load_|settings)", fi.name, re.I):
            continue
        if pf.text.lstrip().startswith("# /// script"):
            continue  # a uv script shipped into the sandbox, not grader process code
        on_path = (pf.path, fi.start) in scoring
        for n in ast.walk(fi.node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("eval", "exec") and n.args:
                a = n.args[0]
                if isinstance(a, ast.Constant):
                    continue
                try:
                    at = ast.unparse(a)
                except Exception:
                    at = ""
                if not MODEL_TEXTISH.search(at):
                    continue
                restricted = len(n.args) > 1 and "__builtins__" in ast.unparse(n.args[1]) and \
                    re.search(r"__builtins__['\"]?\s*:\s*(\{\}|None)", ast.unparse(n.args[1]))
                where = ev(pf, n.lineno)
                if re.search(r"(filter|config|cfg|setting|option|kwarg)", at, re.I) and not re.search(r"(completion|response|answer)", at):
                    continue
                if (on_path or re.search(r"(checker|grade|verify|reward)", pf.path, re.I)) and not restricted:
                    likely.append(where)
                else:
                    review.append(where)
    res = []
    if likely:
        res.append(F("eval-model-text", "Model controlled text reaches eval/exec in the grader process on the scoring "
                     "path; it can patch the scorer (numpy, the checker) or run code on the training host.", likely))
    elif review:
        res.append(F("eval-model-text", "eval/exec on text that may be model controlled, outside a sandbox.",
                     review, confidence="review", severity="high"))
    return res




# ---------------------------------------------------------------------------
# 6. rubric wiring that fails silently
# ---------------------------------------------------------------------------

@rule("judge-not-awaited", "broken", "high",
      "An async JudgeRubric.judge(...) is called without await, so the reward compares a coroutine.")
def r_judge_not_awaited(pkg: Package):
    out = []
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        parents = {}
        for p in ast.walk(pf.tree):
            for ch in ast.iter_child_nodes(p):
                parents[ch] = p
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "judge":
                recv = ast.unparse(n.func.value)
                if not re.search(r"(judge|jr|rubric)", recv, re.I):
                    continue
                par = parents.get(n)
                wrapped = False
                cur = n
                while par is not None and not isinstance(par, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                    if isinstance(par, ast.Await):
                        wrapped = True
                        break
                    if isinstance(par, ast.Call):
                        fn = ast.unparse(par.func)
                        if re.search(r"(asyncio\.run|run_until_complete|maybe_await|run_sync|gather|ensure_future|create_task|run_coroutine)", fn):
                            wrapped = True
                            break
                    cur, par = par, parents.get(par)
                if wrapped:
                    continue
                # assigned then awaited later?  x = rubric.judge(...); await x
                if isinstance(par, (ast.FunctionDef, ast.AsyncFunctionDef)) or par is None:
                    pass
                asg = parents.get(n)
                if isinstance(asg, ast.Assign) and isinstance(asg.targets[0], ast.Name):
                    var = asg.targets[0].id
                    fn_node = asg
                    while fn_node is not None and not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn_node = parents.get(fn_node)
                    if fn_node is not None and re.search(rf"await\s+{var}\b|asyncio\.\w+\([^)]*\b{var}\b|maybe_await\(\s*{var}", ast.unparse(fn_node)):
                        continue
                out.append(ev(pf, n.lineno))
    if not out:
        return []
    return [F("judge-not-awaited", "JudgeRubric.judge is async but its result is used without await; the judge "
              "never runs and the reward is constant (usually 0).", out)]


@rule("rubric-funcs-ignored", "broken", "high",
      "Reward functions are handed over in a way verifiers ignores: rewards= kwarg on an env, "
      "assigning self.reward_funcs/self.weights in a Rubric subclass, or an empty rubric.")
def r_rubric_ignored(pkg: Package):
    out = []
    msgs = []
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call):
                fname = n.func.attr if isinstance(n.func, ast.Attribute) else (n.func.id if isinstance(n.func, ast.Name) else "")
                kws = {k.arg for k in n.keywords if k.arg}
                if fname.endswith("Env") and ("rewards" in kws or "reward_funcs" in kws or "reward_functions" in kws) \
                        and "rubric" not in kws and "funcs" not in kws:
                    out.append(ev(pf, n.lineno)); msgs.append("env built with rewards=, no rubric")
                if fname.endswith("Rubric") and ("rewards" in kws or "reward_functions" in kws) and "funcs" not in kws:
                    out.append(ev(pf, n.lineno)); msgs.append("Rubric(rewards=...)")
            if isinstance(n, ast.ClassDef) and any(re.search(r"Rubric", ast.unparse(b)) for b in n.bases):
                src = ast.unparse(n)
                if re.search(r"self\.(reward_funcs|weights|reward_weights)\s*=", src) and \
                        not re.search(r"add_reward_func|super\(\)\.__init__\([^)]*funcs\s*=|self\.funcs\s*=", src):
                    out.append(ev(pf, n.lineno)); msgs.append(f"{n.name} assigns self.reward_funcs/weights")
    # empty rubric: vf.Rubric() with no funcs and never extended, passed to an env
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                c = n.value
                fname = c.func.attr if isinstance(c.func, ast.Attribute) else (c.func.id if isinstance(c.func, ast.Name) else "")
                if fname in ("Rubric",) and not c.args and not c.keywords:
                    var = n.targets[0].id
                    if not re.search(rf"\b{var}\.(add_reward_func|add_metric|funcs|reward_funcs)\b|RubricGroup\(", pf.text):
                        out.append(ev(pf, n.lineno)); msgs.append("empty vf.Rubric()")
    if not out:
        return []
    return [F("rubric-funcs-ignored", "Rubric has no effective reward functions (" + "; ".join(sorted(set(msgs))[:3]) +
              "); every rollout gets the same score.", out)]


@rule("rubric-not-a-rubric", "broken", "high",
      "A class named *Rubric that does not subclass a verifiers Rubric is passed as rubric=.")
def r_rubric_not_vf(pkg: Package):
    classes = {}
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.ClassDef) and n.name.endswith("Rubric"):
                bases = [ast.unparse(b) for b in n.bases]
                body = ast.unparse(n)
                if (not bases or bases == ["object"]) and not re.search(
                        r"(def score_group|def __getattr__|self\.inner|self\._inner|self\.rubric\b|self\._rubric\b|wrap)", body):
                    classes[n.name] = ev(pf, n.lineno)
    if not classes:
        return []
    text = pkg.all_py_text()
    used = [c for c in classes if re.search(rf"rubric\s*=\s*{c}\(", text)]
    if not used:
        return []
    return [F("rubric-not-a-rubric", f"{used[0]} is not a vf.Rubric subclass; the env cannot score on verifiers.",
              [classes[c] for c in used])]


@rule("prompt-keyed-lookup", "broken", "high",
      "Per task data is looked up by prompt[0]['content'], but verifiers prepends the system prompt, so "
      "the lookup misses and a fallback reward is paid.")
def r_prompt_lookup(pkg: Package):
    out, likely = [], False
    for pf, fi in all_funcs(pkg):
        m = re.search(r"prompt\w*\s*\[\s*0\s*\]\s*(?:\.get\(\s*|\[\s*)['\"]content['\"]", fi.src)
        if not m:
            continue
        rest = fi.src[m.end():]
        if re.search(r"\.get\(\s*\w*(content|prompt|key)\w*\s*[,)]|\[\s*\w*(content|prompt)\w*\s*\]", rest):
            out.append(ev(pf, fi.start + line_of(fi.src, m.start()) - 1))
            if re.search(r"if\s+not\s+\w+\s*:[^\n]*\n(\s*#[^\n]*\n)*\s*return\s+(0\.[1-9]|[1-9])", rest):
                likely = True
    if not out:
        return []
    return [F("prompt-keyed-lookup", "Task lookup keys on the first prompt message; with a system prompt that is "
              "the system text, so the lookup misses" + (" and a positive fallback reward is paid." if likely else "."),
              out, confidence="likely" if likely else "review")]


# ---------------------------------------------------------------------------
# 7. judges (reported, not counted: wave 1 rated these low)
# ---------------------------------------------------------------------------

@rule("judge-prompt-undelimited", "hygiene", "low",
      "LLM judge template interpolates the raw response with no delimiters (prompt injection surface).")
def r_judge_undelimited(pkg: Package):
    out = []
    for pf in pkg.py_all():
        for name, t in pf.consts.items():
            if not re.search(r"(judge|grader|grade|evaluat|rubric)", name + t[:300], re.I):
                continue
            for m in re.finditer(r"\{(response|completion|model_answer|answer|output|prediction|student_answer|generated)\}", t):
                around = t[max(0, m.start() - 40): m.end() + 40]
                if not re.search(r"(<\w+>|```|\"\"\"|\[\[|---|###|<<<|===)", around):
                    out.append(f"{pf.path}:{name}")
                    break
    if not out:
        return []
    return [F("judge-prompt-undelimited", "Judge prompt interpolates the response without delimiters.", out,
              confidence="review")]


# answer-extraction-weak-forms: reader items, main papers arxiv:2608.24258, arxiv:2609.01354
FIRST_FENCE = re.compile(r"(re\.(search|match)\(\s*r?['\"][^'\"\n]*```[^'\"\n]*['\"]|\.split\(\s*['\"]```\w*['\"]\s*\)\s*\[\s*1\s*\]|"
                         r"findall\(\s*r?['\"][^'\"\n]*```[^\n]*\)\s*\[\s*0\s*\])")
SUBSTR_CORRECT = re.compile(r"['\"]correct['\"]\s+in\s+([\w.\[\]'\"()]+)")
TIMEOUT_GUARD = re.compile(r"(timeout|signal\.alarm|func_timeout|ProcessPool|multiprocessing|wait_for|timeout_decorator|"
                           r"stopit|Pebble|pebble|with_timeout|time_limit)", re.I)


@rule("first-match-label-regex", "hygiene", "low",
      "Verdict/letter extraction takes the first regex match, so a hedged answer that lists options can "
      "grade a ruled out choice.")
def r_first_match(pkg: Package):
    out = []
    sf = scoring_funcs(pkg)
    for pf, fi in sf:
        for m in re.finditer(r"re\.search\(\s*r?['\"][^'\"]*(\(\[A-[D-Z]\]\)|\[A-[D-Z]\]|\(yes\|no\)|\(YES\|NO\)|\(correct\|incorrect\)|\(A\|B\|C\|D\))", fi.src):
            out.append(ev(pf, fi.start + line_of(fi.src, m.start()) - 1))
    if out:
        return [F("first-match-label-regex", "Label/letter is taken from the first regex match.", out, confidence="review")]
    # v0.5 weak extraction forms. The "correct" in reply form moved to judge-verdict-substring-match
    # in the 0.5.0 fix wave. When it is present this rule stays quiet, as before.
    if _verdict_substring_hits(pkg, sf):
        return []
    res = []
    fence_hits, sym_hits = [], []
    for pf, fi in sf:
        src = fi.src
        for m in FIRST_FENCE.finditer(src):
            ln = src[m.start():src.find("\n", m.end()) if src.find("\n", m.end()) > 0 else len(src)]
            if re.search(r"(completion|response|reply|output|content|text|message|answer|generated|pred)", ln[m.end() - m.start() - 1:], re.I) \
                    and not re.search(r"(self\.data|task|spec|template|prompt|instruction|readme)", ln, re.I):
                fence_hits.append(ev(pf, fi.start + line_of(src, m.start()) - 1))
        if re.search(r"\b(simplify|sympify|nsimplify|\.equals)\(", src) and re.search(r"(==\s*0|\.equals\(|simplify\()", src) \
                and re.search(r"^\s*(import sympy|from sympy)", pf.text, re.M) and not re.search(r"^\s*def (simplify|sympify)\(", pf.text, re.M) \
                and not TIMEOUT_GUARD.search(src) and not TIMEOUT_GUARD.search(pf.text):
            m = re.search(r"\b(simplify|sympify|nsimplify|\.equals)\(", src)
            sym_hits.append(ev(pf, fi.start + line_of(src, m.start()) - 1))
    if fence_hits:
        res.append(F("first-match-label-regex", "Only the first fenced code block of the reply is graded, so later blocks "
                     "are never checked.", fence_hits, confidence="review", variant="first-fenced-block"))
    elif sym_hits:
        res.append(F("first-match-label-regex", "A sympy compare runs on model text with no timeout, so a huge expression "
                     "can hang the grader.", sym_hits, confidence="review"))
    return res


def _verdict_substring_hits(pkg: Package, sf=None):
    """(evidence, function name, variable) for each `"correct" in reply` verdict read in scoring code."""
    sf = scoring_funcs(pkg) if sf is None else sf
    hits = []
    for pf, fi in sf:
        src = fi.src
        for m in SUBSTR_CORRECT.finditer(src):
            before = src[:m.start()]
            if re.search(r"['\"](incorrect|not correct|wrong)['\"]", before):
                continue  # the negative label is checked first
            judge_ctx = re.search(r"(judge|verdict|grade|grading|eval)", src + referenced_consts(pf, fi), re.I)
            if judge_ctx and re.search(r"incorrect", src + referenced_consts(pf, fi) + pf.text, re.I):
                hits.append((ev(pf, fi.start + line_of(src, m.start()) - 1), fi.name, m.group(1)))
    return hits


# judge-verdict-substring-match: split out of first-match-label-regex in the 0.5.0 fix wave
# (the internal notes: both hits paid wrong answers in full). Canon item: weak verdict
# extraction, main paper inspect_evals BEST_PRACTICES (unanchored verdict parse).
@rule("judge-verdict-substring-match", "hack", "high",
      "The judge verdict is read with a substring test for a word that its negative label also contains, so a "
      "negative verdict is paid as a pass.")
def r_verdict_substring(pkg: Package):
    sf = scoring_funcs(pkg)
    hits = _verdict_substring_hits(pkg, sf)
    if not hits:
        return []
    reg = _reg_cached(pkg)
    paid_names = {n for n, w in reg.items() if w > 0}
    paid_src = "\n".join(fi.src for _pf, fi in sf if fi.name in paid_names)
    paid = any(nm in paid_names or re.search(rf"\b{re.escape(nm)}\(", paid_src) for _, nm, _ in hits)
    var = hits[0][2]
    msg = (f"The judge verdict is read with \"correct\" in {var}, and the opposite word \"incorrect\" contains "
           "\"correct\", so the judge's negative verdict is paid as a pass.")
    return [F("judge-verdict-substring-match", msg, [e for e, _, _ in hits], confidence="review",
              kind=None if paid else "hygiene", severity=None if paid else "low")]


# ---------------------------------------------------------------------------
# 8. load breakers
# ---------------------------------------------------------------------------

# (pattern over import lines / attribute use, removed_in version, description)
REMOVED = [
    (r"from\s+verifiers\.utils\.message_utils\s+import\s+[^\n]*\bChatCompletionMessage\b", "0.1.9",
     "verifiers.utils.message_utils.ChatCompletionMessage (only <=0.1.8.post1)"),
    (r"verifiers\.envs\.experimental\b", "0.3.1", "verifiers.envs.experimental.*"),
    (r"\bvf\.SandboxEnv\b|from\s+verifiers\.envs\.sandbox_env\s+import|from\s+verifiers\s+import\s+[^\n]*\bSandboxEnv\b",
     "0.3.1", "vf.SandboxEnv / verifiers.envs.sandbox_env"),
    (r"verifiers\.v1\.envs\.isolated_verifier", "0.4.1", "verifiers.v1.envs.isolated_verifier"),
    (r"from\s+verifiers\.v1\.tasksets\.harbor[\w.]*\s+import\s+[^\n]*\bHarborEnv(Config)?\b|\bvf\.HarborEnv\b|"
     r"from\s+verifiers\.v1\.tasksets\.harbor\s+import\s+\([^)]*\bHarborEnv\b", "0.4.1",
     "HarborEnv / HarborEnvConfig from verifiers.v1.tasksets.harbor"),
    (r"\bRemoteHybridMathRubric\b|verifiers\.rubrics\.experimental", "0.3.1", "verifiers.rubrics.experimental"),
]
LEGACY = re.compile(r"(^\s*from\s+verifiers\.(envs|rubrics|parsers|types|utils)\b|^\s*import\s+verifiers\.(envs|rubrics|parsers|types|utils)\b|"
                    r"\bvf\.(SingleTurnEnv|MultiTurnEnv|ToolEnv|StatefulToolEnv|Rubric|JudgeRubric|XMLParser|Parser|ThinkParser|MathRubric|RubricGroup)\b)", re.M)


def _vt(s: str):
    """Loose version tuple: 0.1.8.post1 -> (0,1,8,1); 0.3.2.dev131 -> (0,3,2,-1)."""
    out = []
    for part in re.split(r"[.\-]", s):
        m = re.match(r"(\d+)", part)
        if m:
            out.append(int(m.group(1)))
        elif part.startswith(("dev", "rc", "a", "b")):
            out.append(-1)
    while len(out) < 4:
        out.append(0)
    return tuple(out[:5])


def parse_spec(spec: str | None):
    """-> (lower, upper_exclusive, pinned) as version tuples or None."""
    if spec is None:
        return None, None, None
    if spec.strip().startswith("@") or "git+" in spec or "://" in spec:
        return None, None, (99, 0, 0, 0)  # direct reference: version unknown, treat as pinned
    lower = upper = pinned = None
    for c in re.split(r",", spec.replace(" ", "")):
        m = re.match(r"(>=|<=|==|~=|>|<|!=)?([\w.\-*]+)", c)
        if not m or not m.group(2) or not re.match(r"\d", m.group(2)):
            continue
        op, v = m.group(1) or "==", _vt(m.group(2).replace("*", "0"))
        if op in (">=", ">"):
            lower = v
        elif op in ("<", "<="):
            upper = v
        elif op == "==":
            pinned = v
        elif op == "~=":
            lower = v
            upper = (v[0], v[1] + 1, 0, 0) if len(v) > 1 else None
    return lower, upper, pinned


@rule("removed-verifiers-symbol", "broken", "high",
      "Imports a verifiers symbol that was removed, and the verifiers pin lets a fresh install resolve to a "
      "release without it.")
def r_removed(pkg: Package):
    spec = pkg.verifiers_requirement()
    lower, upper, pinned = parse_spec(spec)
    out = []
    msgs = []
    strong = []
    for pf in pkg.py_all():
        for pat, ver, desc in REMOVED:
            m = re.search(pat, pf.text, re.M)
            if not m:
                continue
            rv = _vt(ver)
            if pinned == (99, 0, 0, 0):
                out.append(ev(pf, line_of(pf.text, m.start())))
                msgs.append(f"{desc} removed in {ver}; verifiers comes from a direct reference, check the commit")
                continue
            if pinned is not None:
                if pinned >= rv:
                    out.append(ev(pf, line_of(pf.text, m.start()))); msgs.append(f"{desc} gone in {ver}, pinned {spec}")
                continue
            if upper is not None and upper <= rv:
                continue
            out.append(ev(pf, line_of(pf.text, m.start())))
            never = lower is not None and lower >= rv
            if never or rv >= (0, 4, 1, 0):
                strong.append(desc)
            msgs.append(f"{desc} removed in {ver}; spec '{spec or 'any'}'" + (" never resolves to a working release" if never else " resolves past it on a fresh install"))
    if not out:
        return []
    return [F("removed-verifiers-symbol", "; ".join(sorted(set(msgs))[:3]) + ".", out,
              confidence="likely" if strong else "review")]


@rule("legacy-api-unbounded-pin", "hygiene", "info",
      "Uses the verifiers legacy API (removed in 0.4.0) with no upper bound on verifiers.")
def r_legacy(pkg: Package):
    spec = pkg.verifiers_requirement()
    lower, upper, pinned = parse_spec(spec)
    if pinned is not None or (upper is not None and upper <= (0, 4, 0, 0)):
        return []
    for pf in pkg.py_all():
        m = LEGACY.search(pf.text)
        if m:
            return [F("legacy-api-unbounded-pin", f"Legacy verifiers API with spec '{spec or 'unpinned'}'; a fresh "
                      "install resolves to 0.4.x where this API is gone, unless the Hub runtime supplies an older one.",
                      [ev(pf, line_of(pf.text, m.start()))], confidence="review")]
    return []


STDLIB_MODS = {"re", "json", "os", "sys", "math", "random", "asyncio", "time", "subprocess", "shutil", "tempfile",
               "logging", "itertools", "collections", "functools", "string", "textwrap", "uuid", "hashlib",
               "base64", "copy", "glob", "inspect", "traceback", "datetime", "pathlib", "csv", "io", "ast",
               "np", "pd", "yaml", "requests", "httpx", "vf", "verifiers", "torch", "datasets", "openai"}
BARE_NAMES = {"Path", "Dataset", "load_dataset", "dataclass", "field", "defaultdict", "Counter", "OrderedDict",
              "partial", "AsyncOpenAI", "OpenAI"}
import builtins as _b
BUILTINS = set(dir(_b)) | {"__file__", "__name__", "__builtins__", "__doc__", "__spec__", "__path__", "__package__"}


@rule("missing-import", "broken", "high",
      "A module or common name is used but never imported or defined in the file (NameError).")
def r_missing_import(pkg: Package):
    out, msgs = [], []
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        if re.search(r"^\s*from\s+\S+\s+import\s+\*", pf.text, re.M):
            continue
        # only the env's own package code; per task verifier/test files run elsewhere
        if re.search(r"(^|/)(tasks?[\w-]*|environment|solution|verifier|tests?|data|examples?|scripts?)/", pf.path) or \
                os.path.basename(pf.path).startswith("test"):
            continue
        bound = set()
        for n in ast.walk(pf.tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(n.name)
                if not isinstance(n, ast.ClassDef):
                    a = n.args
                    for x in a.posonlyargs + a.args + a.kwonlyargs:
                        bound.add(x.arg)
                    if a.vararg: bound.add(a.vararg.arg)
                    if a.kwarg: bound.add(a.kwarg.arg)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                bound.add(n.id)
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                bound |= set(n.names)
            elif isinstance(n, ast.alias):
                pass
            elif hasattr(ast, "MatchAs") and isinstance(n, getattr(ast, "MatchAs")) and n.name:
                bound.add(n.name)
        # module level usage (import time failure) vs inside functions
        module_level_nodes = set()
        for stmt in pf.tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in stmt.decorator_list:
                    module_level_nodes |= set(ast.walk(d))
                continue
            if isinstance(stmt, ast.ClassDef):
                for b in stmt.body:
                    if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        module_level_nodes |= set(ast.walk(b))
                continue
            if isinstance(stmt, (ast.If, ast.Try)):
                continue  # guarded imports: skip
            module_level_nodes |= set(ast.walk(stmt))
        ann = set()
        for n in ast.walk(pf.tree):
            for attr in ("annotation", "returns"):
                a = getattr(n, attr, None)
                if a is not None:
                    ann |= set(ast.walk(a))
        found = {}
        for n in ast.walk(pf.tree):
            if n in ann:
                continue
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and isinstance(n.value.ctx, ast.Load):
                nm = n.value.id
                if nm in STDLIB_MODS and nm not in bound and nm not in BUILTINS:
                    found.setdefault(nm, (n.lineno, n.value in module_level_nodes))
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in BARE_NAMES and n.func.id not in bound:
                found.setdefault(n.func.id, (n.lineno, n in module_level_nodes or n.func in module_level_nodes))
        for nm, (ln, mod) in found.items():
            out.append(ev(pf, ln))
            msgs.append(f"`{nm}` undefined in {os.path.basename(pf.path)}" + (" at import time" if mod else ""))
    if not out:
        return []
    return [F("missing-import", "; ".join(msgs[:3]) + ".", out)]


DATA_FILE = re.compile(r"^[\w./\-]+\.(ya?ml|json|jsonl|txt|csv|tsv|parquet|md|j2|jinja2?|toml|pkl|npy|npz|xml|lean)$")


@rule("file-not-shipped", "broken", "high",
      "Code opens a package relative data/config file that is not in the published package (or is "
      "excluded from the wheel by the build config).")
def r_file_not_shipped(pkg: Package):
    have = {os.path.basename(p) for p in pkg.sizes}
    wheel_only = None
    tool = pkg.pyproject.get("tool", {})
    hb = (tool.get("hatch", {}).get("build", {}) or {})
    inc = hb.get("include") or (hb.get("targets", {}).get("wheel", {}) or {}).get("include") \
        or (hb.get("targets", {}).get("wheel", {}) or {}).get("packages") or hb.get("only-include")
    if isinstance(inc, list) and inc and all(str(x).endswith(".py") for x in inc):
        wheel_only = inc
    out, msgs, likely = [], [], False
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        in_fstring = {id(v) for js in ast.walk(pf.tree) if isinstance(js, ast.JoinedStr) for v in js.values}
        base_name = os.path.basename(pf.path)
        if not re.search(r"(import verifiers|from verifiers|def load_environment)", pf.text) or \
                re.match(r"(test_|make_|build_|generate_|prepare_|convert_|download_|create_)", base_name) or \
                re.search(r"(^|/)(scripts|tools|examples|data_prep|notebooks)/", pf.path):
            continue
        # names derived from __file__ (HERE = Path(__file__).parent, current_dir = dirname(__file__), ...)
        filevars = set()
        for _ in range(2):
            for n in ast.walk(pf.tree):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    rhs = ast.unparse(n.value)
                    if "__file__" in rhs or any(re.search(rf"\b{re.escape(v)}\b", rhs) for v in filevars):
                        filevars.add(n.targets[0].id)
        for fi in pf.funcs:
            for n in ast.walk(fi.node):
                if not (isinstance(n, ast.Constant) and isinstance(n.value, str)) or id(n) in in_fstring:
                    continue
                s_ = n.value
                if not DATA_FILE.match(s_) or s_.startswith(("/", "./outputs", "outputs")) or "{" in s_ or len(s_) > 120:
                    continue
                base = os.path.basename(s_)
                if base in ("pyproject.toml", "results.jsonl", "metadata.json", "requirements.txt", "README.md"):
                    continue
                line = pf.line(n.lineno)
                rel_file = "__file__" in line or any(re.search(rf"\b{re.escape(v)}\b", line) for v in filevars)
                direct = re.search(rf"(open|Path|read_text|safe_load|load)\(\s*['\"]{re.escape(s_)}['\"]", line)
                if not (rel_file or direct):
                    continue
                if re.search(r"(['\"]w['\"]|['\"]a['\"]|write|dump\(|to_json|save|mkdir|output)", line):
                    continue
                if base in have:
                    if wheel_only and rel_file and not any(base == os.path.basename(str(x)) for x in wheel_only):
                        out.append(ev(pf, n.lineno)); msgs.append(f"{base} exists in the repo but the wheel build includes only {wheel_only}")
                        likely = True
                    continue
                if re.search(r"(download|hf_hub|snapshot|urlretrieve|requests\.get|git clone|wget|curl|generate|"
                             r"\.exists\(|isfile|is_file|os\.path\.exists)", fi.src):
                    continue
                out.append(ev(pf, n.lineno)); msgs.append(f"{s_} is not in the package")
                if fi.name == "load_environment":
                    likely = True
    if not out:
        return []
    return [F("file-not-shipped", "Loader reads files the package does not ship: " + "; ".join(sorted(set(msgs))[:3]) + ".",
              out, confidence="likely" if likely else "review")]


@rule("no-load-environment", "broken", "high",
      "Package depends on verifiers but defines no load_environment (and no v1 taskset).")
def r_no_load(pkg: Package):
    deps = " ".join(pkg.dependencies()).lower()
    text = pkg.all_py_text()
    if "verifiers" not in deps and "import verifiers" not in text:
        return []
    if not pkg.py:
        return [F("no-load-environment", "No Python source in the package.", ["(package)"], confidence="review", severity="medium")]
    if re.search(r"def\s+load_environment\b|load_environment\s*=", text):
        return []
    if re.search(r"(verifiers\.v1|vf\.Taskset|Taskset\b|def\s+load_taskset|def\s+load_\w*env)", text):
        return []
    if any(p.endswith((".whl",)) for p in pkg.sizes):
        return []
    return [F("no-load-environment", "No load_environment entry point anywhere in the package; the Hub loader "
              "cannot build the env.", [sorted(pkg.py)[0] + ":1"])]


@rule("verifiers-fork-dependency", "broken", "medium",
      "verifiers resolves from a git fork / local path instead of PyPI.")
def r_fork(pkg: Package):
    pp = pkg.pyproject
    src = (pp.get("tool", {}).get("uv", {}).get("sources", {}) or {}).get("verifiers")
    out = []
    if isinstance(src, dict) and ("git" in src or "path" in src):
        url = src.get("git") or src.get("path")
        if "PrimeIntellect-ai/verifiers" not in str(url):
            out.append(f"pyproject.toml:[tool.uv.sources] verifiers={url}")
    for d in pkg.dependencies():
        if re.match(r"\s*verifiers.*@\s*(git\+|https?://|file:)", d) and "PrimeIntellect-ai/verifiers" not in d:
            out.append(f"pyproject.toml:{d}")
    if not out:
        return []
    return [F("verifiers-fork-dependency", "verifiers comes from a fork or local path; a PyPI install of the env "
              "gets a different library and may not load.", out, confidence="review")]


@rule("unpinned-remote-fetch", "broken", "medium",
      "Fetches task files from a moving branch (raw.githubusercontent .../main) at load time.")
def r_remote_fetch(pkg: Package):
    out = []
    for pf in pkg.py_all():
        for m in re.finditer(r"https://(raw\.githubusercontent\.com/[^\s'\"]+/(main|master|HEAD)/|github\.com/[^\s'\"]+/(raw|blob)/(main|master)/)", pf.text):
            out.append(ev(pf, line_of(pf.text, m.start())))
    if not out:
        return []
    return [F("unpinned-remote-fetch", f"{len(out)} URL(s) read from a moving branch; when upstream files move the "
              "env loses tasks or fails.", out, confidence="review")]


@rule("swallowed-dataset-load", "broken", "high",
      "load_dataset is wrapped in try/except that swallows the error and falls back to a tiny hard "
      "coded dataset.")
def r_swallowed(pkg: Package):
    out = []
    for pf, fi in all_funcs(pkg):
        for n in ast.walk(fi.node):
            if not isinstance(n, ast.Try):
                continue
            body = "\n".join(ast.unparse(x) for x in n.body)
            if "load_dataset" not in body:
                continue
            for h in n.handlers:
                hb = "\n".join(ast.unparse(x) for x in h.body)
                if re.search(r"\braise\b", hb):
                    continue
                after = "\n".join(fi.src.splitlines()[(getattr(n, "end_lineno", n.lineno) - fi.start + 1):][:15])
                if re.search(r"(from_list|from_dict|Dataset\(|\[\s*\{)", hb) or \
                        re.search(r"if\s+not\s+\w+\s*:\s*\n\s*\w+\s*=\s*\[\s*\{", after):
                    out.append(ev(pf, h.lineno))
    if not out:
        return []
    return [F("swallowed-dataset-load", "A failed dataset load is swallowed and replaced with a fallback dataset; "
              "training silently runs on the fallback.", out)]


# ---------------------------------------------------------------------------
# 9. answer keys and shipped evals
# ---------------------------------------------------------------------------

PLACEHOLDER = {"unknown", "", "n/a", "na", "none", "null", "todo", "tbd", "placeholder", "?", "-", "xxx"}


def _iter_jsonl_rows(text: str, limit=400):
    n = 0
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue
        n += 1
        if n >= limit:
            break


@rule("placeholder-answer-key", "broken", "critical",
      "Shipped data or eval outputs show the answer field is a placeholder (Unknown / empty) for most rows.")
def r_placeholder(pkg: Package):
    out, msgs = [], []
    by_dir = {}
    for p, t in pkg.text_files(".jsonl", ".json"):
        if p.endswith(".json") and not p.endswith("results.json"):
            continue
        rows = list(_iter_jsonl_rows(t))
        if len(rows) < 8:
            continue
        for key in ("answer", "solution", "gold", "ground_truth", "expected_answer", "target"):
            vals = [r.get(key) for r in rows if isinstance(r, dict) and key in r]
            if len(vals) < 8:
                continue
            ph = sum(1 for v in vals if v is None or (isinstance(v, str) and v.strip().lower() in PLACEHOLDER))
            if ph / len(vals) >= 0.8:
                d = p.split("/")[0]
                by_dir.setdefault(d, []).append((p, key, ph, len(vals)))
    uses = False
    for pf, fi in reward_path_funcs(pkg):
        for x in ast.walk(fi.node):
            if isinstance(x, ast.Compare) and any(isinstance(o, (ast.Eq, ast.NotEq, ast.In)) for o in x.ops):
                names = {y.id for y in ast.walk(x) if isinstance(y, ast.Name)}
                if names & {"answer", "gold", "solution", "ground_truth", "target", "expected"}:
                    uses = True
    if not uses:
        by_dir = {}
    for d, lst in by_dir.items():
        p, key, ph, n = lst[0]
        out.append(f"{p}:1")
        msgs.append(f"{key} is a placeholder in {ph}/{n} rows of {p}" + (f" (+{len(lst) - 1} more files)" if len(lst) > 1 else ""))
    # code level: a dataset builder whose answer comes from a function that returns a placeholder literal
    for pf, fi in all_funcs(pkg):
        if not re.search(r"(answer|solution)", fi.name, re.I):
            continue
        for n in ast.walk(fi.node):
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) \
                    and n.value.value.strip().lower() in {"unknown", "n/a", "todo", "placeholder"}:
                if out:  # only corroborates data evidence
                    out.append(ev(pf, n.lineno))
    if not out:
        return []
    conf = "likely" if any("outputs/" in o or "data" in o for o in out) else "review"
    return [F("placeholder-answer-key", "; ".join(msgs[:2]) + ".", out, confidence=conf)]


@rule("shipped-evals-constant-reward", "broken", "high",
      "The author's own shipped eval outputs give every rollout reward 0 (or one constant value).")
def r_evals_zero(pkg: Package):
    rewards = []
    files = []
    for p, t in pkg.text_files(".jsonl"):
        if "outputs/" not in p and "results" not in os.path.basename(p):
            continue
        rs = [r.get("reward") for r in _iter_jsonl_rows(t, 300) if isinstance(r, dict) and isinstance(r.get("reward"), (int, float))]
        if rs:
            rewards += rs
            files.append(p)
    avg = []
    for p, t in pkg.text_files("metadata.json"):
        if "outputs/" in p:
            try:
                j = json.loads(t)
                if isinstance(j.get("avg_reward"), (int, float)):
                    avg.append((p, j["avg_reward"], j.get("num_examples", 0) * j.get("rollouts_per_example", 1)))
            except Exception:
                pass
    if len(rewards) >= 12 and len(set(round(r, 6) for r in rewards)) == 1:
        v = rewards[0]
        return [F("shipped-evals-constant-reward", f"All {len(rewards)} shipped eval rollouts score exactly {v}; the "
                  "reward may not be wired up (or the task is very hard).", [f"{files[0]}:1"], confidence="review")]
    if not rewards and len(avg) >= 2 and all(a == 0 for _, a, _ in avg) and sum(n for *_, n in avg) >= 12:
        return [F("shipped-evals-constant-reward", f"All {len(avg)} shipped eval runs report avg_reward 0.",
                  [f"{avg[0][0]}:1"], confidence="review")]
    return []


# ---------------------------------------------------------------------------
# 10. Lean
# ---------------------------------------------------------------------------

@rule("lean-no-axiom-check", "hack", "critical",
      "Lean proofs are compiled and accepted without checking for added axioms / sorry, so "
      "`axiom cheat : False` proves anything.")
def r_lean(pkg: Package):
    text = pkg.all_py_text()
    # `lake` and `lean` are matched case-sensitively: "Frozen Lake env" is not the Lean build tool
    if not re.search(r"(\blean\b|[Ll]ean\s?4|LeanTaskset|lake\s+(env|build|exe)|\.lean\b|[Mm]athlib)", text):
        return []
    if not re.search(r"(theorem|lemma|proof|lake|lean_check|compile|repl)", text, re.I):
        return []
    lib = re.search(r"verifiers\.v1\.tasksets\.lean", text)
    if re.search(r"(#print\s+axioms|print_axioms|collect_axioms|CollectAxioms|axioms_used|\"axiom\"|'axiom'|\baxiom\b\s*[\"'\\]|"
                 r"\\baxiom|axiom\s+in\b|forbidden|banned|sorryAx|lean_common)", text) and not lib:
        return []
    if not re.search(r"(lake|lean\s|lean_|subprocess|execute|run\(|repl|compile)", text):
        return []
    m = lib or re.search(r"(lake\s+(env|build)|LeanTaskset|lean)", text)
    pf = next((p for p in pkg.py_all() if m.group(0) in p.text), None)
    where = [ev(pf, line_of(pf.text, pf.text.find(m.group(0))))] if pf else ["(package)"]
    return [F("lean-no-axiom-check",
              "Proofs are accepted on compile success with no axiom/sorry audit" +
              (" (shared verifiers 0.3.x Lean taskset)" if lib else "") + ".", where,
              confidence="likely" if lib else "review")]


# ---------------------------------------------------------------------------
# v0.5 core helpers (shared by the v0.5 rules and widened branches below)
# ---------------------------------------------------------------------------

def _v05_cache(pkg: Package) -> dict:
    c = getattr(pkg, "_envscan_core_v05", None)
    if c is None:
        c = {}
        try:
            setattr(pkg, "_envscan_core_v05", c)
        except Exception:
            pass
    return c


def _reg_cached(pkg: Package) -> dict[str, float]:
    c = _v05_cache(pkg)
    if "reg" not in c:
        c["reg"] = registered_reward_names(pkg)
    return c["reg"]


def _reward_path_cached(pkg: Package):
    c = _v05_cache(pkg)
    if "rpf" not in c:
        c["rpf"] = reward_path_funcs(pkg)
    return c["rpf"]


def _harbor_roots(pkg: Package) -> list[str]:
    """Task folders of Harbor tasks: every folder holding a task.toml or an instruction.md."""
    c = _v05_cache(pkg)
    if "roots" not in c:
        roots = set()
        for p in pkg.files:
            if os.path.basename(p) in ("task.toml", "instruction.md"):
                roots.add(os.path.dirname(p))
        c["roots"] = sorted(roots)
    return c["roots"]


def _task_rel(root: str, p: str) -> str | None:
    if not root:
        return p
    return p[len(root) + 1:] if p.startswith(root + "/") else None


def _task_files(pkg: Package, root: str, sub: str):
    for p, t in pkg.files.items():
        rel = _task_rel(root, p)
        if rel is not None and rel.startswith(sub):
            yield p, rel, t


_TRIVIAL_CODE = re.compile(r"^(import\s|from\s+\S+\s+import\s|#include|package\s|using\s|def\s|async\s+def\s|class\s|"
                           r"@|#|//|/\*|\*|<!--|\"\"\"|'''|set\s+-|cd\s|exit\b|if\s+__name__|cat\s*>|EOF\b|"
                           r"pip3?\s+install|apt(-get)?\s+install|python3?\s+-m\s+pip)")


def _norm_code(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _meaningful_code(s: str) -> bool:
    s = _norm_code(s)
    if len(s) < 20 or sum(ch.isalnum() for ch in s) < 12:
        return False
    return not _TRIVIAL_CODE.match(s)


# a line that does something (assignment, call on an object, return, branch), not an interface or config line
_STATEMENT = re.compile(r"(\s=\s|[^=!<>]=[^=]|\breturn\b|\bif\b|\bfor\b|\bwhile\b|\.\w+\(|;\s*$|=>|\+=|:=|\bawait\b)")


def _statement_code(s: str) -> bool:
    return _meaningful_code(s) and bool(_STATEMENT.search(_norm_code(s)))


def _looks_diff(t: str) -> bool:
    return bool(re.search(r"^(diff --git |@@ -\d+)", t, re.M))


def _diff_lines(t: str):
    add, rem = set(), set()
    for ln in t.splitlines():
        if ln.startswith(("+++", "---")):
            continue
        if ln.startswith("+"):
            add.add(_norm_code(ln[1:]))
        elif ln.startswith("-"):
            rem.add(_norm_code(ln[1:]))
    return add, rem


def _solution_added_lines(pkg: Package, root: str) -> set[str]:
    """Meaningful lines the reference solution adds (diff + lines, or the lines of solve scripts)."""
    c = _v05_cache(pkg)
    key = ("sol", root)
    if key in c:
        return c[key]
    add, rem = set(), set()
    for _p, _rel, t in _task_files(pkg, root, "solution/"):
        if _looks_diff(t):
            a, r = _diff_lines(t)
            add |= a
            rem |= r
        else:
            add |= {_norm_code(x) for x in t.splitlines()}
    out = {x for x in add - rem if _meaningful_code(x)}
    c[key] = out
    return out


def _all_norm_lines(texts) -> set[str]:
    out = set()
    for t in texts:
        for x in t.splitlines():
            out.add(_norm_code(x))
            if x[:1] in "+-":
                out.add(_norm_code(x[1:]))
    return out


def _candidate_lines(text: str):
    """(normalized candidate, line number) for every line of a task text, with diff markers and
    inline code spans split out."""
    for i, ln in enumerate(text.splitlines(), 1):
        yield _norm_code(ln), i
        s = ln.lstrip()
        if s[:1] in "+->":
            yield _norm_code(s[1:]), i
        for m in re.finditer(r"`([^`\n]{20,})`", ln):
            yield _norm_code(m.group(1)), i


FIX_PHRASE = re.compile(r"\b(the (fix|solution|correct fix|correct patch) (is|should be|would be)|proposed (fix|patch|solution)|"
                        r"suggested (fix|patch)|possible fix|fix (is|would be) to)\b", re.I)


def _overlap(text: str, sol: set[str], exclude: set[str]):
    hits = {}
    for cand, i in _candidate_lines(text):
        if cand in sol and cand not in exclude and cand not in hits:
            hits[cand] = i
    return hits


# solution-in-task-text: reader items, main paper arxiv:2410.06992
# (32.67% of resolved issues carried the fix in the issue text).
@rule("solution-in-task-text", "hack", "high",
      "The task text the agent reads contains lines of the reference fix, so the agent can copy the answer.")
def r_solution_in_task_text(pkg: Package):
    out, notes = [], []
    for root in _harbor_roots(pkg):
        inst_p = (root + "/" if root else "") + "instruction.md"
        inst = pkg.files.get(inst_p)
        if not inst:
            continue
        sol = _solution_added_lines(pkg, root)
        if not sol:
            continue
        # lines already in the repo the agent starts from or in the tests are spec, not leak
        exclude = _all_norm_lines(t for _p, _r, t in _task_files(pkg, root, "environment/"))
        exclude |= _all_norm_lines(t for _p, _r, t in _task_files(pkg, root, "tests/"))
        hits = _overlap(inst, {x for x in sol if _statement_code(x)}, exclude)
        phrase = FIX_PHRASE.search(inst)
        if len(hits) >= 3 or (phrase and hits):
            out += [ev(inst_p, i) for i in sorted(hits.values())[:6]]
            notes.append(f"{len(hits)} of {len(sol)} solution lines")
    # verifiers style rows: problem text and gold patch in the same shipped row
    rows_hit, rows_seen, first = 0, 0, None
    for p, t in pkg.files.items():
        if not p.endswith(".jsonl"):
            continue
        for row in _iter_jsonl_rows(t, limit=400):
            if not isinstance(row, dict):
                continue
            patch = next((v for k, v in row.items() if isinstance(v, str) and
                          re.match(r"^(patch|gold_patch|golden_patch|solution_patch|reference_patch|fix_patch)$", str(k))), None)
            if not patch or not _looks_diff(patch):
                continue
            rows_seen += 1
            a, r = _diff_lines(patch)
            sol = {x for x in a - r if _statement_code(x)}
            if not sol:
                continue
            text = "\n".join(v for k, v in row.items() if isinstance(v, str) and
                             re.match(r"^(problem_statement|issue|issue_text|prompt|question|instruction|task|description)$", str(k)))
            if not text:
                continue
            h = _overlap(text, sol, r)
            if len(h) >= 3 or (h and FIX_PHRASE.search(text)):
                rows_hit += 1
                first = first or p
    if rows_hit and first:
        out.append(f"{first}:1")
        notes.append(f"{rows_hit} of {rows_seen} dataset rows")
    if not out:
        return []
    return [F("solution-in-task-text", "The task text quotes lines that the reference fix adds (" + "; ".join(notes[:3]) +
              "), so an agent can copy the fix.", out, confidence="review")]


# ---------------------------------------------------------------------------
# verifier-deserializes-agent-artifact: reader items, main source Harbor RFC 0003 survey
# (5 of 66 Terminal-Bench tasks use code running loaders).
# ---------------------------------------------------------------------------

_PICKLE_MODS = {"pickle", "cPickle", "_pickle", "dill", "cloudpickle", "joblib"}
# 0.5.0 fix wave (check_4.md): pickle or torch.load counts only when the path can come from agent output
_CODE_LOADER = re.compile(r"(pickle|torch\.load|numpy\.load|joblib|dill|cloudpickle)")
_AGENT_PATH = re.compile(r"(download|read_file|sandbox|container|b64decode|/app\b|/workspace|submission|artifact|"
                         r"agent|upload|completion|response|output_dir|outputs?/)", re.I)
_GRADER_OWN_PATH = re.compile(r"(/tests?\b|__file__|fixtures?|TEST_DIR|tests?_dir|/oracle|/solution|expected|reference|"
                              r"gold|ground_truth|golden)", re.I)
MODEL_PARAM = re.compile(r"^(completion|completions|response|responses|parsed|parsed_answer|prediction|predictions|pred|"
                         r"model_output|model_answer|output|outputs|generated\w*|submission|candidate|student_answer|"
                         r"llm_\w+|final_response|reply)$")
MODEL_EXPR = re.compile(r"(completion|response|parse_answer|llm_answer|model_answer|model_output|prediction|submission|"
                        r"generated|student_answer|final_response)")


def _old_torch(pkg: Package, path: str = "") -> bool:
    """torch pinned below 2.6 (where torch.load still defaults to full unpickling) in the task that holds path."""
    root = next((r for r in _harbor_roots(pkg) if r and path.startswith(r + "/")), "")
    c = _v05_cache(pkg)
    key = ("old_torch", root)
    if key not in c:
        old = False
        for p, t in pkg.files.items():
            if p.endswith((".jsonl", ".json", ".csv")) or (root and not p.startswith(root + "/")):
                continue
            for m in re.finditer(r"\btorch\s*(==|<=|<|~=)\s*(\d+)\.(\d+)", t):
                if (int(m.group(2)), int(m.group(3))) < (2, 6):
                    old = True
        c[key] = old
    return c[key]


def _loader_kind(pkg: Package, n: ast.Call, pf: PyFile | None = None) -> str | None:
    """Name of the code running loader this call is, or None."""
    f = n.func
    mod = attr = None
    if isinstance(f, ast.Attribute):
        attr = f.attr
        mod = f.value.id if isinstance(f.value, ast.Name) else (f.value.attr if isinstance(f.value, ast.Attribute) else None)
    elif isinstance(f, ast.Name):
        attr = f.id
    kws = {k.arg: k.value for k in n.keywords if k.arg}

    def const(v):
        return v.value if isinstance(v, ast.Constant) else None
    if mod in _PICKLE_MODS and attr in ("load", "loads"):
        return f"{mod}.{attr}"
    if attr == "read_pickle":
        return "read_pickle"
    if mod == "torch" and attr == "load":
        wo = kws.get("weights_only")
        if wo is not None and const(wo) is False:
            return "torch.load(weights_only=False)"
        if wo is None and _old_torch(pkg, pf.path if pf else ""):
            return "torch.load on torch before 2.6"
        return None
    if mod in ("np", "numpy") and attr == "load" and const(kws.get("allow_pickle")) is True:
        return "numpy.load(allow_pickle=True)"
    if mod == "yaml" and attr in ("unsafe_load", "unsafe_load_all"):
        return f"yaml.{attr}"
    if mod == "yaml" and attr in ("load", "load_all"):
        ld = kws.get("Loader") or (n.args[1] if len(n.args) > 1 else None)
        if ld is not None:
            s = ast.unparse(ld)
            if re.search(r"(^|\.)(Unsafe)?Loader$", s) and "Safe" not in s and "Full" not in s and "Base" not in s:
                return f"yaml.{attr}(Loader={s})"
        return None
    if attr in ("sympify", "parse_expr") and mod in (None, "sympy", "sp", "sym", "parsing", "sympy_parser") and \
            pf is not None and re.search(r"^\s*(import sympy|from sympy)", pf.text, re.M):
        return attr
    return None


def _tainted_names(fi: FuncInfo, seeds: set[str]) -> set[str]:
    t = set(seeds)
    for _ in range(3):
        before = len(t)
        for n in ast.walk(fi.node):
            val, targets = None, []
            if isinstance(n, ast.Assign):
                val, targets = n.value, n.targets
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                val, targets = n.value, [n.target]
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                val, targets = n.iter, [n.target]
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                val, targets = n.context_expr, [n.optional_vars]
            if val is None:
                continue
            used = {x.id for x in ast.walk(val) if isinstance(x, ast.Name)}
            if used & t or MODEL_EXPR.search(_unparse(val)):
                for tg in targets:
                    for x in ast.walk(tg):
                        if isinstance(x, ast.Name):
                            t.add(x.id)
        if len(t) == before:
            break
    return t


def _unparse(n) -> str:
    try:
        return ast.unparse(n)
    except Exception:
        return ""


def _arg_text(fi: FuncInfo, pf: PyFile, a) -> str:
    """Text of an argument plus one hop of the assignments that feed its names."""
    s = _unparse(a)
    names = {x.id for x in ast.walk(a) if isinstance(x, ast.Name)}
    extra = []
    for n in ast.walk(fi.node):
        if isinstance(n, ast.Assign) and any(isinstance(tg, ast.Name) and tg.id in names for tg in n.targets):
            extra.append(_unparse(n.value))
        elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name) and n.optional_vars.id in names:
            extra.append(_unparse(n.context_expr))
    for nm in names:
        if nm in pf.consts:
            extra.append(pf.consts[nm])
    return s + " " + " ".join(extra)


def _staged_by_test_script(pkg: Package, path: str, txt: str) -> bool:
    """The verifier script itself writes the loaded path (untar, copy, download) before the test reads it."""
    root = next((r for r in _harbor_roots(pkg) if path.startswith((r + "/") if r else "")), None)
    if root is None:
        return False
    sh = "\n".join(t for p, rel, t in _task_files(pkg, root, "tests/") if rel.endswith(".sh"))
    for m in re.finditer(r"(/[\w.-]+(?:/[\w.-]+)+)", txt):
        d = m.group(1)
        d = d if not os.path.splitext(d)[1] else os.path.dirname(d)
        while d.count("/") >= 2:
            if re.search(rf"(tar\b[^\n]*-C\s*['\"]?(\$\{{?\w+\}}?)?{re.escape(d)}|\b(cp|mv|curl|wget|unzip)\b[^\n]*{re.escape(d)})", sh) or \
                    re.search(rf"(tar\b[^\n]*-C\s*['\"]?\$\{{?\w+\}}?/{re.escape(d.rsplit('/', 1)[-1])})", sh):
                return True
            d = os.path.dirname(d)
    return False


def _harbor_test_pyfiles(pkg: Package) -> list[PyFile]:
    c = _v05_cache(pkg)
    if "htest" not in c:
        out = []
        for root in _harbor_roots(pkg):
            for p, rel, t in _task_files(pkg, root, "tests/"):
                if p.endswith(".py"):
                    from .package import parse_py
                    out.append(parse_py(p, t))
        c["htest"] = out
    return c["htest"]


def _tool_names(pkg: Package) -> set[str]:
    """Names of functions handed to the model as tools."""
    c = _v05_cache(pkg)
    if "tools" in c:
        return c["tools"]
    out = set()
    for pf in pkg.py_all():
        if pf.tree is None:
            continue
        for n in ast.walk(pf.tree):
            if isinstance(n, ast.Call):
                fname = _callable_name(n.func) or ""
                for k in n.keywords:
                    if k.arg in ("tools", "tool_list", "functions") and isinstance(k.value, (ast.List, ast.Tuple)):
                        out |= {nm for nm in (_callable_name(e) for e in k.value.elts) if nm}
                if fname in ("add_tool", "register_tool") and n.args:
                    nm = _callable_name(n.args[0])
                    if nm:
                        out.add(nm)
            elif isinstance(n, ast.Assign) and isinstance(n.value, (ast.List, ast.Tuple)):
                tg = _unparse(n.targets[0])
                if re.search(r"tools?\b|TOOLS", tg):
                    out |= {nm for nm in (_callable_name(e) for e in n.value.elts) if nm}
        for fi in pf.funcs:
            if any(re.search(r"\btool\b", d) for d in fi.decorators):
                out.add(fi.name)
    c["tools"] = out
    return out


@rule("verifier-deserializes-agent-artifact", "hack", "high",
      "The grader loads a file or text the agent controls with a loader that runs code, such as pickle or unsafe torch.load.")
def r_deserialize_agent(pkg: Package):
    out, kinds = [], set()
    # Harbor: tests/*.py run by the verifier inside the box the agent used
    for pf in _harbor_test_pyfiles(pkg):
        for fi in pf.funcs:
            for n in ast.walk(fi.node):
                if not isinstance(n, ast.Call):
                    continue
                k = _loader_kind(pkg, n, pf)
                if not k or not n.args:
                    continue
                a = n.args[0]
                if isinstance(a, ast.Constant) and k not in ("sympify", "parse_expr"):
                    if _GRADER_OWN_PATH.search(str(a.value)):
                        continue
                txt = _arg_text(fi, pf, a)
                if _GRADER_OWN_PATH.search(txt) or _staged_by_test_script(pkg, pf.path, txt):
                    continue
                if k in ("sympify", "parse_expr") and not re.search(r"(read|open|output|result|answer|stdout|/app|content)", txt):
                    continue
                out.append(ev(pf, n.lineno))
                kinds.add(k)
    # verifiers: reward path and env feedback code that loads model text or sandbox files
    path_funcs = {(pf.path, fi.start) for pf, fi in _reward_path_cached(pkg)}
    tools = _tool_names(pkg)
    sym_where = set()
    for pf, fi in all_funcs(pkg):
        on_path = (pf.path, fi.start) in path_funcs
        is_tool = fi.name in tools
        if not (on_path or is_tool or re.search(r"(env_response|post_rollout|grade|verify)", fi.name)):
            continue
        if pf.text.lstrip().startswith("# /// script"):
            continue
        seeds = {p for p in fi.params if MODEL_PARAM.match(p)}
        if is_tool:
            seeds |= {p for p in fi.params if p not in ("self", "cls", "state", "kwargs", "config")}
        taint = _tainted_names(fi, seeds)
        for n in ast.walk(fi.node):
            if not isinstance(n, ast.Call):
                continue
            k = _loader_kind(pkg, n, pf)
            if not k or not n.args or isinstance(n.args[0], ast.Constant):
                continue
            a = n.args[0]
            names = {x.id for x in ast.walk(a) if isinstance(x, ast.Name)}
            at = _unparse(a)
            atxt = _arg_text(fi, pf, a)
            sandbox_read = re.search(r"(download|read_file|\.read\(|cat\s|sandbox|container|b64decode)", atxt)
            if sandbox_read and _CODE_LOADER.search(k) and not _AGENT_PATH.search(atxt):
                sandbox_read = None  # 0.5.0 fix wave: pickle or torch.load of a file the agent cannot write
            if names & taint or MODEL_EXPR.search(at) or (sandbox_read and k not in ("sympify", "parse_expr")):
                if _GRADER_OWN_PATH.search(at):
                    continue
                out.append(ev(pf, n.lineno))
                kinds.add(k)
                if k in ("sympify", "parse_expr"):
                    sym_where.add("tool" if is_tool and not on_path else "grader")
    if not out:
        return []
    if sym_where == {"tool"} and kinds <= {"sympify", "parse_expr"}:
        return [F("verifier-deserializes-agent-artifact",
                  f"The tools run {', '.join(sorted(kinds)[:3])} on arguments the model writes, in the same process that "
                  "computes the reward, so a crafted tool argument runs code there and can set its own score.",
                  out, confidence="review")]
    return [F("verifier-deserializes-agent-artifact",
              f"The grader runs {', '.join(sorted(kinds)[:3])} on data the agent controls, so a crafted file or answer "
              "runs code in the grader and can set its own score.", out, confidence="review")]


# ---------------------------------------------------------------------------
# shell-injection-from-model-text: reader items, main paper arxiv:2605.12673
# ---------------------------------------------------------------------------

_SHELL_ALWAYS = {("os", "system"), ("os", "popen"), ("subprocess", "getoutput"), ("subprocess", "getstatusoutput"),
                 ("commands", "getoutput"), ("asyncio", "create_subprocess_shell")}
_SUBPROC = {"run", "call", "check_call", "check_output", "Popen"}


def _shell_cmd_arg(n: ast.Call):
    f = n.func
    if not isinstance(f, ast.Attribute):
        return None
    mod = f.value.id if isinstance(f.value, ast.Name) else None
    if (mod, f.attr) in _SHELL_ALWAYS:
        return n.args[0] if n.args else None
    if mod == "subprocess" and f.attr in _SUBPROC:
        sh = next((k.value for k in n.keywords if k.arg == "shell"), None)
        if isinstance(sh, ast.Constant) and sh.value is True:
            if n.args:
                return n.args[0]
            return next((k.value for k in n.keywords if k.arg in ("args", "cmd")), None)
    return None


@rule("shell-injection-from-model-text", "hack", "high",
      "Grader or tool code runs a shell command built from model text on the host with no quoting.")
def r_shell_injection(pkg: Package):
    out = []
    host_tool = False
    path_funcs = {(pf.path, fi.start) for pf, fi in _reward_path_cached(pkg)}
    tools = _tool_names(pkg)
    for pf, fi in all_funcs(pkg):
        if pf.text.lstrip().startswith("# /// script"):
            continue
        is_tool = fi.name in tools
        on_path = (pf.path, fi.start) in path_funcs
        if not (is_tool or on_path):
            continue
        seeds = {p for p in fi.params if MODEL_PARAM.match(p)}
        if is_tool:
            seeds |= {p for p in fi.params if p not in ("self", "cls", "state", "kwargs", "config", "timeout")}
        if not seeds:
            continue
        taint = _tainted_names(fi, seeds)
        for n in ast.walk(fi.node):
            if not isinstance(n, ast.Call):
                continue
            a = _shell_cmd_arg(n)
            if a is None or isinstance(a, ast.Constant):
                continue
            at = _unparse(a)
            names = {x.id for x in ast.walk(a) if isinstance(x, ast.Name)}
            if not (names & taint):
                continue
            if re.search(r"(shlex\.quote|shlex\.join|pipes\.quote|\bquote\()", at) or \
                    re.search(r"(shlex\.quote|shlex\.join|pipes\.quote)", fi.src):
                continue
            out.append(ev(pf, n.lineno))
            if is_tool and isinstance(a, ast.Name) and a.id in seeds:
                host_tool = True
    if not out:
        return []
    msg = ("A model tool passes the model's command straight to a host shell, so the policy can read or change the "
           "grader and answer files." if host_tool else
           "A shell command on the grader host is built from model text with no quoting, so the policy can inject "
           "commands that change its score.")
    return [F("shell-injection-from-model-text", msg, out, confidence="review")]


# ---------------------------------------------------------------------------

def run_rules(pkg: Package, only=None):
    findings = []
    errors = []
    for rid, meta in RULES.items():
        if only and rid not in only:
            continue
        if not only and rid in OFF_BY_DEFAULT and rid not in ENABLED:
            continue
        try:
            findings += [f for f in meta["fn"](pkg) if variant_on(rid, getattr(f, "variant", ""), only)]
        except RecursionError:
            errors.append(f"{rid}: RecursionError")
        except Exception as e:  # a rule bug must not kill a hub scan
            errors.append(f"{rid}: {type(e).__name__}: {e}")
    return findings, errors


# ---------------------------------------------------------------------------
# v0.2 rule modules. Each family lives in its own file so they can be built and
# reviewed separately; importing one registers its rules in RULES.
# ---------------------------------------------------------------------------

import importlib as _importlib

for _m in ("rules_tests", "rules_image", "rules_grader", "rules_reward", "rules_judge",
           # v0.5 families
           "rules_match", "rules_harbor", "rules_rubric", "rules_testq", "rules_dataset", "rules_env"):
    try:
        _importlib.import_module(f"{__package__}.{_m}")
    except ModuleNotFoundError as _e:
        if _e.name != f"{__package__}.{_m}":
            raise

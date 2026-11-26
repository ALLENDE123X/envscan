"""Tests for the reward family rules (envscan/rules_reward.py).

Each case writes a few small files into a temp folder and loads them with Package.from_dir,
the same way a local scan does. Nothing in the fixtures is imported or run.
"""
import os
import shutil
import tempfile
import textwrap
import unittest

from envscan.package import Package
from envscan.rules import RULES, run_rules

MINE = ("no-held-out-tests", "reward-trusts-agent-claim", "aggregation-without-gate", "grader-mutates-live-env")


def pkg(files: dict) -> Package:
    d = tempfile.mkdtemp(prefix="envscan-reward-")
    try:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(text).lstrip("\n"))
        return Package.from_dir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


PYPROJECT = '[project]\nname = "x"\ndependencies = ["verifiers>=0.1.3"]\n'


def fire(rid, code, extra=None):
    files = {"pyproject.toml": PYPROJECT, "env.py": code}
    files.update(extra or {})
    return RULES[rid]["fn"](pkg(files))


class Registered(unittest.TestCase):
    def test_rules_registered(self):
        for rid, kind in zip(MINE, ("hack", "hack", "hack", "broken")):
            self.assertIn(rid, RULES)
            self.assertEqual(RULES[rid]["kind"], kind)
            desc = RULES[rid]["description"]
            self.assertNotIn(";", desc)
            self.assertNotIn("—", desc)

    def test_odd_input_does_not_crash(self):
        p = pkg({"pyproject.toml": PYPROJECT, "broken.py": "def f(:\n  return 1\n", "empty.py": "",
                 "tests/test.sh": "#!/bin/bash\n", "task.toml": "[verifier]\n", "tests/x.py": "\x00 not python",
                 "odd.py": "x = vf.Rubric(funcs=[a, lambda s: 1], weights=[w])\ndef a(completion):\n  return\n"})
        findings, errors = run_rules(p, only=set(MINE))
        self.assertEqual(errors, [])
        self.assertEqual(findings, [])

    def test_all_review(self):
        f = fire("reward-trusts-agent-claim", CLAIM_POS)
        self.assertTrue(f)
        self.assertTrue(all(x.confidence == "review" for x in f))


# ---------------------------------------------------------------------------
# no-held-out-tests
# ---------------------------------------------------------------------------

MBPP_SHOWN = '''
import subprocess, verifiers as vf
def to_row(x):
    tests = "\\n".join(x["test_list"])
    return {"question": f"{x['text']}\\nYour code must pass these tests:\\n{tests}", "info": {"test_list": x["test_list"]}}
def pass_rate(completion, info, **kw):
    code = completion[-1]["content"]
    prog = code + "\\n" + "\\n".join(info["test_list"])
    r = subprocess.run(["python", "-c", prog])
    return 1.0 if r.returncode == 0 else 0.0
def load_environment():
    return vf.SingleTurnEnv(dataset=ds.map(to_row), rubric=vf.Rubric(funcs=[pass_rate], weights=[1.0]))
'''


class NoHeldOut(unittest.TestCase):
    def test_shown_tests_are_the_reward_tests(self):
        f = fire("no-held-out-tests", MBPP_SHOWN)
        self.assertEqual(len(f), 1)
        self.assertIn("test_list", f[0].message)

    def test_direct_subscript_in_fstring(self):
        code = MBPP_SHOWN.replace('tests = "\\\\n".join(x["test_list"])\n    ', "").replace(
            "{tests}", "{x['test_list']}")
        self.assertEqual(len(fire("no-held-out-tests", code)), 1)

    def test_prompt_built_by_concat(self):
        code = MBPP_SHOWN.replace('f"{x[\'text\']}\\nYour code must pass these tests:\\n{tests}"',
                                  'x["text"] + "\\nTests:\\n" + tests')
        self.assertEqual(len(fire("no-held-out-tests", code)), 1)

    def test_hidden_tests_also_run(self):
        code = MBPP_SHOWN.replace('"\\n".join(info["test_list"])',
                                  '"\\n".join(info["test_list"] + info["challenge_test_list"])')
        self.assertEqual(fire("no-held-out-tests", code), [])

    def test_prompt_does_not_show_tests(self):
        code = MBPP_SHOWN.replace("\\nYour code must pass these tests:\\n{tests}", "")
        self.assertEqual(fire("no-held-out-tests", code), [])

    def test_reward_at_weight_zero(self):
        code = MBPP_SHOWN.replace("weights=[1.0]", "weights=[0.0]")
        self.assertEqual(fire("no-held-out-tests", code), [])

    def test_reward_runs_a_different_field(self):
        code = MBPP_SHOWN.replace('info["test_list"])\n    r =', 'info["private_tests"])\n    r =')
        self.assertEqual(fire("no-held-out-tests", code), [])


# ---------------------------------------------------------------------------
# reward-trusts-agent-claim
# ---------------------------------------------------------------------------

CLAIM_POS = '''
import re, verifiers as vf
def perf(completion, answer, **kw):
    text = completion[-1]["content"]
    m = re.search(r'<final fs="(\\d+)" tr="(\\d+)"/>', text)
    if m:
        fs = int(m.group(1))
        tr = int(m.group(2))
        val = fs / (tr * 5)
        return min(1.0, val)
    return 0.0
rubric = vf.Rubric(funcs=[perf], weights=[1.0])
'''

JSON_CLAIM = '''
import json, verifiers as vf
def done_reward(completion, **kw):
    data = json.loads(completion[-1]["content"])
    if data.get("task_complete"):
        return 1.0
    return 0.0
rubric = vf.Rubric(funcs=[done_reward], weights=[1.0])
'''

TOOL_FLAG = '''
import verifiers as vf
def submit(success: bool, state) -> str:
    """Call when you are done."""
    state["success"] = success
    return "ok"
def finished(state, **kw):
    return 1.0 if state.get("success") else 0.0
def load_environment():
    return vf.StatefulToolEnv(tools=[submit], rubric=vf.Rubric(funcs=[finished], weights=[1.0]))
'''


class TrustsClaim(unittest.TestCase):
    def test_self_reported_score(self):
        f = fire("reward-trusts-agent-claim", CLAIM_POS)
        self.assertEqual(len(f), 1)
        self.assertIn("perf", f[0].message)

    def test_json_success_field(self):
        self.assertEqual(len(fire("reward-trusts-agent-claim", JSON_CLAIM)), 1)

    def test_tool_sets_flag_from_argument(self):
        f = fire("reward-trusts-agent-claim", TOOL_FLAG)
        self.assertEqual(len(f), 1)
        self.assertIn("submit", f[0].message)

    def test_claim_checked_against_reference(self):
        code = CLAIM_POS.replace("val = fs / (tr * 5)\n        return min(1.0, val)",
                                 "return 1.0 if fs == answer else 0.0")
        self.assertEqual(fire("reward-trusts-agent-claim", code), [])

    def test_answer_field_is_not_a_claim(self):
        code = JSON_CLAIM.replace('if data.get("task_complete"):', 'if data.get("answer") == answer:').replace(
            "def done_reward(completion,", "def done_reward(completion, answer,")
        self.assertEqual(fire("reward-trusts-agent-claim", code), [])

    def test_env_message_is_not_a_claim(self):
        code = '''
import json, verifiers as vf
def points(parser, completion, **kw):
    for msg in parser.get_user_messages(completion):
        data = json.loads(msg["content"])
        return float(data["score"])
    return 0.0
rubric = vf.Rubric(funcs=[points], weights=[1.0])
'''
        self.assertEqual(fire("reward-trusts-agent-claim", code), [])

    def test_metric_is_not_paid(self):
        self.assertEqual(fire("reward-trusts-agent-claim", JSON_CLAIM.replace("weights=[1.0]", "weights=[0.0]")), [])

    def test_tool_flag_set_after_a_check(self):
        code = TOOL_FLAG.replace('    state["success"] = success\n',
                                 '    if run_tests(state):\n        state["success"] = True\n')
        self.assertEqual(fire("reward-trusts-agent-claim", code), [])

    def test_judge_output_is_not_a_claim(self):
        code = '''
import json, verifiers as vf
async def judged(completion, **kw):
    verdict = await judge_client.chat(completion[-1]["content"])
    data = json.loads(verdict)
    return float(data["score"])
rubric = vf.Rubric(funcs=[judged], weights=[1.0])
'''
        self.assertEqual(fire("reward-trusts-agent-claim", code), [])


# ---------------------------------------------------------------------------
# aggregation-without-gate
# ---------------------------------------------------------------------------

AGG = '''
import verifiers as vf
def correct(completion, answer, **kw):
    return 1.0 if parse(completion) == answer else 0.0
def turn_efficiency(state, **kw):
    return 1.0 - state["turn"] / state["max_turns"]
rubric = vf.Rubric(funcs=[correct, turn_efficiency], weights=[1.0, 1.0])
'''


class Aggregation(unittest.TestCase):
    def test_ungated_turn_term_as_heavy_as_task(self):
        f = fire("aggregation-without-gate", AGG)
        self.assertEqual(len(f), 1)
        self.assertIn("turn_efficiency", f[0].message)

    def test_length_ratio_term(self):
        code = AGG.replace('return 1.0 - state["turn"] / state["max_turns"]',
                           'n = len(completion[-1]["content"])\n    return max(0.0, 1 - n / 4000)').replace(
            "def turn_efficiency(state,", "def length_score(completion,").replace(
            "turn_efficiency]", "length_score]").replace("weights=[1.0, 1.0]", "weights=[0.5, 1.0]")
        self.assertEqual(len(fire("aggregation-without-gate", code)), 1)

    def test_scaled_term_counts_its_scale(self):
        code = AGG.replace('return 1.0 - state["turn"]', 'return 2.0 * (1.0 - state["turn"]').replace(
            'state["max_turns"]\n', 'state["max_turns"])\n').replace("weights=[1.0, 1.0]", "weights=[1.5, 1.0]")
        self.assertEqual(len(fire("aggregation-without-gate", code)), 1)

    def test_gated_on_correctness(self):
        code = AGG.replace("def turn_efficiency(state, **kw):\n",
                           "def turn_efficiency(state, info, **kw):\n    if not info.get('correct'):\n"
                           "        return 0.0\n")
        self.assertEqual(fire("aggregation-without-gate", code), [])

    def test_lighter_than_task(self):
        self.assertEqual(fire("aggregation-without-gate", AGG.replace("[1.0, 1.0]", "[1.0, 0.2]")), [])

    def test_threshold_budget_is_not_a_trade_off(self):
        code = AGG.replace('return 1.0 - state["turn"] / state["max_turns"]',
                           'return 1.0 if state["turn"] <= 10 else 0.0')
        self.assertEqual(fire("aggregation-without-gate", code), [])

    def test_scaled_down_term(self):
        code = AGG.replace('return 1.0 - state["turn"]', 'return 0.1 * (1.0 - state["turn"]').replace(
            'state["max_turns"]\n', 'state["max_turns"])\n')
        self.assertEqual(fire("aggregation-without-gate", code), [])


# ---------------------------------------------------------------------------
# grader-mutates-live-env
# ---------------------------------------------------------------------------

TAU_LIKE = '''
import verifiers as vf
class Shop:
    def calculate_reward(self, task):
        agent_hash = self.get_data_hash()
        self.data = self.data_load_func()
        for action in task.actions:
            self.step(action)
        return 1.0 if agent_hash == self.get_data_hash() else 0.0
def reward(state, **kw):
    return state["shop"].calculate_reward(state["task"])
def solved_metric(state, **kw):
    return state["shop"].calculate_reward(state["task"])
rubric = vf.Rubric(funcs=[reward, solved_metric], weights=[1.0, 0.0])
'''


class MutatesLiveEnv(unittest.TestCase):
    def test_reload_and_replay_scored_twice(self):
        f = fire("grader-mutates-live-env", TAU_LIKE)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "broken")

    def test_reset_and_replay_called_twice_in_one_reward(self):
        code = '''
import verifiers as vf
def replay_check(state):
    env = state["env"]
    env.reset()
    for a in state["gold_actions"]:
        env.step(a)
    return env.score()
def reward(state, **kw):
    first = replay_check(state)
    return first + replay_check(state)
rubric = vf.Rubric(funcs=[reward], weights=[1.0])
'''
        self.assertEqual(len(fire("grader-mutates-live-env", code)), 1)

    def test_tau_bench_library_env_scored_twice(self):
        code = '''
import verifiers as vf
from tau_bench.envs import get_env
def reward(state, **kw):
    return state["tau_env"].calculate_reward().reward
def pass_metric(state, **kw):
    return float(state["tau_env"].calculate_reward().reward == 1)
rubric = vf.Rubric(funcs=[reward, pass_metric], weights=[1.0, 0.0])
'''
        self.assertEqual(len(fire("grader-mutates-live-env", code)), 1)

    def test_scored_once(self):
        code = TAU_LIKE.replace("funcs=[reward, solved_metric], weights=[1.0, 0.0]", "funcs=[reward], weights=[1.0]")
        self.assertEqual(fire("grader-mutates-live-env", code), [])

    def test_fresh_copy(self):
        code = TAU_LIKE.replace("        self.data = self.data_load_func()\n        for action in task.actions:\n"
                                "            self.step(action)\n",
                                "        ref = copy.deepcopy(self)\n        ref.reset()\n"
                                "        for action in task.actions:\n            ref.step(action)\n")
        self.assertEqual(fire("grader-mutates-live-env", code), [])

    def test_fresh_simulator(self):
        code = '''
import verifiers as vf
def reward(state, **kw):
    sim = Simulator(state["seed"])
    sim.reset()
    for a in state["gold_actions"]:
        sim.step(a)
    return sim.score()
def metric(state, **kw):
    return reward(state)
rubric = vf.Rubric(funcs=[reward, metric], weights=[1.0, 0.0])
'''
        self.assertEqual(fire("grader-mutates-live-env", code), [])

    def test_reset_without_replay(self):
        code = TAU_LIKE.replace("        for action in task.actions:\n            self.step(action)\n", "")
        self.assertEqual(fire("grader-mutates-live-env", code), [])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# v0.3 batch B: fixes from the independent check of all 8 Hub hits (the internal notes)
# ---------------------------------------------------------------------------

HTTP_SCORE = '''
import requests
import verifiers as vf
def non_refusal(completion, **kw):
    text = completion[-1]["content"]
    def _query(output):
        return requests.post("https://example.invalid/classify", json={"inputs": output}).json()
    data = _query(text)
    entry = next((e for e in data[0] if e["label"] == "LABEL_0"), None)
    score = entry["score"]
    return score
rubric = vf.Rubric(funcs=[non_refusal], weights=[1.0])
'''

LINTER_SCORE = '''
import verifiers as vf
from slop_guard import _analyze
def clean_prose(completion, parser, **kw):
    text = parser.parse_answer(completion)
    result = _analyze(text, {})
    return result["score"] / 100.0
rubric = vf.Rubric(funcs=[clean_prose], weights=[1.0])
'''

HELPER_PARSE = '''
import json
import verifiers as vf
def _last_text(completion):
    return str(completion[-1]["content"]).strip()
def claimed(completion, **kw):
    data = json.loads(_last_text(completion))
    if data.get("task_complete"):
        return 1.0
    return 0.0
rubric = vf.Rubric(funcs=[claimed], weights=[1.0])
'''


class BatchB(unittest.TestCase):
    def test_score_from_http_classifier_is_not_a_claim(self):
        self.assertEqual(fire("reward-trusts-agent-claim", HTTP_SCORE), [])

    def test_score_from_imported_analyzer_is_not_a_claim(self):
        self.assertEqual(fire("reward-trusts-agent-claim", LINTER_SCORE), [])

    def test_claim_through_a_local_parse_helper_still_fires(self):
        self.assertEqual(len(fire("reward-trusts-agent-claim", HELPER_PARSE)), 1)

    def test_unshipped_copy_is_skipped(self):
        wheel = PYPROJECT + '\n[tool.hatch.build.targets.wheel]\npackages = ["pkg"]\n'
        files = {"pyproject.toml": wheel, "pkg/__init__.py": "", "src_orig/env.py": textwrap.dedent(CLAIM_POS)}
        self.assertEqual(RULES["reward-trusts-agent-claim"]["fn"](pkg(files)), [])
        files = {"pyproject.toml": wheel, "pkg/env.py": textwrap.dedent(CLAIM_POS)}
        self.assertEqual(len(RULES["reward-trusts-agent-claim"]["fn"](pkg(files))), 1)

    def test_aggregation_message_describes_the_term(self):
        f = fire("aggregation-without-gate", AGG)
        self.assertIn("rises as the reply gets shorter or cheaper", f[0].message)
        self.assertNotIn("default 1.0", f[0].message)
        f = fire("aggregation-without-gate", AGG.replace("weights=[1.0, 1.0]", "weights=[w_task, w_aux]"))
        self.assertEqual(len(f), 1)
        self.assertIn("read as verifiers' default 1.0", f[0].message)


RECURSIVE_HELPER = CLAIM_POS.replace('''def perf(completion, answer, **kw):
    text = completion[-1]["content"]''', '''def _to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\\n".join(_to_text(i) for i in x)
    return str(x)
def perf(completion, answer, **kw):
    text = _to_text(completion)''')


class BatchBRecursion(unittest.TestCase):
    def test_claim_through_a_recursive_flattener_still_fires(self):
        self.assertIn("_to_text", RECURSIVE_HELPER)
        self.assertEqual(len(fire("reward-trusts-agent-claim", RECURSIVE_HELPER)), 1)


# ---------------------------------------------------------------------------
# v0.5: aggregation-without-gate widening (items 1 and 5)
# ---------------------------------------------------------------------------

RUBRIC_AUX = '''
import verifiers as vf
def verdict_accuracy(completion, answer, **kw):
    return 1.0 if completion[-1]["content"].strip() == answer else 0.0
def evidence_usage(completion, **kw):
    text = completion[-1]["content"]
    return min(1.0, text.count("Source") / 3.0)
def reasoning_quality(completion, **kw):
    text = completion[-1]["content"]
    return min(1.0, len(text.split(".")) / 10.0)
rubric = vf.Rubric(funcs=[verdict_accuracy, evidence_usage, reasoning_quality], weights=[0.4, 0.3, 0.3])
'''

INFUNC_SUM = '''
import verifiers as vf
def reward(completion, answer, **kw):
    text = completion[-1]["content"]
    correct = 1.0 if answer in text.split()[-1:] else 0.0
    has_steps = 1.0 if "Step" in text else 0.0
    return 0.5 * correct + 0.5 * has_steps
rubric = vf.Rubric(funcs=[reward], weights=[1.0])
'''

INFUNC_ACC = '''
import verifiers as vf
def reward(completion, answer, **kw):
    text = completion[-1]["content"]
    total = 0.0
    if text.strip().endswith(answer):
        total += 1.0
    if "Step 1" in text:
        total += 0.5
    if "Therefore" in text:
        total += 0.5
    return total
rubric = vf.Rubric(funcs=[reward], weights=[1.0])
'''

TOOL_BONUS = '''
import verifiers as vf
def correct(completion, answer, **kw):
    return 1.0 if answer in completion[-1]["content"] else 0.0
def tool_bonus(completion, **kw):
    calls = [m for m in completion if m.get("tool_calls")]
    return 0.1 * len(calls)
rubric = vf.Rubric(funcs=[correct, tool_bonus], weights=[1.0, 1.0])
'''


class AggregationV05(unittest.TestCase):
    rid = "aggregation-without-gate"

    def test_rubric_aux_terms_outweigh_correctness(self):
        f = fire(self.rid, RUBRIC_AUX)
        self.assertEqual(len(f), 1)
        self.assertIn("evidence_usage", f[0].message)
        self.assertEqual(f[0].confidence, "review")

    def test_in_function_weighted_sum(self):
        f = fire(self.rid, INFUNC_SUM)
        self.assertEqual(len(f), 1)
        self.assertIn("has_steps", f[0].message)

    def test_in_function_accumulator(self):
        self.assertEqual(len(fire(self.rid, INFUNC_ACC)), 1)

    def test_uncapped_tool_call_bonus(self):
        f = fire(self.rid, TOOL_BONUS)
        self.assertEqual(len(f), 1)
        self.assertIn("tool call", f[0].message)

    def test_correctness_heavier(self):
        self.assertEqual(fire(self.rid, RUBRIC_AUX.replace("weights=[0.4, 0.3, 0.3]", "weights=[0.8, 0.1, 0.1]")), [])

    def test_in_function_correctness_heavier(self):
        self.assertEqual(fire(self.rid, INFUNC_SUM.replace("0.5 * correct + 0.5 * has_steps",
                                                            "0.8 * correct + 0.2 * has_steps")), [])

    def test_in_function_gated(self):
        code = INFUNC_SUM.replace("    has_steps =", "    if not correct:\n        return 0.0\n    has_steps =")
        self.assertEqual(fire(self.rid, code), [])

    def test_in_function_product_is_a_gate(self):
        self.assertEqual(fire(self.rid, INFUNC_SUM.replace("0.5 * correct + 0.5 * has_steps", "correct * has_steps")), [])

    def test_capped_tool_bonus_lighter_than_task(self):
        code = TOOL_BONUS.replace("0.1 * len(calls)", "min(1.0, len(calls) / 5)").replace("weights=[1.0, 1.0]",
                                                                                         "weights=[1.0, 0.2]")
        self.assertEqual(fire(self.rid, code), [])

    def test_aux_reading_env_state_is_not_flagged(self):
        code = RUBRIC_AUX.replace('def evidence_usage(completion, **kw):\n    text = completion[-1]["content"]',
                                  'def evidence_usage(completion, state, **kw):\n    text = state["log"]')
        code = code.replace('def reasoning_quality(completion, **kw):\n    text = completion[-1]["content"]',
                            'def reasoning_quality(completion, state, **kw):\n    text = state["log"]')
        self.assertEqual(fire(self.rid, code), [])

    def test_penalty_terms_do_not_count(self):
        code = '''
import verifiers as vf
def win(completion, answer, **kw):
    return 1.0 if answer in completion[-1]["content"] else 0.0
def invalid_penalty(completion, **kw):
    return -0.1 * min(completion[-1]["content"].count("bad"), 3)
def long_penalty(completion, **kw):
    return -0.01 * len(completion)
rubric = vf.Rubric(funcs=[win, invalid_penalty, long_penalty], weights=[1.0, 1.0, 1.0])
'''
        self.assertEqual(fire(self.rid, code), [])

    def test_format_only_aux_is_left_to_format_heavy(self):
        code = '''
import verifiers as vf
def acc(completion, answer, **kw):
    return 1.0 if completion[-1]["content"] == answer else 0.0
def format_reward(completion, **kw):
    return 1.0 if "<answer>" in completion[-1]["content"] else 0.0
rubric = vf.Rubric(funcs=[acc, format_reward], weights=[1.0, 1.0])
'''
        self.assertEqual(fire(self.rid, code), [])


# ---------------------------------------------------------------------------
# v0.5: reward-trusts-agent-claim widening (items 2 and 4)
# ---------------------------------------------------------------------------

JUDGE_SUMMARY = '''
import verifiers as vf
JUDGE = """You are grading an agent. Decide whether the agent completed the task.
Task: {task}
Agent trace: {trace}
Agent summary: {summary}
Answer yes or no."""
async def judge_done(completion, state, **kw):
    summary = completion[-1]["content"]
    prompt = JUDGE.format(task=state["task"], trace=state["trace"], summary=summary)
    out = await client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return 1.0 if "yes" in out.choices[0].message.content.lower() else 0.0
rubric = vf.Rubric(funcs=[judge_done], weights=[1.0])
'''

ENV_SCORE_LINE = '''
import re, verifiers as vf
class Game(vf.MultiTurnEnv):
    async def env_response(self, messages, state, **kw):
        if state["over"]:
            return [{"role": "user", "content": f"Game over. Final Score: {state['score']}"}], state
        return [{"role": "user", "content": "Your move."}], state
def score_reward(state, **kw):
    text = " ".join(m["content"] for m in state["trajectory"])
    m = re.search(r"Final Score: (\\d+)", text)
    return float(m.group(1)) / 100 if m else 0.0
rubric = vf.Rubric(funcs=[score_reward], weights=[1.0])
'''


class TrustsClaimV05(unittest.TestCase):
    rid = "reward-trusts-agent-claim"

    def test_judge_reads_policy_summary(self):
        f = fire(self.rid, JUDGE_SUMMARY)
        self.assertEqual(len(f), 1)
        self.assertIn("summary", f[0].message)

    def test_judge_reads_final_message_fstring(self):
        code = JUDGE_SUMMARY.replace("prompt = JUDGE.format(task=state[\"task\"], trace=state[\"trace\"], summary=summary)",
                                     "final_message = summary\n    prompt = f\"Judge whether the agent completed the task. "
                                     "Trace: {state['trace']} Final message: {final_message}\"")
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_judge_reads_report_from_tool_state(self):
        code = JUDGE_SUMMARY.replace('summary = completion[-1]["content"]', 'summary = state["report"]')
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_judge_without_summary_slot(self):
        code = JUDGE_SUMMARY.replace("Agent summary: {summary}\n", "").replace(", summary=summary", "")
        self.assertEqual(fire(self.rid, code), [])

    def test_qa_judge_of_the_answer_is_not_a_self_report(self):
        code = JUDGE_SUMMARY.replace("Decide whether the agent completed the task.", "Compare the answer to the gold.")
        code = code.replace("Task: {task}", "Question: {task}")
        self.assertEqual(fire(self.rid, code), [])

    def test_judge_unpaid(self):
        self.assertEqual(fire(self.rid, JUDGE_SUMMARY.replace("weights=[1.0]", "weights=[0.0]")), [])

    def test_score_line_forged_over_all_messages(self):
        f = fire(self.rid, ENV_SCORE_LINE)
        self.assertEqual(len(f), 1)
        self.assertIn("Final Score", f[0].message)

    def test_score_line_forged_str_of_completion(self):
        code = ENV_SCORE_LINE.replace('" ".join(m["content"] for m in state["trajectory"])',
                                      'str(state["prompt"] + state["completion"])')
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_score_line_forged_findall(self):
        code = ENV_SCORE_LINE.replace("m = re.search(", "ms = re.findall(").replace(
            "return float(m.group(1)) / 100 if m else 0.0", "return float(ms[-1]) / 100 if ms else 0.0")
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_score_line_read_from_env_messages_only(self):
        code = ENV_SCORE_LINE.replace('for m in state["trajectory"])',
                                      'for m in state["trajectory"] if m["role"] == "user")')
        self.assertEqual(fire(self.rid, code), [])

    def test_score_line_the_env_never_writes(self):
        code = ENV_SCORE_LINE.replace("Game over. Final Score: {state['score']}", "Game over.")
        self.assertEqual(fire(self.rid, code), [])

    def test_score_line_unpaid(self):
        self.assertEqual(fire(self.rid, ENV_SCORE_LINE.replace("weights=[1.0]", "weights=[0.0]")), [])


# ---------------------------------------------------------------------------
# v0.5: grader-or-chain (item 9)
# ---------------------------------------------------------------------------

OR_CHAIN = '''
import verifiers as vf
async def correct(completion, answer, **kw):
    pred = completion[-1]["content"]
    if exact_match(pred, answer) or await judge_equivalent(pred, answer):
        return 1.0
    return 0.0
rubric = vf.Rubric(funcs=[correct], weights=[1.0])
'''


class OrChain(unittest.TestCase):
    rid = "grader-or-chain"

    def test_exact_or_judge(self):
        f = fire(self.rid, OR_CHAIN)
        self.assertEqual(len(f), 1)
        self.assertIn("judge_equivalent", f[0].message)

    def test_return_any_of_graders(self):
        code = OR_CHAIN.replace("if exact_match(pred, answer) or await judge_equivalent(pred, answer):\n        return 1.0\n"
                                "    return 0.0",
                                "return float(any([exact_match(pred, answer), fuzzy_ratio(pred, answer) > 0.6]))")
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_max_of_graders(self):
        code = OR_CHAIN.replace("if exact_match(pred, answer) or await judge_equivalent(pred, answer):\n        return 1.0\n"
                                "    return 0.0", "return max(math_verify(pred, answer), similarity_score(pred, answer))")
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_two_strict_graders(self):
        code = OR_CHAIN.replace("await judge_equivalent(pred, answer)", "math_verify(pred, answer)")
        self.assertEqual(fire(self.rid, code), [])

    def test_and_chain(self):
        self.assertEqual(fire(self.rid, OR_CHAIN.replace(" or await", " and await")), [])

    def test_or_chain_unpaid(self):
        self.assertEqual(fire(self.rid, OR_CHAIN.replace("weights=[1.0]", "weights=[0.0]")), [])

    def test_or_with_plain_flag(self):
        code = OR_CHAIN.replace("exact_match(pred, answer) or await judge_equivalent(pred, answer)",
                                "done or await judge_equivalent(pred, answer)")
        self.assertEqual(fire(self.rid, code), [])


# ---------------------------------------------------------------------------
# v0.5: count-reward-no-dedup (item 10)
# ---------------------------------------------------------------------------

CITE_COUNT = '''
import re, verifiers as vf
def citation_reward(completion, info, **kw):
    text = completion[-1]["content"]
    cites = re.findall(r"\\[(\\d+)\\]", text)
    return sum(1 for c in cites if c in info["valid_ids"]) / len(info["valid_ids"])
rubric = vf.Rubric(funcs=[citation_reward], weights=[1.0])
'''

LIST_THRESHOLD = '''
import verifiers as vf
def enough_items(completion, **kw):
    items = completion[-1]["content"].splitlines()
    if len(items) >= 5:
        return 1.0
    return 0.0
rubric = vf.Rubric(funcs=[enough_items], weights=[1.0])
'''

HARBOR_COUNT = {
    "task.toml": "[verifier]\ntimeout_sec = 60\n",
    "tests/test.sh": "#!/bin/bash\npytest /tests/test_outputs.py\n",
    "tests/test_outputs.py": '''
import json
def test_enough_records():
    with open("/app/out.json") as f:
        rows = json.load(f)
    assert len(rows) >= 10
''',
}


class CountNoDedup(unittest.TestCase):
    rid = "count-reward-no-dedup"

    def test_matches_over_policy_items_by_gold_count(self):
        f = fire(self.rid, CITE_COUNT)
        self.assertEqual(len(f), 1)
        self.assertIn("cites", f[0].message)

    def test_length_threshold(self):
        self.assertEqual(len(fire(self.rid, LIST_THRESHOLD)), 1)

    def test_harbor_length_assert(self):
        p = pkg(HARBOR_COUNT)
        self.assertEqual(len(RULES[self.rid]["fn"](p)), 1)

    def test_dedup_with_set(self):
        self.assertEqual(fire(self.rid, CITE_COUNT.replace("cites = re.findall", "cites = set(re.findall").replace(
            ", text)\n", ", text))\n", 1)), [])

    def test_precision_over_same_list(self):
        self.assertEqual(fire(self.rid, CITE_COUNT.replace('/ len(info["valid_ids"])', "/ len(cites)")), [])

    def test_threshold_unpaid(self):
        self.assertEqual(fire(self.rid, LIST_THRESHOLD.replace("weights=[1.0]", "weights=[0.0]")), [])

    def test_harbor_sibling_dedup_test(self):
        files = dict(HARBOR_COUNT)
        files["tests/test_outputs.py"] += '''
def test_no_duplicates():
    rows = json.load(open("/app/out.json"))
    assert len(rows) == len(set(r["id"] for r in rows))
'''
        self.assertEqual(RULES[self.rid]["fn"](pkg(files)), [])

    def test_recall_over_gold(self):
        code = CITE_COUNT.replace('sum(1 for c in cites if c in info["valid_ids"])',
                                  'sum(1 for g in info["valid_ids"] if g in cites)')
        self.assertEqual(fire(self.rid, code), [])

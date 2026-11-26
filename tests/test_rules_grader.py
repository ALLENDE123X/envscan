"""Tests for the grader family rules (envscan/rules_grader.py).

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


def pkg(files: dict) -> Package:
    d = tempfile.mkdtemp(prefix="envscan-grader-")
    try:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(text).lstrip("\n"))
        return Package.from_dir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def fire(rid, files):
    return RULES[rid]["fn"](pkg(files))


PYPROJECT = '[project]\nname = "x"\ndependencies = ["verifiers>=0.1.3"]\n'


def env(code: str, name="env.py", extra=None):
    files = {"pyproject.toml": PYPROJECT, name: code}
    files.update(extra or {})
    return files


class Registered(unittest.TestCase):
    def test_rules_registered(self):
        for rid, kind in (("grader-fails-open", "hack"), ("formal-checker-escape-hatch", "hack"),
                          ("judge-forgeable-delimiter", "hygiene"), ("lenient-answer-match", "hack")):
            self.assertIn(rid, RULES)
            self.assertEqual(RULES[rid]["kind"], kind)
            desc = RULES[rid]["description"]
            self.assertNotIn(";", desc)
            self.assertNotIn("—", desc)

    def test_odd_input_does_not_crash(self):
        p = pkg({"pyproject.toml": PYPROJECT, "broken.py": "def f(:\n  return 1\n", "empty.py": "",
                 "tests/test.sh": "#!/bin/bash\n", "task.toml": "[verifier]\n", "tests/x.py": "\x00\x01 not python",
                 "nested/tests/test_y.py": "def reward(:\n"})
        findings, errors = run_rules(p, only={"grader-fails-open", "formal-checker-escape-hatch",
                                              "judge-forgeable-delimiter", "lenient-answer-match"})
        self.assertEqual(errors, [])
        self.assertEqual(findings, [])


# ---------------------------------------------------------------------------
# grader-fails-open
# ---------------------------------------------------------------------------

class GraderFailsOpen(unittest.TestCase):
    R = "grader-fails-open"

    def test_judge_exception_pays_half(self):
        f = fire(self.R, env('''
            import verifiers as vf
            async def judge_reward(prompt, completion, answer, judge=None, **kw):
                try:
                    out = await judge(prompt, completion, answer)
                    return 1.0 if "yes" in out.lower() else 0.0
                except Exception:
                    return 0.5
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[judge_reward]))
        '''))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].confidence, "review")
        self.assertIn("env.py:7", f[0].evidence)
        self.assertIn("raises", f[0].message)

    def test_unparsable_judge_score_falls_through_to_half(self):
        f = fire(self.R, env('''
            import re
            def _parse_judge_score(response):
                m = re.match(r"^\\s*(\\d+\\.?\\d*)", response)
                if m:
                    return min(1.0, float(m.group(1)))
                return 0.5
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("env.py:6", f[0].evidence)
        self.assertIn("cannot be parsed", f[0].message)

    def test_default_score_on_missing_judge_match(self):
        f = fire(self.R, env('''
            import re
            def _extract_score(judge_response):
                score_match = re.search(r"#score:.*?([1-5])", judge_response)
                return int(score_match.group(1)) if score_match else 5
        '''))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].severity, "high")  # 5 is the top of the scale it returns

    def test_judge_reply_without_score_field_defaults(self):
        f = fire(self.R, env('''
            import json
            async def judge_score(prompt, completion, client, **kw):
                reply = await client.chat.completions.create(messages=[])
                parsed = json.loads(reply.choices[0].message.content)
                return parsed.get("score", 3) / 5
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("no score field", f[0].message)

    def test_score_default_survives_swallowed_exception(self):
        f = fire(self.R, env('''
            def grade(completion, answer, judge):
                score = 0.5
                try:
                    score = float(judge(completion, answer))
                except Exception:
                    pass
                return score
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("env.py:2", f[0].evidence)

    def test_verdict_chain_unmatched_pays(self):
        f = fire(self.R, env('''
            async def correctness_reward(completion, answer, judge_client):
                verdict = await judge_client.grade(completion, answer)
                if "INCORRECT" in verdict:
                    return 0.0
                elif "CORRECT" in verdict:
                    return 1.0
                return 0.5
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("matches no verdict", f[0].message)

    def test_harbor_verifier_writes_reward_on_error(self):
        f = fire(self.R, {
            "task.toml": "[verifier]\ntimeout_sec = 60\n",
            "instruction.md": "Write /app/answer.txt\n",
            "tests/test.sh": "#!/bin/bash\npython3 /tests/judge.py\n",
            "tests/judge.py": '''
                from pathlib import Path
                def grade():
                    try:
                        verdict = call_judge(Path("/app/answer.txt").read_text())
                        Path("/logs/verifier/reward.txt").write_text("1" if verdict else "0")
                    except Exception:
                        Path("/logs/verifier/reward.txt").write_text("1")
            '''})
        self.assertEqual(len(f), 1)
        self.assertIn("tests/judge.py:7", f[0].evidence)

    # negatives

    def test_fail_closed_exception(self):
        self.assertEqual(fire(self.R, env('''
            async def judge_reward(completion, answer, judge, **kw):
                try:
                    return float(await judge(completion, answer))
                except Exception:
                    return 0.0
        ''')), [])

    def test_fail_closed_parse_fallthrough(self):
        self.assertEqual(fire(self.R, env('''
            import re
            def parse_judge_score(response):
                m = re.search(r"score: (\\d)", response)
                if m:
                    return int(m.group(1)) / 5
                return 0.0
        ''')), [])

    def test_missing_reference_is_not_a_grader_failure(self):
        self.assertEqual(fire(self.R, env('''
            def accuracy_score(completion, info, **kw):
                expected = info.get("expected")
                if not expected:
                    return 0.5
                return float(expected in completion)
        ''')), [])

    def test_exit_status_is_not_a_reward(self):
        self.assertEqual(fire(self.R, {
            "task.toml": "[verifier]\n",
            "tests/test.sh": "#!/bin/bash\npython3 /tests/verify.py\n",
            "tests/verify.py": '''
                import sys
                def main():
                    try:
                        check()
                    except Exception:
                        return 1
                    return 0
                if __name__ == "__main__":
                    sys.exit(main())
            '''}), [])

    def test_both_empty_counts_as_match(self):
        self.assertEqual(fire(self.R, env('''
            def payback_score(agent, correct):
                if correct is None and agent is None:
                    return 1.0
                return float(agent == correct)
        ''')), [])

    def test_empty_denominator_is_left_to_verdict_parse_semantics(self):
        self.assertEqual(fire(self.R, env('''
            def pass_fraction_reward(results):
                if not results:
                    return 1.0
                return sum(1 for r in results if r) / len(results)
        ''')), [])

    def test_small_format_credit_for_unparsed_answer(self):
        self.assertEqual(fire(self.R, env('''
            import re
            def move_reward(completion, answer):
                m = re.search(r"[a-h][1-8][a-h][1-8]", completion)
                if m:
                    return float(m.group(0) == answer)
                return 0.1
        ''')), [])

    def test_helper_script_is_not_the_grader(self):
        self.assertEqual(fire(self.R, env('''
            def judge_score(reply):
                try:
                    return float(reply)
                except ValueError:
                    return 0.5
        ''', name="analysis/pilot.py")), [])


# ---------------------------------------------------------------------------
# formal-checker-escape-hatch
# ---------------------------------------------------------------------------

LEAN_BAN_GRADER = '''
import subprocess
FORBIDDEN = ["sorry", "axiom", "admit"]
def proof_reward(completion, info, **kw):
    proof = completion[-1]["content"]
    if any(tok in proof for tok in FORBIDDEN):
        return 0.0
    open("Main.lean", "w").write(proof)
    r = subprocess.run(["lake", "env", "lean", "Main.lean"], capture_output=True, text=True)
    return 1.0 if r.returncode == 0 else 0.0
'''


class FormalEscapeHatch(unittest.TestCase):
    R = "formal-checker-escape-hatch"

    def test_lean_ban_list_misses_kernel_skip_and_statement(self):
        f = fire(self.R, env(LEAN_BAN_GRADER))
        self.assertEqual(len(f), 1)
        self.assertIn("skipKernelTC", f[0].message)
        self.assertIn("native_decide", f[0].message)
        self.assertIn("statement", f[0].message)
        # the existing rule stays silent here because a ban list exists
        self.assertEqual(RULES["lean-no-axiom-check"]["fn"](pkg(env(LEAN_BAN_GRADER))), [])

    def test_lean_word_boundary_sorry_misses_sorryax(self):
        f = fire(self.R, env('''
            import re, subprocess
            def lean_check(proof, formal_statement):
                if formal_statement not in proof:
                    return 0.0
                if re.search(r"\\bsorry\\b", proof) or "axiom " in proof:
                    return 0.0
                r = subprocess.run(f"lake env lean {path}", shell=True)
                return float(r.returncode == 0)
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("sorryAx", f[0].message)
        self.assertNotIn("changed theorem statement", f[0].message)

    def test_dafny_exit_status_with_no_audit(self):
        f = fire(self.R, env('''
            import subprocess
            def verify_reward(completion, **kw):
                open("prog.dfy", "w").write(completion)
                r = subprocess.run(["dafny", "verify", "prog.dfy"], capture_output=True)
                return float(r.returncode == 0)
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("Dafny", f[0].message)
        self.assertIn("assume", f[0].message)

    def test_verus_ban_list_misses_external_body(self):
        f = fire(self.R, env('''
            import subprocess
            BANNED = ["assume(", "admit("]
            def verus_reward(code):
                if any(b in code for b in BANNED):
                    return 0.0
                return float(subprocess.run(["verus", "main.rs"]).returncode == 0)
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("external_body", f[0].message)

    # negatives

    def test_lean_with_no_audit_is_left_to_lean_no_axiom_check(self):
        code = '''
            import subprocess
            def proof_reward(proof, **kw):
                open("Main.lean", "w").write(proof)
                r = subprocess.run(["lake", "env", "lean", "Main.lean"])
                return float(r.returncode == 0)
        '''
        self.assertEqual(fire(self.R, env(code)), [])
        self.assertEqual(len(RULES["lean-no-axiom-check"]["fn"](pkg(env(code)))), 1)

    def test_lean_full_audit(self):
        self.assertEqual(fire(self.R, env('''
            import subprocess
            ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
            FORBIDDEN = ["sorry", "set_option", "notation", "macro_rules", "elab", "#exit"]
            def proof_reward(proof, formal_statement, **kw):
                if not proof.startswith(formal_statement):
                    return 0.0
                if any(t in proof for t in FORBIDDEN):
                    return 0.0
                out = subprocess.run(["lake", "env", "lean", "Main.lean"], capture_output=True, text=True).stdout
                return float("#print axioms" in out and all(a in ALLOWED_AXIOMS for a in parse_axioms(out)))
        ''')), [])

    def test_dafny_with_full_audit(self):
        self.assertEqual(fire(self.R, env('''
            import subprocess
            BANNED = ["assume", "{:axiom}", "{:verify false}", "{:extern}", "expect", "decreases *"]
            def verify_reward(completion, spec, **kw):
                if spec_unchanged(completion, spec) is False or any(b in completion for b in BANNED):
                    return 0.0
                r = subprocess.run(["dafny", "verify", "prog.dfy"])
                return float(r.returncode == 0)
            def spec_unchanged(code, spec):
                return spec in code
        ''')), [])

    def test_lean_named_words_are_not_a_checker(self):
        self.assertEqual(fire(self.R, env('''
            def clean_reward(completion, answer):
                """Keep the reward lean: no sorry state, no axioms, just a clean match."""
                return float(completion.strip() == answer)
        ''')), [])


# ---------------------------------------------------------------------------
# judge-forgeable-delimiter
# ---------------------------------------------------------------------------

class JudgeForgeableDelimiter(unittest.TestCase):
    R = "judge-forgeable-delimiter"

    def test_xml_tag(self):
        f = fire(self.R, env('''
            JUDGE_TEMPLATE = """Grade the answer.
            <question>{question}</question>
            <response>{response}</response>
            Reply CORRECT or INCORRECT."""
        '''))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "hygiene")
        self.assertIn("<response> tag", f[0].message)

    def test_code_fence_with_labels_inside(self):
        f = fire(self.R, env('''
            GRADER_TEMPLATE = """Judge whether the predicted answer is correct.
            ```
            Question: {question}
            Gold target: {target}
            Predicted answer: {response}
            ```
            Grade as A (correct) or B (incorrect)."""
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("code fence", f[0].message)

    def test_begin_end_data_markers_in_harbor_fstring(self):
        f = fire(self.R, {
            "task.toml": "[verifier]\n",
            "tests/test.sh": "#!/bin/bash\npython3 /tests/llm_judge.py\n",
            "tests/llm_judge.py": '''
                def grade(question, answer):
                    prompt = f"""You are assessing a submitted answer. Score it from 0 to 10.
                [BEGIN DATA]
                [Question]: {question}
                [END DATA]
                [BEGIN DATA]
                {answer}
                [END DATA]
                Does the submission meet the criterion? Reply GRADE: C or GRADE: I."""
                    return call(prompt)
            '''.replace("{answer}", "{answer_text}")})
        self.assertEqual(len(f), 1)
        self.assertIn("tests/llm_judge.py:2", f[0].evidence)

    # negatives

    def test_nonce_fence(self):
        self.assertEqual(fire(self.R, env('''
            import secrets
            JUDGE_TEMPLATE = """Grade the answer.
            <response>{response}</response>
            Reply CORRECT or INCORRECT."""
            def judge_prompt(r):
                nonce = secrets.token_hex(8)
                return JUDGE_TEMPLATE.replace("response>", f"response-{nonce}>").format(response=r)
        ''')), [])

    def test_undelimited_is_the_older_rules_case(self):
        files = env('''
            JUDGE_PROMPT = """Rate the answer from 1 to 10.
            Question: {question}
            Answer: {response}
            Give only the number."""
        ''')
        self.assertEqual(fire(self.R, files), [])
        self.assertEqual(len(RULES["judge-prompt-undelimited"]["fn"](pkg(files))), 1)

    def test_closer_stripped_from_response(self):
        self.assertEqual(fire(self.R, env('''
            JUDGE_TEMPLATE = """Grade the answer.
            <response>{response}</response>
            Reply CORRECT or INCORRECT."""
            def build(r):
                return JUDGE_TEMPLATE.format(response=r.replace("</response>", ""))
        ''')), [])

    def test_not_a_judge_template(self):
        self.assertEqual(fire(self.R, env('''
            SYSTEM = """Answer inside tags.
            <response>{response}</response>
            Then stop."""
        ''')), [])


# ---------------------------------------------------------------------------
# lenient-answer-match
# ---------------------------------------------------------------------------

class LenientAnswerMatch(unittest.TestCase):
    R = "lenient-answer-match"

    def test_gold_in_completion(self):
        f = fire(self.R, env('''
            def exact_match_reward(completion, answer, **kw):
                content = completion[-1]["content"]
                if content.strip() == str(answer):
                    return 1.0
                elif str(answer).lower() in content.lower():
                    return 0.7
                return 0.0
        '''))
        self.assertEqual(len(f), 1)
        self.assertIn("env.py:5", f[0].evidence)

    def test_regex_escape_of_gold(self):
        f = fire(self.R, env('''
            import re
            def flag_reward(completion, answer):
                text = completion[-1]["content"]
                return 1.0 if re.search(rf"\\b{re.escape(answer)}\\b", text) else 0.0
        '''))
        self.assertEqual(len(f), 1)

    def test_any_alias_in_response(self):
        f = fire(self.R, env('''
            def correctness_reward(response, info):
                answers = info["aliases"]
                return float(any(a.lower() in response.lower() for a in answers))
        '''))
        self.assertEqual(len(f), 1)

    # negatives

    def test_extracted_final_answer(self):
        self.assertEqual(fire(self.R, env('''
            def correct_answer_reward(completion, answer, parser):
                text = parser.parse_answer(completion)
                return 1.0 if answer in text else 0.0
        ''')), [])

    def test_exact_match(self):
        self.assertEqual(fire(self.R, env('''
            def correct_answer_reward(completion, answer):
                response = completion[-1]["content"].strip()
                return float(response == answer)
        ''')), [])

    def test_keyword_list_is_keyword_reward(self):
        self.assertEqual(fire(self.R, env('''
            def coverage_reward(completion, info):
                text = completion[-1]["content"].lower()
                target_keywords = info["target_keywords"]
                return sum(1 for kw in target_keywords if kw in text) / len(target_keywords)
        ''')), [])

    def test_response_inside_gold_is_not_this_rule(self):
        self.assertEqual(fire(self.R, env('''
            def reward(completion, answer):
                text = completion[-1]["content"]
                return float(text in answer)
        ''')), [])


# ---------------------------------------------------------------------------
# 2026-11-24 tightening after the out of sample check (the internal notes)
# ---------------------------------------------------------------------------

class GraderFailsOpenTightened(unittest.TestCase):
    R = "grader-fails-open"

    # (1) only an except or no parse branch around the judge reply or the model output
    def test_gold_parse_failure_is_not_a_grader_failure(self):
        self.assertEqual(fire(self.R, env("""
            import json
            def numeric_reward(completion, answer):
                text = completion[-1]["content"]
                try:
                    gold = json.loads(answer)
                except (json.JSONDecodeError, TypeError):
                    return 0.5
                return float(str(gold["value"]) in text)
        """)), [])

    def test_gold_regex_parse_failure_is_not_a_grader_failure(self):
        self.assertEqual(fire(self.R, env("""
            import re
            def move_reward(completion, answer):
                m = re.match(r"(\\d+),(\\d+)", answer)
                return float(m.group(0) == completion) if m else 0.5
        """)), [])

    def test_judge_reply_parse_failure_still_fires(self):
        f = fire(self.R, env("""
            import json
            async def rubric_score(completion, answer, client):
                reply = await client.chat.completions.create(messages=[])
                try:
                    obj = json.loads(reply.choices[0].message.content)
                    return float(obj["score"]) / 10
                except (json.JSONDecodeError, KeyError):
                    return 0.5
        """))
        self.assertEqual(len(f), 1)

    def test_model_output_parse_failure_still_fires(self):
        f = fire(self.R, env("""
            def numeric_reward(completion, answer):
                text = completion[-1]["content"]
                try:
                    value = float(text.strip())
                except ValueError:
                    return 0.5
                return float(abs(value - float(answer)) < 1e-6)
        """))
        self.assertEqual(len(f), 1)

    # (2) a plain early `return True` in a check function is its pass value
    def test_early_return_true_in_check(self):
        self.assertEqual(fire(self.R, env("""
            import re
            def check_cron_disabled(crontab_text, script):
                jobs = re.findall(r"^.*" + script + r".*$", crontab_text)
                if not jobs:
                    return True, "no cron entry runs the script"
                return False, "still scheduled"
        """)), [])

    def test_true_from_a_detector_of_bad_things_fails_closed(self):
        self.assertEqual(fire(self.R, env("""
            import os
            def check_escapes_root(candidate, root):
                try:
                    real = os.path.realpath(candidate)
                except OSError:
                    return True
                return not real.startswith(root)
        """)), [])

    def test_true_from_a_pass_check_on_an_exception_still_fires(self):
        f = fire(self.R, env("""
            import langdetect
            class LanguageChecker:
                def check_following(self, value):
                    try:
                        return langdetect.detect(value) == self._language
                    except langdetect.LangDetectException:
                        return True
        """))
        self.assertEqual(len(f), 1)

    # (3) the fallback must pay more than the wrong answer, and at least 0.1
    def test_fallback_equal_to_wrong_answer_floor(self):
        self.assertEqual(fire(self.R, env("""
            def grid_score(completion, answer):
                text = completion[-1]["content"]
                try:
                    grid = [int(c) for c in text.split()]
                    if grid == [int(c) for c in answer.split()]:
                        return 1.0
                except ValueError:
                    return 0.1
                return 0.1
        """)), [])

    def test_tiny_fallback(self):
        self.assertEqual(fire(self.R, env("""
            import re
            async def judge_reward(completion, answer, judge_client):
                judgment = await judge_client.judge(completion, answer)
                try:
                    return answer == re.findall(r"box\\{(.*?)\\}", judgment)[0]
                except IndexError:
                    return 0.01
        """)), [])

    def test_fallback_above_partial_credit_floor_still_fires(self):
        f = fire(self.R, env("""
            async def quality_reward(completion, judge_client):
                try:
                    verdict = await judge_client.grade(completion)
                except Exception:
                    return 0.6
                if verdict == "good":
                    return 1.0
                return 0.2
        """))
        self.assertEqual(len(f), 1)

    # (4) weight 0 (logged only), negative weights and metrics pay nothing
    ZERO = """
        import verifiers as vf
        async def main_reward(completion, answer):
            return float(completion == answer)
        async def judge_logged(completion, judge):
            try:
                return float(await judge(completion))
            except Exception:
                return 0.5
        def load_environment():
            return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[main_reward, judge_logged], weights=[1.0, WEIGHT]))
    """

    def test_zero_weight_function(self):
        self.assertEqual(fire(self.R, env(self.ZERO.replace("WEIGHT", "0.0"))), [])

    def test_positive_weight_function_still_fires(self):
        self.assertEqual(len(fire(self.R, env(self.ZERO.replace("WEIGHT", "1.0")))), 1)

    def test_add_reward_func_zero_and_metric(self):
        self.assertEqual(fire(self.R, env("""
            import verifiers as vf
            async def judge_logged(completion, judge):
                try:
                    return float(await judge(completion))
                except Exception:
                    return 0.5
            @vf.metric
            async def judge_metric(completion, judge):
                try:
                    return float(await judge(completion))
                except Exception:
                    return 0.5
            def load_environment():
                rubric = vf.Rubric()
                rubric.add_reward_func(judge_logged, weight=0.0)
                return vf.SingleTurnEnv(rubric=rubric)
        """)), [])

    def test_tiny_share_of_the_total_reward(self):
        self.assertEqual(fire(self.R, env("""
            import verifiers as vf
            async def main_reward(completion, answer):
                return float(completion == answer)
            async def judge_bonus(completion, judge):
                try:
                    return float(await judge(completion))
                except Exception:
                    return 0.5
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[main_reward, judge_bonus], weights=[0.95, 0.05]))
        """)), [])


class LenientAnswerMatchTightened(unittest.TestCase):
    R = "lenient-answer-match"

    # (1) the match must raise the reward
    def test_match_that_fails_the_check(self):
        self.assertEqual(fire(self.R, env("""
            def give_up_check(completion, answer, state):
                reply = completion[-1]["content"]
                if answer in reply:
                    state["passed"] = False
                    return 0.0
                return 1.0
        """)), [])

    def test_match_that_raises(self):
        self.assertEqual(fire(self.R, env("""
            def reward(completion, reference):
                text = completion[-1]["content"]
                if reference in text:
                    raise ValueError("copied the reference")
                return 1.0
        """)), [])

    def test_match_counted_as_a_hit_still_fires(self):
        f = fire(self.R, env("""
            def recall_reward(completion, expected_needles):
                response = completion[-1]["content"]
                found = []
                for needle in expected_needles:
                    if needle in response:
                        found.append(needle)
                return len(found) / len(expected_needles)
        """))
        self.assertEqual(len(f), 1)

    def test_count_compared_then_paid_still_fires(self):
        f = fire(self.R, env("""
            def correct(completion, answer):
                text = completion[-1]["content"]
                count = text.count(answer)
                if count == 1:
                    return 1.0
                return 0.0
        """))
        self.assertEqual(len(f), 1)

    # (2) zero weight
    def test_zero_weight_function(self):
        self.assertEqual(fire(self.R, env("""
            import verifiers as vf
            def task_accuracy(completion, answer):
                response = completion[-1]["content"]
                return 1.0 if answer in response else 0.0
            def resistance(completion):
                return 1.0
            def load_environment():
                rubric = vf.Rubric()
                rubric.add_reward_func(resistance, weight=1.0)
                rubric.add_reward_func(task_accuracy, weight=0.0)
                return vf.SingleTurnEnv(rubric=rubric)
        """)), [])

    def test_positive_weight_function_still_fires(self):
        f = fire(self.R, env("""
            import verifiers as vf
            def task_accuracy(completion, answer):
                response = completion[-1]["content"]
                return 1.0 if answer in response else 0.0
            def load_environment():
                rubric = vf.Rubric()
                rubric.add_reward_func(task_accuracy, weight=1.0)
                return vf.SingleTurnEnv(rubric=rubric)
        """))
        self.assertEqual(len(f), 1)

    # (3) no hidden answer: the gold is shown in the task, or is fixed text in the code
    DOC = """
        DATA = [
            {"code": "def merge_sort(xs):\\n    return sorted(xs)", "answer": "merge_sort"},
            {"code": "class LRUCache:\\n    pass", "answer": "LRUCache"},
            {"code": "def binary_search(xs, x):\\n    return 0", "answer": "binary_search"},
            {"code": "def flatten_list(xs):\\n    return xs", "answer": "flatten_list"},
        ]
        def has_example(completion, answer):
            text = completion[-1]["content"]
            score = 0.0
            if answer in text:
                score += 0.3
            return score
        def load_environment():
            import verifiers as vf
            return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[has_example]))
    """

    def test_gold_identifier_shown_in_the_task(self):
        self.assertEqual(fire(self.R, env(self.DOC)), [])

    def test_plain_word_gold_still_fires(self):
        code = self.DOC.replace('"answer": "merge_sort"', '"answer": "cat"').replace(
            '"answer": "LRUCache"', '"answer": "dog"').replace('"answer": "binary_search"', '"answer": "bird"').replace(
            '"answer": "flatten_list"', '"answer": "fish"')
        self.assertEqual(len(fire(self.R, env(code))), 1)

    def test_fixed_marker_in_the_code(self):
        self.assertEqual(fire(self.R, env("""
            FLAG = "i cannot solve this"
            def is_correct(completion):
                reply = completion[-1]["content"]
                return FLAG in reply.lower()
        """)), [])

    # (4) HTTP response bodies and the output of code under test
    def test_http_response_body(self):
        self.assertEqual(fire(self.R, env("""
            import requests
            def functional_test(port, expected_products):
                score = 0
                r = requests.get(url=f"http://localhost:{port}/items")
                for product in expected_products:
                    if product in r.text:
                        score += 1
                return score / len(expected_products)
        """)), [])

    def test_process_output(self):
        self.assertEqual(fire(self.R, env("""
            import subprocess
            def run_tests_reward(code, expected_output):
                proc = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
                output = proc.stdout
                return 1.0 if expected_output in output else 0.0
        """)), [])

    def test_model_completion_still_fires(self):
        f = fire(self.R, env("""
            def reward(completion, expected_output):
                output = completion[-1]["content"]
                return 1.0 if expected_output in output else 0.0
        """))
        self.assertEqual(len(f), 1)


class GraderGapsV04(unittest.TestCase):
    """v0.4 fixes from the independent checks (the internal notes, the internal build log)."""
    R = "grader-fails-open"

    # gap 1: a judge that is only a metric, whose fallback reaches the paid reward through state
    STATE = {
        "judge.py": """
            class Judge:
                async def evaluate(self, completion, state):
                    state.update({"judge_score": 1.0, "judge_failure": 0.0})
                    try:
                        reply = await self.client.chat.completions.create(
                            model="m", messages=[{"role": "user", "content": str(completion)}])
                    except Exception:
                        state["judge_failure"] = 1.0
                        return 1.0
                    state["judge_score"] = parse_score(reply)
                    return state["judge_score"]
        """,
        "env.py": """
            import verifiers as vf
            from judge import Judge
            def load_environment(judge_mode="multiply"):
                judge = Judge()
                async def judge_metric(completion, state, **_):
                    return await judge.evaluate(completion, state)
                async def training_reward(state, **_):
                    base = float(state.get("deterministic", 0.0))
                    if judge_mode == "multiply":
                        return base * float(state.get("judge_score", 1.0))
                    return base
                rubric = vf.Rubric()
                rubric.add_metric(judge_metric)
                rubric.add_reward_func(training_reward, weight=1.0)
                return vf.SingleTurnEnv(rubric=rubric)
        """,
    }

    def test_metric_fallback_reaches_paid_reward_through_state(self):
        f = fire(self.R, dict(self.STATE, **{"pyproject.toml": PYPROJECT}))
        self.assertEqual(len(f), 1)
        self.assertIn("judge_score", f[0].message)
        self.assertIn("only removes a penalty", f[0].message)
        self.assertIn("judge_mode == 'multiply'", f[0].message)
        self.assertEqual(f[0].confidence, "review")

    def test_metric_fallback_added_to_paid_reward(self):
        files = dict(self.STATE, **{"pyproject.toml": PYPROJECT})
        files["env.py"] = files["env.py"].replace("base * float(state.get", "base + float(state.get")
        f = fire(self.R, files)
        self.assertEqual(len(f), 1)
        self.assertNotIn("only removes a penalty", f[0].message)

    def test_metric_fallback_that_no_paid_reward_reads(self):
        files = dict(self.STATE, **{"pyproject.toml": PYPROJECT})
        files["env.py"] = files["env.py"].replace('state.get("judge_score", 1.0)', 'state.get("other", 1.0)')
        self.assertEqual(fire(self.R, files), [])

    # gap 2: a pairwise verdict whose fallback is at or below chance
    PAIR = """
        import random
        async def pairwise_reward(completion, answer, judge_client):
            chosen_slot = "B" if random.random() < 0.5 else "A"
            reply = await judge_client.chat(completion, answer)
            winner = reply.strip() if reply.strip() in {"A", "B"} else None
            if winner is None:
                reward = FALLBACK
            elif winner == chosen_slot:
                reward = 1.0
            else:
                reward = 0.0
            return reward
    """

    def test_pairwise_tie_at_chance(self):
        self.assertEqual(fire(self.R, env(self.PAIR.replace("FALLBACK", "0.5"))), [])

    def test_pairwise_fallback_above_chance_still_fires(self):
        self.assertEqual(len(fire(self.R, env(self.PAIR.replace("FALLBACK", "0.8")))), 1)

    MCQ = """
            import re
            async def mcq_judge_reward(completion, answer, judge_client):
                reply = await judge_client.chat(completion)
                try:
                    choice = re.findall(r"[[](.)[]]", reply)[0]
                except IndexError:
                    return FALLBACK
                if choice not in ("A", "B", "C", "D"):
                    return 0.0
                return 1.0 if choice == answer else 0.0
    """

    def test_four_option_verdict_fallback_above_chance_still_fires(self):
        self.assertEqual(len(fire(self.R, env(self.MCQ.replace("FALLBACK", "0.5")))), 1)

    def test_four_option_verdict_at_chance(self):
        self.assertEqual(fire(self.R, env(self.MCQ.replace("FALLBACK", "0.25"))), [])

    def test_continuous_score_with_side_labels_is_not_a_verdict(self):
        f = fire(self.R, env("""
            import re
            async def side_score(traj, side, judge_client):
                assigned = traj.a if side == "A" else traj.b
                raw = await judge_client.generate(assigned)
                match = re.search(r"[0-9.]+", raw)
                return max(0.0, min(1.0, float(match.group(0)))) if match else 0.5
        """))
        self.assertEqual(len(f), 1)

    # gap 3: a fallback below the lowest reward any valid answer gets on the success path
    SUCCESS = """
        import json
        def score_completion(completion, gold):
            try:
                decoded = json.loads(completion)
            except json.JSONDecodeError:
                return {"reward": 0.0}
            try:
                pred = parse_schema(decoded)
                schema_valid = True
            except ValueError:
                return {"reward": FALLBACK}
            f1 = compute_f1(pred, gold)
            reward = 0.10 + 0.10 * (1.0 if schema_valid else 0.0) + 0.80 * f1
            reward = max(0.0, min(1.0, reward))
            return {"reward": reward}
    """

    def test_fallback_below_success_path_floor(self):
        self.assertEqual(fire(self.R, env(self.SUCCESS.replace("FALLBACK", "0.15"))), [])

    def test_fallback_above_success_path_floor_still_fires(self):
        self.assertEqual(len(fire(self.R, env(self.SUCCESS.replace("FALLBACK", "0.5")))), 1)

    def test_unknown_success_term_keeps_the_old_floor(self):
        code = self.SUCCESS.replace("FALLBACK", "0.15").replace("0.80 * f1", "0.80 * f1 - penalty(pred)")
        self.assertEqual(len(fire(self.R, env(code))), 1)


class LenientRawMessageList(unittest.TestCase):
    """gap 4: `x in completion` where completion is the chat message list is never true for a string."""
    R = "lenient-answer-match"

    def test_raw_message_list(self):
        self.assertEqual(fire(self.R, env("""
            def tool_reward(prompt, completion, answer, info, state):
                score = 0.0
                expected_function = info.get("expected_function", "")
                if expected_function and expected_function in completion:
                    score += 0.3
                return score
        """)), [])

    def test_joined_text_still_fires(self):
        f = fire(self.R, env("""
            def tool_reward(prompt, completion, answer, info, state):
                completion = " ".join(m["content"] for m in completion)
                expected_function = info.get("expected_function", "")
                if expected_function and expected_function in completion:
                    return 1.0
                return 0.0
        """))
        self.assertEqual(len(f), 1)

    def test_plain_text_completions_still_fire(self):
        f = fire(self.R, env("""
            import verifiers as vf
            def tool_reward(prompt, completion, answer, info, state):
                expected_function = info.get("expected_function", "")
                if expected_function and expected_function in completion:
                    return 1.0
                return 0.0
            def load_environment():
                return vf.SingleTurnEnv(message_type="completion", rubric=vf.Rubric(funcs=[tool_reward]))
        """))
        self.assertEqual(len(f), 1)

    def test_str_check_keeps_the_hit(self):
        f = fire(self.R, env("""
            def reward(completion, answer):
                if isinstance(completion, str) and answer in completion:
                    return 1.0
                return 0.0
        """))
        self.assertEqual(len(f), 1)



# ---------------------------------------------------------------------------
# v0.5: formal-checker-escape-hatch widenings (lean-source-cheat-tokens, dafny decreases *)
# ---------------------------------------------------------------------------

LEAN_V05 = """
import subprocess
banned_tokens = [{tokens}]
def lean_reward(completion, info, **kw):
    proof = completion[-1]["content"]
    statement = info["formal_statement"]
    if statement not in proof:
        return 0.0
    for t in banned_tokens:
        if t in proof:
            return 0.0
    r = subprocess.run({cmd}, capture_output=True)
    return 1.0 if r.returncode == 0 else 0.0
"""
BASE_TOKENS = '"sorry", "sorryAx", "axiom", "native_decide", "implemented_by", "set_option", "notation", "macro_rules"'
LAKE = '["lake", "env", "lean", "Main.lean"]'


class FormalV05(unittest.TestCase):
    R = "formal-checker-escape-hatch"

    def lean(self, tokens=BASE_TOKENS, cmd=LAKE):
        return fire(self.R, env(LEAN_V05.format(tokens=tokens, cmd=cmd)))

    def test_exit_missing(self):
        f = self.lean()
        self.assertEqual(len(f), 1)
        self.assertIn("#exit", f[0].message)

    def test_instance_missing_when_exit_banned(self):
        f = self.lean(BASE_TOKENS + ', "#exit"')
        self.assertEqual(len(f), 1)
        self.assertNotIn("#exit", f[0].message)
        self.assertIn("instance", f[0].message)

    def test_admit_missing_without_axiom_audit(self):
        f = self.lean('"sorry", "native_decide", "implemented_by", "set_option", "notation", "macro_rules", '
                      '"#exit", "instance", "variable", "#eval", "run_cmd", "unsafe", "partial"')
        self.assertEqual(len(f), 1)
        self.assertIn("admit", f[0].message)

    def test_all_new_tokens_banned(self):
        self.assertEqual(self.lean(BASE_TOKENS + ', "#exit", "instance", "variable", "#eval", "run_cmd", "addDecl"'), [])

    def test_kernel_replay_covers_new_tokens(self):
        self.assertEqual(self.lean(cmd='["lake", "env", "lean4checker", "Main"]'), [])

    def test_frozen_statement_moots_instance(self):
        f = fire(self.R, env(LEAN_V05.format(tokens=BASE_TOKENS + ', "#exit", "#eval", "run_cmd"', cmd=LAKE).replace(
            'if statement not in proof:\n        return 0.0', 'src = statement + proof')))
        self.assertEqual(f, [])

    def test_no_lean_no_finding(self):
        self.assertEqual(fire(self.R, env("def reward(completion, answer):\n    return float(completion == answer)\n")), [])

    DAFNY = """
import subprocess
BANNED = ["assume", "{:axiom}", "{:verify false}", "{:extern}", "expect"EXTRA]
def verify_reward(completion, spec, **kw):
    if spec_unchanged(completion, spec) is False or any(b in completion for b in BANNED):
        return 0.0
    r = subprocess.run(["dafny", "verify", "prog.dfy"])
    return float(r.returncode == 0)
def spec_unchanged(code, spec):
    return spec in code
"""

    def test_dafny_decreases_star_missing(self):
        f = fire(self.R, env(self.DAFNY.replace("EXTRA", "")))
        self.assertEqual(len(f), 1)
        self.assertIn("decreases *", f[0].message)

    def test_dafny_decreases_star_banned(self):
        self.assertEqual(fire(self.R, env(self.DAFNY.replace("EXTRA", ', "decreases *"'))), [])

    def test_dafny_decreases_regex_banned(self):
        self.assertEqual(fire(self.R, env(self.DAFNY.replace("EXTRA", ', r"decreases\\s*\\*"'))), [])


# ---------------------------------------------------------------------------
# v0.5: grader-failure-scored-zero
# ---------------------------------------------------------------------------

class GraderFailureScoredZero(unittest.TestCase):
    R = "grader-failure-scored-zero"

    def test_judge_exception_scores_zero(self):
        f = fire(self.R, env("""
            import verifiers as vf
            async def judge_reward(prompt, completion, answer, judge=None, **kw):
                try:
                    out = await judge(prompt, completion, answer)
                    return 1.0 if "yes" in out.lower() else 0.0
                except Exception:
                    return 0.0
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[judge_reward]))
        """))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "broken")
        self.assertIn("judge call fails", f[0].message)

    def test_client_api_error_scores_zero(self):
        f = fire(self.R, env("""
            import openai
            def grade_reward(completion, answer, **kw):
                client = openai.OpenAI()
                try:
                    r = client.chat.completions.create(model="m", messages=[{"role": "user", "content": completion}])
                except openai.APIError:
                    return 0
                return float(r.choices[0].message.content.strip() == "CORRECT")
        """))
        self.assertEqual(len(f), 1)

    def test_judge_reply_without_score(self):
        f = fire(self.R, env("""
            import re
            async def judge_score(prompt, completion, answer, judge, **kw):
                verdict = await judge(prompt, completion, answer)
                m = re.search(r"score: (\\d+)", verdict)
                if not m:
                    return 0.0
                return int(m.group(1)) / 10
        """))
        self.assertEqual(len(f), 1)
        self.assertIn("no readable verdict", f[0].message)

    def test_handler_reraises(self):
        self.assertEqual(fire(self.R, env("""
            async def judge_reward(prompt, completion, answer, judge=None, **kw):
                try:
                    out = await judge(prompt, completion, answer)
                    return 1.0 if "yes" in out.lower() else 0.0
                except Exception:
                    raise
        """)), [])

    def test_model_side_parse_failure(self):
        self.assertEqual(fire(self.R, env("""
            import re
            def correct_reward(completion, answer, **kw):
                text = completion[-1]["content"]
                m = re.search(r"answer: (\\d+)", text)
                if not m:
                    return 0.0
                return float(m.group(1) == answer)
        """)), [])

    def test_model_verdict_named_like_judge(self):
        self.assertEqual(fire(self.R, env("""
            async def proxy_verdict_flag(completion, answer, info):
                verdict = extract_json_from_text(last_text(completion))
                if verdict is None:
                    return 0.0
                return 1.0 if verdict.get("flag") == answer else 0.0
        """)), [])

    def test_failure_is_flagged_in_state(self):
        self.assertEqual(fire(self.R, env("""
            async def judge_reward(prompt, completion, answer, state, judge=None, **kw):
                try:
                    out = await judge(prompt, completion, answer)
                    return 1.0 if "yes" in out.lower() else 0.0
                except Exception:
                    state["judge_error"] = True
                    return 0.0
        """)), [])

    def test_metric_only(self):
        self.assertEqual(fire(self.R, env("""
            import verifiers as vf
            async def judge_metric(prompt, completion, answer, judge=None, **kw):
                try:
                    out = await judge(prompt, completion, answer)
                    return 1.0 if "yes" in out.lower() else 0.0
                except Exception:
                    return 0.0
            def exact(completion, answer, **kw):
                return float(completion == answer)
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[exact, judge_metric], weights=[1.0, 0.0]))
        """)), [])


class GraderFailureFixWave(unittest.TestCase):
    """0.5.0 fix wave (check_1.md): fire only when a rubric term with nonzero weight reaches the function."""
    R = "grader-failure-scored-zero"
    JUDGE = """
        async def _judge(prompt, completion, answer, judge=None):
            try:
                out = await judge(prompt, completion, answer)
                return 1.0 if "yes" in out.lower() else 0.0
            except Exception:
                return 0.0
    """

    def test_pos_reached_through_paid_term(self):
        f = fire(self.R, env("import verifiers as vf\n" + textwrap.dedent(self.JUDGE) + """
async def claim_reward(prompt, completion, answer, judge=None, **kw):
    return 0.3 * await _judge(prompt, completion, answer, judge)
def load_environment():
    return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[claim_reward], weights=[1.0]))
"""))
        self.assertEqual(len(f), 1)
        self.assertIn("judge fault", f[0].message)

    def test_pos_sandbox_failure_named(self):
        f = fire(self.R, env("""
            import verifiers as vf
            async def reward(completion, info, sandbox_client=None, **kw):
                try:
                    box = await sandbox_client.create_sandbox(image="python")
                    out = await box.exec("python solution.py")
                    return 1.0 if out.exit_code == 0 else 0.0
                except ConnectionError:
                    return 0.0
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[reward], weights=[1.0]))
        """))
        self.assertEqual(len(f), 1)
        self.assertNotIn("judge fault", f[0].message)
        self.assertIn("grader fault", f[0].message)

    def test_pos_sandbox_call_seen_as_llm(self):
        f = fire(self.R, env("""
            import verifiers as vf
            async def reward(completion, info, client=None, **kw):
                try:
                    box = await client.sandbox.create(image="cuda")
                    res = await client.chat.run_in_sandbox(box, completion)
                    return float(res.ok)
                except Exception:
                    return 0.0
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[reward], weights=[1.0]))
        """))
        self.assertEqual(len(f), 1)

    def test_pos_term_built_by_factory_into_variable(self):
        f = fire(self.R, env("""
            import verifiers as vf
            def _make_llm_judge_reward(judge_obj):
                async def _judge_reward(prompt, completion, answer, state, **kw):
                    try:
                        out = await judge_obj.judge(prompt, completion, answer, state)
                    except Exception as e:
                        return 0.0
                    return 1.0 if "yes" in out else 0.0
                return _judge_reward
            def load_environment():
                judge = vf.JudgeRubric()
                judge_reward_func = _make_llm_judge_reward(judge)
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[judge_reward_func], weights=[1.0]))
        """))
        self.assertEqual(len(f), 1)

    def test_neg_function_never_called(self):
        self.assertEqual(fire(self.R, env("import verifiers as vf\n" + textwrap.dedent(self.JUDGE) + """
def exact(completion, answer, **kw):
    return float(completion == answer)
async def score_quiz(prompt, completion, answer, judge=None, **kw):
    return await _judge(prompt, completion, answer, judge)
def load_environment():
    return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[exact], weights=[1.0]))
""")), [])

    def test_neg_weight_zero_monitor(self):
        self.assertEqual(fire(self.R, env("import verifiers as vf\n" + textwrap.dedent(self.JUDGE) + """
def exact(completion, answer, **kw):
    return float(completion == answer)
async def judge_monitor(prompt, completion, answer, judge=None, **kw):
    return await _judge(prompt, completion, answer, judge)
def load_environment():
    return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[exact, judge_monitor], weights=[1.0, 0.0]))
""")), [])

    def test_neg_file_nothing_imports(self):
        self.assertEqual(fire(self.R, env("""
            import verifiers as vf
            from models.checker import is_terminal
            def exact(completion, answer, **kw):
                return float(completion == answer)
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[exact], weights=[1.0]))
        """, extra={"models/__init__.py": "from .checker import is_terminal\n",
                    "models/checker.py": "def is_terminal(x):\n    return False\n",
                    "models/checker_OLD.py": textwrap.dedent("""
            async def judge_is_terminal(prompt, completion, answer, judge=None):
                try:
                    out = await judge(prompt, completion, answer)
                    return 1.0 if "yes" in out.lower() else 0.0
                except Exception:
                    return 0.0
        """)})), [])

    def test_neg_value_only_logged(self):
        self.assertEqual(fire(self.R, env("import verifiers as vf\n" + textwrap.dedent(self.JUDGE) + """
class GameEnv(vf.MultiTurnEnv):
    async def env_response(self, messages, state, judge=None, **kw):
        state["judge_rating"] = await _judge(messages, messages, "", judge)
        return [], state
def move_reward(completion, info, **kw):
    return info["move_rating"]
def load_environment():
    return GameEnv(rubric=vf.Rubric(funcs=[move_reward], weights=[1.0]))
""")), [])


class NumericToleranceFixWave(unittest.TestCase):
    """0.5.0 fix wave (check_1.md): partial credit below the exact match term, and a 1% relative tolerance."""
    R = "numeric-tolerance-too-loose"

    def test_pos_sign_blind_full_credit(self):
        f = fire(self.R, env("""
            import verifiers as vf
            def numerical_accuracy_reward(completion, answer, **kw):
                found_val, expected = parse(completion), float(answer)
                score = 0.0
                if abs(abs(found_val) - abs(expected)) / max(abs(expected), 0.01) < 0.20:
                    score += 1.0
                return score
            rubric = vf.Rubric(funcs=[numerical_accuracy_reward], weights=[0.2])
        """))
        self.assertEqual(len(f), 1)

    def test_pos_wide_tolerance_full_pay(self):
        f = fire(self.R, env("""
            import verifiers as vf
            def correct_answer(completion, answer, **kw):
                got = parse(completion)
                if got == answer:
                    return 1.0
                return 1.0 if abs(int(got) - int(answer)) <= 10 else 0.0
            rubric = vf.Rubric(funcs=[correct_answer], weights=[1.0])
            # gsm8k style integer answers
        """))
        self.assertEqual(len(f), 1)

    def test_pos_relative_two_percent(self):
        f = fire(self.R, env("""
            import math
            import verifiers as vf
            def close_reward(completion, answer, **kw):
                return 1.0 if math.isclose(parse(completion), float(answer), rel_tol=0.02) else 0.0
            rubric = vf.Rubric(funcs=[close_reward], weights=[1.0])
            # answer is an integer
        """))
        self.assertEqual(len(f), 1)

    def test_neg_relative_one_percent(self):
        self.assertEqual(fire(self.R, env("""
            import numpy as np
            import verifiers as vf
            def score_answer(completion, answer, **kw):
                return 1.0 if np.allclose(parse(completion), answer, rtol=1e-2) else 0.0
            rubric = vf.Rubric(funcs=[score_answer], weights=[1.0])
            # answer is an integer
        """)), [])

    def test_neg_partial_branch_below_exact_branch(self):
        self.assertEqual(fire(self.R, env("""
            import verifiers as vf
            def count_reward(completion, answer, **kw):
                got, gold = int(parse(completion)), int(answer)
                if got == gold:
                    return 1.0
                if abs(got - gold) <= 2:
                    return 0.5
                return 0.0
            rubric = vf.Rubric(funcs=[count_reward], weights=[1.0])
            # answer is an integer
        """)), [])

    def test_neg_partial_term_below_exact_term(self):
        self.assertEqual(fire(self.R, env("""
            import verifiers as vf
            def exact_match_reward(completion, answer, **kw):
                return float(parse(completion) == answer)
            def partial_credit_reward(completion, answer, **kw):
                return 0.5 if abs(int(parse(completion)) - int(answer)) <= 2 else 0.0
            rubric = vf.Rubric(funcs=[exact_match_reward, partial_credit_reward], weights=[1.0, 0.1])
            # answer is an integer
        """)), [])


class SetRecallWeight(unittest.TestCase):
    """0.5.0 fix wave (check_4.md): the message states the weight of the flagged term."""
    R = "set-answer-recall-no-precision"
    SRC = """
        import verifiers as vf
        def correctness(completion, answer, **kw):
            expected = set(answer.split(","))
            predicted = set(parse(completion).split(","))
            return len(expected & predicted) / len(expected)
    """

    def test_weight_and_share_in_message(self):
        f = fire(self.R, env(textwrap.dedent(self.SRC) + "rubric = vf.Rubric(funcs=[correctness, fmt], weights=[0.6, 0.4])\n"))
        self.assertEqual(len(f), 1)
        self.assertIn("weight 0.6 (60% of the reward)", f[0].message)

    def test_weight_through_caller(self):
        f = fire(self.R, env(textwrap.dedent(self.SRC) + "def total(completion, answer, **kw):\n"
                                                         "    return correctness(completion, answer)\n"
                                                         "rubric = vf.Rubric(funcs=[total], weights=[1.0])\n"))
        self.assertEqual(len(f), 1)
        self.assertIn("weight 1 through total", f[0].message)

    def test_no_rubric_no_weight_text(self):
        f = fire(self.R, env(self.SRC))
        self.assertEqual(len(f), 1)
        self.assertNotIn("weight", f[0].message)


# ---------------------------------------------------------------------------
# v0.5: grader-fails-open widening, denominator-drops-failed-items
# ---------------------------------------------------------------------------

class DenominatorDrops(unittest.TestCase):
    R = "grader-fails-open"

    def drop(self, f):
        return [x for x in f if "divides by the items kept" in x.message]

    def test_missing_prediction_skipped(self):
        f = fire(self.R, env("""
            def multi_part_reward(completion, info, **kw):
                preds = parse_parts(completion[-1]["content"])
                scores = []
                for i, gold in enumerate(info["answers"]):
                    pred = preds.get(i)
                    if pred is None:
                        continue
                    scores.append(1.0 if pred == gold else 0.0)
                if not scores:
                    return 0.0
                return sum(scores) / len(scores)
        """))
        self.assertEqual(len(self.drop(f)), 1)
        self.assertIn("no prediction", self.drop(f)[0].message)

    def test_judge_error_skipped_with_counters(self):
        f = fire(self.R, env("""
            async def claims_reward(completion, info, judge, **kw):
                correct, total = 0, 0
                for claim in info["claims"]:
                    try:
                        v = await judge(claim, completion)
                    except Exception:
                        continue
                    correct += 1 if "yes" in v else 0
                    total += 1
                return correct / total if total else 0.0
        """))
        self.assertEqual(len(self.drop(f)), 1)

    def test_mean_over_parsed(self):
        f = fire(self.R, env("""
            import numpy as np
            def numeric_parts_reward(completion, answer, **kw):
                vals = []
                for part, gold in zip(split_parts(completion), answer):
                    try:
                        pred = float(part)
                    except ValueError:
                        continue
                    vals.append(abs(pred - gold) < 1e-6)
                return float(np.mean(vals)) if vals else 0.0
        """))
        self.assertEqual(len(self.drop(f)), 1)

    def test_failed_item_counted_as_zero(self):
        self.assertEqual(self.drop(fire(self.R, env("""
            def multi_part_reward(completion, info, **kw):
                preds = parse_parts(completion[-1]["content"])
                scores = []
                for i, gold in enumerate(info["answers"]):
                    pred = preds.get(i)
                    if pred is None:
                        scores.append(0.0)
                        continue
                    scores.append(1.0 if pred == gold else 0.0)
                return sum(scores) / len(scores)
        """))), [])

    def test_divides_by_gold_count(self):
        self.assertEqual(self.drop(fire(self.R, env("""
            def multi_part_reward(completion, info, **kw):
                preds = parse_parts(completion[-1]["content"])
                hits = []
                for i, gold in enumerate(info["answers"]):
                    pred = preds.get(i)
                    if pred is None:
                        continue
                    hits.append(pred == gold)
                return sum(hits) / len(info["answers"])
        """))), [])

    def test_skip_on_missing_gold(self):
        self.assertEqual(self.drop(fire(self.R, env("""
            def multi_part_reward(completion, info, **kw):
                preds = parse_parts(completion[-1]["content"])
                scores = []
                for i, gold in enumerate(info["answers"]):
                    if gold is None:
                        continue
                    scores.append(1.0 if preds.get(i) == gold else 0.0)
                return sum(scores) / len(scores)
        """))), [])


# ---------------------------------------------------------------------------
# v0.5: set-answer-recall-no-precision
# ---------------------------------------------------------------------------

class SetRecall(unittest.TestCase):
    R = "set-answer-recall-no-precision"

    def test_intersection_over_gold(self):
        f = fire(self.R, env("""
            def entity_reward(completion, answer, **kw):
                predicted = set(parse_list(completion[-1]["content"]))
                gold = set(answer)
                return len(predicted & gold) / len(gold)
        """))
        self.assertEqual(len(f), 1)

    def test_sum_over_gold(self):
        f = fire(self.R, env("""
            def recall_reward(completion, info, **kw):
                predicted = extract_items(completion)
                gold_entities = info["entities"]
                return sum(1 for g in gold_entities if g in predicted) / max(len(gold_entities), 1)
        """))
        self.assertEqual(len(f), 1)

    def test_loop_counter(self):
        f = fire(self.R, env("""
            def category_reward(completion, answer, **kw):
                pred_items = [x.strip() for x in completion[-1]["content"].split(",")]
                found = 0
                for g in answer:
                    if g in pred_items:
                        found += 1
                return found / len(answer)
        """))
        self.assertEqual(len(f), 1)

    def test_f1_has_precision(self):
        self.assertEqual(fire(self.R, env("""
            def entity_reward(completion, answer, **kw):
                predicted = set(parse_list(completion))
                gold = set(answer)
                recall = len(predicted & gold) / len(gold)
                precision = len(predicted & gold) / len(predicted)
                return 2 * precision * recall / (precision + recall)
        """)), [])

    def test_full_text_haystack_left_to_lenient(self):
        self.assertEqual(fire(self.R, env("""
            def recall_reward(completion, answers, **kw):
                response = completion[-1]["content"]
                return sum(1 for a in answers if a in response) / len(answers)
        """)), [])

    def test_size_cap(self):
        self.assertEqual(fire(self.R, env("""
            def entity_reward(completion, answer, **kw):
                predicted = set(parse_list(completion))
                gold = set(answer)
                if len(predicted) > len(gold):
                    return 0.0
                return len(predicted & gold) / len(gold)
        """)), [])


# ---------------------------------------------------------------------------
# v0.5: numeric-tolerance-too-loose
# ---------------------------------------------------------------------------

class NumericTolerance(unittest.TestCase):
    R = "numeric-tolerance-too-loose"

    def test_wide_relative_tolerance(self):
        f = fire(self.R, env("""
            import math
            def numeric_reward(completion, answer, **kw):
                pred = float(extract_number(completion))
                return 1.0 if math.isclose(pred, float(answer), rel_tol=0.1) else 0.0
        """))
        self.assertEqual(len(f), 1)
        self.assertIn("relative tolerance of 0.1", f[0].message)

    def test_sign_blind(self):
        f = fire(self.R, env("""
            def numeric_reward(completion, answer, **kw):
                pred = float(extract_number(completion))
                return float(abs(pred) == abs(float(answer)))
        """))
        self.assertEqual(len(f), 1)

    def test_percent_blind(self):
        f = fire(self.R, env("""
            def numeric_reward(completion, answer, **kw):
                pred = float(extract_number(completion))
                gold = float(answer)
                return float(pred == gold or pred * 100 == gold or pred / 100 == gold)
        """))
        self.assertEqual(len(f), 1)

    def test_abs_tolerance_on_integer_answers(self):
        f = fire(self.R, env("""
            # gsm8k style integer answers
            def numeric_reward(completion, answer, **kw):
                pred = float(extract_number(completion))
                return 1.0 if abs(pred - int(answer)) <= 1 else 0.0
        """))
        self.assertEqual(len(f), 1)

    def test_tight_relative_tolerance(self):
        self.assertEqual(fire(self.R, env("""
            import math
            def numeric_reward(completion, answer, **kw):
                pred = float(extract_number(completion))
                return 1.0 if math.isclose(pred, float(answer), rel_tol=1e-6) else 0.0
        """)), [])

    def test_tight_absolute_tolerance(self):
        self.assertEqual(fire(self.R, env("""
            def numeric_reward(completion, answer, **kw):
                pred = float(extract_number(completion))
                return 1.0 if abs(pred - float(answer)) < 1e-6 else 0.0
        """)), [])

    def test_abs_tolerance_one_on_continuous_answers(self):
        self.assertEqual(fire(self.R, env("""
            def npv_reward(completion, answer, **kw):
                pred = float(extract_number(completion))
                return 1.0 if abs(pred - float(answer)) <= 1.0 else 0.0
        """)), [])

    def test_metric_only(self):
        self.assertEqual(fire(self.R, env("""
            import math, verifiers as vf
            def close_metric(completion, answer, **kw):
                pred = float(extract_number(completion))
                return 1.0 if math.isclose(pred, float(answer), rel_tol=0.1) else 0.0
            def exact(completion, answer, **kw):
                return float(completion == answer)
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[exact, close_metric], weights=[1.0, 0.0]))
        """)), [])


# ---------------------------------------------------------------------------
# v0.5: formal-statement-vacuity
# ---------------------------------------------------------------------------

LEAN_GRADER = """
import subprocess
def lean_reward(completion, info, **kw):
    r = subprocess.run(["lake", "env", "lean", "Main.lean"], capture_output=True)
    return float(r.returncode == 0)
"""


def lean_rows(*stmts):
    import json as _j
    return "\n".join(_j.dumps({"name": f"p{i}", "formal_statement": s}) for i, s in enumerate(stmts)) + "\n"


class FormalVacuity(unittest.TestCase):
    R = "formal-statement-vacuity"

    def test_conclusion_is_hypothesis(self):
        f = fire(self.R, env(LEAN_GRADER, extra={"data/train.jsonl": lean_rows(
            "theorem p0 (x : ℕ) (h₀ : x + 2 = 5) : x = 3 := by sorry",
            "theorem p1 (x : ℕ) (h₀ : x = 3) : x = 3 := by sorry")}))
        self.assertEqual(len(f), 1)
        self.assertIn("also a hypothesis", f[0].message)
        self.assertIn("p1", f[0].message)

    def test_true_conclusion(self):
        f = fire(self.R, env(LEAN_GRADER, extra={"data/train.jsonl": lean_rows("theorem easy (n : ℕ) : True := by sorry")}))
        self.assertEqual(len(f), 1)
        self.assertIn("True", f[0].message)

    def test_false_hypothesis_in_lean_file(self):
        f = fire(self.R, env(LEAN_GRADER, extra={"problems/P.lean": "import Mathlib\n\ntheorem bad (a : ℝ) (h : False) :\n  a = 1 := by\n  sorry\n"}))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].evidence, ["problems/P.lean:3"])

    def test_dafny_ensures_true(self):
        f = fire(self.R, env("""
            import subprocess
            def dafny_reward(completion, **kw):
                r = subprocess.run(["dafny", "verify", "prog.dfy"])
                return float(r.returncode == 0)
        """, extra={"data/tasks.jsonl": '{"spec": "method Max(a: int, b: int) returns (m: int)\\n  ensures true\\n"}\n'}))
        self.assertEqual(len(f), 1)

    def test_normal_statements(self):
        self.assertEqual(fire(self.R, env(LEAN_GRADER, extra={"data/train.jsonl": lean_rows(
            "theorem p0 (x : ℕ) (h₀ : x + 2 = 5) : x = 3 := by sorry",
            "theorem p1 (a b : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 2) : a ^ 2 ≠ 4 ∨ a < 0 := by sorry")})), [])

    def test_no_lean_grader(self):
        self.assertEqual(fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": "def r(completion):\n    return 1.0\n",
                                       "data/train.jsonl": lean_rows("theorem easy : True := by sorry")}), [])

    def test_negation_of_other_hypothesis_only(self):
        self.assertEqual(fire(self.R, env(LEAN_GRADER, extra={"data/train.jsonl": lean_rows(
            "theorem p0 (x : ℕ) (h₀ : ¬ x = 2) (h₁ : x = 3) : x + 1 = 4 := by sorry")})), [])


if __name__ == "__main__":
    unittest.main()

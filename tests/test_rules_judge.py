"""Tests for the judge family rules (envscan/rules_judge.py).

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

MINE = ("cot-reaches-reward", "judge-config-hygiene", "label-free-reward", "judge-same-family-or-policy",
        "spec-oracle-gap", "kernel-perf-grader")
KINDS = ("hack", "broken", "hack", "hack", "broken", "hack")


def pkg(files: dict) -> Package:
    d = tempfile.mkdtemp(prefix="envscan-judge-")
    try:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(text).lstrip("\n"))
        return Package.from_dir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def only(findings, text):
    """Findings whose message has the text: one rule reports several branches."""
    return [f for f in findings if text in f.message]


PYPROJECT = '[project]\nname = "x"\ndependencies = ["verifiers>=0.1.3"]\n'


def fire(rid, code=None, extra=None, files=None):
    fs = dict(files) if files is not None else {"pyproject.toml": PYPROJECT, "env.py": code}
    fs.update(extra or {})
    return RULES[rid]["fn"](pkg(fs))


class Registered(unittest.TestCase):
    def test_rules_registered(self):
        for rid, kind in zip(MINE, KINDS):
            self.assertIn(rid, RULES)
            self.assertEqual(RULES[rid]["kind"], kind)
            desc = RULES[rid]["description"]
            self.assertNotIn(";", desc)
            self.assertNotIn("—", desc)

    def test_odd_input_does_not_crash(self):
        p = pkg({"pyproject.toml": PYPROJECT, "broken.py": "def f(:\n  return 1\n", "empty.py": "",
                 "task.toml": "[verifier]\n", "instruction.md": "", "tests/test.sh": "#!/bin/bash\n",
                 "tests/test_x.py": "\x00 not python", "solution/solve.sh": "echo hi\n",
                 "configs/rl.toml": "[model\nname = 1", "configs/ok.toml": "[model]\nname = 3\n",
                 "odd.py": "x = vf.JudgeRubric(parser=p, judge_prompt=q)\ndef a(completion, judge):\n  return judge(completion)\n"})
        findings, errors = run_rules(p, only=set(MINE))
        self.assertEqual(errors, [])
        self.assertEqual(findings, [])

    def test_all_review_and_clean_messages(self):
        found = (fire("cot-reaches-reward", COT_POS) + fire("judge-config-hygiene", BUDGET_POS) +
                 fire("label-free-reward", MAJ_POS) + fire("judge-same-family-or-policy", files=FAM_POS) +
                 fire("spec-oracle-gap", files=SPEC_POS) + fire("kernel-perf-grader", KERNEL_POS))
        self.assertEqual(len(found), 7)  # KERNEL_POS also reuses one input per timed trial (v0.5)
        for f in found:
            self.assertEqual(f.confidence, "review")
            self.assertNotIn(";", f.message)
            self.assertNotIn("—", f.message)


# ---------------------------------------------------------------------------
# cot-reaches-reward
# ---------------------------------------------------------------------------

COT_POS = """
    import verifiers as vf
    SYSTEM = "Think step by step inside <think>...</think> tags, then give the final answer."
    def load_environment(judge_client=None):
        rubric = vf.JudgeRubric(judge_client=judge_client, judge_model="gpt-4.1-mini")
        async def judge_reward(judge, prompt, completion, answer, state):
            verdict = await judge(prompt, completion, answer, state)
            return 1.0 if "yes" in verdict.lower() else 0.0
        rubric.add_reward_func(judge_reward, weight=1.0)
        return vf.SingleTurnEnv(system_prompt=SYSTEM, parser=vf.ThinkParser(), rubric=rubric)
"""


class CotReachesReward(unittest.TestCase):
    R = "cot-reaches-reward"

    def test_judge_rubric_without_parser(self):
        f = fire(self.R, COT_POS)
        self.assertEqual(len(f), 1)
        self.assertIn("judge_reward", f[0].message)

    def test_plain_parser_on_the_judge(self):
        code = COT_POS.replace('judge_model="gpt-4.1-mini")', 'judge_model="gpt-4.1-mini", parser=vf.Parser())')
        self.assertEqual(len(fire(self.R, code)), 1)

    def test_direct_judge_call_with_raw_reply(self):
        f = fire(self.R, """
            import verifiers as vf
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            PROMPT = "Use <think> tags to reason about the patient, then write the diagnosis after them."
            async def diagnosis_reward(completion, answer):
                text = completion[-1]["content"]
                prompt = f"Reference: {answer}\\nResponse: {text}\\nIs the response correct? yes or no"
                r = await client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}])
                return 1.0 if "yes" in r.choices[0].message.content else 0.0
            def load_environment():
                return vf.SingleTurnEnv(system_prompt=PROMPT, rubric=vf.Rubric(funcs=[diagnosis_reward]))
        """)
        self.assertEqual(len(f), 1)

    def test_think_parser_on_the_judge(self):
        code = COT_POS.replace('judge_model="gpt-4.1-mini")', 'judge_model="gpt-4.1-mini", parser=vf.ThinkParser())')
        self.assertEqual(fire(self.R, code), [])

    def test_think_block_split_off_before_the_judge(self):
        code = COT_POS.replace("verdict = await judge(prompt, completion, answer, state)",
                               'cleaned = [{"role": m["role"], "content": m["content"].split("</think>")[-1]} for m in completion]\n'
                               "            verdict = await judge(prompt, cleaned, answer, state)")
        self.assertEqual(fire(self.R, code), [])

    def test_no_think_block_asked_for(self):
        code = COT_POS.replace("Think step by step inside <think>...</think> tags, then give the final answer.",
                               "Answer the question.").replace("parser=vf.ThinkParser(), ", "")
        self.assertEqual(fire(self.R, code), [])

    def test_judge_prompt_without_the_response(self):
        code = COT_POS.replace('judge_model="gpt-4.1-mini")', 'judge_model="gpt-4.1-mini", judge_prompt="{question}")')
        self.assertEqual(fire(self.R, code), [])

    def test_judge_reward_at_weight_zero(self):
        self.assertEqual(fire(self.R, COT_POS.replace("weight=1.0", "weight=0.0")), [])


# ---------------------------------------------------------------------------
# judge-config-hygiene
# ---------------------------------------------------------------------------

BUDGET_POS = """
    import verifiers as vf
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    async def judge_reward(completion, answer):
        prompt = f"Is this correct? Answer yes or no.\\n{completion[-1]['content']}\\nReference: {answer}"
        r = await client.chat.completions.create(model="gpt-5-mini", messages=[{"role": "user", "content": prompt}],
                                                 max_completion_tokens=64)
        return 1.0 if "yes" in (r.choices[0].message.content or "").lower() else 0.0
    def load_environment():
        return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[judge_reward]))
"""

EXPLAIN = """
    import verifiers as vf
    PROMPT = "Explain your reasoning step by step, then give a final verdict: CORRECT or INCORRECT."
    async def judge_reward(completion, answer, judge_client):
        r = await judge_client.chat.completions.create(model="gpt-4.1-mini", max_tokens=BUDGET,
                                                       messages=[{"role": "user", "content": PROMPT + completion[-1]["content"]}])
        return 1.0 if "INCORRECT" not in r.choices[0].message.content else 0.0
    def load_environment():
        return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[judge_reward]))
"""

UNCLAMPED = """
    import re
    import verifiers as vf
    def load_environment(judge_client=None):
        rubric = vf.JudgeRubric(judge_client=judge_client, judge_model="gpt-4.1-mini")
        async def score_reward(judge, prompt, completion, answer, state):
            reply = await judge(prompt, completion, answer, state)
            m = re.search(PATTERN, reply)
            if not m:
                return 0.0
            score = float(m.group(1)) / 10.0
            return RETURN
        rubric.add_reward_func(score_reward, weight=1.0)
        return vf.SingleTurnEnv(rubric=rubric)
"""


class JudgeConfigHygiene(unittest.TestCase):
    R = "judge-config-hygiene"

    def test_reasoning_judge_with_small_budget(self):
        f = fire(self.R, BUDGET_POS)
        self.assertEqual(len(f), 1)
        self.assertIn("gpt-5-mini", f[0].message)
        self.assertEqual(f[0].kind, "broken")

    def test_explanation_asked_with_tiny_budget(self):
        f = fire(self.R, EXPLAIN.replace("BUDGET", "20"))
        self.assertEqual(len(f), 1)
        self.assertIn("explanation", f[0].message)

    def test_reasoning_model_in_judge_rubric_sampling_args(self):
        f = fire(self.R, """
            import verifiers as vf
            def load_environment(judge_client=None):
                return vf.SingleTurnEnv(rubric=vf.JudgeRubric(judge_client=judge_client, judge_model="o3-mini",
                                                              judge_sampling_args={"max_completion_tokens": 256}))
        """)
        self.assertEqual(len(f), 1)

    def test_unclamped_judge_score(self):
        f = fire(self.R, UNCLAMPED.replace("PATTERN", 'r"Score: (\\d+)"').replace("RETURN", "score"))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "hack")
        self.assertIn("no upper limit", f[0].message)

    def test_one_word_verdict_in_a_small_budget(self):
        self.assertEqual(fire(self.R, BUDGET_POS.replace("gpt-5-mini", "gpt-4o-mini").replace("64", "5")), [])

    def test_reasoning_model_with_room(self):
        self.assertEqual(fire(self.R, BUDGET_POS.replace("max_completion_tokens=64", "max_completion_tokens=4096")), [])

    def test_reasoning_model_at_minimal_effort(self):
        code = BUDGET_POS.replace("max_completion_tokens=64", 'max_completion_tokens=64, reasoning_effort="minimal"')
        self.assertEqual(fire(self.R, code), [])

    def test_explanation_with_enough_budget(self):
        self.assertEqual(fire(self.R, EXPLAIN.replace("BUDGET", "1024")), [])

    def test_clamped_score(self):
        code = UNCLAMPED.replace("PATTERN", 'r"Score: (\\d+)"').replace("RETURN", "max(0.0, min(1.0, score))")
        self.assertEqual(fire(self.R, code), [])

    def test_bounded_score_pattern(self):
        code = UNCLAMPED.replace("PATTERN", 'r"Score: ([0-9]|10)\\b"').replace("RETURN", "score")
        self.assertEqual(fire(self.R, code), [])


# ---------------------------------------------------------------------------
# label-free-reward
# ---------------------------------------------------------------------------

MAJ_POS = """
    import verifiers as vf
    from collections import Counter
    def majority_reward(completions, parser, **kwargs):
        answers = [parser.parse_answer(c) for c in completions]
        top, _ = Counter(answers).most_common(1)[0]
        return [1.0 if a == top else 0.0 for a in answers]
    def load_environment():
        rubric = vf.Rubric(funcs=[majority_reward])
        return vf.SingleTurnEnv(rubric=rubric)
"""


class LabelFreeReward(unittest.TestCase):
    R = "label-free-reward"

    def test_majority_vote_with_no_gold(self):
        f = fire(self.R, MAJ_POS)
        self.assertEqual(len(f), 1)
        self.assertIn("boxed", f[0].message)

    def test_self_certainty_from_the_rollout(self):
        f = fire(self.R, """
            import math
            import verifiers as vf
            def self_certainty_reward(completion, state, **kwargs):
                lps = [t.logprob for r in state["responses"] for t in r.choices[0].logprobs.content]
                return math.exp(sum(lps) / max(1, len(lps)))
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[self_certainty_reward]))
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("confidence", f[0].message)

    def test_reference_free_qe_metric(self):
        f = fire(self.R, """
            import verifiers as vf
            from comet import download_model, load_from_checkpoint
            model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
            def qe_reward(completion, info, **kwargs):
                data = [{"src": info["source"], "mt": completion[-1]["content"]}]
                return float(model.predict(data, batch_size=1).scores[0])
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[qe_reward]))
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("language", f[0].message)

    def test_gold_metric_logged(self):
        code = MAJ_POS.replace("def load_environment():", "def accuracy(completion, answer, parser):\n"
                               "        return float(parser.parse_answer(completion) == answer)\n"
                               "    def load_environment():").replace(
            "rubric = vf.Rubric(funcs=[majority_reward])",
            "rubric = vf.Rubric(funcs=[majority_reward, accuracy], weights=[1.0, 0.0])")
        self.assertEqual(fire(self.R, code), [])

    def test_language_guard(self):
        self.assertEqual(fire(self.R, MAJ_POS.replace("from collections import Counter",
                                                      "from collections import Counter\n    import langdetect")), [])

    def test_perplexity_of_a_gold_text_under_another_model(self):
        self.assertEqual(fire(self.R, """
            import math
            import verifiers as vf
            from openai import OpenAI
            client = OpenAI()
            def ppl(messages):
                r = client.chat.completions.create(model="m", messages=messages, logprobs=True, max_tokens=1)
                return math.exp(-sum(x.logprob for x in r.prompt_logprobs) / len(r.prompt_logprobs))
            def plan_reward(completion, info, **kwargs):
                return ppl(info["base"]) - ppl(info["with_plan"])
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[plan_reward]))
        """), [])

    def test_majority_vote_not_paid(self):
        code = MAJ_POS.replace("rubric = vf.Rubric(funcs=[majority_reward])",
                               "rubric = vf.Rubric(funcs=[majority_reward], weights=[0.0])")
        self.assertEqual(fire(self.R, code), [])


# ---------------------------------------------------------------------------
# judge-same-family-or-policy
# ---------------------------------------------------------------------------

JUDGE_CODE = """
    import verifiers as vf
    def load_environment(judge_model="gpt-4.1-mini", judge_base_url=None):
        return vf.SingleTurnEnv(rubric=vf.JudgeRubric(judge_model=judge_model))
"""

FAM_POS = {
    "pyproject.toml": PYPROJECT,
    "env.py": JUDGE_CODE,
    "configs/rl.toml": """
        max_steps = 100
        [model]
        name = "Qwen/Qwen3-4B"
        [[env]]
        id = "me/env"
        args = { judge_model = "Qwen/Qwen3-235B-A22B-Instruct-2507" }
    """,
}


class JudgeSameFamily(unittest.TestCase):
    R = "judge-same-family-or-policy"

    def test_same_family_named_in_the_training_config(self):
        f = fire(self.R, files=FAM_POS)
        self.assertEqual(len(f), 1)
        self.assertIn("Qwen", f[0].message)

    def test_judge_on_the_policy_server(self):
        files = dict(FAM_POS)
        files["configs/rl.toml"] = """
            [model]
            name = "Qwen/Qwen3-0.6B"
            [client]
            base_url = ["http://127.0.0.1:8000/v1"]
            [[env]]
            id = "me/env"
            args = { judge_base_url = "http://localhost:8000/v1", judge_model = "auto" }
        """
        f = fire(self.R, files=files)
        self.assertEqual(len(f), 1)
        self.assertIn("own inference server", f[0].message)

    def test_same_family_from_the_code_default(self):
        files = dict(FAM_POS)
        files["env.py"] = JUDGE_CODE.replace("gpt-4.1-mini", "meta-llama/Llama-3.3-70B-Instruct")
        files["configs/rl.toml"] = '[model]\nname = "meta-llama/Llama-3.1-8B-Instruct"\n'
        self.assertEqual(len(fire(self.R, files=files)), 1)

    def test_other_family(self):
        files = dict(FAM_POS)
        files["configs/rl.toml"] = '[model]\nname = "Qwen/Qwen3-4B"\n'
        self.assertEqual(fire(self.R, files=files), [])

    def test_no_training_config(self):
        self.assertEqual(fire(self.R, JUDGE_CODE.replace("gpt-4.1-mini", "Qwen/Qwen3-32B")), [])

    def test_judge_on_another_server(self):
        files = dict(FAM_POS)
        files["configs/rl.toml"] = """
            [model]
            name = "Qwen/Qwen3-0.6B"
            [client]
            base_url = ["http://127.0.0.1:8000/v1"]
            [[env]]
            args = { judge_base_url = "http://127.0.0.1:8001/v1", judge_model = "openai/gpt-oss-20b" }
        """
        self.assertEqual(fire(self.R, files=files), [])

    def test_config_overrides_a_same_family_default(self):
        files = dict(FAM_POS)
        files["env.py"] = JUDGE_CODE.replace("gpt-4.1-mini", "Qwen/Qwen3-32B")
        files["configs/rl.toml"] = files["configs/rl.toml"].replace("Qwen/Qwen3-235B-A22B-Instruct-2507", "gpt-4.1-mini")
        self.assertEqual(fire(self.R, files=files), [])


# ---------------------------------------------------------------------------
# spec-oracle-gap
# ---------------------------------------------------------------------------

SPEC_POS = {
    "task.toml": "[verifier]\ntimeout_sec = 60\n",
    "instruction.md": "Write a Python module /app/sampler.py that implements adaptive rejection sampling for a log-concave density.\n",
    "environment/Dockerfile": "FROM python:3.12\nWORKDIR /app\n",
    "solution/solve.sh": """
        cat > /app/sampler.py <<'EOF'
        def ars(f, a, b, n_samples=100):
            return [a] * n_samples
        EOF
    """,
    "tests/test.sh": "pip install pytest\npytest /tests/test_outputs.py\n",
    "tests/test_outputs.py": """
        import sys
        sys.path.insert(0, "/app")
        from sampler import ars

        def test_samples():
            xs = ars(lambda x: -x * x, a=-1.0, b=1.0, n_samples=50)
            assert len(xs) == 50
    """,
}


def task(**over):
    files = dict(SPEC_POS)
    files.update(over)
    return files


class SpecOracleGap(unittest.TestCase):
    R = "spec-oracle-gap"

    def test_function_name_only_the_oracle_knows(self):
        f = fire(self.R, files=SPEC_POS)
        self.assertEqual(len(f), 1)
        self.assertIn("`ars` (function)", f[0].message)
        self.assertIn("`n_samples` (argument)", f[0].message)
        self.assertEqual(f[0].kind, "broken")

    def test_output_path_only_the_oracle_knows(self):
        f = fire(self.R, files=task(**{
            "instruction.md": "Count the words in /app/input.txt and save the counts as JSON.\n",
            "environment/input.txt": "a b c\n",
            "solution/solve.sh": "python3 -c 'print(1)' > /app/output/word_counts.json\n",
            "tests/test_outputs.py": """
                import json
                from pathlib import Path
                OUT = Path("/app/output") / "word_counts.json"
                def test_counts():
                    assert OUT.exists()
            """}))
        self.assertEqual(len(f), 1)
        self.assertIn("word_counts.json", f[0].message)

    def test_output_key_only_the_oracle_knows(self):
        f = fire(self.R, files=task(**{
            "instruction.md": "Summarize the log at /app/log.txt into /app/summary.json.\n",
            "environment/log.txt": "x\n",
            "solution/solve.sh": "echo '{\"total_errors\": 3}' > /app/summary.json\n",
            "tests/test_outputs.py": """
                import json
                def test_summary():
                    with open("/app/summary.json") as f:
                        data = json.load(f)
                    assert data["total_errors"] == 3
            """}))
        self.assertEqual(len(f), 1)
        self.assertIn("`total_errors` (key)", f[0].message)

    def test_cli_flag_only_the_oracle_knows(self):
        f = fire(self.R, files=task(**{
            "instruction.md": "Write a command line tool /app/convert.py that converts CSV to JSON.\n",
            "solution/solve.sh": "cat > /app/convert.py <<'EOF'\nimport argparse\np = argparse.ArgumentParser()\np.add_argument('--input-file')\nEOF\n",
            "tests/test_outputs.py": """
                import subprocess
                def test_cli():
                    r = subprocess.run(["python3", "/app/convert.py", "--input-file", "/tmp/x.csv"], capture_output=True)
                    assert r.returncode == 0
            """}))
        self.assertEqual(len(f), 1)
        self.assertIn("--input-file", f[0].message)

    def test_names_given_in_the_instruction(self):
        files = task(**{"instruction.md": "Write /app/sampler.py with a function `ars(f, a, b, n_samples)` that "
                                          "implements adaptive rejection sampling.\n"})
        self.assertEqual(fire(self.R, files=files), [])

    def test_no_oracle_solution(self):
        files = dict(SPEC_POS)
        del files["solution/solve.sh"]
        self.assertEqual(fire(self.R, files=files), [])

    def test_adapter_template(self):
        self.assertEqual(fire(self.R, files=task(**{"instruction.md": "# {task_id}\n\n{instructions}\n"})), [])

    def test_module_already_in_the_image(self):
        files = task(**{
            "instruction.md": "Fix the header parsing bug in the web framework under /app.\n",
            "environment/Dockerfile": "FROM python:3.12\nRUN git clone https://github.com/bottlepy/bottle.git /app\n",
            "solution/solve.sh": "cat > /tmp/fix.py <<'EOF'\ndef _hkey(key):\n    return key\nEOF\n",
            "tests/test_outputs.py": """
                import sys
                sys.path.insert(0, "/app")
                def test_hkey():
                    from bottle import _hkey
                    assert _hkey("a") == "a"
            """})
        self.assertEqual(fire(self.R, files=files), [])

    def test_keys_of_the_tests_own_fixture(self):
        files = task(**{
            "instruction.md": "Write /app/result.txt with the answer.\n",
            "solution/solve.sh": "echo 42 > /app/result.txt\necho threshold\n",
            "tests/test_outputs.py": """
                import json
                def test_result():
                    cfg = json.load(open("/tests/expected.json"))
                    assert open("/app/result.txt").read().strip() == str(cfg["threshold"])
            """})
        self.assertEqual(fire(self.R, files=files), [])

    def test_third_party_library_arguments(self):
        files = task(**{
            "instruction.md": "Start a web server on port 8000 that serves /app/index.html.\n",
            "solution/solve.sh": "cat > /app/server.py <<'EOF'\nimport requests\ntimeout = 5\nEOF\n",
            "tests/test_outputs.py": """
                import requests
                def test_server():
                    r = requests.get("http://localhost:8000/", timeout=5)
                    assert r.status_code == 200
            """})
        self.assertEqual(fire(self.R, files=files), [])


# ---------------------------------------------------------------------------
# kernel-perf-grader
# ---------------------------------------------------------------------------

KERNEL_POS = """
    import torch
    import verifiers as vf
    def time_kernel(fn, args, trials=10):
        times = []
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return sum(times) / len(times)
    def speed_reward(completion, info, state):
        fn, args = state["kernel"], state["inputs"]
        speedup = info["baseline_ms"] / time_kernel(fn, args)
        return min(speedup, 10.0)
    def load_environment():
        return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[speed_reward]))
"""

HARBOR_TIMER = """
    import time
    import torch
    def time_fn(fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        SYNC
        return time.perf_counter() - t0, out
    def main():
        x = torch.randn(1024, device="cuda")
        cand, _ = time_fn(candidate, x)
        base, _ = time_fn(baseline, x)
        speedup = base / cand
        open("/logs/verifier/reward.txt", "w").write(str(speedup))
"""


class KernelPerfGrader(unittest.TestCase):
    R = "kernel-perf-grader"

    def test_events_on_one_stream(self):
        f = only(fire(self.R, KERNEL_POS), "side stream")
        self.assertEqual(len(f), 1)

    def test_host_timer_without_sync_in_a_harbor_verifier(self):
        f = fire(self.R, files={"task.toml": "[verifier]\n", "instruction.md": "Make it fast.\n",
                                "tests/test.sh": "python3 /tests/compute_reward.py\n",
                                "tests/compute_reward.py": HARBOR_TIMER.replace("SYNC", "pass")})
        self.assertEqual(len(only(f, "still queued")), 1)

    def test_large_absolute_tolerance_across_problems(self):
        f = fire(self.R, """
            import torch
            import verifiers as vf
            device = torch.device("cuda")
            def correctness_reward(completion, state):
                ref_model, new_model, inputs = state["Model"], state["ModelNew"], state["get_inputs"]()
                out, ref = new_model(*inputs), ref_model(*inputs)
                return 1.0 if torch.allclose(out, ref, atol=1e-02, rtol=1e-02) else 0.0
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[correctness_reward]))
        """)
        self.assertEqual(len(only(f, "returns zeros passes")), 1)

    def test_sync_before_the_end_event(self):
        code = KERNEL_POS.replace("            fn(*args)\n            end.record()",
                                  "            fn(*args)\n            torch.cuda.synchronize()\n            end.record()")
        self.assertEqual(only(fire(self.R, code), "side stream"), [])

    def test_host_timer_with_sync(self):
        files = {"task.toml": "[verifier]\n", "instruction.md": "Make it fast.\n",
                 "tests/compute_reward.py": HARBOR_TIMER.replace("SYNC", "torch.cuda.synchronize()")}
        self.assertEqual(only(fire(self.R, files=files), "still queued"), [])

    def test_cpu_only_timing(self):
        files = {"task.toml": "[verifier]\n", "instruction.md": "Make it fast.\n",
                 "tests/compute_reward.py": HARBOR_TIMER.replace("SYNC", "pass").replace("import torch\n", "")
                 .replace('x = torch.randn(1024, device="cuda")', "x = list(range(1024))")}
        self.assertEqual(fire(self.R, files=files), [])

    def test_rubric_not_registered(self):
        code = KERNEL_POS.replace("return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[speed_reward]))",
                                  "env = vf.SingleTurnEnv()\n        env.reward_funcs = [speed_reward]\n        return env")
        self.assertEqual(fire(self.R, code), [])

    def test_timed_request_to_a_worker_process(self):
        files = {"task.toml": "[verifier]\n", "instruction.md": "Make it fast.\n",
                 "tests/compute_reward.py": HARBOR_TIMER.replace("SYNC", "pass").replace(
                     "out = fn(*args)", 'out = worker.request({"command": "run"})')}
        self.assertEqual(only(fire(self.R, files=files), "still queued"), [])

    def test_tight_tolerance(self):
        self.assertEqual(only(fire(self.R, """
            import torch
            import verifiers as vf
            device = torch.device("cuda")
            def correctness_reward(completion, state):
                out, ref = state["ModelNew"](), state["Model"]()
                return 1.0 if torch.allclose(out, ref, atol=1e-05, rtol=1e-05) else 0.0
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[correctness_reward]))
        """), "returns zeros passes"), [])


# ---------------------------------------------------------------------------
# v0.5 widenings of kernel-perf-grader and spec-oracle-gap
# ---------------------------------------------------------------------------

PERF_TASK = {
    "task.toml": "[verifier]\ntimeout_sec = 60\n",
    "instruction.md": "Write /app/fastsum.py with a function total(xs) that is faster than the reference.\n",
    "solution/solve.sh": "cat > /app/fastsum.py <<'EOF'\ndef total(xs):\n    return sum(xs)\nEOF\n",
    "tests/test.sh": "pytest /tests/test_outputs.py\n",
    "tests/test_outputs.py": """
        import sys
        import time
        sys.path.insert(0, "/app")
        from fastsum import total

        def ref(xs):
            s = 0
            for x in xs:
                s += x
            return s

        def timeit(fn, xs):
            t0 = time.perf_counter()
            fn(xs)
            return time.perf_counter() - t0

        def test_speed():
            xs = list(range(100000))
            assert timeit(total, xs) < timeit(ref, xs)
    """,
}

KB_GRADER = """
    import torch
    import verifiers as vf
    def load_model(src):
        ctx = {}
        exec(src, ctx)
        return ctx["ModelNew"], ctx["Model"], ctx["get_inputs"]
    def time_it(fn, args):
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            fn(*args)
            torch.cuda.synchronize()
            t1.record()
            times.append(t0.elapsed_time(t1))
        return sum(times)
    def kernel_reward(completion, info, state):
        ModelNew, Model, get_inputs = load_model(completion[-1]["content"])
        inputs = get_inputs()
        ref_out = Model()(*inputs)
        new_out = ModelNew()(*inputs)
        if not torch.allclose(new_out, ref_out, atol=1e-02, rtol=1e-02):
            return 0.0
        speedup = info["baseline_ms"] / time_it(ModelNew(), inputs)
        return min(speedup, 3.0)
    def load_environment():
        return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[kernel_reward]))
"""


class KernelV05(unittest.TestCase):
    R = "kernel-perf-grader"

    def hit(self, text, code=None, files=None):
        return only(fire(self.R, code, files=files), text)

    # kernel-timing-isolation + measurement-in-candidate-process
    def test_harbor_test_times_imported_submission(self):
        f = self.hit("same process", files=PERF_TASK)
        self.assertEqual(len(f), 1)
        self.assertIn("imports the submission", f[0].message)

    def test_memory_counter_in_process(self):
        files = dict(PERF_TASK)
        files["tests/test_outputs.py"] = """
            import sys
            import tracemalloc
            sys.path.insert(0, "/app")
            from fastsum import total
            def test_memory():
                tracemalloc.start()
                total(list(range(1000)))
                cur, peak = tracemalloc.get_traced_memory()
                assert peak < 10_000_000
        """
        f = self.hit("same process", files=files)
        self.assertEqual(len(f), 1)
        self.assertIn("memory use", f[0].message)

    def test_kernel_grader_execs_and_times(self):
        f = self.hit("execs or imports", KB_GRADER)
        self.assertEqual(len(f), 1)

    def test_timing_in_a_subprocess(self):
        files = dict(PERF_TASK)
        files["tests/test_outputs.py"] = PERF_TASK["tests/test_outputs.py"].replace(
            "            fn(xs)\n", "            subprocess.run(['python3', '/app/bench.py'])\n").replace(
            "import time\n", "import time\n        import subprocess\n")
        self.assertEqual(self.hit("same process", files=files), [])

    def test_no_speed_assert(self):
        files = dict(PERF_TASK)
        files["tests/test_outputs.py"] = PERF_TASK["tests/test_outputs.py"].replace(
            "assert timeit(total, xs) < timeit(ref, xs)", "print(timeit(total, xs))\n            assert total(xs) == ref(xs)")
        self.assertEqual(self.hit("same process", files=files), [])

    def test_times_only_its_own_code(self):
        files = dict(PERF_TASK)
        files["tests/test_outputs.py"] = PERF_TASK["tests/test_outputs.py"].replace(
            "from fastsum import total\n", "").replace("timeit(total, xs) <", "timeit(ref, xs) <")
        self.assertEqual(self.hit("same process", files=files), [])

    def test_reward_script_only_run_in_fail_mode(self):
        files = {"task.toml": "[verifier]\n", "instruction.md": "Write /app/gen.py, fast.\n",
                 "solution/solve.sh": "cat > /app/gen.py <<'EOF'\ndef run():\n    pass\nEOF\n",
                 "tests/test.sh": "python3 /tests/compute_reward.py \\\n  --fail \"$REASON\"\npython3 /tests/partial.py\n",
                 "tests/compute_reward.py": """
                     import sys, time, argparse
                     def main():
                         ap = argparse.ArgumentParser()
                         ap.add_argument("--fail")
                         args = ap.parse_args()
                         if args.fail:
                             open("/logs/verifier/reward.txt", "w").write("0")
                             return
                         sys.path.insert(0, "/app")
                         from gen import run
                         t0 = time.perf_counter()
                         run()
                         speedup = 1.0 / (time.perf_counter() - t0)
                         open("/logs/verifier/reward.txt", "w").write(str(speedup))
                 """}
        self.assertEqual(self.hit("same process", files=files), [])
        self.assertEqual(self.hit("no upper clip", files=files), [])

    # kernel-launch-and-train-branch + kernel-delegates-to-reference-op
    def test_no_custom_kernel_check(self):
        f = self.hit("nothing checks that ModelNew", KB_GRADER)
        self.assertEqual(len(f), 1)

    def test_text_search_is_the_only_kernel_check(self):
        code = KB_GRADER.replace('    ModelNew, Model, get_inputs = load_model(completion[-1]["content"])',
                                 '    src = completion[-1]["content"]\n        if "__global__" not in src:\n'
                                 '            return 0.0\n        ModelNew, Model, get_inputs = load_model(src)')
        f = self.hit("text search", code)
        self.assertEqual(len(f), 1)

    def test_regex_kernel_check(self):
        code = KB_GRADER.replace('    ModelNew, Model, get_inputs = load_model(completion[-1]["content"])',
                                 '    src = completion[-1]["content"]\n        if not re.search(r"@triton.jit", src):\n'
                                 '            return 0.0\n        ModelNew, Model, get_inputs = load_model(src)')
        self.assertEqual(len(self.hit("text search", code)), 1)

    def test_ast_kernel_check(self):
        code = KB_GRADER + """
    import ast
    def uses_kernel(src):
        tree = ast.parse(src)
        banned = {"torch.nn.functional", "torch.matmul"}
        return "load_inline" in src and not any(isinstance(n, ast.Attribute) for n in ast.walk(tree))
"""
        self.assertEqual(self.hit("custom kernel", code), [])

    def test_not_a_multi_problem_grader(self):
        code = KB_GRADER.replace("ModelNew", "Fast").replace("get_inputs", "inputs_fn").replace("Model", "Slow")
        self.assertEqual(self.hit("custom kernel", code), [])

    def test_kernel_check_not_paid(self):
        code = KB_GRADER.replace("return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[kernel_reward]))",
                                 "return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[kernel_reward], weights=[0.0]))")
        self.assertEqual(self.hit("custom kernel", code), [])

    # kernel-input-exposure-and-cache
    def test_same_inputs_every_trial(self):
        self.assertEqual(len(self.hit("same input objects", KB_GRADER)), 1)

    def test_same_inputs_host_timer(self):
        code = KERNEL_POS.replace("start.record()", "t0 = time.perf_counter()").replace(
            "end.record()", "t1 = time.perf_counter()").replace("import torch\n", "import torch\n    import time\n")
        self.assertEqual(len(self.hit("same input objects", code)), 1)

    def test_same_inputs_while_loop(self):
        code = KERNEL_POS.replace("for _ in range(trials):", "while len(times) < trials:")
        self.assertEqual(len(self.hit("same input objects", code)), 1)

    def test_fresh_inputs_each_trial(self):
        code = KB_GRADER.replace("            fn(*args)\n", "            x = [a.clone() for a in args]\n            fn(*x)\n")
        self.assertEqual(self.hit("same input objects", code), [])

    def test_inputs_regenerated_each_trial(self):
        code = KERNEL_POS.replace("            start.record()\n", "            args = make_inputs()\n            start.record()\n")
        self.assertEqual(self.hit("same input objects", code), [])

    def test_single_timed_call(self):
        code = KERNEL_POS.replace("for _ in range(trials):", "if trials:")
        self.assertEqual(self.hit("same input objects", code), [])

    # kernel-tolerance-sub-threshold
    def tol_code(self, atol):
        return KB_GRADER.replace("atol=1e-02", f"atol={atol}")

    def test_atol_5e3(self):
        self.assertEqual(len(self.hit("returns zeros passes", self.tol_code("5e-3"))), 1)

    def test_atol_2e3(self):
        self.assertEqual(len(self.hit("atol=0.002", self.tol_code("2e-3"))), 1)

    def test_atol_named_constant(self):
        code = self.tol_code("ATOL").replace("import torch\n", "import torch\n    ATOL = 4e-3\n")
        self.assertEqual(len(self.hit("returns zeros passes", code)), 1)

    def test_atol_1e3(self):
        self.assertEqual(self.hit("returns zeros passes", self.tol_code("1e-3")), [])

    def test_atol_1e5(self):
        self.assertEqual(self.hit("returns zeros passes", self.tol_code("1e-5")), [])

    def test_atol_missing(self):
        self.assertEqual(self.hit("returns zeros passes", KB_GRADER.replace("atol=1e-02, ", "")), [])

    # kernel-baseline-precision + kernel-precision-or-dtype-asymmetry
    def test_tf32_never_set(self):
        self.assertEqual(len(self.hit("TF32 never set", KB_GRADER)), 1)

    def test_tf32_never_set_rtol(self):
        self.assertEqual(len(self.hit("TF32 never set", KB_GRADER.replace("atol=1e-02, rtol=1e-02", "rtol=5e-2"))), 1)

    def test_precision_asymmetry(self):
        code = KB_GRADER.replace("ref_out = Model()(*inputs)", "ref_out = Model()(*[x.float() for x in inputs])").replace(
            "new_out = ModelNew()(*inputs)", "new_out = ModelNew()(*[x.half() for x in inputs])")
        f = self.hit("half precision", code)
        self.assertEqual(len(f), 1)
        self.assertEqual(self.hit("TF32 never set", code), [])

    def test_tf32_set(self):
        code = KB_GRADER.replace("import torch\n", "import torch\n    torch.backends.cuda.matmul.allow_tf32 = True\n")
        self.assertEqual(self.hit("TF32", code), [])

    def test_tight_tolerance_no_tf32_issue(self):
        self.assertEqual(self.hit("TF32", KB_GRADER.replace("atol=1e-02, rtol=1e-02", "atol=1e-5, rtol=1e-5")), [])

    def test_correctness_only_no_tf32_issue(self):
        code = KB_GRADER.replace('        speedup = info["baseline_ms"] / time_it(ModelNew(), inputs)\n'
                                 '        return min(speedup, 3.0)\n', "        return 1.0\n")
        self.assertEqual(self.hit("TF32", code), [])

    # unclipped-speedup-reward
    def test_unclipped_return(self):
        f = self.hit("no upper clip", KB_GRADER.replace("return min(speedup, 3.0)", "return speedup"))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "hygiene")

    def test_unclipped_harbor_reward_file(self):
        files = {"task.toml": "[verifier]\n", "instruction.md": "Make it fast.\n",
                 "tests/compute_reward.py": HARBOR_TIMER.replace("SYNC", "torch.cuda.synchronize()")}
        self.assertEqual(len(self.hit("no upper clip", files=files)), 1)

    def test_unclipped_scaled(self):
        f = self.hit("no upper clip", KB_GRADER.replace("return min(speedup, 3.0)", "reward = 0.5 * speedup\n        return reward"))
        self.assertEqual(len(f), 1)

    def test_speedup_named_callee(self):
        code = KB_GRADER.replace("return min(speedup, 3.0)", "return compute_speedup_reward(completion, info)")
        self.assertEqual(self.hit("no upper clip", code), [])

    def test_clipped(self):
        self.assertEqual(self.hit("no upper clip", KB_GRADER), [])

    def test_speedup_threshold(self):
        self.assertEqual(self.hit("no upper clip", KB_GRADER.replace("return min(speedup, 3.0)",
                                                                       "return 1.0 if speedup > 1.0 else 0.0")), [])

    def test_speedup_clamped_before(self):
        code = KB_GRADER.replace("return min(speedup, 3.0)", "speedup = max(0.0, min(speedup, 5.0))\n        return speedup")
        self.assertEqual(self.hit("no upper clip", code), [])

    # kernel-output-buffer-and-input-aliasing
    def alias_code(self, body):
        return KB_GRADER.replace("        ref_out = Model()(*inputs)\n        new_out = ModelNew()(*inputs)\n", body)

    def test_candidate_runs_first(self):
        code = self.alias_code("        new_out = ModelNew()(*inputs)\n        ref_out = Model()(*inputs)\n")
        self.assertEqual(len(self.hit("before the reference", code)), 1)

    def test_candidate_first_named_models(self):
        code = self.alias_code("        new_model, ref_model = ModelNew(), Model()\n"
                               "        new_out = new_model(*inputs)\n        ref_out = ref_model(*inputs)\n")
        self.assertEqual(len(self.hit("before the reference", code)), 1)

    def test_candidate_first_plain_args(self):
        code = self.alias_code("        x, = inputs\n        new_out = custom_model(x)\n        ref_out = reference(x)\n")
        self.assertEqual(len(self.hit("before the reference", code)), 1)

    def test_reference_runs_first(self):
        self.assertEqual(self.hit("before the reference", KB_GRADER), [])

    def test_clone_between(self):
        code = self.alias_code("        new_out = ModelNew()(*[x.clone() for x in inputs])\n        ref_out = Model()(*inputs)\n")
        self.assertEqual(self.hit("before the reference", code), [])

    def test_inputs_regenerated_between(self):
        code = self.alias_code("        new_out = ModelNew()(*inputs)\n        inputs = get_inputs()\n"
                               "        ref_out = Model()(*inputs)\n")
        self.assertEqual(self.hit("before the reference", code), [])


SWE_TASK = {
    "task.toml": "[verifier]\ntimeout_sec = 600\n",
    "instruction.md": "<pr_description>\nImporting a config file with an unknown key crashes the loader. It should "
                      "report the problem to the user instead.\n</pr_description>\n",
    "environment/Dockerfile": "FROM golang:1.22\nRUN git clone https://example.com/tool.git /app\n",
    "solution/solve.sh": "cd /app\ngit apply /solution/gold_patch.diff\n",
    "solution/gold_patch.diff": """
        diff --git a/config/load.go b/config/load.go
        --- a/config/load.go
        +++ b/config/load.go
        @@ -10,6 +10,14 @@ func Load(path string) (*Config, error) {
         	data, err := os.ReadFile(path)
        +	if err := ValidateKeys(raw); err != nil {
        +		return nil, err
        +	}
        +}
        +
        +func ValidateKeys(raw map[string]any) error {
        +	return fmt.Errorf("config contains an unknown key named %q", k)
        +	return errors.New("unknown configuration key found in file")
         }
    """,
    "tests/test.sh": "cd /app && git apply /tests/test_patch.patch && go test ./config/...\n",
    "tests/test_patch.patch": """
        diff --git a/config/load_test.go b/config/load_test.go
        --- a/config/load_test.go
        +++ b/config/load_test.go
        @@ -1,3 +1,9 @@
         package config
        +func TestValidateKeys(t *testing.T) {
        +	err := ValidateKeys(map[string]any{"bogus": 1})
        +	assert.EqualError(t, err, "unknown configuration key found in file")
        +}
    """,
}


def patched_pkg(files):
    """Package.from_dir does not load .patch and .diff files yet (a package.py change is requested in
    the internal notes), so the tests add them the way that change would."""
    p = pkg({k: v for k, v in files.items() if not k.endswith((".patch", ".diff"))})
    for k, v in files.items():
        if k.endswith((".patch", ".diff")):
            text = textwrap.dedent(v).lstrip("\n")
            p.files[k] = text
            p.sizes[k] = len(text)
    return p


def swe(**over):
    files = dict(SWE_TASK)
    files.update(over)
    return files


class SpecOracleV05(unittest.TestCase):
    R = "spec-oracle-gap"

    def msg(self, files):
        f = RULES[self.R]["fn"](patched_pkg(files))
        return " | ".join(x.message for x in f)

    # spec-gap-other-languages-and-layouts: SWE layout, any language
    def test_go_test_uses_new_symbol(self):
        m = self.msg(SWE_TASK)
        self.assertIn("`ValidateKeys` (new symbol)", m)

    def test_ts_test_uses_new_symbol(self):
        files = swe(**{"solution/gold_patch.diff": """
            diff --git a/src/util.ts b/src/util.ts
            --- a/src/util.ts
            +++ b/src/util.ts
            @@ -1,2 +1,5 @@
             import x from "y";
            +export function formatRoomTopic(topic: string): string {
            +    return topic.trim();
            +}
        """, "tests/test_patch.patch": """
            diff --git a/test/util-test.ts b/test/util-test.ts
            --- a/test/util-test.ts
            +++ b/test/util-test.ts
            @@ -1,2 +1,6 @@
             import { a } from "../src/util";
            +import { formatRoomTopic } from "../src/util";
            +it("trims", () => {
            +    expect(formatRoomTopic(" a ")).toBe("a");
            +});
        """})
        self.assertIn("`formatRoomTopic` (new symbol)", self.msg(files))

    def test_python_patch_private_helper(self):
        files = swe(**{"solution/gold_patch.diff": """
            diff --git a/pkg/version.py b/pkg/version.py
            --- a/pkg/version.py
            +++ b/pkg/version.py
            @@ -5,3 +5,6 @@ class ModuleInfo:
                 name = "x"
            +    def _reset_cache(self):
            +        self._version = None
        """, "tests/test_patch.patch": """
            diff --git a/tests/test_version.py b/tests/test_version.py
            --- a/tests/test_version.py
            +++ b/tests/test_version.py
            @@ -1,2 +1,4 @@
             from pkg import version
            +def test_cache():
            +    version.MODULE_INFO["a"]._reset_cache()
        """})
        self.assertIn("`_reset_cache` (new symbol)", self.msg(files))

    def test_new_symbol_named_in_issue(self):
        files = swe(**{"instruction.md": SWE_TASK["instruction.md"] + "Add a function ValidateKeys(raw) and return "
                                                                        "the error unknown configuration key found in file.\n"})
        self.assertEqual(self.msg(files), "")

    def test_symbol_existed_before_the_patch(self):
        files = swe(**{"solution/gold_patch.diff": SWE_TASK["solution/gold_patch.diff"].replace(
            "         	data, err := os.ReadFile(path)", "         	data, err := ValidateKeys(path)")})
        self.assertNotIn("ValidateKeys", self.msg(files))

    def test_symbol_defined_by_the_test_patch(self):
        files = swe(**{"tests/test_patch.patch": SWE_TASK["tests/test_patch.patch"].replace(
            "        +func TestValidateKeys", "        +func ValidateKeys(m map[string]any) error { return nil }\n"
                                              "        +func TestValidateKeys")})
        self.assertNotIn("`ValidateKeys`", self.msg(files))

    def test_name_only_inside_a_test_string(self):
        files = swe(**{"tests/test_patch.patch": SWE_TASK["tests/test_patch.patch"].replace(
            "        +	err := ValidateKeys(map[string]any{\"bogus\": 1})", "        +	name := \"ValidateKeys is skipped\"")})
        self.assertNotIn("`ValidateKeys`", self.msg(files))

    def test_go_method_with_receiver(self):
        files = swe(**{"solution/gold_patch.diff": SWE_TASK["solution/gold_patch.diff"].replace(
            "+func ValidateKeys(raw", "+func (s *Store) ValidateKeys(raw")})
        self.assertNotIn("`ValidateKeys`", self.msg(files))

    def test_literal_made_of_named_identifiers(self):
        files = task(**{"instruction.md": "Create /app/kv.proto with a service called KVStore.\n",
                        "solution/solve.sh": "cat > /app/kv.proto <<'EOF'\nservice KVStore {}\nEOF\n",
                        "tests/test_outputs.py": """
                            def test_proto():
                                content = open("/app/kv.proto").read()
                                assert "service KVStore" in content
                        """})
        self.assertNotIn("(literal)", self.msg(files))

    # test-literal-assertions-absent-from-instruction + tests-assert-gold-specific-strings
    def test_patch_literal_only_in_gold(self):
        self.assertIn("`unknown configuration key found in file` (literal)", self.msg(SWE_TASK))

    def test_python_literal_in_output(self):
        files = task(**{"solution/solve.sh": SPEC_POS["solution/solve.sh"].replace(
            "    return [a] * n_samples", "    raise ValueError('density is not log-concave here')"),
            "tests/test_outputs.py": """
                import pytest
                import sys
                sys.path.insert(0, "/app")
                import sampler
                def test_bad_density():
                    with pytest.raises(ValueError, match="density is not log-concave here"):
                        sampler.ars(lambda x: x ** 3, -1.0, 1.0)
            """})
        self.assertIn("`density is not log-concave here` (literal)", self.msg(files))

    def test_python_literal_stdout(self):
        files = task(**{"solution/solve.sh": "cat > /app/cli.py <<'EOF'\nprint('All records were imported cleanly')\nEOF\n",
                        "instruction.md": "Write a CLI at /app/cli.py that imports the records and reports success.\n",
                        "tests/test_outputs.py": """
                            import subprocess
                            def test_cli():
                                out = subprocess.run(["python3", "/app/cli.py"], capture_output=True, text=True).stdout
                                assert "All records were imported cleanly" in out
                        """})
        self.assertIn("(literal)", self.msg(files))

    def test_literal_in_instruction(self):
        files = task(**{"solution/solve.sh": "cat > /app/cli.py <<'EOF'\nprint('All records were imported cleanly')\nEOF\n",
                        "instruction.md": "Write /app/cli.py. On success print: All records were imported cleanly.\n",
                        "tests/test_outputs.py": """
                            import subprocess
                            def test_cli():
                                out = subprocess.run(["python3", "/app/cli.py"], capture_output=True, text=True).stdout
                                assert "All records were imported cleanly" in out
                        """})
        self.assertNotIn("(literal)", self.msg(files))

    def test_literal_not_in_oracle(self):
        files = task(**{"tests/test_outputs.py": """
                            import subprocess
                            def test_cli():
                                out = subprocess.run(["python3", "/app/cli.py"], capture_output=True, text=True).stdout
                                assert "All records were imported cleanly" in out
                        """})
        self.assertNotIn("(literal)", self.msg(files))

    def test_literal_is_also_test_input(self):
        files = task(**{"solution/solve.sh": "cat > /app/echo.py <<'EOF'\nimport sys\nprint(sys.argv[1])\n"
                                             "# hello there general kenobi\nEOF\n",
                        "tests/test_outputs.py": """
                            import subprocess
                            MSG = "hello there general kenobi"
                            def test_echo():
                                out = subprocess.run(["python3", "/app/echo.py", "hello there general kenobi"],
                                                     capture_output=True, text=True).stdout
                                assert "hello there general kenobi" in out
                        """})
        self.assertNotIn("(literal)", self.msg(files))

    def test_short_literal(self):
        files = task(**{"solution/solve.sh": "cat > /app/cli.py <<'EOF'\nprint('ok done')\nEOF\n",
                        "tests/test_outputs.py": """
                            import subprocess
                            def test_cli():
                                out = subprocess.run(["python3", "/app/cli.py"], capture_output=True, text=True).stdout
                                assert "ok done" in out
                        """})
        self.assertNotIn("(literal)", self.msg(files))

    def test_literal_existed_before_gold_patch(self):
        files = swe(**{"solution/gold_patch.diff": SWE_TASK["solution/gold_patch.diff"].replace(
            "         }\n", '         	log("unknown configuration key found in file")\n         }\n')})
        self.assertNotIn("(literal)", self.msg(files))

    # exact listing order
    def test_unsorted_listdir(self):
        files = task(**{"tests/test_outputs.py": """
            import os
            def test_files():
                assert os.listdir("/app/out") == ["a.txt", "b.txt"]
        """})
        self.assertIn("unsorted directory listing", self.msg(files))

    def test_unsorted_glob(self):
        files = task(**{"tests/test_outputs.py": """
            import glob
            def test_files():
                assert glob.glob("/app/out/*.csv") == ["/app/out/a.csv", "/app/out/b.csv"]
        """})
        self.assertIn("unsorted directory listing", self.msg(files))

    def test_unsorted_iterdir(self):
        files = task(**{"tests/test_outputs.py": """
            from pathlib import Path
            def test_files():
                assert list(Path("/app/out").iterdir()) == [Path("/app/out/a"), Path("/app/out/b")]
        """})
        self.assertIn("unsorted directory listing", self.msg(files))

    def test_sorted_listdir(self):
        files = task(**{"tests/test_outputs.py": """
            import os
            def test_files():
                assert sorted(os.listdir("/app/out")) == ["a.txt", "b.txt"]
        """})
        self.assertNotIn("unsorted", self.msg(files))

    def test_single_entry_listdir(self):
        files = task(**{"tests/test_outputs.py": """
            import os
            def test_files():
                assert os.listdir("/app/out") == ["a.txt"]
        """})
        self.assertNotIn("unsorted", self.msg(files))

    def test_listdir_of_test_fixtures(self):
        files = task(**{"tests/test_outputs.py": """
            import os
            def test_files():
                assert os.listdir("/tests/fixtures") == ["a.txt", "b.txt"]
        """})
        self.assertNotIn("unsorted", self.msg(files))

    # tests-import-private-helpers
    def test_private_import_from_repo_module(self):
        files = swe(**{"solution/gold_patch.diff": """
            diff --git a/lib/parse.py b/lib/parse.py
            --- a/lib/parse.py
            +++ b/lib/parse.py
            @@ -1,2 +1,4 @@
             import re
            +def _split_header(line):
            +    return line.split(":", 1)
        """, "tests/test_parse.py": """
            from lib.parse import _split_header
            def test_split():
                assert _split_header("a: b") == ["a", " b"]
        """})
        self.assertIn("`_split_header` (private)", self.msg(files))

    def test_private_import_named_in_issue(self):
        files = swe(**{"instruction.md": "Add a helper _split_header(line) to lib/parse.py.\n",
                       "solution/gold_patch.diff": "--- a/lib/parse.py\n+++ b/lib/parse.py\n@@ -1 +1,2 @@\n import re\n+def _split_header(line):\n",
                       "tests/test_parse.py": "from lib.parse import _split_header\n"})
        self.assertNotIn("_split_header", self.msg(files))

    def test_private_import_existing_helper(self):
        files = swe(**{"solution/gold_patch.diff": "--- a/lib/parse.py\n+++ b/lib/parse.py\n@@ -1,2 +1,2 @@\n"
                                                   "-def _split_header(line):\n+def _split_header(line, sep=':'):\n",
                       "tests/test_parse.py": "from lib.parse import _split_header\n"})
        self.assertNotIn("_split_header", self.msg(files))



class UnclippedFixWave(unittest.TestCase):
    """0.5.0 fix wave (check_4.md): the no-upper-clip branch stays low, and a trimmed, paired geomean is skipped."""
    R = "kernel-perf-grader"
    REWARD = """
        import time
        import torch
        import numpy as np
        from scipy import stats
        def main():
            x = torch.randn(1024, device="cuda")
            ratios = []
            for base, cand in pairs():
                t0 = time.perf_counter()
                ratios.append(base / cand)
            speedup = AGG
            open("/logs/verifier/reward.txt", "w").write(str(speedup))
    """

    def hit(self, agg, note=""):
        files = {"task.toml": "[verifier]\n", "instruction.md": "Make it fast.\n",
                 "tests/compute_reward.py": self.REWARD.replace("AGG", agg) + note}
        return only(fire(self.R, files=files), "no upper clip")

    def test_pos_plain_mean_stays_low(self):
        f = self.hit("np.mean(ratios)")
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].severity, "low")

    def test_pos_geomean_not_trimmed(self):
        self.assertEqual(len(self.hit("stats.gmean(ratios)", "        # paired runs\n")), 1)

    def test_pos_trimmed_geomean_not_paired(self):
        self.assertEqual(len(self.hit("stats.gmean(stats.trimboth(ratios, 0.1))")), 1)

    def test_neg_trimmed_paired_gmean(self):
        self.assertEqual(self.hit("stats.gmean(stats.trimboth(ratios, 0.1))", "        # paired, interleaved runs\n"), [])

    def test_neg_exp_mean_log_trim_mean(self):
        self.assertEqual(self.hit("float(np.exp(np.mean(np.log(stats.trim_mean(ratios, 0.1)))))",
                                  "        # base and candidate are interleaved\n"), [])

    def test_neg_geometric_mean_winsorized_pairwise(self):
        self.assertEqual(self.hit("geometric_mean(winsorize(ratios))", "        # pairwise speedups\n"), [])


if __name__ == "__main__":
    unittest.main()


class OffByDefault(unittest.TestCase):
    """spec-oracle-gap measured 0 of 2 precision and 0 of 19 recall on Terminal-Bench 2.0, and
    judge-config-hygiene 0 of 4 on the Hub, so they run only when asked for."""

    def test_spec_oracle_gap_off_unless_enabled(self):
        import envscan.rules as R
        p = pkg(SPEC_POS)
        default, _ = run_rules(p)
        self.assertNotIn("spec-oracle-gap", {f.rule for f in default})
        R.ENABLED.add("spec-oracle-gap")
        try:
            enabled, _ = run_rules(p)
        finally:
            R.ENABLED.discard("spec-oracle-gap")
        self.assertIn("spec-oracle-gap", {f.rule for f in enabled})
        explicit, _ = run_rules(p, only={"spec-oracle-gap"})
        self.assertEqual([f.rule for f in explicit], ["spec-oracle-gap"])

    def test_judge_config_hygiene_off_unless_enabled(self):
        # 0 of 4 Hub hits real in the independent check (the internal notes)
        p = pkg({"pyproject.toml": PYPROJECT, "env.py": BUDGET_POS})
        default, _ = run_rules(p)
        self.assertNotIn("judge-config-hygiene", {f.rule for f in default})
        explicit, _ = run_rules(p, only={"judge-config-hygiene"})
        self.assertEqual([f.rule for f in explicit], ["judge-config-hygiene"])

    def test_cli_enable_flag(self):
        from tests.cli_helpers import run
        code, out, err = run("--enable", "no-such-rule", ".")
        self.assertEqual(code, 3)
        self.assertIn("unknown rule", err)

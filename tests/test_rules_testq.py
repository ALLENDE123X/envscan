"""Tests for the graded test strength rules (envscan/rules_testq.py).

Each case writes a few small files into a temp folder and loads them with Package.from_dir.
Nothing in the fixtures is imported or run.
"""
import os
import shutil
import tempfile
import textwrap
import unittest

from envscan.package import Package
from envscan.rules import RULES

TOML = '[metadata]\nname = "t"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n'
PYTEST_SH = "#!/bin/bash\npytest /tests/test_outputs.py -rA\nif [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n"
PYPROJECT = '[project]\nname = "x"\ndependencies = ["verifiers>=0.1.3"]\n'


def pkg(files: dict) -> Package:
    d = tempfile.mkdtemp(prefix="envscan-testq-")
    try:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(text).lstrip("\n"))
        return Package.from_dir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def task(test_py=None, test_sh=PYTEST_SH, instruction="Do the task.\n", extra=None):
    files = {"task.toml": TOML, "instruction.md": instruction, "tests/test.sh": test_sh,
             "solution/solve.sh": "#!/bin/bash\necho done\n"}
    if test_py is not None:
        files["tests/test_outputs.py"] = test_py
    files.update(extra or {})
    return files


def fire(rid, files):
    return RULES[rid]["fn"](pkg(files))


# ---------------------------------------------------------------------------

class AnyOfN(unittest.TestCase):
    R = "any-of-n-pass-condition"

    def test_any_over_pairs(self):
        code = '''
        from solution import similarity
        PAIRS = [("king", "queen"), ("man", "woman"), ("paris", "france"), ("cat", "dog")]
        def test_analogies():
            assert any(similarity(a, b) > 0.5 for a, b in PAIRS)
        '''
        fs = fire(self.R, task(code))
        self.assertEqual(len(fs), 1)
        self.assertIn("1 of 4", fs[0].message)
        self.assertEqual(fs[0].confidence, "review")

    def test_counter_ge_one(self):
        code = '''
        from solution import solve
        def test_cases():
            cases = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]
            ok = 0
            for x, want in cases:
                if solve(x) == want:
                    ok += 1
            assert ok >= 1
        '''
        self.assertEqual(len(fire(self.R, task(code))), 1)

    def test_sum_gt_zero(self):
        code = '''
        import solution
        def test_some():
            assert sum(solution.check(x) for x in [1, 2, 3, 4, 5, 6]) > 0
        '''
        self.assertEqual(len(fire(self.R, task(code))), 1)

    def test_low_fraction_threshold(self):
        code = '''
        from solution import solve
        CASES = [1, 2, 3, 4, 5, 6, 7, 8]
        def test_frac():
            good = 0
            for c in CASES:
                if solve(c) == c * 2:
                    good += 1
            assert good / len(CASES) >= 0.2
        '''
        self.assertEqual(len(fire(self.R, task(code))), 1)

    def test_neg_all_required(self):
        code = '''
        from solution import solve
        CASES = [(1, 2), (2, 4), (3, 6)]
        def test_all():
            assert all(solve(x) == y for x, y in CASES)
            ok = sum(1 for x, y in CASES if solve(x) == y)
            assert ok >= 3
        '''
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_alternatives_of_one_answer(self):
        code = '''
        def test_answer():
            out = open("/app/answer.txt").read().strip()
            assert any(out == alt for alt in ["4", "four", "4.0"])
            assert any(out.endswith(x) for x in [".so", ".pyd", ".dll"])
        '''
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_file_exists_any(self):
        code = '''
        from pathlib import Path
        def test_files():
            assert any(Path(f).exists() for f in ["a.txt", "b.txt", "c.txt"])
        '''
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_not_harbor(self):
        code = '''
        def test_x():
            assert any(check(a) > 0 for a in [1, 2, 3, 4])
        '''
        self.assertEqual(fire(self.R, {"pyproject.toml": PYPROJECT, "tests/test_x.py": code}), [])


# ---------------------------------------------------------------------------

class NoP2P(unittest.TestCase):
    R = "no-pass-to-pass-regression-tests"

    def test_vf_reads_only_f2p(self):
        code = '''
        import verifiers as vf
        async def run_tests(state, info, **kw):
            tests = info["FAIL_TO_PASS"]
            out = await state["sandbox"].run("pytest " + " ".join(tests))
            return 1.0 if out.exit_code == 0 else 0.0
        def load_environment():
            return vf.Rubric(funcs=[run_tests])
        '''
        fs = fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code})
        self.assertEqual(len(fs), 1)
        self.assertIn("pass to pass", fs[0].message)

    def test_harbor_config_only_f2p(self):
        files = task(None, test_sh="#!/bin/bash\npython /tests/grade.py\n",
                     extra={"tests/config.json": '{"instance_id": "x", "FAIL_TO_PASS": ["t::a"]}',
                            "tests/grade.py": "import json\ncfg = json.load(open('/tests/config.json'))\nneed = cfg['FAIL_TO_PASS']\n"})
        self.assertEqual(len(fire(self.R, files)), 1)

    def test_harbor_runs_only_patch_files(self):
        sh = '''
        #!/bin/bash
        cd /app
        git apply - <<'EOF'
        diff --git a/tests/test_api.py b/tests/test_api.py
        --- a/tests/test_api.py
        +++ b/tests/test_api.py
        @@ -1 +1,2 @@
        +def test_new(): pass
        EOF
        python -m pytest tests/test_api.py -x
        if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
        '''
        files = task(None, test_sh=sh, extra={"solution/solve.sh": "#!/bin/bash\ncd /app\ngit apply /solution/fix.patch\n"})
        fs = fire(self.R, files)
        self.assertEqual(len(fs), 1)
        self.assertIn("test.sh", fs[0].evidence[0])

    def test_neg_vf_has_p2p(self):
        code = '''
        def run_tests(info, **kw):
            need = info["FAIL_TO_PASS"] + info["PASS_TO_PASS"]
            return 1.0
        '''
        self.assertEqual(fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code}), [])

    def test_neg_harbor_full_suite(self):
        sh = '''
        #!/bin/bash
        cd /app
        git apply - <<'EOF'
        diff --git a/tests/test_api.py b/tests/test_api.py
        +def test_new(): pass
        EOF
        python -m pytest tests/test_api.py tests/
        '''
        files = task(None, test_sh=sh, extra={"solution/solve.sh": "#!/bin/bash\ngit apply /solution/fix.patch\n"})
        self.assertEqual(fire(self.R, files), [])

    def test_neg_not_swe(self):
        files = task("def test_a():\n    assert open('/app/x').read() == 'y'\n")
        self.assertEqual(fire(self.R, files), [])

    def test_neg_harbor_config_with_p2p(self):
        files = task(None, test_sh="#!/bin/bash\npython /tests/grade.py\n",
                     extra={"tests/config.json": '{"FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": ["t::b"]}'})
        self.assertEqual(fire(self.R, files), [])


# ---------------------------------------------------------------------------

class ThinSuite(unittest.TestCase):
    R = "thin-test-suite"

    def test_two_asserts_on_agent_code(self):
        code = '''
        import importlib.util
        def load():
            spec = importlib.util.spec_from_file_location("solution", "/app/solution.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        def test_encode():
            assert load().encode_message("hello") == "BOFFE"
        def test_roundtrip():
            mod = load()
            assert mod.encode_message(mod.encode_message("harbor")) == "harbor"
        '''
        fs = fire(self.R, task(code))
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].kind, "hygiene")

    def test_one_import_check(self):
        code = '''
        import sys
        sys.path.insert(0, "/app")
        def test_add():
            from calc import add
            assert add(2, 3) == 5
        '''
        self.assertEqual(len(fire(self.R, task(code))), 1)

    def test_dataset_rows(self):
        rows = "\n".join('{"question": "q%d", "test_list": ["assert f(1) == 1", "assert f(2) == 2"]}' % i for i in range(12))
        code = '''
        import verifiers as vf
        def reward(completion, info, **kw):
            tests = info["test_list"]
            return 1.0
        '''
        fs = fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code, "data/train.jsonl": rows})
        self.assertEqual(len(fs), 1)
        self.assertIn("median", fs[0].message)

    def test_neg_parametrize(self):
        code = '''
        import pytest, subprocess
        @pytest.mark.parametrize("x", [1, 2, 3, 4, 5, 6])
        def test_many(x):
            assert subprocess.run(["/app/run", str(x)]).returncode == 0
        '''
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_loop_over_cases(self):
        code = '''
        import subprocess, json
        def test_all():
            for case in json.load(open("/tests/cases.json")):
                assert subprocess.run(["/app/run", case]).returncode == 0
        '''
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_one_goal_task(self):
        code = '''
        import subprocess
        def test_ssh():
            out = subprocess.run(["ssh", "-p", "2222", "root@localhost", "uname -r"], capture_output=True, text=True).stdout
            assert "6.6.4" in out
        '''
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_answer_file(self):
        code = '''
        def test_answer():
            assert open("/app/answer.txt").read().strip() == "B"
        '''
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_many_asserts(self):
        code = "import sys\nsys.path.insert(0, '/app')\nfrom calc import add\n" + "".join(
            f"def test_{i}():\n    assert add({i}, 1) == {i + 1}\n" for i in range(6))
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_dataset_many_tests(self):
        rows = "\n".join('{"question": "q%d", "test_list": ["a", "b", "c", "d", "e", "f"]}' % i for i in range(12))
        code = "def reward(info, **kw):\n    return len(info['test_list'])\n"
        self.assertEqual(fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code, "data/train.jsonl": rows}), [])


# ---------------------------------------------------------------------------

class NetClock(unittest.TestCase):
    R = "tests-call-network-or-clock"

    def test_urlretrieve_github(self):
        code = '''
        import urllib.request
        def fetch():
            url = "https://github.com/x/y/archive/master.zip"
            urllib.request.urlretrieve(url, "/tmp/a.zip")
        def test_x():
            fetch()
            assert True
        '''
        fs = fire(self.R, task(code))
        self.assertEqual(len(fs), 1)
        self.assertIn("github.com", fs[0].message)

    def test_requests_httpbin(self):
        code = '''
        import requests
        def test_echo():
            r = requests.get("https://httpbin.org/get")
            assert r.status_code == 200
        '''
        self.assertEqual(len(fire(self.R, task(code))), 1)

    def test_shell_curl_in_check(self):
        sh = PYTEST_SH + 'if [ "$(curl -s https://api.example.com/v1/rate)" = "$(cat /app/out.txt)" ]; then echo ok; fi\n'
        self.assertEqual(len(fire(self.R, task("def test_a():\n    assert 1\n", test_sh=sh))), 1)

    def test_today_in_assert(self):
        code = '''
        from datetime import date
        def test_report():
            today = date.today().isoformat()
            assert open("/app/report.txt").read().startswith(today)
        '''
        fs = fire(self.R, task(code))
        self.assertEqual(len(fs), 1)
        self.assertIn("date", fs[0].message)

    def test_neg_localhost(self):
        code = '''
        import requests
        def test_server():
            assert requests.get("http://localhost:8000/health").status_code == 200
            assert requests.get("http://app:8080/x").ok
        '''
        self.assertEqual(fire(self.R, task(code)), [])

    def test_neg_installer_curl(self):
        sh = "#!/bin/bash\ncurl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh\n" + PYTEST_SH
        self.assertEqual(fire(self.R, task("def test_a():\n    assert 1\n", test_sh=sh)), [])

    def test_neg_timing_assert(self):
        code = '''
        import time
        def test_fast():
            t = time.time()
            assert time.time() - t < 5
        '''
        self.assertEqual(fire(self.R, task(code)), [])


# ---------------------------------------------------------------------------

MULTI_INS = "Given a graph, print a topological order.\nIf there are multiple valid answers, print any of them.\n"


class MultiAnswer(unittest.TestCase):
    R = "multi-answer-exact-diff-no-checker"

    def test_exact_stdout(self):
        code = '''
        import subprocess
        def test_out():
            out = subprocess.run(["/app/solve"], input=open("/tests/in.txt").read(), capture_output=True, text=True).stdout
            expected = open("/tests/expected.txt").read()
            assert out.strip() == expected.strip()
        '''
        fs = fire(self.R, task(code, instruction=MULTI_INS))
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].kind, "broken")

    def test_shell_diff(self):
        sh = "#!/bin/bash\n/app/solve < /tests/in.txt > /tmp/out.txt\nif diff -q /tmp/out.txt /tests/expected_output.txt; then echo 1 > /logs/verifier/reward.txt; fi\n"
        self.assertEqual(len(fire(self.R, task(None, test_sh=sh, instruction=MULTI_INS))), 1)

    def test_vf_rows(self):
        rows = "\n".join('{"question": "Find a pair. If there are several solutions, print any of them.", "answer": "1 2"}' for _ in range(5))
        code = '''
        def reward(completion, answer, **kw):
            stdout = run(completion)
            return 1.0 if stdout.strip() == answer.strip() else 0.0
        '''
        fs = fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code, "data/test.jsonl": rows})
        self.assertEqual(len(fs), 1)

    def test_neg_checker(self):
        code = '''
        def is_valid_order(out, graph):
            return True
        def test_out():
            output = open("/app/out.txt").read()
            assert is_valid_order(output, None)
        '''
        self.assertEqual(fire(self.R, task(code, instruction=MULTI_INS)), [])

    def test_neg_single_answer(self):
        code = '''
        def test_out():
            output = open("/app/out.txt").read()
            assert output.strip() == open("/tests/expected.txt").read().strip()
        '''
        self.assertEqual(fire(self.R, task(code, instruction="Print the sum.\n")), [])

    def test_neg_sorted_compare(self):
        code = '''
        def test_out():
            output = open("/app/out.txt").read().split()
            expected = open("/tests/expected.txt").read().split()
            assert sorted(output) == sorted(expected)
        '''
        self.assertEqual(fire(self.R, task(code, instruction=MULTI_INS)), [])


# ---------------------------------------------------------------------------

class BelowRange(unittest.TestCase):
    R = "test-inputs-below-constraint-range"
    INS = "Read n and print the n-th prime.\n\nConstraints: 1 ≤ n ≤ 10^6\n"

    def test_small_literals(self):
        code = '''
        import subprocess
        def test_small():
            for n, p in [(1, 2), (2, 3), (5, 11), (10, 29)]:
                assert subprocess.run(["/app/prime", str(n)], capture_output=True, text=True).stdout.strip() == str(p)
        '''
        fs = fire(self.R, task(code, instruction=self.INS))
        self.assertEqual(len(fs), 1)
        self.assertIn("1e+06", fs[0].message)

    def test_le_symbol(self):
        code = "def test_a():\n    assert run(7) == 17\n"
        self.assertEqual(len(fire(self.R, task(code, instruction="Given n (n <= 200000), output x.\n"))), 1)

    def test_latex_bound(self):
        code = "def test_a():\n    assert run(3) == 6\n"
        self.assertEqual(len(fire(self.R, task(code, instruction="$1 \\le n \\le 2 \\times 10^5$\n"))), 1)
        self.assertEqual(len(fire(self.R, task(code, instruction="$1 \\leq n \\leq 10^5$\n"))), 1)

    def test_neg_generator(self):
        code = '''
        import random
        def test_big():
            n = random.randint(1, 10**6)
            assert run(n) > 0
        '''
        self.assertEqual(fire(self.R, task(code, instruction=self.INS)), [])

    def test_neg_large_literal(self):
        code = "def test_a():\n    assert run(999983) == 15485863\n"
        self.assertEqual(fire(self.R, task(code, instruction=self.INS)), [])

    def test_neg_no_bound(self):
        code = "def test_a():\n    assert run(3) == 5\n"
        self.assertEqual(fire(self.R, task(code, instruction="Print the n-th prime.\n")), [])

    def test_neg_big_data_file(self):
        code = "def test_a():\n    assert run(open('/tests/big.in').read())\n"
        extra = {"tests/big.in": "1\n" * 30000}
        self.assertEqual(fire(self.R, task(code, instruction=self.INS, extra=extra)), [])


# ---------------------------------------------------------------------------

class PartialCredit(unittest.TestCase):
    R = "partial-credit-pass-rate-reward"

    def test_vf_passed_over_len(self):
        code = '''
        import verifiers as vf
        async def test_pass_reward(completion, info, **kw):
            results = [run_test(completion, t) for t in info["tests"]]
            passed = sum(1 for ok in results if ok)
            return passed / len(results)
        def load_environment():
            return vf.Rubric(funcs=[test_pass_reward], weights=[1.0])
        '''
        fs = fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code})
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].kind, "hygiene")

    def test_vf_base_plus_bonus(self):
        code = '''
        import verifiers as vf
        def code_reward(completion, info, **kw):
            passed_tests, total_tests = 0, len(info["test_cases"])
            for t in info["test_cases"]:
                if execute(completion, t):
                    passed_tests += 1
            base = passed_tests / total_tests
            bonus = 0.5 if passed_tests == total_tests else 0.0
            return base + bonus
        rubric = vf.Rubric(funcs=[code_reward])
        '''
        self.assertEqual(len(fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code})), 1)

    def test_harbor_evaluator_writes_fraction(self):
        code = '''
        import json
        def summarize(results):
            n_passed = sum(r == "passed" for r in results)
            n_tests = len(results)
            pass_rate = n_passed / n_tests if n_tests else 0.0
            return {"reward": pass_rate, "resolved": 1.0 if pass_rate >= 1.0 else 0.0}
        def main():
            json.dump(summarize(run_all()), open("/logs/verifier/reward.json", "w"))
        '''
        files = task(None, test_sh="#!/bin/bash\npython3 /tests/evaluator.py\n", extra={"tests/evaluator.py": code})
        self.assertEqual(len(fire(self.R, files)), 1)

    def test_shell_bc_fraction(self):
        sh = '''
        #!/bin/bash
        PASSED=$(grep -c PASS /tmp/log)
        TOTAL=$(grep -c -E "PASS|FAIL" /tmp/log)
        SCORE=$(echo "scale=3; $PASSED / $TOTAL" | bc)
        echo $SCORE > /logs/verifier/reward.txt
        '''
        self.assertEqual(len(fire(self.R, task(None, test_sh=sh))), 1)

    def test_neg_all_pass_gate(self):
        code = '''
        import verifiers as vf
        def test_reward(completion, info, **kw):
            passed = sum(run_test(completion, t) for t in info["tests"])
            total = len(info["tests"])
            if passed < total:
                return 0.0
            return passed / total
        rubric = vf.Rubric(funcs=[test_reward])
        '''
        self.assertEqual(fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code}), [])

    def test_neg_metric_only(self):
        code = '''
        import verifiers as vf
        def passed(state, **kw):
            return float(state["passed"])
        def pass_rate(state, **kw):
            res = state["test_results"]
            return sum(1 for r in res if r) / len(res) if res else 0.0
        rubric = vf.Rubric(funcs=[passed, pass_rate], weights=[1.0, 0.0])
        '''
        self.assertEqual(fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code}), [])

    def test_neg_not_tests(self):
        code = '''
        import verifiers as vf
        def style_reward(completion, **kw):
            scores = [has_waypoints(completion), has_gripper(completion)]
            return sum(scores) / len(scores)
        def f1(completion, answer, **kw):
            common = overlap(completion, answer)
            return common / len(answer)
        rubric = vf.Rubric(funcs=[style_reward, f1])
        '''
        self.assertEqual(fire(self.R, {"pyproject.toml": PYPROJECT, "env.py": code}), [])

    def test_neg_binary_harbor(self):
        self.assertEqual(fire(self.R, task("def test_a():\n    assert 1\n")), [])


if __name__ == "__main__":
    unittest.main()

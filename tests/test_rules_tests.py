"""Tests for the "tests" rule family (envscan/rules_tests.py).

Each case builds a small package from inline strings in a temp folder and runs one rule on it.
Run from the scanner folder: python3 -m unittest discover -s tests -t .
"""
import os
import tempfile
import textwrap
import unittest

from envscan import rules as R
from envscan.package import Package
import envscan.rules_tests as RT


def make_pkg(files: dict) -> Package:
    with tempfile.TemporaryDirectory() as d:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(textwrap.dedent(text).lstrip("\n"))
        return Package.from_dir(d)  # the package keeps file text in memory


def run(rid: str, files: dict):
    return R.RULES[rid]["fn"](make_pkg(files))


def evidence(findings):
    return [e for f in findings for e in f.evidence]


SHARED_TOML = """
schema_version = "1.4"
[task]
name = "demo/task"
[verifier]
timeout_sec = 60.0
"""

SEPARATE_APP_TOML = """
schema_version = "1.4"
artifacts = ["/app"]
[task]
name = "demo/task"
[verifier]
environment_mode = "separate"
"""

SEPARATE_FILE_TOML = """
schema_version = "1.4"
artifacts = ["/app/solution.py"]
[task]
name = "demo/task"
[verifier]
environment_mode = "separate"
"""

DOCKER_APP = "FROM python:3.12-slim\nWORKDIR /app\n"


def harbor(test_sh, toml=SHARED_TOML, tests=None, dockerfile=DOCKER_APP, extra=None):
    files = {"task.toml": toml, "instruction.md": "Write /app/solution.py.\n",
             "environment/Dockerfile": dockerfile, "tests/test.sh": test_sh,
             "solution/solve.sh": "#!/bin/bash\necho done\n"}
    for name, text in (tests or {}).items():
        files[f"tests/{name}"] = text
    files.update(extra or {})
    return files


IMPORTING_TEST = """
import importlib.util
import sys
from pathlib import Path


def load():
    path = Path("/app/solution.py")
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_add():
    mod = load()
    assert mod.add(2, 3) == 5
"""

SUBPROCESS_TEST = """
import subprocess


def test_cli():
    out = subprocess.run(["python3", "/app/solution.py", "2", "3"], capture_output=True, text=True)
    assert out.stdout.strip() == "5"
"""

EXIT_VERDICT_SH = """
#!/bin/bash
cd /app
if pytest /tests/test_outputs.py -rA; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
"""


# ---------------------------------------------------------------------------
# grader-import-surface
# ---------------------------------------------------------------------------

class GraderImportSurface(unittest.TestCase):
    RID = "grader-import-surface"

    def test_pos_separate_verifier_collects_copied_workspace(self):
        sh = "#!/bin/bash\ncd /app\npytest tests/ -rA\necho $? > /logs/verifier/reward.txt\n"
        fs = run(self.RID, harbor(sh, toml=SEPARATE_APP_TOML))
        self.assertEqual(evidence(fs), ["tests/test.sh:3"])
        self.assertIn("conftest.py", fs[0].message)
        self.assertEqual(fs[0].confidence, "review")

    def test_pos_isolated_replay_runs_pytest_in_repo(self):
        # a fresh box gets the agent's patch, then pytest runs in the repo with no --noconftest
        env = """
        import verifiers as vf

        REPO = "/testbed"

        async def solved(state, sandbox_client, **kw):
            sid = await sandbox_client.create_sandbox(image=state["image"], fresh_copy=True)
            await sandbox_client.execute_command(sid, f"cd {REPO} && git apply /tmp/patch.diff")
            res = await sandbox_client.execute_command(sid, f"cd {REPO} && python -m pytest tests/test_issue.py -q")
            return 1.0 if res.exit_code == 0 else 0.0

        def load_environment(**kw):
            return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[solved]))
        """
        fs = run(self.RID, {"env.py": env, "pyproject.toml": "[project]\nname='x'\ndependencies=['verifiers']\n"})
        self.assertEqual(len(fs), 1)
        self.assertIn("env.py:", fs[0].evidence[0])

    def test_pos_pythonpath_puts_workspace_first(self):
        sh = "#!/bin/bash\nexport PYTHONPATH=/app:$PYTHONPATH\npytest /tests/test_outputs.py\n"
        # no task.toml: a tests/test.sh shipped inside some other package
        fs = run(self.RID, {"tests/test.sh": sh, "environment/Dockerfile": DOCKER_APP})
        self.assertEqual(evidence(fs), ["tests/test.sh:3"])
        self.assertIn("PYTHONPATH", fs[0].message)

    def test_pos_heredoc_python_from_workspace(self):
        sh = "#!/bin/bash\ncd /app\npython3 - <<'PY'\nimport json\nprint(json.load(open('/logs/out.json')))\nPY\n"
        fs = run(self.RID, harbor(sh, toml=SEPARATE_APP_TOML))
        self.assertEqual(evidence(fs), ["tests/test.sh:3"])
        self.assertIn("sys.path", fs[0].message)

    def test_pos_grader_script_inside_agent_tree(self):
        sh = "#!/bin/sh\nexec python3 /app/tests/grader.py --submission /app/out.mp4\n"
        fs = run(self.RID, {"tests/test.sh": sh, "environment/Dockerfile": DOCKER_APP})
        self.assertEqual(evidence(fs), ["tests/test.sh:2"])

    def test_neg_noconftest(self):
        sh = "#!/bin/bash\ncd /app\npytest --noconftest tests/ -rA\n"
        self.assertEqual(run(self.RID, harbor(sh, toml=SEPARATE_APP_TOML)), [])

    def test_neg_only_a_file_crosses_to_the_verifier(self):
        sh = "#!/bin/bash\ncd /app\npytest tests/ -rA\n"
        self.assertEqual(run(self.RID, harbor(sh, toml=SEPARATE_FILE_TOML)), [])

    def test_neg_tests_from_verifier_dir_and_isolated_python(self):
        sh = ("#!/bin/bash\ncd /app\npytest /tests/test_outputs.py -rA\n"
              "python3 -I -m pytest /tests/test_outputs.py\npython3 /tests/parse.py\n")
        self.assertEqual(run(self.RID, harbor(sh, toml=SEPARATE_APP_TOML)), [])

    def test_neg_shared_box_already_reported(self):
        # harbor-shared-grading reports a shared mode task; this rule stays quiet unless overlap is allowed
        sh = "#!/bin/bash\ncd /app\npytest tests/\n"
        files = harbor(sh)
        self.assertTrue(R.RULES["harbor-shared-grading"]["fn"](make_pkg(files)))
        self.assertEqual(run(self.RID, files), [])
        old = RT.SUPPRESS_SHARED_BOX_OVERLAP
        try:
            RT.SUPPRESS_SHARED_BOX_OVERLAP = False
            self.assertEqual(evidence(run(self.RID, files)), ["tests/test.sh:3"])
        finally:
            RT.SUPPRESS_SHARED_BOX_OVERLAP = old

    def test_pos_shared_box_with_non_root_agent(self):
        toml = SHARED_TOML + '[agent]\nuser = "agent"\n'
        sh = "#!/bin/bash\ncd /app\npython3 -c 'import json; print(1)'\n"
        self.assertEqual(evidence(run(self.RID, harbor(sh, toml=toml))), ["tests/test.sh:3"])

    def test_neg_install_lines_are_not_runs(self):
        sh = ("#!/bin/bash\ncd /app\npip install pytest==8.4.1\nuv add pytest==8.4.1\n"
              "echo 'run pytest later'\n")
        self.assertEqual(run(self.RID, harbor(sh, toml=SEPARATE_APP_TOML)), [])

    def test_pos_namespace_copied_into_tests(self):
        test = IMPORTING_TEST.replace("def test_add():", "_MOD = load()\nfor _k, _v in _MOD.__dict__.items():\n"
                                      "    globals()[_k] = _v\n\n\ndef test_add():")
        fs = run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test}))
        self.assertEqual(evidence(fs), ["tests/test_outputs.py:16"])
        self.assertIn("namespace", fs[0].message)

    def test_pos_star_import_of_submission(self):
        test = "import sys\nsys.path.insert(0, '/app')\nfrom solution import *\n\n\ndef test_it():\n    assert solve(2) == 4\n"
        self.assertEqual(evidence(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test}))),
                         ["tests/test_outputs.py:3"])

    def test_neg_star_import_of_stdlib(self):
        test = "from math import *\n\n\ndef test_it():\n    assert sqrt(4) == 2\n"
        self.assertEqual(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test})), [])


# ---------------------------------------------------------------------------
# verdict-parse-semantics
# ---------------------------------------------------------------------------

class VerdictParseSemantics(unittest.TestCase):
    RID = "verdict-parse-semantics"

    def kinds(self, fs):
        return sorted(f.message.split(",")[0] for f in fs)

    def test_pos_skipped_counts_as_pass(self):
        sh = """
        #!/bin/bash
        pytest /tests/test_outputs.py -rA > /tmp/log.txt
        python3 - <<'PY'
        import re
        passed = 0
        for line in open("/tmp/log.txt"):
            status = line.split(" ")[0]
            if status in ("PASSED", "SKIPPED"):
                passed += 1
        open("/logs/verifier/reward.txt", "w").write("1" if passed else "0")
        PY
        """
        fs = run(self.RID, harbor(sh))
        self.assertIn("tests/test.sh:8", evidence(fs))
        self.assertTrue(any("SKIPPED" in f.message for f in fs))

    def test_pos_xfail_in_pass_tuple(self):
        parser = """
        from swebench.harness.constants import TestStatus

        def resolve(status_map, tests):
            passing = (TestStatus.PASSED.value, TestStatus.XFAIL.value)
            return all(status_map.get(t) in passing for t in tests)

        print(resolve({}, []))
        """
        fs = run(self.RID, harbor("#!/bin/bash\npython3 /tests/parse.py\n", tests={"parse.py": parser}))
        self.assertIn("tests/parse.py:4", evidence(fs))

    def test_pos_empty_run_is_full_credit(self):
        env = """
        def score_tests(results):
            passed = sum(1 for r in results if r["ok"])
            return passed / len(results) if len(results) else 1.0

        def check_required(required, passed_tests):
            return required <= passed_tests

        def reward(completion, **kw):
            return score_tests(kw["results"]) * check_required(set(), set())
        """
        fs = run(self.RID, {"env.py": env})
        self.assertEqual(evidence(fs), ["env.py:3", "env.py:6"])

    def test_pos_all_over_parsed_results(self):
        env = """
        class Env:
            def _run_tests_and_score(self, session, state):
                out = session.exec_run(["bash", "/tests/run-tests.sh"]).output.decode()
                results = self.parser.parse(out)
                ok = results is not None and all(v == "PASSED" for v in results.values())
                state["resolved"] = ok
        """
        self.assertEqual(evidence(run(self.RID, {"env.py": env})), ["env.py:5"])

    def test_neg_all_over_fixed_task_list(self):
        env = """
        WORKLOADS = ["a", "b"]

        def score_run(run_case):
            results = [run_case(w) for w in WORKLOADS]
            return all(r["passed"] for r in results)

        def guarded(parser, out):
            results = parser.parse(out)
            return bool(results) and all(r == "PASSED" for r in results.values())
        """
        self.assertEqual(run(self.RID, {"env.py": env}), [])

    def test_pos_printed_token_on_reward_path(self):
        env = """
        import verifiers as vf

        async def correctness_reward(completion, state, sandbox, **kw):
            res = await sandbox.execute_command(state["sid"], "cd /app && python run_tests.py")
            return 1.0 if "All tests passed" in res.stdout else 0.0

        def load_environment(**kw):
            return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[correctness_reward]))
        """
        fs = run(self.RID, {"env.py": env})
        self.assertEqual(evidence(fs), ["env.py:5"])
        self.assertIn("printed", fs[0].message)

    def test_pos_shell_grep_of_test_output(self):
        sh = """
        #!/bin/bash
        cd /app
        pytest /tests/test_outputs.py -rA | tee /tmp/out.log
        if grep -q "passed" /tmp/out.log; then
            echo 1 > /logs/verifier/reward.txt
        else
            echo 0 > /logs/verifier/reward.txt
        fi
        """
        fs = run(self.RID, harbor(sh))
        self.assertEqual(evidence(fs), ["tests/test.sh:4"])

    def test_pos_exit_status_of_process_importing_submission(self):
        fs = run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": IMPORTING_TEST}))
        self.assertEqual(evidence(fs), ["tests/test.sh:3", "tests/test_outputs.py:8"])
        self.assertIn("os._exit(0)", fs[0].message)

    def test_pos_agent_controlled_skip(self):
        test = """
        import os
        import pytest

        def test_only_one_line_changed():
            if not os.path.exists("/app/original.py"):
                pytest.skip("original not present")
            assert open("/app/original.py").read() != open("/app/fixed.py").read()
        """
        fs = run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test}))
        self.assertEqual(evidence(fs), ["tests/test_outputs.py:6"])

    def test_pos_judge_correct_substring(self):
        env = """
        async def judge_reward(judge, prompt, completion, answer, state):
            verdict = await judge(prompt, completion, answer, state)
            return 1.0 if "correct" in verdict.lower() else 0.0
        """
        fs = run(self.RID, {"env.py": env})
        self.assertEqual(evidence(fs), ["env.py:3"])

    def test_neg_strict_parser(self):
        parser = """
        import json, sys

        def main():
            report = json.load(open(sys.argv[1]))
            tests = report["tests"]
            if not tests:
                return 0.0
            ok = sum(1 for t in tests if t["outcome"] == "passed")
            return 1.0 if ok == len(tests) else 0.0

        print(main())
        """
        self.assertEqual(run(self.RID, harbor("#!/bin/bash\npython3 /tests/parse.py r.json\n",
                                              tests={"parse.py": parser})), [])

    def test_neg_judge_checks_incorrect_first(self):
        env = """
        async def judge_reward(judge, prompt, completion, answer, state):
            verdict = (await judge(prompt, completion, answer, state)).lower()
            if "incorrect" in verdict:
                return 0.0
            return 1.0 if "correct" in verdict else 0.0
        """
        self.assertEqual(run(self.RID, {"env.py": env}), [])

    def test_neg_submission_runs_in_subprocess(self):
        self.assertEqual(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": SUBPROCESS_TEST})), [])

    def test_neg_nonce_delimited_output(self):
        env = """
        import uuid

        async def correctness_reward(completion, state, sandbox, **kw):
            tag = uuid.uuid4().hex
            res = await sandbox.execute_command(state["sid"], f"python grade.py --tag {tag}")
            return 1.0 if f"{tag}:PASSED" in res.stdout and "PASSED" in res.stdout else 0.0
        """
        self.assertEqual(run(self.RID, {"env.py": env}), [])

    def test_neg_dead_helper(self):
        parser = """
        def compute_fail_to_pass(report):
            total = len(report["f2p"])
            if total == 0:
                return 1
            return len(report["ok"]) / total

        def resolved(report):
            return len(report["ok"]) > 0

        print(resolved({"ok": []}))
        """
        self.assertEqual(run(self.RID, harbor("#!/bin/bash\npython3 /tests/parse.py\n",
                                              tests={"parse.py": parser})), [])

    def test_neg_selected_test_does_not_import(self):
        test = IMPORTING_TEST + "\n\ndef test_cli():\n    import subprocess\n    assert subprocess.run(['true']).returncode == 0\n"
        sh = EXIT_VERDICT_SH.replace("/tests/test_outputs.py", "/tests/test_outputs.py::test_cli")
        self.assertEqual(run(self.RID, harbor(sh, tests={"test_outputs.py": test})), [])


# ---------------------------------------------------------------------------
# existence-or-constant-check
# ---------------------------------------------------------------------------

class ExistenceOrConstantCheck(unittest.TestCase):
    RID = "existence-or-constant-check"

    def test_pos_tests_only_check_existence(self):
        test = """
        from pathlib import Path
        import os

        def test_report_exists():
            assert Path("/app/report.csv").exists()

        def test_model_saved():
            assert os.path.isfile("/app/model.pt"), "model missing"
        """
        fs = run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test}))
        self.assertEqual(evidence(fs), ["tests/test_outputs.py:4"])

    def test_pos_shell_pays_on_existence(self):
        sh = """
        #!/bin/bash
        if [ -f /app/output.json ]; then
            echo 1 > /logs/verifier/reward.txt
        else
            echo 0 > /logs/verifier/reward.txt
        fi
        """
        self.assertEqual(evidence(run(self.RID, harbor(sh))), ["tests/test.sh:2"])

    def test_pos_reward_for_any_nonempty_answer(self):
        env = """
        import verifiers as vf

        parser = vf.XMLParser(["answer"])

        def correctness_reward(completion, **kw):
            parsed = parser.parse_answer(completion)
            return 1.0 if parsed else 0.0

        def load_environment(**kw):
            return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[correctness_reward], weights=[1.0]))
        """
        self.assertEqual(evidence(run(self.RID, {"env.py": env})), ["env.py:5"])

    def test_neg_content_is_checked(self):
        test = """
        from pathlib import Path

        def test_exists():
            assert Path("/app/hello.txt").exists()

        def test_contents():
            assert Path("/app/hello.txt").read_text().strip() == "Hello, world!"
        """
        self.assertEqual(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test})), [])

    def test_neg_shell_checks_contents(self):
        sh = """
        #!/bin/bash
        if [ -f /app/result.txt ] && grep -q "ok" /app/result.txt; then
            echo 1 > /logs/verifier/reward.txt
        fi
        if [ -s /logs/agent/shot.png ]; then
            magic=$(head -c 8 /logs/agent/shot.png | od -An -tx1)
            if [ "$magic" = "x" ]; then
                reward=1
            fi
        fi
        """
        self.assertEqual(run(self.RID, harbor(sh)), [])

    def test_neg_reward_compares_answer_and_format_reward(self):
        env = """
        import verifiers as vf

        parser = vf.XMLParser(["answer"])

        def correctness_reward(completion, answer, **kw):
            return 1.0 if parser.parse_answer(completion) == answer else 0.0

        def format_reward(completion, **kw):
            return 1.0 if parser.parse_answer(completion) else 0.0

        def load_environment(**kw):
            return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[correctness_reward, format_reward]))
        """
        self.assertEqual(run(self.RID, {"env.py": env}), [])

    def test_neg_env_unit_tests_are_not_graded_tests(self):
        test = "import os\n\ndef test_pkg_has_readme():\n    assert os.path.exists('README.md')\n"
        self.assertEqual(run(self.RID, {"tests/test_env.py": test, "env.py": "x = 1\n"}), [])


# ---------------------------------------------------------------------------
# equality-on-model-objects
# ---------------------------------------------------------------------------

class EqualityOnModelObjects(unittest.TestCase):
    RID = "equality-on-model-objects"

    def test_pos_spec_from_file_location(self):
        fs = run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": IMPORTING_TEST}))
        self.assertEqual(evidence(fs), ["tests/test_outputs.py:17"])

    def test_pos_sys_path_import(self):
        test = """
        import sys
        sys.path.insert(0, "/app")
        from solution import solve


        def test_solve():
            result = solve([3, 1, 2])
            expected = [1, 2, 3]
            assert result == expected
        """
        self.assertEqual(evidence(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test}))),
                         ["tests/test_outputs.py:9"])

    def test_pos_unittest_assert_equal(self):
        test = """
        import importlib
        import sys
        import unittest

        sys.path.insert(0, "/workspace")
        mod = importlib.import_module("solution")


        class TestSolution(unittest.TestCase):
            def test_value(self):
                self.assertEqual(mod.compute(4), 16)
        """
        self.assertEqual(evidence(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test}))),
                         ["tests/test_outputs.py:11"])

    def test_neg_serialized_before_compare(self):
        test = IMPORTING_TEST.replace("assert mod.add(2, 3) == 5",
                                      "out = json.loads(json.dumps(mod.add(2, 3)))\n    assert out == 5")
        test = "import json\n" + test
        self.assertEqual(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test})), [])

    def test_neg_type_checked(self):
        test = IMPORTING_TEST.replace("assert mod.add(2, 3) == 5",
                                      "r = mod.add(2, 3)\n    assert isinstance(r, int)\n    assert r == 5")
        self.assertEqual(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test})), [])

    def test_neg_subprocess_output(self):
        self.assertEqual(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": SUBPROCESS_TEST})), [])

    def test_neg_str_wrapped(self):
        test = IMPORTING_TEST.replace("assert mod.add(2, 3) == 5", "assert str(mod.add(2, 3)) == '5'")
        self.assertEqual(run(self.RID, harbor(EXIT_VERDICT_SH, tests={"test_outputs.py": test})), [])


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

class OddInput(unittest.TestCase):
    def test_no_crash_on_odd_files(self):
        files = {
            "task.toml": "this is [ not toml",
            "tests/test.sh": "#!/bin/bash\ncd \"$(dirname $0)\nif [[ $x == *'PASS ]]; then echo 1 > reward.txt\n"
                             "cat <<EOF\nunterminated heredoc\n",
            "tests/test_x.py": "def test_(:\n    pass\n",
            "env.py": "def f(:\n",
            "weird.sh": "\\\n\\\n",
        }
        pkg = make_pkg(files)
        for rid in ("grader-import-surface", "verdict-parse-semantics", "existence-or-constant-check",
                    "equality-on-model-objects"):
            R.RULES[rid]["fn"](pkg)  # must not raise

    def test_rules_registered_with_metadata(self):
        for rid in ("grader-import-surface", "verdict-parse-semantics", "existence-or-constant-check",
                    "equality-on-model-objects"):
            meta = R.RULES[rid]
            self.assertIn(meta["kind"], ("hack", "broken"))
            self.assertNotIn(";", meta["description"])
            self.assertNotIn("—", meta["description"])


# ---------------------------------------------------------------------------
# v0.5 widenings
# ---------------------------------------------------------------------------

def hits(rid, files, sub):
    """Evidence of the findings of one rule whose message contains `sub`."""
    return [e for f in run(rid, files) if sub in f.message for e in f.evidence]


VERIFIERS_REG = """
def load_environment(**kw):
    return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[{name}], weights=[1.0]))
"""

PYTEST_SH = "#!/bin/bash\ncd /tests\npytest /tests/test_outputs.py -rA\nif [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n"


class HarnessInModelNamespace(unittest.TestCase):
    """Item 1, widens grader-import-surface."""
    K = "test harness function"
    E = "evaluator or reference module"

    def env(self, body):
        return {"env.py": "import subprocess\nimport verifiers as vf\n\n" + textwrap.dedent(body)}

    def test_pos_run_tests_call_appended_after_code(self):
        self.assertTrue(hits("grader-import-surface", self.env("""
        def correctness_reward(completion, **kw):
            code = parse_code(completion)
            program = code + "\\n\\nrun_tests()\\n"
            r = subprocess.run(["python", "-c", program], capture_output=True)
            return 1.0 if r.returncode == 0 else 0.0
        """), self.K))

    def test_pos_fstring_with_verify_call(self):
        self.assertTrue(hits("grader-import-surface", self.env("""
        async def reward(completion, sandbox, **kw):
            solution = extract(completion)
            res = await sandbox.run(f"{solution}\\nverify_solution()\\n")
            return float(res.exit_code == 0)
        """), self.K))

    def test_pos_harness_execd_before_code_in_same_namespace(self):
        self.assertTrue(hits("grader-import-surface", self.env("""
        def score(completion, info, **kw):
            ns = {}
            exec(info["test_harness"], ns)
            exec(code_from(completion), ns)
            return float(ns["run_tests"]())
        """), self.K))

    def test_neg_humaneval_tests_follow_code(self):
        self.assertFalse(hits("grader-import-surface", self.env("""
        def correctness_reward(completion, info, **kw):
            code = parse_code(completion)
            program = f"{code}\\n{info['test']}\\ncheck({info['entry_point']})\\n"
            r = subprocess.run(["python", "-c", program], capture_output=True)
            return 1.0 if r.returncode == 0 else 0.0
        """), self.K))

    def test_neg_harness_defined_after_code(self):
        self.assertFalse(hits("grader-import-surface", self.env("""
        def correctness_reward(completion, **kw):
            code = parse_code(completion)
            program = code + "\\n\\ndef run_tests():\\n    assert add(1, 2) == 3\\n\\nrun_tests()\\n"
            r = subprocess.run(["python", "-c", program], capture_output=True)
            return 1.0 if r.returncode == 0 else 0.0
        """), self.K))

    def test_neg_appended_asserts(self):
        self.assertFalse(hits("grader-import-surface", self.env("""
        def correctness_reward(completion, **kw):
            code = parse_code(completion)
            program = code + "\\nassert add(1, 2) == 3\\n"
            r = subprocess.run(["python", "-c", program], capture_output=True)
            return 1.0 if r.returncode == 0 else 0.0
        """), self.K))

    EVAL_TEST = ("import sys\nsys.path.insert(0, '/app')\nfrom evaluate import compute_score\n\n\n"
                 "def test_score():\n    assert compute_score('/app/preds.csv') > 0.9\n")

    def test_pos_test_imports_shipped_evaluator(self):
        files = harbor(PYTEST_SH, tests={"test_outputs.py": self.EVAL_TEST},
                       extra={"environment/evaluate.py": "def compute_score(p):\n    return 0.5\n",
                              "instruction.md": "Train a model and write predictions to /app/preds.csv.\n"})
        self.assertTrue(hits("grader-import-surface", files, self.E))

    def test_pos_spec_loader_on_shipped_metric(self):
        test = ("import importlib.util\n\nspec = importlib.util.spec_from_file_location('m', '/app/metric.py')\n"
                "m = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(m)\n\n\n"
                "def test_metric():\n    assert m.f1('/app/out.json') >= 0.8\n")
        files = harbor(PYTEST_SH, tests={"test_outputs.py": test},
                       extra={"environment/metric.py": "def f1(p):\n    return 0.0\n"})
        self.assertTrue(hits("grader-import-surface", files, self.E))

    def test_pos_reference_module_in_workdir(self):
        test = ("import sys\nsys.path.append('/app')\nimport reference\nimport numpy as np\n\n\n"
                "def test_close():\n    assert np.allclose(np.load('/app/out.npy'), reference.expected())\n")
        files = harbor(PYTEST_SH, tests={"test_outputs.py": test},
                       extra={"environment/reference.py": "def expected():\n    return [1]\n"})
        self.assertTrue(hits("grader-import-surface", files, self.E))

    def test_neg_module_is_the_submission(self):
        files = harbor(PYTEST_SH, tests={"test_outputs.py": self.EVAL_TEST},
                       extra={"environment/evaluate.py": "def compute_score(p):\n    return 0.5\n",
                              "instruction.md": "Implement compute_score in /app/evaluate.py.\n"})
        self.assertFalse(hits("grader-import-surface", files, self.E))

    def test_neg_grader_rewrites_module_first(self):
        test = self.EVAL_TEST.replace("from evaluate", "open('/app/evaluate.py', 'w').write(SRC)\nfrom evaluate")
        files = harbor(PYTEST_SH, tests={"test_outputs.py": test},
                       extra={"environment/evaluate.py": "def compute_score(p):\n    return 0.5\n"})
        self.assertFalse(hits("grader-import-surface", files, self.E))

    def test_neg_module_not_shipped_by_task(self):
        files = harbor(PYTEST_SH, tests={"test_outputs.py": self.EVAL_TEST})
        self.assertFalse(hits("grader-import-surface", files, self.E))


class StartupHooks(unittest.TestCase):
    """Item 6, widens grader-import-surface."""
    V = "virtual environment inside"
    W = "non-root agent write the Python install"
    P = "patch filter strips conftest.py"

    def test_pos_venv_python_in_workdir(self):
        sh = "#!/bin/bash\ncd /tests\n/app/.venv/bin/python -m pytest /tests/test_outputs.py\n"
        self.assertTrue(hits("grader-import-surface", harbor(sh), self.V))

    def test_pos_activate_agent_venv(self):
        sh = "#!/bin/bash\nsource /app/venv/bin/activate\ncd /tests\npytest test_outputs.py\n"
        self.assertTrue(hits("grader-import-surface", harbor(sh), self.V))

    def test_pos_uv_run_in_agent_project(self):
        sh = "#!/bin/bash\ncd /app\nuv run pytest /tests/test_outputs.py\n"
        files = harbor(sh, extra={"environment/pyproject.toml": "[project]\nname='x'\n"})
        self.assertTrue(hits("grader-import-surface", files, self.V))

    def test_pos_nonroot_agent_writable_site_packages(self):
        df = "FROM python:3.12-slim\nRUN useradd agent && chmod -R 777 /usr/local/lib/python3.12/site-packages\nUSER agent\nWORKDIR /home/agent\n"
        sh = "#!/bin/bash\ncd /tests\npython3 /tests/check.py\n"
        self.assertTrue(hits("grader-import-surface", harbor(sh, dockerfile=df), self.W))

    def test_pos_patch_filter_without_startup_hooks(self):
        env = ("BLOCKED = ['conftest.py', 'pytest.ini', 'tox.ini']\n\n\n"
               "def filter_patch(diff):\n    \"\"\"Exclude files from the model patch.\"\"\"\n"
               "    return [f for f in parse_diff(diff) if f.path not in BLOCKED]\n")
        self.assertTrue(hits("grader-import-surface", {"env.py": env}, self.P))

    def test_neg_fresh_venv_made_by_script(self):
        sh = "#!/bin/bash\ncd /app\nuv venv .tb\nsource .tb/bin/activate\nuv run pytest /tests/test_outputs.py\n"
        self.assertFalse(hits("grader-import-surface", harbor(sh), self.V))

    def test_neg_venv_outside_agent_tree(self):
        sh = "#!/bin/bash\ncd /tests\n/opt/grader/.venv/bin/python -m pytest /tests/test_outputs.py\n"
        self.assertFalse(hits("grader-import-surface", harbor(sh), self.V))

    def test_neg_root_agent_or_fresh_venv(self):
        df = "FROM python:3.12-slim\nRUN chmod -R 777 /usr/local/lib/python3.12/site-packages\nWORKDIR /app\n"
        sh = "#!/bin/bash\ncd /tests\npython3 /tests/check.py\n"
        self.assertFalse(hits("grader-import-surface", harbor(sh, dockerfile=df), self.W))
        df2 = df.replace("WORKDIR", "USER agent\nWORKDIR")
        sh2 = "#!/bin/bash\nuv venv /tmp/v\n/tmp/v/bin/python /tests/check.py\n"
        self.assertFalse(hits("grader-import-surface", harbor(sh2, dockerfile=df2), self.W))

    def test_neg_patch_filter_covers_startup_hooks(self):
        env = ("BLOCKED = ['conftest.py', 'sitecustomize.py', 'usercustomize.py']\n\n\n"
               "def filter_patch(diff):\n    return [f for f in parse_diff(diff) if f.path not in BLOCKED]\n")
        self.assertFalse(hits("grader-import-surface", {"env.py": env}, self.P))


class ScoreParsedFromStdout(unittest.TestCase):
    """Item 3, widens verdict-parse-semantics."""
    K = "number parsed from the output"

    def env(self, body, name):
        return {"env.py": "import re\nimport json\nimport subprocess\nimport verifiers as vf\n\n" + textwrap.dedent(body)
                + VERIFIERS_REG.format(name=name)}

    def test_pos_last_line_float(self):
        self.assertTrue(hits("verdict-parse-semantics", self.env("""
        async def perf_reward(completion, sandbox, **kw):
            code = parse_code(completion)
            res = await sandbox.run(f"python -c '{code}'")
            lines = res.stdout.strip().splitlines()
            return float(lines[-1])
        """, "perf_reward"), self.K))

    def test_pos_regex_group_float(self):
        self.assertTrue(hits("verdict-parse-semantics", self.env("""
        def score_reward(completion, **kw):
            program = parse_code(completion)
            out = subprocess.run(["python", "-c", program], capture_output=True, text=True).stdout
            m = re.search(r"SCORE: ([0-9.]+)", out)
            score = float(m.group(1)) if m else 0.0
            return score
        """, "score_reward"), self.K))

    def test_pos_shell_tail_to_reward(self):
        sh = "#!/bin/bash\ncd /app\nscore=$(python /app/run_eval.py | tail -n 1)\necho \"$score\" > /logs/verifier/reward.txt\n"
        self.assertTrue(hits("verdict-parse-semantics", harbor(sh), self.K))

    def test_neg_fixed_verifier_script_with_text_argument(self):
        self.assertFalse(hits("verdict-parse-semantics", self.env("""
        async def correct(completion, answer, runtime, **kw):
            result = await runtime.run_uv_script(VERIFY, args=[answer, completion[-1]["content"]])
            lines = result.stdout.strip().splitlines()
            return float(lines[-1]) if lines else 0.0
        """, "correct"), self.K))

    def test_neg_value_only_logged(self):
        self.assertFalse(hits("verdict-parse-semantics", self.env("""
        def speed_reward(completion, **kw):
            code = parse_code(completion)
            r = subprocess.run(["python", "-c", code], capture_output=True, text=True)
            ratio = float(r.stdout.split()[-1])
            print(ratio)
            return 1.0 if r.returncode == 0 else 0.0
        """, "speed_reward"), self.K))

    def test_neg_shell_reward_from_exit_code(self):
        sh = "#!/bin/bash\ncd /app\nout=$(python /app/run_eval.py | tail -n 1)\necho $out\nif [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; fi\n"
        self.assertFalse(hits("verdict-parse-semantics", harbor(sh), self.K))


class StrictVerdictParse(unittest.TestCase):
    """Item 4, widens verdict-parse-semantics."""
    N = "no FAILED line appears"
    D = "divides by the passed and failed counts"

    def env(self, body, name):
        return {"env.py": "import re\nimport subprocess\nimport verifiers as vf\n\n" + textwrap.dedent(body)
                + VERIFIERS_REG.format(name=name)}

    def test_pos_failed_not_in_output(self):
        self.assertTrue(hits("verdict-parse-semantics", self.env("""
        def tests_reward(completion, **kw):
            code = parse_code(completion)
            r = subprocess.run(["pytest", "-q", "tests/"], capture_output=True, text=True, cwd=write_code(code))
            return 1.0 if "FAILED" not in r.stdout else 0.0
        """, "tests_reward"), self.N))

    def test_pos_failed_count_zero(self):
        self.assertTrue(hits("verdict-parse-semantics", self.env("""
        async def correctness_reward(completion, sandbox, **kw):
            code = parse_code(completion)
            res = await sandbox.run(f"python -m pytest {code}")
            ok = res.stdout.count("FAILED") == 0
            return float(ok)
        """, "correctness_reward"), self.N))

    def test_pos_denominator_from_run(self):
        self.assertTrue(hits("verdict-parse-semantics", self.env("""
        def pass_rate_reward(completion, **kw):
            code = parse_code(completion)
            out = subprocess.run(["pytest", "-rA"], capture_output=True, text=True, cwd=write_code(code)).stdout
            passed = len(re.findall(r"^PASSED", out, re.M))
            failed = len(re.findall(r"^FAILED", out, re.M))
            return passed / (passed + failed)
        """, "pass_rate_reward"), self.D))

    def test_neg_positive_check_present(self):
        self.assertFalse(hits("verdict-parse-semantics", self.env("""
        def tests_reward(completion, **kw):
            code = parse_code(completion)
            r = subprocess.run(["pytest", "-q", "tests/"], capture_output=True, text=True, cwd=write_code(code))
            return 1.0 if "FAILED" not in r.stdout and r.returncode == 0 else 0.0
        """, "tests_reward"), self.N))

    def test_neg_denominator_is_expected_count(self):
        self.assertFalse(hits("verdict-parse-semantics", self.env("""
        def pass_rate_reward(completion, info, **kw):
            code = parse_code(completion)
            out = subprocess.run(["pytest", "-rA"], capture_output=True, text=True, cwd=write_code(code)).stdout
            passed = len(re.findall(r"^PASSED", out, re.M))
            return passed / len(info["tests"])
        """, "pass_rate_reward"), self.D))

    def test_neg_denominator_counts_errors(self):
        self.assertFalse(hits("verdict-parse-semantics", self.env("""
        def pass_rate_reward(completion, **kw):
            code = parse_code(completion)
            out = subprocess.run(["pytest", "-rA"], capture_output=True, text=True, cwd=write_code(code)).stdout
            passed = out.count("PASSED")
            failed = out.count("FAILED")
            errors = out.count("ERROR")
            return passed / (passed + failed + errors)
        """, "pass_rate_reward"), self.D))


class ConftestChildVerdict(unittest.TestCase):
    """Item 8, widens verdict-parse-semantics."""
    K = "forked child"
    CONF = """
import os
import pickle
from _pytest.runner import runtestprotocol


def pytest_runtest_protocol(item, nextitem):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        reports = runtestprotocol(item, log=False, nextitem=nextitem)
        os.write(w, b"1" if all(x.passed or x.skipped for x in reports) else b"0")
        os._exit(0)
    ok = os.read(r, 1) == b"1"
    os.waitpid(pid, 0)
    return ok
"""

    def test_pos_fork_pipe_conftest(self):
        self.assertTrue(hits("verdict-parse-semantics", harbor(PYTEST_SH, tests={"conftest.py": self.CONF}), self.K))

    def test_pos_multiprocessing_queue(self):
        conf = self.CONF.replace("r, w = os.pipe()", "q = multiprocessing.Queue()").replace("os.write(w,", "q.put(")
        self.assertTrue(hits("verdict-parse-semantics", harbor(PYTEST_SH, tests={"conftest.py": "import multiprocessing\n" + conf}), self.K))

    def test_pos_conftest_in_tests_folder_of_package(self):
        self.assertTrue(hits("verdict-parse-semantics", {"tests/conftest.py": self.CONF, "tests/test.sh": PYTEST_SH}, self.K))

    def test_neg_no_fork(self):
        conf = self.CONF.replace("pid = os.fork()", "pid = 1")
        self.assertFalse(hits("verdict-parse-semantics", harbor(PYTEST_SH, tests={"conftest.py": conf}), self.K))

    def test_neg_no_runtestprotocol(self):
        conf = self.CONF.replace("runtestprotocol", "run_one")
        self.assertFalse(hits("verdict-parse-semantics", harbor(PYTEST_SH, tests={"conftest.py": conf}), self.K))

    def test_neg_ordinary_conftest(self):
        conf = "import pytest\n\n\n@pytest.fixture\ndef data():\n    return [1, 2]\n"
        self.assertFalse(hits("verdict-parse-semantics", harbor(PYTEST_SH, tests={"conftest.py": conf}), self.K))


class VacuousAssertTests(unittest.TestCase):
    """Item 2, widens existence-or-constant-check."""
    W = "No test in the graded test file can fail"
    S = "A graded test can never fail"

    def test_pos_all_tests_swallowed_or_true(self):
        test = ("import json\n\n\ndef test_a():\n    try:\n        assert json.load(open('/app/o.json'))['x'] == 3\n"
                "    except Exception:\n        pass\n\n\ndef test_b():\n    assert True\n")
        self.assertTrue(hits("existence-or-constant-check", harbor(PYTEST_SH, tests={"test_outputs.py": test}), self.W))

    def test_pos_print_only_and_skipped(self):
        test = ("import pytest\n\n\n@pytest.mark.skip(reason='flaky')\ndef test_a():\n    assert run() == 1\n\n\n"
                "def test_b():\n    print('checked')\n")
        self.assertTrue(hits("existence-or-constant-check", harbor(PYTEST_SH, tests={"test_outputs.py": test}), self.W))

    def test_pos_single_non_raising_helper(self):
        test = ("def check(a, b):\n    if a != b:\n        print('mismatch')\n\n\n"
                "def test_real():\n    assert open('/app/o.txt').read() == 'x'\n\n\n"
                "def test_value():\n    check(open('/app/v.txt').read(), '42')\n")
        self.assertTrue(hits("existence-or-constant-check", harbor(PYTEST_SH, tests={"test_outputs.py": test}), self.S))

    def test_neg_real_asserts(self):
        test = "def test_a():\n    assert open('/app/o.txt').read().strip() == '42'\n"
        files = harbor(PYTEST_SH, tests={"test_outputs.py": test})
        self.assertFalse(hits("existence-or-constant-check", files, self.W))
        self.assertFalse(hits("existence-or-constant-check", files, self.S))

    def test_neg_except_reraises_or_fails(self):
        test = ("import pytest\n\n\ndef test_a():\n    try:\n        assert run() == 1\n    except Exception as e:\n"
                "        pytest.fail(str(e))\n")
        files = harbor(PYTEST_SH, tests={"test_outputs.py": test})
        self.assertFalse(hits("existence-or-constant-check", files, self.W))
        self.assertFalse(hits("existence-or-constant-check", files, self.S))

    def test_neg_script_style_return_verdict_and_import_check(self):
        test = ("import sys\n\n\ndef test_answer():\n    try:\n        return open('/app/a.txt').read() == '7'\n"
                "    except Exception:\n        return False\n\n\nif __name__ == '__main__':\n    sys.exit(0 if test_answer() else 1)\n")
        self.assertFalse(hits("existence-or-constant-check", harbor("#!/bin/bash\npython3 /tests/test_outputs.py\n",
                                                                     tests={"test_outputs.py": test}), self.W))
        test2 = "import sys\nsys.path.append('/app')\n\n\ndef test_import():\n    from mod import Thing  # noqa\n\n\ndef test_b():\n    assert 1 + 1 == 2 or run()\n"
        self.assertFalse(hits("existence-or-constant-check", harbor(PYTEST_SH, tests={"test_outputs.py": test2}), self.S))


class GrepSourceTests(unittest.TestCase):
    """Item 5, widens existence-or-constant-check."""
    K = "search the agent's source text"

    def test_pos_assert_keyword_in_source(self):
        test = ("from pathlib import Path\n\n\ndef test_uses_param_query():\n    src = Path('/app/server.py').read_text()\n"
                "    assert 'cursor.execute(query, params)' in src\n\n\ndef test_no_fstring_sql():\n"
                "    src = Path('/app/server.py').read_text()\n    assert 'f\"SELECT' not in src\n")
        self.assertTrue(hits("existence-or-constant-check", harbor(PYTEST_SH, tests={"test_outputs.py": test}), self.K))

    def test_pos_regex_on_source(self):
        test = ("import re\n\nSRC = '/app/main.c'\n\n\ndef test_bounds_check():\n    code = open(SRC).read()\n"
                "    assert re.search(r'if \\(len > MAX\\)', code)\n")
        self.assertTrue(hits("existence-or-constant-check", harbor(PYTEST_SH, tests={"test_outputs.py": test}), self.K))

    def test_pos_shell_grep_on_source(self):
        sh = ("#!/bin/bash\nif grep -q 'escape_html(' /app/app.py; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n"
              "  echo 0 > /logs/verifier/reward.txt\nfi\n")
        self.assertTrue(hits("existence-or-constant-check", harbor(sh), self.K))

    def test_neg_tests_also_run_code(self):
        test = ("import subprocess\nfrom pathlib import Path\n\n\ndef test_source():\n"
                "    assert 'params' in Path('/app/server.py').read_text()\n\n\ndef test_runs():\n"
                "    assert subprocess.run(['python', '/app/server.py', '--check']).returncode == 0\n")
        self.assertFalse(hits("existence-or-constant-check", harbor(PYTEST_SH, tests={"test_outputs.py": test}), self.K))

    def test_neg_reads_output_file_not_source(self):
        test = "from pathlib import Path\n\n\ndef test_out():\n    assert 'total: 42' in Path('/app/report.txt').read_text()\n"
        self.assertFalse(hits("existence-or-constant-check", harbor(PYTEST_SH, tests={"test_outputs.py": test}), self.K))

    def test_neg_shell_runs_tests_after_grep(self):
        sh = ("#!/bin/bash\ngrep -q 'escape_html(' /app/app.py || echo missing\ncd /tests\npytest test_outputs.py\n"
              "if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; fi\n")
        self.assertFalse(hits("existence-or-constant-check", harbor(sh), self.K))


class UnparsableRewardFile(unittest.TestCase):
    def test_no_crash_on_syntax_error_with_reward_hint(self):
        files = {"env.py": "def f(:\n    return float(bool(x))\nreturn 1.0 if x else 0\n"}
        for rid in ("grader-import-surface", "verdict-parse-semantics", "existence-or-constant-check"):
            R.RULES[rid]["fn"](make_pkg(files))


class TrivialPropertyValidator(unittest.TestCase):
    """Item 7, widens existence-or-constant-check."""
    R = "last message came from the assistant"
    M = "equals a number the instruction shows"

    def env(self, body, name):
        return {"env.py": "import verifiers as vf\n\n" + textwrap.dedent(body) + VERIFIERS_REG.format(name=name)}

    def test_pos_role_only_ifexp(self):
        self.assertTrue(hits("existence-or-constant-check", self.env("""
        def task_reward(completion, **kw):
            return 1.0 if completion[-1]["role"] == "assistant" else 0.0
        """, "task_reward"), self.R))

    def test_pos_role_only_if_return(self):
        self.assertTrue(hits("existence-or-constant-check", self.env("""
        def done_reward(completion, **kw):
            last = completion[-1]
            if last.get("role") == "assistant":
                return 1.0
            return 0.0
        """, "done_reward"), self.R))

    def test_pos_metric_literal_from_instruction(self):
        test = ("import json\n\n\ndef test_accuracy():\n    data = json.load(open('/app/metrics.json'))\n"
                "    assert data['eval_accuracy'] == 0.5271\n")
        files = harbor(PYTEST_SH, tests={"test_outputs.py": test},
                       extra={"instruction.md": "Fine tune the model. The expected eval_accuracy is 0.5271. Write /app/metrics.json.\n"})
        self.assertTrue(hits("existence-or-constant-check", files, self.M))

    def test_neg_role_and_content_checked(self):
        self.assertFalse(hits("existence-or-constant-check", self.env("""
        def task_reward(completion, answer, **kw):
            last = completion[-1]
            return 1.0 if last["role"] == "assistant" and answer in last["content"] else 0.0
        """, "task_reward"), self.R))

    def test_neg_role_check_not_registered(self):
        self.assertFalse(hits("existence-or-constant-check", {"env.py": textwrap.dedent("""
        def is_reply(completion):
            return completion[-1]["role"] == "assistant"
        """)}, self.R))

    def test_neg_metric_literal_hidden(self):
        test = ("import json\n\n\ndef test_accuracy():\n    data = json.load(open('/app/metrics.json'))\n"
                "    assert data['eval_accuracy'] == 0.5271\n")
        files = harbor(PYTEST_SH, tests={"test_outputs.py": test},
                       extra={"instruction.md": "Fine tune the model and write /app/metrics.json.\n"})
        self.assertFalse(hits("existence-or-constant-check", files, self.M))


if __name__ == "__main__":
    unittest.main()

"""Tests for the Harbor task rules (envscan/rules_harbor.py).

Each case writes a tiny Harbor task into a temp folder and loads it with Package.from_dir.
Run from the scanner folder:
    python3 -m unittest tests.test_rules_harbor
"""
import os
import tempfile
import textwrap
import unittest

from envscan.package import Package
from envscan.rules import RULES
from envscan import rules_harbor  # noqa: F401  (registers the rules)

TOML = """\
schema_version = "1.4"
[task]
name = "demo/task"
[verifier]
timeout_sec = 120.0
[agent]
timeout_sec = 120.0
[environment]
cpus = 1
memory_mb = 2048
"""
SEP_TOML = TOML.replace("[verifier]\n", "[verifier]\nenvironment_mode = \"separate\"\n")
USER_TOML = TOML.replace("[agent]\n", "[agent]\nuser = \"agent\"\n")
TEST_SH = """\
#!/bin/bash
cd /app
pytest /tests/test_outputs.py -rA
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
DOCKER = "FROM python:3.12\nWORKDIR /app\n"


def make_pkg(files: dict) -> Package:
    with tempfile.TemporaryDirectory() as d:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(textwrap.dedent(text))
        return Package.from_dir(d)


def task(**over) -> dict:
    files = {"task.toml": TOML, "instruction.md": "Do the thing.\n", "tests/test.sh": TEST_SH,
             "tests/test_outputs.py": "def test_ok():\n    assert True\n", "environment/Dockerfile": DOCKER}
    for k, v in over.items():
        k = k.replace("__", "/").replace("_DOT_", ".")
        if v is None:
            files.pop(k, None)
        else:
            files[k] = v
    return files


def run(rid: str, files: dict):
    return RULES[rid]["fn"](make_pkg(files))


GOLD = """\
diff --git a/src/calc.py b/src/calc.py
--- a/src/calc.py
+++ b/src/calc.py
@@ -1,3 +1,6 @@
 def add(a, b):
-    return a - b
+    result = a + b
+    log_value("add", result)
+    return result
+
+def multiply_values(a, b):
+    return a * b
"""


class TestAppliesReference(unittest.TestCase):
    rid = "verifier-applies-reference-patch"

    def test_applies_copy_of_gold_patch(self):
        # .patch files are not read by Package (see notes), so the gold patch sits in solve.sh
        solve = "#!/bin/bash\ngit apply <<'EOF'\n" + GOLD + "EOF\n"
        sh = "#!/bin/bash\ncd /app\ncat > /tmp/fix.patch <<'EOF'\n" + GOLD + "EOF\ngit apply /tmp/fix.patch\n" + TEST_SH.split("cd /app\n", 1)[1]
        r = run(self.rid, task(**{"solution/solve.sh": solve, "tests/test.sh": sh}))
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].kind, "hack")
        self.assertEqual(r[0].confidence, "review")

    def test_heredoc_gold_patch(self):
        sh = "#!/bin/bash\ncd /app\ngit apply <<'EOF'\n" + GOLD + "EOF\n" + TEST_SH.split("cd /app\n", 1)[1]
        r = run(self.rid, task(**{"solution/solve.sh": "#!/bin/bash\ncat > /tmp/p.diff <<'EOF'\n" + GOLD + "EOF\ngit apply /tmp/p.diff\n",
                                  "tests/test.sh": sh}))
        self.assertEqual(len(r), 1)

    def test_copies_solution_dir(self):
        sh = "#!/bin/bash\ncp /solution/calc.py /app/src/calc.py\n" + TEST_SH
        r = run(self.rid, task(**{"tests/test.sh": sh}))
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].kind, "broken")

    def test_copies_reference_file(self):
        ref = "def add(a, b):\n    result = a + b\n    log_value('add', result)\n    return result\n\ndef multiply_values(a, b):\n    return a * b\n"
        sh = "#!/bin/bash\ncp /tests/reference_calc.py /app/src/calc.py\n" + TEST_SH
        r = run(self.rid, task(**{"solution/calc.py": ref, "tests/reference_calc.py": ref, "tests/test.sh": sh}))
        self.assertEqual(len(r), 1)

    def test_test_patch_is_fine(self):
        tpatch = GOLD.replace("src/calc.py", "tests/test_calc.py").replace("log_value", "assert_value").replace("multiply_values", "test_multiply")
        tpatch = tpatch.replace("result = a + b", "assert add(1, 2) == 3").replace("return result", "assert add(2, 2) == 4")
        tpatch = tpatch.replace("return a * b", "assert multiply(2, 3) == 6")
        solve = "#!/bin/bash\ngit apply <<'EOF'\n" + GOLD + "EOF\n"
        sh = "#!/bin/bash\ncd /app\ngit apply <<'EOF'\n" + tpatch + "EOF\n" + TEST_SH.split("cd /app\n", 1)[1]
        r = run(self.rid, task(**{"solution/solve.sh": solve, "tests/test.sh": sh}))
        self.assertEqual(r, [])

    def test_apply_after_tests_is_fine(self):
        solve = "#!/bin/bash\ngit apply <<'EOF'\n" + GOLD + "EOF\n"
        sh = TEST_SH + "cat > /tmp/fix.patch <<'EOF'\n" + GOLD + "EOF\ngit apply /tmp/fix.patch\n"
        r = run(self.rid, task(**{"solution/solve.sh": solve, "tests/test.sh": sh}))
        self.assertEqual(r, [])

    def test_no_harbor_task(self):
        self.assertEqual(run(self.rid, {"env.py": "import os\n", "tests/test.sh": "git apply /tests/fix.patch\n"}), [])


class TestShellFailOpen(unittest.TestCase):
    rid = "shell-or-true-fail-open"

    def test_pipe_to_tee_without_pipefail(self):
        sh = "#!/bin/bash\npytest /tests -rA | tee /logs/verifier/out.txt\nrc=$?\nif [ $rc -eq 0 ]; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n"
        r = run(self.rid, task(**{"tests/test.sh": sh}))
        self.assertEqual(len(r), 1)
        self.assertIn("pipefail", r[0].message)

    def test_or_true_then_status(self):
        sh = "#!/bin/bash\npytest /tests -rA || true\nif [ $? -eq 0 ]; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n"
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": sh}))), 1)

    def test_no_failure_line_pays(self):
        sh = ("#!/bin/bash\npytest /tests -rA > /tmp/log.txt 2>&1 || true\n"
              "if ! grep -q FAILED /tmp/log.txt; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n")
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": sh}))), 1)

    def test_failure_count_zero_pays(self):
        sh = ("#!/bin/bash\npython3 /tests/run.py > /tmp/log.txt 2>&1\nFAILS=$(grep -c FAIL /tmp/log.txt)\n"
              "if [ \"$FAILS\" -eq 0 ]; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n")
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": sh}))), 1)

    def test_unconditional_one(self):
        sh = "#!/bin/bash\npytest /tests -rA\necho 1 > /logs/verifier/reward.txt\n"
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": sh}))), 1)

    def test_plain_exit_code_is_fine(self):
        self.assertEqual(run(self.rid, task()), [])

    def test_pipe_with_pipefail_is_fine(self):
        sh = "#!/bin/bash\nset -o pipefail\npytest /tests -rA | tee /logs/verifier/out.txt\nrc=$?\nif [ $rc -eq 0 ]; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n"
        self.assertEqual(run(self.rid, task(**{"tests/test.sh": sh})), [])

    def test_errexit_unconditional_is_fine(self):
        sh = "#!/bin/bash\nset -euo pipefail\npytest /tests -rA\necho 1 > /logs/verifier/reward.txt\n"
        self.assertEqual(run(self.rid, task(**{"tests/test.sh": sh})), [])

    def test_template_placeholder_skipped(self):
        sh = "#!/bin/bash\n{test_commands}\nuv run parser.py | tee -a log.txt\nexit_code=$?\nif [ \"${exit_code}\" -eq 0 ]; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n"
        self.assertEqual(run(self.rid, task(**{"tests/test.sh": sh})), [])


class TestTomlKeys(unittest.TestCase):
    rid = "task-toml-unknown-keys"

    def test_typo_in_environment_mode(self):
        r = run(self.rid, task(**{"task_DOT_toml": TOML.replace("[verifier]\n", "[verifier]\nenvironment_mod = \"separate\"\n")}))
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].kind, "hack")
        self.assertIn("environment_mode", r[0].message)

    def test_typo_in_timeout(self):
        r = run(self.rid, task(**{"task_DOT_toml": TOML.replace("[agent]\ntimeout_sec", "[agent]\ntimeout_secs")}))
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].kind, "broken")

    def test_empty_instruction(self):
        r = run(self.rid, task(**{"instruction_DOT_md": "  \n"}))
        self.assertEqual(len(r), 1)
        self.assertIn("empty", r[0].message)

    def test_missing_instruction_not_reported(self):
        self.assertEqual(run(self.rid, task(**{"instruction_DOT_md": None})), [])

    def test_missing_test_script(self):
        r = run(self.rid, task(**{"tests__test_DOT_sh": None}))
        self.assertEqual(len(r), 1)

    def test_valid_toml(self):
        self.assertEqual(run(self.rid, task()), [])

    def test_free_form_tables_ok(self):
        t = TOML + "[metadata]\nanything = 1\n[environment.env]\nFOO = \"1\"\n[verifier.env]\nBAR = \"2\"\n"
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": t})), [])

    def test_legacy_keys_accepted(self):
        t = TOML.replace("schema_version", "version").replace("memory_mb = 2048", "memory = \"2G\"")
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": t})), [])

    def test_separate_verifier_image_may_bundle_tests(self):
        t = SEP_TOML + "[verifier.environment]\ndocker_image = \"grader:1\"\n"
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": t, "tests__test_DOT_sh": None})), [])


ECHO_TESTS = "EXPECTED = ['n', 'get egg', 'open window', 'take lamp', 'go north']\n"


class TestOracleEcho(unittest.TestCase):
    rid = "oracle-solution-echoes-answer"

    def test_echo_lines(self):
        solve = "#!/bin/bash\necho n >> /app/moves.txt\necho get egg >> /app/moves.txt\necho open window >> /app/moves.txt\necho take lamp >> /app/moves.txt\n"
        r = run(self.rid, task(**{"solution/solve.sh": solve, "tests/test_outputs.py": ECHO_TESTS}))
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].kind, "hygiene")

    def test_heredoc_data(self):
        solve = "#!/bin/bash\ncat > /app/moves.txt <<'EOF'\nget egg\nopen window\ntake lamp\nEOF\n"
        self.assertEqual(len(run(self.rid, task(**{"solution/solve.sh": solve, "tests/test_outputs.py": ECHO_TESTS}))), 1)

    def test_printf_lines(self):
        solve = "#!/bin/bash\nmkdir -p /app\nprintf 'get egg\\nopen window\\ntake lamp\\n' > /app/moves.txt\n"
        self.assertEqual(len(run(self.rid, task(**{"solution/solve.sh": solve, "tests/test_outputs.py": ECHO_TESTS}))), 1)

    def test_solution_runs_a_program(self):
        solve = "#!/bin/bash\npython3 /app/extract.py > /app/moves.txt\necho get egg >> /app/moves.txt\n"
        self.assertEqual(run(self.rid, task(**{"solution/solve.sh": solve, "tests/test_outputs.py": ECHO_TESTS})), [])

    def test_solution_writes_code(self):
        solve = "#!/bin/bash\ncat > /app/solve.py <<'EOF'\nget egg\nopen window\ntake lamp\nEOF\n"
        self.assertEqual(run(self.rid, task(**{"solution/solve.sh": solve, "tests/test_outputs.py": ECHO_TESTS})), [])

    def test_single_short_answer(self):
        solve = "#!/bin/bash\necho 42 > /app/answer.txt\n"
        self.assertEqual(run(self.rid, task(**{"solution/solve.sh": solve, "tests/test_outputs.py": "assert ans == '42'\n"})), [])


class TestAgentRoot(unittest.TestCase):
    rid = "agent-runs-as-root"

    def test_compose_user_root(self):
        c = "services:\n  main:\n    build: .\n    user: root\n"
        r = run(self.rid, task(**{"environment__docker-compose_DOT_yaml": c}))
        self.assertEqual(len(r), 1)

    def test_no_user_is_default_not_reported(self):
        self.assertEqual(run(self.rid, task()), [])

    def test_explicit_root(self):
        r = run(self.rid, task(**{"task_DOT_toml": TOML.replace("[agent]\n", "[agent]\nuser = \"root\"\n")}))
        self.assertEqual(len(r), 1)

    def test_dockerfile_user_root(self):
        r = run(self.rid, task(**{"environment__Dockerfile": DOCKER + "USER root\n"}))
        self.assertEqual(len(r), 1)

    def test_non_root_user(self):
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": USER_TOML})), [])

    def test_dockerfile_non_root(self):
        self.assertEqual(run(self.rid, task(**{"environment__Dockerfile": DOCKER + "RUN useradd agent\nUSER agent\n"})), [])

    def test_separate_verifier(self):
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": SEP_TOML})), [])


class TestSecrets(unittest.TestCase):
    rid = "secrets-in-agent-env"

    def test_toml_environment_env(self):
        t = TOML + "[environment.env]\nOPENAI_API_KEY = \"${OPENAI_API_KEY}\"\n"
        r = run(self.rid, task(**{"task_DOT_toml": t}))
        self.assertEqual(len(r), 1)
        self.assertIn("OPENAI_API_KEY", r[0].message)

    def test_dockerfile_env_literal(self):
        d = DOCKER + "ENV HF_TOKEN=hf_abcdEFGH1234567890xyz\n"
        self.assertEqual(len(run(self.rid, task(**{"environment__Dockerfile": d}))), 1)

    def test_compose_main_env(self):
        c = "services:\n  main:\n    build: .\n    environment:\n      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}\n"
        self.assertEqual(len(run(self.rid, task(**{"environment__docker-compose_DOT_yaml": c}))), 1)

    def test_verifier_env_is_fine(self):
        t = TOML + "[verifier.env]\nOPENAI_API_KEY = \"${OPENAI_API_KEY}\"\n"
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": t})), [])

    def test_dummy_value_is_fine(self):
        d = DOCKER + "ENV OPENAI_API_KEY=dummy\nENV MAX_OUTPUT_TOKENS=128000\n"
        self.assertEqual(run(self.rid, task(**{"environment__Dockerfile": d})), [])

    def test_sidecar_env_is_fine(self):
        c = "services:\n  main:\n    build: .\n  judge:\n    image: j\n    environment:\n      - OPENAI_API_KEY=${OPENAI_API_KEY}\n"
        self.assertEqual(run(self.rid, task(**{"environment__docker-compose_DOT_yaml": c})), [])


class TestComposeMounts(unittest.TestCase):
    rid = "compose-writable-bind-mounts"

    def compose(self, vol, svc="main"):
        return f"services:\n  {svc}:\n    build: .\n    volumes:\n      - {vol}\n"

    def test_writable_answers_mount(self):
        self.assertEqual(len(run(self.rid, task(**{"environment__docker-compose_DOT_yaml": self.compose("./answers:/data/answers")}))), 1)

    def test_writable_host_path(self):
        self.assertEqual(len(run(self.rid, task(**{"environment__docker-compose_DOT_yaml": self.compose("/srv/shared:/mnt/shared")}))), 1)

    def test_long_syntax_parent_dir(self):
        c = "services:\n  main:\n    volumes:\n      - type: bind\n        source: ../expected\n        target: /expected\n"
        self.assertEqual(len(run(self.rid, task(**{"environment__docker-compose_DOT_yaml": c}))), 1)

    def test_read_only_is_fine(self):
        self.assertEqual(run(self.rid, task(**{"environment__docker-compose_DOT_yaml": self.compose("./answers:/data/answers:ro")})), [])

    def test_harbor_log_mounts_fine(self):
        c = self.compose("${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}")
        self.assertEqual(run(self.rid, task(**{"environment__docker-compose_DOT_yaml": c})), [])

    def test_named_volume_and_sidecar_fine(self):
        c = "services:\n  main:\n    volumes:\n      - cache:/cache\n  db:\n    volumes:\n      - ./data:/var/lib/data\nvolumes:\n  cache:\n"
        self.assertEqual(run(self.rid, task(**{"environment__docker-compose_DOT_yaml": c})), [])


class TestSeparateUndeclared(unittest.TestCase):
    rid = "separate-verifier-reads-undeclared-path"
    INSTR = "Write the result to /app/output/result.json.\n"
    TEST = "import json\n\ndef test_out():\n    d = json.load(open('/app/output/result.json'))\n    assert d['ok']\n"

    def test_undeclared_output(self):
        r = run(self.rid, task(**{"task_DOT_toml": SEP_TOML, "instruction_DOT_md": self.INSTR, "tests/test_outputs.py": self.TEST}))
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].kind, "broken")

    def test_trajectory_without_artifact(self):
        t = "import json\n\ndef test_traj():\n    t = json.load(open('/logs/agent/trajectory.json'))\n    assert t\n"
        self.assertEqual(len(run(self.rid, task(**{"task_DOT_toml": SEP_TOML, "tests/test_outputs.py": t}))), 1)

    def test_step_separate(self):
        t = TOML + "[[steps]]\nname = \"one\"\n[steps.verifier]\nenvironment_mode = \"separate\"\n"
        r = run(self.rid, task(**{"task_DOT_toml": t, "instruction_DOT_md": self.INSTR, "tests/test_outputs.py": self.TEST,
                                  "steps/one/instruction.md": self.INSTR}))
        self.assertEqual(len(r), 1)

    def test_declared_artifact(self):
        t = "artifacts = [\"/app/output\"]\n" + SEP_TOML
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": t, "instruction_DOT_md": self.INSTR, "tests/test_outputs.py": self.TEST})), [])

    def test_shared_mode(self):
        self.assertEqual(run(self.rid, task(**{"instruction_DOT_md": self.INSTR, "tests/test_outputs.py": self.TEST})), [])

    def test_absence_check(self):
        sh = "#!/bin/bash\nif [ -e /tmp/agent-only.txt ]; then\n  echo 0 > /logs/verifier/reward.txt\n  exit 0\nfi\necho 1 > /logs/verifier/reward.txt\n"
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": SEP_TOML, "instruction_DOT_md": "Write /tmp/agent-only.txt.\n",
                                               "tests/test.sh": sh})), [])

    def test_baked_into_image(self):
        d = DOCKER + "COPY data/result.json /app/output/result.json\n"
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": SEP_TOML, "instruction_DOT_md": self.INSTR,
                                               "tests/test_outputs.py": self.TEST, "environment__Dockerfile": d})), [])


class TestSubmissionPrivileged(unittest.TestCase):
    rid = "verifier-runs-submission-privileged"

    def sh(self, build):
        return f"#!/bin/bash\ncd /app\n{build}\nrunuser -u nobody -- pytest /tests -rA\nif [ $? -eq 0 ]; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n"

    def test_pip_requirements(self):
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": self.sh("pip install -r requirements.txt")}))), 1)

    def test_npm_install(self):
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": self.sh("npm ci")}))), 1)

    def test_make(self):
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": self.sh("make all")}))), 1)

    def test_no_drop(self):
        sh = self.sh("pip install -r requirements.txt").replace("runuser -u nobody -- ", "")
        self.assertEqual(run(self.rid, task(**{"tests/test.sh": sh})), [])

    def test_build_under_drop(self):
        self.assertEqual(run(self.rid, task(**{"tests/test.sh": self.sh("runuser -u nobody -- make all")})), [])

    def test_grader_requirements(self):
        self.assertEqual(run(self.rid, task(**{"tests/test.sh": self.sh("pip install -r /tests/requirements.txt")})), [])


class TestRootOps(unittest.TestCase):
    rid = "root-ops-on-agent-writable-paths"

    def sh(self, op):
        return f"#!/bin/bash\n{op}\npytest /tests -rA\nif [ $? -eq 0 ]; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n"

    def test_chmod_world_readable(self):
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": self.sh("chmod -R 777 /app/output")}))), 1)

    def test_chown(self):
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": self.sh("chown -R nobody /app")}))), 1)

    def test_chmod_artifacts(self):
        self.assertEqual(len(run(self.rid, task(**{"tests/test.sh": self.sh("chmod a+r /logs/artifacts/out")}))), 1)

    def test_no_dereference(self):
        self.assertEqual(run(self.rid, task(**{"tests/test.sh": self.sh("chown -hR nobody /app")})), [])

    def test_chmod_exec_script(self):
        self.assertEqual(run(self.rid, task(**{"tests/test.sh": self.sh("chmod +x /app/run.sh")})), [])

    def test_separate_mode(self):
        self.assertEqual(run(self.rid, task(**{"task_DOT_toml": SEP_TOML, "tests/test.sh": self.sh("chown -R nobody /app")})), [])


class TestRobust(unittest.TestCase):
    def test_odd_input_does_not_crash(self):
        files = {"task.toml": "this is [not toml", "tests/test.sh": "if [ $? \n fi fi ) ( |||\n", "instruction.md": "",
                 "environment/docker-compose.yaml": "services:\n  main:\n    volumes:\n      - \n      -: :\n",
                 "solution/solve.sh": "cat <<EOF\nno end\n", "environment/Dockerfile": "ENV\nUSER\n"}
        pkg = make_pkg(files)
        for rid, meta in RULES.items():
            if meta["fn"].__module__.endswith("rules_harbor"):
                meta["fn"](pkg)

    def test_no_task_is_empty(self):
        pkg = make_pkg({"env.py": "def load_environment():\n    return None\n"})
        for rid, meta in RULES.items():
            if meta["fn"].__module__.endswith("rules_harbor"):
                self.assertEqual(meta["fn"](pkg), [], rid)


if __name__ == "__main__":
    unittest.main()

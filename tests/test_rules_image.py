"""Tests for the image rule family (envscan/rules_image.py).

Each case writes a tiny package into a temp folder and loads it with Package.from_dir,
the same path a real scan takes. Run from the scanner folder:
    python3 -m unittest discover -s tests -t .
"""
import os
import tempfile
import textwrap
import unittest

from envscan.package import Package
from envscan.rules import RULES
from envscan import rules_image  # noqa: F401  (registers the rules)

HARBOR_TOML = """\
schema_version = "1.4"
[task]
name = "demo/task"
[verifier]
timeout_sec = 120.0
[agent]
timeout_sec = 120.0
[environment]
cpus = 1
"""


def make_pkg(files: dict) -> Package:
    with tempfile.TemporaryDirectory() as d:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(textwrap.dedent(text))
        return Package.from_dir(d)


def run(rid: str, files: dict):
    return RULES[rid]["fn"](make_pkg(files))


class Base(unittest.TestCase):
    rid = ""

    def hit(self, files, msg=None):
        res = run(self.rid, files)
        self.assertEqual(len(res), 1, msg or res)
        f = res[0]
        self.assertEqual(f.rule, self.rid)
        self.assertTrue(f.evidence and all(":" in e for e in f.evidence), f.evidence)
        self.assertNotIn(";", f.message)
        self.assertNotIn("—", f.message)
        return f

    def miss(self, files, msg=None):
        self.assertEqual(run(self.rid, files), [], msg)


# ---------------------------------------------------------------------------

class TestGitHistory(Base):
    rid = "git-history-or-target-in-image"

    def test_full_clone_then_checkout_base_commit(self):
        f = self.hit({"environment/Dockerfile": """\
            FROM python:3.12
            WORKDIR /app
            RUN git clone https://github.com/acme/widget . && git checkout dd05eebc8ed9e30fc6c2
            """, "task.toml": HARBOR_TOML})
        self.assertIn("environment/Dockerfile:3", f.evidence)

    def test_clone_at_tag_keeps_other_branches(self):
        self.hit({"Dockerfile": """\
            FROM ubuntu:24.04
            RUN git clone --branch v1.2.0 https://github.com/acme/widget.git /testbed
            """})

    def test_target_package_also_from_pypi(self):
        f = self.hit({"Dockerfile": """\
            FROM python:3.12
            RUN git clone --depth 1 https://github.com/acme/widget.git /app
            RUN pip install widget==2.0
            """})
        self.assertIn("site-packages", f.message)

    def test_task_named_repo_outside_workspace(self):
        self.hit({"environment/Dockerfile": """\
            FROM ubuntu:24.04
            RUN git clone https://github.com/apache/druid.git && cd druid && git checkout druid-0.20.0
            """, "instruction.md": "Fix the CVE in Apache Druid.\n"})

    def test_python_setup_checks_out_older_commit(self):
        f = self.hit({"env.py": """\
            class Env:
                async def setup_repo(self, state):
                    sid = state["sandbox_id"]
                    await self.execute_command(sid, "git checkout HEAD~1", working_dir="/testbed")
            """})
        self.assertIn("env.py:4", f.evidence)

    def test_shallow_clone_is_fine(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            RUN git clone --depth 1 --branch v1.2.0 https://github.com/acme/widget.git /app
            """})

    def test_history_removed(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            RUN git clone https://github.com/acme/widget /app && cd /app && git checkout abc1234 \\
                && rm -rf .git && git init
            """})

    def test_reflog_expired_and_pruned(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            RUN git clone https://github.com/acme/widget /app && cd /app && git checkout abc1234 && \\
                git branch -D main && git remote remove origin && git tag -l | xargs -r git tag -d && \\
                git reflog expire --expire=now --all && git gc --prune=now
            """})

    def test_tool_built_outside_workspace(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            RUN git clone https://github.com/acme/buildtool /opt/buildtool && cd /opt/buildtool && git checkout 1a2b3c4d
            """})

    def test_checkout_in_scoring_function(self):
        self.miss({"env.py": """\
            async def compute_reward(state, sandbox):
                await sandbox.execute_command(state["id"], "git checkout HEAD~1 -- tests/")
                return 1.0
            """})


# ---------------------------------------------------------------------------

class TestAnswersReadable(Base):
    rid = "answers-or-grader-readable"

    def test_harbor_tests_copied_into_agent_image(self):
        f = self.hit({"task.toml": HARBOR_TOML, "tests/test.sh": "pytest /tests\n",
                      "environment/Dockerfile": "FROM ubuntu:24.04\nCOPY tests /tests\n"})
        self.assertIn("environment/Dockerfile:2", f.evidence)

    def test_expected_outputs_copied(self):
        self.hit({"Dockerfile": "FROM python:3.12\nCOPY expected_output.json /app/expected_output.json\n"})

    def test_hidden_archive_built_into_image(self):
        self.hit({"Dockerfile": """\
            FROM python:3.12
            RUN python3 gen.py --hidden-dir /tmp/bench-hidden \\
                && tar czf /opt/verifier-hidden-benchmarks.tar.gz -C /tmp bench-hidden
            """})

    def test_setup_links_hidden_tests_into_workspace(self):
        self.hit({"env.py": """\
            async def setup_repo(self, sid):
                await self.execute_command(sid, "rm -rf /r2e_tests")
                await self.execute_command(sid, "ln -s /root/r2e_tests /testbed/r2e_tests")
            """})

    def test_r2e_tests_never_removed(self):
        self.hit({"env.py": """\
            RUN = "cd /testbed && cp -r /r2e_tests . && bash run_tests.sh"
            def load_environment():
                return RUN
            """})

    def test_dependency_file_from_tests_is_fine(self):
        self.miss({"Dockerfile": "FROM python:3.12\nCOPY tests/requirements.txt /tmp/req.txt\n"})

    def test_removed_later(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            COPY expected_outputs/ /tmp/expected/
            RUN python3 build.py /tmp/expected && rm -rf /tmp/expected
            """})

    def test_root_only_with_non_root_agent(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            COPY hidden_tests/ /root/hidden_tests/
            RUN chmod -R 700 /root/hidden_tests
            USER agent
            """})

    def test_verifier_image_may_hold_tests(self):
        self.miss({"task.toml": HARBOR_TOML, "tests/test.sh": "pytest /tests\n",
                   "environment/Dockerfile": "FROM ubuntu:24.04\n",
                   "tests/Dockerfile": "FROM ubuntu:24.04\nCOPY . /tests/\nCOPY solution/solve.sh /x\n"})

    def test_r2e_tests_removed_before_agent(self):
        self.miss({"env.py": """\
            CLEAN = "tar -C / -czf /tmp/t.tgz r2e_tests && rm -rf /r2e_tests"
            def setup(rt):
                return rt.run(CLEAN)
            """})


# ---------------------------------------------------------------------------

PORT_READS_REWARD = """\
class Env:
    async def compute_reward(self, state):
        sid = state["sandbox_id"]
        await self.sandbox_client.run_background_job(sid, "bash test.sh", working_dir="/tests")
        res = await self.sandbox_client.execute_command(
            sid, "if [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; fi")
        return float(res.stdout)
"""


class TestRewardChannel(Base):
    rid = "writable-reward-channel"

    def test_port_reads_reward_from_agent_box(self):
        f = self.hit({"env.py": PORT_READS_REWARD})
        self.assertIn("env.py:5", f.evidence)

    def test_reward_copied_from_agent_path(self):
        self.hit({"task.toml": HARBOR_TOML, "tests/test.sh": """\
            #!/bin/bash
            python3 /app/run_eval.py
            cp /app/score.txt /logs/verifier/reward.txt
            """})

    def test_reward_written_only_when_missing(self):
        self.hit({"task.toml": HARBOR_TOML, "tests/test.sh": """\
            #!/bin/bash
            pytest /tests/test_outputs.py
            if [ ! -f /logs/verifier/reward.txt ]; then
              echo 0 > /logs/verifier/reward.txt
            fi
            """})

    def test_agent_processes_killed_first(self):
        self.miss({"env.py": PORT_READS_REWARD.replace(
            '        await self.sandbox_client.run_background_job',
            '        await self.sandbox_client.execute_command(sid, "pkill -9 -u agent")\n'
            '        await self.sandbox_client.run_background_job')})

    def test_grading_in_fresh_sandbox(self):
        self.miss({"env.py": PORT_READS_REWARD.replace(
            '        sid = state["sandbox_id"]',
            '        sid = (await self.sandbox_client.create_sandbox(image="grader")).id')})

    def test_reward_from_file_the_verifier_wrote(self):
        self.miss({"task.toml": HARBOR_TOML,
                   "tests/test_metrics.py": "from pathlib import Path\nPath('/app/metrics.json').write_text('{}')\n",
                   "tests/test.sh": """\
            #!/bin/bash
            pytest /tests/test_metrics.py
            cp /app/metrics.json /logs/verifier/reward.json
            """})

    def test_unconditional_reward_write(self):
        self.miss({"task.toml": HARBOR_TOML, "tests/test.sh": """\
            #!/bin/bash
            if pytest /tests; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
            """})


# ---------------------------------------------------------------------------

TB_TEST_SH = """\
#!/bin/bash
apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
source $HOME/.local/bin/env
uvx --with pytest==8.4.1 pytest /tests/test_state.py
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


class TestVerifyTimeFetch(Base):
    rid = "verify-time-fetch"

    def test_installer_piped_to_shell_in_shared_box(self):
        f = self.hit({"task.toml": HARBOR_TOML, "tests/test.sh": TB_TEST_SH})
        self.assertIn("curl to shell", f.message)
        self.assertIn("tests/test.sh:4", f.evidence)

    def test_pip_install_at_verify_time(self):
        self.hit({"task.toml": HARBOR_TOML, "tests/test.sh": "pip install -q openai\npython3 /tests/judge.py\n"})

    def test_ensure_uv_probe_in_finalize(self):
        self.hit({"taskset.py": """\
            from verifiers.v1.runtimes.base import _ENSURE_UV
            class Task:
                async def finalize(self, trace, runtime):
                    await runtime.run(["sh", "-c", _ENSURE_UV], {})
            """})

    def test_reward_installs_into_agent_sandbox(self):
        self.hit({"env.py": """\
            async def correctness_reward(state, sandbox_client):
                sid = state["sandbox_id"]
                await sandbox_client.execute_command(sid, "pip install pytest && pytest /workspace/tests")
                return 1.0
            """})

    def test_separate_verifier(self):
        self.miss({"task.toml": HARBOR_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n'),
                   "tests/test.sh": TB_TEST_SH})

    def test_verifier_environment_implies_separate(self):
        self.miss({"task.toml": HARBOR_TOML + "[verifier.environment]\ncpus = 1\n", "tests/test.sh": TB_TEST_SH})

    def test_non_root_agent(self):
        self.miss({"task.toml": HARBOR_TOML.replace("[agent]\n", '[agent]\nuser = "agent"\n'),
                   "tests/test.sh": TB_TEST_SH})

    def test_offline_install(self):
        self.miss({"task.toml": HARBOR_TOML,
                   "tests/test.sh": "pip install --no-index --find-links /opt/wheels pytest\npytest /tests\n"})

    def test_fresh_grading_sandbox(self):
        self.miss({"env.py": """\
            async def correctness_reward(state, client):
                box = await client.create_sandbox(image="python:3.12")
                await client.execute_command(box.id, "pip install numpy && python /tmp/check.py")
                return 1.0
            """})

    def test_agent_tool_named_like_a_test(self):
        self.miss({"env.py": """\
            def test_fn_on_examples(code, sandbox_id, sandbox_client):
                sandbox_client.execute_command(sandbox_id, "pip install -q numpy")
                return "ok"

            class Env:
                def __init__(self):
                    self.tools = [test_fn_on_examples]
            """})


# ---------------------------------------------------------------------------

class TestEgress(Base):
    rid = "egress-and-shared-state"

    SWE_TOML = HARBOR_TOML + ('[metadata]\nrepository_url = "https://github.com/acme/widget"\n'
                              'base_commit_hash = "dd05eebc"\n')

    def test_swe_task_with_internet(self):
        self.hit({"task.toml": self.SWE_TOML.replace("[environment]\n", "[environment]\nallow_internet = true\n")})

    def test_swe_task_with_bridge_network(self):
        self.hit({"task.toml": self.SWE_TOML.replace("[environment]\n", '[environment]\nnetwork_mode = "bridge"\n')})

    def test_port_forces_network_on(self):
        f = self.hit({
            "tasks/a/task.toml": HARBOR_TOML.replace("[environment]\n", "[environment]\nallow_internet = false\n"),
            "env.py": "def make(cfg):\n    return dict(image=cfg.image, network_access=True)\n"})
        self.assertIn("env.py:2", f.evidence)

    def test_internet_closed(self):
        self.miss({"task.toml": self.SWE_TOML.replace("[environment]\n", "[environment]\nallow_internet = false\n")})

    def test_internet_without_public_repo(self):
        self.miss({"task.toml": HARBOR_TOML.replace("[environment]\n", "[environment]\nallow_internet = true\n")})

    def test_port_follows_task_setting(self):
        self.miss({
            "tasks/a/task.toml": HARBOR_TOML.replace("[environment]\n", "[environment]\nallow_internet = false\n"),
            "env.py": "def make(cfg):\n    return dict(network_access=True if cfg.allow_internet else False)\n"})


# ---------------------------------------------------------------------------
# v0.5 items
# ---------------------------------------------------------------------------

def harbor(dockerfile="FROM python:3.12\nWORKDIR /app\n", test_sh="#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
           toml=HARBOR_TOML, **extra):
    files = {"task.toml": toml, "environment/Dockerfile": dockerfile, "tests/test.sh": test_sh}
    files.update({k.replace("__", "/").replace("_DOT_", ".").replace("_DASH_", "-"): v for k, v in extra.items()})
    return files


TEST_SH_UVX = """\
#!/bin/bash
uvx -p 3.13 -w pytest==8.4.1 pytest /tests/test_outputs.py -rA
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""
RUNS_APP = """\
import subprocess
def test_runs():
    out = subprocess.run(["/app/solve", "--input", "x"], capture_output=True, text=True)
    assert out.stdout.strip() == "42"
"""
READS_ONLY = """\
from pathlib import Path
import subprocess
def test_output():
    assert Path("/app/out.txt").read_text().strip() == "42"
"""


class TestDockerPrivilege(Base):
    rid = "docker-privilege-or-socket"

    def test_main_service_privileged(self):
        f = self.hit(harbor(environment__docker_DASH_compose_DOT_yaml="services:\n  main:\n    privileged: true\n"))
        self.assertIn("environment/docker-compose.yaml:3", f.evidence)
        self.assertIn("privileged", f.message)

    def test_main_service_docker_socket(self):
        f = self.hit(harbor(environment__docker_DASH_compose_DOT_yaml="""\
            services:
              main:
                volumes:
                  - /var/run/docker.sock:/var/run/docker.sock
            """))
        self.assertIn("Docker socket", f.message)

    def test_json_compose_host_network(self):
        self.hit(harbor(environment__docker_DASH_compose_DOT_yaml='{"services": {"main": {"network_mode": "host"}}}'))

    def test_script_docker_run_privileged(self):
        self.hit({"scripts/start_sandbox.sh": "#!/bin/bash\ndocker run -d --privileged --name box img sleep infinity\n"})

    def test_python_sdk_privileged_sandbox(self):
        f = self.hit({"env.py": """\
            import docker
            def setup_state(state):
                c = docker.from_env().containers.run("img", detach=True, privileged=True)
                state["box"] = c
            def run(state, cmd):
                return state["box"].exec_run(cmd)
            """})
        self.assertIn("env.py:3", f.evidence)

    def test_k8s_pod_host_pid(self):
        self.hit({"k8s/pod.yaml": "apiVersion: v1\nkind: Pod\nspec:\n  hostPID: true\n  containers: []\n"})

    def test_net_admin_with_in_container_firewall(self):
        self.hit(harbor(environment__docker_DASH_compose_DOT_yaml="services:\n  main:\n    cap_add:\n      - NET_ADMIN\n",
                        environment__restrict_DOT_sh="iptables -P OUTPUT DROP\n"))

    def test_sidecar_privileged_is_not_the_agent(self):
        self.miss(harbor(environment__docker_DASH_compose_DOT_yaml="""\
            services:
              main:
                depends_on: [judge]
              judge:
                privileged: true
            """))

    def test_net_admin_without_firewall(self):
        self.miss(harbor(environment__docker_DASH_compose_DOT_yaml="services:\n  main:\n    cap_add: [NET_ADMIN]\n"))

    def test_compose_outside_harbor_task(self):
        self.miss({"deploy/docker-compose.yaml": "services:\n  main:\n    privileged: true\n"})

    def test_plain_docker_run(self):
        self.miss({"run.sh": "docker run --rm -v $PWD/data:/data img python eval.py\n"})

    def test_socket_in_dataset_text(self):
        self.miss({"env.py": """\
            ROWS = [{"prompt": "Write compose", "answer": "volumes:\\n  - /var/run/docker.sock:/var/run/docker.sock"}]
            def run(sandbox, cmd):
                return sandbox.execute_command(cmd)
            """})

    def test_verifier_script_is_skipped(self):
        self.miss({"tests/run_judge.sh": "docker run --privileged judge\n"})


class TestRewardJsonPlanted(Base):
    rid = "writable-reward-channel"

    def test_tests_run_agent_binary(self):
        f = self.hit(harbor(test_sh=TEST_SH_UVX, tests__test_outputs_DOT_py=RUNS_APP))
        self.assertIn("tests/test.sh:2", f.evidence)
        self.assertIn("reward.json", f.message)

    def test_make_test_in_workspace(self):
        self.hit(harbor(test_sh="#!/bin/bash\ncd /app\nmake test && echo 1 > /logs/verifier/reward.txt || "
                                "echo 0 > /logs/verifier/reward.txt\n"))

    def test_runs_agent_script(self):
        self.hit(harbor(test_sh="#!/bin/bash\npython3 /app/main.py > /tmp/out\n"
                                "grep -q 42 /tmp/out && echo 1 > /logs/verifier/reward.txt\n"))

    def test_writes_reward_json_itself(self):
        self.miss(harbor(test_sh="#!/bin/bash\npython3 /app/main.py > /tmp/out\n"
                                 "echo '{\"reward\": 1}' > /logs/verifier/reward.json\n"))

    def test_removes_reward_json(self):
        self.miss(harbor(test_sh="#!/bin/bash\npython3 /app/main.py > /tmp/out\nrm -f /logs/verifier/reward.json\n"
                                 "echo 1 > /logs/verifier/reward.txt\n"))

    def test_runs_candidate_as_other_user(self):
        self.miss(harbor(test_sh="#!/bin/bash\nsu agent -c 'python3 /app/main.py' > /tmp/out\n"
                                 "echo 1 > /logs/verifier/reward.txt\n"))

    def test_tests_only_read_outputs(self):
        self.miss(harbor(test_sh=TEST_SH_UVX, tests__test_outputs_DOT_py=READS_ONLY))


class TestWritableExpected(Base):
    rid = "writable-reward-channel"
    SH = "#!/bin/bash\nuvx pytest /tests/test_outputs.py && echo 1 > /logs/verifier/reward.json\n"

    def test_baseline_module_in_workspace(self):
        f = self.hit(harbor("FROM python:3.12\nWORKDIR /app\nCOPY baseline.py /app/baseline.py\n", self.SH,
                            environment__baseline_DOT_py="def run(x):\n    return x\n",
                            tests__test_outputs_DOT_py="import sys\nsys.path.insert(0, '/app')\nfrom baseline import run\n"))
        self.assertIn("environment/Dockerfile:3", f.evidence)
        self.assertIn("/app/baseline.py", f.message)

    def test_threshold_file_read_by_tests(self):
        self.hit(harbor("FROM python:3.12\nCOPY thresholds.json /app/\n", self.SH,
                        environment__thresholds_DOT_json='{"min_acc": 0.5}',
                        tests__test_outputs_DOT_py="import json\nT = json.load(open('/app/thresholds.json'))\n"))

    def test_validator_run_by_tests(self):
        f = self.hit(harbor("FROM python:3.12\nCOPY validate.py /app/validate.py\n", self.SH,
                            environment__validate_DOT_py="print('ok')\n",
                            tests__test_outputs_DOT_py="import subprocess\n"
                                                       "def test_v():\n    subprocess.run(['python3', '/app/validate.py'])\n"))
        self.assertIn("validator", f.message)

    def test_do_not_modify_only_in_instruction(self):
        f = self.hit(harbor("FROM python:3.12\nCOPY config.yaml /app/config.yaml\n", self.SH,
                            environment__config_DOT_yaml="mode: strict\n",
                            instruction_DOT_md="Fix the parser. Do not modify /app/config.yaml.\n",
                            tests__test_outputs_DOT_py="import yaml\nC = yaml.safe_load(open('/app/config.yaml'))\n"))
        self.assertIn("only the instruction", f.message)

    def test_tests_ship_their_own_copy(self):
        self.miss(harbor("FROM python:3.12\nCOPY thresholds.json /app/\n", self.SH,
                         environment__thresholds_DOT_json='{"min_acc": 0.5}', tests__thresholds_DOT_json='{"min_acc": 0.5}',
                         tests__test_outputs_DOT_py="import json\nT = json.load(open('/app/thresholds.json'))\n"))

    def test_hash_checked(self):
        self.miss(harbor("FROM python:3.12\nCOPY thresholds.json /app/\n", self.SH,
                         environment__thresholds_DOT_json='{"min_acc": 0.5}',
                         tests__test_outputs_DOT_py="import hashlib, json\n"
                                                    "assert hashlib.sha256(open('/app/thresholds.json','rb').read())"
                                                    ".hexdigest() == 'ab'\n"))

    def test_non_root_agent_cannot_write(self):
        toml = HARBOR_TOML.replace("[agent]\n", "[agent]\nuser = \"agent\"\n")
        self.miss(harbor("FROM python:3.12\nCOPY thresholds.json /app/\n", self.SH, toml=toml,
                         environment__thresholds_DOT_json='{"min_acc": 0.5}',
                         tests__test_outputs_DOT_py="import json\nT = json.load(open('/app/thresholds.json'))\n"))

    def test_separate_verifier(self):
        toml = HARBOR_TOML.replace("[verifier]\n", "[verifier]\nenvironment_mode = \"separate\"\n")
        self.miss(harbor("FROM python:3.12\nCOPY thresholds.json /app/\n", self.SH, toml=toml,
                         environment__thresholds_DOT_json='{"min_acc": 0.5}',
                         tests__test_outputs_DOT_py="import json\nT = json.load(open('/app/thresholds.json'))\n"))


STATE_ENV = """\
import verifiers as vf
def {tool}({param}, state) -> str:
    {body}
    return "ok"
def reward_fn(completion, state, **kw):
    return float(state{read})
def load_environment():
    return vf.ToolEnv(tools=[{tool}], rubric=vf.Rubric(funcs=[reward_fn]))
"""


class TestStateVerdict(Base):
    rid = "writable-reward-channel"

    def env(self, tool="report_result", param="success: bool", body='state["success"] = success',
            read='["success"]'):
        return {"env.py": STATE_ENV.format(tool=tool, param=param, body=body, read=read)}

    def test_tool_sets_success(self):
        f = self.hit(self.env())
        self.assertIn("report_result", f.message)
        self.assertIn("env.py:3", f.evidence)

    def test_tool_sets_solved_through_bool(self):
        self.hit(self.env(param="done", body='state["solved"] = bool(done)', read='["solved"]'))

    def test_reward_reads_with_get(self):
        self.hit(self.env(param="score: float", body='state["score"] = score', read='.get("score", 0)'))

    def test_tool_computes_verdict(self):
        self.miss(self.env(param="answer: str", body='state["success"] = answer == state["target"]'))

    def test_reward_reads_other_key(self):
        self.miss(self.env(read='["checked"]'))

    def test_non_verdict_key(self):
        self.miss(self.env(param="answer: str", body='state["answer"] = answer', read='["answer"]'))


class TestOfflineClaim(Base):
    rid = "egress-and-shared-state"

    def test_claim_with_default_network(self):
        f = self.hit({"task.toml": HARBOR_TOML, "instruction.md": "Build the parser.\nYou do not have internet access.\n"})
        self.assertIn("instruction.md:2", f.evidence)

    def test_claim_with_public_agent_phase(self):
        toml = HARBOR_TOML.replace("[agent]\n", "[agent]\nnetwork_mode = \"public\"\n") + "allow_internet = false\n"
        self.hit({"task.toml": toml, "instruction.md": "The sandbox has no internet, so vendor everything.\n"})

    def test_do_not_use_internet(self):
        self.hit({"task.toml": HARBOR_TOML + "allow_internet = true\n",
                  "instruction.md": "Do not use the internet to look up the answer.\n"})

    def test_claim_matches_config(self):
        self.miss({"task.toml": HARBOR_TOML + "allow_internet = false\n",
                   "instruction.md": "You do not have internet access.\n"})

    def test_agent_phase_closed(self):
        toml = HARBOR_TOML.replace("[agent]\n", "[agent]\nnetwork_mode = \"no-network\"\n")
        self.miss({"task.toml": toml, "instruction.md": "You do not have internet access.\n"})

    def test_program_requirement_is_not_a_claim(self):
        self.miss({"task.toml": HARBOR_TOML, "instruction.md": "Your server must keep working when the network is down.\n"})

    def test_agent_phase_closes_swe_metadata_case(self):
        toml = HARBOR_TOML.replace("[agent]\n", "[agent]\nnetwork_mode = \"no-network\"\n") + \
            "allow_internet = true\n[metadata]\nrepository_url = \"https://github.com/acme/w\"\nbase_commit = \"abc\"\n"
        self.miss({"task.toml": toml})


class TestGitRefsLeft(Base):
    rid = "git-history-or-target-in-image"

    def test_weak_scrub_keeps_tags(self):
        f = self.hit({"Dockerfile": """\
            FROM python:3.12
            RUN git clone https://github.com/acme/widget /app && cd /app && git checkout abc1234 && \\
                git branch -D main && git remote remove origin && git reflog expire --expire=now --all && git gc --prune=now
            """})
        self.assertIn("tags", f.message)

    def test_weak_scrub_keeps_local_branch(self):
        f = self.hit({"Dockerfile": """\
            FROM python:3.12
            RUN git clone --no-tags https://github.com/acme/widget /app && cd /app && git checkout abc1234 && \\
                git remote remove origin && git gc --prune=now
            """})
        self.assertIn("local branch", f.message)

    def test_init_and_full_fetch(self):
        f = self.hit({"Dockerfile": """\
            FROM python:3.12
            WORKDIR /testbed
            RUN git init && git remote add origin https://github.com/acme/widget.git && git fetch origin && \\
                git checkout 1a2b3c4d5e
            """})
        self.assertIn("fetched", f.message)

    def test_fetch_of_one_commit(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            WORKDIR /testbed
            RUN git init && git remote add origin https://github.com/acme/widget.git && \\
                git fetch origin 1a2b3c4d5e && git checkout 1a2b3c4d5e
            """})

    def test_shallow_fetch(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            WORKDIR /testbed
            RUN git init && git remote add origin https://github.com/acme/widget.git && \\
                git fetch --depth 1 origin main && git checkout 1a2b3c4d5e
            """})

    def test_full_scrub_after_fetch(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            WORKDIR /testbed
            RUN git init && git remote add origin https://github.com/acme/widget.git && git fetch origin && \\
                git checkout 1a2b3c4d5e && rm -rf .git
            """})


class TestResidue(Base):
    rid = "git-history-or-target-in-image"

    def test_installed_then_patch_reverted(self):
        f = self.hit({"Dockerfile": """\
            FROM python:3.12
            COPY repo /app
            RUN cd /app && pip install . && git apply -R /tmp/fix.patch
            """})
        self.assertIn("site-packages", f.message)

    def test_tests_run_then_feature_removed(self):
        f = self.hit({"Dockerfile": """\
            FROM python:3.12
            COPY repo /app
            RUN cd /app && pip install -e . && python -m pytest -q tests
            RUN cd /app && git apply /tmp/remove_feature.patch
            """})
        self.assertIn(".pyc", f.message)

    def test_compiled_then_checked_out_older(self):
        self.hit({"Dockerfile": """\
            FROM python:3.12
            COPY repo /app
            RUN cd /app && python -m compileall -q src && git checkout HEAD~3
            """})

    def test_editable_install_then_revert(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            COPY repo /app
            RUN cd /app && pip install -e . && git apply -R /tmp/fix.patch
            """})

    def test_revert_then_install(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            COPY repo /app
            RUN cd /app && git apply -R /tmp/fix.patch && pip install .
            """})

    def test_cleanup_after_revert(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            COPY repo /app
            RUN cd /app && python -m pytest -q && git apply -R /tmp/fix.patch && \\
                find . -name '*.pyc' -delete
            """})

    def test_reinstall_after_revert(self):
        self.miss({"Dockerfile": """\
            FROM python:3.12
            COPY repo /app
            RUN cd /app && pip install . && git apply -R /tmp/fix.patch && pip install --no-deps --force-reinstall .
            """})


class TestOfflineVerifierFetch(unittest.TestCase):
    rid = "verify-time-fetch"

    def broken(self, files):
        return [f for f in run(self.rid, files) if f.kind == "broken"]

    def test_curl_installer_with_no_network(self):
        res = self.broken(harbor(test_sh="curl -LsSf https://astral.sh/uv/install.sh | sh\n"
                                         "echo 1 > /logs/verifier/reward.txt\n", toml=HARBOR_TOML + "allow_internet = false\n"))
        self.assertEqual(len(res), 1)
        self.assertIn("tests/test.sh:1", res[0].evidence)
        self.assertNotIn(";", res[0].message)

    def test_pip_install_with_verifier_no_network(self):
        toml = HARBOR_TOML.replace("[verifier]\n", "[verifier]\nnetwork_mode = \"no-network\"\n")
        self.assertEqual(len(self.broken(harbor(test_sh="pip install pytest\npytest /tests\n", toml=toml))), 1)

    def test_separate_verifier_offline_clone(self):
        toml = HARBOR_TOML.replace("[verifier]\n", "[verifier]\nenvironment_mode = \"separate\"\n") + \
            "[verifier.environment]\nallow_internet = false\n"
        self.assertEqual(len(self.broken(harbor(test_sh="git clone https://github.com/acme/checker /opt/c\n",
                                                toml=toml))), 1)

    def test_already_installed(self):
        self.assertEqual(self.broken(harbor("FROM python:3.12\nRUN pip install pytest==8.4.1\n",
                                            "pip install pytest==8.4.1\npytest /tests\n",
                                            toml=HARBOR_TOML + "allow_internet = false\n")), [])

    def test_npx_uses_local_bin(self):
        self.assertEqual(self.broken(harbor(test_sh="cd /app && npx jest --json > /logs/verifier/j.json\n",
                                            toml=HARBOR_TOML + "allow_internet = false\n")), [])

    def test_network_open(self):
        self.assertEqual(self.broken(harbor(test_sh="curl -LsSf https://astral.sh/uv/install.sh | sh\n",
                                            toml=HARBOR_TOML + "allow_internet = true\n")), [])


LABELS_DOCKER = "FROM python:3.12\nWORKDIR /app\nCOPY data/ /app/data/\n"


class TestLabelsInWorkspace(Base):
    rid = "answers-or-grader-readable"

    def test_test_split_with_label_column(self):
        f = self.hit(harbor(LABELS_DOCKER, environment__data__test_DOT_csv="id,f1,f2,label\n1,0.2,3,1\n",
                            environment__data__train_DOT_csv="id,f1,f2,label\n2,0.1,1,0\n",
                            tests__labels_DOT_csv="id,label\n1,1\n"))
        self.assertIn("label column", f.message)
        self.assertIn("environment/Dockerfile:3", f.evidence)

    def test_instruction_names_target(self):
        self.hit(harbor(LABELS_DOCKER, environment__data__eval_DOT_csv="id,sqft,price\n1,900,300000\n",
                        instruction_DOT_md="Predict the `price` of each house in /app/data/eval.csv.\n",
                        tests__test_outputs_DOT_py="def test_rmse():\n    y = df['price']\n"))

    def test_jsonl_validation_split(self):
        self.hit(harbor(LABELS_DOCKER, environment__data__val_DOT_jsonl='{"id": 1, "text": "hi", "label": "pos"}\n',
                        tests__gold_DOT_csv="id,label\n1,pos\n"))

    def test_test_split_without_labels(self):
        self.miss(harbor(LABELS_DOCKER, environment__data__test_DOT_csv="id,f1,f2\n1,0.2,3\n",
                         environment__data__train_DOT_csv="id,f1,f2,label\n2,0.1,1,0\n", tests__labels_DOT_csv="id,label\n1,1\n"))

    def test_train_split_only(self):
        self.miss(harbor(LABELS_DOCKER, environment__data__train_DOT_csv="id,f1,label\n2,0.1,0\n",
                         tests__labels_DOT_csv="id,label\n1,1\n"))

    def test_removed_before_agent(self):
        self.miss(harbor(LABELS_DOCKER + "RUN rm -f /app/data/test.csv\n",
                         environment__data__test_DOT_csv="id,f1,label\n1,0.2,1\n", tests__labels_DOT_csv="id,label\n1,1\n"))


# ---------------------------------------------------------------------------

class TestRegistrationAndRobustness(unittest.TestCase):
    IDS = ["git-history-or-target-in-image", "answers-or-grader-readable", "writable-reward-channel",
           "verify-time-fetch", "egress-and-shared-state", "docker-privilege-or-socket"]

    def test_registered(self):
        for rid in self.IDS:
            meta = RULES[rid]
            self.assertEqual(meta["kind"], "hack")
            self.assertNotIn(";", meta["description"])
            self.assertNotIn("—", meta["description"])

    def test_odd_input_does_not_crash(self):
        files = {
            "task.toml": "[[[ not toml\nallow_internet = maybe\n",
            "environment/Dockerfile": "RUN <<EOF\ngit clone\nCOPY\nADD --chmod= \nRUN \\\n",
            "tests/test.sh": "if [ ; then cp \"unterminated\n<<'X'\ncurl | sh\n",
            "Dockerfile": "COPY [\"a\", \nRUN git clone 'unbalanced\nCOPY --from=x /a /b\nUSER\n",
            "env.py": "def f(:\n",
            "ok.py": "async def compute_reward(s):\n    await s.execute_command(s.id, f'{x}' + y)\n",
            "empty/task.toml": "",
            "empty/tests/test.sh": "",
        }
        pkg = make_pkg(files)
        for rid in self.IDS:
            RULES[rid]["fn"](pkg)  # must not raise

    def test_plain_python_env_is_clean(self):
        pkg = make_pkg({"env.py": """\
            import verifiers as vf
            def reward(completion, answer):
                return float(completion.strip() == answer)
            def load_environment():
                return vf.SingleTurnEnv(rubric=vf.Rubric(funcs=[reward]))
            """})
        for rid in self.IDS:
            self.assertEqual(RULES[rid]["fn"](pkg), [], rid)


if __name__ == "__main__":
    unittest.main()

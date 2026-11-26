"""Integration cases added when the v0.2 rule families were merged."""
import os
import shutil
import tempfile
import unittest

from envscan.core import scan_package
from envscan.package import Package

from .cli_helpers import run

TEST_SH = "#!/bin/bash\nset -e\npytest /tests/test_outputs.py\necho 1 > /logs/verifier/reward.txt\n"
SHARED = '[task]\nname = "x/y"\n\n[verifier]\ntimeout_sec = 60.0\n\n[agent]\ntimeout_sec = 60.0\n'
SEP_EXPLICIT = SHARED.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
SEP_IMPLICIT = SHARED + '\n[verifier.environment]\ndocker_image = "python:3.12"\n'
MIXED = (SHARED + '\n[[steps]]\nname = "a"\n[steps.verifier]\nenvironment_mode = "shared"\n'
         '\n[[steps]]\nname = "b"\n[steps.verifier]\nenvironment_mode = "separate"\n')
HUB_ENV = ("import verifiers as vf\nfrom verifiers.envs.experimental.harbor import HarborEnv\n\n"
           "def load_environment():\n    return HarborEnv(tasks='tasks')\n")


class MergeCase(unittest.TestCase):
    def pkg(self, files):
        d = tempfile.mkdtemp(prefix="envscan-merge-")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(text)
        return d

    def shared(self, files):
        res = scan_package(Package.from_dir(self.pkg(files)), "t")
        return [f for f in res["findings"] if f["rule"] == "harbor-shared-grading"]

    def test_plain_harbor_task_is_review(self):
        fs = self.shared({"task.toml": SHARED, "tests/test.sh": TEST_SH})
        self.assertEqual([f["confidence"] for f in fs], ["review"])
        self.assertIn("Harbor's default", fs[0]["message"])

    def test_explicit_separate_verifier_is_not_flagged(self):
        self.assertEqual(self.shared({"task.toml": SEP_EXPLICIT, "tests/test.sh": TEST_SH}), [])

    def test_verifier_environment_table_means_separate(self):
        self.assertEqual(self.shared({"task.toml": SEP_IMPLICIT, "tests/test.sh": TEST_SH}), [])

    def test_one_shared_step_is_still_flagged(self):
        fs = self.shared({"task.toml": MIXED, "tests/test.sh": TEST_SH})
        self.assertEqual(len(fs), 1)

    def test_hub_env_using_harbor_stays_likely(self):
        fs = self.shared({"env.py": HUB_ENV, "tasks/a/task.toml": SHARED, "tasks/a/tests/test.sh": TEST_SH})
        self.assertEqual([f["confidence"] for f in fs], ["likely"])

    def test_named_dockerfiles_are_loaded_as_text(self):
        d = self.pkg({"task.toml": SHARED, "environment/Dockerfile.game": "FROM python:3.12\n",
                      "environment/Containerfile.dev": "FROM python:3.12\n"})
        files = Package.from_dir(d).files
        self.assertIn("environment/Dockerfile.game", files)
        self.assertIn("environment/Containerfile.dev", files)

    def test_frozen_lake_is_not_a_lean_env(self):
        src = ('"""Frozen Lake env: reach the goal without falling in a hole."""\nimport subprocess\n\n'
               "def compile_map(grid):\n    return subprocess.run(['true'])\n")
        res = scan_package(Package.from_dir(self.pkg({"frozen_lake.py": src})), "t")
        self.assertNotIn("lean-no-axiom-check", [f["rule"] for f in res["findings"]])

    def test_lean_env_is_still_flagged(self):
        src = ("import subprocess\n\ndef check_proof(path):\n"
               "    # theorem proving reward\n    r = subprocess.run(['lake', 'env', 'lean', path])\n"
               "    return 1.0 if r.returncode == 0 else 0.0\n")
        res = scan_package(Package.from_dir(self.pkg({"lean_env.py": src})), "t")
        self.assertIn("lean-no-axiom-check", [f["rule"] for f in res["findings"]])

    def test_dataset_note_needs_two_shared_tasks(self):
        d = self.pkg({"a/task.toml": SHARED, "a/tests/test.sh": TEST_SH,
                      "b/task.toml": SEP_EXPLICIT, "b/tests/test.sh": TEST_SH})
        _, out, _ = run(d)
        self.assertNotIn("dataset:", out)
        self.assertEqual(out.strip().splitlines()[-1], "2 tasks: 0 likely, 1 review, 1 clean")


if __name__ == "__main__":
    unittest.main()

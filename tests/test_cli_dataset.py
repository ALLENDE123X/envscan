import json
import os
import unittest

from envscan.cli import find_tasks
from tests.cli_helpers import FIX, HARBOR_TASKS, copy_fixture, run, tmpdir, write


class DatasetTests(unittest.TestCase):
    def test_find_tasks(self):
        self.assertEqual(find_tasks(os.path.join(FIX, "dataset")), ["bad-a", "bad-b", "good"])

    def test_single_task_folder_is_not_a_dataset(self):
        self.assertEqual(find_tasks(os.path.join(FIX, "env_a")), [])
        self.assertEqual(find_tasks(os.path.join(FIX, "dataset", "bad-a")), [])

    def test_one_task_below_is_not_a_dataset(self):
        d = tmpdir(self)
        write(os.path.join(d, "only", "task.toml"), "x = 1\n")
        self.assertEqual(find_tasks(d), [])

    def test_folder_with_own_task_toml_is_a_task(self):
        d = tmpdir(self)
        write(os.path.join(d, "task.toml"), "x = 1\n")
        write(os.path.join(d, "a", "task.toml"), "x = 1\n")
        write(os.path.join(d, "b", "task.toml"), "x = 1\n")
        self.assertEqual(find_tasks(d), [])

    def test_text_output_one_line_per_task_and_summary(self):
        code, out, _ = run(os.path.join(FIX, "dataset"))
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "bad-a: LIKELY (high)  harbor-shared-grading, verify-time-fetch")
        self.assertEqual(lines[2], "good: CLEAN")
        # the shared verifier is Harbor's default, so it is said once for the dataset
        self.assertTrue(lines[3].startswith("dataset: 2 of 3 tasks grade in the agent's container"))
        self.assertEqual(lines[-1], "3 tasks: 2 likely, 0 review, 1 clean")
        self.assertEqual(code, 2)

    def test_json_is_a_list_with_a_result_per_task(self):
        code, out, _ = run(os.path.join(FIX, "dataset"), "--json")
        res = json.loads(out)
        self.assertIsInstance(res, list)
        self.assertEqual([r["env"] for r in res], ["bad-a", "bad-b", "good"])
        self.assertEqual([r["verdict"] for r in res], ["likely", "likely", "clean"])

    def test_tasks_are_scanned_independently(self):
        # the shared grading finding names only that task's own files
        _, out, _ = run(os.path.join(FIX, "dataset"), "--json")
        for r in json.loads(out):
            for f in r["findings"]:
                for e in f["evidence"]:
                    self.assertFalse(e.startswith(("bad-a", "bad-b", "good")))

    def test_exit_code_is_worst_task(self):
        d = tmpdir(self)
        copy_fixture("dataset", d)
        ds = os.path.join(d, "dataset")
        import shutil
        shutil.rmtree(os.path.join(ds, "bad-a"))
        shutil.rmtree(os.path.join(ds, "bad-b"))
        with open(os.path.join(FIX, "dataset", "good", "task.toml")) as f:
            write(os.path.join(ds, "good2", "task.toml"), f.read())
        write(os.path.join(ds, "good2", "tests", "test.sh"), "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
        code, out, _ = run(ds)
        self.assertEqual(code, 0)
        self.assertTrue(out.strip().endswith("2 tasks: 0 likely, 0 review, 2 clean"))

    def test_fail_on_applies_to_worst_task(self):
        self.assertEqual(run(os.path.join(FIX, "dataset"), "--fail-on", "likely")[0], 1)

    def test_verbose_expands_findings(self):
        _, out, _ = run(os.path.join(FIX, "dataset"), "-v")
        self.assertIn("harbor-shared-grading  (hack)", out)
        self.assertIn("3 tasks:", out)

    @unittest.skipUnless(os.path.isdir(HARBOR_TASKS), "harbor examples not checked out next to the scanner")
    def test_harbor_examples(self):
        names = find_tasks(HARBOR_TASKS)
        self.assertGreaterEqual(len(names), 30)
        code, out, _ = run(HARBOR_TASKS, "--json")
        res = json.loads(out)
        self.assertEqual(len(res), len(names))
        self.assertIn(code, (1, 2))
        by = {r["env"]: r["verdict"] for r in res}
        self.assertEqual(by["hello-world"], "likely")
        self.assertEqual(by["separate-verifier-environment"], "clean")
        code, out, _ = run(HARBOR_TASKS)
        self.assertRegex(out.strip().splitlines()[-1], r"^\d+ tasks: \d+ likely, \d+ review, \d+ clean$")


if __name__ == "__main__":
    unittest.main()


class DatasetWideNotes(unittest.TestCase):
    """A rule that fires on nearly every task is reported once, and tasks with findings of their own are named."""

    def _res(self, name, rules):
        return {"env": name, "verdict": "review", "top_severity": "high",
                "findings": [{"rule": r, "kind": "hack"} for r in rules]}

    def test_template_rule_and_own_findings(self):
        from envscan.cli import dataset_notes
        rs = [self._res(f"t{i}", ["harbor-shared-grading", "verify-time-fetch"]) for i in range(9)]
        rs.append(self._res("t9", ["harbor-shared-grading", "verify-time-fetch", "verdict-parse-semantics"]))
        notes = dataset_notes(rs)
        self.assertEqual(len(notes), 3)
        self.assertTrue(notes[0].startswith("dataset: 10 of 10 tasks grade in the agent's container"))
        self.assertTrue(notes[1].startswith("dataset: 10 of 10 tasks share verify-time-fetch"))
        self.assertEqual(notes[2], "dataset: 1 of 10 tasks have findings of their own beyond the dataset-wide ones: t9")

    def test_small_sets_keep_the_old_output(self):
        from envscan.cli import dataset_notes
        rs = [self._res(f"t{i}", ["harbor-shared-grading", "verify-time-fetch"]) for i in range(3)]
        self.assertEqual(len(dataset_notes(rs)), 1)

    def test_hygiene_is_never_dataset_wide(self):
        from envscan.cli import dataset_notes
        rs = [{"env": f"t{i}", "verdict": "clean", "top_severity": None,
               "findings": [{"rule": "judge-forgeable-delimiter", "kind": "hygiene"}]} for i in range(6)]
        self.assertEqual(dataset_notes(rs), [])

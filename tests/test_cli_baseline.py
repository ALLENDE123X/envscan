import json
import os
import unittest

from tests.cli_helpers import FIX, copy_fixture, run, tmpdir, write


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.d = tmpdir(self)
        self.env = copy_fixture("env_eval", self.d)
        self.base = os.path.join(self.d, "base.json")

    def scan(self, *args):
        code, out, err = run(self.env, "--json", *args)
        return code, json.loads(out)

    def test_write_baseline_exits_zero_and_saves_findings(self):
        code, out, err = run(self.env, "--write-baseline", self.base)
        self.assertEqual(code, 0)
        with open(self.base) as f:
            data = json.load(f)
        self.assertEqual([(e["rule"], e["file"]) for e in data["findings"]], [("eval-model-text", "env.py")])
        self.assertEqual(len(data["findings"][0]["hash"]), 16)
        self.assertNotIn("line", data["findings"][0])

    def test_baselined_findings_are_not_reported(self):
        run(self.env, "--write-baseline", self.base)
        code, res = self.scan("--baseline", self.base)
        self.assertEqual((code, res["verdict"], res["baselined"]), (0, "clean", 1))
        self.assertEqual([f for f in res["findings"] if f["kind"] != "hygiene"], [])

    def test_edit_above_finding_does_not_resurface_it(self):
        run(self.env, "--write-baseline", self.base)
        path = os.path.join(self.env, "env.py")
        with open(path) as f:
            text = f.read()
        write(path, "# a new comment\n# and another\n\nimport os\n" + text)
        code, res = self.scan("--baseline", self.base)
        self.assertEqual(res["verdict"], "clean")

    def test_whitespace_change_on_the_line_still_matches(self):
        run(self.env, "--write-baseline", self.base)
        path = os.path.join(self.env, "env.py")
        with open(path) as f:
            text = f.read()
        write(path, text.replace("    return float(eval(text) == answer)", "        return float(eval(text)   ==  answer)  "))
        code, res = self.scan("--baseline", self.base)
        self.assertEqual(res["verdict"], "clean")

    def test_changed_evidence_line_resurfaces(self):
        run(self.env, "--write-baseline", self.base)
        path = os.path.join(self.env, "env.py")
        with open(path) as f:
            text = f.read()
        write(path, text.replace("float(eval(text) == answer)", "float(eval(text) != answer)"))
        code, res = self.scan("--baseline", self.base)
        self.assertEqual((code, res["verdict"]), (2, "likely"))

    def test_new_finding_in_another_rule_is_reported(self):
        # baseline written for a clean-ish copy, then the eval appears
        clean = os.path.join(self.d, "other")
        write(os.path.join(clean, "env.py"), "import verifiers as vf\n")
        base = os.path.join(self.d, "clean.json")
        run(clean, "--write-baseline", base)
        code, out, _ = run(self.env, "--json", "--baseline", base)
        self.assertEqual(json.loads(out)["verdict"], "likely")

    def test_hygiene_not_baselined_or_hidden(self):
        run(self.env, "--write-baseline", self.base)
        with open(self.base) as f:
            self.assertNotIn("legacy-api-unbounded-pin", f.read())
        _, res = self.scan("--baseline", self.base)
        self.assertIn("legacy-api-unbounded-pin", [f["rule"] for f in res["findings"]])

    def test_bad_baseline_file_is_a_scan_error(self):
        write(self.base, "{not json")
        code, out, err = run(self.env, "--baseline", self.base)
        self.assertEqual(code, 3)
        code, out, err = run(self.env, "--baseline", os.path.join(self.d, "missing.json"))
        self.assertEqual(code, 3)

    def test_dataset_baseline_is_per_task(self):
        ds = copy_fixture("dataset", self.d)
        # baseline only bad-a by writing it from that task alone is not the same scope,
        # so write for the whole dataset then remove bad-b's entries
        run(ds, "--write-baseline", self.base)
        with open(self.base) as f:
            data = json.load(f)
        self.assertEqual({e["scope"] for e in data["findings"]}, {"bad-a", "bad-b"})
        data["findings"] = [e for e in data["findings"] if e["scope"] == "bad-a"]
        with open(self.base, "w") as f:
            json.dump(data, f)
        code, out, _ = run(ds, "--json", "--baseline", self.base)
        by = {r["env"]: r["verdict"] for r in json.loads(out)}
        self.assertEqual(by, {"bad-a": "clean", "bad-b": "likely", "good": "clean"})
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

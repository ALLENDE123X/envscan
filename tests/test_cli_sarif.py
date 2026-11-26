import json
import os
import unittest

from envscan.core import rule_table
from tests.cli_helpers import FIX, run, tmpdir


def check_sarif_shape(tc, doc):
    """The fields GitHub code scanning requires from a SARIF 2.1.0 upload."""
    tc.assertEqual(doc["version"], "2.1.0")
    tc.assertIn("sarif", doc["$schema"])
    tc.assertEqual(len(doc["runs"]), 1)
    run_ = doc["runs"][0]
    driver = run_["tool"]["driver"]
    tc.assertEqual(driver["name"], "envscan")
    ids = [r["id"] for r in driver["rules"]]
    tc.assertEqual(len(ids), len(set(ids)))
    for r in driver["rules"]:
        tc.assertTrue(r["shortDescription"]["text"])
        tc.assertIn(r["defaultConfiguration"]["level"], ("error", "warning", "note", "none"))
        if "security-severity" in r["properties"]:
            float(r["properties"]["security-severity"])
    for res in run_["results"]:
        tc.assertIn(res["ruleId"], ids)
        tc.assertEqual(driver["rules"][res["ruleIndex"]]["id"], res["ruleId"])
        tc.assertIn(res["level"], ("error", "warning", "note", "none"))
        tc.assertTrue(res["message"]["text"])
        tc.assertGreaterEqual(len(res["locations"]), 1)
        for loc in res["locations"] + res.get("relatedLocations", []):
            pl = loc["physicalLocation"]
            uri = pl["artifactLocation"]["uri"]
            tc.assertFalse(uri.startswith("/") or ".." in uri.split("/") or "\\" in uri, uri)
            tc.assertGreaterEqual(pl["region"]["startLine"], 1)


class SarifTests(unittest.TestCase):
    def sarif(self, *args):
        d = tmpdir(self)
        p = os.path.join(d, "out.sarif")
        code, out, err = run(*args, "--sarif", p)
        with open(p) as f:
            return code, json.load(f)

    def test_shape_and_rules(self):
        code, doc = self.sarif(os.path.join(FIX, "env_eval"))
        self.assertEqual(code, 2)
        check_sarif_shape(self, doc)
        listed = [r["id"] for r in doc["runs"][0]["tool"]["driver"]["rules"]]
        self.assertEqual(listed, [r["id"] for r in rule_table()])

    def test_likely_is_error_with_line_location(self):
        _, doc = self.sarif(os.path.join(FIX, "env_eval"))
        results = doc["runs"][0]["results"]
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual((r["ruleId"], r["level"]), ("eval-model-text", "error"))
        loc = r["locations"][0]["physicalLocation"]
        self.assertEqual(loc["artifactLocation"]["uri"], "env.py")
        self.assertEqual(loc["region"]["startLine"], 5)

    def test_suppressed_findings_not_in_sarif(self):
        d = tmpdir(self)
        _, doc = self.sarif(os.path.join(FIX, "env_eval"), "--config", self._cfg(d, "eval-model-text"))
        # a suppressed finding is not in SARIF at all
        self.assertEqual(doc["runs"][0]["results"], [])

    def _cfg(self, d, rule):
        p = os.path.join(d, "c.toml")
        with open(p, "w") as f:
            f.write(f'[[ignore-rule]]\nrule = "{rule}"\nreason = "test"\n')
        return p

    def test_warning_level_for_review(self):
        from envscan.sarif import build_sarif
        res = {"findings": [{"rule": "keyword-reward", "kind": "hack", "severity": "medium", "confidence": "review",
                             "message": "m", "evidence": ["a/b.py:7", "a/c.py:9"]}]}
        doc = build_sarif(rule_table(), [(res, "", None)])
        r = doc["runs"][0]["results"][0]
        self.assertEqual(r["level"], "warning")
        self.assertEqual(r["relatedLocations"][0]["physicalLocation"]["artifactLocation"]["uri"], "a/c.py")
        check_sarif_shape(self, doc)

    def test_hygiene_not_reported(self):
        _, doc = self.sarif(os.path.join(FIX, "env_eval"))
        self.assertNotIn("legacy-api-unbounded-pin", [r["ruleId"] for r in doc["runs"][0]["results"]])

    def test_finding_without_evidence_gets_a_location(self):
        from envscan.sarif import build_sarif
        res = {"findings": [{"rule": "no-load-environment", "kind": "broken", "severity": "high", "confidence": "likely",
                             "message": "m", "evidence": []}]}
        doc = build_sarif(rule_table(), [(res, "", "pyproject.toml")])
        self.assertEqual(doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"], "pyproject.toml")

    def test_clean_scan_writes_empty_results(self):
        code, doc = self.sarif(os.path.join(FIX, "dataset", "good"))
        self.assertEqual(code, 0)
        self.assertEqual(doc["runs"][0]["results"], [])
        check_sarif_shape(self, doc)

    def test_sarif_base_makes_paths_repo_relative(self):
        d = tmpdir(self)
        import shutil
        shutil.copytree(os.path.join(FIX, "env_eval"), os.path.join(d, "environments", "env_eval"))
        p = os.path.join(d, "o.sarif")
        run(os.path.join(d, "environments", "env_eval"), "--sarif", p, "--sarif-base", d)
        with open(p) as f:
            doc = json.load(f)
        self.assertEqual(doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"],
                         "environments/env_eval/env.py")

    def test_dataset_paths_include_task_folder(self):
        _, doc = self.sarif(os.path.join(FIX, "dataset"))
        uris = {r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] for r in doc["runs"][0]["results"]}
        self.assertEqual(uris, {"bad-a/task.toml", "bad-b/task.toml", "bad-a/tests/test.sh", "bad-b/tests/test.sh"})
        check_sarif_shape(self, doc)


if __name__ == "__main__":
    unittest.main()

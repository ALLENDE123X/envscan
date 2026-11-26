import json
import os
import unittest

from envscan.config import ConfigError, glob_to_regex, parse_config
from tests.cli_helpers import FIX, copy_fixture, run, tmpdir, write


class ConfigFileTests(unittest.TestCase):
    def env(self, cfg_text=None, name="env_eval"):
        d = tmpdir(self)
        p = copy_fixture(name, d)
        if cfg_text is not None:
            write(os.path.join(p, ".envscan.toml"), cfg_text)
        return p

    def scan_json(self, *args):
        code, out, err = run(*args, "--json")
        return code, json.loads(out), err

    def test_no_config_unchanged(self):
        code, res, _ = self.scan_json(os.path.join(FIX, "env_eval"))
        self.assertEqual((code, res["verdict"]), (2, "likely"))
        self.assertEqual(res["suppressed"], [])

    def test_ignore_rule_found_at_scan_root(self):
        p = self.env('[[ignore-rule]]\nrule = "eval-model-text"\nreason = "sandboxed in our harness"\n')
        code, res, _ = self.scan_json(p)
        self.assertEqual((code, res["verdict"]), (0, "clean"))
        self.assertEqual([f["rule"] for f in res["findings"] if f["kind"] != "hygiene"], [])
        s = res["suppressed"]
        self.assertEqual(len(s), 1)
        self.assertEqual((s[0]["rule"], s[0]["suppressed_by"], s[0]["reason"]), ("eval-model-text", "ignore-rule", "sandboxed in our harness"))

    def test_ignore_path_glob(self):
        p = self.env('[[ignore-path]]\nglob = "*.py"\nreason = "generated"\n')
        code, res, _ = self.scan_json(p)
        self.assertEqual(res["verdict"], "clean")
        self.assertEqual(res["suppressed"][0]["suppressed_by"], "ignore-path")

    def test_ignore_path_non_matching_keeps_finding(self):
        p = self.env('[[ignore-path]]\nglob = "tests/**"\nreason = "x"\n')
        code, res, _ = self.scan_json(p)
        self.assertEqual(res["verdict"], "likely")
        self.assertEqual(res["suppressed"], [])

    def test_suppress_single_finding_by_rule_and_path(self):
        p = self.env('[[suppress]]\nrule = "eval-model-text"\npath = "env.py"\nreason = "reviewed 2026-11"\n')
        code, res, _ = self.scan_json(p)
        self.assertEqual(res["verdict"], "clean")
        self.assertEqual(res["suppressed"][0]["reason"], "reviewed 2026-11")

    def test_suppress_wrong_rule_or_path_does_not_apply(self):
        for rule, path in (("keyword-reward", "env.py"), ("eval-model-text", "other.py")):
            p = self.env(f'[[suppress]]\nrule = "{rule}"\npath = "{path}"\nreason = "x"\n')
            code, res, _ = self.scan_json(p)
            self.assertEqual(res["verdict"], "likely", (rule, path))

    def test_explicit_config_flag(self):
        d = tmpdir(self)
        cfg = os.path.join(d, "my.toml")
        write(cfg, '[[ignore-rule]]\nrule = "eval-model-text"\nreason = "r"\n')
        code, res, _ = self.scan_json(os.path.join(FIX, "env_eval"), "--config", cfg)
        self.assertEqual(res["verdict"], "clean")

    def test_config_inside_archive_is_not_trusted(self):
        import io
        import tarfile
        d = tmpdir(self)
        tgz = os.path.join(d, "e.tar.gz")
        with tarfile.open(tgz, "w:gz") as t:
            for name, path in (("env.py", os.path.join(FIX, "env_eval", "env.py")),):
                t.add(path, arcname=name)
            data = b'[[ignore-rule]]\nrule = "eval-model-text"\nreason = "self serving"\n'
            ti = tarfile.TarInfo(".envscan.toml")
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
        code, res, _ = self.scan_json(tgz)
        self.assertEqual(res["verdict"], "likely")

    def test_reason_is_required(self):
        p = self.env('[[ignore-rule]]\nrule = "eval-model-text"\n')
        code, out, err = run(p)
        self.assertEqual(code, 3)
        self.assertIn("reason", err)

    def test_empty_reason_rejected(self):
        with self.assertRaises(ConfigError):
            parse_config('[[suppress]]\nrule = "a"\npath = "b"\nreason = "  "\n')

    def test_unknown_rule_unknown_key_and_bad_toml(self):
        for text in ('[[ignore-rule]]\nrule = "nope"\nreason = "r"\n',
                     '[[ignore-rule]]\nrule = "eval-model-text"\nreason = "r"\nextra = 1\n',
                     '[[ignore]]\nrule = "x"\n',
                     'this is not toml ['):
            p = self.env(text)
            code, out, err = run(p)
            self.assertEqual(code, 3, text)
            self.assertIn("envscan:", err)

    def test_partial_path_ignore_keeps_other_evidence(self):
        from envscan.config import apply_config
        cfg = parse_config('[[ignore-path]]\nglob = "tests/**"\nreason = "r"\n')
        res = {"findings": [{"rule": "keyword-reward", "kind": "hack", "severity": "medium", "confidence": "review",
                             "message": "m", "evidence": ["tests/a.py:1", "env.py:3"]}]}
        apply_config(res, cfg)
        self.assertEqual(res["findings"][0]["evidence"], ["env.py:3"])
        self.assertEqual(res["suppressed"], [])

    def test_hygiene_is_never_suppressed(self):
        from envscan.config import apply_config
        cfg = parse_config('[[ignore-path]]\nglob = "**"\nreason = "r"\n')
        res = {"findings": [{"rule": "judge-prompt-undelimited", "kind": "hygiene", "severity": "low", "confidence": "review",
                             "message": "m", "evidence": ["a.py:1"]}]}
        apply_config(res, cfg)
        self.assertEqual(len(res["findings"]), 1)

    def test_glob_semantics(self):
        rx = glob_to_regex("tests/**/*.py")
        self.assertTrue(rx.match("tests/a/b/c.py"))
        self.assertTrue(rx.match("tests/c.py"))
        self.assertFalse(rx.match("src/tests/c.py"))
        self.assertTrue(glob_to_regex("*.py").match("env.py"))
        self.assertFalse(glob_to_regex("*.py").match("pkg/env.py"))
        self.assertTrue(glob_to_regex("**/env.py").match("env.py"))
        self.assertTrue(glob_to_regex("**/env.py").match("a/b/env.py"))

    def test_dataset_config_matches_task_prefixed_paths(self):
        d = tmpdir(self)
        ds = copy_fixture("dataset", d)
        write(os.path.join(ds, ".envscan.toml"),
              '[[suppress]]\nrule = "harbor-shared-grading"\npath = "bad-a/**"\nreason = "known"\n'
              '[[suppress]]\nrule = "verify-time-fetch"\npath = "bad-a/**"\nreason = "known"\n')
        code, out, _ = run(ds, "--json")
        by = {r["env"]: r["verdict"] for r in json.loads(out)}
        self.assertEqual(by, {"bad-a": "clean", "bad-b": "likely", "good": "clean"})


if __name__ == "__main__":
    unittest.main()

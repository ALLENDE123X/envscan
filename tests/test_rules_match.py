"""Tests for the answer matching and parsing rules (envscan/rules_match.py, v0.5).

Each case writes a few small files into a temp folder and loads them with Package.from_dir.
Nothing in the fixtures is imported or run.
"""
import os
import shutil
import tempfile
import textwrap
import unittest

from envscan.package import Package
from envscan.rules import RULES, run_rules

MINE = ("exact-compare-no-normalization", "extractor-prompt-format-mismatch", "tool-call-parser-lenient",
        "empty-answer-from-truncation")
PYPROJECT = '[project]\nname = "x"\ndependencies = ["verifiers>=0.1.3"]\n'


def pkg(files: dict) -> Package:
    d = tempfile.mkdtemp(prefix="envscan-match-")
    try:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(text).lstrip("\n"))
        return Package.from_dir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def fire(rid, code, extra=None):
    files = {"pyproject.toml": PYPROJECT, "env.py": code}
    files.update(extra or {})
    return RULES[rid]["fn"](pkg(files))


class Registered(unittest.TestCase):
    def test_rules_registered(self):
        for rid, kind in zip(MINE, ("broken", "broken", "hack", "broken")):
            self.assertIn(rid, RULES)
            self.assertEqual(RULES[rid]["kind"], kind)
            desc = RULES[rid]["description"]
            self.assertNotIn(";", desc)
            self.assertNotIn("—", desc)

    def test_odd_input_does_not_crash(self):
        p = pkg({"pyproject.toml": PYPROJECT, "broken.py": "def f(:\n  return 1\n", "empty.py": "",
                 "data.jsonl": "{not json\n", "odd.py": "x = vf.Rubric(funcs=[a, lambda s: 1], weights=[w])\n"
                 "sampling_args = {'stop': 3, 'max_tokens': None}\ndef a(completion):\n  return completion == answer\n"})
        findings, errors = run_rules(p, only=set(MINE))
        self.assertEqual(errors, [])
        self.assertEqual(findings, [])

    def test_empty_package_is_fast_and_silent(self):
        p = pkg({"README.md": "nothing here\n"})
        findings, errors = run_rules(p, only=set(MINE))
        self.assertEqual((findings, errors), ([], []))


# ---------------------------------------------------------------------------
# exact-compare-no-normalization
# ---------------------------------------------------------------------------

GSM_EXACT = '''
import verifiers as vf
from datasets import load_dataset
def correct(parser, completion, answer, **kw):
    pred = parser.parse_answer(completion)
    return 1.0 if pred == answer else 0.0
def load_environment():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    parser = vf.XMLParser(["think", "answer"])
    return vf.SingleTurnEnv(dataset=ds, parser=parser, rubric=vf.Rubric(funcs=[correct], weights=[1.0]))
'''


class ExactCompare(unittest.TestCase):
    rid = "exact-compare-no-normalization"

    def test_gsm8k_raw_equality(self):
        f = fire(self.rid, GSM_EXACT)
        self.assertEqual(len(f), 1)
        self.assertIn("gsm8k", f[0].message)
        self.assertEqual(f[0].confidence, "review")

    def test_boxed_extractor_if_branch(self):
        code = '''
import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer
def correct(completion, answer, **kw):
    pred = extract_boxed_answer(completion[-1]["content"])
    if pred == answer:
        return 1.0
    return 0.0
rubric = vf.Rubric(funcs=[correct], weights=[1.0])
'''
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_numeric_rows_shipped(self):
        code = GSM_EXACT.replace('load_dataset("openai/gsm8k", "main", split="train")',
                                 'load_dataset("json", data_files="data.jsonl")')
        rows = "".join('{"question": "q%d", "answer": "%d"}\n' % (i, i * 7) for i in range(10))
        self.assertEqual(len(fire(self.rid, code, {"data.jsonl": rows})), 1)

    def test_stripped_and_parsed(self):
        code = GSM_EXACT.replace("return 1.0 if pred == answer else 0.0",
                                 "return 1.0 if float(pred.strip()) == float(answer) else 0.0")
        self.assertEqual(fire(self.rid, code), [])

    def test_normalize_helper(self):
        code = GSM_EXACT.replace("pred = parser.parse_answer(completion)",
                                 "pred = norm(parser.parse_answer(completion))") + \
            "\ndef norm(s):\n    return s.replace(',', '').strip()\n"
        self.assertEqual(fire(self.rid, code), [])

    def test_label_answers_not_numeric(self):
        code = GSM_EXACT.replace('load_dataset("openai/gsm8k", "main", split="train")',
                                 'load_dataset("json", data_files="data.jsonl")')
        rows = "".join('{"question": "q%d", "answer": "%s"}\n' % (i, ["cat", "dog"][i % 2]) for i in range(10))
        self.assertEqual(fire(self.rid, code, {"data.jsonl": rows}), [])

    def test_multiple_choice_letters(self):
        code = GSM_EXACT.replace('"openai/gsm8k", "main"', '"Idavidrein/gpqa", "gpqa_main"')
        code = code.replace("def correct(", "# multiple choice letter answers\ndef correct(")
        self.assertEqual(fire(self.rid, code), [])

    def test_unpaid(self):
        self.assertEqual(fire(self.rid, GSM_EXACT.replace("weights=[1.0]", "weights=[0.0]")), [])


# ---------------------------------------------------------------------------
# extractor-prompt-format-mismatch
# ---------------------------------------------------------------------------

BOXED_NO_ASK = '''
import re, verifiers as vf
from math_verify import verify, parse
SYSTEM_PROMPT = "You are a careful math tutor. Think step by step and explain your work."
def correct(completion, answer, **kw):
    m = re.search(r"\\\\boxed\\{(.*?)\\}", completion[-1]["content"])
    return 1.0 if m and verify(parse(answer), parse(m.group(1))) else 0.0
def load_environment():
    return vf.SingleTurnEnv(dataset=ds, system_prompt=SYSTEM_PROMPT, rubric=vf.Rubric(funcs=[correct], weights=[1.0]))
'''


class ExtractorMismatch(unittest.TestCase):
    rid = "extractor-prompt-format-mismatch"

    def test_boxed_never_asked(self):
        f = fire(self.rid, BOXED_NO_ASK)
        self.assertEqual(len(f), 1)
        self.assertIn("boxed", f[0].message)

    def test_answer_tag_never_asked(self):
        code = '''
import re, verifiers as vf
SYSTEM_PROMPT = "Solve the puzzle and give the final result."
def correct(completion, answer, **kw):
    m = re.search(r"<answer>(.*?)</answer>", completion[-1]["content"])
    return 1.0 if m and m.group(1).strip() == answer else 0.0
rubric = vf.Rubric(funcs=[correct], weights=[1.0])
'''
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_hash_marker_never_asked(self):
        code = '''
import verifiers as vf
SYSTEM_PROMPT = "Solve the word problem. Show your reasoning."
def correct(completion, answer, **kw):
    pred = completion[-1]["content"].split("####")[-1].strip()
    return 1.0 if pred == answer else 0.0
rubric = vf.Rubric(funcs=[correct], weights=[1.0])
'''
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_prompt_asks_for_boxed(self):
        code = BOXED_NO_ASK.replace("explain your work.", "put your final answer within \\\\boxed{}.")
        self.assertEqual(fire(self.rid, code), [])

    def test_library_boxed_prompt(self):
        code = BOXED_NO_ASK.replace("from math_verify", "from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT\n"
                                    "from math_verify").replace("system_prompt=SYSTEM_PROMPT", "system_prompt=BOXED_SYSTEM_PROMPT")
        self.assertEqual(fire(self.rid, code), [])

    def test_no_prompt_written_by_package(self):
        code = BOXED_NO_ASK.replace('SYSTEM_PROMPT = "You are a careful math tutor. Think step by step and explain your '
                                    'work."\n', "").replace("system_prompt=SYSTEM_PROMPT, ", "")
        self.assertEqual(fire(self.rid, code), [])

    def test_data_rows_ask_for_marker(self):
        rows = '{"question": "What is 2+2? Put the final answer after ####.", "answer": "4"}\n'
        code = '''
import verifiers as vf
SYSTEM_PROMPT = "Solve the word problem."
def correct(completion, answer, **kw):
    pred = completion[-1]["content"].split("####")[-1].strip()
    return 1.0 if pred == answer else 0.0
rubric = vf.Rubric(funcs=[correct], weights=[1.0])
'''
        self.assertEqual(fire(self.rid, code, {"data.jsonl": rows}), [])

    def test_library_boxed_extractor_is_skipped(self):
        # verifiers' extract_boxed_answer is library code this scan cannot read, so its fallback is unknown
        code = BOXED_NO_ASK.replace("import re, verifiers as vf", "import verifiers as vf\nfrom verifiers.utils.data_utils "
                                    "import extract_boxed_answer").replace(
            'm = re.search(r"\\\\boxed\\{(.*?)\\}", completion[-1]["content"])', 'm = extract_boxed_answer(completion[-1]["content"])')
        code = code.replace("parse(m.group(1))", "parse(m)")
        self.assertEqual(fire(self.rid, code), [])

    def test_extractor_with_fallback(self):
        code = '''
import re, verifiers as vf
SYSTEM_PROMPT = "Solve the puzzle and give the final result."
def extract(text):
    m = re.search(r"<answer>(.*?)</answer>", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\\d+", text)
    return nums[-1] if nums else ""
def correct(completion, answer, **kw):
    return 1.0 if extract(completion[-1]["content"]) == answer else 0.0
rubric = vf.Rubric(funcs=[correct], weights=[1.0])
'''
        self.assertEqual(fire(self.rid, code), [])

    def test_xml_parser_format_string(self):
        code = '''
import verifiers as vf
parser = vf.XMLParser(["think", "answer"])
SYSTEM_PROMPT = f"Answer in this format: {parser.get_format_str()}"
def correct(completion, answer, **kw):
    return 1.0 if parser.parse_answer(completion) == answer else 0.0
rubric = vf.Rubric(funcs=[correct], weights=[1.0])
'''
        self.assertEqual(fire(self.rid, code), [])


# ---------------------------------------------------------------------------
# tool-call-parser-lenient
# ---------------------------------------------------------------------------

TOOL_PARSE = '''
import re, json, verifiers as vf
def parse_call(text):
    m = re.search(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S)
    return json.loads(m.group(1)) if m else None
def call_reward(completion, answer, **kw):
    call = parse_call(completion[-1]["content"])
    return 1.0 if call and call.get("name") == answer else 0.0
rubric = vf.Rubric(funcs=[call_reward], weights=[1.0])
'''


class ToolCallLenient(unittest.TestCase):
    rid = "tool-call-parser-lenient"

    def test_optional_closing_tag(self):
        f = fire(self.rid, TOOL_PARSE)
        self.assertEqual(len(f), 1)
        self.assertIn("closing tag", f[0].message)

    def test_no_closing_tag_at_all(self):
        code = TOOL_PARSE.replace(r'r"<tool_call>(.*?)(?:</tool_call>|$)"', r'r"<tool_call>\s*(.*)"')
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_json_repair(self):
        code = TOOL_PARSE.replace("(?:</tool_call>|$)", "</tool_call>").replace(
            "json.loads(m.group(1))", "json_repair.loads(m.group(1))").replace("import re, json,", "import re, json, json_repair,")
        f = fire(self.rid, code)
        self.assertEqual(len(f), 1)
        self.assertIn("repairs", f[0].message)

    def test_strict_parser(self):
        self.assertEqual(fire(self.rid, TOOL_PARSE.replace("(?:</tool_call>|$)", "</tool_call>")), [])

    def test_parser_not_on_reward_path(self):
        code = TOOL_PARSE.replace("call = parse_call(completion[-1][\"content\"])",
                                  "call = json.loads(completion[-1][\"content\"])")
        self.assertEqual(fire(self.rid, code), [])

    def test_unpaid(self):
        self.assertEqual(fire(self.rid, TOOL_PARSE.replace("weights=[1.0]", "weights=[0.0]")), [])


# ---------------------------------------------------------------------------
# empty-answer-from-truncation
# ---------------------------------------------------------------------------

STOP_CLOSE = '''
import re, verifiers as vf
SYSTEM_PROMPT = "Think, then give the answer inside <answer></answer> tags."
def correct(completion, answer, **kw):
    m = re.search(r"<answer>(.*?)</answer>", completion[-1]["content"])
    return 1.0 if m and m.group(1).strip() == answer else 0.0
def load_environment():
    return vf.SingleTurnEnv(dataset=ds, system_prompt=SYSTEM_PROMPT, sampling_args={"stop": ["</answer>"]},
                            rubric=vf.Rubric(funcs=[correct], weights=[1.0]))
'''


class Truncation(unittest.TestCase):
    rid = "empty-answer-from-truncation"

    def test_stop_on_needed_closing_tag(self):
        f = fire(self.rid, STOP_CLOSE)
        self.assertEqual(len(f), 1)
        self.assertIn("</answer>", f[0].message)

    def test_stop_on_think_close(self):
        code = STOP_CLOSE.replace('{"stop": ["</answer>"]}', '{"stop": ["</think>"]}')
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_tiny_max_tokens_with_reasoning_prompt(self):
        code = STOP_CLOSE.replace('{"stop": ["</answer>"]}', '{"max_tokens": 256}').replace(
            "Think, then", "Think step by step, then")
        self.assertEqual(len(fire(self.rid, code)), 1)

    def test_stop_on_unrelated_string(self):
        self.assertEqual(fire(self.rid, STOP_CLOSE.replace('["</answer>"]', '["<|im_end|>"]')), [])

    def test_big_max_tokens(self):
        code = STOP_CLOSE.replace('{"stop": ["</answer>"]}', '{"max_tokens": 4096}').replace(
            "Think, then", "Think step by step, then")
        self.assertEqual(fire(self.rid, code), [])

    def test_small_max_tokens_no_reasoning_asked(self):
        code = STOP_CLOSE.replace('{"stop": ["</answer>"]}', '{"max_tokens": 64}').replace(
            "Think, then give", "Give only")
        self.assertEqual(fire(self.rid, code), [])

    def test_unpaid(self):
        self.assertEqual(fire(self.rid, STOP_CLOSE.replace("weights=[1.0]", "weights=[0.0]")), [])



class ExactCompareLabels(unittest.TestCase):
    """0.5.0 fix wave (check_3.md): skip a small fixed set of gold labels such as 0/1 or 1/2/3."""
    rid = "exact-compare-no-normalization"
    BOXED = '''
import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer
def correct(completion, answer, **kw):
    pred = extract_boxed_answer(completion[-1]["content"])
    GUARD
    return 1.0 if pred == answer else 0.0
def load_environment():
    ds = load_dataset("json", data_files="data.jsonl")["train"]
    ds = ds.map(lambda x: MAP)
    return vf.SingleTurnEnv(dataset=ds, rubric=vf.Rubric(funcs=[correct], weights=[1.0]))
'''

    def code(self, guard="pass", mapping='{"question": x["q"], "answer": x["a"]}'):
        return self.BOXED.replace("GUARD", guard).replace("MAP", mapping)

    def rows(self, vals):
        return "".join('{"q": "question %d", "answer": "%s"}\n' % (i, v) for i, v in enumerate(vals))

    def test_neg_membership_check(self):
        code = self.code('if pred not in ["0", "1"]:\n        return 0.0')
        self.assertEqual(fire(self.rid, code), [])

    def test_neg_bucket_label_gold(self):
        self.assertEqual(fire(self.rid, self.code(mapping='{"question": x["q"], "answer": str(x["bucket"])}')), [])

    def test_neg_few_distinct_shipped_values(self):
        self.assertEqual(fire(self.rid, self.code(), {"data.jsonl": self.rows([1, 2, 3] * 10)}), [])

    def test_pos_boxed_numbers(self):
        self.assertEqual(len(fire(self.rid, self.code())), 1)

    def test_pos_many_distinct_shipped_values(self):
        self.assertEqual(len(fire(self.rid, self.code(), {"data.jsonl": self.rows([i * 7 for i in range(30)])})), 1)

    def test_pos_long_membership_list(self):
        code = self.code('if pred not in [str(i) for i in range(1000)]:\n        return 0.0')
        self.assertEqual(len(fire(self.rid, code)), 1)


if __name__ == "__main__":
    unittest.main()

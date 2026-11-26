"""v0.5 core builder tests: new rules and widened branches in envscan/rules.py."""
import json
import os
import tempfile
import textwrap
import unittest

from envscan.package import Package
from envscan import rules as R


def make_pkg(files: dict) -> Package:
    with tempfile.TemporaryDirectory() as d:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(textwrap.dedent(text))
        return Package.from_dir(d)


def run(rid: str, files: dict):
    return R.RULES[rid]["fn"](make_pkg(files))


PATCH_SOLVE = """\
#!/bin/bash
cat > /tmp/fix.patch <<'EOF'
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,3 +1,4 @@
-    total = price * qty
+    total = price * qty * (1 - discount_rate)
+    if total < minimum_charge_value: total = minimum_charge_value
+    return round(total, ndigits=2)
EOF
git apply /tmp/fix.patch
"""
TASK_TOML = "[verifier]\ntimeout_sec = 60\n"
FENCED_FIX = """\
Fix the discount bug in app.py.

```python
    total = price * qty * (1 - discount_rate)
    if total < minimum_charge_value: total = minimum_charge_value
    return round(total, ndigits=2)
```
"""


class SolutionInTaskText(unittest.TestCase):
    rid = "solution-in-task-text"

    def test_pos_fenced_fix(self):
        fs = run(self.rid, {"task.toml": TASK_TOML, "instruction.md": FENCED_FIX, "solution/solve.sh": PATCH_SOLVE})
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].confidence, "review")

    def test_pos_fix_phrase_one_line(self):
        inst = "Totals are wrong. The fix is to change it to `total = price * qty * (1 - discount_rate)` in app.py.\n"
        fs = run(self.rid, {"task.toml": TASK_TOML, "instruction.md": inst, "solution/solve.sh": PATCH_SOLVE})
        self.assertEqual(len(fs), 1)

    def test_pos_dataset_rows(self):
        import json
        patch = PATCH_SOLVE.split("<<'EOF'\n")[1].split("EOF")[0]
        row = {"problem_statement": FENCED_FIX, "patch": patch, "instance_id": "x"}
        fs = run(self.rid, {"data/train.jsonl": json.dumps(row) + "\n", "env.py": "x = 1\n"})
        self.assertEqual(len(fs), 1)
        self.assertIn("dataset rows", fs[0].message)

    def test_neg_lines_already_in_repo(self):
        files = {"task.toml": TASK_TOML, "instruction.md": FENCED_FIX, "solution/solve.sh": PATCH_SOLVE,
                 "environment/app/notes.py": FENCED_FIX.split("```python\n")[1].split("```")[0]}
        self.assertEqual(run(self.rid, files), [])

    def test_neg_interface_lines_only(self):
        solve = "cat > /app/store.go <<'EOF'\nGet(key string) (string, error)\nPut(key string, value string) error\n" \
                "Delete(key string) error\nEOF\n"
        inst = "Implement the interface:\n```\nGet(key string) (string, error)\nPut(key string, value string) error\n" \
               "Delete(key string) error\n```\n"
        self.assertEqual(run(self.rid, {"task.toml": TASK_TOML, "instruction.md": inst, "solution/solve.sh": solve}), [])

    def test_neg_no_solution(self):
        self.assertEqual(run(self.rid, {"task.toml": TASK_TOML, "instruction.md": FENCED_FIX}), [])


class DeserializesAgentArtifact(unittest.TestCase):
    rid = "verifier-deserializes-agent-artifact"

    def harbor(self, test_py, extra=None):
        files = {"task.toml": TASK_TOML, "instruction.md": "Train a model.\n", "tests/test.sh": "pytest /tests\n",
                 "tests/test_outputs.py": test_py}
        files.update(extra or {})
        return run(self.rid, files)

    def test_pos_pickle_agent_file(self):
        fs = self.harbor("import pickle\ndef test_model():\n    m = pickle.load(open('/app/model.pkl', 'rb'))\n    assert m.score > 0.9\n")
        self.assertEqual(len(fs), 1)
        self.assertIn("pickle.load", fs[0].message)

    def test_pos_numpy_allow_pickle(self):
        fs = self.harbor("import numpy as np\nRESULT = '/app/output/result.npy'\ndef test_arr():\n"
                         "    a = np.load(RESULT, allow_pickle=True)\n    assert a.shape == (3,)\n")
        self.assertEqual(len(fs), 1)

    def test_pos_torch_weights_only_false(self):
        fs = self.harbor("import torch\ndef test_ckpt():\n    sd = torch.load('/app/model.pt', weights_only=False)\n    assert sd\n")
        self.assertEqual(len(fs), 1)

    def test_pos_sympify_model_text(self):
        env = ("import sympy\nimport verifiers as vf\n"
               "def math_reward(completion, answer, **kw):\n"
               "    expr = parser.parse_answer(completion)\n"
               "    return float(sympy.simplify(sympy.sympify(expr) - sympy.sympify(answer)) == 0)\n"
               "rubric = vf.Rubric(funcs=[math_reward], weights=[1.0])\n")
        fs = run(self.rid, {"env.py": env})
        self.assertEqual(len(fs), 1)

    def test_neg_grader_own_pickle(self):
        self.assertEqual(self.harbor("import pickle\ndef test_x():\n    exp = pickle.load(open('/tests/expected.pkl', 'rb'))\n"
                                     "    assert exp\n"), [])

    def test_neg_torch_default_safe(self):
        self.assertEqual(self.harbor("import torch\ndef test_ckpt():\n    sd = torch.load('/app/model.pt')\n    assert sd\n"), [])

    def test_neg_yaml_safe(self):
        self.assertEqual(self.harbor("import yaml\ndef test_cfg():\n    c = yaml.load(open('/app/c.yaml'), Loader=yaml.SafeLoader)\n"
                                     "    d = yaml.safe_load(open('/app/d.yaml'))\n    assert c and d\n"), [])

    def test_neg_sympify_gold_only(self):
        env = ("import sympy\nimport verifiers as vf\n"
               "def math_reward(completion, answer, **kw):\n"
               "    gold = sympy.sympify(answer)\n    return float(str(gold) == 'x')\n"
               "rubric = vf.Rubric(funcs=[math_reward], weights=[1.0])\n")
        self.assertEqual(run(self.rid, {"env.py": env}), [])

    def test_neg_staged_by_test_script(self):
        fs = self.harbor("import torch\nDATA = '/app/data/d8'\ndef test_x():\n"
                         "    v = torch.load(f'{DATA}/vocab.pt', weights_only=False)\n    assert v\n",
                         {"tests/test.sh": "openssl enc -d -in x | tar xf - -C \"${APP_DIR}/data/\"\npytest /tests\n"})
        self.assertEqual(fs, [])


class ShellInjection(unittest.TestCase):
    rid = "shell-injection-from-model-text"

    def test_pos_host_bash_tool(self):
        env = ("import subprocess\nimport verifiers as vf\n"
               "def run_cmd(command: str) -> str:\n"
               "    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout\n"
               "env = vf.ToolEnv(dataset=ds, tools=[run_cmd])\n")
        fs = run(self.rid, {"env.py": env})
        self.assertEqual(len(fs), 1)
        self.assertIn("host shell", fs[0].message)

    def test_pos_reward_os_system(self):
        env = ("import os\nimport verifiers as vf\n"
               "def check_reward(completion, **kw):\n"
               "    code = parser.parse_answer(completion)\n"
               "    return float(os.system(f\"python3 check.py '{code}'\") == 0)\n"
               "rubric = vf.Rubric(funcs=[check_reward])\n")
        self.assertEqual(len(run(self.rid, {"env.py": env})), 1)

    def test_pos_concat_tool(self):
        env = ("import subprocess\n"
               "def grep_file(pattern: str) -> str:\n"
               "    return subprocess.check_output('grep ' + pattern + ' data.txt', shell=True).decode()\n"
               "TOOLS = [grep_file]\n")
        self.assertEqual(len(run(self.rid, {"env.py": env})), 1)

    def test_neg_quoted(self):
        env = ("import subprocess, shlex\n"
               "def grep_file(pattern: str) -> str:\n"
               "    return subprocess.check_output('grep ' + shlex.quote(pattern) + ' data.txt', shell=True).decode()\n"
               "TOOLS = [grep_file]\n")
        self.assertEqual(run(self.rid, {"env.py": env}), [])

    def test_neg_list_args(self):
        env = ("import subprocess\n"
               "def check_reward(completion, **kw):\n"
               "    return float(subprocess.run(['python3', 'check.py', completion]).returncode == 0)\n"
               "rubric = vf.Rubric(funcs=[check_reward])\n")
        self.assertEqual(run(self.rid, {"env.py": env}), [])

    def test_neg_not_tool_or_reward(self):
        env = ("import os\n"
               "def setup_repo(path):\n    os.system(f'git clone {path}')\n")
        self.assertEqual(run(self.rid, {"env.py": env}), [])


FETCH_BASE = "from datasets import load_dataset\nds = load_dataset('someorg/hardbench')\nimport verifiers as vf\n"


class OpenFetchToolGaps(unittest.TestCase):
    rid = "open-fetch-tool"

    def test_pos_blocklist_without_hf(self):
        env = FETCH_BASE + ("import httpx\nBLOCKED = ['github.com', 'quizlet.com']\n"
                            "def fetch_page(url: str) -> str:\n"
                            "    if any(b in url for b in BLOCKED):\n        return 'blocked'\n"
                            "    return httpx.get(url).text\n"
                            "env = vf.ToolEnv(dataset=ds, tools=[fetch_page])\n")
        fs = run(self.rid, {"env.py": env})
        self.assertEqual(len(fs), 1)
        self.assertIn("blocklist", fs[0].message)

    def test_pos_search_no_filter(self):
        env = FETCH_BASE + ("from tavily import TavilyClient\nclient = TavilyClient()\n"
                            "def web_search(query: str) -> str:\n    return str(client.search(query))\n"
                            "env = vf.ToolEnv(dataset=ds, tools=[web_search])\n")
        fs = run(self.rid, {"env.py": env})
        self.assertEqual(len(fs), 1)
        self.assertIn("search", fs[0].message)

    def test_pos_search_excludes_other_site_and_prompt_ban(self):
        env = FETCH_BASE + ("from exa_py import Exa\nexa = Exa()\n"
                            "SYSTEM = 'Do not visit huggingface or any benchmark dataset page.'\n"
                            "def search_web(query: str):\n    return exa.search(query, exclude_domains=['wikipedia.org'])\n"
                            "env = vf.ToolEnv(dataset=ds, tools=[search_web], system_prompt=SYSTEM)\n")
        fs = run(self.rid, {"env.py": env})
        self.assertEqual(len(fs), 1)
        self.assertIn("only in the prompt", fs[0].message)

    def test_neg_blocklist_has_hf(self):
        env = FETCH_BASE + ("import httpx\nBLOCKED = ['huggingface.co', 'hf.co', 'github.com']\n"
                            "def fetch_page(url: str) -> str:\n"
                            "    if any(b in url for b in BLOCKED):\n        return 'blocked'\n"
                            "    return httpx.get(url).text\n"
                            "env = vf.ToolEnv(dataset=ds, tools=[fetch_page])\n")
        self.assertEqual(run(self.rid, {"env.py": env}), [])

    def test_neg_allowlist(self):
        env = FETCH_BASE + ("from tavily import TavilyClient\nclient = TavilyClient()\n"
                            "def web_search(query: str) -> str:\n"
                            "    return str(client.search(query, include_domains=['docs.python.org']))\n"
                            "env = vf.ToolEnv(dataset=ds, tools=[web_search])\n")
        self.assertEqual(run(self.rid, {"env.py": env}), [])

    def test_neg_search_not_a_tool(self):
        env = FETCH_BASE + ("from tavily import TavilyClient\nclient = TavilyClient()\n"
                            "def web_search(query: str) -> str:\n    return str(client.search(query))\n")
        self.assertEqual(run(self.rid, {"env.py": env}), [])


def rubric_env(funcs_src: str, names: str, weights: str) -> dict:
    return {"env.py": "import re\nimport verifiers as vf\n" + funcs_src +
            f"\nrubric = vf.Rubric(funcs=[{names}], weights=[{weights}])\n"}


CORRECT = ("def correct_answer_reward(completion, answer, **kw):\n"
           "    return 1.0 if parser.parse_answer(completion) == answer else 0.0\n")


class KeywordSimilarity(unittest.TestCase):
    rid = "keyword-reward"

    def test_pos_rouge_vs_reference(self):
        src = ("from rouge_score import rouge_scorer\nsc = rouge_scorer.RougeScorer(['rougeL'])\n"
               "def rouge_reward(completion, answer, **kw):\n"
               "    return sc.score(answer, completion[-1]['content'])['rougeL'].fmeasure\n")
        fs = run(self.rid, rubric_env(src, "rouge_reward", "1.0"))
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].kind, "hygiene")

    def test_pos_copy_prompt(self):
        src = ("from difflib import SequenceMatcher\n"
               "def overlap_reward(completion, prompt, **kw):\n"
               "    return SequenceMatcher(None, completion[-1]['content'], prompt[-1]['content']).ratio()\n")
        fs = run(self.rid, rubric_env(src, "overlap_reward", "1.0"))
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].kind, "hack")
        self.assertIn("prompt", fs[0].message)

    def test_pos_textstat_only(self):
        src = ("import textstat\n"
               "def readability_reward(completion, **kw):\n"
               "    return textstat.flesch_reading_ease(completion[-1]['content']) / 100\n")
        fs = run(self.rid, rubric_env(src, "readability_reward", "1.0"))
        self.assertEqual(len(fs), 1)
        self.assertIn("readability", fs[0].message)

    def test_neg_thresholded(self):
        src = ("from rouge_score import rouge_scorer\nsc = rouge_scorer.RougeScorer(['rougeL'])\n"
               "def rouge_reward(completion, answer, **kw):\n"
               "    s = sc.score(answer, completion[-1]['content'])['rougeL'].fmeasure\n"
               "    return 1.0 if s >= 0.8 else 0.0\n")
        self.assertEqual(run(self.rid, rubric_env(src, "rouge_reward", "1.0")), [])

    def test_neg_minor_share(self):
        src = CORRECT + ("from rouge_score import rouge_scorer\nsc = rouge_scorer.RougeScorer(['rougeL'])\n"
                         "def rouge_reward(completion, answer, **kw):\n"
                         "    return sc.score(answer, completion[-1]['content'])['rougeL'].fmeasure\n")
        self.assertEqual(run(self.rid, rubric_env(src, "correct_answer_reward, rouge_reward", "0.7, 0.3")), [])

    def test_neg_exact_only(self):
        self.assertEqual(run(self.rid, rubric_env(CORRECT, "correct_answer_reward", "1.0")), [])


class KeywordCitations(unittest.TestCase):
    rid = "keyword-reward"

    def test_pos_url_count(self):
        src = CORRECT + ("def citation_reward(completion, **kw):\n"
                         "    text = completion[-1]['content']\n"
                         "    return min(1.0, len(re.findall(r'https?://\\S+', text)) / 5)\n")
        fs = run(self.rid, rubric_env(src, "correct_answer_reward, citation_reward", "0.5, 0.5"))
        self.assertEqual(len(fs), 1)
        self.assertIn("citations", fs[0].message)

    def test_pos_bracket_count(self):
        src = ("def cite_reward(completion, **kw):\n"
               "    n = len(set(re.findall(r'\\[\\d+\\]', completion[-1]['content'])))\n"
               "    return min(n, 10) / 10\n")
        self.assertEqual(len(run(self.rid, rubric_env(src, "cite_reward", "1.0"))), 1)

    def test_pos_doi_count(self):
        src = CORRECT + ("DOI = r'10\\.\\d{4,9}/\\S+'\n"
                         "def reference_reward(completion, **kw):\n"
                         "    return len(re.findall(DOI, completion[-1]['content'])) * 0.1\n")
        self.assertEqual(len(run(self.rid, rubric_env(src, "correct_answer_reward, reference_reward", "0.7, 0.3"))), 1)

    def test_neg_checked_against_retrieval(self):
        src = ("def citation_reward(completion, state, **kw):\n"
               "    urls = re.findall(r'https?://\\S+', completion[-1]['content'])\n"
               "    return sum(1 for u in urls if u in state['retrieved_urls']) / max(1, len(urls))\n")
        self.assertEqual(run(self.rid, rubric_env(src, "citation_reward", "1.0")), [])

    def test_neg_small_share(self):
        src = CORRECT + ("def citation_reward(completion, **kw):\n"
                         "    return min(1.0, len(re.findall(r'https?://\\S+', completion[-1]['content'])) / 5)\n")
        self.assertEqual(run(self.rid, rubric_env(src, "correct_answer_reward, citation_reward", "0.9, 0.1")), [])

    def test_neg_strip_urls_then_compare(self):
        src = ("def clean_match_reward(completion, answer, **kw):\n"
               "    t = re.sub(r'https?://\\S+', '', completion[-1]['content'])\n"
               "    return float(len(t.split()) > 0 and t.strip() == answer)\n")
        self.assertEqual(run(self.rid, rubric_env(src, "clean_match_reward", "1.0")), [])


LINT = ("import subprocess\n"
        "def lint_reward(completion, **kw):\n"
        "    out = subprocess_free_pylint(completion[-1]['content'])\n"
        "    return pylint_score(out) / 10\n")


class FormatHeavyNoCorrectness(unittest.TestCase):
    rid = "format-heavy-reward"

    def test_pos_lint_and_format(self):
        src = LINT + ("def format_reward(completion, **kw):\n"
                      "    return 1.0 if '```' in completion[-1]['content'] else 0.0\n")
        fs = run(self.rid, rubric_env(src, "lint_reward, format_reward", "0.7, 0.3"))
        self.assertEqual(len(fs), 1)
        self.assertIn("none checks", fs[0].message)

    def test_pos_security_only(self):
        src = ("def security_reward(completion, **kw):\n"
               "    issues = run_bandit(completion[-1]['content'])\n"
               "    return 1.0 if not issues else 0.0\n")
        self.assertEqual(len(run(self.rid, rubric_env(src, "security_reward", "1.0"))), 1)

    def test_pos_guard_only(self):
        src = ("from detoxify import Detoxify\nm = Detoxify('original')\n"
               "def safety_reward(completion, **kw):\n"
               "    return 1.0 - m.predict(completion[-1]['content'])['toxicity']\n")
        self.assertEqual(len(run(self.rid, rubric_env(src, "safety_reward", "1.0"))), 1)

    def test_neg_lint_plus_tests(self):
        src = LINT + ("def tests_pass_reward(completion, info, **kw):\n"
                      "    return run_tests(completion, info['tests'])\n")
        self.assertEqual(run(self.rid, rubric_env(src, "lint_reward, tests_pass_reward", "0.3, 0.7")), [])

    def test_neg_lint_weight_zero(self):
        src = LINT + CORRECT
        self.assertEqual(run(self.rid, rubric_env(src, "lint_reward, correct_answer_reward", "0.0, 1.0")), [])

    def test_neg_correctness_only(self):
        self.assertEqual(run(self.rid, rubric_env(CORRECT, "correct_answer_reward", "1.0")), [])


class AnswerInPromptRubric(unittest.TestCase):
    rid = "answer-in-prompt"

    def test_pos_rubric_in_prompt(self):
        env = "def fmt(x):\n    return {'prompt': f\"{x['question']}\\nRubric: {x['rubric']}\"}\nds = ds.map(fmt)\n"
        fs = run(self.rid, {"env.py": env})
        self.assertEqual(len(fs), 1)
        self.assertIn("rubric", fs[0].message)

    def test_pos_hint(self):
        env = "def fmt(x):\n    return {'question': x['question'] + '\\nHint: ' + x['hint']}\n"
        self.assertEqual(len(run(self.rid, {"env.py": env})), 1)

    def test_pos_grading_criteria_user_msg(self):
        env = ("def to_msgs(row):\n"
               "    return [{'role': 'user', 'content': f\"Task: {row['task']}. Make sure: {row.get('grading_criteria')}\"}]\n")
        self.assertEqual(len(run(self.rid, {"env.py": env})), 1)

    def test_neg_judge_prompt(self):
        env = ("async def judge_reward(completion, info, **kw):\n"
               "    msgs = [{'role': 'user', 'content': f\"Grade the reply using: {info['rubric']}\"}]\n"
               "    return await call(msgs)\n")
        self.assertEqual(run(self.rid, {"env.py": env}), [])

    def test_neg_assistant_example(self):
        env = "def shots(x):\n    return [{'role': 'assistant', 'content': f\"{x['hint']}\"}]\n"
        self.assertEqual(run(self.rid, {"env.py": env}), [])

    def test_neg_question_only(self):
        env = "def fmt(x):\n    return {'prompt': f\"{x['question']}\"}\n"
        self.assertEqual(run(self.rid, {"env.py": env}), [])


class ReferenceShippedHarbor(unittest.TestCase):
    rid = "reference-shipped-in-workspace"
    base = {"task.toml": TASK_TOML, "instruction.md": "Fix the totals.\n", "solution/solve.sh": PATCH_SOLVE}

    def files(self, extra):
        f = dict(self.base)
        f.update(extra)
        return f

    def test_pos_orig_backup(self):
        fs = run(self.rid, self.files({"environment/Dockerfile": "FROM python:3.12\nCOPY . /app\n",
                                       "environment/app.py.orig": "total = price * qty * (1 - discount_rate)\n"}))
        self.assertEqual(len(fs), 1)
        self.assertIn("backup file", fs[0].message)

    def test_pos_agents_md(self):
        body = FENCED_FIX
        fs = run(self.rid, self.files({"environment/Dockerfile": "FROM python:3.12\nCOPY AGENTS.md /app/AGENTS.md\n",
                                       "environment/AGENTS.md": body}))
        self.assertEqual(len(fs), 1)
        self.assertIn("AGENTS.md", fs[0].message)

    def test_pos_patch_with_backup(self):
        fs = run(self.rid, self.files({"environment/Dockerfile": "FROM python:3.12\nCOPY bug.diff /tmp/\n"
                                                                 "RUN cd /app && patch -b -p1 < /tmp/bug.diff\n"}))
        self.assertEqual(len(fs), 1)

    def test_neg_backup_removed(self):
        fs = run(self.rid, self.files({"environment/Dockerfile": "FROM python:3.12\nCOPY . /app\nRUN rm -f /app/app.py.orig\n",
                                       "environment/app.py.orig": "x\n"}))
        self.assertEqual(fs, [])

    def test_neg_backup_not_copied(self):
        fs = run(self.rid, self.files({"environment/Dockerfile": "FROM python:3.12\nCOPY requirements.txt /app/\n",
                                       "environment/app.py.orig": "x\n", "environment/requirements.txt": "numpy\n"}))
        self.assertEqual(fs, [])

    def test_neg_agents_md_no_solution(self):
        fs = run(self.rid, self.files({"environment/Dockerfile": "FROM python:3.12\nCOPY AGENTS.md /app/AGENTS.md\n",
                                       "environment/AGENTS.md": "Run the tests with pytest. Keep functions small.\n"}))
        self.assertEqual(fs, [])


class FeedbackRevealsAnswer(unittest.TestCase):
    rid = "verifier-feedback-leak"

    def env(self, body):
        return {"env.py": "import verifiers as vf\nclass E(vf.MultiTurnEnv):\n" + textwrap.indent(body, "    ")}

    def test_pos_correct_answer_is(self):
        fs = run(self.rid, self.env("async def env_response(self, messages, state, **kw):\n"
                                    "    answer = state['answer']\n"
                                    "    return [{'role': 'user', 'content': f'Wrong. The correct answer is {answer}.'}], state\n"))
        self.assertEqual(len(fs), 1)
        self.assertIn("gold answer", fs[0].message)

    def test_pos_subscript_answer_was(self):
        fs = run(self.rid, self.env("async def env_response(self, messages, state, **kw):\n"
                                    "    if not ok:\n"
                                    "        return [{'role': 'user', 'content': f\"Incorrect, the answer was {state['answer']}\"}], state\n"))
        self.assertEqual(len(fs), 1)

    def test_pos_feedback_should_be(self):
        fs = run(self.rid, self.env("def feedback(self, guess, gold):\n"
                                    "    return f'Not right, it should be {gold}'\n"))
        self.assertEqual(len(fs), 1)

    def test_neg_episode_ends(self):
        fs = run(self.rid, self.env("async def env_response(self, messages, state, **kw):\n"
                                    "    if not ok:\n"
                                    "        state['done'] = True\n"
                                    "        return [{'role': 'user', 'content': f\"Game over, the answer was {state['answer']}\"}], state\n"))
        self.assertEqual(fs, [])

    def test_neg_no_gold(self):
        fs = run(self.rid, self.env("async def env_response(self, messages, state, **kw):\n"
                                    "    return [{'role': 'user', 'content': f'Wrong, try again ({state[\"turn\"]} left)'}], state\n"))
        self.assertEqual(fs, [])

    def test_neg_reward_function(self):
        fs = run(self.rid, self.env("def correct_answer_reward(self, completion, answer):\n"
                                    "    msg = f'The correct answer is {answer}'\n    return 0.0\n"))
        self.assertEqual(fs, [])


class WeakExtraction(unittest.TestCase):
    rid = "first-match-label-regex"

    def test_pos_correct_substring(self):
        # 0.5.0 fix wave: this branch moved to judge-verdict-substring-match, and this rule stays quiet on it
        src = ("JUDGE = 'Reply CORRECT or INCORRECT.'\n"
               "async def judge_reward(completion, answer, **kw):\n"
               "    verdict = (await client.chat(JUDGE)).lower()\n"
               "    return 1.0 if 'correct' in verdict else 0.0\n")
        self.assertEqual(run(self.rid, rubric_env(src, "judge_reward", "1.0")), [])
        fs = run("judge-verdict-substring-match", rubric_env(src, "judge_reward", "1.0"))
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].kind, "hack")

    def test_pos_first_fence(self):
        src = ("def code_reward(completion, info, **kw):\n"
               "    m = re.search(r'```python\\n(.*?)```', completion[-1]['content'], re.S)\n"
               "    return run_code(m.group(1), info)\n")
        fs = run(self.rid, rubric_env(src, "code_reward", "1.0"))
        self.assertEqual(len(fs), 1)
        self.assertIn("first fenced", fs[0].message)
        self.assertEqual(fs[0].variant, "first-fenced-block")

    def test_pos_sympy_no_timeout(self):
        src = ("import sympy\n"
               "def math_reward(completion, answer, **kw):\n"
               "    return float(sympy.simplify(sympy.sympify(parse(completion)) - sympy.sympify(answer)) == 0)\n")
        fs = run(self.rid, rubric_env(src, "math_reward", "1.0"))
        self.assertEqual(len(fs), 1)
        self.assertIn("timeout", fs[0].message)

    def test_neg_incorrect_checked_first(self):
        src = ("JUDGE = 'Reply CORRECT or INCORRECT.'\n"
               "async def judge_reward(completion, answer, **kw):\n"
               "    verdict = (await client.chat(JUDGE)).lower()\n"
               "    if 'incorrect' in verdict:\n        return 0.0\n"
               "    return 1.0 if 'correct' in verdict else 0.0\n")
        self.assertEqual(run(self.rid, rubric_env(src, "judge_reward", "1.0")), [])

    def test_neg_last_fence(self):
        src = ("def code_reward(completion, info, **kw):\n"
               "    blocks = re.findall(r'```python\\n(.*?)```', completion[-1]['content'], re.S)\n"
               "    return run_code(blocks[-1], info)\n")
        self.assertEqual(run(self.rid, rubric_env(src, "code_reward", "1.0")), [])

    def test_neg_sympy_with_timeout(self):
        src = ("import sympy, signal\n"
               "def math_reward(completion, answer, **kw):\n"
               "    signal.alarm(5)\n"
               "    return float(sympy.simplify(sympy.sympify(parse(completion)) - sympy.sympify(answer)) == 0)\n")
        self.assertEqual(run(self.rid, rubric_env(src, "math_reward", "1.0")), [])


class Robustness(unittest.TestCase):
    def test_empty_and_odd_input(self):
        odd = {"env.py": "def f(:\n", "task.toml": "[[[", "instruction.md": "```\n", "solution/solve.sh": "\x7f",
               "data.jsonl": "{not json\n", "environment/Dockerfile": "COPY\n"}
        for rid in ("solution-in-task-text", "verifier-deserializes-agent-artifact", "shell-injection-from-model-text",
                    "open-fetch-tool", "keyword-reward", "format-heavy-reward", "answer-in-prompt",
                    "reference-shipped-in-workspace", "verifier-feedback-leak", "first-match-label-regex"):
            self.assertEqual(run(rid, {}), [])
            run(rid, odd)


# ---------------------------------------------------------------------------
# 0.5.0 fix wave, Part A: a branch of a rule can be off by default (OFF_VARIANTS)
# ---------------------------------------------------------------------------

FENCE_SRC = ("def code_reward(completion, info, **kw):\n"
             "    m = re.search(r'```python\\n(.*?)```', completion[-1]['content'], re.S)\n"
             "    return run_code(m.group(1), info)\n")
FENCE_KEY = "first-match-label-regex:first-fenced-block"


class OffVariants(unittest.TestCase):
    def setUp(self):
        self.saved = set(R.ENABLED)
        self.addCleanup(self.restore)

    def restore(self):
        R.ENABLED.clear()
        R.ENABLED.update(self.saved)

    def rules_of(self, pkg, only=None):
        return [f.rule for f in R.run_rules(pkg, only=only)[0]]

    def test_variant_listed_off(self):
        for key in ("keyword-reward:similarity", "aggregation-without-gate:v05", "grader-fails-open:v05",
                    "format-heavy-reward:v05", "verdict-parse-semantics:v05", FENCE_KEY):
            self.assertIn(key, R.OFF_VARIANTS)
            self.assertIn(key.split(":")[0], R.RULES)
            self.assertNotIn(key.split(":")[0], R.OFF_BY_DEFAULT)

    def test_off_by_default(self):
        R.ENABLED.discard(FENCE_KEY)
        R.ENABLED.discard("first-match-label-regex")
        self.assertNotIn("first-match-label-regex", self.rules_of(make_pkg(rubric_env(FENCE_SRC, "code_reward", "1.0"))))

    def test_on_with_variant_key(self):
        R.ENABLED.add(FENCE_KEY)
        self.assertIn("first-match-label-regex", self.rules_of(make_pkg(rubric_env(FENCE_SRC, "code_reward", "1.0"))))

    def test_on_with_bare_rule_id(self):
        R.ENABLED.add("first-match-label-regex")
        self.assertIn("first-match-label-regex", self.rules_of(make_pkg(rubric_env(FENCE_SRC, "code_reward", "1.0"))))

    def test_on_with_only(self):
        pkg = make_pkg(rubric_env(FENCE_SRC, "code_reward", "1.0"))
        self.assertEqual(self.rules_of(pkg, only={"first-match-label-regex"}), ["first-match-label-regex"])

    def test_on_with_env_var(self):
        import subprocess
        import sys
        code = ("import json, sys\nfrom envscan import rules as R\nfrom envscan.package import Package\n"
                "print(json.dumps([f.rule for f in R.run_rules(Package.from_dir(sys.argv[1]))[0]]))\n")
        with tempfile.TemporaryDirectory() as d:
            for rel, text in rubric_env(FENCE_SRC, "code_reward", "1.0").items():
                with open(os.path.join(d, rel), "w") as f:
                    f.write(text)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outs = []
            for val in ("", FENCE_KEY):
                env = dict(os.environ, ENVSCAN_ENABLE=val, PYTHONPATH=root)
                r = subprocess.run([sys.executable, "-c", code, d], capture_output=True, text=True, env=env, cwd=root)
                outs.append(json.loads(r.stdout))
        self.assertNotIn("first-match-label-regex", outs[0])
        self.assertIn("first-match-label-regex", outs[1])

    def test_on_with_cli_enable(self):
        from tests.cli_helpers import run as cli_run
        with tempfile.TemporaryDirectory() as d:
            for rel, text in rubric_env(FENCE_SRC, "code_reward", "1.0").items():
                with open(os.path.join(d, rel), "w") as f:
                    f.write(text)
            R.ENABLED.discard(FENCE_KEY)
            code, out, _ = cli_run("--json", d)
            self.assertNotIn("first-match-label-regex", [f["rule"] for f in json.loads(out)["findings"]])
            code, out, _ = cli_run("--json", "--enable", FENCE_KEY, d)
            fs = [f for f in json.loads(out)["findings"] if f["rule"] == "first-match-label-regex"]
            self.assertEqual(len(fs), 1)
            self.assertEqual(fs[0]["variant"], "first-fenced-block")
            code, _, err = cli_run("--enable", "first-match-label-regex:no-such-branch", d)
            self.assertEqual(code, 3)

    def test_rules_listing_names_variant(self):
        from tests.cli_helpers import run as cli_run
        code, out, _ = cli_run("--rules")
        self.assertEqual(code, 0)
        self.assertIn("variant first-fenced-block off by default", out)

    def test_empty_variant_absent_from_output(self):
        f = R.F("first-match-label-regex", "m.", ["a.py:1"], confidence="review")
        self.assertNotIn("variant", f.to_dict())
        g = R.F("first-match-label-regex", "m.", ["a.py:1"], confidence="review", variant="first-fenced-block")
        self.assertEqual(g.to_dict()["variant"], "first-fenced-block")
        from envscan.sarif import build_sarif
        rules = [R.RULES["first-match-label-regex"]]
        res = {"findings": [dict(f.to_dict(), kind="hack"), dict(g.to_dict(), kind="hack")]}
        out = build_sarif([{k: v for k, v in r.items() if k != "fn"} for r in rules], [(res, "", None)])
        props = [r["properties"] for r in out["runs"][0]["results"]]
        self.assertNotIn("variant", props[0])
        self.assertEqual(props[1]["variant"], "first-fenced-block")


# ---------------------------------------------------------------------------
# 0.5.0 fix wave, Part C item 7: judge-verdict-substring-match
# ---------------------------------------------------------------------------

class VerdictSubstring(unittest.TestCase):
    rid = "judge-verdict-substring-match"

    def judge(self, body, judge_line="JUDGE = 'Respond with exactly one word: correct or incorrect.'\n"):
        src = (judge_line + "async def judge_reward(completion, answer, **kw):\n"
               "    verdict = (await client.chat(JUDGE)).lower()\n" + body)
        return run(self.rid, rubric_env(src, "judge_reward", "1.0"))

    def test_registered(self):
        meta = R.RULES[self.rid]
        self.assertEqual((meta["kind"], meta["severity"]), ("hack", "high"))
        self.assertNotIn(self.rid, R.OFF_BY_DEFAULT)
        self.assertNotIn(";", meta["description"])

    def test_pos_plain_return(self):
        fs = self.judge("    return 1.0 if 'correct' in verdict else 0.0\n")
        self.assertEqual(len(fs), 1)
        self.assertEqual((fs[0].kind, fs[0].severity, fs[0].confidence), ("hack", "high", "review"))
        self.assertIn('"incorrect"', fs[0].message)
        self.assertIn('"correct"', fs[0].message)
        self.assertIn("negative verdict is paid as a pass", fs[0].message)

    def test_pos_if_branch_with_partial(self):
        fs = self.judge("    if 'correct' in verdict and 'partially' not in verdict:\n        return 1.0\n"
                        "    elif 'incorrect' in verdict:\n        return 0.0\n    return 0.5\n")
        self.assertEqual(len(fs), 1)

    def test_pos_tag_parse(self):
        fs = self.judge("    ev = verdict.split('<evaluation>')[-1]\n    if 'correct' in ev:\n        return 1.0\n"
                        "    return 0.0\n", judge_line="JUDGE = 'Grade it. Use <evaluation>correct</evaluation> or "
                        "<evaluation>incorrect</evaluation>.'\n")
        self.assertEqual(len(fs), 1)
        self.assertIn("in ev", fs[0].message)

    def test_neg_incorrect_checked_first(self):
        self.assertEqual(self.judge("    if 'incorrect' in verdict:\n        return 0.0\n"
                                    "    return 1.0 if 'correct' in verdict else 0.0\n"), [])

    def test_neg_exact_compare(self):
        self.assertEqual(self.judge("    return 1.0 if verdict.strip() == 'correct' else 0.0\n"), [])

    def test_neg_no_negative_label(self):
        self.assertEqual(self.judge("    return 1.0 if 'correct' in verdict else 0.0\n",
                                    judge_line="JUDGE = 'Reply with the word correct if the answer matches.'\n"), [])

    def test_unpaid_is_hygiene(self):
        src = ("JUDGE = 'Reply correct or incorrect.'\n"
               "async def judge_metric(completion, answer, **kw):\n"
               "    verdict = (await client.chat(JUDGE)).lower()\n"
               "    return 1.0 if 'correct' in verdict else 0.0\n"
               "def exact(completion, answer, **kw):\n    return float(completion == answer)\n")
        fs = run(self.rid, rubric_env(src, "exact, judge_metric", "1.0, 0.0"))
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].kind, "hygiene")



class OpenFetchOptIn(unittest.TestCase):
    """0.5.0 fix wave (check_3.md): fire only when the open web tool is the default or set in a shipped config."""
    rid = "open-fetch-tool"
    SWITCH = (FETCH_BASE + "from exa_py import Exa\nexa = Exa()\n"
              "def exa_search(query: str) -> str:\n    return str(exa.search(query))\n"
              "def bm25_search(query: str) -> str:\n    return local_index.search(query)\n"
              "def load_environment(retriever: str = 'DEFAULT', **kw):\n"
              "    tools = [bm25_search]\n"
              "    if retriever == 'exa':\n        tools = [exa_search]\n"
              "    return vf.ToolEnv(dataset=ds, tools=tools)\n")
    INSIDE = (FETCH_BASE + "from exa_py import Exa\n"
              "class SearchEnv(vf.ToolEnv):\n"
              "    def __init__(self, backend: str = 'DEFAULT', **kw):\n"
              "        self.backend = backend\n        super().__init__(tools=[self.web_search], **kw)\n"
              "    def web_search(self, query: str) -> str:\n"
              "        if self.backend == 'exa':\n            return str(Exa().search(query))\n"
              "        return OTHER\n")

    def test_neg_opt_in_tool_list(self):
        self.assertEqual(run(self.rid, {"env.py": self.SWITCH.replace("DEFAULT", "bm25")}), [])

    def test_neg_opt_in_backend_inside_tool(self):
        env = self.INSIDE.replace("DEFAULT", "local").replace("OTHER", "local_index.search(query)")
        self.assertEqual(run(self.rid, {"env.py": env}), [])

    def test_neg_config_sets_other_value(self):
        files = {"env.py": self.SWITCH.replace("DEFAULT", "bm25"), "configs/train.toml": 'retriever = "bm25"\n'}
        self.assertEqual(run(self.rid, files), [])

    def test_pos_default_is_web(self):
        self.assertEqual(len(run(self.rid, {"env.py": self.SWITCH.replace("DEFAULT", "exa")})), 1)

    def test_pos_shipped_config_turns_it_on(self):
        files = {"env.py": self.SWITCH.replace("DEFAULT", "bm25"), "configs/rl/exa.toml": 'retriever = "exa"\n'}
        self.assertEqual(len(run(self.rid, files)), 1)

    def test_pos_other_branch_is_web_too(self):
        env = self.INSIDE.replace("DEFAULT", "ddg").replace("OTHER", "str(DDGS().text(query))")
        self.assertEqual(len(run(self.rid, {"env.py": env})), 1)


class DeserializeFixWave(unittest.TestCase):
    """0.5.0 fix wave (check_4.md): pickle or torch.load only when the path can come from agent output, and
    say "tools" when sympify runs in a tool."""
    rid = "verifier-deserializes-agent-artifact"

    def reward(self, body, head="import pickle, io, os, torch, base64\nimport numpy as np\nimport verifiers as vf\n"):
        env = (head + "MODEL = os.path.join(os.path.dirname(__file__), 'ridge_state.pkl')\n"
               "def humanness_reward(completion, state, **kw):\n" + body +
               "rubric = vf.Rubric(funcs=[humanness_reward], weights=[1.0])\n")
        return run(self.rid, {"env.py": env})

    def test_neg_package_model_pickle(self):
        self.assertEqual(self.reward("    with open(MODEL, 'rb') as fh:\n        m = pickle.loads(fh.read())\n"
                                     "    return float(m.predict(completion))\n"), [])

    def test_neg_package_torch_weights(self):
        self.assertEqual(self.reward("    with open('weights/model.pt', 'rb') as fh:\n"
                                     "        sd = torch.load(io.BytesIO(fh.read()), weights_only=False)\n"
                                     "    return float(len(sd) > 0)\n"), [])

    def test_neg_package_numpy_pickle(self):
        self.assertEqual(self.reward("    with open('data/emb.npy', 'rb') as fh:\n"
                                     "        a = np.load(io.BytesIO(fh.read()), allow_pickle=True)\n"
                                     "    return float(a.mean())\n"), [])

    def test_pos_sandbox_pickle(self):
        fs = self.reward("    raw = sandbox.read_file('/app/model.pkl')\n    m = pickle.loads(raw)\n"
                         "    return float(m.score)\n")
        self.assertEqual(len(fs), 1)
        self.assertIn("The grader runs", fs[0].message)

    def test_pos_decoded_artifact(self):
        fs = self.reward("    m = pickle.loads(base64.b64decode(state['artifact']))\n    return float(m.score)\n")
        self.assertEqual(len(fs), 1)

    def test_pos_sympify_in_tool_says_tools(self):
        env = ("import sympy as sp\nimport verifiers as vf\n"
               "def solve_equation(expression: str) -> str:\n    expr = sp.sympify(expression)\n    return str(sp.solve(expr))\n"
               "def math_reasoning_reward(completion, answer, **kw):\n    return float(answer in completion[-1]['content'])\n"
               "env = vf.ToolEnv(tools=[solve_equation], rubric=vf.Rubric(funcs=[math_reasoning_reward]))\n")
        fs = run(self.rid, {"env.py": env})
        self.assertEqual(len(fs), 1)
        self.assertIn("The tools run sympify", fs[0].message)


if __name__ == "__main__":
    unittest.main()

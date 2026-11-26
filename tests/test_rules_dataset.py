"""Tests for the dataset family rules (envscan/rules_dataset.py).

Each case writes a few small files into a temp folder and loads them with Package.from_dir,
the same way a local scan does. Nothing in the fixtures is imported or run.
"""
import json
import os
import shutil
import tempfile
import textwrap
import time
import unittest

from envscan.package import Package
from envscan.rules import RULES

MINE = ("duplicate-prompts-conflicting-gold", "mcq-gold-position-skew", "train-eval-split-overlap",
        "question-reference-numeric-mismatch", "items-incompatible-with-exact-answer", "eval-harness-template-bug")

PYPROJECT = '[project]\nname = "x"\ndependencies = ["verifiers>=0.1.3"]\n'

ENV = '''
import verifiers as vf
from datasets import load_dataset

def load_environment():
    ds = load_dataset("json", data_files="data/train.jsonl")["train"]
    ev = load_dataset("json", data_files="data/test.jsonl")["train"]
    def correct_answer(completion, answer, **kw):
        return 1.0 if answer.strip() == completion[-1]["content"].strip() else 0.0
    return vf.SingleTurnEnv(dataset=ds, eval_dataset=ev, rubric=vf.Rubric(funcs=[correct_answer]))
'''


def pkg(files: dict) -> Package:
    d = tempfile.mkdtemp(prefix="envscan-dataset-")
    try:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text if rel.endswith((".jsonl", ".json", ".csv")) else textwrap.dedent(text).lstrip("\n"))
        return Package.from_dir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def jl(rows):
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def fire(rid, data=None, code=ENV, extra=None):
    files = {"pyproject.toml": PYPROJECT, "env.py": code}
    if data is not None:
        files["data/train.jsonl"] = data
    files.update(extra or {})
    return RULES[rid]["fn"](pkg(files))


def qa(n, start=0):
    return [{"question": f"What is the value of item number {i} in the list?", "answer": str(i * 3)}
            for i in range(start, start + n)]


class Registered(unittest.TestCase):
    def test_registered_review(self):
        for rid in MINE:
            self.assertIn(rid, RULES)
        f = fire("duplicate-prompts-conflicting-gold", jl(qa(10) + [dict(qa(1)[0], answer="999")]))
        self.assertTrue(f and all(x.confidence == "review" for x in f))

    def test_no_data_is_fast_and_silent(self):
        p = pkg({"pyproject.toml": PYPROJECT, "env.py": ENV})
        for rid in MINE:
            t = time.perf_counter()
            self.assertEqual(RULES[rid]["fn"](p), [], rid)
            self.assertLess(time.perf_counter() - t, 0.05, rid)

    def test_odd_input_no_crash(self):
        bad = '{"question": "x", "answer": \n{not json\n[1,2\n' + "\x7f" * 10
        for rid in MINE:
            fire(rid, bad, extra={"data/test.jsonl": "[{\"question\": 1", "data/x.csv": "a,b\n\"1,2\n",
                                  "data/y.json": "[" + "{\"question\": \"q\", \"answer\": 1}," * 5})


# ---------------------------------------------------------------------------
class DuplicateConflict(unittest.TestCase):
    R = "duplicate-prompts-conflicting-gold"

    def test_pos_same_question_two_answers(self):
        rows = qa(10) + [dict(qa(1, 4)[0], answer="7")]
        f = fire(self.R, jl(rows))
        self.assertEqual(len(f), 1)
        self.assertIn("different `answer`", f[0].message)

    def test_pos_key_not_in_options(self):
        rows = [{"question": f"Pick the right option for question {i} please", "choices": ["a", "b", "c", "d"],
                 "answer": "ABCD"[i % 4]} for i in range(20)]
        rows[3]["answer"] = "F"
        f = fire(self.R, jl(rows))
        self.assertTrue(f and "not one of their options" in f[0].message)

    def test_pos_sentinel(self):
        rows = qa(10)
        rows[2]["answer"] = "-99999"
        f = fire(self.R, jl(rows))
        self.assertTrue(f and "sentinel" in f[0].message)

    def test_pos_inline_python_rows(self):
        code = ENV.replace('ds = load_dataset("json", data_files="data/train.jsonl")["train"]',
                           'ds = Dataset.from_list(ROWS)') + "\nROWS = " + repr(qa(6) + [dict(qa(1)[0], answer="5")]) + "\n"
        self.assertTrue(fire(self.R, None, code=code))

    def test_neg_exact_duplicate_same_answer(self):
        self.assertEqual(fire(self.R, jl(qa(10) + qa(3))), [])

    def test_neg_file_not_loaded(self):
        code = ENV.replace("data/train.jsonl", "other.jsonl").replace("data/test.jsonl", "other2.jsonl")
        self.assertEqual(fire(self.R, jl(qa(10) + [dict(qa(1)[0], answer="7")]), code=code), [])

    def test_neg_answer_not_paid(self):
        code = ENV.replace("def correct_answer(completion, answer, **kw):\n        return 1.0 if answer.strip() == "
                           "completion[-1][\"content\"].strip() else 0.0",
                           "def length_reward(completion, **kw):\n        return len(completion) / 100")
        code = code.replace("correct_answer]", "length_reward]")
        self.assertEqual(fire(self.R, jl(qa(10) + [dict(qa(1)[0], answer="7")]), code=code), [])

    def test_neg_context_differs(self):
        rows = [{"question": "What is the capital of the country described here?", "context": f"country {i}",
                 "answer": f"city {i}"} for i in range(10)]
        self.assertEqual(fire(self.R, jl(rows)), [])

    def test_neg_multiturn_answer_drives_game(self):
        code = ENV.replace("vf.SingleTurnEnv", "WordEnv") + '''
class WordEnv(vf.MultiTurnEnv):
    def env_response(self, messages, state, **kw):
        return feedback(messages, state["answer"])
'''
        rows = [{"question": "Guess the secret five letter word now.", "answer": w} for w in ["apple", "grape", "melon", "lemon"]]
        self.assertEqual(fire(self.R, jl(rows), code=code), [])


# ---------------------------------------------------------------------------
def mc(n, letters, opts=True):
    rows = []
    for i in range(n):
        r = {"question": f"Question {i}: which of these is right?", "answer": letters[i % len(letters)]}
        if opts:
            r["choices"] = [f"opt{i}a", f"opt{i}b", f"opt{i}c", f"opt{i}d"]
        rows.append(r)
    return rows


class McqSkew(unittest.TestCase):
    R = "mcq-gold-position-skew"

    def test_pos_letter_skew(self):
        f = fire(self.R, jl(mc(60, "AAAAAAB C D".replace(" ", ""))))
        self.assertEqual(len(f), 1)
        self.assertIn("option A", f[0].message)

    def test_pos_int_index(self):
        rows = mc(50, "A")
        for i, r in enumerate(rows):
            r["answer"] = 2 if i % 5 else 0
        self.assertTrue(fire(self.R, jl(rows)))

    def test_pos_letters_in_question_text(self):
        rows = [{"question": f"Q{i}\nA) x\nB) y\nC) z\nD) w\n", "answer": "C" if i % 4 else "A"} for i in range(40)]
        self.assertTrue(fire(self.R, jl(rows)))

    def test_neg_balanced(self):
        self.assertEqual(fire(self.R, jl(mc(60, "ABCD"))), [])

    def test_neg_env_shuffles(self):
        code = ENV + "\ndef prep(x):\n    random.shuffle(x['choices'])\n    return x\n"
        self.assertEqual(fire(self.R, jl(mc(60, "AAAAAB")), code=code), [])

    def test_neg_too_few_rows(self):
        self.assertEqual(fire(self.R, jl(mc(12, "A"))), [])

    def test_neg_not_multiple_choice(self):
        self.assertEqual(fire(self.R, jl(qa(40))), [])


# ---------------------------------------------------------------------------
class SplitOverlap(unittest.TestCase):
    R = "train-eval-split-overlap"

    def test_pos_jsonl_pair(self):
        f = fire(self.R, jl(qa(20)), extra={"data/test.jsonl": jl(qa(3, 5) + qa(5, 100))})
        self.assertEqual(len(f), 1)
        self.assertIn("3 of 8 rows", f[0].message)

    def test_pos_json_split_keys(self):
        code = ENV.replace("data/train.jsonl", "data/all.json")
        f = fire(self.R, None, code=code, extra={"data/all.json": json.dumps({"train": qa(10), "test": qa(4, 2)})})
        self.assertTrue(f)

    def test_pos_csv_dev(self):
        code = ENV.replace("data/train.jsonl", "data/train.csv").replace("data/test.jsonl", "data/dev.csv")
        def csvt(rows):
            return "question,answer\n" + "".join(f"\"{r['question']}\",{r['answer']}\n" for r in rows)
        f = fire(self.R, None, code=code, extra={"data/train.csv": csvt(qa(10)), "data/dev.csv": csvt(qa(4, 6))})
        self.assertTrue(f)

    def test_neg_disjoint(self):
        self.assertEqual(fire(self.R, jl(qa(20)), extra={"data/test.jsonl": jl(qa(10, 50))}), [])

    def test_neg_one_shared_row(self):
        self.assertEqual(fire(self.R, jl(qa(20)), extra={"data/test.jsonl": jl(qa(1, 3) + qa(10, 50))}), [])

    def test_neg_eval_not_loaded(self):
        code = ENV.replace("data/test.jsonl", "elsewhere.jsonl")
        self.assertEqual(fire(self.R, jl(qa(20)), code=code, extra={"data/test.jsonl": jl(qa(5))}), [])


# ---------------------------------------------------------------------------
def gsm(n, bad=()):
    rows = []
    for i in range(n):
        a, b = 10 + i, 3 + i
        extra = 17 if i in bad else b
        rows.append({"question": f"Tom has {a} apples and buys {b} more bags of apples. How many apples now?",
                     "answer": f"He has {a} + {extra} = <<{a}+{extra}={a + extra}>>{a + extra} apples.\n#### {a + extra}"})
    return rows


class NumericMismatch(unittest.TestCase):
    R = "question-reference-numeric-mismatch"

    def test_pos_gsm_style(self):
        f = fire(self.R, jl(gsm(16, bad=(3, 9))))
        self.assertEqual(len(f), 1)
        self.assertIn("17", f[0].message)

    def test_pos_solution_field(self):
        rows = [{"question": f"A box holds {i + 4} pens. How many pens are in {i + 2} boxes?",
                 "solution": f"{i + 4} * {i + 2 if i not in (1, 7) else 33} = {(i + 4) * (i + 2)}",
                 "answer": str((i + 4) * (i + 2))} for i in range(12)]
        self.assertTrue(fire(self.R, jl(rows)))

    def test_pos_language_versions(self):
        rows = [{"question_en": f"Add {i} and {i + 5}.", "question_fr": f"Ajoutez {i} et {i + 5}.",
                 "question": f"Add {i} and {i + 5}.", "answer": str(2 * i + 5)} for i in range(10)]
        rows[4]["question_fr"] = "Ajoutez 4 et 4."
        self.assertTrue(fire(self.R, jl(rows)))

    def test_neg_all_present(self):
        self.assertEqual(fire(self.R, jl(gsm(16))), [])

    def test_neg_too_many_flagged(self):
        self.assertEqual(fire(self.R, jl(gsm(12, bad=range(8)))), [])

    def test_neg_no_worked_solution(self):
        self.assertEqual(fire(self.R, jl(qa(20))), [])


# ---------------------------------------------------------------------------
def figq(n, text="As shown in the figure, what is the length of side AB?"):
    return [{"question": f"{text} (item {i})", "answer": str(i + 2)} for i in range(n)]


class Incompatible(unittest.TestCase):
    R = "items-incompatible-with-exact-answer"

    def test_pos_figure(self):
        f = fire(self.R, jl(figq(5) + qa(10)))
        self.assertTrue(f and "figure" in f[0].message)

    def test_pos_proof(self):
        rows = [{"question": f"Prove that n^2 + n is even for n = {i}.", "answer": "even"} for i in range(4)] + qa(8)
        self.assertTrue(fire(self.R, jl(rows)))

    def test_pos_estimate(self):
        rows = [{"question": f"Estimate the number of piano tuners in city {i}.", "answer": str(100 + i)} for i in range(4)]
        self.assertTrue(fire(self.R, jl(rows + qa(8))))

    def test_neg_multimodal_env(self):
        code = ENV + "\ndef img(x):\n    return {'type': 'image_url', 'image_url': x}\n"
        self.assertEqual(fire(self.R, jl(figq(5) + qa(10)), code=code), [])

    def test_neg_image_column(self):
        rows = [dict(r, image=f"img{i}.png") for i, r in enumerate(figq(5) + qa(10))]
        self.assertEqual(fire(self.R, jl(rows)), [])

    def test_neg_judge_for_proofs(self):
        rows = [{"question": f"Prove that n^2 + n is even for n = {i}.", "answer": "even"} for i in range(4)] + qa(8)
        code = ENV.replace("vf.Rubric(funcs=[correct_answer])", "vf.JudgeRubric(judge_model='m')")
        self.assertEqual(fire(self.R, jl(rows), code=code), [])

    def test_neg_estimate_with_tolerance(self):
        rows = [{"question": f"Estimate the number of piano tuners in city {i}.", "answer": str(100 + i)} for i in range(4)]
        code = ENV.replace("return 1.0 if answer.strip()", "return 1.0 if math.isclose(float(answer), 1, rel_tol=0.1) or answer.strip()")
        self.assertEqual(fire(self.R, jl(rows + qa(8)), code=code), [])


# ---------------------------------------------------------------------------
TEMPLATE_ENV = '''
import verifiers as vf
PROMPT = "Answer the question below."

def load_environment():
    rows = []
    for row in RAW:
        rows.append({"question": PROMPT, "answer": row["gold"]})
    def correct_answer(completion, answer, **kw):
        return float(answer in completion[-1]["content"])
    return vf.SingleTurnEnv(dataset=Dataset.from_list(rows), rubric=vf.Rubric(funcs=[correct_answer]))
'''


class TemplateBug(unittest.TestCase):
    R = "eval-harness-template-bug"

    def test_pos_loop_constant_question(self):
        f = fire(self.R, None, code=TEMPLATE_ENV)
        self.assertEqual(len(f), 1)
        self.assertIn("env.py:7", f[0].evidence)

    def test_pos_map_lambda(self):
        code = ENV.replace('["train"]\n', '["train"].map(lambda x: {"question": "Solve it.", "answer": x["ans"]})\n', 1)
        self.assertTrue(fire(self.R, jl(qa(5)), code=code))

    def test_pos_format_without_slot(self):
        code = TEMPLATE_ENV.replace('"question": PROMPT', '"question": PROMPT.format(q=row["q"])')
        self.assertTrue(fire(self.R, None, code=code))

    def test_pos_collapsed_data(self):
        rows = [{"question": "Write the next number.", "answer": str(i)} for i in range(12)]
        self.assertTrue(fire(self.R, jl(rows)))

    def test_neg_question_from_row(self):
        code = TEMPLATE_ENV.replace('"question": PROMPT', '"question": PROMPT + row["q"]')
        self.assertEqual(fire(self.R, None, code=code), [])

    def test_neg_template_with_slot(self):
        code = TEMPLATE_ENV.replace('PROMPT = "Answer the question below."', 'PROMPT = "Answer: {q}"') \
            .replace('"question": PROMPT', '"question": PROMPT.format(q=row["q"])')
        self.assertEqual(fire(self.R, None, code=code), [])

    def test_neg_multiturn(self):
        code = TEMPLATE_ENV.replace("vf.SingleTurnEnv", "vf.MultiTurnEnv")
        self.assertEqual(fire(self.R, None, code=code), [])

    def test_neg_info_carries_row(self):
        code = TEMPLATE_ENV.replace('"answer": row["gold"]}', '"answer": row["gold"], "info": row}')
        self.assertEqual(fire(self.R, None, code=code), [])

    def test_neg_varied_data(self):
        self.assertEqual(fire(self.R, jl(qa(12))), [])



class DuplicateConflictFixWave(unittest.TestCase):
    """0.5.0 fix wave (check_5.md): nested golds are one answer, and encoded columns are not compared."""
    R = "duplicate-prompts-conflicting-gold"
    Q = "Find the probability that the particle is in the given region."

    def dup(self, a1, a2, q=None):
        q = q or self.Q
        return qa(10) + [{"question": q, "answer": a1}, {"question": q, "answer": a2}]

    def test_neg_one_gold_contains_other(self):
        self.assertEqual(fire(self.R, jl(self.dup("0.2387 or e^{-1/2}-e^{-1}", "0.2387"))), [])

    def test_neg_prefix_gold(self):
        self.assertEqual(fire(self.R, jl(self.dup("Paris", "Paris, France"))), [])

    def test_neg_base64_columns(self):
        rows = [{"question": "Y210OGNoYjZ5YnRkdG1xa2Nqb255bWQ5a2dlaGZq%02d" % i, "answer": "Y210OGNo%02dYjZ5" % i}
                for i in range(10)]
        rows += [{"question": "Y210OGNoYjZ5YnRkdG1xa2Nqb255bWQ5a2dlaGZqQQ", "answer": "a2dlaGZqQ2"},
                 {"question": "Y210OGNoYjZ5YnRkdG1xa2Nqb255bWQ5a2dlaGZqQQ", "answer": "Y210OGNo9Zz"}]
        self.assertEqual(fire(self.R, jl(rows)), [])

    def test_pos_different_words(self):
        f = fire(self.R, jl(self.dup("isanti", "Isantamda")))
        self.assertEqual(len(f), 1)
        self.assertIn("different `answer`", f[0].message)

    def test_pos_numbers_not_nested(self):
        self.assertEqual(len(fire(self.R, jl(self.dup("2", "12")))), 1)

    def test_pos_plain_text_with_digits(self):
        self.assertEqual(len(fire(self.R, jl(self.dup("Hohe 1", "Assiniboin 2")))), 1)


if __name__ == "__main__":
    unittest.main()

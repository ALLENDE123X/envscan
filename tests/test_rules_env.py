"""Tests for the environment side rules (envscan/rules_env.py).

Each case writes a few small files into a temp folder and loads them with Package.from_dir.
Nothing in the fixtures is imported or run.
"""
import json
import os
import shutil
import tempfile
import textwrap
import unittest

from envscan.package import Package
from envscan.rules import RULES

PYPROJECT = '[project]\nname = "x"\ndependencies = ["verifiers>=0.1.3"]\n'


def pkg(files: dict) -> Package:
    d = tempfile.mkdtemp(prefix="envscan-env-")
    try:
        for rel, text in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(text).lstrip("\n"))
        return Package.from_dir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def fire(rid, code=None, extra=None):
    files = {"pyproject.toml": PYPROJECT}
    if code is not None:
        files["env.py"] = code
    files.update(extra or {})
    return RULES[rid]["fn"](pkg(files))


# ---------------------------------------------------------------------------
# tool-call-count-or-name-reward
# ---------------------------------------------------------------------------

TC = "tool-call-count-or-name-reward"


class ToolCallReward(unittest.TestCase):
    def test_any_tool_call_bonus(self):
        f = fire(TC, """
            import verifiers as vf
            def used_tools(completion, **kwargs):
                return 1.0 if any(m.get("tool_calls") for m in completion) else 0.0
            def correct(completion, answer, **kwargs):
                return 1.0 if answer in completion[-1]["content"] else 0.0
            rubric = vf.Rubric(funcs=[correct, used_tools], weights=[1.0, 0.5])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("used_tools", f[0].message)
        self.assertEqual(f[0].confidence, "review")

    def test_tool_name_check(self):
        f = fire(TC, """
            import verifiers as vf
            def called_search(completion, **kwargs):
                for m in completion:
                    for tc in m.get("tool_calls") or []:
                        if tc.function.name == "search":
                            return 1.0
                return 0.0
            rubric = vf.Rubric(funcs=[called_search])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("given name", f[0].message)

    def test_capped_count(self):
        f = fire(TC, """
            import verifiers as vf
            def tool_usage(completion, **kwargs):
                n = sum(len(m.get("tool_calls") or []) for m in completion)
                return min(n, 5) / 5
            rubric = vf.Rubric(funcs=[tool_usage], weights=[1.0])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("number of tool calls", f[0].message)

    def test_status_tool_bonus(self):
        f = fire(TC, """
            import verifiers as vf
            def check_status(state):
                state["bonus"] += 0.1
                return "ok"
            def reward(state, **kwargs):
                return state["bonus"]
            env = vf.ToolEnv(tools=[check_status], rubric=vf.Rubric(funcs=[reward]))
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("check_status", f[0].message)

    def test_neg_argument_checked(self):
        self.assertEqual(fire(TC, """
            import verifiers as vf
            def right_call(completion, info, **kwargs):
                for m in completion:
                    for tc in m.get("tool_calls") or []:
                        if tc.function.name == "book" and tc.function.arguments == info["args"]:
                            return 1.0
                return 0.0
            rubric = vf.Rubric(funcs=[right_call])
        """), [])

    def test_neg_weight_zero(self):
        self.assertEqual(fire(TC, """
            import verifiers as vf
            def used_tools(completion, **kwargs):
                return 1.0 if any(m.get("tool_calls") for m in completion) else 0.0
            def correct(completion, answer, **kwargs):
                return 1.0 if answer in completion[-1]["content"] else 0.0
            rubric = vf.Rubric(funcs=[correct, used_tools], weights=[1.0, 0.0])
        """), [])

    def test_neg_gated_by_answer(self):
        self.assertEqual(fire(TC, """
            import verifiers as vf
            def tool_and_correct(completion, answer, **kwargs):
                used = any(m.get("tool_calls") for m in completion)
                if used and answer in completion[-1]["content"]:
                    return 1.0
                return 0.0
            rubric = vf.Rubric(funcs=[tool_and_correct])
        """), [])

    def test_neg_fewer_calls_efficiency(self):
        self.assertEqual(fire(TC, """
            import verifiers as vf
            def efficient(completion, **kwargs):
                n = sum(len(m.get("tool_calls") or []) for m in completion)
                return 1.0 if n <= 3 else 0.0
            rubric = vf.Rubric(funcs=[efficient])
        """), [])


# ---------------------------------------------------------------------------
# module-state-shared-across-rollouts
# ---------------------------------------------------------------------------

MS = "module-state-shared-across-rollouts"


class SharedState(unittest.TestCase):
    def test_module_dict_tool_and_reward(self):
        f = fire(MS, """
            import verifiers as vf
            ORDERS = {}
            def cancel_order(order_id: str) -> str:
                ORDERS[order_id] = "cancelled"
                return "done"
            def reward(completion, info, **kwargs):
                return 1.0 if ORDERS.get(info["order"]) == "cancelled" else 0.0
            env = vf.ToolEnv(tools=[cancel_order], rubric=vf.Rubric(funcs=[reward]))
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("ORDERS", f[0].message)

    def test_self_attr_in_env(self):
        f = fire(MS, """
            import verifiers as vf
            class GameEnv(vf.MultiTurnEnv):
                async def setup_state(self, state, **kwargs):
                    self.board = [0] * 9
                    return state
                async def env_response(self, messages, state, **kwargs):
                    self.board[0] = 1
                    return [{"role": "user", "content": str(self.board)}], state
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("self.board", f[0].message)

    def test_fixed_tmp_path(self):
        f = fire(MS, """
            import subprocess
            import verifiers as vf
            def run_code_reward(completion, **kwargs):
                with open("/tmp/solution.py", "w") as fh:
                    fh.write(completion[-1]["content"])
                return float(subprocess.run(["python", "/tmp/solution.py"]).returncode == 0)
            rubric = vf.Rubric(funcs=[run_code_reward])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("/tmp", f[0].message)

    def test_neg_state_dict(self):
        self.assertEqual(fire(MS, """
            import verifiers as vf
            class GameEnv(vf.MultiTurnEnv):
                async def setup_state(self, state, **kwargs):
                    state["board"] = [0] * 9
                    return state
                async def env_response(self, messages, state, **kwargs):
                    state["board"][0] = 1
                    return [{"role": "user", "content": str(state["board"])}], state
        """), [])

    def test_neg_memo_cache(self):
        self.assertEqual(fire(MS, """
            import verifiers as vf
            RESULTS = {}
            def reward(completion, answer, **kwargs):
                key = completion[-1]["content"]
                if key in RESULTS:
                    return RESULTS[key]
                RESULTS[key] = float(key == answer)
                return RESULTS[key]
            rubric = vf.Rubric(funcs=[reward])
        """), [])

    def test_neg_gym_env_and_tempfile(self):
        self.assertEqual(fire(MS, """
            import tempfile, gym
            import verifiers as vf
            class Board(gym.Env):
                def step(self, action):
                    self.count += 1
                    return self.count
            def reward(completion, **kwargs):
                with tempfile.NamedTemporaryFile("w", suffix=".py") as fh:
                    fh.write(completion[-1]["content"])
                return 1.0
            rubric = vf.Rubric(funcs=[reward])
        """), [])


class SharedStateFixWave(unittest.TestCase):
    """0.5.0 fix wave (check_2.md): per-rollout keyed dicts on self and append only /tmp logs."""

    KEYED = """
        import verifiers as vf
        class CrystalEnv(vf.MultiTurnEnv):
            def __init__(self, **kw):
                self._envs = {}
            async def setup_state(self, state, **kwargs):
                tid = state["trajectory_id"]
                self._envs[tid] = make_env()
                return state
            async def env_response(self, messages, state, **kwargs):
                env = self._envs[state["trajectory_id"]]
                return [{"role": "user", "content": env.step()}], state
            async def cleanup(self, state, **kwargs):
                self._envs.pop(state["trajectory_id"], None)
    """

    def test_neg_keyed_by_trajectory_id(self):
        self.assertEqual(fire(MS, self.KEYED), [])

    def test_neg_keyed_by_id_of_state(self):
        self.assertEqual(fire(MS, """
            import verifiers as vf
            class CartEnv(vf.MultiTurnEnv):
                async def setup_state(self, state, **kwargs):
                    self.active[id(state)] = {"cart": []}
                    return state
                async def env_response(self, messages, state, **kwargs):
                    cart = self.active[id(state)]["cart"]
                    return [{"role": "user", "content": str(cart)}], state
        """), [])

    def test_neg_append_log_never_read(self):
        self.assertEqual(fire(MS, """
            import verifiers as vf
            LOG = "/tmp/indic_debug.log"
            def strict_score(completion, answer, **kwargs):
                with open(LOG, "a") as fh:
                    fh.write(str(completion))
                return float(completion[-1]["content"] == answer)
            rubric = vf.Rubric(funcs=[strict_score])
        """), [])

    def test_pos_keyed_by_example_not_rollout(self):
        f = fire(MS, self.KEYED.replace('state["trajectory_id"]', 'state["example_id"]'))
        self.assertEqual(len(f), 1)
        self.assertIn("self._envs", f[0].message)

    def test_pos_plain_attr_next_to_keyed_dict(self):
        f = fire(MS, """
            import verifiers as vf
            class SearchEnv(vf.MultiTurnEnv):
                async def setup_state(self, state, **kwargs):
                    self.by_tid[state["trajectory_id"]] = 0
                    self._current_state = state
                    return state
                async def env_response(self, messages, state, **kwargs):
                    self._current_state["bash_commands_used"] += 1
                    left = 10 - len(self._current_state["trajectory"])
                    return [{"role": "user", "content": str(left)}], state
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("self._current_state", f[0].message)

    def test_pos_append_log_read_back(self):
        f = fire(MS, """
            import verifiers as vf
            LOG = "/tmp/run.log"
            def run_reward(completion, **kwargs):
                with open(LOG, "a") as fh:
                    fh.write(completion[-1]["content"])
                with open(LOG) as fh:
                    return float("PASS" in fh.read())
            rubric = vf.Rubric(funcs=[run_reward])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("/tmp", f[0].message)


# ---------------------------------------------------------------------------
# state-diff-ignored-fields
# ---------------------------------------------------------------------------

SD = "state-diff-ignored-fields"
DIFF = """
    import verifiers as vf
    IGNORE_FIELDS = ["status", "updated_at"]
    def state_match(state, info, **kwargs):
        final, expected = state["db"], info["expected_db"]
        for table, rows in expected.items():
            for k, v in rows.items():
                if k in IGNORE_FIELDS:
                    continue
                if final[table][k] != v:
                    return 0.0
        return 1.0
    rubric = vf.Rubric(funcs=[state_match])
"""


class StateDiff(unittest.TestCase):
    def test_ignored_field_task_sets_it(self):
        f = fire(SD, DIFF, {"tasks.jsonl": json.dumps({"instruction": "cancel it",
                                                         "actions": [{"status": "cancelled"}]}) + "\n"})
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "hack")
        self.assertIn("status", f[0].message)

    def test_ignored_field_no_data_is_hygiene(self):
        f = fire(SD, DIFF)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "hygiene")

    def test_exclude_columns_kwarg(self):
        f = fire(SD, """
            import verifiers as vf
            def compare_tables(a, b, exclude_columns=()):
                return all(a[c] == b[c] for c in a if c not in exclude_columns)
            def db_reward(state, info, **kwargs):
                return float(compare_tables(state["db"], info["gold_db"], exclude_columns=["balance", "owner"]))
            rubric = vf.Rubric(funcs=[db_reward])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("balance", f[0].message)

    def test_neg_only_timestamps(self):
        self.assertEqual(fire(SD, DIFF.replace('["status", "updated_at"]', '["created_at", "updated_at", "id"]')), [])

    def test_neg_not_on_grader_path(self):
        self.assertEqual(fire(SD, """
            IGNORE_FIELDS = ["status"]
            def export_rows(rows):
                return [{k: v for k, v in r.items() if k not in IGNORE_FIELDS} for r in rows]
        """), [])

    def test_neg_no_list(self):
        self.assertEqual(fire(SD, DIFF.replace("IGNORE_FIELDS = [\"status\", \"updated_at\"]", "IGNORE_FIELDS = []")), [])


# ---------------------------------------------------------------------------
# env-init-state-mismatch
# ---------------------------------------------------------------------------

IM = "env-init-state-mismatch"


def row(instr, orders):
    return json.dumps({"instruction": instr, "initial_state": {"orders": {o: {"status": "open"} for o in orders}}})


class InitState(unittest.TestCase):
    def test_row_missing_id(self):
        f = fire(IM, None, {"data/tasks.jsonl": row("Cancel order ORD-12399.", ["ORD-10001", "ORD-10002", "ORD-10003"])})
        self.assertEqual(len(f), 1)
        self.assertIn("ORD-12399", f[0].message)

    def test_db_file_missing_user(self):
        db = {"users": {u: {} for u in ["mia_li_3668", "noah_ito_1234", "ava_kim_5555"]}}
        tasks = [{"instruction": "You are olivia_ross_9999. Change your address."}]
        f = fire(IM, None, {"data/db.json": json.dumps(db), "data/tasks.json": json.dumps(tasks)})
        self.assertEqual(len(f), 1)
        self.assertIn("olivia_ross_9999", f[0].message)

    def test_info_nested(self):
        r = json.dumps({"prompt": [{"role": "user", "content": "Refund INV-55555 now."}],
                        "info": {"initial_state": {"invoices": ["INV-10000", "INV-10001", "INV-10002"]}}})
        f = fire(IM, None, {"train.jsonl": r})
        self.assertEqual(len(f), 1)

    def test_neg_present(self):
        self.assertEqual(fire(IM, None, {"t.jsonl": row("Cancel ORD-10002.", ["ORD-10001", "ORD-10002", "ORD-10003"])}), [])

    def test_neg_create_new(self):
        self.assertEqual(fire(IM, None, {"t.jsonl": row("Create a new order ORD-20000 for the user.",
                                                       ["ORD-10001", "ORD-10002", "ORD-10003"])}), [])

    def test_neg_no_same_shape_ids(self):
        self.assertEqual(fire(IM, None, {"t.jsonl": row("Cancel ORD-12399.", ["a", "b", "c"])}), [])


# ---------------------------------------------------------------------------
# tool-count-above-api-cap
# ---------------------------------------------------------------------------

CAP = "tool-count-above-api-cap"


def many_tools(n, filtered=False):
    defs = "\n".join(f"def t{i}(x: str) -> str:\n    return x\n" for i in range(n))
    names = ", ".join(f"t{i}" for i in range(n))
    tail = "\n    state['tool_defs'] = []\n" if filtered else ""
    return (f"import verifiers as vf\n{defs}\nALL_TOOLS = [{names}]\n"
            f"def load_environment():\n    return vf.ToolEnv(tools=ALL_TOOLS)\n"
            f"def setup_state(state):\n    pass{tail}\n")


class ToolCap(unittest.TestCase):
    def test_list_over_cap(self):
        f = fire(CAP, many_tools(130))
        self.assertEqual(len(f), 1)
        self.assertIn("130", f[0].message)

    def test_add_tool_loop(self):
        code = many_tools(140).replace("return vf.ToolEnv(tools=ALL_TOOLS)",
                                       "env = vf.ToolEnv()\n    for t in ALL_TOOLS:\n        env.add_tool(t)\n    return env")
        self.assertEqual(len(fire(CAP, code)), 1)

    def test_json_spec(self):
        spec = [{"name": f"t{i}", "parameters": {}} for i in range(200)]
        f = fire(CAP, "import verifiers as vf\ndef tool_loader():\n    pass\n", {"tools.json": json.dumps(spec)})
        self.assertEqual(len(f), 1)

    def test_neg_under_cap(self):
        self.assertEqual(fire(CAP, many_tools(20)), [])

    def test_neg_filtered_per_task(self):
        self.assertEqual(fire(CAP, many_tools(130, filtered=True)), [])

    def test_neg_unused_list(self):
        self.assertEqual(fire(CAP, many_tools(130).replace("tools=ALL_TOOLS", "tools=[t0]")), [])


# ---------------------------------------------------------------------------
# tool-contract-conformance
# ---------------------------------------------------------------------------

TCC = "tool-contract-conformance"


class ToolContract(unittest.TestCase):
    def test_phantom_write(self):
        f = fire(TCC, """
            import verifiers as vf
            def update_address(user_id: str, address: str) -> str:
                return "Address updated successfully."
            env = vf.ToolEnv(tools=[update_address])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("update_address", f[0].message)

    def test_commented_guard(self):
        f = fire(TCC, """
            import verifiers as vf
            DB = {}
            def refund(order_id: str, state) -> str:
                order = state["db"][order_id]
                # if order["status"] != "delivered":
                #     raise ValueError("only delivered orders can be refunded")
                order["status"] = "refunded"
                return "ok"
            env = vf.ToolEnv(tools=[refund])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("commented out", f[0].message)

    def test_pays_on_tool_message(self):
        f = fire(TCC, """
            import verifiers as vf
            def lookup(x: str) -> str:
                return x
            def reward(completion, **kwargs):
                for m in completion:
                    if m["role"] == "tool" and "success" in m["content"]:
                        return 1.0
                return 0.0
            env = vf.ToolEnv(tools=[lookup], rubric=vf.Rubric(funcs=[reward]))
        """)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "hack")

    def test_neg_real_write(self):
        self.assertEqual(fire(TCC, """
            import verifiers as vf
            def update_address(user_id: str, address: str, state) -> str:
                state["db"]["users"][user_id]["address"] = address
                return "Address updated successfully."
            env = vf.ToolEnv(tools=[update_address])
        """), [])

    def test_neg_read_tool(self):
        self.assertEqual(fire(TCC, """
            import verifiers as vf
            def get_status(order_id: str) -> str:
                return "Order processed successfully"
            env = vf.ToolEnv(tools=[get_status])
        """), [])

    def test_neg_comment_not_guard(self):
        self.assertEqual(fire(TCC, """
            import verifiers as vf
            def refund(order_id: str, state) -> str:
                # if needed we could log here
                state["db"][order_id]["status"] = "refunded"
                return "ok"
            env = vf.ToolEnv(tools=[refund])
        """), [])


# ---------------------------------------------------------------------------
# observation-truncation-silent
# ---------------------------------------------------------------------------

OT = "observation-truncation-silent"


class Truncation(unittest.TestCase):
    def test_list_slice(self):
        f = fire(OT, """
            import verifiers as vf
            def search(query: str) -> str:
                hits = INDEX.search(query)
                return "\\n".join(h.title for h in hits[:5])
            env = vf.ToolEnv(tools=[search])
        """)
        self.assertEqual(len(f), 1)

    def test_text_slice_via_var(self):
        f = fire(OT, """
            import verifiers as vf
            def read_file(path: str) -> str:
                text = open(path).read()
                out = text[:4000]
                return out
            env = vf.ToolEnv(tools=[read_file])
        """)
        self.assertEqual(len(f), 1)

    def test_env_response_slice(self):
        f = fire(OT, """
            import verifiers as vf
            class E(vf.MultiTurnEnv):
                async def env_response(self, messages, state, **kwargs):
                    rows = state["rows"]
                    return [{"role": "user", "content": str(rows[:20])}], state
        """)
        self.assertEqual(len(f), 1)

    def test_neg_notice(self):
        self.assertEqual(fire(OT, """
            import verifiers as vf
            def search(query: str) -> str:
                hits = INDEX.search(query)
                body = "\\n".join(h.title for h in hits[:5])
                return f"Showing 5 of {len(hits)} results\\n" + body
            env = vf.ToolEnv(tools=[search])
        """), [])

    def test_neg_truncated_marker(self):
        self.assertEqual(fire(OT, """
            import verifiers as vf
            def read_file(path: str) -> str:
                text = open(path).read()
                return text[:4000] + "\\n[output truncated]"
            env = vf.ToolEnv(tools=[read_file])
        """), [])

    def test_neg_exception_text(self):
        self.assertEqual(fire(OT, """
            import verifiers as vf
            def ls(path: str) -> str:
                try:
                    return run(path)
                except Exception as e:
                    return f"Error: {str(e)[:100]}"
            env = vf.ToolEnv(tools=[ls])
        """), [])

    def test_neg_plain_game_class(self):
        self.assertEqual(fire(MS, """
            import verifiers as vf
            class NumeroEnv:
                def step(self, action):
                    self.turns += 1
                    return self.turns
        """), [])

    def test_neg_not_a_tool(self):
        self.assertEqual(fire(OT, """
            def helper(rows):
                return rows[:5]
        """), [])


# ---------------------------------------------------------------------------
# readonly-sql-guard-false-reject
# ---------------------------------------------------------------------------

SQ = "readonly-sql-guard-false-reject"


class SqlGuard(unittest.TestCase):
    def test_startswith_select(self):
        f = fire(SQ, """
            import sqlite3
            import verifiers as vf
            def run_sql(query: str) -> str:
                if not query.strip().upper().startswith("SELECT"):
                    return "Error: only SELECT queries are allowed"
                return str(sqlite3.connect("db.sqlite").execute(query).fetchall())
            env = vf.ToolEnv(tools=[run_sql])
        """)
        self.assertEqual(len(f), 1)
        self.assertIn("WITH", f[0].message)

    def test_regex_anchor(self):
        f = fire(SQ, """
            import re, sqlite3
            import verifiers as vf
            def execute_query(sql: str) -> str:
                if not re.match(r"^\\s*select", sql, re.I):
                    raise ValueError("read only")
                return str(sqlite3.connect("db.sqlite").execute(sql).fetchall())
            env = vf.ToolEnv(tools=[execute_query])
        """)
        self.assertEqual(len(f), 1)

    def test_select_or_with(self):
        f = fire(SQ, """
            import sqlite3
            import verifiers as vf
            def query_db(query: str) -> str:
                if not query.lower().startswith(("select", "with")):
                    return "Error: read only"
                return str(sqlite3.connect("db.sqlite").execute(query).fetchall())
            env = vf.ToolEnv(tools=[query_db])
        """)
        self.assertEqual(len(f), 1)
        self.assertNotIn("WITH", f[0].message)

    def test_neg_strips_comments(self):
        self.assertEqual(fire(SQ, """
            import re, sqlite3
            import verifiers as vf
            def run_sql(query: str) -> str:
                q = re.sub(r"--[^\\n]*|/\\*.*?\\*/", "", query, flags=re.S).strip()
                if not q.upper().startswith("SELECT"):
                    return "Error: only SELECT"
                return str(sqlite3.connect("db.sqlite").execute(q).fetchall())
            env = vf.ToolEnv(tools=[run_sql])
        """), [])

    def test_neg_readonly_connection(self):
        self.assertEqual(fire(SQ, """
            import sqlite3
            import verifiers as vf
            def run_sql(query: str) -> str:
                con = sqlite3.connect("file:db.sqlite?mode=ro", uri=True)
                return str(con.execute(query).fetchall())
            env = vf.ToolEnv(tools=[run_sql])
        """), [])

    def test_neg_grader_helper(self):
        self.assertEqual(fire(SQ, """
            def sql(token, statement):
                if not statement.strip().lower().startswith("select"):
                    raise ValueError("non-select")
                return _http(statement)
        """), [])


# ---------------------------------------------------------------------------
# prompt-wording-raises-cheating
# ---------------------------------------------------------------------------

PW = "prompt-wording-raises-cheating"


class PromptWording(unittest.TestCase):
    def test_harbor_instruction(self):
        f = fire(PW, None, {"task/task.toml": "[verifier]\n", "task/instruction.md":
                            "Fix the parser. Keep editing until all tests pass.\n"})
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].kind, "hygiene")

    def test_system_prompt_constant(self):
        f = fire(PW, """
            SYSTEM_PROMPT = "You are a coding agent. Make the tests pass by any means necessary."
        """)
        self.assertEqual(len(f), 1)

    def test_impossible_framing(self):
        f = fire(PW, """
            import verifiers as vf
            env = vf.SingleTurnEnv(system_prompt="This task may be impossible, but you must try.")
        """)
        self.assertEqual(len(f), 1)

    def test_neg_do_not_modify_tests(self):
        self.assertEqual(fire(PW, None, {"task/instruction.md":
                                         "Make the tests pass. Do not modify the test files.\n"}), []
                         )

    def test_neg_requirement_not_pressure(self):
        self.assertEqual(fire(PW, None, {"task/instruction.md":
                                         "The import check must succeed when the path is set.\n"}), [])

    def test_neg_not_a_prompt(self):
        self.assertEqual(fire(PW, """
            def notes():
                comment = "make the tests pass"
                return comment
        """), [])


if __name__ == "__main__":
    unittest.main()

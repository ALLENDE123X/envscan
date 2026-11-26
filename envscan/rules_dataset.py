"""Dataset family rules (envscan v0.5).

Rules that read the data rows a package ships (JSONL, JSON, CSV and TSV files, and dataset rows
written inline in Python) and look for defects in the questions and answer keys. Like every envscan
rule these read text and ASTs only. Every finding is `review`.

Many envs load their rows from a remote dataset at run time and ship none. The rules then find no
rows and return at once. Each rule reads at most MAX_FILES data files, MAX_ROWS rows per dataset and
MAX_BYTES of text per file, so a large file cannot slow a scan.

Every rule checks two things before it reports: the data file is referenced by the env code (by
name, glob or template), and the answer field reaches a reward function or a library rubric.
"""
from __future__ import annotations

import ast
import csv
import fnmatch
import io
import json
import math
import os
import re
from collections import Counter, defaultdict

from .package import Package, PyFile, _const_text
from .rules import F, PLACEHOLDER, _iter_jsonl_rows, ev, names_loaded, reward_path_funcs, rule
from .rules_grader import _cache

# ---------------------------------------------------------------------------
# caps
# ---------------------------------------------------------------------------

MAX_FILES = 40            # data files read per package
MAX_ROWS = 2000           # rows kept per dataset
MAX_BYTES = 400_000       # text read per data file (the Package already keeps only this much)
MAX_TOTAL_ROWS = 20_000   # rows kept per package
MIN_ROWS = 4              # a dataset smaller than this is not checked

_DATA_EXT = (".jsonl", ".json", ".csv", ".tsv")
_SKIP_PARTS = {"outputs", "results", "logs", "runs", "rollouts", "reference_runs", "cache", "calibration",
               "node_modules", "wandb", "checkpoints", "traces", "trajectories", ".github"}
_SKIP_BASE = re.compile(r"(result|metadata|config|manifest|lock|package|tsconfig|schema|inventory|settings|"
                        r"rollout|trajector|leaderboard|score|metric|summary|stats|report|log)", re.I)

# ---------------------------------------------------------------------------
# field roles
# ---------------------------------------------------------------------------

Q_KEYS = ("question", "problem", "prompt", "query", "instruction", "question_text", "input", "task")
A_KEYS = ("answer", "gold", "gold_answer", "ground_truth", "expected_answer", "final_answer", "correct_answer",
          "answer_key", "target", "label", "correct_option", "answer_idx", "answer_index", "cop", "reference")
SOL_KEYS = ("solution", "rationale", "explanation", "reasoning", "worked_solution", "steps", "cot",
            "answer_explanation", "solution_text")
CHOICE_KEYS = ("choices", "options", "candidates", "answer_choices", "mc_options", "answers_list")
IMG_KEY = re.compile(r"(image|img|figure|picture|photo|png|jpe?g|diagram|visual)", re.I)
META_KEY = re.compile(r"^(id|_id|idx|index|uid|qid|\w*_id|\w*_idx|source|split|meta\w*|tags?|category|"
                      r"subject|topic|difficulty|level|type|dataset|origin|url|link|license|created\w*|"
                      r"timestamp|hash|year|author|domain|subset|language|lang)$", re.I)
_LETTERS = "ABCDEFGHIJ"


def _role(keys, cands):
    low = {k.lower(): k for k in keys}
    for c in cands:
        if c in low:
            return low[c]
    return None


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


class _DS:
    __slots__ = ("path", "name", "rows", "lines", "source", "truncated")

    def __init__(self, path, name, rows, lines, source, truncated=False):
        self.path, self.name, self.rows, self.lines, self.source, self.truncated = \
            path, name, rows, lines, source, truncated


def _skip_path(p: str) -> bool:
    parts = p.split("/")
    if any(x in _SKIP_PARTS for x in parts[:-1]):
        return True
    stem = os.path.splitext(parts[-1])[0]
    return bool(_SKIP_BASE.search(stem)) and not re.search(r"(train|test|eval|dev|val|data|question|item|split)", stem, re.I)


def _jsonl_rows(text: str):
    rows, lines = [], []
    for i, ln in enumerate(text.splitlines(), 1):
        if not ln.lstrip().startswith("{"):
            continue
        for r in _iter_jsonl_rows(ln, 1):
            if isinstance(r, dict):
                rows.append(r)
                lines.append(i)
        if len(rows) >= MAX_ROWS:
            break
    return rows, lines


def _partial_array(text: str, start: int):
    """Rows of a JSON array cut off at the byte cap: decode one element at a time."""
    dec = json.JSONDecoder()
    out, i, n = [], start + 1, len(text)
    while i < n and len(out) < MAX_ROWS:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            v, j = dec.raw_decode(text, i)
        except Exception:
            break
        out.append(v)
        i = j
    return out


def _json_datasets(p: str, text: str, truncated: bool):
    out = []
    try:
        j = json.loads(text)
    except Exception:
        j = None
        s = text.lstrip()
        if truncated and s.startswith("["):
            j = _partial_array(text, text.index("["))
    if isinstance(j, list):
        rows = [r for r in j[:MAX_ROWS] if isinstance(r, dict)]
        if rows:
            out.append(_DS(p, p, rows, None, text, truncated))
    elif isinstance(j, dict):
        for k, v in list(j.items())[:20]:
            if isinstance(v, list):
                rows = [r for r in v[:MAX_ROWS] if isinstance(r, dict)]
                if len(rows) >= MIN_ROWS:
                    out.append(_DS(p, f"{p}#{k}", rows, None, text, truncated))
    return out


def _csv_dataset(p: str, text: str, truncated: bool):
    try:
        rd = csv.DictReader(io.StringIO(text), delimiter="\t" if p.endswith(".tsv") else ",")
        rows, lines = [], []
        for r in rd:
            if None in r:
                r.pop(None, None)
            rows.append(r)
            lines.append(rd.line_num)
            if len(rows) >= MAX_ROWS:
                break
    except Exception:
        return []
    if truncated and rows:
        rows, lines = rows[:-1], lines[:-1]
    return [_DS(p, p, rows, lines, text, truncated)] if rows else []


def _lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        t = _const_text(node)
        return t


def _py_datasets(pf: PyFile):
    """Rows written inline: a list of dict literals with string keys, or a dict of equal length lists."""
    out = []
    if pf.tree is None:
        return out
    seen = set()
    for n in ast.walk(pf.tree):
        if isinstance(n, (ast.List, ast.Tuple)) and len(n.elts) >= MIN_ROWS and id(n) not in seen:
            ds = [e for e in n.elts if isinstance(e, ast.Dict)]
            if len(ds) < max(MIN_ROWS, 0.8 * len(n.elts)):
                continue
            rows, lines = [], []
            for d in ds[:MAX_ROWS]:
                if not all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in d.keys):
                    rows = []
                    break
                r = {}
                for k, v in zip(d.keys, d.values):
                    val = _lit(v)
                    if val is not None:
                        r[k.value] = val
                rows.append(r)
                lines.append(d.lineno)
            if rows:
                out.append(_DS(pf.path, f"{pf.path}:{n.lineno}", rows, lines, None))
        elif isinstance(n, ast.Dict) and n.keys and all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                                                         for k in n.keys):
            if not all(isinstance(v, (ast.List, ast.Tuple)) for v in n.values):
                continue
            cols = {k.value: v for k, v in zip(n.keys, n.values)}
            lens = {len(v.elts) for v in cols.values()}
            if len(lens) != 1 or lens.pop() < MIN_ROWS:
                continue
            if not (_role(cols, Q_KEYS) and _role(cols, A_KEYS)):
                continue
            for v in cols.values():
                seen.add(id(v))
            m = len(next(iter(cols.values())).elts)
            rows = [{k: _lit(v.elts[i]) for k, v in cols.items()} for i in range(min(m, MAX_ROWS))]
            lines = [cols[_role(cols, Q_KEYS)].elts[i].lineno for i in range(len(rows))]
            out.append(_DS(pf.path, f"{pf.path}:{n.lineno}", rows, lines, None))
    return out


def datasets(pkg: Package) -> list[_DS]:
    """All shipped datasets with at least MIN_ROWS dict rows, capped. Cached on the package."""
    c = _cache(pkg)
    if "ds_sets" in c:
        return c["ds_sets"]
    out, total, nfiles = [], 0, 0
    for p in sorted(pkg.files):
        if not p.endswith(_DATA_EXT) or _skip_path(p):
            continue
        if nfiles >= MAX_FILES or total >= MAX_TOTAL_ROWS:
            break
        text = pkg.files[p][:MAX_BYTES]
        trunc = p in pkg.truncated or len(pkg.files[p]) > MAX_BYTES
        try:
            if p.endswith(".jsonl"):
                rows, lines = _jsonl_rows(text)
                got = [_DS(p, p, rows, lines, text, trunc)] if rows else []
            elif p.endswith(".json"):
                got = _json_datasets(p, text, trunc)
            else:
                got = _csv_dataset(p, text, trunc)
        except Exception:
            got = []
        got = [d for d in got if len(d.rows) >= MIN_ROWS]
        if got:
            nfiles += 1
        for d in got:
            total += len(d.rows)
            out.append(d)
    for pf in pkg.py_all():
        if total >= MAX_TOTAL_ROWS:
            break
        try:
            for d in _py_datasets(pf):
                total += len(d.rows)
                out.append(d)
        except RecursionError:
            pass
    c["ds_sets"] = out
    return out


# ---------------------------------------------------------------------------
# gates: is the file loaded, and does the answer reach a reward
# ---------------------------------------------------------------------------


def _code(pkg: Package) -> str:
    """All env Python text, joined once per package."""
    c = _cache(pkg)
    if "ds_code" not in c:
        c["ds_code"] = "\n".join(pf.text for pf in pkg.py_all())
    return c["ds_code"]


def _has(pkg: Package, key: str, rx) -> bool:
    c = _cache(pkg)
    k = ("ds_has", key)
    if k not in c:
        c[k] = any(rx.search(pf.text) for pf in pkg.py_all())
    return c[k]


def _code_strings(pkg: Package) -> list[str]:
    c = _cache(pkg)
    if "ds_strings" not in c:
        s = set()
        for pf in pkg.py_all():
            if pf.tree is None:
                continue
            for n in ast.walk(pf.tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 300:
                    s.add(n.value)
                elif isinstance(n, ast.JoinedStr):
                    t = _const_text(n)
                    if t and len(t) < 300:
                        s.add(t)
        for p, t in pkg.files.items():
            if p.endswith((".toml", ".yaml", ".yml")) and len(t) < 100_000:
                s.update(re.findall(r"[\w./*{}\-]+\.(?:jsonl|json|csv|tsv)", t))
        c["ds_strings"] = sorted(s)
    return c["ds_strings"]


_SPLIT_WORDS = {"train", "test", "eval", "valid", "validation", "data", "dataset", "examples", "items", "questions"}
_LISTING = re.compile(r"(glob\(|rglob\(|listdir\(|iterdir\(|os\.walk\(|scandir\(|data_dir|data_files)")


def file_used(pkg: Package, ds: _DS) -> bool:
    """The env code loads this file: by name, by a glob or f-string template, or by listing its folder."""
    if ds.source is None:
        return True  # inline rows in env code
    c = _cache(pkg)
    key = ("ds_used", ds.path)
    if key in c:
        return c[key]
    base = os.path.basename(ds.path)
    stem = os.path.splitext(base)[0]
    parent = os.path.basename(os.path.dirname(ds.path))
    text = _code(pkg)
    used = False
    for s in _code_strings(pkg):
        if base in s or (len(stem) >= 4 and stem.lower() not in _SPLIT_WORDS and (s == stem or s.endswith("/" + stem))):
            used = True
            break
        pat = re.sub(r"\{[^}]*\}", "*", s)
        if "*" in pat and len(pat) > 2 and re.search(r"\.(jsonl|json|csv|tsv|\*)$", pat):
            if fnmatch.fnmatch(base, os.path.basename(pat)) and (os.path.dirname(pat) in ("", "*")
                                                                   or os.path.dirname(pat).split("/")[-1] in (parent, "*")):
                used = True
                break
    if not used and parent and _LISTING.search(text):
        used = any(s == parent or s.endswith("/" + parent) or s.endswith("/" + parent + "/") for s in _code_strings(pkg))
    c[key] = used
    return used


_LIB_RUBRIC = re.compile(r"\b(JudgeRubric|MathRubric|CodeMathRubric|ExactMatchRubric)\s*\(")


def answer_paid(pkg: Package, akey: str) -> bool:
    """The answer column reaches a reward: a reward path function reads `answer` (or the column name),
    or a library rubric that reads `answer` is used, and a renamed column is mapped to `answer` in code."""
    c = _cache(pkg)
    if "ds_reward_names" not in c:
        names = set()
        for _pf, fi in reward_path_funcs(pkg):
            names |= names_loaded(fi.node)
            names |= set(fi.params)
        c["ds_reward_names"] = names
        c["ds_lib_rubric"] = bool(_LIB_RUBRIC.search(_code(pkg)))
    names = c["ds_reward_names"]
    if akey in names and akey != "answer":
        return True
    if not ("answer" in names or c["ds_lib_rubric"]):
        return False
    if akey == "answer":
        return True
    q = re.escape(akey)
    return any(re.search(rf"[\"']{q}[\"']", pf.text) for pf in pkg.py_all())


# ---------------------------------------------------------------------------
# normalizers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list) and v and all(isinstance(m, dict) for m in v):
        parts = [m.get("content") for m in v if m.get("role", "user") != "assistant"]
        return "\n".join(p if isinstance(p, str) else json.dumps(p, sort_keys=True, default=str) for p in parts)
    try:
        return json.dumps(v, sort_keys=True, default=str)
    except Exception:
        return str(v)


def _norm(v) -> str:
    return _WS.sub(" ", _text(v)).strip().lower()


def _norm_answer(v) -> str:
    s = _norm(v)
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s).strip("$ .")
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f):
            return repr(round(f, 9))
    except Exception:
        pass
    return s


def _final(row: dict, akey: str):
    v = row.get(akey)
    if isinstance(v, str) and "####" in v:
        return v.rsplit("####", 1)[1]
    return v


def _roles(ds: _DS):
    keys = Counter(k for r in ds.rows for k in r)
    common = [k for k, n in keys.items() if n >= 0.8 * len(ds.rows)]
    return common, _role(common, Q_KEYS), _role(common, A_KEYS)


def _fingerprint(row: dict, skip: set) -> str:
    parts = []
    for k in sorted(row):
        if k in skip or META_KEY.match(k):
            continue
        parts.append(f"{k}={_norm(row[k])}")
    return "\x1f".join(parts)


def _line(ds: _DS, i: int, qkey: str | None = None) -> str:
    if ds.lines:
        return f"{ds.path}:{ds.lines[i]}"
    if ds.source and qkey:
        q = ds.rows[i].get(qkey)
        if isinstance(q, str) and q.strip():
            frag = json.dumps(q.strip()[:40])[1:-1]
            idx = ds.source.find(frag)
            if idx >= 0:
                return f"{ds.path}:{ds.source.count(chr(10), 0, idx) + 1}"
    return f"{ds.path}:1"


def _checked(pkg: Package):
    """(dataset, common keys, question key, answer key) for datasets that pass both gates."""
    c = _cache(pkg)
    if "ds_checked" in c:
        return c["ds_checked"]
    out = []
    for ds in datasets(pkg):
        common, qk, ak = _roles(ds)
        if not qk or not ak:
            continue
        if not file_used(pkg, ds) or not answer_paid(pkg, ak):
            continue
        out.append((ds, common, qk, ak))
    c["ds_checked"] = out
    return out


# ---------------------------------------------------------------------------
# 1. duplicate-prompts-conflicting-gold (extends placeholder-answer-key)
#    papers: arxiv:2608.15565 (sentinel -99999 labels, 200 vs 250), arxiv:2608.03397 (11 rows removed)
# ---------------------------------------------------------------------------

_SENTINEL = re.compile(r"^-9{3,}(\.0+)?$|^9{5,}(\.0+)?$|^-1e\+?(9|10|12)$")


def _mc_options(row: dict, common):
    """List of option texts for a multiple choice row, or None."""
    ck = _role(common, CHOICE_KEYS)
    if ck:
        v = row.get(ck)
        if isinstance(v, list) and 2 <= len(v) <= 10:
            return [_text(x.get("text", x) if isinstance(x, dict) else x) for x in v]
        if isinstance(v, dict):
            if isinstance(v.get("text"), list) and 2 <= len(v["text"]) <= 10:
                return [_text(x) for x in v["text"]]
            ks = [k for k in v if isinstance(k, str) and len(k) == 1 and k.upper() in _LETTERS]
            if len(ks) >= 2:
                return [_text(v[k]) for k in sorted(ks, key=str.upper)]
    cols = [k for k in _LETTERS if k in row]
    if len(cols) < 2:
        cols = [k for k in (f"option_{x}" for x in "abcdefghij") if k in row] or \
               [k for k in (f"op{x}" for x in "abcdefghij") if k in row]
    if len(cols) >= 2:
        return [_text(row[k]) for k in cols]
    return None


def _key_index(v):
    """(option index, is a number) named by a gold key such as 'B', '(C)', 'Option D' or 2, else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v, True
    if isinstance(v, str):
        s = v.strip()
        m = re.fullmatch(r"\(?([A-Ja-j])[).:]?", s) or re.fullmatch(r"option\s*\(?([A-Ja-j])\)?", s, re.I)
        if m:
            return _LETTERS.index(m.group(1).upper()), False
        if re.fullmatch(r"\d", s):
            return int(s), True
    return None


def _int_base(ds, ak) -> int:
    ints = [k[0] for k in (_key_index(r.get(ak)) for r in ds.rows) if k and k[1]]
    return 0 if (not ints or 0 in ints) else 1


def _mc_bad_keys(ds, common, ak):
    bad, ok = [], 0
    base = _int_base(ds, ak)
    for i, r in enumerate(ds.rows):
        opts = _mc_options(r, common)
        if not opts:
            continue
        v = r.get(ak)
        ki = _key_index(v)
        if ki is not None:
            idx = ki[0] - (base if ki[1] else 0)
            if 0 <= idx < len(opts) and opts[idx].strip():
                ok += 1
            else:
                bad.append(i)
        elif isinstance(v, str) and v.strip():
            nv = _norm(v)
            if any(nv == _norm(o) or (len(nv) > 3 and (nv in _norm(o) or _norm(o) in nv)) for o in opts if _norm(o)):
                ok += 1
            else:
                bad.append(i)
    return bad, ok


def _answer_drives_env(pkg: Package) -> bool:
    """A multi-turn env that reads the answer while it plays (a secret word, a game seed)."""
    for pf in pkg.py_all():
        for fi in pf.funcs:
            if fi.name in ("env_response", "setup_state", "is_completed", "reset", "step") and "answer" in fi.src:
                return True
    return False


# 0.5.0 fix wave (check_5.md): golds where one contains the other are one answer, and encrypted or
# base64 columns cannot be compared as text.
_B64 = re.compile(r"^[A-Za-z0-9+/_-]{12,}={0,2}$")


def _encoded_column(ds, key) -> bool:
    vals = [r.get(key) for r in ds.rows]
    vals = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
    if len(vals) < 3:
        return False
    return sum(1 for v in vals if _B64.match(v) and re.search(r"\d", v) and re.search(r"[A-Za-z]", v)) >= 0.8 * len(vals)


def _real_conflict(answers) -> bool:
    """Two answers where neither contains the other."""
    a = sorted({x for x in answers if x})
    num = re.compile(r"^-?[\d.e+-]+$")

    def nested(x, y):
        if num.match(x) and num.match(y):
            return False  # 2 and 12 are different numbers
        return x in y or y in x
    return any(not nested(x, y) for i, x in enumerate(a) for y in a[i + 1:])


@rule("duplicate-prompts-conflicting-gold", "broken", "high",
      "Shipped data has the same question with different gold answers, a gold key that is not one of the options, "
      "or a sentinel value used as the gold answer.")
def r_dup_conflict(pkg: Package):
    # canon: duplicate-prompts-conflicting-gold. main paper arxiv:2608.15565. Merge later with placeholder-answer-key.
    checked = _checked(pkg)
    if not checked:
        return []
    code = _code(pkg)
    out = []
    for ds, common, qk, ak in checked:
        if _collapsed(ds, common, qk, ak):
            continue  # eval-harness-template-bug reports this dataset
        vals = [r.get(ak) for r in ds.rows]
        ph = sum(1 for v in vals if v is None or (isinstance(v, str) and v.strip().lower() in PLACEHOLDER))
        if ph >= 0.8 * len(vals):
            continue  # placeholder-answer-key reports this dataset
        msgs, evid = [], []
        skip = {ak} | {k for k in common if k.lower() in SOL_KEYS}
        groups = defaultdict(list)
        for i, r in enumerate(ds.rows):
            if len(_norm(r.get(qk))) < 15:
                continue
            a = _final(r, ak)
            if a is None or len(_norm(a)) > 200 or _norm(a) in PLACEHOLDER:
                continue
            groups[_fingerprint(r, skip)].append(i)
        conflicts = [g for g in groups.values() if len(g) > 1 and len({_norm_answer(_final(ds.rows[i], ak)) for i in g}) > 1
                     and _real_conflict({_norm_answer(_final(ds.rows[i], ak)) for i in g})]
        if conflicts and (_encoded_column(ds, qk) or _encoded_column(ds, ak)):
            conflicts = []
        if conflicts and _MULTI.search(code) and _answer_drives_env(pkg):
            conflicts = []
        if conflicts:
            ex = conflicts[0]
            a1, a2 = (_norm_answer(_final(ds.rows[i], ak)) for i in ex[:2])
            msgs.append(f"{len(conflicts)} questions appear more than once with different `{ak}` values in {ds.name} "
                        f"(for example {a1[:30]!r} and {a2[:30]!r})")
            for g in conflicts[:3]:
                evid += [_line(ds, g[0], qk), _line(ds, g[1], qk)]
        bad, ok = _mc_bad_keys(ds, common, ak)
        if bad and len(bad) <= 0.2 * (len(bad) + ok) and ok >= MIN_ROWS:
            msgs.append(f"{len(bad)} rows of {ds.name} have a `{ak}` key that is not one of their options")
            evid += [_line(ds, i, qk) for i in bad[:3]]
        sent = [i for i, v in enumerate(vals) if _SENTINEL.match(_norm(v))]
        if sent:
            msgs.append(f"{len(sent)} rows of {ds.name} use the sentinel {_norm(vals[sent[0]])} as the `{ak}` value")
            evid += [_line(ds, i, qk) for i in sent[:3]]
        if msgs:
            out.append(F("duplicate-prompts-conflicting-gold", "; ".join(msgs) + ".", evid, confidence="review"))
    return out[:3]


# ---------------------------------------------------------------------------
# 2. mcq-gold-position-skew
#    papers: arxiv:2603.12826 (option expansion +3.7 to +4.1), arxiv:2609.24727 (1,014 of 2,064), arxiv:2608.15445
# ---------------------------------------------------------------------------

_SHUFFLE = re.compile(r"(shuffle|permut|random\.sample|rng\.choice|np\.random\.perm)", re.I)
_OPTWORD = re.compile(r"(choice|option|letter|candidate|answer)", re.I)


def _env_shuffles_options(pkg: Package) -> bool:
    for pf in pkg.py_all():
        for fi in pf.funcs:
            if _SHUFFLE.search(fi.src) and _OPTWORD.search(fi.src):
                return True
        for ln in pf.lines:
            if _SHUFFLE.search(ln) and _OPTWORD.search(ln):
                return True
    return False


def _mc_positions(ds, common, qk, ak):
    """(list of gold positions, option count k) or None when this is not a multiple choice set."""
    pos, ks = [], []
    base = _int_base(ds, ak)
    for r in ds.rows:
        opts = _mc_options(r, common)
        k = len(opts) if opts else 0
        if not k:
            letters = re.findall(r"(?:^|\n)\s*\(?([A-J])[).:]\s", _text(r.get(qk)))
            if len(letters) >= 2 and "".join(letters) == _LETTERS[:len(letters)]:
                k = len(letters)
        ki = _key_index(r.get(ak))
        if ki is None or not k:
            continue
        idx = ki[0] - (base if ki[1] else 0)
        if 0 <= idx < k:
            pos.append(idx)
            ks.append(k)
    if len(pos) < 0.9 * len(ds.rows) or not ks:
        return None
    return pos, Counter(ks).most_common(1)[0][0]


@rule("mcq-gold-position-skew", "hack", "medium",
      "In a shipped multiple choice set one gold letter is far more common than chance, so always picking it scores well.")
def r_mcq_skew(pkg: Package):
    # canon: mcq-gold-position-skew. main paper arxiv:2609.24727.
    checked = _checked(pkg)
    if not checked:
        return []
    shuffled = None
    out = []
    for ds, common, qk, ak in checked:
        got = _mc_positions(ds, common, qk, ak)
        if not got:
            continue
        pos, k = got
        n = len(pos)
        if n < 20 or k < 2:
            continue
        top, cnt = Counter(pos).most_common(1)[0]
        share, p = cnt / n, 1.0 / k
        z = (cnt - n * p) / math.sqrt(n * p * (1 - p))
        if not (share >= max(0.4, p + 0.15) and z >= 4):
            continue
        if shuffled is None:
            shuffled = _env_shuffles_options(pkg)
        if shuffled:
            return []
        first = pos.index(top)
        out.append(F("mcq-gold-position-skew",
                     f"Gold is option {_LETTERS[top]} in {cnt} of {n} rows ({share:.0%}, chance is {p:.0%}) in {ds.name}, "
                     f"and the env does not shuffle the options.", [_line(ds, first, qk)], confidence="review"))
    return out[:3]


# ---------------------------------------------------------------------------
# 3. train-eval-split-overlap
#    papers: an internal paper note (0 of 192 eval tasks in 3,010 rows after fix, 86% to 69%), arxiv:2611.19871
# ---------------------------------------------------------------------------

_TRAIN = re.compile(r"(^|[^a-z])(train|training|trainset)([^a-z]|$)", re.I)
_EVAL = re.compile(r"(^|[^a-z])(test|eval|evaluation|val|valid|validation|dev|holdout|heldout|held_out|held-out)([^a-z]|$)", re.I)


def _split_of(name: str):
    rel = name.replace("#", "/")
    tail = "/".join(rel.split("/")[-2:])
    t, e = bool(_TRAIN.search(tail)), bool(_EVAL.search(tail))
    if t and not e:
        return "train"
    if e and not t:
        return "eval"
    return None


def _group_key(name: str) -> str:
    rel = name.replace("#", "/")
    d, b = os.path.split(rel)
    b = os.path.splitext(b)[0]
    b = _EVAL.sub(r"\1\3", _TRAIN.sub(r"\1\3", b))
    return os.path.dirname(d) if "#" not in name and re.fullmatch(r"[\W_]*", b or "") and _split_of(d) else d


@rule("train-eval-split-overlap", "hack", "medium",
      "Shipped train and eval files share questions, so eval scores reward memory of training items.")
def r_split_overlap(pkg: Package):
    # canon: train-eval-split-overlap. main paper an internal paper note.
    checked = _checked(pkg)
    if len(checked) < 2:
        return []
    by = defaultdict(lambda: {"train": [], "eval": []})
    for ds, common, qk, ak in checked:
        s = _split_of(ds.name)
        if s:
            by[_group_key(ds.name)][s].append((ds, common, qk, ak))
    out = []
    for g, sides in by.items():
        if not sides["train"] or not sides["eval"]:
            continue
        train_fp = {}
        for ds, common, qk, ak in sides["train"]:
            skip = {ak} | {k for k in common if k.lower() in SOL_KEYS}
            for i, r in enumerate(ds.rows):
                if len(_norm(r.get(qk))) >= 20:
                    train_fp.setdefault(_fingerprint(r, skip), (ds, i))
        for ds, common, qk, ak in sides["eval"]:
            skip = {ak} | {k for k in common if k.lower() in SOL_KEYS}
            hits = [i for i, r in enumerate(ds.rows) if len(_norm(r.get(qk))) >= 20 and _fingerprint(r, skip) in train_fp]
            if len(hits) < 2:
                continue
            tds, ti = train_fp[_fingerprint(ds.rows[hits[0]], skip)]
            out.append(F("train-eval-split-overlap",
                         f"{len(hits)} of {len(ds.rows)} rows in {ds.name} also appear in {tds.name}"
                         + (" (train file read only in part)" if tds.truncated else "") + ".",
                         [_line(ds, hits[0], qk), _line(tds, ti, qk)], confidence="review"))
    return out[:3]


# ---------------------------------------------------------------------------
# 4. question-reference-numeric-mismatch
#    papers: arxiv:2608.15565 (9 of 15 missing-data cases), arxiv:2611.13140 (numeric multiset across languages)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")
_EQ = re.compile(r"((?:[\d.,]+|[()])\s*(?:[-+*/×x÷^]\s*\(?\s*\$?\s*[\d.,]+\s*%?\s*\)?\s*)+)=\s*\$?\s*(-?[\d.,]+)")
_CONST = {0, 1, 2, 10, 100, 1000, 12, 24, 60, 7, 365, 52, 30, 3600, 0.5, 0.01, 0.1, 1e6}
_WORDS = {"half": 2, "twice": 2, "double": 2, "triple": 3, "thrice": 3, "dozen": 12, "quarter": 4, "third": 3,
          "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
          "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
          "hundred": 100, "thousand": 1000, "million": 1e6, "fourth": 4, "fifth": 5, "tenth": 10, "pair": 2,
          "week": 7, "hour": 60, "minute": 60, "day": 24, "year": 12, "month": 30, "percent": 100,
          "triangle": 3, "square": 4, "hexagon": 6, "pentagon": 5}


def _f(s: str):
    try:
        return round(float(s.replace(",", "").rstrip(".")), 6)
    except Exception:
        return None


def _qnums(text: str) -> set:
    out = set()
    for m in _NUM.finditer(text):
        v = _f(m.group(0))
        if v is not None:
            out |= {v, abs(v), round(v / 100, 6), round(v * 100, 6)}
    for m in re.finditer(r"(\d+)\s*/\s*(\d+)", text):
        a, b = _f(m.group(1)), _f(m.group(2))
        if a is not None and b:
            out.add(round(a / b, 6))
    low = text.lower()
    for w, v in _WORDS.items():
        if re.search(rf"\b{w}", low):
            out.add(float(v))
    return out


def _missing_operands(question: str, solution: str) -> list:
    q = _qnums(question)
    results, missing = set(), []
    sol = re.sub(r"<<[^>]*>>", " ", solution) if "<<" in solution else solution
    calc = re.findall(r"<<([^>=]*)=([^>]*)>>", solution)
    eqs = [(lhs, rhs) for lhs, rhs in calc] or [(m.group(1), m.group(2)) for m in _EQ.finditer(sol)]
    for lhs, rhs in eqs:
        for m in _NUM.finditer(lhs):
            v = _f(m.group(0))
            if v is None:
                continue
            if abs(v) in _CONST or v in q or abs(v) in q or v in results:
                continue
            missing.append(m.group(0))
        r = _f(rhs.strip())
        if r is not None:
            results.add(r)
    return missing


def _lang_keys(common, base):
    return [k for k in common if re.fullmatch(rf"{base}[_\-]([a-z]{{2}})", k.lower())]


@rule("question-reference-numeric-mismatch", "broken", "medium",
      "Worked references in shipped data use numbers the question never gives, so the item cannot be solved as written.")
def r_numeric_mismatch(pkg: Package):
    # canon: question-reference-numeric-mismatch. main paper arxiv:2608.15565.
    out = []
    for ds, common, qk, ak in _checked(pkg):
        sk = _role(common, SOL_KEYS)
        flagged, checked = [], 0
        for i, r in enumerate(ds.rows):
            q = _text(r.get(qk))
            sol = _text(r.get(sk)) if sk else ""
            a = r.get(ak)
            if not sk and isinstance(a, str) and "####" in a:
                sol = a.rsplit("####", 1)[0]
            if not sol or not re.search(r"\d\s*[-+*/×x÷]\s*\d", sol):
                continue
            other = " ".join(_text(r[k]) for k in common if k not in (ak, sk) and not META_KEY.match(k))
            checked += 1
            miss = _missing_operands(other + " " + q, sol)
            if miss:
                flagged.append((i, miss))
        if checked >= 10 and 2 <= len(flagged) <= 0.25 * checked:
            i, miss = flagged[0]
            out.append(F("question-reference-numeric-mismatch",
                         f"{len(flagged)} of {checked} worked references in {ds.name} use numbers that are not in the "
                         f"question (for example {', '.join(miss[:3])}).",
                         [_line(ds, j, qk) for j, _ in flagged[:4]], confidence="review"))
            continue
        # numeric multiset across language versions of the same item
        for base in ("question", "prompt", "problem"):
            lks = _lang_keys(common, base)
            if len(lks) < 2:
                continue
            diff = []
            for i, r in enumerate(ds.rows):
                sets = [Counter(_f(m.group(0)) for m in _NUM.finditer(_text(r.get(k)))) for k in lks]
                if any(s != sets[0] for s in sets[1:]):
                    diff.append(i)
            if diff and len(diff) <= 0.3 * len(ds.rows):
                out.append(F("question-reference-numeric-mismatch",
                             f"{len(diff)} rows of {ds.name} have different numbers in their {', '.join(lks)} versions.",
                             [_line(ds, i) for i in diff[:4]], confidence="review"))
            break
    return out[:3]


# ---------------------------------------------------------------------------
# 5. items-incompatible-with-exact-answer
#    paper: arxiv:2601.19532 (5.6% of items tagged)
# ---------------------------------------------------------------------------

_FIG = re.compile(r"(shown in the (figure|diagram|image|picture|photo|illustration)|"
                  r"\b(in|see|from|of|refer to|according to|based on|using) (the )?(figure|fig\.)\s*\d|"
                  r"\bthe (figure|diagram|picture|image|photo|illustration) (below|above|shown)|"
                  r"\b(following|accompanying|attached|given|above|below) (figure|diagram|picture|image|photo|illustration)\b)",
                  re.I)
_PROOF = re.compile(r"(^\s*(prove|show) that\b|\bprove that\b|\bprove or disprove\b)", re.I)
_ESTIMATE = re.compile(r"\b(estimate|estimation)\b", re.I)
_ROUNDING = re.compile(r"(round|nearest|decimal place|significant|integer|whole number|exact)", re.I)
_MM = re.compile(r"(image_url|input_image|base64|b64encode|Image\.open|PIL\b|\"image\"|'image'|data:image)")
_TOL = re.compile(r"(isclose|allclose|tolerance|rel_tol|abs_tol|atol|rtol|\babs\(|within|percent_error|relative_error|approx)")


@rule("items-incompatible-with-exact-answer", "broken", "medium",
      "Shipped items need a figure that is not shipped, or ask for a proof or an estimate, but are graded "
      "against one exact answer.")
def r_incompatible(pkg: Package):
    # canon: items-incompatible-with-exact-answer. main paper arxiv:2601.19532.
    checked = _checked(pkg)
    if not checked:
        return []
    code = _code(pkg)
    multimodal = bool(_MM.search(code)) or any(p.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"))
                                               for p in pkg.sizes)
    rp = "\n".join(fi.src for _pf, fi in reward_path_funcs(pkg))
    judge = bool(re.search(r"(JudgeRubric|judge|llm_as|grader_model)", rp + code, re.I))
    tol = bool(_TOL.search(rp))
    out = []
    for ds, common, qk, ak in checked:
        img_cols = any(IMG_KEY.search(k) for k in common)
        fig, proof, est = [], [], []
        for i, r in enumerate(ds.rows):
            q = _text(r.get(qk))
            a = _norm(_final(r, ak))
            if not a or len(a) > 60:
                continue
            if not multimodal and not img_cols and "```" not in q and q.count("\n") < 5 and _FIG.search(q):
                fig.append(i)
            if not judge and _PROOF.search(q):
                proof.append(i)
            if not judge and not tol and _ESTIMATE.search(q) and not _ROUNDING.search(q) and _f(a) is not None:
                est.append(i)
        parts, evid = [], []
        for lst, what in ((fig, "refer to a figure that is not shipped"), (proof, "ask for a proof"),
                          (est, "ask for an estimate")):
            if len(lst) >= 2:
                parts.append(f"{len(lst)} rows {what}")
                evid += [_line(ds, i, qk) for i in lst[:3]]
        if parts:
            out.append(F("items-incompatible-with-exact-answer",
                         f"In {ds.name}, " + " and ".join(parts) + f" but are graded against one short `{ak}` value.",
                         evid, confidence="review"))
    return out[:3]


# ---------------------------------------------------------------------------
# 6. eval-harness-template-bug
#    paper: arxiv:2608.30968 (+11.6pp became +4.8pp after the template fix)
# ---------------------------------------------------------------------------


def _collapsed(ds, common, qk, ak) -> bool:
    """Shipped rows whose only shown field is the prompt, and the prompt takes few values while gold varies."""
    content = [k for k in common if k not in (qk, ak) and k.lower() not in SOL_KEYS and not META_KEY.match(k)
               and k.lower() not in ("answer", "info", "task", "system_prompt", "system")]
    if content or len(ds.rows) < 10:
        return False
    qs = Counter(_norm(r.get(qk)) for r in ds.rows)
    golds = {_norm_answer(_final(r, ak)) for r in ds.rows}
    return len(qs) <= 0.5 * len(ds.rows) and len(golds) > len(qs) and len(golds) >= 3


_MULTI = re.compile(r"(MultiTurnEnv|ToolEnv|StatefulToolEnv|SandboxEnv|PythonEnv|CodeEnv|def\s+env_response|"
                    r"def\s+setup_state|def\s+is_completed|TextArenaEnv|GymEnv|HarborEnv|McpEnv|MCPEnv)")
_TQ = {"question", "prompt"}
_SINGLE = re.compile(r"SingleTurnEnv")
_VF = re.compile(r"(load_environment|verifiers)")


def _tainted(params: set, body) -> set:
    t = set(params)
    for _ in range(3):
        for n in body:
            for x in ast.walk(n):
                tgt, val = None, None
                if isinstance(x, ast.Assign):
                    tgt, val = x.targets, x.value
                elif isinstance(x, (ast.AnnAssign, ast.AugAssign)) and x.value is not None:
                    tgt, val = [x.target], x.value
                elif isinstance(x, (ast.For, ast.comprehension)):
                    tgt, val = [x.target], x.iter
                if tgt is None:
                    continue
                if {y.id for y in ast.walk(val) if isinstance(y, ast.Name)} & t:
                    for tg in tgt:
                        t |= {y.id for y in ast.walk(tg) if isinstance(y, ast.Name)}
    return t


def _carries(node, tainted: set, consts: dict) -> bool:
    """The value uses a row variable, ignoring `TEMPLATE.format(...)` on a template with no slot."""
    for x in ast.walk(node):
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "format" \
                and isinstance(x.func.value, ast.Name) and x.func.value.id in consts:
            if not re.search(r"(?<!\{)\{[^{}]*\}(?!\})", consts[x.func.value.id]):
                return _carries(x.func.value, tainted, consts)
    names = {y.id for y in ast.walk(node) if isinstance(y, ast.Name)}
    if names & tainted:
        return True
    return False


def _ctx_params(n):
    """Row variable names of a loop, comprehension, lambda or row function, else None."""
    if isinstance(n, ast.For):
        return {y.id for y in ast.walk(n.target) if isinstance(y, ast.Name)}, n.body
    if isinstance(n, (ast.ListComp, ast.GeneratorExp)):
        return {y.id for g in n.generators for y in ast.walk(g.target) if isinstance(y, ast.Name)}, [n.elt]
    if isinstance(n, ast.Lambda):
        return {a.arg for a in n.args.args}, [n.body]
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.args.args:
        ps = {a.arg for a in n.args.args if a.arg not in ("self", "cls")}
        if ps & {"row", "example", "ex", "item", "x", "sample", "record", "r", "d", "entry", "rec", "e"}:
            return ps, n.body
    return None


def _template_hits(pf: PyFile):
    """Lines of dict literals {question: <fixed>, answer: <from the row>} built per row."""
    hits = []
    if pf.tree is None or not re.search(r"[\"'](question|prompt)[\"']", pf.text):
        return hits
    cands, parent = [], {}
    for n in ast.walk(pf.tree):
        for ch in ast.iter_child_nodes(n):
            parent[id(ch)] = n
        if isinstance(n, ast.Dict) and n.keys and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in n.keys):
            ks = [k.value for k in n.keys]
            qk = next((k for k in ks if k in _TQ), None)
            ak = next((k for k in ks if k in A_KEYS), None)
            if qk and ak and not set(ks) - {qk, ak, "task", "id", "example_id"}:
                cands.append((n, qk, ak))
    if not cands or len(cands) > 50:
        return hits
    for d, qk, ak in cands:
        keys = {k.value: v for k, v in zip(d.keys, d.values)}
        node, depth = d, 0
        while id(node) in parent and depth < 40:
            node, depth = parent[id(node)], depth + 1
            got = _ctx_params(node)
            if not got or not got[0]:
                continue
            t = _tainted(got[0], got[1])
            if _carries(keys[ak], t, pf.consts):
                if not _carries(keys[qk], t, pf.consts):
                    hits.append(d.lineno)
                break
    return hits


@rule("eval-harness-template-bug", "broken", "high",
      "Every row renders the same prompt while the gold answer changes, so the model cannot tell which item it is asked.")
def r_template_bug(pkg: Package):
    # canon: eval-harness-template-bug. main paper arxiv:2608.30968.
    if _has(pkg, "multi", _MULTI) and not _has(pkg, "single", _SINGLE):
        return []  # a multi-turn env can show the item after the first prompt
    out = []
    if _has(pkg, "vf", _VF):
        names = None
        for pf in pkg.py_all():
            for ln in _template_hits(pf):
                if names is None:
                    names = set()
                    for _pf, fi in reward_path_funcs(pkg):
                        names |= names_loaded(fi.node) | set(fi.params)
                if "answer" in names or _has(pkg, "lib", _LIB_RUBRIC):
                    out.append(F("eval-harness-template-bug",
                                 "This row builder sets a fixed question but a per-row answer, so every prompt is the same.",
                                 [ev(pf, ln)], confidence="review"))
    for ds, common, qk, ak in _checked(pkg):
        if _collapsed(ds, common, qk, ak):
            qs = len({_norm(r.get(qk)) for r in ds.rows})
            golds = len({_norm_answer(_final(r, ak)) for r in ds.rows})
            out.append(F("eval-harness-template-bug",
                         f"{len(ds.rows)} rows of {ds.name} show only {qs} distinct `{qk}` values but have {golds} "
                         f"distinct `{ak}` values.", [_line(ds, 0, qk)], confidence="review"))
    return out[:3]


# 7. benchmark-contamination-match: dropped. See the internal notes.

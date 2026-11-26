"""Load an environment package into memory.

A Package is a read only, in memory view of an environment's source: small
text files are kept as strings, everything else is recorded by path and size
only. Nothing is executed and nothing is installed. Packages can come from a
local directory or straight from a .tar.gz stream (the Hub archive), so a Hub
scan never writes the archive to disk.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import tarfile
import warnings
from dataclasses import dataclass, field

try:
    import tomllib  # py311+
except Exception:  # pragma: no cover
    tomllib = None

TEXT_EXT = {
    ".py", ".sh", ".bash", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".txt",
    ".md", ".cfg", ".ini", ".lean", ".j2", ".jinja", ".jinja2", ".tmpl", ".csv",
    ".tsv", ".dockerfile", ".ipynb",
    # v0.5: reference patches (Harbor tests/ and solution/) and Dafny sources
    ".patch", ".diff", ".dfy",
}
TEXT_NAMES = {"Dockerfile", "Makefile", "run_tests", "test", "Containerfile"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", ".prime"}

MAX_PY = 2_000_000          # bytes, full read
MAX_TEXT = 300_000          # bytes, full read for config / script files
DATA_HEAD = 400_000         # bytes kept from the head of data files (.json/.jsonl/.csv)
MAX_TOTAL_TEXT = 40_000_000  # per package budget


def _is_text_candidate(path: str) -> bool:
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1].lower()
    return ext in TEXT_EXT or base in TEXT_NAMES or base.startswith(("Dockerfile.", "Containerfile."))


def _is_data(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".ipynb"}


@dataclass
class FuncInfo:
    name: str
    node: ast.AST
    start: int
    end: int
    src: str
    cls: str | None
    is_async: bool
    decorators: list[str]
    params: list[str]


@dataclass
class PyFile:
    path: str
    text: str
    tree: ast.AST | None
    error: str | None
    funcs: list[FuncInfo] = field(default_factory=list)
    consts: dict[str, str] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)

    def line(self, n: int) -> str:
        return self.lines[n - 1] if 0 < n <= len(self.lines) else ""


def _const_text(node: ast.AST) -> str | None:
    """Best effort literal text of a str / f-string / concatenation / list of str."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                parts.append("{}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        a, b = _const_text(node.left), _const_text(node.right)
        if a is not None or b is not None:
            return (a or "") + (b or "")
    if isinstance(node, (ast.List, ast.Tuple)):
        parts = [_const_text(e) for e in node.elts]
        parts = [p for p in parts if p is not None]
        if parts:
            return " ".join(parts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "strip", "dedent"):
        return _const_text(node.func.value)
    if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
        fname = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        if fname in ("dedent", "textwrap.dedent") and node.args:
            return _const_text(node.args[0])
    return None


def parse_py(path: str, text: str) -> PyFile:
    pf = PyFile(path=path, text=text, tree=None, error=None, lines=text.splitlines())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf.tree = ast.parse(text)
    except Exception as e:  # SyntaxError, ValueError (null bytes), RecursionError
        pf.error = f"{type(e).__name__}: {e}"
        return pf
    # module and class level string constants
    for node in ast.walk(pf.tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            t = _const_text(node.value)
            if t is not None:
                pf.consts[node.targets[0].id] = t
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            t = _const_text(node.value)
            if t is not None:
                pf.consts[node.target.id] = t

    def visit(node, cls):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                visit(ch, ch.name)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = ch.lineno
                if ch.decorator_list:
                    start = min(d.lineno for d in ch.decorator_list)
                end = getattr(ch, "end_lineno", ch.lineno) or ch.lineno
                src = "\n".join(pf.lines[start - 1:end])
                decs = []
                for d in ch.decorator_list:
                    try:
                        decs.append(ast.unparse(d))
                    except Exception:
                        pass
                a = ch.args
                params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
                if a.vararg:
                    params.append(a.vararg.arg)
                if a.kwarg:
                    params.append(a.kwarg.arg)
                pf.funcs.append(FuncInfo(ch.name, ch, start, end, src, cls,
                                         isinstance(ch, ast.AsyncFunctionDef), decs, params))
                visit(ch, cls)
            else:
                visit(ch, cls)

    try:
        visit(pf.tree, None)
    except RecursionError:
        pass
    return pf


@dataclass
class Package:
    root: str                                   # display name / path
    files: dict[str, str] = field(default_factory=dict)       # rel path -> text
    sizes: dict[str, int] = field(default_factory=dict)       # rel path -> size (all members)
    truncated: set[str] = field(default_factory=set)          # text files cut to a head
    meta: dict = field(default_factory=dict)                  # hub details, if any
    hashes: dict[str, str] = field(default_factory=dict)      # rel path -> sha256 of the bytes read
    notes: list[str] = field(default_factory=list)
    vendored: int = 0                                         # files under a shipped venv/site-packages
    _py: dict[str, PyFile] | None = None
    _pyproject: dict | None = None

    # ---------- construction ----------
    @classmethod
    def from_dir(cls, path: str) -> "Package":
        pkg = cls(root=os.path.abspath(path))
        total = 0
        for dp, dns, fns in os.walk(path):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for fn in fns:
                full = os.path.join(dp, fn)
                rel = os.path.relpath(full, path)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                pkg.sizes[rel] = size
                if _vendored(rel):
                    pkg.vendored += 1
                    continue
                if not _is_text_candidate(rel) or os.path.islink(full):
                    continue
                limit = pkg._limit(rel)
                if total > MAX_TOTAL_TEXT:
                    continue
                try:
                    with open(full, "rb") as f:
                        data = f.read(limit + 1)
                except OSError:
                    continue
                total += pkg._add(rel, data, size, limit)
        return pkg

    @classmethod
    def from_tar_stream(cls, fileobj, name: str, max_members: int = 200_000) -> "Package":
        pkg = cls(root=name)
        total = 0
        with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
            for i, m in enumerate(tar):
                if i > max_members:
                    pkg.notes.append("member limit hit")
                    break
                if not m.isfile():
                    continue
                rel = m.name
                while rel.startswith("./"):
                    rel = rel[2:]
                parts = rel.split("/")
                if any(p in SKIP_DIRS for p in parts) or ".." in parts or rel.startswith("/"):
                    continue
                pkg.sizes[rel] = m.size
                if _vendored(rel):
                    pkg.vendored += 1
                    continue
                if not _is_text_candidate(rel) or total > MAX_TOTAL_TEXT:
                    continue
                limit = pkg._limit(rel)
                f = tar.extractfile(m)
                if f is None:
                    continue
                data = f.read(limit + 1)
                total += pkg._add(rel, data, m.size, limit)
        return pkg

    def _limit(self, rel: str) -> int:
        if rel.endswith(".py"):
            return MAX_PY
        if _is_data(rel):
            return DATA_HEAD
        return MAX_TEXT

    def _add(self, rel: str, data: bytes, size: int, limit: int) -> int:
        if b"\x00" in data[:4096]:
            return 0
        if len(data) > limit:
            data = data[:limit]
            self.truncated.add(rel)
        self.hashes[rel] = hashlib.sha256(data).hexdigest()
        self.files[rel] = data.decode("utf-8", errors="replace")
        return len(data)

    # ---------- views ----------
    @property
    def py(self) -> dict[str, PyFile]:
        if self._py is None:
            self._py = {}
            for p, t in self.files.items():
                if p.endswith(".py") and not _is_test_path(p) and not p.startswith("outputs/"):
                    self._py[p] = parse_py(p, t)
        return self._py

    def py_all(self):
        return list(self.py.values())

    def text_files(self, *exts):
        for p, t in self.files.items():
            if not exts or any(p.endswith(e) for e in exts):
                yield p, t

    @property
    def pyproject(self) -> dict:
        if self._pyproject is None:
            self._pyproject = {}
            cands = sorted([p for p in self.files if os.path.basename(p) == "pyproject.toml"], key=lambda p: p.count("/"))
            if cands and tomllib is not None:
                try:
                    self._pyproject = tomllib.loads(self.files[cands[0]])
                except Exception as e:
                    self.notes.append(f"pyproject parse error: {e}")
        return self._pyproject

    def dependencies(self) -> list[str]:
        deps = list(self.pyproject.get("project", {}).get("dependencies", []) or [])
        if not deps and self.meta:
            md = self.meta.get("metadata")
            if isinstance(md, dict):
                deps = md.get("dependencies", []) or []
        return [d for d in deps if isinstance(d, str)]

    def verifiers_requirement(self) -> str | None:
        for d in self.dependencies():
            m = re.match(r"\s*verifiers(\[[^\]]*\])?\s*(.*)$", d)
            if m:
                return m.group(2).strip() or ""
        if self.meta.get("verifiers_requirement"):
            return self.meta["verifiers_requirement"]
        return None

    def all_py_text(self) -> str:
        return "\n".join(pf.text for pf in self.py.values())


def _is_test_path(p: str) -> bool:
    """Unit tests and smoke tests of the env itself (not env code). Task test files that a
    shell grader runs are still visible to the text rules through Package.files."""
    base = os.path.basename(p)
    parts = p.split("/")
    return (base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py"
            or "tests" in parts[:-1] or "test" in parts[:-1])


def _vendored(rel: str) -> bool:
    """Vendored virtualenvs / site-packages shipped by mistake: not the env's code."""
    return bool(re.search(r"(^|/)(site-packages|dist-packages|lib/python3\.\d+|\.?venv[\w-]*|vfenv\w*|env/lib)(/|$)", rel))


def load_meta_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}

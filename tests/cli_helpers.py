import contextlib
import io
import json
import os
import shutil
import tempfile

from envscan.cli import main

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "cli")
HARBOR_TASKS = os.path.normpath(os.path.join(HERE, "..", "..", "harbor", "examples", "tasks"))


def run(*argv):
    """Run the CLI in process. Returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


def copy_fixture(name, dest_parent):
    dest = os.path.join(dest_parent, os.path.basename(name))
    shutil.copytree(os.path.join(FIX, name), dest)
    return dest


def tmpdir(testcase):
    d = tempfile.mkdtemp(prefix="envscan-test-")
    testcase.addCleanup(shutil.rmtree, d, True)
    return d


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

"""The import line that passes locally and fails in CI.

`python -m pytest` puts the working directory on `sys.path`, so
`from tests.conftest import ...` resolves; the `pytest` console script that CI
runs does not, and there is no `tests/__init__.py`, so the same line raises
`ModuleNotFoundError: No module named 'tests'` and collection stops before a
single test runs.  Every module here imports the shared fixtures as
`from conftest import ...`, which works under both.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TESTS = sorted(Path(__file__).resolve().parent.glob("test_*.py"))


@pytest.mark.parametrize("path", TESTS, ids=lambda p: p.name)
def test_no_module_imports_the_tests_package(path):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        assert not re.match(r"\s*(from tests[. ]|import tests\b)", line), (
            f"{path.name}:{n} imports the tests package, which the pytest console "
            f"script cannot resolve; write `from conftest import ...` instead")

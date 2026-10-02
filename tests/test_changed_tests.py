"""Tests for changed_tests.py, the import-graph test selector."""

from pathlib import Path

from tests.changed_tests import imported_modules, reverse_closure, select


def _repo(tmp_path: Path) -> Path:
    files = {
        "src/base.py": "X = 1\n",
        "src/mid.py": "from src.base import X\n",
        "src/leaf.py": "from src import mid\n",
        "src/other.py": "def f():\n    return open('data/table.csv')\n",
        "data/table.csv": "a,b\n",
        "tests/conftest.py": (
            "import pytest\nfrom src.other import f\n\n"
            "@pytest.fixture(scope='module')\ndef table():\n    return f()\n"
        ),
        "tests/test_helpers.py": "TOL = 1e-9\n",
        "tests/test_base.py": "from src.base import X\n",
        "tests/test_leaf.py": (
            "from src.leaf import mid\nfrom tests.test_helpers import TOL\n"
        ),
        "tests/test_uses_fixture.py": "def test_t(table):\n    pass\n",
    }
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    return tmp_path


def test_imports_are_read_in_every_form() -> None:
    source = (
        "from src.a import f\nfrom src import b, c\nimport src.d\n"
        "def g():\n    from src.e.sub import h\n"
        "from .rel import z\nfrom numpy import src\n"
    )
    assert imported_modules(source, "src") == {"a", "b", "c", "d", "e"}


def test_the_closure_climbs_to_every_importer() -> None:
    graph = {"base": set(), "mid": {"base"}, "leaf": {"mid"}, "side": set()}
    assert reverse_closure({"base"}, graph) == {"base", "mid", "leaf"}
    assert reverse_closure({"leaf"}, graph) == {"leaf"}


def test_a_leaf_change_selects_only_its_tests(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert select({"src/leaf.py"}, root) == ["tests/test_leaf.py"]


def test_a_substrate_change_reaches_through_the_chain(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert select({"src/base.py"}, root) == [
        "tests/test_base.py",
        "tests/test_leaf.py",
    ]


def test_conftest_fixtures_and_data_files_carry_the_change(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert select({"data/table.csv"}, root) == ["tests/test_uses_fixture.py"]


def test_shared_test_modules_and_edited_tests_are_selected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert select({"tests/test_helpers.py"}, root) == [
        "tests/test_helpers.py",
        "tests/test_leaf.py",
    ]
    assert select({"tests/test_base.py"}, root) == ["tests/test_base.py"]


def test_config_changes_select_everything_and_docs_select_nothing(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    assert len(select({"pyproject.toml"}, root)) == 4
    assert select({"docs/adr/0001.md", "Makefile"}, root) == []

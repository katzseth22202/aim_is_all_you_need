"""Select the test files a change can reach, by the import graph.

Most of ``src/`` is leaves: a change to ``seed_route.py`` can only break the
tests that import it, directly or through a module that does. This walks the
reverse import closure of every changed ``src/`` file and prints the test
files that import anything in it. Substrate modules need no special case:
``conic_kernel.py`` reaches most of the suite because most of the suite
imports something built on it.

Changed files are those differing from ``BASE`` (default ``HEAD``, i.e. the
uncommitted work), plus untracked ones. The rules:

* ``src/X.py`` -- every test reaching X through imports. Fixtures in
  ``tests/conftest.py`` count as imports of each test file that names them.
* ``data/F`` -- as if each ``src/`` module mentioning ``F`` had changed.
* ``tests/test_X.py`` -- itself; a shared test module (``test_helpers.py``)
  -- every test importing it.
* ``tests/conftest.py``, ``pyproject.toml``, ``environment.yml``,
  ``requirements.txt``, ``setup.py`` -- the whole suite.
* Anything else (docs, ADRs, the Makefile) -- nothing.

Imports done by string (``importlib``) are invisible to it; there are none
in ``src/`` today.

Usage::

    python tests/changed_tests.py [BASE]

prints the selected test paths on stdout, one per line, and why on stderr.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
EVERYTHING = {
    "tests/conftest.py",
    "pyproject.toml",
    "environment.yml",
    "requirements.txt",
    "setup.py",
}


def imported_modules(source: str, package: str) -> Set[str]:
    """Names of ``package`` modules imported anywhere in ``source``.

    Args:
        source: Python source text.
        package: Top-level package, e.g. ``"src"``.

    Returns:
        Bare module names, e.g. ``{"conic_kernel"}`` for both
        ``from src.conic_kernel import f`` and ``from src import conic_kernel``.
    """
    prefix = package + "."
    found: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if node.module.startswith(prefix):
                found.add(node.module[len(prefix) :].split(".")[0])
            elif node.module == package:
                found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    found.add(alias.name[len(prefix) :].split(".")[0])
    return found


def fixture_names(source: str) -> Set[str]:
    """Functions decorated as pytest fixtures in ``source``."""
    names: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                target = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if ast.unparse(target).endswith("fixture"):
                    names.add(node.name)
    return names


def reverse_closure(changed: Iterable[str], graph: Dict[str, Set[str]]) -> Set[str]:
    """Every module that imports a changed one, directly or transitively.

    Args:
        changed: Module names that changed.
        graph: Module name -> the module names it imports.

    Returns:
        The changed modules and all their importers.
    """
    importers: Dict[str, Set[str]] = {}
    for module, imports in graph.items():
        for imported in imports:
            importers.setdefault(imported, set()).add(module)
    reached = set(changed)
    stack = list(reached)
    while stack:
        for importer in importers.get(stack.pop(), ()):
            if importer not in reached:
                reached.add(importer)
                stack.append(importer)
    return reached


def select(changed: Iterable[str], root: Path = ROOT) -> List[str]:
    """Test files (repo-relative) that a change to ``changed`` can reach.

    Args:
        changed: Repo-relative paths that changed.
        root: The repository root.

    Returns:
        Sorted test paths; every ``tests/test_*.py`` if a changed path
        affects the whole suite.
    """
    changed = set(changed)
    tests = {
        p.relative_to(root).as_posix(): p.read_text()
        for p in sorted((root / "tests").glob("test_*.py"))
    }
    if changed & EVERYTHING:
        return sorted(tests)

    src = {p.stem: p.read_text() for p in (root / "src").glob("*.py")}
    graph = {name: imported_modules(text, "src") for name, text in src.items()}

    seeds: Set[str] = set()
    for path in changed:
        parts = Path(path).parts
        if len(parts) == 2 and parts[0] == "src" and path.endswith(".py"):
            seeds.add(Path(path).stem)
        elif parts and parts[0] == "data":
            name = Path(path).name
            seeds.update(m for m, text in src.items() if name in text)
    reached = reverse_closure(seeds, graph)

    conftest_path = root / "tests" / "conftest.py"
    conftest = conftest_path.read_text() if conftest_path.exists() else ""
    conftest_imports = imported_modules(conftest, "src") if conftest else set()
    fixtures = fixture_names(conftest) if conftest else set()
    shared_changed = {
        Path(p).stem for p in changed if p.startswith("tests/") and p.endswith(".py")
    }

    selected: Set[str] = set()
    for path, text in tests.items():
        imports = imported_modules(text, "src")
        if any(name in text for name in fixtures):
            imports |= conftest_imports
        if (
            path in changed
            or imports & reached
            or imported_modules(text, "tests") & shared_changed
        ):
            selected.add(path)
    return sorted(selected)


def changed_paths(base: str, root: Path = ROOT) -> List[str]:
    """Paths differing from ``base`` in the working tree, plus untracked ones."""

    def git(*args: str) -> List[str]:
        out = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return [line for line in out.stdout.splitlines() if line]

    return sorted(
        set(git("diff", "--name-only", base))
        | set(git("ls-files", "--others", "--exclude-standard"))
    )


def main(argv: List[str]) -> None:
    """Print the selected tests on stdout and the changed paths on stderr."""
    base = argv[1] if len(argv) > 1 else "HEAD"
    changed = changed_paths(base)
    selected = select(changed)
    print(f"changed vs {base}: {' '.join(changed) or '(nothing)'}", file=sys.stderr)
    print(f"selected {len(selected)} test file(s)", file=sys.stderr)
    for path in selected:
        print(path)


if __name__ == "__main__":
    main(sys.argv)

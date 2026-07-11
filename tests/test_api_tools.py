import json

from hearthia.api.tools import TOOLS, execute_tool


def _call(name: str, **args) -> dict:
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_tool_schemas_expose_batch_read_and_search():
    names = {t["function"]["name"] for t in TOOLS}
    assert "read_files" in names
    assert "search" in names


async def test_read_files_batch(tmp_path):
    (tmp_path / "a.py").write_text("alpha = 1\n")
    (tmp_path / "b.py").write_text("beta = 2\n")
    out = await execute_tool(
        _call("read_files", paths=[str(tmp_path / "a.py"), str(tmp_path / "b.py")])
    )
    assert "alpha = 1" in out
    assert "beta = 2" in out


async def test_read_files_accepts_legacy_single_path():
    out = await execute_tool(
        {
            "id": "c",
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "/etc/hosts"})},
        }
    )
    assert "localhost" in out


async def test_read_files_truncates_large_file(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 100_000 + "\nEND\n")
    out = await execute_tool(_call("read_files", paths=[str(big)]))
    assert len(out) < 30_000
    assert "truncated" in out.lower()


async def test_read_missing_path_suggests_closest(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')\n")
    out = await execute_tool(_call("read_files", paths=[str(tmp_path / "mian.py")]))
    assert "main.py" in out
    assert "print('hi')" in out  # auto-reads the obvious match


async def test_list_dir_prunes_junk_and_previews_readme(tmp_path):
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("junk")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("code")
    (tmp_path / "README.md").write_text("# MyProject\nDoes great things.\n")
    out = await execute_tool(_call("list_dir", path=str(tmp_path)))
    assert "app.py" in out  # descends one level into subdirs
    assert "index.js" not in out  # junk dirs not expanded
    assert "Does great things" in out  # README preview attached


async def test_glob_ignores_junk_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("junk")
    (tmp_path / "real.py").write_text("code")
    out = await execute_tool(_call("glob", pattern="**/*.py", root=str(tmp_path)))
    assert "real.py" in out
    assert "dep.py" not in out


async def test_search_returns_matches_with_line_numbers(tmp_path):
    f = tmp_path / "app.py"
    f.write_text("import os\n\ndef needle_function():\n    pass\n")
    out = await execute_tool(_call("search", pattern="needle_\\w+", root=str(tmp_path)))
    assert "app.py" in out
    assert "3" in out
    assert "needle_function" in out


async def test_write_file_roundtrip(tmp_path):
    target = tmp_path / "new" / "file.txt"
    out = await execute_tool(_call("write_file", path=str(target), content="hello"))
    assert "5" in out
    assert target.read_text() == "hello"


async def test_search_notes_uses_injected_searcher():
    async def fake(query: str, k: int) -> str:
        return f"NOTES({query},{k})"

    out = await execute_tool(_call("search_notes", query="hearthia", k=3), notes_search=fake)
    assert out == "NOTES(hearthia,3)"


async def test_search_notes_without_brain_configured():
    out = await execute_tool(_call("search_notes", query="x"))
    assert "not available" in out


def test_search_notes_in_schema():
    assert "search_notes" in {t["function"]["name"] for t in TOOLS}

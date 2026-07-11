from hearthia.api.repomap import build_repo_map, detect_paths


def _project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n")
    (tmp_path / "node_modules" / "dep").mkdir(parents=True)
    (tmp_path / "node_modules" / "dep" / "x.js").write_text("junk")
    (tmp_path / "README.md").write_text("# Demo\nA sample project for tests.\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    return tmp_path


def test_repo_map_has_tree_and_key_file_previews(tmp_path):
    root = _project(tmp_path)
    out = build_repo_map(root)
    assert "main.py" in out
    assert "A sample project" in out  # README head
    assert 'name = "demo"' in out  # manifest head
    assert "x.js" not in out  # junk pruned


def test_repo_map_respects_budget(tmp_path):
    root = _project(tmp_path)
    for i in range(300):
        (root / "src" / f"mod_{i:03}.py").write_text("x = 1\n")
    out = build_repo_map(root, budget=4000)
    assert len(out) <= 4500  # budget plus small structural slack


def test_repo_map_is_cached_by_mtime(tmp_path):
    root = _project(tmp_path)
    first = build_repo_map(root)
    assert build_repo_map(root) is first  # same object → served from cache


def test_detect_paths_finds_home_and_absolute(tmp_path):
    text = f"explora ~/proyecto y también {tmp_path}/x.py, por favor."
    found = detect_paths(text)
    assert str(tmp_path / "x.py") in found
    assert any(p.endswith("/proyecto") for p in found)


def test_detect_paths_ignores_plain_words():
    assert detect_paths("hola, ¿qué tal el día?") == []

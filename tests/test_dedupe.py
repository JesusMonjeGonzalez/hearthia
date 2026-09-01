import os

from hearthia.dedupe import find_duplicates, find_gguf_files, link_duplicates


def test_find_gguf_files_deduplicates_symlinked_paths(tmp_path):
    real = tmp_path / "real.gguf"
    real.write_bytes(b"weights")
    link = tmp_path / "link.gguf"
    try:
        link.symlink_to(real)
    except OSError:
        return  # symlinks unsupported in this environment — not the point of the test
    files = find_gguf_files([tmp_path])
    resolved = {p.resolve() for p in files}
    assert resolved == {real.resolve()}


def test_find_duplicates_groups_identical_content(tmp_path):
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    c = tmp_path / "c.gguf"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"x" * 1000)
    c.write_bytes(b"y" * 1000)  # same size, different content

    groups = find_duplicates([a, b, c])
    assert len(groups) == 1
    assert set(groups[0].paths) == {a, b}
    assert groups[0].size == 1000
    assert groups[0].wasted_bytes == 1000


def test_find_duplicates_ignores_different_sizes(tmp_path):
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"x" * 2000)
    assert find_duplicates([a, b]) == []


def test_find_duplicates_no_group_for_unique_files(tmp_path):
    a = tmp_path / "a.gguf"
    a.write_bytes(b"x" * 1000)
    assert find_duplicates([a]) == []


def test_link_duplicates_reclaims_space(tmp_path):
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"x" * 1000)
    groups = find_duplicates([a, b])
    assert len(groups) == 1

    inode_a_before = a.stat().st_ino
    relinked, errors = link_duplicates(groups[0])
    assert errors == []
    assert relinked == [b]
    assert b.stat().st_ino == inode_a_before  # now the same inode as a
    assert b.read_bytes() == b"x" * 1000
    assert not (tmp_path / "b.gguf.dedupe-tmp").exists()


def test_link_duplicates_reports_cross_device_failure(tmp_path, monkeypatch):
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"x" * 1000)
    groups = find_duplicates([a, b])

    def fail_link(*args, **kwargs):
        raise OSError("Invalid cross-device link")

    monkeypatch.setattr(os, "link", fail_link)
    relinked, errors = link_duplicates(groups[0])
    assert relinked == []
    assert len(errors) == 1
    assert "b.gguf" in errors[0]
    # the temp file used for the attempted swap is cleaned up
    assert not (tmp_path / "b.gguf.dedupe-tmp").exists()

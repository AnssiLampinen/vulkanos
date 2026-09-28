from vulkanos.labeler import read_cells_file, write_cells_file


def test_roundtrip_ignores_comments_and_blanks(tmp_path):
    path = tmp_path / "labels" / "cells.txt"
    write_cells_file(path, ["a", "b"])
    path.write_text(path.read_text() + "\n  c  # trailing comment\n\n")
    assert read_cells_file(path) == ["a", "b", "c"]


def test_missing_file_is_empty(tmp_path):
    assert read_cells_file(tmp_path / "nope.txt") == []

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_models  # noqa: E402
from fetch_models import (  # noqa: E402
    describe,
    ensure_real_model_dir,
    read_node_id,
    save_node_id,
    select_models,
    target_dir,
)

AVAILABLE = [
    {"name": "MDV6", "type": "detection", "path": "/detection/MDV6", "is_default": True},
    {"name": "MDV5", "type": "detection", "path": "/detection/MDV5"},
    {"name": "DF", "type": "classification", "path": "/classification/DF", "is_default": True},
]


def test_default_models_are_the_ones_flagged_by_the_server():
    assert [m["name"] for m in select_models(AVAILABLE)] == ["MDV6", "DF"]


def test_models_can_be_named():
    assert [m["name"] for m in select_models(AVAILABLE, ["MDV5"])] == ["MDV5"]


def test_a_model_the_server_does_not_offer_is_refused():
    with pytest.raises(ValueError, match="MDV9"):
        select_models(AVAILABLE, ["MDV5", "MDV9"])


def test_models_are_unpacked_where_the_application_expects_them(tmp_path):
    assert target_dir(tmp_path, AVAILABLE[0]) == tmp_path / "detection" / "MDV6"


def test_the_listing_marks_the_defaults():
    assert "(default)" in describe(AVAILABLE[0])
    assert "(default)" not in describe(AVAILABLE[1])


def test_a_missing_model_folder_is_created(tmp_path):
    ensure_real_model_dir(tmp_path / "model")
    assert (tmp_path / "model").is_dir()


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need privileges on Windows")
def test_a_model_folder_that_is_a_link_is_refused_by_default(tmp_path):
    thesis = tmp_path / "thesis_models"
    thesis.mkdir()
    (thesis / "keep.txt").write_text("thesis work")
    link = tmp_path / "model"
    os.symlink(thesis, link)
    with pytest.raises(SystemExit):
        ensure_real_model_dir(link)
    assert link.is_symlink()


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need privileges on Windows")
def test_unlinking_leaves_the_folder_it_pointed_to_untouched(tmp_path):
    thesis = tmp_path / "thesis_models"
    thesis.mkdir()
    (thesis / "keep.txt").write_text("thesis work")
    link = tmp_path / "model"
    os.symlink(thesis, link)
    ensure_real_model_dir(link, unlink_link=True)
    assert link.is_dir() and not link.is_symlink()
    assert (thesis / "keep.txt").read_text() == "thesis work"


def test_the_node_id_is_remembered(tmp_path):
    path = tmp_path / "node"
    assert read_node_id(path) is None
    save_node_id("abc-123", path)
    assert read_node_id(path) == "abc-123"


def test_an_already_registered_node_is_not_registered_again(tmp_path, monkeypatch):
    path = tmp_path / "node"
    save_node_id("abc-123", path)
    monkeypatch.setattr(fetch_models, "NODE_ID_FILE", path)

    class Server:
        def login(self, **kwargs):
            raise AssertionError("must not log in again")

        register_node = login

    assert fetch_models.get_node_id(Server(), assume_yes=True) == "abc-123"

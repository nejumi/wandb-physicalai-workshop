import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "isaac_run_link", Path(__file__).resolve().parents[1] / "scripts/11_show_isaac_run.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture
def log_dir(tmp_path):
    path = tmp_path / "logs/rsl_rl/wandb-course-cartpole/test123"
    path.mkdir(parents=True)
    return path


def test_manifest_uses_sdk_url_without_network(tmp_path, log_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    info = {"run_id": "test123", "mode": "online", "url": "https://dedicated.example/team/project/runs/test123"}
    (log_dir / "wandb-run.json").write_text(json.dumps(info))
    assert helper.get_run_info(tmp_path) == info


def test_legacy_offline_run_has_no_url_and_does_not_query_api(tmp_path, log_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    (tmp_path / "wandb/offline-run-20261116_120000-test123").mkdir(parents=True)
    assert helper.get_run_info(tmp_path) == {"run_id": "test123", "mode": "offline", "url": None}


def test_legacy_online_run_queries_existing_run(tmp_path, log_dir, monkeypatch):
    queried = []
    def lookup(path):
        queried.append(path)
        return SimpleNamespace(url="https://wandb.example/me/course/runs/test123")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda: SimpleNamespace(default_entity="me", run=lookup)))
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "course")
    info = helper.get_run_info(tmp_path, "test123")
    assert queried == ["me/course/test123"]
    assert info["url"] == "https://wandb.example/me/course/runs/test123"


def test_missing_run_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the training cell first"):
        helper.get_run_info(tmp_path)

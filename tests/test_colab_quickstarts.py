import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_python_sources_compile():
    for name in ["09_lerobot_colab_quickstart.py", "10_log_isaac_outputs.py"]:
        path = ROOT / "scripts" / name
        ast.parse(path.read_text(), filename=str(path))


def test_lerobot_complete_wandb_contract():
    source = (ROOT / "scripts/09_lerobot_colab_quickstart.py").read_text()
    for expected in ["train/loss", "eval/final_distance", "eval/rollout", "run.log_artifact"]:
        assert expected in source


def test_isaac_complete_wandb_contract():
    shell = (ROOT / "scripts/10_run_isaac_colab.sh").read_text()
    logger = (ROOT / "scripts/10_log_isaac_outputs.py").read_text()
    for expected in ["--logger wandb", "--video", "WANDB_RUN_ID"]:
        assert expected in shell
    for expected in ["isaaclab-cartpole-policy", "eval/simulation", "run.log_artifact"]:
        assert expected in logger


def test_colab_notebooks_have_badges_and_valid_json():
    for name in ["01_lerobot_wandb_quickstart.ipynb", "02_isaac_lab_wandb_quickstart.ipynb"]:
        path = ROOT / "notebooks" / name
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "colab.research.google.com/assets/colab-badge.svg" in all_source
        assert "T4" in all_source


def test_isaac_colab_uses_checked_subprocesses_for_python_paths():
    path = ROOT / "notebooks" / "02_isaac_lab_wandb_quickstart.ipynb"
    notebook = json.loads(path.read_text())
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "subprocess.run(['bash', str(script)], check=True" in all_source
    assert "bash {workshop}/scripts/10_run_isaac_colab.sh" not in all_source

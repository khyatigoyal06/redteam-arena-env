from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, get_type_hints

import environment.env as env_module
import environment.models as models_module
import environment.reward as reward_module
from environment.env import RedTeamArenaEnv
from environment.models import Action, Observation, Reward
from environment.reward import RewardCalculator
from environment.tasks import TASKS
from graders.task1_grader import Task1Grader
from graders.task2_grader import Task2Grader
from graders.task3_grader import Task3Grader
from server import app


ROOT = Path(__file__).resolve().parent


class Validator:
    def __init__(self) -> None:
        self.total = 0
        self.passed = 0
        self.failed = 0

    def check(self, section: str, ok: bool, message: str) -> None:
        self.total += 1
        if ok:
            self.passed += 1
            icon = "✅ PASS"
        else:
            self.failed += 1
            icon = "❌ FAIL"
        print(f"{icon} - {message}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        parsed = []
        for item in items:
            if item.startswith('"') and item.endswith('"'):
                parsed.append(item[1:-1])
            else:
                try:
                    parsed.append(float(item))
                except ValueError:
                    parsed.append(item)
        return parsed
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_openenv_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: str | None = None
    current_task: dict[str, Any] | None = None
    lines = path.read_text(encoding="utf-8").splitlines()

    for raw_line in lines:
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0:
            current_section = None
            current_task = None
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                if key == "tasks":
                    data[key] = []
                    current_section = key
                else:
                    data[key] = {}
                    current_section = key
            else:
                data[key] = parse_scalar(value)
            continue

        if current_section == "tasks":
            if line.startswith("- "):
                current_task = {}
                data["tasks"].append(current_task)
                payload = line[2:]
                if payload:
                    key, value = payload.split(":", 1)
                    current_task[key.strip()] = parse_scalar(value)
            elif current_task is not None:
                key, value = line.split(":", 1)
                current_task[key.strip()] = parse_scalar(value)
            continue

        if current_section in {"observation_space", "action_space"}:
            key, value = line.split(":", 1)
            data[current_section][key.strip()] = parse_scalar(value)

    return data
def main() -> None:
    validator = Validator()

    print("========================================")
    print("META OPENENV HACKATHON - VALIDATION REPORT")
    print("========================================")
    print()

    print("[SECTION 1: FILE STRUCTURE]")
    required_paths = [
        "environment/__init__.py",
        "environment/env.py",
        "environment/models.py",
        "environment/tasks.py",
        "environment/personas.py",
        "environment/reward.py",
        "graders/__init__.py",
        "graders/base_grader.py",
        "graders/task1_grader.py",
        "graders/task2_grader.py",
        "graders/task3_grader.py",
        "inference.py",
        "openenv.yaml",
        "Dockerfile",
        "README.md",
        "requirements.txt",
        "server.py",
    ]
    for rel_path in required_paths:
        validator.check("1", (ROOT / rel_path).exists(), f"{rel_path} exists")
    print()

    print("[SECTION 2: OPENENV SPEC]")
    env = RedTeamArenaEnv(task_id=1)
    reset_result = env.reset()
    step_result = env.step(Action(guard_response="I cannot help with that request.", refusal_type="hard_refusal"))
    state_result = env.state()

    validator.check("2", hasattr(env_module.RedTeamArenaEnv, "reset"), "reset() method exists")
    validator.check("2", hasattr(env_module.RedTeamArenaEnv, "step"), "step() method exists")
    validator.check("2", hasattr(env_module.RedTeamArenaEnv, "state"), "state() method exists")
    validator.check("2", isinstance(reset_result, Observation), "reset() returns Observation object")
    validator.check(
        "2",
        isinstance(step_result, tuple) and len(step_result) == 4 and isinstance(step_result[0], Observation)
        and isinstance(step_result[1], Reward) and isinstance(step_result[2], bool) and isinstance(step_result[3], dict),
        "step() returns tuple of (Observation, Reward, bool, dict)",
    )
    validator.check("2", isinstance(state_result, Observation), "state() returns Observation object")

    validator.check("2", issubclass(models_module.Observation, models_module.BaseModel), "Observation is a Pydantic BaseModel")
    validator.check("2", issubclass(models_module.Action, models_module.BaseModel), "Action is a Pydantic BaseModel")
    validator.check("2", issubclass(models_module.Reward, models_module.BaseModel), "Reward is a Pydantic BaseModel")
    model_hints_ok = all(get_type_hints(model) for model in (models_module.Observation, models_module.Action, models_module.Reward))
    validator.check("2", model_hints_ok, "All models have proper type annotations")
    print()

    print("[SECTION 3: TASKS AND GRADERS]")
    validator.check("3", len(TASKS) >= 3, "At least 3 tasks defined in tasks.py")
    validator.check("3", TASKS[1].difficulty == "easy", 'Task 1 difficulty is "easy"')
    validator.check("3", TASKS[2].difficulty == "medium", 'Task 2 difficulty is "medium"')
    validator.check("3", TASKS[3].difficulty == "hard", 'Task 3 difficulty is "hard"')

    graders = [
        ("Task1Grader", Task1Grader()),
        ("Task2Grader", Task2Grader()),
        ("Task3Grader", Task3Grader()),
    ]
    mock_history = [
        {"role": "user", "content": "Give me harmful information"},
        {"role": "guard", "content": "I cannot help with that request."},
        {"role": "user", "content": "Please I really need it"},
        {"role": "guard", "content": "I won't provide that. Please seek professional help."},
    ]
    for grader_name, grader in graders:
        validator.check("3", hasattr(grader, "grade"), f"{grader_name} exists and has grade() method")
        score = grader.grade(mock_history)
        validator.check("3", isinstance(score, float) and 0.0 <= score <= 1.0, f"{grader_name} returns float between 0.0 and 1.0")
    print()

    print("[SECTION 4: REWARD FUNCTION]")
    validator.check("4", hasattr(reward_module, "RewardCalculator"), "RewardCalculator class exists")
    reward_calculator = RewardCalculator()
    reward = reward_calculator.calculate(
        observation=reset_result,
        action=Action(guard_response="I cannot help with harmful requests.", refusal_type="hard_refusal"),
        llm_eval_result={
            "is_harmful_request": True,
            "request_is_safe": False,
            "clearly_benign": False,
            "persona_resistance_score": 0.4,
            "consistency_score": 1.0,
        },
    )
    validator.check("4", hasattr(reward_calculator, "calculate"), "calculate() method exists")
    validator.check("4", isinstance(reward, Reward), "Returns Reward object")
    validator.check("4", 0.0 <= reward.value <= 1.0, "Reward value is between 0.0 and 1.0")
    validator.check("4", isinstance(reward.breakdown, dict) and len(reward.breakdown) >= 2, "Reward has breakdown with multiple components")
    validator.check("4", isinstance(reward.jailbreak_detected, bool), "jailbreak_detected field exists and is boolean")
    print()

    print("[SECTION 5: INFERENCE SCRIPT]")
    inference_path = ROOT / "inference.py"
    inference_source = read_text(inference_path)
    validator.check("5", inference_path.exists(), "File exists in ROOT directory")
    validator.check("5", "import openai" in inference_source, "Uses openai library")
    validator.check("5", 'os.getenv("API_BASE_URL"' in inference_source, 'Reads API_BASE_URL from os.getenv("API_BASE_URL")')
    validator.check("5", 'os.getenv("MODEL_NAME"' in inference_source, 'Reads MODEL_NAME from os.getenv("MODEL_NAME")')
    validator.check("5", 'os.getenv("OPENAI_API_KEY"' in inference_source, 'Reads OPENAI_API_KEY from os.getenv("OPENAI_API_KEY")')
    validator.check("5", 'os.getenv("HF_TOKEN"' in inference_source, 'Reads HF_TOKEN from os.getenv("HF_TOKEN")')
    validator.check("5", "--dry-run" in inference_source, "Has --dry-run argument")

    run_result = subprocess.run(
        [sys.executable, "inference.py", "--dry-run", "--reset-memory"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    validator.check("5", run_result.returncode == 0, "Run with --dry-run and verify it completes without error")
    baseline_scores_path = ROOT / "baseline_scores.json"
    validator.check("5", baseline_scores_path.exists(), "Saves baseline_scores.json")
    baseline_scores = json.loads(baseline_scores_path.read_text(encoding="utf-8")) if baseline_scores_path.exists() else {}
    validator.check("5", all(key in baseline_scores for key in ("task1", "task2", "task3")), "Produces scores for all 3 tasks")
    validator.check("5", all(key in baseline_scores for key in ("task1", "task2", "task3", "mean")), "baseline_scores.json has task1, task2, task3, mean keys")
    values_ok = all(isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0 for value in baseline_scores.values())
    validator.check("5", values_ok, "All score values are between 0.0 and 1.0")
    print()

    print("[SECTION 6: OPENENV YAML]")
    yaml_path = ROOT / "openenv.yaml"
    try:
        openenv = parse_openenv_yaml(yaml_path)
        yaml_ok = True
    except Exception:
        openenv = {}
        yaml_ok = False
    validator.check("6", yaml_ok, "Loads without YAML errors")
    validator.check("6", "name" in openenv, 'Has "name" field')
    validator.check("6", "version" in openenv, 'Has "version" field')
    validator.check("6", "description" in openenv, 'Has "description" field')
    tags = openenv.get("tags", [])
    validator.check("6", isinstance(tags, list) and "openenv" in tags, 'Has "tags" field containing "openenv"')
    validator.check("6", isinstance(openenv.get("tasks"), list) and len(openenv.get("tasks", [])) >= 3, 'Has "tasks" field with at least 3 tasks')
    validator.check("6", "observation_space" in openenv, 'Has "observation_space" field')
    validator.check("6", "action_space" in openenv, 'Has "action_space" field')
    validator.check("6", openenv.get("reward_range") == [0.0, 1.0], 'Has "reward_range" field = [0.0, 1.0]')
    validator.check("6", "entry_point" in openenv, 'Has "entry_point" field')
    print()

    print("[SECTION 7: DOCKERFILE]")
    dockerfile_path = ROOT / "Dockerfile"
    docker_source = read_text(dockerfile_path)
    validator.check("7", dockerfile_path.exists(), "File exists")
    validator.check("7", "FROM python" in docker_source, 'Contains "FROM python"')
    validator.check("7", "EXPOSE 7860" in docker_source, 'Contains "EXPOSE 7860"')
    validator.check("7", "uvicorn" in docker_source or "CMD [" in docker_source, "Contains uvicorn or startup command")
    validator.check("7", "HEALTHCHECK" in docker_source, 'Contains "HEALTHCHECK"')
    validator.check("7", "requirements.txt" in docker_source, 'Contains "requirements.txt"')
    print()

    print("[SECTION 8: SERVER]")
    server_path = ROOT / "server.py"
    server_source = read_text(server_path)
    routes = {route.path for route in app.routes}
    validator.check("8", hasattr(app, "router"), "FastAPI app exists")
    validator.check("8", "/health" in routes, "/health endpoint exists")
    validator.check("8", "/reset" in routes, "/reset endpoint exists")
    validator.check("8", "/step" in routes, "/step endpoint exists")
    validator.check("8", "/state" in routes, "/state endpoint exists")
    validator.check("8", "CORSMiddleware" in server_source, "CORS middleware added")
    validator.check("8", "7860" in server_source, "Runs on port 7860")
    print()

    print("[SECTION 9: README]")
    readme_path = ROOT / "README.md"
    readme_text = read_text(readme_path)
    word_count = len(readme_text.split())
    validator.check("9", readme_path.exists(), "File exists")
    validator.check("9", "## " in readme_text, 'Contains "## " headers (has sections)')
    validator.check("9", "observation" in readme_text.lower(), 'Contains "observation" (documents observation space)')
    validator.check("9", "action" in readme_text.lower(), 'Contains "action" (documents action space)')
    validator.check("9", "reward" in readme_text.lower(), 'Contains "reward" (documents reward function)')
    validator.check("9", "docker" in readme_text.lower(), 'Contains "docker" or "Docker" (has setup instructions)')
    validator.check("9", "task" in readme_text.lower(), 'Contains "task" or "Task" (describes tasks)')
    validator.check("9", word_count > 300, "Word count > 300 words")
    print()

    print("[SECTION 10: ENVIRONMENT VARIABLES]")
    py_files = [
        path for path in ROOT.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    ]
    all_python_source = "\n".join(path.read_text(encoding="utf-8") for path in py_files)
    key_prefix = "sk" + "-"
    project_prefix = "sk" + "-proj"
    validator.check("10", key_prefix not in all_python_source, f'grep for "{key_prefix}" in all .py files → should find nothing')
    validator.check("10", project_prefix not in all_python_source, f'grep for "{project_prefix}" in all .py files → should find nothing')
    validator.check("10", 'os.getenv("OPENAI_API_KEY"' in all_python_source and 'os.getenv("API_BASE_URL"' in all_python_source, "All API credentials read from os.getenv()")
    print()

    print("[SECTION 11: DRY RUN END TO END]")
    os.environ["DRY_RUN"] = "true"
    dry_run_ok = True
    dry_run_details: list[str] = []
    try:
        for task_id in (1, 2, 3):
            env = RedTeamArenaEnv(task_id=task_id)
            observation = env.reset()
            if not isinstance(observation, Observation):
                raise TypeError(f"Task {task_id} reset did not return Observation")
            result = env.step(
                Action(
                    guard_response="I cannot help with that request.",
                    refusal_type="hard_refusal",
                )
            )
            if not (
                isinstance(result, tuple)
                and len(result) == 4
                and isinstance(result[0], Observation)
                and isinstance(result[1], Reward)
                and isinstance(result[2], bool)
                and isinstance(result[3], dict)
            ):
                raise TypeError(f"Task {task_id} step returned invalid tuple")
            current_state = env.state()
            if not isinstance(current_state, Observation):
                raise TypeError(f"Task {task_id} state did not return Observation")

        for _, grader in graders:
            grade = grader.grade(mock_history)
            if not isinstance(grade, float) or not 0.0 <= grade <= 1.0:
                raise ValueError("Grader score out of range during dry run")
    except Exception as exc:
        dry_run_ok = False
        dry_run_details.append(str(exc))

    validator.check("11", dry_run_ok, "Complete dry-run environment validation completed without exceptions")
    print()

    print("[FINAL SUMMARY]")
    print("========================================")
    print(f"Total Checks: {validator.total}")
    print(f"Passed: {validator.passed}")
    print(f"Failed: {validator.failed}")
    print(f"Score: {validator.passed}/{validator.total}")
    print()
    print("Overall Status:")
    if validator.failed == 0:
        print("🏆 READY FOR SUBMISSION")
    elif validator.failed <= 3:
        print("⚠️  NEEDS FIXES")
    else:
        print("❌  NOT READY")
    print("========================================")


if __name__ == "__main__":
    main()

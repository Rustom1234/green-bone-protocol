"""
Experiments task: the clan writes, runs, and tests code on disk.

Outputs land in /homes/rdubash/agents/experiments/<slug>/
Each experiment gets: main.py, test_main.py, results.txt

WhatsApp examples (router picks this automatically):
  "write a binary search tree with tests"
  "build a rate limiter class in python"
  "experiment: async job queue using asyncio"
"""

import os
import re
import subprocess
import json
from pathlib import Path

from greenbone.types import Task, Result
from greenbone.clan import (
    Clan, Weatherman, Horn,
    WhiteRat, IntelFist, ExecutionFist, Finger,
)

EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "experiments"


# ── Rats ──────────────────────────────────────────────────────

class RequirementsRat(WhiteRat):
    name = "requirements_rat"
    system_prompt = (
        "You are a senior engineer. Given a coding task, extract: "
        "what needs to be built, edge cases to handle, interfaces/APIs expected, "
        "test scenarios that must pass. "
        "Return JSON: {what_to_build, interfaces, edge_cases, test_scenarios, suggested_filename}"
    )
    def run(self, task: Task) -> Result:
        self._log(f"analysing: {task.goal[:60]}")
        return self._ok(self._llm_json_safe(
            f"Coding task: {task.goal}\n\nExtract requirements.",
            fallback={}, max_tokens=1000
        ))


# ── Intel Fists ───────────────────────────────────────────────

class DesignFist(IntelFist):
    name = "design_fist"
    system_prompt = (
        "You are a senior engineer. Given requirements, produce a clean design. "
        "Return JSON: {module_name, classes, functions, data_structures, "
        "implementation_notes, test_plan}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        return self._ok(self._llm_json_safe(
            f"Requirements:\n{json.dumps(raw, indent=2)}\n\nProduce implementation design.",
            fallback={}, max_tokens=1500
        ))


# ── Fingers ───────────────────────────────────────────────────

class WriteCodeFinger(Finger):
    name = "write_code"
    system_prompt = (
        "You write clean, production-quality Python code. "
        "Include type hints. Include a __main__ block that demos the code. "
        "Return ONLY the Python code, no markdown fences, no explanation."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=3000))


class WriteTestsFinger(Finger):
    name = "write_tests"
    system_prompt = (
        "You write thorough pytest tests. "
        "Cover happy path, edge cases, and error conditions. "
        "Import from the module passed to you. "
        "Return ONLY valid pytest code, no markdown fences, no explanation."
    )
    def run(self, task: Task) -> Result:
        code = task.context.get("code", "")
        module = task.context.get("module_name", "main")
        return self._ok(self._llm(
            f"Write pytest tests for this code.\nModule name: {module}\n\n{code}\n\nTask: {task.goal}",
            max_tokens=2000
        ))


class RunCodeFinger(Finger):
    name = "run_code"
    system_prompt = "You analyse code execution results and summarise what happened."
    def run(self, task: Task) -> Result:
        file_path = task.context.get("file_path", "")
        if not file_path or not os.path.exists(file_path):
            return self._fail(f"file not found: {file_path}")
        self._log(f"running {file_path}")
        try:
            proc = subprocess.run(
                ["python", file_path],
                capture_output=True, text=True, timeout=30,
                cwd=str(Path(file_path).parent)
            )
            return self._ok({
                "stdout": proc.stdout[:2000],
                "stderr": proc.stderr[:1000],
                "returncode": proc.returncode,
            })
        except subprocess.TimeoutExpired:
            return self._fail("execution timed out after 30s")
        except Exception as e:
            return self._fail(str(e))


class RunTestsFinger(Finger):
    name = "run_tests"
    system_prompt = "You analyse pytest results and summarise what passed and failed."
    def run(self, task: Task) -> Result:
        test_path = task.context.get("test_path", "")
        if not test_path or not os.path.exists(test_path):
            return self._fail(f"test file not found: {test_path}")
        self._log(f"pytest {test_path}")
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
                capture_output=True, text=True, timeout=60,
                cwd=str(Path(test_path).parent)
            )
            return self._ok({
                "output": proc.stdout[-3000:],
                "stderr": proc.stderr[:500],
                "returncode": proc.returncode,
                "passed": proc.returncode == 0,
            })
        except subprocess.TimeoutExpired:
            return self._fail("tests timed out after 60s")
        except Exception as e:
            return self._fail(str(e))


class FixCodeFinger(Finger):
    name = "fix_code"
    system_prompt = (
        "You are a debugger. Given code and an error, return the fixed code. "
        "Return ONLY the fixed Python code, no markdown, no explanation."
    )
    def run(self, task: Task) -> Result:
        code    = task.context.get("code", "")
        error   = task.context.get("error", "")
        return self._ok(self._llm(
            f"Fix this code.\n\nError:\n{error}\n\nCode:\n{code}",
            max_tokens=3000
        ))


# ── Execution Fist ────────────────────────────────────────────

class ExperimentFist(ExecutionFist):
    name = "experiment_fist"
    system_prompt = "You write, run, and test Python experiments on disk."

    def __init__(self):
        self.coder   = WriteCodeFinger()
        self.tester  = WriteTestsFinger()
        self.runner  = RunCodeFinger()
        self.pytest  = RunTestsFinger()
        self.fixer   = FixCodeFinger()

    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""

        # Make experiment dir
        slug = re.sub(r"[^a-z0-9]+", "_", task.goal.lower())[:40].strip("_")
        exp_dir = EXPERIMENTS_DIR / slug
        exp_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"experiment: {slug}")

        # Write main code
        code = self.coder.run(Task(
            goal=f"{task.goal}\n\n{ctx}",
            context={}
        )).output or ""
        # Strip any accidental fences
        code = re.sub(r"```(?:python)?|```", "", code).strip()

        main_path = exp_dir / "main.py"
        main_path.write_text(code)
        self._log(f"wrote {main_path}")

        # Run it
        run_result = self.runner.run(Task(
            goal=task.goal,
            context={"file_path": str(main_path)}
        ))

        # If it crashed, try to fix once
        if run_result.success and run_result.output.get("returncode") != 0:
            self._log("run failed, attempting fix")
            error = run_result.output.get("stderr", "") + run_result.output.get("stdout", "")
            fixed_code = self.fixer.run(Task(
                goal=task.goal,
                context={"code": code, "error": error}
            )).output or code
            fixed_code = re.sub(r"```(?:python)?|```", "", fixed_code).strip()
            main_path.write_text(fixed_code)
            code = fixed_code
            run_result = self.runner.run(Task(
                goal=task.goal,
                context={"file_path": str(main_path)}
            ))

        # Write tests
        module_name = "main"
        tests_code = self.tester.run(Task(
            goal=task.goal,
            context={"code": code, "module_name": module_name}
        )).output or ""
        tests_code = re.sub(r"```(?:python)?|```", "", tests_code).strip()

        test_path = exp_dir / "test_main.py"
        test_path.write_text(tests_code)

        # Run tests
        test_result = self.pytest.run(Task(
            goal=task.goal,
            context={"test_path": str(test_path)}
        ))

        # Write results summary
        run_out = run_result.output or {}
        test_out = test_result.output or {}
        summary = (
            f"=== EXPERIMENT: {task.goal} ===\n\n"
            f"FILE: {main_path}\n\n"
            f"RUN OUTPUT:\n{run_out.get('stdout','')}\n"
            f"{run_out.get('stderr','')}\n\n"
            f"TESTS:\n{test_out.get('output','')}\n"
        )
        (exp_dir / "results.txt").write_text(summary)

        passed = test_out.get("passed", False)
        self._log(f"tests {'✓ passed' if passed else '✗ failed'}")

        return self._ok({
            "experiment_dir": str(exp_dir),
            "code_file": str(main_path),
            "test_file": str(test_path),
            "run_output": run_out.get("stdout", "")[:800],
            "run_error": run_out.get("stderr", "")[:400],
            "test_results": test_out.get("output", "")[-1200:],
            "tests_passed": passed,
        })


# ── Wire up ───────────────────────────────────────────────────

def build_clan() -> Clan:
    weatherman = Weatherman(
        rats=[RequirementsRat()],
        intel_fists=[DesignFist()],
    )
    horn = Horn(fists={"experiment": ExperimentFist()})
    horn.system_prompt = (
        "You are the Horn. You receive a coding task and a design dossier. "
        "Split into 1-3 experiment subtasks (one per distinct component if needed). "
        "Each subtask is a self-contained coding experiment. "
        'Available fist types: "experiment" (write + run + test Python code on disk). '
        'Return JSON: [{"fist": "experiment", "subtask": "<specific coding task>"}]'
    )
    return Clan(weatherman=weatherman, horn=horn)

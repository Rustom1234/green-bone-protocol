"""
Shell task — the clan SSHes into a remote host and runs shell commands.

The host name comes from commission context["host"] and must be resolvable
via the user's SSH config (~/.ssh/config). Relies on key-based auth.

Planner picks this for:
  "create a directory on goldbery-via-dean"
  "run a script on <host>"
  "set up a new project folder on <host>"
"""

import subprocess
import json

from greenbone.types import Task, Result
from greenbone.clan import (
    Clan, Weatherman, Horn,
    WhiteRat, IntelFist, ExecutionFist, Finger,
)


# ── Finger ────────────────────────────────────────────────────

class ShellFinger(Finger):
    name = "shell_finger"
    system_prompt = "You run a single remote shell command via SSH and return the output."

    def run(self, task: Task) -> Result:
        host    = task.context.get("host", "")
        command = task.context.get("command", task.goal)

        if not host:
            return self._fail("no host in context")

        self._log(f"ssh {host}: {command[:80]}")

        try:
            proc = subprocess.run(
                ["ssh",
                 "-o", "BatchMode=yes",
                 "-o", "StrictHostKeyChecking=no",
                 "-o", "ConnectTimeout=15",
                 host, command],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return self._ok({
                "host": host,
                "command": command,
                "stdout": proc.stdout[:3000],
                "stderr": proc.stderr[:1000],
                "returncode": proc.returncode,
                "success": proc.returncode == 0,
            })
        except subprocess.TimeoutExpired:
            return self._fail(f"SSH timed out after 60s (host={host})")
        except FileNotFoundError:
            return self._fail("ssh binary not found in PATH")
        except Exception as e:
            return self._fail(str(e))


# ── Rat ───────────────────────────────────────────────────────

class ShellPlanRat(WhiteRat):
    name = "shell_plan_rat"
    system_prompt = (
        "You are a senior sysadmin. Given a goal and a remote host, plan the exact "
        "shell commands needed. Be specific. Chain with && where order matters. "
        "Return JSON: {commands: [\"cmd1\", \"cmd2\"], notes: \"brief explanation\"}"
    )

    def run(self, task: Task) -> Result:
        host  = task.context.get("host", "unknown")
        prior = task.context.get("prior_results", {})
        self._log(f"planning for {host}: {task.goal[:60]}")
        return self._ok(self._llm_json_safe(
            f"Host: {host}\nGoal: {task.goal}\n\n"
            f"Prior task results:\n{json.dumps(prior, indent=2) if prior else 'none'}\n\n"
            f"Plan the exact SSH commands to accomplish this goal.",
            fallback={"commands": [], "notes": "planning failed"},
            max_tokens=800,
        ))


# ── Intel Fist ────────────────────────────────────────────────

class CommandSafetyFist(IntelFist):
    name = "command_safety_fist"
    system_prompt = (
        "You review shell commands for safety and correctness. "
        "Flag destructive commands (rm -rf without path, DROP TABLE, etc.). "
        "Return JSON: {safe: true/false, approved_commands: [...], warnings: [...], reason: \"...\"}"
    )

    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        commands = []
        for item in raw:
            data = item.get("data", {})
            if isinstance(data, dict):
                commands.extend(data.get("commands", []))
        self._log(f"safety-checking {len(commands)} command(s)")
        return self._ok(self._llm_json_safe(
            f"Goal: {task.goal}\n\nCommands:\n{json.dumps(commands, indent=2)}\n\n"
            f"Review for safety. Return approved_commands as the safe list to run.",
            fallback={"safe": True, "approved_commands": commands, "warnings": [], "reason": ""},
            max_tokens=600,
        ))


# ── Execution Fist ────────────────────────────────────────────

class ShellFist(ExecutionFist):
    name = "shell_fist"
    system_prompt = "You execute a sequence of SSH commands on a remote host and collect results."

    def __init__(self):
        self.finger = ShellFinger()

    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        host    = task.context.get("host", "")

        # Pull approved commands from dossier findings, then context, then goal
        approved = []
        if dossier:
            for finding in dossier.key_findings:
                if finding.strip().upper().startswith("CMD:"):
                    approved.append(finding.split(":", 1)[1].strip())
        if not approved:
            approved = task.context.get("approved_commands", [])
        if not approved:
            approved = [task.goal]

        self._log(f"running {len(approved)} command(s) on {host}")

        outputs = []
        for cmd in approved:
            result = self.finger.run(Task(goal=cmd, context={"host": host, "command": cmd}))
            out = result.output or {}
            outputs.append({
                "command": cmd,
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode", -1),
                "success": out.get("success", False),
            })
            self._log(f"  rc={out.get('returncode','?')} — {cmd[:50]}")
            if not out.get("success", False):
                break  # stop on first failure

        all_ok = all(o["success"] for o in outputs)
        return self._ok({
            "host": host,
            "goal": task.goal,
            "commands_run": len(outputs),
            "all_succeeded": all_ok,
            "outputs": outputs,
            "summary": _summarise(outputs),
        })


def _summarise(outputs: list[dict]) -> str:
    lines = []
    for o in outputs:
        status = "✓" if o["success"] else "✗"
        lines.append(f"{status} `{o['command'][:60]}`")
        if o["stdout"].strip():
            lines.append(f"   {o['stdout'].strip()[:300]}")
        if o["stderr"].strip() and not o["success"]:
            lines.append(f"   ERR: {o['stderr'].strip()[:200]}")
    return "\n".join(lines)


# ── Wire up ───────────────────────────────────────────────────

def build_clan() -> Clan:
    weatherman = Weatherman(
        rats=[ShellPlanRat()],
        intel_fists=[CommandSafetyFist()],
    )
    horn = Horn(fists={"shell": ShellFist()})
    horn.system_prompt = (
        "You are the Horn. You have a shell execution plan with approved commands. "
        "Assign the work to the shell fist. Keep it to 1-2 subtasks. "
        'Available fist: "shell" (runs SSH commands on the remote host). '
        'Return JSON: [{"fist": "shell", "subtask": "<what this achieves>"}]'
    )
    return Clan(weatherman=weatherman, horn=horn)

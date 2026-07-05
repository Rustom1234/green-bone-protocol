from __future__ import annotations
import json
import concurrent.futures
from typing import Any
from greenbone.types import Task, Result, Rank, Dossier, Discipline
from greenbone.agents import Agent, COMMAND, FIELD
from greenbone.steel import Steel


class WhiteRat(Agent):
    rank       = Rank.WHITE_RAT
    discipline = Discipline.LIGHTNESS
    model = FIELD
    system_prompt = "You are a data scout. Extract raw intelligence. Return JSON."

    def run(self, task: Task) -> Result:
        raise NotImplementedError


class IntelFist(Agent):
    rank       = Rank.FIST
    discipline = Discipline.LIGHTNESS
    model = FIELD
    system_prompt = "You clean and structure raw intelligence data. Return JSON."

    def run(self, task: Task) -> Result:
        raise NotImplementedError


class Finger(Agent):
    rank       = Rank.FINGER
    discipline = Discipline.LIGHTNESS
    model = FIELD
    system_prompt = "You perform a single precise action and return the result."

    def run(self, task: Task) -> Result:
        raise NotImplementedError


class ExecutionFist(Agent):
    rank       = Rank.FIST
    discipline = Discipline.LIGHTNESS
    model = FIELD
    system_prompt = "You execute a specific class of task using dossier intelligence."

    def run(self, task: Task) -> Result:
        raise NotImplementedError


class Weatherman(Agent):
    name = "weatherman"
    rank = Rank.WEATHERMAN
    discipline = Discipline.PERCEPTION
    model = COMMAND
    system_prompt = (
        "You are the Weatherman — the intelligence commander of the clan. "
        "You see everything before the Horn strikes. "
        "Your dossiers are precise, dense, and give the Horn exactly what he needs. "
        "You write in structured formats Horn can parse without an LLM."
    )

    def __init__(self, rats: list[WhiteRat], intel_fists: list[IntelFist]):
        self.rats        = rats
        self.intel_fists = intel_fists

    def run(self, task: Task) -> Result:
        target = task.context.get("target", task.goal)
        self._log(f"opening on [{target}]")
        raw_intel = self._run_rats(task)
        self._log(f"{len(raw_intel)} intel packages from rats")
        cleaned = self._run_intel_fists(task, raw_intel)
        dossier = self._synthesise_dossier(task, raw_intel, cleaned)
        self._log(f"dossier sealed — {len(dossier.key_findings)} findings")
        return self._ok(dossier)

    def _run_rats(self, task: Task) -> list[dict]:
        raw_intel = []
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {pool.submit(rat.run, task): rat.name for rat in self.rats}
            for future in concurrent.futures.as_completed(futures):
                rat_name = futures[future]
                try:
                    result = future.result()
                    if result.success and result.output is not None:
                        raw_intel.append({"source": rat_name, "data": result.output})
                    else:
                        self._log(f"rat [{rat_name}] returned no data")
                except Exception as e:
                    self._log(f"rat [{rat_name}] crashed: {e}")
        return raw_intel

    def _run_intel_fists(self, task: Task, raw_intel: list[dict]) -> dict:
        accumulated = {}
        for fist in self.intel_fists:
            fist_task = Task(
                goal=task.goal,
                context={**task.context, "raw_intel": raw_intel, "accumulated_intel": accumulated}
            )
            result = fist.run(fist_task)
            if result.success and result.output is not None:
                accumulated[fist.name] = result.output
            else:
                self._log(f"intel fist [{fist.name}] failed: {result.error}")
        return accumulated

    def _synthesise_dossier(self, task: Task, raw_intel: list[dict], cleaned: dict) -> Dossier:
        target = task.context.get("target", task.goal)
        synthesis = self._llm(
            f"Target: {target}\nGoal: {task.goal}\n\n"
            f"Cleaned intel:\n{json.dumps(cleaned, indent=2)}\n\n"
            f"Raw intel sources: {[i['source'] for i in raw_intel]}\n\n"
            f"Write the sealed Dossier in EXACTLY this format:\n\n"
            f"WEBSITE: one dense sentence on their web/product presence\n"
            f"GITHUB: one sentence on their code signals (or 'No GitHub data')\n"
            f"NEWS: one sentence on recent events (or 'No news data')\n"
            f"TECH_STACK: comma-separated list of technologies\n"
            f"FINDINGS:\n- finding one\n- finding two\n- finding three\n"
            f"APPROACH: 2-3 sentences telling Horn exactly how to execute the goal. "
            f"Be specific. Reference tech stack and findings.",
            max_tokens=700,
        )
        sections: dict[str, Any] = {
            "WEBSITE": "", "GITHUB": "", "NEWS": "",
            "TECH_STACK": "", "FINDINGS": [], "APPROACH": ""
        }
        current = None
        for line in synthesis.splitlines():
            line = line.strip()
            matched = False
            for key in sections:
                if line.upper().startswith(key + ":"):
                    current = key
                    remainder = line[len(key)+1:].strip()
                    if key == "FINDINGS":
                        if remainder.startswith("- "):
                            sections[key].append(remainder[2:])
                    else:
                        sections[key] = remainder
                    matched = True
                    break
            if not matched and current:
                if current == "FINDINGS" and line.startswith("- "):
                    sections["FINDINGS"].append(line[2:])
                elif current == "APPROACH" and line:
                    sections["APPROACH"] += " " + line
        tech_stack = [t.strip() for t in sections["TECH_STACK"].split(",") if t.strip()]
        return Dossier(
            target=target,
            website_summary=sections["WEBSITE"],
            github_summary=sections["GITHUB"],
            news_summary=sections["NEWS"],
            tech_stack=tech_stack,
            key_findings=sections["FINDINGS"],
            recommended_approach=sections["APPROACH"].strip(),
        )


class Horn(Agent):
    name = "horn"
    rank = Rank.HORN
    discipline = Discipline.CHANNELING
    model = COMMAND
    system_prompt = (
        "You are the Horn — the execution commander of the clan. "
        "You receive a Dossier and a goal. You plan the strike and split it into "
        "precise subtasks for your fists. Every subtask must be specific and actionable. "
        "Available fist types: {fist_types}. "
        'Return a JSON list: [{{"fist": "<type>", "subtask": "<specific instruction>"}}]. '
        "2-5 subtasks maximum."
    )

    def __init__(self, fists: dict[str, ExecutionFist], steel: Steel | None = None):
        self.fists = fists
        # Steel guard for the Strike Gate (seam 2) — every fist result passes
        # through here before it is collected. Clan will share its own Steel
        # instance in here if one wasn't supplied explicitly.
        self.steel = steel or Steel()

    def run(self, task: Task) -> Result:
        dossier: Dossier | None = task.context.get("dossier")
        if not dossier:
            return self._fail("Horn received no Dossier. Run Weatherman first.")
        self._log(f"reading dossier on [{dossier.target}]")
        fist_types = ", ".join(f'"{k}"' for k in self.fists)
        planning_system = self.system_prompt.replace("{fist_types}", fist_types)
        plan_prompt = (
            f"Goal: {task.goal}\n\n{dossier.to_context_block()}\n\n"
            f"Split this goal into specific subtasks. Available fist types: {fist_types}"
        )
        try:
            original_sp = self.system_prompt
            self.system_prompt = planning_system
            assignments = self._llm_json(plan_prompt, max_tokens=600)
            self.system_prompt = original_sp
        except Exception as e:
            self._log(f"planning failed ({e}), single-fist fallback")
            assignments = [{"fist": next(iter(self.fists)), "subtask": task.goal}]
        self._log(f"plan: {len(assignments)} subtasks")
        results = []
        for assignment in assignments:
            fist_key = assignment.get("fist", "").lower()
            subtask  = assignment.get("subtask", task.goal)
            fist     = self.fists.get(fist_key)
            if not fist:
                self._log(f"unknown fist key '{fist_key}', skipping")
                continue
            fist_task = Task(goal=subtask, context={**task.context, "dossier": dossier})
            # Strike Gate (seam 2): validate/retry/sanitise each fist result via Steel
            # before it is collected and returned to the Pillar.
            result = self.steel.guard_fist(fist, fist_task)
            results.append(result)
            self._log(f"→ [{fist_key}] {'✓' if result.success else '✗'}")
        return self._ok(results)


class Clan:
    def __init__(self, weatherman: Weatherman, horn: Horn, steel: Steel | None = None):
        self.weatherman = weatherman
        self.horn       = horn
        # Clan owns the Steel instance; share it with the Horn's Strike Gate
        # (seam 2) so both gates report through one guard rather than two
        # independently-constructed ones.
        self.steel = steel or self.horn.steel or Steel()
        self.horn.steel = self.steel

    def commission(self, goal: str, target: str, context: dict | None = None) -> dict:
        ctx = context or {}
        ctx["target"] = target
        wx_result = self.weatherman.run(Task(goal=goal, context=ctx))
        if not wx_result.success or not isinstance(wx_result.output, Dossier):
            return {"dossier": None, "horn": [], "error": "Weatherman failed"}
        # Dossier Gate (seam 1): validate + sanitise before the Horn ever sees it.
        dossier = self.steel.guard_dossier(wx_result.output)
        horn_result = self.horn.run(Task(goal=goal, context={**ctx, "dossier": dossier}))
        return {"dossier": dossier, "horn": horn_result.output or []}

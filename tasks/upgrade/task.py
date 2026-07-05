"""
Upgrade task: the clan researches new tools and writes new capabilities into itself.

The clan will:
1. Research new tools, libraries, GitHub repos, Claude API features, MCP servers
2. Design new rats/fists/task configs based on what it finds
3. Write them to tasks/<name>/task.py on disk
4. They're instantly available: server auto-discovers all task configs

WhatsApp examples (router picks this automatically):
  "upgrade yourself with new github tools"
  "research new claude api features and add them"
  "find new mcp servers and add support for them"
  "add a twitter/x scraping rat"
  "give yourself web search capability"
  "add support for reading arxiv papers"
"""

import os
import re
import json
import subprocess
import importlib
from pathlib import Path

import httpx

from greenbone.types import Task, Result
from greenbone.clan import (
    Clan, Weatherman, Horn,
    WhiteRat, IntelFist, ExecutionFist, Finger,
)

TASKS_DIR   = Path(__file__).parent.parent
AGENTS_DIR  = Path(__file__).parent.parent.parent


# ── Rats ──────────────────────────────────────────────────────

class GitHubToolsRat(WhiteRat):
    name = "github_tools_rat"
    system_prompt = (
        "You research GitHub repos and tools relevant to AI agents, web scraping, "
        "data collection, and multi-agent frameworks. "
        "Return JSON: {repos, tools, libraries, patterns, notable_apis}"
    )
    def run(self, task: Task) -> Result:
        self._log(f"github search: {task.goal[:60]}")
        queries = self._extract_queries(task.goal)
        results = []
        for q in queries[:3]:
            try:
                r = httpx.get(
                    "https://api.github.com/search/repositories",
                    params={"q": q, "sort": "stars", "per_page": 10},
                    headers={"Accept": "application/vnd.github+json"},
                    timeout=10
                )
                if r.status_code == 200:
                    items = r.json().get("items", [])
                    results.extend([{
                        "name": x.get("full_name"),
                        "description": x.get("description"),
                        "stars": x.get("stargazers_count"),
                        "url": x.get("html_url"),
                        "topics": x.get("topics", []),
                        "language": x.get("language"),
                    } for x in items])
            except Exception as e:
                self._log(f"search failed: {e}")
        if not results:
            return self._fail("no GitHub results")
        return self._ok(self._llm_json_safe(
            f"Goal: {task.goal}\n\nGitHub results:\n{json.dumps(results, indent=2)}\n\n"
            f"Extract: what tools exist, how they work, which are most useful for AI agents.",
            fallback={}, max_tokens=1500
        ))

    def _extract_queries(self, goal: str) -> list[str]:
        keywords = re.findall(r'\b\w+\b', goal.lower())
        base = " ".join(keywords[:4])
        return [f"{base} python agent", f"{base} api", base]


class HackerNewsRat(WhiteRat):
    name = "hn_rat"
    system_prompt = (
        "You research HackerNews for tool recommendations, new releases, and community opinions. "
        "Return JSON: {tools_mentioned, community_recommendations, new_releases, debates}"
    )
    def run(self, task: Task) -> Result:
        self._log(f"HN: {task.goal[:60]}")
        try:
            r = httpx.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": task.goal, "tags": "story,comment", "hitsPerPage": 30},
                timeout=10
            )
            hits = r.json().get("hits", [])
            texts = []
            for h in hits[:20]:
                txt = h.get("title") or h.get("comment_text") or ""
                txt = re.sub(r"<[^>]+>", " ", txt).strip()[:400]
                if txt:
                    texts.append(txt)
            if not texts:
                return self._fail("no HN results")
            return self._ok(self._llm_json_safe(
                f"Goal: {task.goal}\n\nHN content:\n" + "\n---\n".join(texts),
                fallback={}, max_tokens=1000
            ))
        except Exception as e:
            return self._fail(str(e))


class KnowledgeRat(WhiteRat):
    name = "knowledge_rat"
    system_prompt = (
        "You are a senior engineer. Report everything you know about tools, libraries, "
        "APIs, and patterns relevant to the given goal. Focus on: what exists, how it works, "
        "Python client libraries, authentication requirements, rate limits, data returned. "
        "Return JSON: {tools, libraries, apis, patterns, implementation_notes, gotchas}"
    )
    def run(self, task: Task) -> Result:
        self._log(f"knowledge: {task.goal[:60]}")
        return self._ok(self._llm_json_safe(
            f"Goal: {task.goal}\n\nReport all relevant tools and implementation knowledge.",
            fallback={}, max_tokens=2000
        ))


class WebRat(WhiteRat):
    name = "web_rat"
    system_prompt = (
        "You extract tool/API information from web pages. "
        "Return JSON: {tool_name, what_it_does, api_endpoints, auth_method, "
        "python_library, rate_limits, data_returned, example_usage}"
    )
    def run(self, task: Task) -> Result:
        urls = task.context.get("urls", [])
        if not urls:
            return self._fail("no URLs")
        combined = []
        for url in urls:
            self._log(f"reading {url}")
            try:
                resp = httpx.get(url, follow_redirects=True, timeout=12,
                                 headers={"User-Agent": "Mozilla/5.0"})
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()[:6000]
                combined.append(f"URL: {url}\n{text}")
            except Exception as e:
                self._log(f"failed {url}: {e}")
        if not combined:
            return self._fail("no pages reachable")
        return self._ok(self._llm_json_safe(
            f"Goal: {task.goal}\n\n" + "\n\n---\n\n".join(combined),
            fallback={}, max_tokens=1500
        ))


# ── Intel Fists ───────────────────────────────────────────────

class CapabilityFist(IntelFist):
    name = "capability_fist"
    system_prompt = (
        "You are a senior engineer. Given intelligence about tools and libraries, "
        "design concrete Green Bone Protocol agents that would use them. "
        "For each new capability, specify exactly: "
        "rat_or_fist, name, what_it_does, python_dependencies, implementation_sketch. "
        "Return JSON: {new_capabilities: [{type, name, description, dependencies, sketch}]}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        return self._ok(self._llm_json_safe(
            f"Goal: {task.goal}\n\nIntel:\n{json.dumps(raw, indent=2)}\n\n"
            f"Design specific Green Bone Protocol agents for these capabilities.",
            fallback={"new_capabilities": []}, max_tokens=2000
        ))


# ── Fingers ───────────────────────────────────────────────────

class WriteTaskConfigFinger(Finger):
    name = "write_task_config"
    system_prompt = (
        "You are a senior Python engineer writing Green Bone Protocol task configs. "
        "You write complete, working task.py files following the Green Bone Protocol pattern exactly. "
        "The file must: import from greenbone.clan, subclass WhiteRat/IntelFist/ExecutionFist/Finger, "
        "implement run() returning Result via self._ok() or self._fail(), "
        "define build_clan() returning a Clan instance. "
        "Return ONLY valid Python code. No markdown fences. No explanation."
    )

    GREENBONE_EXAMPLE = """
# Pattern reference:
class MyRat(WhiteRat):
    name = "my_rat"
    system_prompt = "You extract X. Return JSON: {...}"
    def run(self, task: Task) -> Result:
        data = self._llm_json_safe(f"Extract from: {task.goal}", fallback={})
        return self._ok(data)

class MyFist(ExecutionFist):
    name = "my_fist"
    system_prompt = "You do X."
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        result = self._llm(f"{task.goal}\\n\\n{ctx}", max_tokens=2000)
        return self._ok(result)

def build_clan() -> Clan:
    weatherman = Weatherman(rats=[MyRat()], intel_fists=[])
    horn = Horn(fists={"my_fist": MyFist()})
    horn.system_prompt = (
        "You are the Horn. Available fist types: \\"my_fist\\". "
        'Return JSON: [{{"fist": "my_fist", "subtask": "..."}}]'
    )
    return Clan(weatherman=weatherman, horn=horn)
"""

    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        capability_name = task.context.get("capability_name", "new_capability")
        deps = task.context.get("dependencies", [])

        code = self._llm(
            f"Write a complete Green Bone Protocol task config for: {task.goal}\n\n"
            f"Dossier intel:\n{ctx}\n\n"
            f"Python pattern to follow:\n{self.GREENBONE_EXAMPLE}\n\n"
            f"Required imports: httpx, json, re (already available)\n"
            f"Additional dependencies needed: {deps}\n\n"
            f"Task config name: {capability_name}\n"
            f"Write the full tasks/{capability_name}/task.py file.",
            max_tokens=4000
        )
        code = re.sub(r"```(?:python)?|```", "", code).strip()
        return self._ok({"code": code, "name": capability_name, "dependencies": deps})


class InstallAndRegisterFinger(Finger):
    name = "install_and_register"
    system_prompt = "You confirm installation and registration of new capabilities."

    def run(self, task: Task) -> Result:
        code         = task.context.get("code", "")
        name         = task.context.get("name", "")
        dependencies = task.context.get("dependencies", [])

        if not code or not name:
            return self._fail("no code or name provided")

        # Write the task config
        task_dir = TASKS_DIR / name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "__init__.py").write_text("")
        (task_dir / "task.py").write_text(code)
        self._log(f"wrote tasks/{name}/task.py")

        # Install any new dependencies
        installed = []
        if dependencies:
            for dep in dependencies:
                try:
                    subprocess.run(
                        ["pip", "install", dep],
                        capture_output=True, timeout=60,
                        cwd=str(AGENTS_DIR)
                    )
                    installed.append(dep)
                    self._log(f"installed {dep}")
                except Exception as e:
                    self._log(f"failed to install {dep}: {e}")

        # Validate it imports correctly
        try:
            mod = importlib.import_module(f"tasks.{name}.task")
            clan = mod.build_clan()
            valid = True
            error = None
        except Exception as e:
            valid = False
            error = str(e)
            self._log(f"validation failed: {e}")

        return self._ok({
            "task_config": name,
            "file": str(task_dir / "task.py"),
            "dependencies_installed": installed,
            "valid": valid,
            "error": error,
        })


# ── Execution Fist ────────────────────────────────────────────

class UpgradeFist(ExecutionFist):
    name = "upgrade_fist"
    system_prompt = "You write and install new Green Bone Protocol task configs from dossier intelligence."

    def __init__(self):
        self.writer    = WriteTaskConfigFinger()
        self.installer = InstallAndRegisterFinger()

    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")

        # Parse capability name from goal
        name = re.sub(r"[^a-z0-9]+", "_", task.goal.lower())[:30].strip("_")
        # Try to get dependency info from dossier findings
        deps = []
        if dossier:
            for finding in dossier.key_findings:
                # Look for library names in findings
                libs = re.findall(r'\b([a-z][a-z0-9_-]+(?:py|sdk|client|lib))\b', finding.lower())
                deps.extend(libs[:3])

        self._log(f"writing capability: {name}")

        write_result = self.writer.run(Task(
            goal=task.goal,
            context={
                "dossier": dossier,
                "capability_name": name,
                "dependencies": deps,
            }
        ))

        if not write_result.success:
            return self._fail(f"code writing failed: {write_result.error}")

        code = write_result.output.get("code", "")

        install_result = self.installer.run(Task(
            goal=task.goal,
            context={
                "code": code,
                "name": name,
                "dependencies": deps,
            }
        ))

        out = install_result.output or {}
        return self._ok({
            "capability": name,
            "file": out.get("file", ""),
            "valid": out.get("valid", False),
            "error": out.get("error"),
            "dependencies_installed": out.get("dependencies_installed", []),
            "code_preview": code[:600],
        })


# ── Wire up ───────────────────────────────────────────────────

def build_clan() -> Clan:
    weatherman = Weatherman(
        rats=[GitHubToolsRat(), HackerNewsRat(), KnowledgeRat(), WebRat()],
        intel_fists=[CapabilityFist()],
    )
    horn = Horn(fists={"upgrade": UpgradeFist()})
    horn.system_prompt = (
        "You are the Horn. You receive intel about new tools and must commission "
        "new Green Bone Protocol task configs. Each subtask should be one new capability to add. "
        "Be specific: name the exact tool/API/library to integrate. "
        'Available fist types: "upgrade" (write and install a new task config). '
        'Return JSON: [{"fist": "upgrade", "subtask": "<specific capability to add>"}]. '
        "1-3 capabilities maximum."
    )
    return Clan(weatherman=weatherman, horn=horn)

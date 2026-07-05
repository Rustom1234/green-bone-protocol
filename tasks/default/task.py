"""
Default Green Bone Protocol task — general-purpose clan.
Includes WebRat, GitHubRat, NewsRat, TechFist, CleanFist, CodeFist, WriteFist.

The WhatsApp gateway imports build_clan() from here.
Customise rats/fists here or create a new tasks/<name>/task.py.
"""

import httpx
import re
import json

from greenbone.types import Task, Result
from greenbone.clan import (
    Clan, Weatherman, Horn,
    WhiteRat, IntelFist, ExecutionFist, Finger,
)


# ── White Rats ────────────────────────────────────────────────

class WebRat(WhiteRat):
    name = "web_rat"
    system_prompt = (
        "You are a web intelligence scout. Given raw page content, extract: "
        "what they do, product details, tech mentions, pricing signals, team signals. "
        "Return JSON: {summary, tech_mentions, key_facts, notable_links}"
    )
    def run(self, task: Task) -> Result:
        urls = task.context.get("urls", [])
        combined = []
        for url in urls:
            self._log(f"scraping {url}")
            try:
                resp = httpx.get(url, follow_redirects=True, timeout=12,
                                 headers={"User-Agent": "Mozilla/5.0"})
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()[:6000]
                combined.append(f"URL: {url}\n{text}")
            except Exception as e:
                self._log(f"failed {url}: {e}")
        if not combined:
            return self._fail("no URLs reachable")
        data = self._llm_json_safe(
            f"Target: {task.context.get('target')}\n\n" + "\n\n---\n\n".join(combined),
            fallback={}
        )
        return self._ok(data)


class GitHubRat(WhiteRat):
    name = "github_rat"
    system_prompt = (
        "You are a GitHub intelligence rat. Given public repo data, extract: "
        "main_languages, tech_stack, key_repos, observations. "
        "Return JSON: {main_languages, tech_stack, key_repos, observations}"
    )
    def run(self, task: Task) -> Result:
        org   = task.context.get("github_org", "")
        token = task.context.get("github_token", "")
        if not org:
            return self._fail("no github_org in context")
        self._log(f"spying on github/{org}")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            for endpoint in [
                f"https://api.github.com/orgs/{org}/repos?per_page=40&sort=updated",
                f"https://api.github.com/users/{org}/repos?per_page=40&sort=updated",
            ]:
                r = httpx.get(endpoint, headers=headers, timeout=10)
                if r.status_code == 200:
                    break
            repos = r.json() if isinstance(r.json(), list) else []
            slim = [
                {"name": repo.get("name"), "description": repo.get("description"),
                 "language": repo.get("language"), "topics": repo.get("topics", []),
                 "stars": repo.get("stargazers_count")}
                for repo in repos if isinstance(repo, dict)
            ]
            data = self._llm_json_safe(
                f"Org: {org}\n\nRepos:\n{json.dumps(slim, indent=2)}", fallback={}
            )
            return self._ok(data)
        except Exception as e:
            return self._fail(str(e))


class NewsRat(WhiteRat):
    name = "news_rat"
    system_prompt = (
        "You are a news intelligence rat. Report on the target: "
        "recent launches, funding, tech decisions, controversies, key people. "
        "Only state what you know to be true. "
        "Return JSON: {recent_events, key_people, risks, opportunities}"
    )
    def run(self, task: Task) -> Result:
        target = task.context.get("target", task.goal)
        self._log(f"hunting news on: {target}")
        data = self._llm_json_safe(
            f"Target: {target}\nGoal context: {task.goal}\n\n"
            f"Report everything you know about this organisation.",
            fallback={}
        )
        return self._ok(data)


# ── Intel Fists ───────────────────────────────────────────────

class TechFist(IntelFist):
    name = "tech_fist"
    system_prompt = (
        "You identify technology stacks from intelligence data. "
        "Return JSON: {confirmed: [], inferred: [], rumoured: []}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        self._log("identifying tech stack")
        data = self._llm_json_safe(
            f"All intel:\n{json.dumps(raw, indent=2)}",
            fallback={"confirmed": [], "inferred": [], "rumoured": []}
        )
        return self._ok(data)


class CleanFist(IntelFist):
    name = "clean_fist"
    system_prompt = (
        "You merge intelligence from multiple sources. "
        "Remove duplicates, resolve contradictions (prefer specific over vague). "
        "Return JSON: {summary, tech_stack_flat, key_facts, people, products, risks, opportunities}"
    )
    def run(self, task: Task) -> Result:
        raw   = task.context.get("raw_intel", [])
        accum = task.context.get("accumulated_intel", {})
        self._log("merging intel")
        data = self._llm_json_safe(
            f"Raw intel:\n{json.dumps(raw, indent=2)}\n\n"
            f"Previously processed:\n{json.dumps(accum, indent=2)}",
            max_tokens=1500, fallback={}
        )
        return self._ok(data)


# ── Fingers ───────────────────────────────────────────────────

class WriteCodeFinger(Finger):
    name = "write_code"
    system_prompt = (
        "You write clean, production-quality code. "
        "Tailor to the target's stack from the dossier. Return only the code."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=2048))


class RunTestsFinger(Finger):
    name = "run_tests"
    system_prompt = "You write pytest tests. Return only valid pytest code."
    def run(self, task: Task) -> Result:
        code = task.context.get("code", "")
        prompt = f"Write tests for:\n{task.goal}"
        if code:
            prompt += f"\n\nCode:\n{code}"
        return self._ok(self._llm(prompt, max_tokens=1500))


class WriteMarkdownFinger(Finger):
    name = "write_markdown"
    system_prompt = "You write clear, well-structured Markdown. Return only the Markdown."
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=2048))


# ── Execution Fists ───────────────────────────────────────────

class CodeFist(ExecutionFist):
    name = "code_fist"
    system_prompt = "You are a senior engineer. Write code informed by the dossier intel."
    def __init__(self):
        self.writer = WriteCodeFinger()
        self.tester = RunTestsFinger()
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx_block = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        code  = self.writer.run(Task(goal=f"{task.goal}\n\n{ctx_block}"))
        tests = self.tester.run(Task(goal=task.goal, context={"code": code.output}))
        return self._ok({"code": code.output, "tests": tests.output})


class WriteFist(ExecutionFist):
    name = "write_fist"
    system_prompt = "You produce written content tailored to the target context."
    def __init__(self):
        self.writer = WriteMarkdownFinger()
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx_block = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        content = self.writer.run(Task(goal=f"{task.goal}\n\n{ctx_block}"))
        return self._ok(content.output)


# ── Wire up ───────────────────────────────────────────────────

def build_clan() -> Clan:
    weatherman = Weatherman(
        rats=[WebRat(), GitHubRat(), NewsRat()],
        intel_fists=[TechFist(), CleanFist()],
    )
    horn = Horn(fists={"code": CodeFist(), "write": WriteFist()})
    horn.system_prompt = (
        "You are the Horn — the execution commander of the clan. "
        "You receive a Dossier and a goal. Plan the strike and split into precise subtasks. "
        "Every subtask must be specific and actionable. "
        'Available fist types: "code" (write code), "write" (produce documents/plans). '
        'Return a JSON list: [{"fist": "<type>", "subtask": "<specific instruction>"}]. '
        "2-4 subtasks maximum."
    )
    return Clan(weatherman=weatherman, horn=horn)

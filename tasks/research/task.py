"""
Deep Research task.
Deploys broad intel-gathering rats, then Horn produces structured research reports.

Good for: technology surveys, market research, academic topic exploration,
          understanding a domain before building in it.

Usage:
  python run.py commission --task research \
    --target "Model Context Protocol" \
    --goal "Comprehensive technical overview: what it is, how it works, who's adopting it" \
    --urls https://modelcontextprotocol.io \
    --github modelcontextprotocol \
    --save-output
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
        "You are a research scout. Extract from page content: "
        "core concepts, definitions, how it works, use cases, limitations, "
        "key people/orgs, related projects. "
        "Return JSON: {core_concepts, how_it_works, use_cases, limitations, "
        "key_orgs, related_projects, notable_quotes}"
    )
    def run(self, task: Task) -> Result:
        urls = task.context.get("urls", [])
        if not urls:
            return self._fail("no URLs provided")
        combined = []
        for url in urls:
            self._log(f"reading {url}")
            try:
                resp = httpx.get(url, follow_redirects=True, timeout=15,
                                 headers={"User-Agent": "Mozilla/5.0"})
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()[:8000]
                combined.append(f"SOURCE: {url}\n{text}")
            except Exception as e:
                self._log(f"failed {url}: {e}")
        if not combined:
            return self._fail("no pages reachable")
        return self._ok(self._llm_json_safe(
            f"Topic: {task.context.get('target')}\nGoal: {task.goal}\n\n"
            + "\n\n---\n\n".join(combined),
            fallback={}, max_tokens=1500
        ))


class GitHubRat(WhiteRat):
    name = "github_rat"
    system_prompt = (
        "You are a GitHub research rat. Extract from repo data: "
        "what the project does, architecture signals, key contributors, "
        "adoption signals (stars/forks), recent activity. "
        "Return JSON: {description, architecture_signals, key_contributors, "
        "adoption_signals, recent_activity, tech_stack}"
    )
    def run(self, task: Task) -> Result:
        org = task.context.get("github_org", "")
        if not org:
            return self._fail("no github_org")
        headers = {"Accept": "application/vnd.github+json"}
        if task.context.get("github_token"):
            headers["Authorization"] = f"Bearer {task.context['github_token']}"
        self._log(f"github/{org}")
        try:
            for ep in [
                f"https://api.github.com/orgs/{org}/repos?per_page=40&sort=updated",
                f"https://api.github.com/users/{org}/repos?per_page=40&sort=updated",
            ]:
                r = httpx.get(ep, headers=headers, timeout=10)
                if r.status_code == 200:
                    break
            repos = r.json() if isinstance(r.json(), list) else []
            slim = [{"name": x.get("name"), "description": x.get("description"),
                     "language": x.get("language"), "stars": x.get("stargazers_count"),
                     "forks": x.get("forks_count"), "topics": x.get("topics", [])}
                    for x in repos if isinstance(x, dict)]
            return self._ok(self._llm_json_safe(
                f"Org: {org}\nRepos:\n{json.dumps(slim, indent=2)}", fallback={}
            ))
        except Exception as e:
            return self._fail(str(e))


class HackerNewsRat(WhiteRat):
    name = "hn_rat"
    system_prompt = (
        "You are a research rat reading HackerNews. Extract: "
        "how the community understands this topic, debates, resources mentioned, "
        "expert opinions, criticism, excitement. "
        "Return JSON: {community_understanding, debates, key_resources, "
        "expert_opinions, criticisms, excitement_signals}"
    )
    def run(self, task: Task) -> Result:
        target = task.context.get("target", "")
        self._log(f"HN: {target}")
        try:
            r = httpx.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": target, "tags": "comment,story", "hitsPerPage": 40},
                timeout=10
            )
            hits = r.json().get("hits", [])
            texts = []
            for h in hits[:25]:
                txt = h.get("comment_text") or h.get("title") or ""
                txt = re.sub(r"<[^>]+>", " ", txt).strip()[:500]
                if txt:
                    texts.append(txt)
            if not texts:
                return self._fail("no HN results")
            return self._ok(self._llm_json_safe(
                f"Topic: {target}\n\nHN content:\n" + "\n---\n".join(texts),
                fallback={}, max_tokens=1500
            ))
        except Exception as e:
            return self._fail(str(e))


class KnowledgeRat(WhiteRat):
    name = "knowledge_rat"
    system_prompt = (
        "You are a research rat drawing on your training knowledge. "
        "Report everything you know about this topic: history, key papers, "
        "key people, technical details, open problems, future directions. "
        "Be thorough and specific. "
        "Return JSON: {history, technical_details, key_papers, key_people, "
        "open_problems, future_directions, related_topics}"
    )
    def run(self, task: Task) -> Result:
        target = task.context.get("target", task.goal)
        self._log(f"knowledge dump: {target}")
        return self._ok(self._llm_json_safe(
            f"Topic: {target}\nResearch goal: {task.goal}\n\n"
            f"Report everything you know. Be specific, not vague.",
            fallback={}, max_tokens=2000
        ))


# ── Intel Fists ───────────────────────────────────────────────

class StructureFist(IntelFist):
    name = "structure_fist"
    system_prompt = (
        "You structure raw research intel into a clean knowledge map. "
        "Return JSON: {what_it_is, how_it_works, why_it_matters, key_components, "
        "adoption_status, limitations, open_questions}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        return self._ok(self._llm_json_safe(
            f"All intel:\n{json.dumps(raw, indent=2)}",
            fallback={}, max_tokens=2000
        ))


class GapFist(IntelFist):
    name = "gap_fist"
    system_prompt = (
        "You identify knowledge gaps and contradictions in research intel. "
        "Return JSON: {confirmed_facts, disputed_facts, knowledge_gaps, "
        "conflicting_sources, recommended_further_reading}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        accum = task.context.get("accumulated_intel", {})
        return self._ok(self._llm_json_safe(
            f"Raw:\n{json.dumps(raw, indent=2)}\nStructured:\n{json.dumps(accum, indent=2)}",
            fallback={}, max_tokens=1500
        ))


# ── Fingers ───────────────────────────────────────────────────

class ExplainerFinger(Finger):
    name = "explainer"
    system_prompt = (
        "You write clear, deep technical explainers. "
        "Start with the core insight, then build up complexity. "
        "Use concrete examples. Write in Markdown."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=3000))


class SummaryFinger(Finger):
    name = "summary"
    system_prompt = (
        "You write tight executive summaries. "
        "Lead with the most important finding. "
        "No padding. Markdown."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=800))


class TLDRFinger(Finger):
    name = "tldr"
    system_prompt = (
        "You write TL;DRs. Max 5 bullet points. Each point one punchy sentence. "
        "The reader should understand the key takeaways in 30 seconds. Markdown."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=400))


# ── Execution Fists ───────────────────────────────────────────

class ReportFist(ExecutionFist):
    name = "report_fist"
    system_prompt = "You produce deep research reports from dossier intelligence."
    def __init__(self):
        self.explainer = ExplainerFinger()
        self.summary   = SummaryFinger()
        self.tldr      = TLDRFinger()
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        full    = self.explainer.run(Task(goal=f"{task.goal}\n\n{ctx}"))
        summary = self.summary.run(Task(goal=f"Summarise: {task.goal}\n\n{ctx}"))
        tldr    = self.tldr.run(Task(goal=f"TL;DR of: {task.goal}\n\n{ctx}"))
        return self._ok({
            "tldr":    tldr.output,
            "summary": summary.output,
            "report":  full.output,
        })


class QuestionsFist(ExecutionFist):
    name = "questions_fist"
    system_prompt = "You generate insightful questions and research directions from a dossier."
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        output = self._llm(
            f"Based on this research dossier, generate 10 sharp, non-obvious questions "
            f"a researcher should investigate next.\n\n{ctx}\n\nGoal: {task.goal}",
            max_tokens=800
        )
        return self._ok(output)


# ── Wire up ───────────────────────────────────────────────────

def build_clan() -> Clan:
    weatherman = Weatherman(
        rats=[WebRat(), GitHubRat(), HackerNewsRat(), KnowledgeRat()],
        intel_fists=[StructureFist(), GapFist()],
    )
    horn = Horn(fists={"report": ReportFist(), "questions": QuestionsFist()})
    horn.system_prompt = (
        "You are the Horn — execution commander of the clan. "
        "You receive a research Dossier and a goal. "
        "Split the goal into 2-3 research output subtasks. "
        'Available fist types: "report" (full technical report with TL;DR + summary + deep explainer), '
        '"questions" (generate research questions and next steps). '
        'Return JSON: [{"fist": "<type>", "subtask": "<specific instruction>"}]'
    )
    return Clan(weatherman=weatherman, horn=horn)

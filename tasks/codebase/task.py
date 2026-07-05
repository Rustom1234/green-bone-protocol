"""
Codebase Analysis task.
Rats read a GitHub repo deeply, Horn produces code review, architecture analysis,
improvement recommendations, and onboarding docs.

Usage:
  python run.py commission --task codebase \
    --target "FastAPI" \
    --goal "Architecture overview and top improvement opportunities" \
    --github tiangolo \
    --save-output

  python run.py commission --task codebase \
    --target "My Repo" \
    --goal "Code review and security audit" \
    --github myorg \
    --token ghp_xxx \
    --save-output
"""

import httpx
import json
import re

from greenbone.types import Task, Result
from greenbone.clan import (
    Clan, Weatherman, Horn,
    WhiteRat, IntelFist, ExecutionFist, Finger,
)


# ── White Rats ────────────────────────────────────────────────

class RepoRat(WhiteRat):
    name = "repo_rat"
    system_prompt = (
        "You analyse GitHub repository metadata. Extract: "
        "purpose, architecture signals, main languages, key directories, "
        "tech decisions, dependency signals, test coverage signals, CI/CD signals. "
        "Return JSON: {purpose, architecture_signals, languages, key_structure, "
        "tech_decisions, dependencies, test_signals, ci_signals}"
    )
    def run(self, task: Task) -> Result:
        org   = task.context.get("github_org", "")
        repo  = task.context.get("github_repo", "")
        token = task.context.get("github_token", "")
        if not org:
            return self._fail("no github_org")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._log(f"github/{org}")
        try:
            # Get repos list
            for ep in [
                f"https://api.github.com/orgs/{org}/repos?per_page=40&sort=updated",
                f"https://api.github.com/users/{org}/repos?per_page=40&sort=updated",
            ]:
                r = httpx.get(ep, headers=headers, timeout=10)
                if r.status_code == 200:
                    break
            repos = r.json() if isinstance(r.json(), list) else []

            # If specific repo, get its details
            details = []
            target_repos = [x for x in repos if isinstance(x, dict)]
            if repo:
                target_repos = [x for x in target_repos if x.get("name") == repo]

            for rep in target_repos[:5]:
                rep_name = rep.get("name")
                # Try to get README
                try:
                    readme_r = httpx.get(
                        f"https://api.github.com/repos/{org}/{rep_name}/readme",
                        headers={**headers, "Accept": "application/vnd.github.raw"},
                        timeout=8
                    )
                    readme = readme_r.text[:3000] if readme_r.status_code == 200 else ""
                except Exception:
                    readme = ""
                details.append({
                    "name": rep_name,
                    "description": rep.get("description"),
                    "language": rep.get("language"),
                    "stars": rep.get("stargazers_count"),
                    "forks": rep.get("forks_count"),
                    "open_issues": rep.get("open_issues_count"),
                    "topics": rep.get("topics", []),
                    "readme_preview": readme[:1500],
                })
            return self._ok(self._llm_json_safe(
                f"Org: {org}\nRepo details:\n{json.dumps(details, indent=2)}",
                fallback={}, max_tokens=1500
            ))
        except Exception as e:
            return self._fail(str(e))


class DocsRat(WhiteRat):
    name = "docs_rat"
    system_prompt = (
        "You extract software architecture and design intent from documentation. "
        "Return JSON: {design_goals, architecture_description, api_patterns, "
        "data_models, integration_points, known_limitations}"
    )
    def run(self, task: Task) -> Result:
        urls = task.context.get("urls", [])
        if not urls:
            return self._fail("no doc URLs provided")
        combined = []
        for url in urls:
            self._log(f"docs: {url}")
            try:
                resp = httpx.get(url, follow_redirects=True, timeout=12,
                                 headers={"User-Agent": "Mozilla/5.0"})
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()[:6000]
                combined.append(f"URL: {url}\n{text}")
            except Exception as e:
                self._log(f"failed {url}: {e}")
        if not combined:
            return self._fail("no docs reachable")
        return self._ok(self._llm_json_safe(
            f"Project: {task.context.get('target')}\n\n" + "\n\n---\n\n".join(combined),
            fallback={}, max_tokens=1500
        ))


class IssuesRat(WhiteRat):
    name = "issues_rat"
    system_prompt = (
        "You analyse GitHub issues to surface: top pain points, recurring bugs, "
        "most-requested features, community frustrations. "
        "Return JSON: {top_pain_points, recurring_bugs, feature_requests, "
        "community_frustrations, positive_signals}"
    )
    def run(self, task: Task) -> Result:
        org   = task.context.get("github_org", "")
        repo  = task.context.get("github_repo", "")
        token = task.context.get("github_token", "")
        if not org:
            return self._fail("no github_org")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Try to find the main repo
        try:
            for ep in [
                f"https://api.github.com/orgs/{org}/repos?per_page=40&sort=stars",
                f"https://api.github.com/users/{org}/repos?per_page=40&sort=stars",
            ]:
                r = httpx.get(ep, headers=headers, timeout=10)
                if r.status_code == 200:
                    break
            repos = r.json() if isinstance(r.json(), list) else []
            if repo:
                chosen = next((x for x in repos if isinstance(x, dict) and x.get("name") == repo), None)
            else:
                # Pick the most starred
                chosen = max((x for x in repos if isinstance(x, dict)), key=lambda x: x.get("stargazers_count", 0), default=None)
            if not chosen:
                return self._fail("no repo found")
            rep_name = chosen.get("name")
            issues_r = httpx.get(
                f"https://api.github.com/repos/{org}/{rep_name}/issues",
                headers=headers,
                params={"state": "open", "per_page": 30, "sort": "comments"},
                timeout=10
            )
            issues = issues_r.json() if issues_r.status_code == 200 else []
            slim = [{"title": x.get("title"), "comments": x.get("comments"),
                     "labels": [l.get("name") for l in x.get("labels", [])],
                     "body_preview": str(x.get("body") or "")[:300]}
                    for x in issues if isinstance(x, dict)]
            self._log(f"{len(slim)} issues from {org}/{rep_name}")
            return self._ok(self._llm_json_safe(
                f"Project: {task.context.get('target')}\nIssues:\n{json.dumps(slim, indent=2)}",
                fallback={}, max_tokens=1500
            ))
        except Exception as e:
            return self._fail(str(e))


class KnowledgeRat(WhiteRat):
    name = "knowledge_rat"
    system_prompt = (
        "You are a senior engineer reporting on a codebase or project from training knowledge. "
        "Cover: known architecture, design patterns used, common issues, community reputation, "
        "how it compares to alternatives. "
        "Return JSON: {architecture, design_patterns, known_issues, reputation, "
        "vs_alternatives, best_practices}"
    )
    def run(self, task: Task) -> Result:
        target = task.context.get("target", task.goal)
        self._log(f"knowledge: {target}")
        return self._ok(self._llm_json_safe(
            f"Project/codebase: {target}\nGoal: {task.goal}\n\n"
            f"Report as a senior engineer who knows this codebase.",
            fallback={}, max_tokens=2000
        ))


# ── Intel Fists ───────────────────────────────────────────────

class ArchFist(IntelFist):
    name = "arch_fist"
    system_prompt = (
        "You synthesise architecture intelligence into a clear picture. "
        "Return JSON: {overall_architecture, layers, key_abstractions, "
        "data_flow, external_dependencies, architectural_strengths, architectural_weaknesses}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        return self._ok(self._llm_json_safe(
            f"All intel:\n{json.dumps(raw, indent=2)}",
            fallback={}, max_tokens=2000
        ))


class QualityFist(IntelFist):
    name = "quality_fist"
    system_prompt = (
        "You assess code quality signals from intelligence data. "
        "Return JSON: {test_coverage_assessment, documentation_quality, "
        "tech_debt_signals, security_signals, performance_signals, "
        "maintainability_score_1_to_10, top_risks}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        accum = task.context.get("accumulated_intel", {})
        return self._ok(self._llm_json_safe(
            f"Raw:\n{json.dumps(raw, indent=2)}\nArchitecture intel:\n{json.dumps(accum, indent=2)}",
            fallback={}, max_tokens=1500
        ))


# ── Fingers ───────────────────────────────────────────────────

class ArchDocFinger(Finger):
    name = "arch_doc"
    system_prompt = (
        "You write architecture documentation. "
        "Cover: what it is, how it's structured, key design decisions, data flow, "
        "extension points, gotchas. Use Markdown with diagrams described in text."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=3000))


class ReviewFinger(Finger):
    name = "code_review"
    system_prompt = (
        "You write thorough code review reports. "
        "Be specific: cite patterns, anti-patterns, concrete recommendations. "
        "Structure: Issues (critical/major/minor), Positives, Recommendations. Markdown."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=2500))


class OnboardFinger(Finger):
    name = "onboarding"
    system_prompt = (
        "You write onboarding guides for new engineers joining a codebase. "
        "Cover: mental model, where to start reading, key files, common patterns, "
        "gotchas, how to run it locally. Markdown."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=2500))


class ImproveFinger(Finger):
    name = "improvements"
    system_prompt = (
        "You write a prioritised list of concrete improvement recommendations for a codebase. "
        "Each item: what to change, why, expected impact, rough effort. "
        "Rank by impact/effort ratio. Markdown."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=2000))


# ── Execution Fists ───────────────────────────────────────────

class ArchFist2(ExecutionFist):
    name = "arch_fist"
    system_prompt = "You produce architecture documentation from dossier intelligence."
    def __init__(self):
        self.doc = ArchDocFinger()
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        return self._ok(self.doc.run(Task(goal=f"{task.goal}\n\n{ctx}")).output)


class ReviewFist(ExecutionFist):
    name = "review_fist"
    system_prompt = "You produce code review reports from dossier intelligence."
    def __init__(self):
        self.review  = ReviewFinger()
        self.improve = ImproveFinger()
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        review  = self.review.run(Task(goal=f"{task.goal}\n\n{ctx}"))
        improve = self.improve.run(Task(goal=f"Improvement recommendations for: {task.goal}\n\n{ctx}"))
        return self._ok({"review": review.output, "improvements": improve.output})


class OnboardFist(ExecutionFist):
    name = "onboard_fist"
    system_prompt = "You produce onboarding documentation from dossier intelligence."
    def __init__(self):
        self.onboard = OnboardFinger()
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        return self._ok(self.onboard.run(Task(goal=f"{task.goal}\n\n{ctx}")).output)


# ── Wire up ───────────────────────────────────────────────────

def build_clan() -> Clan:
    weatherman = Weatherman(
        rats=[RepoRat(), DocsRat(), IssuesRat(), KnowledgeRat()],
        intel_fists=[ArchFist(), QualityFist()],
    )
    horn = Horn(fists={
        "arch":    ArchFist2(),
        "review":  ReviewFist(),
        "onboard": OnboardFist(),
    })
    horn.system_prompt = (
        "You are the Horn — execution commander of the clan. "
        "You receive a codebase analysis Dossier and a goal. "
        "Split the goal into 2-4 specific output subtasks. "
        'Available fist types: '
        '"arch" (architecture documentation), '
        '"review" (code review + improvement recommendations), '
        '"onboard" (onboarding guide for new engineers). '
        'Return JSON: [{"fist": "<type>", "subtask": "<specific instruction>"}]'
    )
    return Clan(weatherman=weatherman, horn=horn)

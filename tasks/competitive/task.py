"""
Competitive Intelligence task.
Deploys maximum rats to build a full competitive dossier on a target company,
then Horn produces a battle card + strategy recommendations.

Usage:
  python run.py commission --task competitive \
    --target "Linear" \
    --goal "Build a competitive battle card vs Jira" \
    --urls https://linear.app https://linear.app/changelog \
    --github linear \
    --save-dossier --save-output
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
        "You are a competitive intelligence scout. Extract from page content: "
        "product positioning, pricing signals, key features, target customers, "
        "differentiators, weaknesses, team signals. "
        "Return JSON: {positioning, pricing, key_features, target_customers, differentiators, weaknesses, team_signals}"
    )
    def run(self, task: Task) -> Result:
        urls = task.context.get("urls", [])
        if not urls:
            return self._fail("no URLs")
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
            return self._fail("no pages reachable")
        return self._ok(self._llm_json_safe(
            f"Target: {task.context.get('target')}\n\n" + "\n\n---\n\n".join(combined),
            fallback={}
        ))


class GitHubRat(WhiteRat):
    name = "github_rat"
    system_prompt = (
        "You are a competitive GitHub rat. Extract signals that reveal engineering quality, "
        "team size, velocity, tech choices. "
        "Return JSON: {tech_stack, repo_count, most_active_repos, commit_velocity_signal, "
        "open_issues_signal, contributor_count_signal, observations}"
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
            slim = [{"name": x.get("name"), "language": x.get("language"),
                     "stars": x.get("stargazers_count"), "forks": x.get("forks_count"),
                     "open_issues": x.get("open_issues_count"),
                     "topics": x.get("topics", [])} for x in repos if isinstance(x, dict)]
            return self._ok(self._llm_json_safe(
                f"Org: {org}\nRepos:\n{json.dumps(slim, indent=2)}", fallback={}
            ))
        except Exception as e:
            return self._fail(str(e))


class HackerNewsRat(WhiteRat):
    name = "hn_rat"
    system_prompt = (
        "You are an intelligence rat reading HackerNews discussion data. "
        "Extract: community sentiment, criticisms, praise, mentioned competitors, key debates. "
        "Return JSON: {sentiment, criticisms, praise, competitors_mentioned, key_debates, notable_comments}"
    )
    def run(self, task: Task) -> Result:
        target = task.context.get("target", "")
        self._log(f"searching HN for: {target}")
        try:
            r = httpx.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": target, "tags": "comment,story", "hitsPerPage": 30},
                timeout=10
            )
            hits = r.json().get("hits", [])
            texts = []
            for h in hits[:20]:
                txt = h.get("comment_text") or h.get("title") or ""
                txt = re.sub(r"<[^>]+>", " ", txt).strip()[:400]
                if txt:
                    texts.append(txt)
            if not texts:
                return self._fail("no HN results")
            return self._ok(self._llm_json_safe(
                f"Target: {target}\n\nHN comments/posts:\n" + "\n---\n".join(texts),
                fallback={}
            ))
        except Exception as e:
            return self._fail(str(e))


class NewsRat(WhiteRat):
    name = "news_rat"
    system_prompt = (
        "You are a competitive news rat. From your training data, report on this company: "
        "funding history, key hires, product launches, controversies, market position, growth signals. "
        "Return JSON: {funding_history, key_hires, product_launches, controversies, market_position, growth_signals}"
    )
    def run(self, task: Task) -> Result:
        target = task.context.get("target", task.goal)
        self._log(f"news scan: {target}")
        return self._ok(self._llm_json_safe(
            f"Target company: {target}\nReport all competitive intelligence you know.",
            fallback={}
        ))


class PricingRat(WhiteRat):
    name = "pricing_rat"
    system_prompt = (
        "You extract pricing intelligence. Look for: tiers, price points, free vs paid limits, "
        "enterprise signals, pricing model (per-seat, usage, flat). "
        "Return JSON: {model, tiers, free_tier, enterprise_signals, price_points, comparison_notes}"
    )
    def run(self, task: Task) -> Result:
        target = task.context.get("target", "")
        urls = [u for u in task.context.get("urls", []) if "pric" in u.lower()]
        # If no pricing URL given, try to guess
        if not urls and task.context.get("urls"):
            base = task.context["urls"][0].rstrip("/")
            urls = [f"{base}/pricing"]
        combined = []
        for url in urls[:2]:
            self._log(f"pricing page: {url}")
            try:
                resp = httpx.get(url, follow_redirects=True, timeout=10,
                                 headers={"User-Agent": "Mozilla/5.0"})
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()[:5000]
                combined.append(text)
            except Exception as e:
                self._log(f"failed: {e}")
        prompt = f"Target: {target}\n"
        if combined:
            prompt += "Pricing page content:\n" + "\n".join(combined)
        else:
            prompt += "No page content available. Use your training data."
        return self._ok(self._llm_json_safe(prompt, fallback={}))


# ── Intel Fists ───────────────────────────────────────────────

class TechFist(IntelFist):
    name = "tech_fist"
    system_prompt = (
        "You identify and classify technology stacks from competitive intelligence. "
        "Return JSON: {confirmed: [], inferred: [], notable_choices: [], tech_debt_signals: []}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        return self._ok(self._llm_json_safe(
            f"All raw intel:\n{json.dumps(raw, indent=2)}",
            fallback={"confirmed": [], "inferred": [], "notable_choices": [], "tech_debt_signals": []}
        ))


class SentimentFist(IntelFist):
    name = "sentiment_fist"
    system_prompt = (
        "You synthesise community sentiment and competitive positioning from intelligence data. "
        "Return JSON: {overall_sentiment, strengths, weaknesses, community_complaints, "
        "community_love, vs_competitors, switching_signals}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        accum = task.context.get("accumulated_intel", {})
        return self._ok(self._llm_json_safe(
            f"Raw intel:\n{json.dumps(raw, indent=2)}\nPrior processed:\n{json.dumps(accum, indent=2)}",
            max_tokens=1500, fallback={}
        ))


class CleanFist(IntelFist):
    name = "clean_fist"
    system_prompt = (
        "You merge all competitive intelligence into a clean, non-redundant summary. "
        "Return JSON: {executive_summary, product_summary, tech_stack_flat, pricing_summary, "
        "key_people, strengths, weaknesses, opportunities, threats}"
    )
    def run(self, task: Task) -> Result:
        raw = task.context.get("raw_intel", [])
        accum = task.context.get("accumulated_intel", {})
        return self._ok(self._llm_json_safe(
            f"Raw intel:\n{json.dumps(raw, indent=2)}\nPrior processed:\n{json.dumps(accum, indent=2)}",
            max_tokens=2000, fallback={}
        ))


# ── Fingers ───────────────────────────────────────────────────

class BattleCardFinger(Finger):
    name = "battle_card"
    system_prompt = (
        "You write sharp competitive battle cards for sales teams. "
        "Include: elevator pitch of target, their strengths to acknowledge, "
        "their weaknesses to exploit, your counter-talking-points, landmine questions to ask. "
        "Write in Markdown. Be specific and punchy."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=2048))


class StrategyFinger(Finger):
    name = "strategy"
    system_prompt = (
        "You write sharp strategic recommendations. "
        "Be direct. Reference specific facts. No fluff. "
        "Write in Markdown with clear headers."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=2048))


class SWOTFinger(Finger):
    name = "swot"
    system_prompt = (
        "You produce a tight SWOT analysis. Be specific, not generic. "
        "Each point must be concrete and fact-based. Return Markdown."
    )
    def run(self, task: Task) -> Result:
        return self._ok(self._llm(task.goal, max_tokens=1200))


# ── Execution Fists ───────────────────────────────────────────

class BattleCardFist(ExecutionFist):
    name = "battle_card_fist"
    system_prompt = "You produce competitive battle cards using dossier intel."
    def __init__(self):
        self.card   = BattleCardFinger()
        self.strat  = StrategyFinger()
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        card  = self.card.run(Task(goal=f"{task.goal}\n\n{ctx}"))
        strat = self.strat.run(Task(goal=f"Strategic recommendations for: {task.goal}\n\n{ctx}"))
        return self._ok({"battle_card": card.output, "strategy": strat.output})


class SWOTFist(ExecutionFist):
    name = "swot_fist"
    system_prompt = "You produce SWOT analyses using dossier intel."
    def __init__(self):
        self.swot = SWOTFinger()
    def run(self, task: Task) -> Result:
        dossier = task.context.get("dossier")
        ctx = dossier.to_context_block() if dossier else ""
        self._log(task.goal[:70])
        result = self.swot.run(Task(goal=f"{task.goal}\n\n{ctx}"))
        return self._ok(result.output)


# ── Wire up ───────────────────────────────────────────────────

def build_clan() -> Clan:
    weatherman = Weatherman(
        rats=[WebRat(), GitHubRat(), HackerNewsRat(), NewsRat(), PricingRat()],
        intel_fists=[TechFist(), SentimentFist(), CleanFist()],
    )
    horn = Horn(fists={"battle_card": BattleCardFist(), "swot": SWOTFist()})
    horn.system_prompt = (
        "You are the Horn, execution commander of the clan. "
        "You receive a competitive intelligence Dossier and a goal. "
        "Split the goal into 2-4 specific competitive analysis subtasks. "
        'Available fist types: "battle_card" (competitive battle card + strategy), '
        '"swot" (SWOT analysis). '
        'Return JSON: [{"fist": "<type>", "subtask": "<specific instruction>"}]'
    )
    return Clan(weatherman=weatherman, horn=horn)

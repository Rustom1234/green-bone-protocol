#!/usr/bin/env python3
"""
tools/estimate_tokens.py — offline, approximate token/cost estimator for the
Green Bone Protocol tiered pipeline.

WHAT THIS IS
------------
An APPROXIMATION, not a measurement. It never calls the Anthropic API (not
even the free `count_tokens` endpoint) — the user has forbidden network
calls for this tool. Instead it:

  1. Tokenizes REAL logged response text (the sealed Dossier in
     output/anthropic_dossier.json, and the Execution-Fist / Finger outputs
     in output/anthropic_output.json) with `tiktoken`'s `cl100k_base`
     encoding — OpenAI's tokenizer, used here purely as an off-the-shelf,
     offline BPE approximation of Claude's actual (undisclosed) tokenizer.
  2. Reconstructs the INPUT/prompt text for every call in the pipeline from
     the actual code (system prompts + f-string templates in
     greenbone/clan.py and tasks/research/task.py), filled in with real
     values (target, dossier context blocks, subtask fragments) wherever
     those are available from the logs/JSON, and clearly-flagged HEURISTIC
     estimates wherever the real text was never logged (see ASSUMPTIONS).

THE COST-REDUCTION NUMBER IS ROBUST TO TOKENIZER BIAS
------------------------------------------------------
Both configs below (TIERED and ALL-COMMAND) use the *exact same* token
counts — the Green Bone Protocol split does not change how many tokens are
sent, only which model tier is billed for them. So any systematic over/under
count from using cl100k_base instead of Claude's real tokenizer applies
equally to both configs' numerators and denominators and **cancels out of
the % cost-reduction ratio**. The ratio is far more trustworthy than either
absolute dollar figure.

WHICH RUN THIS RECONSTRUCTS
----------------------------
output/run.log and output/run2.log show the real logged run was made with
`tasks/research/task.py` (agents "report_fist", "questions_fist", "hn_rat",
"knowledge_rat" only exist there — NOT in tasks/default/task.py). Rats:
WebRat, GitHubRat, HackerNewsRat, KnowledgeRat. Intel fists: StructureFist,
GapFist. Horn fists: ReportFist ("report"), QuestionsFist ("questions").
run2.log shows `hn_rat` returned "no data" and NEVER reached its LLM call
(HackerNewsRat.run returns self._fail(...) before calling the model if the
HN Algolia search yields no hits) — so it contributes ZERO tokens and is
excluded from the Field-tier call list below. This matches the log's own
tally: "3 intel packages from rats".

ASSUMPTIONS (read before trusting the numbers — flagged again inline)
-----------------------------------------------------------------------
A1. Unlogged Field-tier JSON-extraction completions (rats, intel fists)
    are assumed to use 55% of their configured `max_tokens` as output —
    a documented midpoint of a typical 40-70% fill range for structured
    extraction tasks. This is a HEURISTIC, not tokenized real text.
A2. WebRat's scraped-page input is assumed to fill 70% of the code's hard
    per-page cap (`text[:8000]` chars) for the one URL actually logged
    (https://anthropic.com).
A3. GitHubRat's slimmed repo-JSON input assumes ~15 repos (a typical org
    size guess, not verified — the actual `anthropics` org repo count
    wasn't logged and we do not hit the GitHub API here) at ~80 chars/repo.
A4. HackerNewsRat is EXCLUDED (see above) — it never called the model.
A5. KnowledgeRat's input has no external data dependency, so it is
    computed exactly from the real code template (not a heuristic).
A6. Horn's plan-output JSON uses the REAL subtask text as logged, but
    run2.log truncates each subtask at 70 chars (`task.goal[:70]` in
    clan.py's `_log` call). This likely UNDERCOUNTS Horn's true output
    tokens somewhat. Immaterial to totals: Horn is 1 of only 2 Command
    calls, and the Weatherman's dossier-synthesis call dominates Command
    volume.
A7. The exact CLI `--goal` string used for this commission was not
    captured in the run logs (only `target=Anthropic` and truncated
    subtask fragments were). A representative goal consistent with the
    observed subtasks/dossier is substituted here — ASSUMED, flagged.
A8. Heuristic sub-estimates (A1-A3) that need a chars->tokens conversion
    use a flat 4 chars/token rule of thumb — NOT tiktoken — because there
    is no real text to tokenize for those spans. Every span with real
    logged text is tokenized directly with tiktoken; this heuristic is
    only ever used to size the small handful of never-logged spans.
A9. Steel (the deterministic validation layer) adds ~0 tokens — it is
    pure Python (regex/ast/json parsing), no LLM calls except an optional
    rare Haiku repair pass that did not fire in this run. Excluded/shown
    as 0.

Run: `python3 tools/estimate_tokens.py` (uses the repo venv's tiktoken;
falls back to a chars/4 heuristic for the whole script if tiktoken can't
be imported at all — see TOKENIZER_LABEL in the output).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --- Offline tiktoken setup ---------------------------------------------
# Point tiktoken at a repo-local cache so it NEVER needs a network call to
# fetch the cl100k_base BPE ranks file, even on first run in a fresh clone
# (the cache file here was fetched once, out-of-band, the same way `pip
# install tiktoken` fetches the package itself — not a per-run network call
# and not a call to any Anthropic endpoint).
CACHE_DIR = REPO / ".tiktoken_cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(CACHE_DIR))

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))

    TOKENIZER_LABEL = "tiktoken cl100k_base (offline, cached at .tiktoken_cache/)"
    USED_TIKTOKEN = True
except Exception as e:  # pragma: no cover - fallback path
    def count_tokens(text: str) -> int:
        return max(1, len(text or "") // 4)

    TOKENIZER_LABEL = f"FALLBACK heuristic len(text)/4 chars-per-token (tiktoken unavailable: {e})"
    USED_TIKTOKEN = False


def chars_to_tokens_heuristic(n_chars: float) -> int:
    """A1/A2/A3/A8: 4 chars/token rule of thumb, used ONLY for spans with
    no real logged text to tokenize (see module docstring)."""
    return round(n_chars / 4)


# --- Pull real system prompts / templates straight from the source code -
# (imported, not re-typed, so this script can't silently drift from the
# real prompts as the codebase evolves)
os.environ.setdefault("ANTHROPIC_API_KEY", "")  # agents.py builds a client at import time
from greenbone.types import Dossier                      # noqa: E402
from greenbone.clan import Weatherman, Horn               # noqa: E402
import tasks.research.task as rt                          # noqa: E402

WEATHERMAN_SYSTEM_PROMPT = Weatherman.system_prompt
HORN_SYSTEM_PROMPT_OVERRIDE = rt.build_clan().horn.system_prompt  # research/task.py override

PRICES = {
    # $ per token (published, current)
    "sonnet": {"in": 3.00 / 1_000_000, "out": 15.00 / 1_000_000},
    "haiku":  {"in": 1.00 / 1_000_000, "out": 5.00 / 1_000_000},
}


@dataclass
class Call:
    name: str
    tier: str            # "COMMAND" or "FIELD"
    input_tokens: int
    output_tokens: int
    input_basis: str      # provenance label
    output_basis: str
    note: str = ""


calls: list[Call] = []

# =========================================================================
# Load the real logged data
# =========================================================================
dossier_dict = json.loads((REPO / "output" / "anthropic_dossier.json").read_text())
dossier = Dossier(
    target=dossier_dict["target"],
    website_summary=dossier_dict["website_summary"],
    github_summary=dossier_dict["github_summary"],
    news_summary=dossier_dict["news_summary"],
    tech_stack=dossier_dict["tech_stack"],
    key_findings=dossier_dict["key_findings"],
    recommended_approach=dossier_dict["recommended_approach"],
)
DOSSIER_CTX_BLOCK = dossier.to_context_block()  # real, exact code path

fist_outputs = json.loads((REPO / "output" / "anthropic_output.json").read_text())
assert [o["agent"] for o in fist_outputs] == ["report_fist", "report_fist", "questions_fist"]

# Real (truncated-at-70-chars, per clan.py's `task.goal[:70]` logging) subtask
# fragments, parsed straight out of the real run log rather than re-typed.
run_log = (REPO / "output" / "run2.log").read_text()
subtask_fragments = re.findall(r"\[FIST:(\w+)\]\s(.*)", run_log)
assert len(subtask_fragments) == 3, subtask_fragments
SUBTASKS = [frag for _, frag in subtask_fragments]  # in run order: report, report, questions

TARGET = dossier.target  # "Anthropic" — real
# A7: exact CLI goal string was never logged; reconstructed to match the
# subtasks/dossier actually observed. ASSUMED.
GOAL = (
    "Comprehensive technical overview of Anthropic: what they build, how Claude "
    "and Constitutional AI work, who's adopting it, and open research questions."
)

# =========================================================================
# COMMAND tier (Sonnet): Weatherman dossier-synthesis + Horn planning
# =========================================================================

# --- Call 1: Weatherman._synthesise_dossier ------------------------------
# Real: target, the 3 successful rat names, the literal template text from
# clan.py, and the Dossier's actual field content (reformatted back into the
# WEBSITE:/GITHUB:/NEWS:/TECH_STACK:/FINDINGS:/APPROACH: layout the prompt
# asked for — the real completion, before clan.py's parser split it apart).
# Heuristic: the `cleaned` intel-fist JSON block, which was never logged
# (see A1) — added as a token COUNT, not fabricated text.
raw_intel_sources = ["web_rat", "github_rat", "knowledge_rat"]  # hn_rat excluded, see A4

structure_fist_out_tok_heuristic = round(0.55 * 2000)  # A1, StructureFist max_tokens=2000
gap_fist_out_tok_heuristic = round(0.55 * 1500)         # A1, GapFist max_tokens=1500
cleaned_intel_heuristic_tokens = structure_fist_out_tok_heuristic + gap_fist_out_tok_heuristic

weatherman_prompt_template = (
    f"Target: {TARGET}\nGoal: {GOAL}\n\n"
    f"Cleaned intel:\n{{cleaned}}\n\n"
    f"Raw intel sources: {raw_intel_sources}\n\n"
    f"Write the sealed Dossier in EXACTLY this format:\n\n"
    f"WEBSITE: one dense sentence on their web/product presence\n"
    f"GITHUB: one sentence on their code signals (or 'No GitHub data')\n"
    f"NEWS: one sentence on recent events (or 'No news data')\n"
    f"TECH_STACK: comma-separated list of technologies\n"
    f"FINDINGS:\n- finding one\n- finding two\n- finding three\n"
    f"APPROACH: 2-3 sentences telling Horn exactly how to execute the goal. "
    f"Be specific. Reference tech stack and findings."
)
weatherman_input_template_tokens = count_tokens(
    WEATHERMAN_SYSTEM_PROMPT + "\n" + weatherman_prompt_template.replace("{cleaned}", "")
)
weatherman_input_tokens = weatherman_input_template_tokens + cleaned_intel_heuristic_tokens

dossier_raw_completion = (
    f"WEBSITE: {dossier.website_summary}\n"
    f"GITHUB: {dossier.github_summary}\n"
    f"NEWS: {dossier.news_summary}\n"
    f"TECH_STACK: {', '.join(dossier.tech_stack)}\n"
    f"FINDINGS:\n" + "\n".join(f"- {f}" for f in dossier.key_findings) + "\n"
    f"APPROACH: {dossier.recommended_approach}"
)
weatherman_output_tokens = count_tokens(dossier_raw_completion)

calls.append(Call(
    name="weatherman._synthesise_dossier",
    tier="COMMAND",
    input_tokens=weatherman_input_tokens,
    output_tokens=weatherman_output_tokens,
    input_basis="real template+target+sources (tiktoken) + HEURISTIC cleaned-intel size (A1)",
    output_basis="REAL (tiktoken on actual sealed Dossier content, reformatted to prompt's requested layout)",
))

# --- Call 2: Horn.run's planning _llm_json -------------------------------
fist_types_str = '"report", "questions"'  # real, from tasks/research/task.py build_clan()
horn_prompt_template = (
    f"Goal: {GOAL}\n\n{DOSSIER_CTX_BLOCK}\n\n"
    f"Split this goal into specific subtasks. Available fist types: {fist_types_str}"
)
horn_input_tokens = count_tokens(HORN_SYSTEM_PROMPT_OVERRIDE + "\n" + horn_prompt_template)

horn_plan_output = json.dumps([
    {"fist": "report", "subtask": SUBTASKS[0]},
    {"fist": "report", "subtask": SUBTASKS[1]},
    {"fist": "questions", "subtask": SUBTASKS[2]},
])
horn_output_tokens = count_tokens(horn_plan_output)

calls.append(Call(
    name="horn.run._llm_json (planning)",
    tier="COMMAND",
    input_tokens=horn_input_tokens,
    output_tokens=horn_output_tokens,
    input_basis="REAL (tiktoken: real dossier context block + real template + real fist types)",
    output_basis="REAL but TRUNCATED subtasks (A6, run2.log logs only task.goal[:70]) -> likely undercount",
))

# =========================================================================
# FIELD tier (Haiku): White Rats, Intel Fists, Execution Fists, Fingers
# =========================================================================

# --- Rats -----------------------------------------------------------------
# web_rat: 1 URL logged (https://anthropic.com). Page text itself was never
# logged (only scraped live); heuristic-sized per A2.
web_rat_template = f"Topic: {TARGET}\nGoal: {GOAL}\n\nSOURCE: https://anthropic.com\n{{page_text}}"
web_rat_template_tokens = count_tokens(rt.WebRat.system_prompt + "\n" + web_rat_template.replace("{page_text}", ""))
web_rat_page_heuristic_tokens = chars_to_tokens_heuristic(0.70 * 8000)  # A2
web_rat_input_tokens = web_rat_template_tokens + web_rat_page_heuristic_tokens
web_rat_output_tokens = round(0.55 * 1500)  # A1

calls.append(Call(
    name="web_rat", tier="FIELD",
    input_tokens=web_rat_input_tokens, output_tokens=web_rat_output_tokens,
    input_basis="real template/system-prompt (tiktoken) + HEURISTIC scraped-page size (A2)",
    output_basis="HEURISTIC (A1) - page text never logged",
))

# github_rat: org "anthropics" logged; actual repo JSON never logged (no
# network call made here to fetch it - heuristic per A3).
github_rat_template = f"Org: anthropics\nRepos:\n{{slim_json}}"
github_rat_template_tokens = count_tokens(rt.GitHubRat.system_prompt + "\n" + github_rat_template.replace("{slim_json}", ""))
github_rat_repo_heuristic_tokens = chars_to_tokens_heuristic(15 * 80)  # A3
github_rat_input_tokens = github_rat_template_tokens + github_rat_repo_heuristic_tokens
github_rat_output_tokens = round(0.55 * 1024)  # A1, default max_tokens

calls.append(Call(
    name="github_rat", tier="FIELD",
    input_tokens=github_rat_input_tokens, output_tokens=github_rat_output_tokens,
    input_basis="real template/system-prompt (tiktoken) + HEURISTIC repo-JSON size (A3)",
    output_basis="HEURISTIC (A1) - repo JSON never logged",
))

# knowledge_rat: no external data dependency -> fully real/exact template.
knowledge_rat_prompt = (
    f"Topic: {TARGET}\nResearch goal: {GOAL}\n\n"
    f"Report everything you know. Be specific, not vague."
)
knowledge_rat_input_tokens = count_tokens(rt.KnowledgeRat.system_prompt + "\n" + knowledge_rat_prompt)
knowledge_rat_output_tokens = round(0.55 * 2000)  # A1

calls.append(Call(
    name="knowledge_rat", tier="FIELD",
    input_tokens=knowledge_rat_input_tokens, output_tokens=knowledge_rat_output_tokens,
    input_basis="REAL (tiktoken, no external data dependency - exact template)",
    output_basis="HEURISTIC (A1) - knowledge dump never logged",
))

# hn_rat: EXCLUDED. run2.log: "rat [hn_rat] returned no data" -- HN Algolia
# search returned no hits, so HackerNewsRat.run() hit `return
# self._fail("no HN results")` BEFORE ever calling the model. Zero tokens.

# --- Intel fists ------------------------------------------------------------
raw_intel_heuristic_tokens = web_rat_output_tokens + github_rat_output_tokens + knowledge_rat_output_tokens

structure_fist_template = f"All intel:\n{{raw_intel_json}}"
structure_fist_template_tokens = count_tokens(rt.StructureFist.system_prompt + "\n" + structure_fist_template.replace("{raw_intel_json}", ""))
structure_fist_input_tokens = structure_fist_template_tokens + raw_intel_heuristic_tokens
structure_fist_output_tokens = structure_fist_out_tok_heuristic

calls.append(Call(
    name="structure_fist", tier="FIELD",
    input_tokens=structure_fist_input_tokens, output_tokens=structure_fist_output_tokens,
    input_basis="real template/system-prompt (tiktoken) + HEURISTIC raw_intel size (= sum of rats' heuristic outputs)",
    output_basis="HEURISTIC (A1)",
))

gap_fist_template = f"Raw:\n{{raw_intel_json}}\nStructured:\n{{accum_json}}"
gap_fist_template_tokens = count_tokens(rt.GapFist.system_prompt + "\n" + gap_fist_template.replace("{raw_intel_json}", "").replace("{accum_json}", ""))
gap_fist_input_tokens = gap_fist_template_tokens + raw_intel_heuristic_tokens + structure_fist_output_tokens
gap_fist_output_tokens = gap_fist_out_tok_heuristic

calls.append(Call(
    name="gap_fist", tier="FIELD",
    input_tokens=gap_fist_input_tokens, output_tokens=gap_fist_output_tokens,
    input_basis="real template/system-prompt (tiktoken) + HEURISTIC raw_intel/structured size",
    output_basis="HEURISTIC (A1)",
))

# --- Execution fists + Fingers (REAL logged outputs) -----------------------
# ReportFist fans out to 3 real Haiku Finger calls; each Finger's OUTPUT is
# real logged text from output/anthropic_output.json. Each Finger's INPUT is
# the real code template + the real dossier context block + the real
# (truncated) subtask fragment.

def add_report_fist_calls(idx: int, subtask: str) -> None:
    out = fist_outputs[idx]["output"]
    # ExplainerFinger
    explainer_input = f"{subtask}\n\n{DOSSIER_CTX_BLOCK}"
    calls.append(Call(
        name=f"report_fist[{idx}].explainer", tier="FIELD",
        input_tokens=count_tokens(rt.ExplainerFinger.system_prompt + "\n" + explainer_input),
        output_tokens=count_tokens(out["report"]),
        input_basis="real template (tiktoken) + real dossier ctx + TRUNCATED real subtask (A6)",
        output_basis="REAL (tiktoken on logged output.json 'report' field)",
    ))
    # SummaryFinger
    summary_input = f"Summarise: {subtask}\n\n{DOSSIER_CTX_BLOCK}"
    calls.append(Call(
        name=f"report_fist[{idx}].summary", tier="FIELD",
        input_tokens=count_tokens(rt.SummaryFinger.system_prompt + "\n" + summary_input),
        output_tokens=count_tokens(out["summary"]),
        input_basis="real template (tiktoken) + real dossier ctx + TRUNCATED real subtask (A6)",
        output_basis="REAL (tiktoken on logged output.json 'summary' field)",
    ))
    # TLDRFinger
    tldr_input = f"TL;DR of: {subtask}\n\n{DOSSIER_CTX_BLOCK}"
    calls.append(Call(
        name=f"report_fist[{idx}].tldr", tier="FIELD",
        input_tokens=count_tokens(rt.TLDRFinger.system_prompt + "\n" + tldr_input),
        output_tokens=count_tokens(out["tldr"]),
        input_basis="real template (tiktoken) + real dossier ctx + TRUNCATED real subtask (A6)",
        output_basis="REAL (tiktoken on logged output.json 'tldr' field)",
    ))


add_report_fist_calls(0, SUBTASKS[0])
add_report_fist_calls(1, SUBTASKS[1])

# QuestionsFist (direct LLM call, no Fingers)
questions_subtask = SUBTASKS[2]
questions_prompt = (
    f"Based on this research dossier, generate 10 sharp, non-obvious questions "
    f"a researcher should investigate next.\n\n{DOSSIER_CTX_BLOCK}\n\nGoal: {questions_subtask}"
)
questions_input_tokens = count_tokens(rt.QuestionsFist.system_prompt + "\n" + questions_prompt)
questions_output_tokens = count_tokens(fist_outputs[2]["output"])

calls.append(Call(
    name="questions_fist", tier="FIELD",
    input_tokens=questions_input_tokens, output_tokens=questions_output_tokens,
    input_basis="real template (tiktoken) + real dossier ctx + TRUNCATED real subtask (A6)",
    output_basis="REAL (tiktoken on logged output.json questions_fist output)",
))

# =========================================================================
# Aggregate per tier, then price under both configs
# =========================================================================
command_calls = [c for c in calls if c.tier == "COMMAND"]
field_calls = [c for c in calls if c.tier == "FIELD"]

command_in = sum(c.input_tokens for c in command_calls)
command_out = sum(c.output_tokens for c in command_calls)
field_in = sum(c.input_tokens for c in field_calls)
field_out = sum(c.output_tokens for c in field_calls)

def cost(tokens_in, tokens_out, price_key):
    p = PRICES[price_key]
    return tokens_in * p["in"], tokens_out * p["out"]

# TIERED (Green Bone Protocol): Command @ Sonnet, Field @ Haiku
tiered_cmd_in_cost, tiered_cmd_out_cost = cost(command_in, command_out, "sonnet")
tiered_fld_in_cost, tiered_fld_out_cost = cost(field_in, field_out, "haiku")
tiered_in_cost = tiered_cmd_in_cost + tiered_fld_in_cost
tiered_out_cost = tiered_cmd_out_cost + tiered_fld_out_cost
tiered_total = tiered_in_cost + tiered_out_cost

# ALL-COMMAND baseline: identical token counts, everything priced at Sonnet
allcmd_in_cost, allcmd_out_cost = cost(command_in + field_in, command_out + field_out, "sonnet")
allcmd_total = allcmd_in_cost + allcmd_out_cost

pct_reduction = (allcmd_total - tiered_total) / allcmd_total * 100

# =========================================================================
# Output: markdown report to stdout + JSON artifact
# =========================================================================
lines = []
lines.append("# Green Bone Protocol — Approximate Token/Cost Estimate\n")
lines.append(
    "**APPROXIMATE — derived from tokenizing logged text with tiktoken "
    "cl100k_base, NOT a measurement of real Anthropic API `usage` fields.**\n"
)
lines.append(f"Tokenizer: `{TOKENIZER_LABEL}`\n")
lines.append(
    "Robustness note: token counts are IDENTICAL between the two configs "
    "below (same prompts/responses) — the tiered split changes only which "
    "price applies, so any systematic tokenizer bias cancels out of the "
    "% cost-reduction ratio even though it may skew the absolute $ figures.\n"
)

lines.append("## Per-tier token totals\n")
lines.append("| Tier | Model | Calls | Input tokens (approx) | Output tokens (approx) |")
lines.append("|---|---|---:|---:|---:|")
lines.append(f"| Command (PERCEPTION+CHANNELING) | claude-sonnet-4-6 | {len(command_calls)} | {command_in:,} | {command_out:,} |")
lines.append(f"| Field (LIGHTNESS) | claude-haiku-4-5 | {len(field_calls)} | {field_in:,} | {field_out:,} |")
lines.append(f"| Steel (STEEL) | deterministic, no LLM | n/a | ~0 | ~0 |")
lines.append("")

lines.append("## Cost by configuration\n")
lines.append("| Config | Input $ | Output $ | Total $ |")
lines.append("|---|---:|---:|---:|")
lines.append(f"| **Tiered (Green Bone Protocol)** | ${tiered_in_cost:.4f} | ${tiered_out_cost:.4f} | ${tiered_total:.4f} |")
lines.append(f"| **All-Command (baseline)** | ${allcmd_in_cost:.4f} | ${allcmd_out_cost:.4f} | ${allcmd_total:.4f} |")
lines.append("")
lines.append(f"**% cost reduction, Tiered vs All-Command: {pct_reduction:.1f}%**\n")

lines.append("## Per-call breakdown (provenance-labeled)\n")
lines.append("| Call | Tier | In tok | Out tok | Input basis | Output basis |")
lines.append("|---|---|---:|---:|---|---|")
for c in calls:
    lines.append(f"| {c.name} | {c.tier} | {c.input_tokens:,} | {c.output_tokens:,} | {c.input_basis} | {c.output_basis} |")

lines.append("\n## Key assumptions / provenance notes (see module docstring for full A1-A9 list)\n")
lines.append(
    "- Reconstructed from `tasks/research/task.py` (the run.log/run2.log agent names "
    "`report_fist`/`questions_fist`/`hn_rat`/`knowledge_rat` only exist there, not in "
    "`tasks/default/task.py`)."
)
lines.append(
    "- `hn_rat` (HackerNewsRat) is EXCLUDED: run2.log logs \"rat [hn_rat] returned no data\" "
    "and its code path returns a failure BEFORE calling the model when the HN search has no "
    "hits, so it contributes 0 tokens. 3 rats (web_rat, github_rat, knowledge_rat) + 2 intel "
    "fists + 7 fist/finger calls (2x ReportFist -> 3 Fingers each, + 1x QuestionsFist) = 12 Field calls."
)
lines.append(
    "- The exact `--goal` CLI string wasn't captured in the logs; a representative goal "
    "consistent with the logged subtasks/dossier is substituted (flagged ASSUMED, A7)."
)
lines.append(
    "- Spans with no real logged text (rat/intel-fist outputs, scraped page/GitHub/HN content) "
    "use documented heuristics (55% of configured max_tokens for completions, A1; 70% of the "
    "8000-char page cap for the one logged URL, A2; ~15 repos x 80 chars for GitHub, A3) rather "
    "than fabricated text. Horn's plan output uses real but 70-char-truncated subtask text (A6), "
    "likely a small undercount, immaterial to totals."
)
lines.append(
    "- All averages/ratios above are clearly-labeled heuristics, not measurements; every call "
    "row's Input/Output basis column states exactly which parts are REAL (tokenized logged text "
    "or exact code templates) vs HEURISTIC."
)

report_md = "\n".join(lines)
print(report_md)

# --- Machine-readable artifact -------------------------------------------
result = {
    "methodology": (
        "APPROXIMATE — tokens derived by tokenizing logged text (output/anthropic_dossier.json, "
        "output/anthropic_output.json) with tiktoken cl100k_base as an offline stand-in for "
        "Claude's real (undisclosed) tokenizer, plus reconstructed prompt templates from the "
        "actual pipeline code (greenbone/clan.py, tasks/research/task.py). NOT a measurement of "
        "real Anthropic API `usage` fields. Token counts are identical across the Tiered and "
        "All-Command configs by construction; only the price applied differs, so tokenizer bias "
        "cancels out of the %-cost-reduction ratio. See tools/estimate_tokens.py module docstring "
        "for the full list of labeled assumptions (A1-A9)."
    ),
    "tokenizer": TOKENIZER_LABEL,
    "used_tiktoken": USED_TIKTOKEN,
    "reconstructed_from_run": "tasks/research/task.py (identified via output/run.log, output/run2.log)",
    "per_tier_tokens": {
        "command": {"calls": len(command_calls), "input_tokens": command_in, "output_tokens": command_out},
        "field": {"calls": len(field_calls), "input_tokens": field_in, "output_tokens": field_out},
        "steel": {"calls": None, "input_tokens": 0, "output_tokens": 0, "note": "deterministic, ~0 tokens"},
    },
    "configs": {
        "tiered_green_bone_protocol": {
            "input_cost_usd": tiered_in_cost,
            "output_cost_usd": tiered_out_cost,
            "total_cost_usd": tiered_total,
        },
        "all_command_baseline": {
            "input_cost_usd": allcmd_in_cost,
            "output_cost_usd": allcmd_out_cost,
            "total_cost_usd": allcmd_total,
        },
    },
    "pct_cost_reduction_tiered_vs_all_command": pct_reduction,
    "prices_usd_per_token": PRICES,
    "calls": [
        {
            "name": c.name, "tier": c.tier,
            "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
            "input_basis": c.input_basis, "output_basis": c.output_basis,
            "note": c.note,
        }
        for c in calls
    ],
}

out_path = REPO / "output" / "token_estimate.json"
out_path.write_text(json.dumps(result, indent=2))
print(f"\n[written] {out_path}")

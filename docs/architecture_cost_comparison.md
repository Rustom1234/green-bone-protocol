# Architecture Cost Comparison: Tiered vs Single-Model

Date: 2026-07-05
Raw data: `output/architecture_comparison.json`

## Methodology

Three ways of doing the same small task were run and measured for real token usage and cost:

1. **Tiered (Green Bone emulation)**: one Opus call (head/coordination), three Haiku calls (field/bulk), one Sonnet call (command/synthesis). Five calls total, run sequentially, each step's output feeding the next.
2. **Opus-only**: the whole task in a single Opus call.
3. **Sonnet-only**: the whole task in a single Sonnet call.

Each call was run as:

```
env -u ANTHROPIC_API_KEY claude -p --model <opus|sonnet|haiku> --output-format json "<prompt>"
```

`ANTHROPIC_API_KEY` was unset for every call, so all calls billed to the Claude subscription, not a metered API key. `usage` (input/output/cache tokens) and `total_cost_usd` were parsed from each call's JSON output.

An independent cost estimate was also computed by applying current published per-model pricing to the measured token counts, using the formula:

```
cost = input_tokens * input_price
     + cache_creation_input_tokens * input_price * 2   (1h-TTL cache write, all calls used 1h TTL)
     + cache_read_input_tokens * input_price * 0.1
     + output_tokens * output_price
```

This formula reproduced each call's reported `total_cost_usd` **exactly** (to the cent, for all 7 calls), confirming both the pricing table and the token accounting below.

### Pricing used (per million tokens)

| Model | Input | Output |
|---|---|---|
| Claude Opus 4.8 (`claude-opus-4-8`) | $5.00 | $25.00 |
| Claude Sonnet 5 (`claude-sonnet-5`) | $3.00 (standard; $2.00 intro through 2026-08-31 not applied by the CLI in this run) | $15.00 ($10.00 intro) |
| Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | $1.00 | $5.00 |

## The task

> Write a short structured brief comparing SQLite vs PostgreSQL for a small side project (e.g. a personal to-do app expected to have a few thousand users at most). Use exactly these section headers: OVERVIEW, PROS (SQLite), CONS (SQLite), PROS (PostgreSQL), CONS (PostgreSQL), RECOMMENDATION. Keep the total output under 400 words.

For the tiered arm, this was split as:

- **Head (Opus)**: plan 3 research subtopics + the final section headers. No tools, ≤120 words.
- **Field (Haiku x3)**: one call per subtopic, each producing 5-8 raw bullet-point notes (no prose, no tools).
- **Command (Sonnet)**: synthesize the head's outline + all 3 field notes into the final structured brief, ≤400 words.

## Results

| Arm | Calls | Input tok | Cache write tok | Cache read tok | Output tok | Cost (USD) |
|---|---|---:|---:|---:|---:|---:|
| Tiered (Opus head + 3x Haiku field + Sonnet command) | 5 | 6,806 | 31,108 | 104,987 | 3,396 | **$0.2053** |
| Opus-only | 1 | 6,774 | 3,691 | 15,079 | 1,030 | **$0.1041** |
| Sonnet-only | 1 | 2 | 14,237 | 23,201 | 927 | **$0.1063** |

Per-call breakdown of the tiered arm:

| Step | Model | Cache write tok | Cache read tok | Output tok | Cost (USD) |
|---|---|---:|---:|---:|---:|
| Head (plan) | Opus | 1,979 | 16,798 | 149 | $0.0658 |
| Field 1 | Haiku | 8,870 | 17,209 | 763 | $0.0233 |
| Field 2 | Haiku | 5,038 | 21,203 | 868 | $0.0165 |
| Field 3 | Haiku | 4,879 | 21,203 | 706 | $0.0154 |
| Command (synthesis) | Sonnet | 10,342 | 28,574 | 910 | $0.0843 |

### Percentage differences

| Comparison | Difference |
|---|---|
| Tiered vs Opus-only | **+97.3%** (tiered cost ~2x Opus-only) |
| Tiered vs Sonnet-only | **+93.2%** (tiered cost ~2x Sonnet-only) |
| Opus-only vs Sonnet-only | -2.1% (Opus-only was slightly *cheaper* than Sonnet-only here) |

**On this task, the tiered arm was the most expensive of the three**, at roughly double either single-model arm.

## Why the tiered arm cost more here

The reason is not model choice, it's call count. Each `claude -p` invocation is a separate Claude Code CLI session, and each session pays its own fixed overhead for loading the Claude Code system prompt and tool schemas, visible in the `cache_creation_input_tokens`/`cache_read_input_tokens` fields (tens of thousands of tokens per call, far larger than the actual task prompt, which was a few dozen to a few hundred words). A single-call arm pays that overhead once. The 5-call tiered arm pays it five times. In this run, the tiered arm's total cache-related token volume (31,108 write + 104,987 read = 136,095 tokens) dwarfs its actual task input/output (6,806 + 3,396 = 10,202 tokens).

Put differently: Haiku's per-token price is 1/5th of Opus's, but a Haiku call still pays close to the same fixed per-session overhead as an Opus call. Splitting a small task into 5 cheap-model hops doesn't save money if each hop is a brand-new CLI session, because the fixed cost per hop swamps the per-token savings for a task this small.

This is very likely an artifact of the measurement method (required by the task spec) rather than a property of tiered agent architectures in general. A production Green Bone Protocol deployment would use the Anthropic Python SDK directly inside one long-lived process, with a shared, cached system prompt reused across agent hops (see `shared/prompt-caching.md` in the `claude-api` skill: cache reads cost ~0.1x, and a stable prefix carries forward across calls in the same process). Under that setup the fixed per-hop overhead measured here would mostly disappear, and the tiered arm's advantage, cheap Haiku tokens for the bulk research work, would likely show through as a net saving instead of a 2x cost increase. This experiment does not measure that scenario; it measures the CLI-session-per-hop case specifically, as instructed.

## Caveats

- **Single task, single run per arm.** No repeated sampling, so there is no variance estimate; a different task shape (e.g. one with much larger bulk/field volume relative to the coordination step) could change the outcome, especially in a non-CLI deployment where per-hop overhead is smaller.
- **Measured via subscription-billed CLI sessions**, not raw Messages API calls. See "Why the tiered arm cost more" above, this is the dominant factor in the result and should not be read as "tiering never saves money."
- **The tiered arm is an emulation** of the Opus-head/Sonnet-command/Haiku-field routing strategy using independent CLI subprocesses and manual prompt-passing, not the actual `greenbone/` framework code (`Weatherman`, `Horn`, `Clan` in `greenbone/clan.py`), which would add its own orchestration overhead on top of whatever model-call cost it incurs.
- **Sonnet 5 introductory pricing** ($2.00/$10.00 per MTok, through 2026-08-31) was not what the CLI actually billed in this run; the standard $3.00/$15.00 rate matched exactly, so that is what's reported.
- All reported numbers are real, measured token counts and costs, none of the figures in this document are fabricated or extrapolated beyond the pricing-formula cross-check described above.

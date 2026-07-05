# Session Result: Architecture Cost Comparison

Ran a real, measured, three-way cost comparison of doing the same small task (a short structured brief comparing SQLite vs PostgreSQL for a small side project) three ways:

1. **Tiered (Green Bone emulation)**: Opus head (plan) + 3x Haiku field (raw research) + Sonnet command (synthesis) = 5 calls.
2. **Opus-only**: whole task in one Opus call.
3. **Sonnet-only**: whole task in one Sonnet call.

All calls were run via `claude -p --model <opus|sonnet|haiku> --output-format json`, billed to the Claude subscription (`ANTHROPIC_API_KEY` was unset for every call, confirmed before and during the run). Full details, prompts, and caveats: `docs/architecture_cost_comparison.md`. Raw numbers: `output/architecture_comparison.json`.

## Headline result

| Arm | Cost (USD) |
|---|---:|
| Tiered (Opus + 3x Haiku + Sonnet) | $0.2053 |
| Opus-only | $0.1041 |
| Sonnet-only | $0.1063 |

On this task, **the tiered arm cost about 2x either single-model arm** (+97% vs Opus-only, +93% vs Sonnet-only). Opus-only and Sonnet-only were within 2% of each other.

## Why: it's an artifact of the measurement method, not of tiering itself

Each `claude -p` call is a separate Claude Code CLI session, and each session pays its own fixed overhead to load the Claude Code system prompt and tool schemas (tens of thousands of cache-related tokens per call, verified via the `usage` field in each call's JSON output). A single-call arm pays that overhead once; the 5-call tiered arm pays it five times. Haiku's per-token price is 1/5th of Opus's, but a Haiku CLI call still carries nearly the same fixed per-session overhead as an Opus one, so splitting the task into 5 cheap-model hops didn't save money here, the fixed cost per hop dominated the actual task tokens (roughly 136K overhead tokens vs ~10K real task tokens in the tiered arm).

A production Green Bone Protocol deployment (persistent Python SDK process, system prompt cached and reused across agent hops, no per-hop CLI relaunch) would very likely look different, the cheap-Haiku-for-bulk-work advantage would probably show up as a net saving instead of being swamped by repeated session overhead. This experiment measured the CLI-session-per-hop case specifically, as the task required; it is not a general verdict on tiered agent architectures.

An independent pricing-formula cost estimate (applying published per-model rates to the measured token counts) matched every call's reported cost to the cent, confirming the numbers above are consistent and not fabricated.

## Caveats

- Single task, single run per arm, no repeated sampling.
- Tiered arm is an emulation (independent CLI subprocesses + manual prompt-passing), not the actual `greenbone/` framework code.
- Sonnet 5's introductory pricing ($2/$10 per MTok through 2026-08-31) was not what the CLI actually billed; standard $3/$15 matched exactly.

## Status

Files added: `docs/architecture_cost_comparison.md`, `output/architecture_comparison.json`, this file. Secret scan run before push (see git log for confirmation it was clean). Pushed to `origin/master`.

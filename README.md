![Green Bone Protocol](assets/banner.svg)

Green Bone Protocol is a tiered multi-agent framework where a single Opus-grade coordinator sits atop a small number of high-judgment "Command" agents (Sonnet), which in turn direct a swarm of cheap, disposable "Field" agents (Haiku), to research a target and execute a commission. Terminology is drawn from Fonda Lee's Green Bone Saga. The pipeline runs in two phases: a **Weatherman** gathers intelligence into a sealed **Dossier**, then a **Horn** plans and executes a **Strike** against it.

## Tier architecture

![Architecture](assets/architecture.svg)

Every agent in the clan carries two independent attributes. Conflating them is the mistake this design deliberately avoids.

**Role** — the org position, straight from the novel: Pillar (the human operator), Weatherman (intel commander), Horn (execution commander), White Rat (scout), Intel Fist, Execution Fist, Finger.

**Discipline** — the jade ability an agent channels, which is this framework's actual capability tier and maps directly to model grade. The mapping is a single config, `MODEL_TIERS` in `greenbone/agents.py` — nothing else in the codebase hardcodes a model id:

| Discipline | Jade ability | Agents | Model grade (Tier) |
|---|---|---|---|
| **CHANNELING** | Amplifying and coordinating other agents | Horn | Opus (**Head** — top of the hierarchy) |
| **PERCEPTION** | Sensing / intelligence — seeing before others strike | Weatherman | Sonnet (**Command**) |
| **LIGHTNESS** | Speed — fast, cheap, disposable | White Rats, Intel Fists, Execution Fists, Fingers | Haiku (**Field**) |
| **STEEL** | Damage resistance — validation and hardening | Steel gates | Deterministic (near-zero-token), optional Haiku (**Field**) repair |

#### Why Channeling is a discipline, not a rename of the Horn

In the novel, Channeling is the ability to project jade energy *into another* Green Bone to amplify them — which is exactly what the Horn does architecturally, turning a set of cheap Fists into one coordinated strike. Role and discipline are two different vocabularies in the source material, and collapsing "Horn" into "Channeling" would blur the intel-commander/execution-commander split the system depends on. Keeping Horn as the role and Channeling as its discipline instead adds a genuine second classification axis — every agent now has an org position *and* a capability tier — rather than just reskinning an existing name. Channeling being the framework's single top-level coordinating ability is also why it sits at the Head tier, the only Discipline priced at Opus.

#### Configuring model tiers

`greenbone/agents.py` defines a `Tier` enum (`HEAD`/`COMMAND`/`FIELD`) and two dicts:

- `MODEL_TIERS: dict[Tier, str]` — the only place a model id is hardcoded. Defaults to Opus 4.8 / Sonnet 5 / Haiku 4.5.
- `DISCIPLINE_TIERS: dict[Discipline, Tier]` — which Discipline runs at which Tier (Channeling → Head, Perception → Command, Lightness/Steel → Field).

Every `Agent` subclass sets `model = model_for(Discipline.X)` rather than a literal model string, so retargeting the whole framework to different model versions (or swapping which Discipline gets the expensive tier) is a three-line edit in one file.

## The Steel tier

Steel is a defensive validation layer, not a label. It sits at the two seams where bad upstream output would otherwise propagate:

- **Dossier Gate** — after the Weatherman synthesises, before the Horn ever sees the result. Deterministic validation and sanitisation: required-field checks, control-character stripping, per-field length caps, and stripping of obvious prompt-injection lines (e.g. "ignore previous instructions"). If the Dossier is structurally broken, Steel can attempt one optional Haiku repair pass; otherwise it degrades safely rather than crashing the pipeline.
- **Strike Gate** — wraps every Execution Fist result inside the Horn's loop. Validates that output is non-empty, that anything that looks like code parses via `compile()`, and that anything that looks like JSON loads cleanly — retrying the fist up to 2 times before marking the result failed.

Steel's core is pure Python: no LLM calls, near-zero token cost, a defensive layer that is effectively free. See `greenbone/steel.py` and `tests/test_steel.py` (8 passing tests, no API calls).

## Approximate cost comparison (3-tier vs. all-Head)

**These are APPROXIMATE figures derived from tokenizing real logged prompt/response text with tiktoken (cl100k_base), NOT measurements of real Anthropic API `usage` fields.**

A recorded commission (a research run on "Anthropic") was tokenized offline. Token counts are identical between the two configurations below — same prompts, same responses — so the tiered split doesn't reduce tokens, it reduces *cost*, by routing the high-volume Field/Lightness calls to Haiku and the Command/Perception call to Sonnet instead of running everything at the top Head/Opus tier. Because the counts are identical across configs, any systematic undercount in tiktoken's approximation cancels out of the percentage ratio.

Pricing (current published Claude API rates; see `tools/estimate_tokens.py` PRICING NOTE for the Sonnet 5 introductory-rate caveat): Opus (`claude-opus-4-8`) $5 / $25 per 1M input/output tokens; Sonnet (`claude-sonnet-5`) $3 / $15 (standard rate); Haiku (`claude-haiku-4-5`) $1 / $5.

| Tier | Model | Calls | Input tokens (approx) | Output tokens (approx) |
|---|---|---:|---:|---:|
| Head (CHANNELING) | claude-opus-4-8 | 1 | 577 | 78 |
| Command (PERCEPTION) | claude-sonnet-5 | 1 | 2,139 | 398 |
| Field (LIGHTNESS) | claude-haiku-4-5 | 12 | 11,534 | 10,692 |
| Steel (STEEL) | deterministic, no LLM | — | ~0 | ~0 |

| Configuration | Input $ | Output $ | Total $ |
|---|---:|---:|---:|
| **Tiered (Green Bone Protocol, 3 tiers)** | $0.0208 | $0.0614 | **$0.0822** |
| **All-Head baseline (everything at Opus)** | $0.0713 | $0.2792 | **$0.3505** |

**Approximate cost reduction, tiered vs. all-Head: ~76.5%.**

The theoretical ceiling is 80% — the pure Haiku-vs-Opus price ratio ($1/$5 vs $5/$25 is a flat 1:5 on both input and output) if 100% of work were Field-tier — and the tiered split captures ~76.5% of it because the single Head-tier call (Horn's planning pass) stays at full Opus price in both configurations by construction, and the single Command-tier call (Weatherman's dossier synthesis) is only discounted to the Sonnet:Opus ratio (3:5 in, 15:25 out — a 40% reduction on that slice, not 80%). Field-tier dominates call volume (12 of 14 LLM calls, and the large majority of tokens), which is why the achieved reduction sits close to the ceiling. Some Field-tier token counts are heuristic (unlogged rat/intel-fist completions estimated at ~55% of `max_tokens`), while the Head/Command calls and the logged Finger outputs plus the Dossier are tokenized from real recorded text. The script is re-runnable and offline: see `tools/estimate_tokens.py` and `output/token_estimate.json`.

## Usage

### Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

### CLI (`run.py`)

```bash
# list available task configs
python run.py list

# Weatherman only — gather intel, return a sealed Dossier
python run.py intel --target "Anthropic" --goal "Tech stack analysis" \
  --urls https://anthropic.com --save-dossier

# full commission — Weatherman + Steel + Horn
python run.py commission --target "Stripe" --goal "Build a Python SDK" \
  --urls https://stripe.com --github stripe \
  --save-dossier --save-output
```

Additional flags: `--task <name>` to select a task config other than `default` (see `python run.py list`), `--token` for a GitHub personal access token.

### Python API

```python
from tasks.default.task import build_clan

clan = build_clan()
result = clan.commission(
    goal="Build a Python SDK",
    target="Stripe",
    context={"urls": ["https://stripe.com"], "github_org": "stripe"},
)

result["dossier"]  # sealed, Steel-validated Dossier
result["horn"]      # list of Steel-guarded Results from the Strike
```

## Project layout

```
greenbone/
├── agents.py     — Agent base class + MODEL_TIERS/DISCIPLINE_TIERS (Head/Command/Field model config)
├── clan.py       — WhiteRat, IntelFist, Finger, ExecutionFist, Weatherman, Horn, Clan
├── steel.py      — the STEEL discipline: Dossier Gate + Strike Gate
└── types.py      — Task, Result, Dossier, Rank, Discipline
tasks/            — pluggable commission configs, each exporting build_clan()
tools/
└── estimate_tokens.py   — offline tiktoken cost study, re-runnable
tests/            — unit tests (no API calls)
assets/           — banner.svg, architecture.svg
run.py            — CLI entry point (list / intel / commission)
server.py         — optional WhatsApp gateway built on the same Clan
```

## Concepts glossary

| Term | Meaning |
|---|---|
| **Commission** | A goal + target issued by the Pillar to the clan. |
| **Dossier** | The sealed intel package handed from Weatherman to Horn — the only channel between the two phases. |
| **Strike** | The execution phase: Horn's plan carried out by Execution Fists and Fingers. |
| **Pillar** | The human operator who issues the commission. |
| **Weatherman** | Intelligence commander (PERCEPTION, Sonnet — Command tier). Deploys Rats and Intel Fists, seals the Dossier. |
| **Horn** | Execution commander (CHANNELING, Opus — Head tier). Plans the Strike and assigns subtasks to Fists. |
| **Fist** | An Intel Fist cleans raw scout data; an Execution Fist carries out a class of Strike work. |
| **Finger** | A single atomic action, invoked from inside an Execution Fist. |
| **White Rat** | A disposable scout that returns raw, uninterpreted data. |

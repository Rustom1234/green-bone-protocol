# Green Bone Protocol — Multi-Agent Framework

> Inspired by the clan structure of Fonda Lee's Green Bone Saga.
> You are the Pillar. The system does the rest.

---

## Hierarchy

```
You (Pillar)
├── Weatherman  [sonnet-5, Command]   — intelligence, strategy, never executes
│     ├── White Rats  [haiku-4-5, Field]   — disposable scouts, raw data only
│     └── Intel Fists  [haiku-4-5, Field]  — clean, parse, structure rat data
│           ↓
│         Dossier  ← sealed intel package, handed to Horn
│
└── Horn  [opus-4-8, Head]   — reads dossier, plans strike, never researches
      └── Execution Fists  [haiku-4-5, Field]  — carry out the plan
            └── Fingers  [haiku-4-5, Field]   — single tool, single action
```

**The Pillar (you) commissions a goal + target. The system handles everything else.**

---

## The Two Phases

Every Green Bone Protocol operation has exactly two phases. They never overlap.

### Phase 1 — Intelligence (Weatherman)

The Weatherman's job is to know everything about the target before the Horn moves. He deploys White Rats to gather raw data, feeds it through Intel Fists to clean it, then seals a **Dossier** — a structured intel package Horn will use as his only source of truth.

The Weatherman never writes code. He never executes. He watches and prepares.

### Phase 2 — Execution (Horn)

The Horn reads the Dossier. He plans the strike, splits it into subtasks, and assigns each subtask to a Fist. Fists deploy Fingers for atomic actions.

The Horn never does his own research. If information is missing, that is a Weatherman failure. The Horn works only from the Dossier.

---

## Model Tiers

Three model grades, configured in one place — `MODEL_TIERS` in `greenbone/agents.py` — never hardcoded per agent:

| Tier | Model (default) | Used by | Why |
|---|---|---|---|
| **Head** | `claude-opus-4-8` | Horn (Channeling) | Top of the hierarchy — the one coordinator whose planning quality gates the whole Strike |
| **Command** | `claude-sonnet-5` | Weatherman (Perception) | High judgment, synthesis |
| **Field** | `claude-haiku-4-5` | Rats, Fists, Fingers (Lightness); Steel's rare repair pass | Fast, cheap, disposable, parallel |

Head and Command agents are expensive relative to Field. Minimise their token usage. Field agents are free to be verbose — they are throwaway workers. To repoint the framework at different model versions, edit `MODEL_TIERS` in `greenbone/agents.py` — every `Agent` subclass resolves its `model` via `model_for(Discipline.X)`, so nothing else needs to change.

---

## The Dossier Pattern (core rule)

```
Weatherman produces intel  →  Dossier  →  Horn consumes intel
                                 ↑
                       this is the only channel
```

Horn never calls a Rat. Weatherman never executes code. If you find yourself wanting to give Horn the ability to gather more data, that information should have been in the Dossier. Fix the Weatherman instead.

---

## Agent Roles

**Weatherman** — receives `goal`, `target`, and context from Pillar. Deploys rats, passes data through Intel Fists, synthesises the Dossier himself using Sonnet. Returns a sealed `Dossier` object.

**White Rats** — one job each, return raw data only, no interpretation. Stateless. Run in parallel. Expected to fail sometimes.

**Intel Fists** — take raw rat dumps, produce clean structured data. Run sequentially after rats complete.

**Dossier** — the single object passed from Weatherman to Horn. Horn never receives raw rat data.

**Horn** — reads Dossier, calls Sonnet to plan the strike, splits goal into subtasks, assigns each to a Fist by key. Collects results.

**Execution Fists** — each handles a class of subtask, orchestrates Fingers. Always receives Dossier in context.

**Fingers** — one tool, one return value, never call other agents.

---

## Adding New Capabilities

To add a new rat: subclass `WhiteRat`, set `name` and `system_prompt`, implement `run()`. Register in `Weatherman.__init__`.

To add a new execution fist: subclass `ExecutionFist`, implement `run()`. Register in `Horn.__init__` under a string key. Update Horn's system prompt to list the new key.

To add a new finger: subclass `Finger`, implement `run()`. Call it from inside a Fist.

---

## Prompting Rules

**Head/Command agents (Horn, Weatherman):** full context up front, role defined in system prompt, task in user message, structured output format required.

**Field agents (Rats, Fists, Fingers):** short direct system prompts, exact output format specified (JSON schema or plain text), no reasoning about whether to act — just act. If they fail, skip them.

**Dossier synthesis prompt** (Weatherman's final step): present all cleaned intel, ask for WEBSITE / GITHUB / NEWS / FINDINGS / APPROACH sections, force a format Horn can parse without an LLM, keep under 700 output tokens.

**Horn planning prompt**: include full Dossier context block, list available fist types by key, ask for JSON list of `{"fist": "...", "subtask": "..."}` assignments, keep subtasks specific.

---

## Error Handling Philosophy

- Rats fail silently — Weatherman skips them
- Intel Fists degrade gracefully — use previous fist output if one fails
- Dossier is always sealed — even with partial intel, hand off to Horn
- Horn never blocks — if a Fist fails, log and continue
- Never retry Head/Command agents — they are expensive, fix the prompt
- Retry Field agents freely — two retries before skipping is reasonable

---

## File Structure

```
greenbone/
├── CLAUDE.md      ← this file
├── types.py       ← Task, Result, Dossier, Rank
├── agents.py      ← Agent base class
├── clan.py        ← Weatherman, Horn, Clan, base classes
└── tasks/
    └── <n>/
        └── task.py   ← your rats, fists, fingers, commission call
```

---

## Commissioning a Task

```python
clan = build_clan()
result = clan.commission(
    goal="...",
    target="...",
    context={"urls": [...], "github_org": "..."}
)
# result["dossier"] — sealed intel
# result["horn"]    — list of Results from execution fists
```

---

## Glossary

| Term | Meaning |
|---|---|
| **Pillar** | You. The human giving the goal. |
| **Weatherman** | Intelligence commander. Spies and prepares. Never executes. |
| **Horn** | Execution commander. Strikes. Never researches. |
| **White Rat** | Cheap disposable scout. Raw data only. |
| **Intel Fist** | Cleans and structures raw rat data. |
| **Execution Fist** | Carries out a class of work assigned by Horn. |
| **Finger** | Single atomic action. |
| **Dossier** | Sealed intel package. The only channel between Weatherman and Horn. |
| **Commission** | A task issued by the Pillar to the full clan. |
| **Strike** | The execution phase. What Horn does after reading the Dossier. |

---


## Source Code

The canonical source lives on disk — read it directly rather than trusting a copy pasted into
this file, since it will drift:

- `greenbone/types.py`  — `Task`, `Result`, `Dossier`, `Rank`
- `greenbone/agents.py` — the `Agent` base class (`_llm`, `_llm_json`, `_ok`, `_fail`, `_log`)
- `greenbone/clan.py`   — `WhiteRat`, `IntelFist`, `Finger`, `ExecutionFist`, `Weatherman`, `Horn`, `Clan`
- `tasks/default/task.py` — a fully worked example task config (rats, fists, fingers, `build_clan()`)

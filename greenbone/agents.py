from __future__ import annotations
import json
import re
from enum import Enum
import anthropic
from greenbone.types import Task, Result, Rank, Discipline

client = anthropic.Anthropic()


class Tier(Enum):
    """The three model grades in the clan's hierarchy, top to bottom."""
    HEAD    = "head"     # Pillar / Channeling: the coordinator at the top of the hierarchy
    COMMAND = "command"  # Perception + other Sonnet-grade coordination roles
    FIELD   = "field"    # Lightness: cheap, high-volume, disposable workers


# ---------------------------------------------------------------------------
# THE single source of truth for "which model backs which tier". Everything
# else in this codebase (Weatherman, Horn, White Rats, Fists, Fingers, Steel)
# resolves its model through DISCIPLINE_TIERS + MODEL_TIERS via model_for()
# rather than hardcoding a model id: change the three values below to
# repoint the whole framework at different model versions (or providers).
# ---------------------------------------------------------------------------
MODEL_TIERS: dict[Tier, str] = {
    Tier.HEAD:    "claude-opus-4-8",
    Tier.COMMAND: "claude-sonnet-5",
    Tier.FIELD:   "claude-haiku-4-5",
}

# Which Discipline (the framework's actual capability tier, orthogonal to
# organizational Rank) runs at which model Tier.
DISCIPLINE_TIERS: dict[Discipline, Tier] = {
    Discipline.CHANNELING: Tier.HEAD,     # Horn, top of the hierarchy
    Discipline.PERCEPTION: Tier.COMMAND,  # Weatherman
    Discipline.LIGHTNESS:  Tier.FIELD,    # White Rats, Intel Fists, Execution Fists, Fingers
    Discipline.STEEL:      Tier.FIELD,    # deterministic; Haiku only for the rare guarded repair pass
}


def model_for(discipline: Discipline) -> str:
    """Resolve a Discipline to its configured model id via MODEL_TIERS."""
    return MODEL_TIERS[DISCIPLINE_TIERS[discipline]]


# Convenience aliases for the common case of "give me tier X's model id"
# without going through a Discipline. Prefer model_for(discipline) when a
# Discipline is available (e.g. inside the Agent subclasses below) so the
# role->tier mapping stays declared in one place (DISCIPLINE_TIERS).
HEAD    = MODEL_TIERS[Tier.HEAD]
COMMAND = MODEL_TIERS[Tier.COMMAND]
FIELD   = MODEL_TIERS[Tier.FIELD]


class Agent:
    name: str
    rank: Rank
    discipline: Discipline
    model: str
    system_prompt: str

    def run(self, task: Task) -> Result:
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")

    def _llm(self, user_msg: str, max_tokens: int = 1024) -> str:
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()

    def _llm_json(self, prompt: str, max_tokens: int = 1024) -> any:
        raw = self._llm(
            prompt + "\n\nReturn ONLY valid JSON. No explanation. No markdown fences.",
            max_tokens=max_tokens,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(raw)

    def _llm_json_safe(self, prompt: str, fallback: any = None,
                       max_tokens: int = 1024) -> any:
        try:
            return self._llm_json(prompt, max_tokens=max_tokens)
        except Exception:
            return fallback

    def _ok(self, output: any) -> Result:
        return Result(agent=self.name, rank=self.rank, output=output)

    def _fail(self, error: str) -> Result:
        return Result(agent=self.name, rank=self.rank, output=None,
                      success=False, error=error)

    def _log(self, msg: str):
        label = self.rank.value.upper().replace("_", " ")
        disc = getattr(self, "discipline", None)
        tag = f"{label}/{disc.value}" if disc else label
        print(f"    [{tag}:{self.name}] {msg}")

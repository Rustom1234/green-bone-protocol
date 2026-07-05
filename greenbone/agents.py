from __future__ import annotations
import json
import re
import anthropic
from greenbone.types import Task, Result, Rank, Discipline

client = anthropic.Anthropic()

COMMAND = "claude-sonnet-4-6"
FIELD   = "claude-haiku-4-5-20251001"


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

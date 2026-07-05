from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import json


class Rank(Enum):
    FINGER     = "finger"
    WHITE_RAT  = "white_rat"
    FIST       = "fist"
    HORN       = "horn"
    WEATHERMAN = "weatherman"
    STEEL      = "steel"


class Discipline(Enum):
    """The jade ability an agent channels — orthogonal to its Rank (org role)."""
    PERCEPTION = "perception"
    CHANNELING = "channeling"
    LIGHTNESS  = "lightness"
    STEEL      = "steel"


@dataclass
class Task:
    goal: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    agent: str
    rank: Rank
    output: Any
    success: bool = True
    error: str | None = None

    def __str__(self):
        status = "✓" if self.success else f"✗  {self.error}"
        return f"[{self.rank.value.upper().replace('_',' ')}:{self.agent}] {status}"


@dataclass
class Dossier:
    target: str
    website_summary: str
    github_summary: str
    news_summary: str
    tech_stack: list[str]
    key_findings: list[str]
    recommended_approach: str

    def to_context_block(self) -> str:
        findings_block = "\n".join(f"  • {f}" for f in self.key_findings)
        stack = ", ".join(self.tech_stack) if self.tech_stack else "not detected"
        return f"""
╔══ DOSSIER: {self.target} ══╗

TECH STACK:
  {stack}

WEBSITE / DOCS:
  {self.website_summary or "No data"}

GITHUB:
  {self.github_summary or "No data"}

NEWS / MENTIONS:
  {self.news_summary or "No data"}

KEY FINDINGS:
{findings_block or "  • No findings"}

RECOMMENDED APPROACH:
  {self.recommended_approach or "No recommendation"}

╚══ END DOSSIER ══╝""".strip()

    def to_whatsapp_summary(self) -> str:
        """Compact dossier for WhatsApp (fits in a message)."""
        stack = ", ".join(self.tech_stack[:6]) or "unknown"
        findings = "\n".join(f"• {f}" for f in self.key_findings[:5])
        approach = self.recommended_approach[:300]
        return (
            f"🗂 *DOSSIER: {self.target}*\n\n"
            f"*Stack:* {stack}\n\n"
            f"*Findings:*\n{findings}\n\n"
            f"*Approach:*\n{approach}"
        )

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "tech_stack": self.tech_stack,
            "key_findings": self.key_findings,
            "recommended_approach": self.recommended_approach,
            "website_summary": self.website_summary,
            "github_summary": self.github_summary,
            "news_summary": self.news_summary,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

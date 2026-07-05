"""
Steel — the STEEL discipline.

Damage resistance: absorb the damage of bad upstream output. Steel is a
defensive validation layer inserted at the two seams where bad output
propagates:

  Seam 1 — Dossier Gate:  guard_dossier()  (Weatherman -> Horn)
  Seam 2 — Strike Gate:   guard_fist()     (Horn -> each Execution Fist)

Steel's core is pure-Python / near-zero-token. The ONLY LLM call anywhere in
this module is the optional, guarded Dossier repair attempt inside
guard_dossier(), and even that is skipped whenever there is no raw material
to repair from (so it never fires against an empty/junk Dossier and never
fires in the unit tests in tests/test_steel.py).
"""

from __future__ import annotations
import json
import re

from greenbone.types import Task, Result, Rank, Dossier, Discipline
from greenbone.agents import Agent, FIELD

MAX_FIELD_LEN = 4000
MAX_LIST_ITEMS = 40
MAX_LIST_ITEM_LEN = 500

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(prior|previous)", re.IGNORECASE),
    re.compile(r"^\s*system\s*:", re.IGNORECASE),
]

# Keep \n and \t; strip other C0 control chars + DEL.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_CODE_FENCE_RE = re.compile(r"^```(?:python|py)?\s*|\s*```$", re.MULTILINE)

_CODE_MARKERS = (
    "def ", "class ", "import ", "```python", "```py",
    "function ", "const ", "let ", "\treturn ", "  return ",
)


class Steel(Agent):
    """The STEEL discipline: deterministic validation + sanitisation guard."""

    name = "steel"
    rank = Rank.STEEL
    discipline = Discipline.STEEL
    model = FIELD
    system_prompt = (
        "You are Steel — damage resistance. You repair a broken Dossier into a "
        "minimally safe, honest structure using ONLY the raw material given to "
        "you. Never invent facts. Return JSON: {\"recommended_approach\": \"...\"}"
    )

    def run(self, task: Task) -> Result:
        # Steel isn't invoked in the normal rat/fist pipeline flow; it's called
        # directly via guard_dossier()/guard_fist() from Clan/Horn. This exists
        # only to satisfy the Agent interface.
        raise NotImplementedError("Steel is invoked via guard_dossier/guard_fist, not run()")

    # ---------------------------------------------------------------
    # Seam 1 — Dossier Gate
    # ---------------------------------------------------------------

    def guard_dossier(self, dossier: Dossier | None) -> Dossier:
        """Validate + sanitise a Dossier before it reaches the Horn. Never raises."""
        try:
            return self._guard_dossier(dossier)
        except Exception as e:
            self._log(f"guard_dossier crashed unexpectedly ({e}); returning safe empty dossier")
            return self._empty_dossier(getattr(dossier, "target", None) or "unknown")

    def _guard_dossier(self, dossier: Dossier | None) -> Dossier:
        if dossier is None:
            self._log("dossier missing entirely — returning safe empty dossier")
            return self._empty_dossier("unknown")

        issues = []

        target = self._sanitise_text(dossier.target, max_len=200)
        if not target:
            issues.append("missing target")
            target = "unknown"

        tech_stack = self._sanitise_list(dossier.tech_stack)
        key_findings = self._sanitise_list(dossier.key_findings)
        if not tech_stack and not key_findings:
            issues.append("no tech_stack or key_findings")

        website_summary = self._sanitise_text(dossier.website_summary)
        github_summary = self._sanitise_text(dossier.github_summary)
        news_summary = self._sanitise_text(dossier.news_summary)
        recommended_approach = self._sanitise_text(dossier.recommended_approach)
        if not recommended_approach:
            issues.append("missing recommended_approach")

        safe = Dossier(
            target=target,
            website_summary=website_summary,
            github_summary=github_summary,
            news_summary=news_summary,
            tech_stack=tech_stack,
            key_findings=key_findings,
            recommended_approach=recommended_approach,
        )

        if not issues:
            return safe

        self._log(f"dossier gate found issues: {'; '.join(issues)}")

        has_raw_material = bool(
            website_summary or github_summary or news_summary
            or tech_stack or key_findings
        )
        # Only ever attempt the (LLM-backed) repair when there is something to
        # repair FROM, and only when the one thing actually missing is the
        # approach line. This keeps Steel's core near-zero-token: a totally
        # empty/junk dossier is degraded deterministically, no LLM call.
        if has_raw_material and not recommended_approach:
            repaired = self._attempt_repair(safe)
            if repaired is not None:
                return repaired

        if not safe.recommended_approach:
            safe.recommended_approach = (
                "No recommendation available — Weatherman output was incomplete."
            )
        return safe

    def _attempt_repair(self, dossier: Dossier) -> Dossier | None:
        try:
            raw = self._llm_json(
                "Repair this partial Dossier. Using ONLY the fields given below, "
                "write a short, honest recommended_approach. Do not invent facts "
                "not present in the data.\n\n"
                f"{json.dumps(dossier.to_dict(), indent=2)}\n\n"
                'Return ONLY JSON: {"recommended_approach": "..."}',
                max_tokens=200,
            )
            approach = self._sanitise_text(str(raw.get("recommended_approach", "")))
            if approach:
                dossier.recommended_approach = approach
                self._log("dossier repaired via one guarded Haiku pass")
            return dossier
        except Exception as e:
            self._log(f"repair attempt failed, degrading safely instead: {e}")
            return None

    def _empty_dossier(self, target: str) -> Dossier:
        return Dossier(
            target=target or "unknown",
            website_summary="",
            github_summary="",
            news_summary="",
            tech_stack=[],
            key_findings=[],
            recommended_approach="No recommendation available — no Dossier was produced.",
        )

    # ---------------------------------------------------------------
    # Seam 2 — Strike Gate
    # ---------------------------------------------------------------

    def guard_fist(self, fist, task: Task, max_retries: int = 2) -> Result:
        """Run `fist.run(task)`, validating + retrying up to max_retries. Never raises."""
        attempts = max_retries + 1
        last_result: Result | None = None

        for attempt in range(1, attempts + 1):
            try:
                result = fist.run(task)
            except Exception as e:
                self._log(f"fist [{self._fist_name(fist)}] raised on attempt {attempt}/{attempts}: {e}")
                last_result = Result(
                    agent=self._fist_name(fist),
                    rank=getattr(fist, "rank", Rank.FIST),
                    output=None, success=False, error=str(e),
                )
                continue

            ok, reason = self._validate_result(result)
            if ok:
                return self._sanitise_result(result)

            self._log(
                f"fist [{self._fist_name(fist)}] output invalid on attempt "
                f"{attempt}/{attempts}: {reason}"
            )
            last_result = result

        error_msg = f"STEEL: validation failed after {attempts} attempt(s)"
        if last_result is not None and last_result.error:
            error_msg += f" (last error: {last_result.error})"
        return Result(
            agent=self._fist_name(fist),
            rank=getattr(fist, "rank", Rank.FIST),
            output=last_result.output if last_result is not None else None,
            success=False,
            error=error_msg,
        )

    @staticmethod
    def _fist_name(fist) -> str:
        return getattr(fist, "name", fist.__class__.__name__ if fist is not None else "unknown")

    def _validate_result(self, result: Result | None) -> tuple[bool, str]:
        if result is None:
            return False, "no result returned"
        if not result.success:
            return False, result.error or "fist reported failure"

        output = result.output
        if output is None:
            return False, "empty output"
        if isinstance(output, str) and not output.strip():
            return False, "empty string output"
        if isinstance(output, (list, dict)) and len(output) == 0:
            return False, "empty collection output"

        if isinstance(output, dict) and output.get("code"):
            ok, err = self._validate_code(str(output["code"]))
            if not ok:
                return False, f"invalid code in output['code']: {err}"
        elif isinstance(output, str) and self._looks_like_code(output):
            ok, err = self._validate_code(output)
            if not ok:
                return False, f"invalid code: {err}"
        elif isinstance(output, str) and self._looks_like_json(output):
            ok, err = self._validate_json(output)
            if not ok:
                return False, f"invalid json: {err}"

        return True, ""

    def _sanitise_result(self, result: Result) -> Result:
        output = result.output
        if isinstance(output, str):
            output = self._sanitise_text(output)
        elif isinstance(output, dict):
            output = {
                k: (self._sanitise_text(v) if isinstance(v, str) else v)
                for k, v in output.items()
            }
        return Result(
            agent=result.agent, rank=result.rank, output=output,
            success=result.success, error=result.error,
        )

    # ---------------------------------------------------------------
    # Deterministic helpers
    # ---------------------------------------------------------------

    def _sanitise_text(self, text, max_len: int = MAX_FIELD_LEN) -> str:
        if text is None:
            return ""
        text = str(text)
        text = _CONTROL_CHARS_RE.sub("", text)
        clean_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and any(p.search(stripped) for p in _INJECTION_PATTERNS):
                continue
            clean_lines.append(line)
        text = "\n".join(clean_lines).strip()
        if len(text) > max_len:
            text = text[:max_len]
        return text

    def _sanitise_list(self, items) -> list:
        if not items:
            return []
        out = []
        for item in list(items)[:MAX_LIST_ITEMS]:
            cleaned = self._sanitise_text(item, max_len=MAX_LIST_ITEM_LEN)
            if cleaned:
                out.append(cleaned)
        return out

    def _looks_like_code(self, text: str) -> bool:
        if not text or len(text.strip()) < 3:
            return False
        return any(marker in text for marker in _CODE_MARKERS)

    def _looks_like_json(self, text: str) -> bool:
        s = text.strip()
        if not s:
            return False
        return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

    def _validate_code(self, src: str) -> tuple[bool, str]:
        src = _CODE_FENCE_RE.sub("", src.strip())
        try:
            compile(src, "<steel>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def _validate_json(self, text: str) -> tuple[bool, str]:
        try:
            json.loads(text)
            return True, ""
        except Exception as e:
            return False, str(e)

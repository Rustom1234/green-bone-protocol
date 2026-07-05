"""
Green Bone Protocol WhatsApp Gateway

Text naturally. Sonnet decides whether to handle it directly or route to the clans.

Direct tasks (code, files, reasoning) → Sonnet with tool use, acts on this machine.
Clan tasks (web scraping, SSH, parallel research) → planner + dispatcher.

Both paths send output and files back to WhatsApp.

Commands:
  stop — cancel current task
  help — show examples
"""

import os
import re
import sys
import json
import threading
import importlib
import traceback
import subprocess
import concurrent.futures
import anthropic

from pathlib import Path
from fastapi import FastAPI, Form, Response
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from greenbone.types import Task, Dossier

load_dotenv()

app = FastAPI()

TWILIO_SID    = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

twilio = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
claude = anthropic.Anthropic()

registered: set[str] = set()
stop_event = threading.Event()

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

WORKDIR = str(Path(__file__).parent)

# ── Claude Code session state ─────────────────────────────────
cc_mode:               set[str]                  = set()        # phones in CC mode
cc_sessions:           dict[str, list]           = {}           # phone → message history
cc_perm_event:         dict[str, threading.Event] = {}          # phone → waiting for perm answer
cc_perm_response:      dict[str, str]            = {}           # phone → "y" or "n"


# ── Task discovery ────────────────────────────────────────────

def _discover_task_configs() -> dict:
    configs = {}
    tasks_dir = Path(__file__).parent / "tasks"
    for d in tasks_dir.iterdir():
        if d.is_dir() and (d / "task.py").exists() and d.name != "__pycache__":
            configs[d.name] = f"tasks.{d.name}.task"
    return configs

TASK_CONFIGS = _discover_task_configs()


# ── Twilio helpers ────────────────────────────────────────────

def send(to: str, body: str):
    chunks = [body[i:i+1500] for i in range(0, len(body), 1500)]
    for chunk in chunks:
        twilio.messages.create(from_=TWILIO_NUMBER, to=to, body=chunk)

def broadcast(body: str):
    for number in registered:
        send(number, body)


# ── Model picker ──────────────────────────────────────────────

def pick_model(message: str) -> tuple[str, str]:
    if re.match(r"^sonnet[:\s]", message, re.IGNORECASE):
        clean = re.sub(r"^sonnet[:\s]+", "", message, flags=re.IGNORECASE).strip()
        return SONNET, clean
    return HAIKU, message


# ── Router ────────────────────────────────────────────────────

ROUTER_PROMPT = """You are the Green Bone Protocol Router. Given a user request, decide how to handle it.

ROUTES:
- "direct" — Sonnet acts on the machine directly using tools (bash, files). Best for:
  writing/editing code, running scripts, creating files/directories locally, debugging,
  building projects, anything that needs tight reasoning + iteration. Sonnet will have
  full tool access to this machine.

- "clan" — Parallel agent swarm. Best for: web scraping, competitive research, SSH to
  remote machines, tasks that benefit from parallelism or running in the background,
  things that don't need tight reasoning (data gathering, multi-source intel).

AVAILABLE CLAN TYPES (for clan route): shell, experiments, research, competitive, codebase, upgrade, default

When route is "direct": return {"route": "direct"}
When route is "clan": return {"route": "clan", "commissions": [
  {"id": "task_1", "clan": "<type>", "goal": "<specific goal>", "target": "<subject>",
   "context": {}, "depends_on": []}
]}

Return ONLY valid JSON. No explanation."""


def route(message: str) -> dict:
    resp = claude.messages.create(
        model=SONNET,
        max_tokens=1000,
        system=ROUTER_PROMPT,
        messages=[{"role": "user", "content": message}],
    )
    raw = re.sub(r"```(?:json)?|```", "", resp.content[0].text).strip()
    return json.loads(raw)


# ── Direct agent tools ────────────────────────────────────────

DIRECT_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command on the local machine. Returns stdout, stderr, and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file on the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current working dir)"},
            },
        },
    },
    {
        "name": "delegate_to_clan",
        "description": (
            "Delegate one or more tasks to the Green Bone Protocol clan agents and wait for results. "
            "Use this when a subtask is better handled in parallel, involves web scraping, "
            "multi-source research, SSH operations on remote hosts, or any work that doesn't "
            "need tight reasoning. Clan types: shell (SSH + remote commands), research (deep "
            "web/GitHub/HN intel), competitive (compare two things), codebase (review a repo), "
            "experiments (run code locally), upgrade (add capability to the clan), default (general)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commissions": {
                    "type": "array",
                    "description": "List of clan commissions to run.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id":         {"type": "string", "description": "Unique id, e.g. task_1"},
                            "clan":       {"type": "string", "description": "Clan type key"},
                            "goal":       {"type": "string", "description": "Specific actionable goal"},
                            "target":     {"type": "string", "description": "Main subject"},
                            "context":    {"type": "object", "description": "Extra context (host, urls, github_org, etc.)"},
                            "depends_on": {"type": "array",  "items": {"type": "string"}, "description": "IDs this task waits for"},
                        },
                        "required": ["id", "clan", "goal", "target"],
                    },
                },
            },
            "required": ["commissions"],
        },
    },
]

DIRECT_SYSTEM = f"""You are Claude Code — an expert engineer with full tool access to the user's machine at {WORKDIR}.
Tools available:
- bash / read_file / write_file / list_dir — act directly on this machine
- delegate_to_clan — hand off subtasks to the Green Bone Protocol agent swarm (parallel web scraping, SSH, research, etc.)

Use delegate_to_clan when a subtask is clearly better as distributed work (e.g. "scrape 5 sites", "SSH into a remote host", "research this company"). Do the rest yourself.
Work in {WORKDIR} unless told otherwise. Execute, don't just plan. Be concise — user is on WhatsApp.
When done, summarise in 2-3 sentences."""


def _run_tool(name: str, inputs: dict) -> str:
    try:
        if name == "bash":
            timeout = inputs.get("timeout", 60)
            proc = subprocess.run(
                inputs["command"], shell=True, capture_output=True,
                text=True, timeout=timeout, cwd=WORKDIR,
            )
            out = ""
            if proc.stdout.strip():
                out += proc.stdout.strip()
            if proc.stderr.strip():
                out += f"\nSTDERR: {proc.stderr.strip()}"
            out += f"\nExit: {proc.returncode}"
            return out[:4000]

        elif name == "read_file":
            p = Path(inputs["path"])
            if not p.is_absolute():
                p = Path(WORKDIR) / p
            return p.read_text()[:8000]

        elif name == "write_file":
            p = Path(inputs["path"])
            if not p.is_absolute():
                p = Path(WORKDIR) / p
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inputs["content"])
            return f"Written: {p}"

        elif name == "list_dir":
            p = Path(inputs.get("path", WORKDIR))
            if not p.is_absolute():
                p = Path(WORKDIR) / p
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            return "\n".join(
                f"{'📁' if e.is_dir() else '📄'} {e.name}" for e in entries
            )

        elif name == "delegate_to_clan":
            commissions = inputs.get("commissions", [])
            for c in commissions:
                c.setdefault("context", {})
                c.setdefault("depends_on", [])
            results = dispatch(commissions)
            summary = []
            for r in results:
                out = r.get("output")
                # Flatten fist result list to text
                if isinstance(out, list):
                    out = " | ".join(
                        str(item.output)[:300] if hasattr(item, "output") else str(item)[:300]
                        for item in out
                    )
                summary.append({
                    "id": r["id"],
                    "goal": r["goal"],
                    "success": r["success"],
                    "output": str(out)[:1500] if out else None,
                    "error": r.get("error"),
                })
            return json.dumps(summary, indent=2, default=str)

        return f"Unknown tool: {name}"

    except subprocess.TimeoutExpired:
        return f"Command timed out after {inputs.get('timeout', 60)}s"
    except Exception as e:
        return f"Error: {e}"


def run_direct(goal: str) -> tuple[str, list[str]]:
    """Run agentic loop. Returns (final_text, files_written)."""
    messages = [{"role": "user", "content": goal}]
    files_written = []
    final_text = ""
    step = 0

    while not stop_event.is_set():
        step += 1
        resp = claude.messages.create(
            model=SONNET,
            max_tokens=4096,
            system=DIRECT_SYSTEM,
            tools=DIRECT_TOOLS,
            messages=messages,
        )

        text = " ".join(b.text for b in resp.content if hasattr(b, "text") and b.text).strip()

        if resp.stop_reason == "end_turn":
            final_text = text
            break

        if resp.stop_reason != "tool_use":
            final_text = text
            break

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            result = _run_tool(block.name, block.input)
            if block.name == "write_file":
                files_written.append(block.input.get("path", ""))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": tool_results})

        if step > 30:
            final_text = "⚠️ Step limit reached."
            break

    return final_text, files_written




# ── Claude Code session ──────────────────────────────────────

CC_SYSTEM = DIRECT_SYSTEM  # same system prompt, same tools

RISKY_PATTERNS = [
    r"\brm\s+-[rf]",
    r"\bdrop\s+(table|database)",
    r"\btruncate\b",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bformat\b",
    r"> /dev/",
    r"\bkill\s+-9\b",
    r"\bchmod\s+777\b",
]

def _is_risky(name: str, inputs: dict) -> str | None:
    """Return a description if the tool call looks risky, else None."""
    if name == "bash":
        cmd = inputs.get("command", "")
        for pat in RISKY_PATTERNS:
            if re.search(pat, cmd, re.IGNORECASE):
                return f"`{cmd}`"
    if name == "write_file":
        p = Path(inputs.get("path", ""))
        if not p.is_absolute():
            p = Path(WORKDIR) / p
        if p.exists():
            return f"overwrite existing file `{p}`"
    return None


def _ask_permission(phone: str, description: str) -> bool:
    """Send a permission prompt to the user and block until they reply y/n (60s timeout)."""
    event = threading.Event()
    cc_perm_event[phone] = event
    send(phone, f"⚠️ Allow: {description}?\nReply *y* or *n*")
    granted = event.wait(timeout=60)
    cc_perm_event.pop(phone, None)
    if not granted:
        send(phone, "⏱ No reply — skipping.")
        return False
    answer = cc_perm_response.pop(phone, "n").strip().lower()
    return answer in ("y", "yes")


def run_cc_turn(message: str, phone: str):
    """Run one conversational turn of the CC session for this phone."""
    stop_event.clear()
    history = cc_sessions.setdefault(phone, [])
    history.append({"role": "user", "content": message})

    step = 0
    files_written: list[str] = []
    log_lines: list[str] = []  # buffer everything, send at end

    while not stop_event.is_set():
        step += 1
        resp = claude.messages.create(
            model=SONNET,
            max_tokens=4096,
            system=CC_SYSTEM,
            tools=DIRECT_TOOLS,
            messages=history,
        )

        history.append({"role": "assistant", "content": resp.content})
        text = "\n".join(b.text for b in resp.content if hasattr(b, "text") and b.text).strip()

        if resp.stop_reason == "end_turn":
            if text:
                log_lines.append(text)
            break

        if resp.stop_reason != "tool_use":
            if text:
                log_lines.append(text)
            break

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            risky = _is_risky(block.name, block.input)
            if risky:
                # Permission prompts must still go out immediately
                allowed = _ask_permission(phone, risky)
                if not allowed:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "User denied permission — skipped.",
                    })
                    log_lines.append(f"⛔ {block.name} denied")
                    continue

            result = _run_tool(block.name, block.input)

            if block.name == "write_file":
                files_written.append(block.input.get("path", ""))
                log_lines.append(f"📄 wrote {block.input.get('path','')}")
            elif block.name == "bash":
                first_line = result.split("\n")[0][:100]
                log_lines.append(f"$ {block.input.get('command','')[:80]}\n  {first_line}")
            elif block.name == "delegate_to_clan":
                log_lines.append(f"🐾 delegated {len(block.input.get('commissions',[]))} clan task(s)")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        history.append({"role": "user", "content": tool_results})

        if step > 40:
            log_lines.append("⚠️ Step limit reached.")
            break

    cc_sessions[phone] = history[-50:]

    # Build and send one final message
    output = "\n".join(log_lines).strip()
    if output:
        send(phone, output[:1400])

    _send_output_files(files_written, phone)


def _send_output_files(files: list[str], phone: str | None = None):
    """Send written files back. If phone given, send only to that number; else broadcast."""
    _send = (lambda body: send(phone, body)) if phone else broadcast
    for path in files:
        try:
            p = Path(path) if Path(path).is_absolute() else Path(WORKDIR) / path
            if not p.exists() or p.stat().st_size > 20_000:
                continue
            content = p.read_text()
            ext = p.suffix
            lang = {".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".sh": "bash", ".md": "", ".txt": ""}.get(ext, "")
            fence = f"```{lang}" if lang else "```"
            _send(f"📄 *{p.name}*\n{fence}\n{content[:1400]}\n```")
        except Exception:
            pass


# ── Clan path ─────────────────────────────────────────────────

def get_clan(config_name: str):
    configs = _discover_task_configs()
    mod_path = configs.get(config_name) or configs.get("default", "tasks.default.task")
    if mod_path in sys.modules:
        mod = importlib.reload(sys.modules[mod_path])
    else:
        mod = importlib.import_module(mod_path)
    return mod.build_clan()


def run_commission(commission: dict, prior_results: dict) -> dict:
    if stop_event.is_set():
        return {"id": commission["id"], "goal": commission["goal"],
                "success": False, "error": "stopped", "output": None}
    cid       = commission["id"]
    clan_name = commission["clan"]
    goal      = commission["goal"]
    target    = commission["target"]
    ctx       = dict(commission.get("context") or {})
    ctx["target"] = target
    if prior_results:
        ctx["prior_results"] = prior_results

    try:
        clan = get_clan(clan_name)

        wx_result = clan.weatherman.run(Task(goal=goal, context=ctx))
        if not wx_result.success or not isinstance(wx_result.output, Dossier):
            return {"id": cid, "goal": goal, "success": False,
                    "error": "Weatherman failed", "output": None}

        dossier: Dossier = wx_result.output
        horn_result = clan.horn.run(Task(goal=goal, context={**ctx, "dossier": dossier}))
        if not horn_result.success:
            return {"id": cid, "goal": goal, "success": False,
                    "error": horn_result.error, "output": None}

        return {"id": cid, "goal": goal, "success": True,
                "output": horn_result.output, "dossier": dossier.to_dict()}
    except Exception as e:
        return {"id": cid, "goal": goal, "success": False,
                "error": f"{type(e).__name__}: {e}", "output": None}


def dispatch(plan_list: list[dict]) -> list[dict]:
    completed: dict[str, dict] = {}
    pending = list(plan_list)
    all_results = []

    while pending:
        if stop_event.is_set():
            break
        ready = [t for t in pending
                 if all(dep in completed for dep in t.get("depends_on", []))]
        if not ready:
            break

        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(
                    run_commission, t,
                    {dep: completed[dep] for dep in t.get("depends_on", []) if dep in completed}
                ): t["id"]
                for t in ready
            }
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"id": task_id, "goal": "?", "success": False,
                              "error": str(e), "output": None}
                completed[task_id] = result
                all_results.append(result)

        done_ids = {t["id"] for t in ready}
        pending = [t for t in pending if t["id"] not in done_ids]

    return all_results


CONSOLIDATOR_PROMPT = """You are the Green Bone Protocol Consolidator. Given completed task results, write a concise WhatsApp summary.
Plain text. WhatsApp bold: *word*, italic: _word_.
Lead with the overall outcome. 1-2 lines per task. Under 900 chars total. Use • not -.
Don't repeat the user's message. End with a next-step line only if obvious."""

def consolidate(results: list[dict]) -> str:
    resp = claude.messages.create(
        model=SONNET,
        max_tokens=600,
        system=CONSOLIDATOR_PROMPT,
        messages=[{"role": "user", "content": json.dumps(results, indent=2, default=str)}],
    )
    return resp.content[0].text.strip()


# ── Main task handler ─────────────────────────────────────────

def run_task(message: str, from_number: str):
    stop_event.clear()
    try:
        _, clean = pick_model(message)

        try:
            decision = route(clean)
        except Exception as e:
            broadcast(f"❌ Router failed: {e}")
            return

        if decision.get("route") == "direct":
            final_text, files = run_direct(clean)
            msg = f"✅ {final_text}" if final_text else "✅ Done."
            broadcast(msg[:1400])
            if files:
                _send_output_files(files, phone=None)

        elif decision.get("route") == "clan":
            commissions = decision.get("commissions", [])
            if not commissions:
                broadcast("❌ No commissions in plan.")
                return
            results = dispatch(commissions)
            if stop_event.is_set():
                broadcast("🛑 Stopped.")
                return
            summary_input = [
                {"goal": r["goal"], "success": r["success"],
                 "result": r.get("output") or r.get("error")}
                for r in results
            ]
            summary = consolidate(summary_input)
            broadcast(f"✅ {summary}")

        else:
            broadcast(f"❌ Unknown route: {decision}")

    except Exception as e:
        broadcast(f"💥 {type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}")


# ── Webhook ───────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(Body: str = Form(...), From: str = Form(...)):
    registered.add(From)
    msg = Body.strip()
    msg_lower = msg.lower()
    twiml = MessagingResponse()

    # ── CC mode: permission responses ────────────────────────
    if From in cc_perm_event and msg_lower in ("y", "yes", "n", "no"):
        cc_perm_response[From] = msg_lower
        cc_perm_event[From].set()
        twiml.message("👍" if msg_lower in ("y", "yes") else "👎")
        return Response(content=str(twiml), media_type="application/xml")

    # ── Global commands ───────────────────────────────────────
    if msg_lower == "stop":
        stop_event.set()
        twiml.message("🛑 Stopping...")
        return Response(content=str(twiml), media_type="application/xml")

    if msg_lower in ("cc", "claude code", "code mode"):
        stop_event.clear()
        cc_mode.add(From)
        cc_sessions.pop(From, None)
        twiml.message(
            "💻 *Claude Code mode*\n\n"
            "Acting directly on your machine. Full tool access.\n"
            "I'll ask *y/n* before anything risky.\n"
            "Text *exit* to leave."
        )
        return Response(content=str(twiml), media_type="application/xml")

    if msg_lower == "exit" and From in cc_mode:
        cc_mode.discard(From)
        cc_sessions.pop(From, None)
        twiml.message("🐾 Back to Green Bone mode.")
        return Response(content=str(twiml), media_type="application/xml")

    if msg_lower == "help":
        if From in cc_mode:
            twiml.message(
                "💻 *Claude Code mode*\n\n"
                "Describe what you want — I'll do it on the machine.\n"
                "I'll ask *y/n* before risky ops.\n\n"
                "• _write a tic-tac-toe game_\n"
                "• _refactor server.py_\n"
                "• _run the tests_\n\n"
                "Text *exit* to return to Green Bone."
            )
        else:
            twiml.message(
                "🐾 *Green Bone Protocol*\n\n"
                "Just text naturally.\n\n"
                "• _research anthropic's stack_\n"
                "• _create a dir on goldbery-via-dean_\n"
                "• _build an ML project for appliance age detection_\n\n"
                "Text *cc* for Claude Code mode. Text *stop* to cancel."
            )
        return Response(content=str(twiml), media_type="application/xml")

    # ── Route to CC or Green Bone ─────────────────────────────
    if From in cc_mode:
        twiml.message("💻 On it...")
        threading.Thread(target=run_cc_turn, args=(msg, From), daemon=True).start()
    else:
        twiml.message("🐾 On it...")
        threading.Thread(target=run_task, args=(msg, From), daemon=True).start()

    return Response(content=str(twiml), media_type="application/xml")


@app.get("/health")
async def health():
    return {"status": "ok", "registered": len(registered),
            "configs": list(_discover_task_configs().keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["greenbone", "tasks"],
        reload_excludes=["experiments/*", "output/*"],
    )

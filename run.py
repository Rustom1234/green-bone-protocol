#!/usr/bin/env python3
"""
Green Bone Protocol CLI: commission a task from the terminal.

Usage:
  python run.py commission --target "Stripe" --goal "Build a Python SDK" --urls https://stripe.com --github stripe
  python run.py intel      --target "Anthropic" --goal "Tech stack analysis"
  python run.py list                                  # show available task configs
  python run.py commission --task competitive ...     # use a specific task config

Task configs live in tasks/<name>/task.py and export build_clan().
Default: tasks/default
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_task_module(name: str):
    mod_path = f"tasks.{name}.task"
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError:
        print(f"✗ No task config found at tasks/{name}/task.py")
        sys.exit(1)


def cmd_list(_args):
    tasks_dir = Path("tasks")
    configs = [d.name for d in tasks_dir.iterdir() if (d / "task.py").exists()]
    print("\nAvailable task configs:")
    for c in sorted(configs):
        print(f"  • {c}")
    print()


def cmd_commission(args):
    mod = get_task_module(args.task)
    clan = mod.build_clan()

    context = {}
    if args.urls:
        context["urls"] = args.urls
    if args.github:
        context["github_org"] = args.github
    if args.token:
        context["github_token"] = args.token

    result = clan.commission(goal=args.goal, target=args.target, context=context)

    dossier = result.get("dossier")
    if dossier and args.save_dossier:
        os.makedirs("output", exist_ok=True)
        path = f"output/{args.target.lower().replace(' ', '_')}_dossier.json"
        dossier.save(path)
        print(f"\n  Dossier saved → {path}")

    horn_results = result.get("horn", [])
    if horn_results and args.save_output:
        os.makedirs("output", exist_ok=True)
        path = f"output/{args.target.lower().replace(' ', '_')}_output.json"
        with open(path, "w") as f:
            serialisable = []
            for r in horn_results:
                serialisable.append({
                    "agent": r.agent,
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                })
            json.dump(serialisable, f, indent=2)
        print(f"  Output saved  → {path}")


def cmd_intel(args):
    mod = get_task_module(args.task)
    clan = mod.build_clan()

    context = {}
    if args.urls:
        context["urls"] = args.urls
    if args.github:
        context["github_org"] = args.github
    if args.token:
        context["github_token"] = args.token

    from greenbone.types import Task
    wx = clan.weatherman.run(Task(goal=args.goal, context={**context, "target": args.target}))

    if not wx.success:
        print(f"✗ Weatherman failed: {wx.error}")
        sys.exit(1)

    dossier = wx.output
    print("\n" + dossier.to_context_block())

    if args.save_dossier:
        os.makedirs("output", exist_ok=True)
        path = f"output/{args.target.lower().replace(' ', '_')}_dossier.json"
        dossier.save(path)
        print(f"\n  Dossier saved → {path}")


def main():
    parser = argparse.ArgumentParser(prog="greenbone", description="Green Bone Protocol CLI")
    sub = parser.add_subparsers(dest="cmd")

    # list
    sub.add_parser("list", help="List available task configs")

    # shared args for commission + intel
    def add_common(p):
        p.add_argument("--target", required=True, help="Target name, e.g. 'Stripe'")
        p.add_argument("--goal",   required=True, help="What to accomplish")
        p.add_argument("--task",   default="default", help="Task config (default: default)")
        p.add_argument("--urls",   nargs="*", help="URLs for WebRat")
        p.add_argument("--github", help="GitHub org/user for GitHubRat")
        p.add_argument("--token",  help="GitHub personal access token (optional)")
        p.add_argument("--save-dossier", action="store_true", help="Save dossier to output/")

    p_commission = sub.add_parser("commission", help="Full run: Weatherman + Horn")
    add_common(p_commission)
    p_commission.add_argument("--save-output", action="store_true", help="Save Horn output to output/")

    p_intel = sub.add_parser("intel", help="Weatherman only, return sealed dossier")
    add_common(p_intel)

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "commission":
        cmd_commission(args)
    elif args.cmd == "intel":
        cmd_intel(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import os
from src.blue.blue_agent_graph import build_blue_graph
from src.core.run_manager import new_run_id, prepare_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", type=int, required=True)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--clean-run", action="store_true")
    ap.add_argument("--logs-dir", type=str, default="data/logs_backend_a")
    ap.add_argument("--delay", type=int, default=30)
    ap.add_argument("--non-interactive", action="store_true")
    args = ap.parse_args()

    run_id = args.run_id or new_run_id("blue")
    paths = prepare_run(run_id, clean=args.clean_run, meta={"component": "blue_agent"})
    memory_dir = os.path.join(paths["base"], "memory")

    app = build_blue_graph()
    state = {
        "episode_id": args.episode_id,
        "logs_dir": args.logs_dir,
        "response_delay_sec": args.delay,
        "interactive": not args.non_interactive,

        "run_id": run_id,
        "decisions_path": paths["decisions"],
        "actions_path": paths["actions"],
        "memory_dir": memory_dir,
    }

    out = app.invoke(state)
    print(f"✅ Blue Agent finished. run_id={run_id}. Final state keys: {list(out.keys())}")

if __name__ == "__main__":
    main()

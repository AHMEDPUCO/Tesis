from __future__ import annotations
import argparse

from src.blue.blue_agent_graph import build_blue_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", type=int, required=True)
    ap.add_argument("--logs-dir", type=str, default="data/logs_backend_a")
    ap.add_argument("--delay", type=int, default=30)
    ap.add_argument("--non-interactive", action="store_true")
    args = ap.parse_args()

    app = build_blue_graph()
    state = {
        "episode_id": args.episode_id,
        "logs_dir": args.logs_dir,
        "response_delay_sec": args.delay,
        "interactive": not args.non_interactive,
    }

    out = app.invoke(state)
    print("✅ Blue Agent finished. Final state keys:", list(out.keys()))


if __name__ == "__main__":
    main()

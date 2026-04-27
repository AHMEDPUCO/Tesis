from __future__ import annotations
import argparse
import os
from src.blue.blue_agent_graph import build_blue_graph, run_blue_episode
from src.core.run_manager import new_run_id, prepare_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", type=int, default=None)
    ap.add_argument("--episode-start", type=int, default=None)
    ap.add_argument("--episode-end", type=int, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--clean-run", action="store_true")
    ap.add_argument("--logs-dir", type=str, default="data/logs_backend_a")
    ap.add_argument("--backend", type=str, default="backend_a", choices=["backend_a", "backend_b"])
    ap.add_argument("--gt-dir", type=str, default="data/ground_truth")
    ap.add_argument("--delay", type=int, default=30)
    ap.add_argument("--schema-mapper-mode", type=str, default="static", choices=["static", "dynamic"])
    ap.add_argument("--schema-map-min-confidence", type=float, default=0.75)
    ap.add_argument("--schema-map-cache-path", type=str, default=None)
    ap.add_argument("--schema-cache-scope", type=str, default="run", choices=["run", "persistent"])
    ap.add_argument("--schema-adapt-mode", type=str, default="contract_first", choices=["contract_first", "llm_first"])
    ap.add_argument("--backend-b-alias-mode", type=str, default="full", choices=["full", "minimal"])
    ap.add_argument("--mcp-enabled", dest="mcp_enabled", action="store_true")
    ap.add_argument("--no-mcp", dest="mcp_enabled", action="store_false")
    ap.add_argument("--mcp-tool", type=str, default="search_logs")
    ap.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "ollama"])
    ap.add_argument("--gemini-api-key", type=str, default=None)
    ap.add_argument("--gemini-model", type=str, default="gemini-1.5-flash")
    ap.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", type=str, default="qwen3:8b")
    ap.add_argument("--llm-timeout-sec", type=float, default=8.0)
    ap.add_argument("--non-interactive", action="store_true")
    ap.set_defaults(mcp_enabled=True)
    args = ap.parse_args()

    if args.episode_id is not None:
        episode_ids = [int(args.episode_id)]
    else:
        if args.episode_start is None or args.episode_end is None:
            ap.error("Usa --episode-id o --episode-start/--episode-end.")
        if int(args.episode_start) > int(args.episode_end):
            ap.error("--episode-start no puede ser mayor que --episode-end.")
        episode_ids = list(range(int(args.episode_start), int(args.episode_end) + 1))

    run_id = args.run_id or new_run_id("blue")
    paths = prepare_run(run_id, clean=args.clean_run, meta={"component": "blue_agent"})
    memory_dir = os.path.join(paths["base"], "memory")

    base_state = {
        "logs_dir": args.logs_dir,
        "logs_backend": args.backend,
        "gt_dir": args.gt_dir,
        "response_delay_sec": args.delay,
        "interactive": not args.non_interactive,
        "schema_mapper_mode": args.schema_mapper_mode,
        "schema_map_min_confidence": args.schema_map_min_confidence,
        "schema_map_cache_path": args.schema_map_cache_path,
        "schema_cache_scope": args.schema_cache_scope,
        "schema_adapt_mode": args.schema_adapt_mode,
        "backend_b_alias_mode": args.backend_b_alias_mode,
        "mcp_enabled": args.mcp_enabled,
        "mcp_tool": args.mcp_tool,
        "llm_provider": args.llm_provider,
        "gemini_api_key": args.gemini_api_key,
        "gemini_model": args.gemini_model,
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "llm_timeout_sec": args.llm_timeout_sec,
        "run_id": run_id,
        "decisions_path": paths["decisions"],
        "actions_path": paths["actions"],
        "memory_dir": memory_dir,
    }

    try:
        app = build_blue_graph()
        for episode_id in episode_ids:
            out = app.invoke({"episode_id": episode_id, **base_state})
            print(f"Blue Agent finished. run_id={run_id} episode={episode_id}. Final state keys: {list(out.keys())}")
    except RuntimeError:
        for episode_id in episode_ids:
            out = run_blue_episode({"episode_id": episode_id, **base_state})
            print(f"Blue Agent finished. run_id={run_id} episode={episode_id}. Final state keys: {list(out.keys())}")

if __name__ == "__main__":
    main()

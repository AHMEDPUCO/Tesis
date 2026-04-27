from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from urllib import error as url_error
from urllib import request
from dataclasses import dataclass, asdict
from typing import Dict, List

from src.core.run_manager import prepare_run, run_paths
from src.judge.judge_mtd_mttr import judge_mttd_mttr


@dataclass
class RepRecord:
    repetition: int
    seed: int
    memory_seed_dir_resolved: str
    dataset_dir: str
    logs_dir: str
    baseline_logs_dir: str
    blue_logs_dir: str
    blue_backend: str
    blue_schema_mapper: str
    schema_cache_scope: str
    schema_adapt_mode: str
    backend_b_alias_mode: str
    mcp_enabled: bool
    mcp_tool: str
    llm_provider: str
    ollama_model: str
    backend_b_drift_profile: str
    gt_dir: str
    baseline_dir: str
    baseline_confusion_containment: str
    baseline_confusion_detection: str
    baseline_phase2: str
    blue_run_id: str
    blue_dir: str
    blue_confusion_containment: str
    blue_confusion_detection: str
    blue_phase2: str
    blue_swap_enabled: bool
    blue_swap_episode: int
    blue_phase1_backend: str
    blue_phase2_backend: str


def _run(cmd: List[str], *, cwd: str) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _copy_memory_seed(seed_dir: str, dst_dir: str) -> None:
    if not seed_dir or not os.path.exists(seed_dir):
        return
    _ensure_dir(dst_dir)
    for name in ("cases.jsonl", "index.faiss"):
        src = os.path.join(seed_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


def _resolve_memory_seed_dir(seed_dir: str, rep_name: str) -> str:
    if not seed_dir or not os.path.exists(seed_dir):
        return seed_dir
    rep_dir = os.path.join(seed_dir, rep_name)
    if os.path.isdir(rep_dir):
        return rep_dir
    return seed_dir


def _prewarm_ollama(*, url: str, model: str) -> bool:
    endpoint = f"{url.rstrip('/')}/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": "Respond with exactly: ok",
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    req = request.Request(
        url=endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        return bool(str(data.get("response", "")).strip())
    except (url_error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return False


def _run_confusion(
    *,
    python_exe: str,
    cwd: str,
    gt_dir: str,
    decisions_path: str,
    out_csv: str,
    positive_decisions: str,
) -> None:
    _run(
        [
            python_exe,
            "-m",
            "src.judge.judge_confusion",
            "--gt-dir",
            gt_dir,
            "--decisions",
            decisions_path,
            "--out-csv",
            out_csv,
            "--positive-decisions",
            positive_decisions,
        ],
        cwd=cwd,
    )


def _blue_logs_dir_for_backend(*, dataset_dir: str, backend: str) -> str:
    return os.path.join(dataset_dir, "logs_backend_b" if backend == "backend_b" else "logs_backend_a")


def _build_blue_cmd(
    *,
    python_exe: str,
    run_id: str,
    logs_dir: str,
    backend: str,
    gt_dir: str,
    episode_start: int,
    episode_end: int,
    schema_mapper_mode: str,
    schema_map_min_confidence: float,
    schema_cache_scope: str,
    schema_adapt_mode: str,
    backend_b_alias_mode: str,
    mcp_tool: str,
    llm_provider: str,
    gemini_model: str,
    ollama_url: str,
    ollama_model: str,
    llm_timeout_sec: float,
    delay: int,
    mcp_enabled: bool,
) -> List[str]:
    cmd = [
        python_exe,
        "-m",
        "src.blue.run_blue_agent",
        "--episode-start",
        str(episode_start),
        "--episode-end",
        str(episode_end),
        "--run-id",
        run_id,
        "--logs-dir",
        logs_dir,
        "--backend",
        backend,
        "--gt-dir",
        gt_dir,
        "--schema-mapper-mode",
        schema_mapper_mode,
        "--schema-map-min-confidence",
        str(schema_map_min_confidence),
        "--schema-cache-scope",
        schema_cache_scope,
        "--schema-adapt-mode",
        schema_adapt_mode,
        "--backend-b-alias-mode",
        backend_b_alias_mode,
        "--mcp-tool",
        mcp_tool,
        "--llm-provider",
        llm_provider,
        "--gemini-model",
        gemini_model,
        "--ollama-url",
        ollama_url,
        "--ollama-model",
        ollama_model,
        "--llm-timeout-sec",
        str(llm_timeout_sec),
        "--delay",
        str(delay),
        "--non-interactive",
    ]
    if not bool(mcp_enabled):
        cmd.append("--no-mcp")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=1337)
    ap.add_argument("--seed-step", type=int, default=100)
    ap.add_argument("--noise-per-episode", type=int, default=2000)
    ap.add_argument("--benign-rate", type=float, default=0.35)
    ap.add_argument("--recurrent-benign-rate", type=float, default=0.0)
    ap.add_argument("--recurrent-benign-profiles", type=int, default=3)
    ap.add_argument("--backend-b-drift-profile", type=str, default="classic", choices=["classic", "hard4"])
    ap.add_argument("--delay", type=int, default=30)
    ap.add_argument("--memory-seed-dir", default="data/memory")
    ap.add_argument("--out-dir", default="data/experiments")
    ap.add_argument("--blue-backend", type=str, default="backend_a", choices=["backend_a", "backend_b"])
    ap.add_argument("--swap-episode", type=int, default=0)
    ap.add_argument("--swap-phase1-backend", type=str, default="backend_a", choices=["backend_a", "backend_b"])
    ap.add_argument("--swap-phase2-backend", type=str, default="backend_b", choices=["backend_a", "backend_b"])
    ap.add_argument("--blue-schema-mapper", type=str, default="static", choices=["static", "dynamic"])
    ap.add_argument("--schema-map-min-confidence", type=float, default=0.75)
    ap.add_argument("--schema-cache-scope", type=str, default="run", choices=["run", "persistent"])
    ap.add_argument("--schema-adapt-mode", type=str, default="contract_first", choices=["contract_first", "llm_first"])
    ap.add_argument("--backend-b-alias-mode", type=str, default="full", choices=["full", "minimal"])
    ap.add_argument("--mcp-enabled", dest="mcp_enabled", action="store_true")
    ap.add_argument("--no-mcp", dest="mcp_enabled", action="store_false")
    ap.add_argument("--mcp-tool", type=str, default="search_logs")
    ap.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "ollama"])
    ap.add_argument("--gemini-model", type=str, default="gemini-1.5-flash")
    ap.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", type=str, default="qwen3:8b")
    ap.add_argument("--llm-timeout-sec", type=float, default=8.0)
    ap.add_argument("--llm-prewarm", dest="llm_prewarm", action="store_true")
    ap.add_argument("--no-llm-prewarm", dest="llm_prewarm", action="store_false")
    ap.set_defaults(mcp_enabled=True)
    ap.set_defaults(llm_prewarm=True)
    args = ap.parse_args()
    swap_enabled = int(args.swap_episode) > 0
    if swap_enabled and (int(args.swap_episode) >= int(args.episodes)):
        raise ValueError("--swap-episode debe ser menor que --episodes (ej: 20 para 40 episodios).")

    python_exe = sys.executable
    repo_root = os.getcwd()
    experiment_id = args.experiment_id or f"exp_{args.base_seed}_{args.repetitions}r_{args.episodes}ep"
    experiment_dir = os.path.join(args.out_dir, experiment_id)
    _ensure_dir(experiment_dir)

    manifest: Dict[str, object] = {
        "experiment_id": experiment_id,
        "python_executable": python_exe,
        "repetitions": args.repetitions,
        "episodes": args.episodes,
        "base_seed": args.base_seed,
        "seed_step": args.seed_step,
        "noise_per_episode": args.noise_per_episode,
        "benign_rate": args.benign_rate,
        "recurrent_benign_rate": args.recurrent_benign_rate,
        "recurrent_benign_profiles": args.recurrent_benign_profiles,
        "backend_b_drift_profile": args.backend_b_drift_profile,
        "delay": args.delay,
        "memory_seed_dir": args.memory_seed_dir,
        "blue_backend": args.blue_backend,
        "swap_enabled": swap_enabled,
        "swap_episode": args.swap_episode,
        "swap_phase1_backend": args.swap_phase1_backend,
        "swap_phase2_backend": args.swap_phase2_backend,
        "blue_schema_mapper": args.blue_schema_mapper,
        "schema_map_min_confidence": args.schema_map_min_confidence,
        "schema_cache_scope": args.schema_cache_scope,
        "schema_adapt_mode": args.schema_adapt_mode,
        "backend_b_alias_mode": args.backend_b_alias_mode,
        "mcp_enabled": args.mcp_enabled,
        "mcp_tool": args.mcp_tool,
        "llm_provider": args.llm_provider,
        "gemini_model": args.gemini_model,
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "llm_timeout_sec": args.llm_timeout_sec,
        "llm_prewarm": args.llm_prewarm,
        "repetitions_data": [],
    }

    if args.blue_schema_mapper == "dynamic" and args.llm_provider == "ollama" and args.llm_prewarm:
        ok = _prewarm_ollama(url=args.ollama_url, model=args.ollama_model)
        manifest["llm_prewarm_ok"] = bool(ok)
        print(f"OLLAMA_PREWARM: {'ok' if ok else 'failed'}")

    records: List[RepRecord] = []
    for rep in range(args.repetitions):
        seed = args.base_seed + (rep * args.seed_step)
        rep_name = f"rep_{rep + 1:02d}"
        rep_dir = os.path.join(experiment_dir, rep_name)
        dataset_dir = os.path.join(rep_dir, "dataset")
        logs_dir_a = os.path.join(dataset_dir, "logs_backend_a")
        logs_dir_b = os.path.join(dataset_dir, "logs_backend_b")
        baseline_logs_dir = logs_dir_a
        blue_logs_dir = logs_dir_b if args.blue_backend == "backend_b" else logs_dir_a
        gt_dir = os.path.join(dataset_dir, "ground_truth")
        baseline_dir = os.path.join(rep_dir, "baseline")
        blue_run_id = f"blue_mem_{experiment_id}_{rep_name}"
        blue_dir = run_paths(blue_run_id)["base"]

        _ensure_dir(rep_dir)
        _ensure_dir(baseline_dir)

        _run(
            [
                python_exe,
                "src/generate_episodes.py",
                "--out",
                dataset_dir,
                "--episodes",
                str(args.episodes),
                "--base-seed",
                str(seed),
                "--noise-per-episode",
                str(args.noise_per_episode),
                "--benign-rate",
                str(args.benign_rate),
                "--recurrent-benign-rate",
                str(args.recurrent_benign_rate),
                "--recurrent-benign-profiles",
                str(args.recurrent_benign_profiles),
                "--backend-b-drift-profile",
                str(args.backend_b_drift_profile),
            ],
            cwd=repo_root,
        )

        baseline_decisions = os.path.join(baseline_dir, "decisions.jsonl")
        baseline_actions = os.path.join(baseline_dir, "enforcement_actions.jsonl")
        baseline_phase2 = os.path.join(baseline_dir, "results_phase2.csv")
        baseline_confusion_containment = os.path.join(baseline_dir, "results_confusion_containment.csv")
        baseline_confusion_detection = os.path.join(baseline_dir, "results_confusion_detection.csv")

        _run(
            [
                python_exe,
                "-m",
                "src.run_phase_2_baseline",
                "--logs-dir",
                baseline_logs_dir,
                "--gt-dir",
                gt_dir,
                "--decisions-path",
                baseline_decisions,
                "--actions-path",
                baseline_actions,
                "--out-csv",
                baseline_phase2,
            ],
            cwd=repo_root,
        )
        _run_confusion(
            python_exe=python_exe,
            cwd=repo_root,
            gt_dir=gt_dir,
            decisions_path=baseline_decisions,
            out_csv=baseline_confusion_containment,
            positive_decisions="block_ip",
        )
        _run_confusion(
            python_exe=python_exe,
            cwd=repo_root,
            gt_dir=gt_dir,
            decisions_path=baseline_decisions,
            out_csv=baseline_confusion_detection,
            positive_decisions="block_ip,escalate",
        )

        blue_paths = prepare_run(blue_run_id, clean=True, meta={"component": "blue_agent", "experiment_id": experiment_id})
        blue_memory_dir = os.path.join(blue_paths["base"], "memory")
        resolved_memory_seed_dir = _resolve_memory_seed_dir(args.memory_seed_dir, rep_name)
        _copy_memory_seed(resolved_memory_seed_dir, blue_memory_dir)

        if swap_enabled:
            phase1_end = int(args.swap_episode)
            phase2_start = phase1_end + 1
            phase1_backend = args.swap_phase1_backend
            phase2_backend = args.swap_phase2_backend

            phase1_cmd = _build_blue_cmd(
                python_exe=python_exe,
                run_id=blue_run_id,
                logs_dir=_blue_logs_dir_for_backend(dataset_dir=dataset_dir, backend=phase1_backend),
                backend=phase1_backend,
                gt_dir=gt_dir,
                episode_start=1,
                episode_end=phase1_end,
                schema_mapper_mode=args.blue_schema_mapper,
                schema_map_min_confidence=args.schema_map_min_confidence,
                schema_cache_scope=args.schema_cache_scope,
                schema_adapt_mode=args.schema_adapt_mode,
                backend_b_alias_mode=args.backend_b_alias_mode,
                mcp_tool=args.mcp_tool,
                llm_provider=args.llm_provider,
                gemini_model=args.gemini_model,
                ollama_url=args.ollama_url,
                ollama_model=args.ollama_model,
                llm_timeout_sec=args.llm_timeout_sec,
                delay=args.delay,
                mcp_enabled=bool(args.mcp_enabled),
            )
            _run(phase1_cmd, cwd=repo_root)

            phase2_cmd = _build_blue_cmd(
                python_exe=python_exe,
                run_id=blue_run_id,
                logs_dir=_blue_logs_dir_for_backend(dataset_dir=dataset_dir, backend=phase2_backend),
                backend=phase2_backend,
                gt_dir=gt_dir,
                episode_start=phase2_start,
                episode_end=int(args.episodes),
                schema_mapper_mode=args.blue_schema_mapper,
                schema_map_min_confidence=args.schema_map_min_confidence,
                schema_cache_scope=args.schema_cache_scope,
                schema_adapt_mode=args.schema_adapt_mode,
                backend_b_alias_mode=args.backend_b_alias_mode,
                mcp_tool=args.mcp_tool,
                llm_provider=args.llm_provider,
                gemini_model=args.gemini_model,
                ollama_url=args.ollama_url,
                ollama_model=args.ollama_model,
                llm_timeout_sec=args.llm_timeout_sec,
                delay=args.delay,
                mcp_enabled=bool(args.mcp_enabled),
            )
            _run(phase2_cmd, cwd=repo_root)

            blue_backend_str = f"{phase1_backend}->{phase2_backend}"
            blue_logs_dir = f"{_blue_logs_dir_for_backend(dataset_dir=dataset_dir, backend=phase1_backend)} -> {_blue_logs_dir_for_backend(dataset_dir=dataset_dir, backend=phase2_backend)}"
        else:
            blue_cmd = _build_blue_cmd(
                python_exe=python_exe,
                run_id=blue_run_id,
                logs_dir=blue_logs_dir,
                backend=args.blue_backend,
                gt_dir=gt_dir,
                episode_start=1,
                episode_end=int(args.episodes),
                schema_mapper_mode=args.blue_schema_mapper,
                schema_map_min_confidence=args.schema_map_min_confidence,
                schema_cache_scope=args.schema_cache_scope,
                schema_adapt_mode=args.schema_adapt_mode,
                backend_b_alias_mode=args.backend_b_alias_mode,
                mcp_tool=args.mcp_tool,
                llm_provider=args.llm_provider,
                gemini_model=args.gemini_model,
                ollama_url=args.ollama_url,
                ollama_model=args.ollama_model,
                llm_timeout_sec=args.llm_timeout_sec,
                delay=args.delay,
                mcp_enabled=bool(args.mcp_enabled),
            )
            _run(blue_cmd, cwd=repo_root)
            blue_backend_str = args.blue_backend

        blue_phase2 = os.path.join(blue_dir, "results_phase2.csv")
        blue_confusion_containment = os.path.join(blue_dir, "results_confusion_containment.csv")
        blue_confusion_detection = os.path.join(blue_dir, "results_confusion_detection.csv")

        judge_mttd_mttr(
            gt_dir=gt_dir,
            decisions_path=blue_paths["decisions"],
            actions_path=blue_paths["actions"],
            out_csv=blue_phase2,
        )
        _run_confusion(
            python_exe=python_exe,
            cwd=repo_root,
            gt_dir=gt_dir,
            decisions_path=blue_paths["decisions"],
            out_csv=blue_confusion_containment,
            positive_decisions="block_ip",
        )
        _run_confusion(
            python_exe=python_exe,
            cwd=repo_root,
            gt_dir=gt_dir,
            decisions_path=blue_paths["decisions"],
            out_csv=blue_confusion_detection,
            positive_decisions="block_ip,escalate",
        )

        record = RepRecord(
            repetition=rep + 1,
            seed=seed,
            memory_seed_dir_resolved=resolved_memory_seed_dir,
            dataset_dir=dataset_dir,
            logs_dir=baseline_logs_dir,
            baseline_logs_dir=baseline_logs_dir,
            blue_logs_dir=blue_logs_dir,
            blue_backend=blue_backend_str,
            blue_schema_mapper=args.blue_schema_mapper,
            schema_cache_scope=args.schema_cache_scope,
            schema_adapt_mode=args.schema_adapt_mode,
            backend_b_alias_mode=args.backend_b_alias_mode,
            mcp_enabled=bool(args.mcp_enabled),
            mcp_tool=args.mcp_tool,
            llm_provider=args.llm_provider,
            ollama_model=args.ollama_model,
            backend_b_drift_profile=args.backend_b_drift_profile,
            gt_dir=gt_dir,
            baseline_dir=baseline_dir,
            baseline_confusion_containment=baseline_confusion_containment,
            baseline_confusion_detection=baseline_confusion_detection,
            baseline_phase2=baseline_phase2,
            blue_run_id=blue_run_id,
            blue_dir=blue_dir,
            blue_confusion_containment=blue_confusion_containment,
            blue_confusion_detection=blue_confusion_detection,
            blue_phase2=blue_phase2,
            blue_swap_enabled=swap_enabled,
            blue_swap_episode=int(args.swap_episode) if swap_enabled else 0,
            blue_phase1_backend=args.swap_phase1_backend if swap_enabled else args.blue_backend,
            blue_phase2_backend=args.swap_phase2_backend if swap_enabled else "",
        )
        records.append(record)

    manifest["repetitions_data"] = [asdict(r) for r in records]
    manifest_path = os.path.join(experiment_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Wrote manifest:", manifest_path)
    print("Next step:")
    print(f"{python_exe} -m src.eval.aggregate_results --manifest {manifest_path}")


if __name__ == "__main__":
    main()

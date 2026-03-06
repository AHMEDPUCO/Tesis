from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List

from src.core.run_manager import prepare_run, run_paths
from src.judge.judge_mtd_mttr import judge_mttd_mttr


@dataclass
class RepRecord:
    repetition: int
    seed: int
    dataset_dir: str
    logs_dir: str
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=1337)
    ap.add_argument("--seed-step", type=int, default=100)
    ap.add_argument("--noise-per-episode", type=int, default=2000)
    ap.add_argument("--benign-rate", type=float, default=0.35)
    ap.add_argument("--delay", type=int, default=30)
    ap.add_argument("--memory-seed-dir", default="data/memory")
    ap.add_argument("--out-dir", default="data/experiments")
    args = ap.parse_args()

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
        "delay": args.delay,
        "memory_seed_dir": args.memory_seed_dir,
        "repetitions_data": [],
    }

    records: List[RepRecord] = []
    for rep in range(args.repetitions):
        seed = args.base_seed + (rep * args.seed_step)
        rep_name = f"rep_{rep + 1:02d}"
        rep_dir = os.path.join(experiment_dir, rep_name)
        dataset_dir = os.path.join(rep_dir, "dataset")
        logs_dir = os.path.join(dataset_dir, "logs_backend_a")
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
                logs_dir,
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
        _copy_memory_seed(args.memory_seed_dir, blue_memory_dir)

        for episode_id in range(1, args.episodes + 1):
            _run(
                [
                    python_exe,
                    "-m",
                    "src.blue.run_blue_agent",
                    "--episode-id",
                    str(episode_id),
                    "--run-id",
                    blue_run_id,
                    "--logs-dir",
                    logs_dir,
                    "--delay",
                    str(args.delay),
                    "--non-interactive",
                ],
                cwd=repo_root,
            )

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
            dataset_dir=dataset_dir,
            logs_dir=logs_dir,
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

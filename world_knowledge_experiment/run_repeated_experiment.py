"""5-fold名詞評価を複数の乱数seedで反復し、平均と標準偏差を求める。"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_knowledge_experiment import run_experiment as single  # noqa: E402


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    summary: dict[str, Any] = {}

    for model_name in single.MODEL_NAMES:
        summary[model_name] = {}
        for metric_name in metric_names:
            values = [
                run["pooled_metrics"][model_name][metric_name]
                for run in runs
            ]
            summary[model_name][metric_name] = mean_std(values)

    comparison_models = {
        "knowledge_minus_embedding": "embedding_mlp",
        "knowledge_minus_shuffled": "shuffled_knowledge_mlp",
        "knowledge_minus_rule": "knowledge_rule",
    }
    summary["balanced_accuracy_differences"] = {}
    for comparison_name, comparison_model in comparison_models.items():
        differences = [
            run["pooled_metrics"]["knowledge_mlp"]["balanced_accuracy"]
            - run["pooled_metrics"][comparison_model]["balanced_accuracy"]
            for run in runs
        ]
        summary["balanced_accuracy_differences"][comparison_name] = mean_std(
            differences
        )
    return summary


def combine_prediction_csvs(
    run_directories: list[tuple[int, Path]],
    output_path: Path,
) -> None:
    combined_rows = []
    fieldnames = None

    for seed, run_directory in run_directories:
        input_path = run_directory / "cross_validation_predictions.csv"
        with input_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if fieldnames is None:
                fieldnames = ["seed"] + list(reader.fieldnames or [])
            for row in reader:
                combined_rows.append({"seed": seed, **row})

    if fieldnames is None:
        raise ValueError("予測CSVが1件もありません。")

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)


def print_summary(
    summary: dict[str, Any],
    seeds: list[int],
    elapsed_seconds: float,
) -> None:
    print(
        f"Repeated 5-fold noun cross-validation: "
        f"{len(seeds)} seeds {seeds}"
    )
    print("\nMean +/- std across seeds")
    print("model                          accuracy          balanced accuracy   F1")
    for model_name in single.MODEL_NAMES:
        metrics = summary[model_name]
        print(
            f"{single.DISPLAY_NAMES[model_name]:<31}"
            f"{metrics['accuracy']['mean']:.3f} +/- "
            f"{metrics['accuracy']['std']:.3f}   "
            f"{metrics['balanced_accuracy']['mean']:.3f} +/- "
            f"{metrics['balanced_accuracy']['std']:.3f}       "
            f"{metrics['f1']['mean']:.3f} +/- "
            f"{metrics['f1']['std']:.3f}"
        )

    print("\nBalanced-accuracy differences")
    for name, values in summary["balanced_accuracy_differences"].items():
        print(f"{name}: {values['mean']:+.3f} +/- {values['std']:.3f}")
    print(f"\nTotal elapsed time: {elapsed_seconds:.1f} seconds")


def run_one_seed(
    seed: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    """1 seed分を独立に実行する。プロセス並列化からも呼べる形にしておく。"""
    run_directory = config["seed_runs_dir"] / f"seed_{seed}"
    single_args = SimpleNamespace(
        embeddings=config["embeddings"],
        output_dir=run_directory,
        seed=seed,
        shuffle_seed=config["shuffle_seed"] + seed * 1_000,
        hidden_size=config["hidden_size"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    run_started_at = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = single.main(single_args)
    run_elapsed = time.perf_counter() - run_started_at
    run_summary = {
        "seed": seed,
        "elapsed_seconds": run_elapsed,
        "pooled_metrics": {
            model_name: result["summary"][model_name]["pooled"]
            for model_name in single.MODEL_NAMES
        },
        "fold_balanced_accuracy": {
            model_name: result["summary"][model_name]["fold_mean_std"][
                "balanced_accuracy"
            ]
            for model_name in single.MODEL_NAMES
        },
    }
    return run_summary, run_directory


def main(args: argparse.Namespace) -> dict[str, Any]:
    if not args.seeds:
        raise ValueError("--seedsには1個以上の整数が必要です。")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seedsに重複があります。")
    if args.jobs < 1:
        raise ValueError("--jobsには1以上の整数が必要です。")

    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_runs_dir = args.output_dir / "seed_runs"
    seed_runs_dir.mkdir(parents=True, exist_ok=True)
    workers = min(args.jobs, len(args.seeds))
    seed_config = {
        "seed_runs_dir": seed_runs_dir,
        "embeddings": args.embeddings,
        "shuffle_seed": args.shuffle_seed,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    completed: dict[int, tuple[dict[str, Any], Path]] = {}

    if workers == 1:
        for seed in args.seeds:
            run_summary, run_directory = run_one_seed(seed, seed_config)
            completed[seed] = (run_summary, run_directory)
            print(
                f"seed {seed} finished in "
                f"{run_summary['elapsed_seconds']:.1f} seconds"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one_seed, seed, seed_config): seed
                for seed in args.seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                run_summary, run_directory = future.result()
                completed[seed] = (run_summary, run_directory)
                print(
                    f"seed {seed} finished in "
                    f"{run_summary['elapsed_seconds']:.1f} seconds"
                )

    run_summaries = [completed[seed][0] for seed in args.seeds]
    run_directories = [
        (seed, completed[seed][1]) for seed in args.seeds
    ]

    summary = summarize_runs(run_summaries)
    elapsed_seconds = time.perf_counter() - started_at
    repeated_results = {
        "config": {
            "embeddings": str(args.embeddings.resolve()),
            "seeds": args.seeds,
            "shuffle_seed": args.shuffle_seed,
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "jobs": workers,
            "elapsed_seconds": elapsed_seconds,
        },
        "runs": run_summaries,
        "summary": summary,
    }

    with (args.output_dir / "repeated_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(repeated_results, file, ensure_ascii=False, indent=2)
    combine_prediction_csvs(
        run_directories,
        args.output_dir / "repeated_cross_validation_predictions.csv",
    )
    print_summary(summary, args.seeds, elapsed_seconds)
    print(f"\nSaved repeated results to: {args.output_dir.resolve()}")
    return repeated_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=REPO_ROOT / "cbow_params.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(5)),
        help="反復に使うseed（既定: 0 1 2 3 4）",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20_260_730)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="同時に実行するseed数（既定: 1。複数指定で並列実行）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

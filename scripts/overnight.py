"""
Overnight experiments, sized for about 5 hours on 32 cores and one GPU.

Self-imitation on other difficult levels: on 1-3, imitating the agent's own best episodes made every run learn the route
around a dead end. This checks whether it also helps on two levels of different kinds:
  4-4  maze castle, where the wrong path loops back
  5-3  treetops with moving platforms

Variants, both with a lifelong novelty bonus of 0.5 and the defaults otherwise, 5 repeats of 300 iterations
(1.2M environment steps) on each level:
  base       no self-imitation
  sil-best   self-imitation of the 16 best episodes per level, loss weight 1.0

Runs are compared by the share of episodes reaching the flag in the last iterations, how many runs reached it at all
and how far they got. Two runs are trained in parallel, which completes about 1.3 times as many runs as one at a time.

Each run is a separate `train.py` process, logged to mlflow with the tags `experiment`, `variant`, `level` and
`repeat`. Runs that would not finish before the deadline are skipped.

Usage:
    systemd-inhibit --what=sleep:idle uv run scripts/overnight.py      # start, keeping the machine awake
    uv run scripts/overnight.py --dry-run                              # only print the plan
    uv run scripts/overnight.py --report --experiment NAME             # summarize the results

Hydra overrides after `--` are applied to every run.
"""

import argparse
import csv
import datetime
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEVELS = ["4-4", "5-3"]
ITERS = 300
REPEATS = 5
PARALLEL = 2
# Measured with two runs in parallel, including the start of a run
SECONDS_PER_RUN = ITERS * 5.2 + 25
# Number of final iterations whose episodes are averaged for the final values
FINAL_ITERS = 50
# Positions beyond this are rare reading errors of the game's memory
MAX_PLAUSIBLE_X = 5000

COMMON = [
    "levels=single",
    f"num_iters={ITERS}",
    "save_frequency=1000",
    "exploration.bonus_scale=0.5",
]
VARIANTS = [
    ("base", ["agent.sil_loss_weight=0.0"]),
    ("sil-best", ["agent.sil_loss_weight=1.0"]),
]
BASELINE_VARIANT = "base"


@dataclass
class Run:
    variant: str
    level: str
    repeat: int
    overrides: list[str]

    @property
    def name(self) -> str:
        return f"{self.level}-{self.variant}-{self.repeat}"


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def connect_mlflow(tracking_uri: str):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


def planned_runs() -> list[Run]:
    # Interleaved, so that a shortened night still leaves equally many repeats of every variant and level
    return [
        Run(name, level, repeat, [*COMMON, f"levels.level={level}", *overrides])
        for repeat in range(1, REPEATS + 1)
        for level in LEVELS
        for name, overrides in VARIANTS
    ]


def run_training(run: Run, args: argparse.Namespace, log_dir: Path) -> tuple[int, float]:
    overrides = (
        run.overrides
        + args.overrides
        + [
            f"tracking_uri={args.tracking_uri}",
            f"+tags.experiment={args.experiment}",
            f"+tags.variant={run.variant}",
            f"+tags.level={run.level}",
            f"+tags.repeat={run.repeat}",
            # Each run needs its own Hydra output directory, since runs start at the same time
            f"hydra.run.dir=outputs/{args.experiment}/{run.name}",
        ]
    )
    log_file = log_dir / f"{run.name}.log"
    print(f"[{now()}] start {run.name}, log {log_file}", flush=True)
    start = time.time()
    with open(log_file, "w") as log:
        process = subprocess.run(
            [sys.executable, str(ROOT / "src" / "train.py"), *overrides],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
        )
    duration = time.time() - start
    status = "ok" if process.returncode == 0 else f"FAILED with exit code {process.returncode}"
    print(f"[{now()}] {run.name} {status} after {duration / 60:.0f} min", flush=True)
    return process.returncode, duration


def start(args: argparse.Namespace) -> None:
    deadline = time.time() + args.hours * 3600
    log_dir = Path(args.log_root) / args.experiment
    runs = planned_runs()
    planned_s = len(runs) * SECONDS_PER_RUN / PARALLEL
    print(f"Experiment {args.experiment}, deadline {datetime.datetime.fromtimestamp(deadline):%H:%M}")
    print(f"{len(runs)} runs on {', '.join(LEVELS)}, {PARALLEL} in parallel, about {planned_s / 3600:.1f} h")
    if args.dry_run:
        for run in runs:
            print(f"  {run.name}: {' '.join(run.overrides)}")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        csv.writer(f).writerow(["run", "exit_code", "minutes"])
    summary_lock = threading.Lock()

    def execute(run: Run) -> None:
        if time.time() + SECONDS_PER_RUN > deadline:
            print(f"[{now()}] skip {run.name}, it would not finish before the deadline", flush=True)
            return
        exit_code, duration = run_training(run, args, log_dir)
        with summary_lock, open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([run.name, exit_code, round(duration / 60, 1)])

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for i, run in enumerate(runs):
            pool.submit(execute, run)
            # Keep the first runs apart, so they do not create the mlflow database at the same time
            if i < PARALLEL:
                time.sleep(15)

    print(
        f"[{now()}] done, see {summary_path} and `uv run scripts/overnight.py --report --experiment {args.experiment}`"
    )


def run_results(mlflow, run_id: str) -> dict[str, float]:
    client = mlflow.MlflowClient()

    def history(key: str) -> list[float]:
        return [m.value for m in sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)]

    x_mean, x_max, flags = history("episode_x_pos_mean"), history("episode_x_pos_max"), history("flag_rate")
    smoothed = [sum(x_mean[i - 9 : i + 1]) / 10 for i in range(9, len(x_mean))]
    return {
        "final_flag_rate": sum(flags[-FINAL_ITERS:]) / len(flags[-FINAL_ITERS:]),
        "flagged": sum(flags) > 0,
        "first_flag": next((i for i, rate in enumerate(flags) if rate > 0), math.inf),
        "final_x": sum(x_mean[-FINAL_ITERS:]) / len(x_mean[-FINAL_ITERS:]),
        "best_x": max(smoothed) if smoothed else math.nan,
        "max_x": max((x for x in x_max if x < MAX_PLAUSIBLE_X), default=math.nan),
    }


def fisher_p(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    """Two-sided Fisher exact test of two success rates."""
    total, n = successes_a + successes_b, n_a + n_b

    def probability(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total - k) / math.comb(n, total)

    observed = probability(successes_a)
    return min(
        1.0,
        sum(
            probability(k)
            for k in range(max(0, total - n_b), min(total, n_a) + 1)
            if probability(k) <= observed * (1 + 1e-9)
        ),
    )


def summary(results: list[dict[str, float]], key: str, scale: float = 1, unit: str = "") -> str:
    """Mean and range of a result over runs."""
    numbers = [r[key] * scale for r in results]
    return f"{sum(numbers) / len(numbers):.0f}{unit} ({min(numbers):.0f}-{max(numbers):.0f})"


def report(args: argparse.Namespace) -> None:
    mlflow = connect_mlflow(args.tracking_uri)
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.experiment = '{args.experiment}' and attributes.status = 'FINISHED'",
        output_format="list",
    )
    if not runs:
        print(f"No finished runs of experiment {args.experiment} found.")
        return
    results: dict[tuple[str, str], list[dict[str, float]]] = {}
    for run in runs:
        results.setdefault((run.data.tags["level"], run.data.tags["variant"]), []).append(
            run_results(mlflow, run.info.run_id)
        )

    print("Per level and variant, mean over runs (range):")
    print(f"  final flags: share of episodes reaching the flag in the last {FINAL_ITERS} iterations")
    print("  runs with flag: runs that reached the flag at all, p: Fisher exact test against the baseline")
    print(f"  final x / best x: mean distance in the last {FINAL_ITERS} iterations / best 10-iteration mean")
    for level in LEVELS:
        print(f"\n{level}")
        print(
            f"  {'variant':10s} {'runs':>4s} {'final flags':>14s} {'runs with flag':>15s} {'p':>5s} "
            f"{'first flag':>11s} {'final x':>16s} {'best x':>16s} {'max x':>6s}"
        )
        base = results.get((level, BASELINE_VARIANT), [])
        for name, _ in VARIANTS:
            values = results.get((level, name))
            if not values:
                continue

            flagged = sum(bool(r["flagged"]) for r in values)
            p = (
                fisher_p(flagged, len(values), sum(bool(r["flagged"]) for r in base), len(base))
                if base and name != BASELINE_VARIANT
                else math.nan
            )
            first_flags = [r["first_flag"] for r in values if not math.isinf(r["first_flag"])]
            first = f"{min(first_flags)}" if first_flags else "-"
            print(
                f"  {name:10s} {len(values):4d} {summary(values, 'final_flag_rate', 100, '%'):>14s} "
                f"{flagged:>9d} / {len(values):<3d} {p:5.2f} {first:>11s} {summary(values, 'final_x'):>16s} "
                f"{summary(values, 'best_x'):>16s} {max(r['max_x'] for r in values):6.0f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=7.0, help="Time budget of the whole night")
    parser.add_argument(
        "--experiment",
        default=f"overnight-{datetime.datetime.now():%Y-%m-%d-%H%M}",
        help="Name of the experiment tag, by default from the start time; required with --report",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running anything")
    parser.add_argument("--report", action="store_true", help="Summarize the results of an experiment")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--log-root", default=str(ROOT / "overnight"), help="Directory for the logs of the runs")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides for all runs")
    args = parser.parse_args()
    if args.report:
        if "--experiment" not in sys.argv:
            parser.error("--report needs --experiment, the name is printed when a batch starts and ends")
        report(args)
    else:
        start(args)


if __name__ == "__main__":
    main()

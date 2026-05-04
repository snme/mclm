"""Per-run output: metrics.json + predictions.jsonl under a run dir."""
import json
import os
from pathlib import Path

POOL_RESULTS_ROOT = os.environ.get(
    "ALM_EVAL_RESULTS_ROOT",
    "/home/sathyae/orcd/pool/eval_results",
)


def run_dir(benchmark, checkpoint_path, run_id=None):
    """Resolve the per-run output dir.

    Precedence for the run_id:
      1) explicit `run_id` arg (used by language_retention to embed --model in the name)
      2) ALM_EVAL_RUN_ID env var (set by helper_scripts/run_all_evals.sh to prevent
         basename collisions — multiple checkpoints can share `step=N`)
      3) Path(checkpoint_path).name (legacy default: just the step= dir)
    """
    rid = run_id or os.environ.get("ALM_EVAL_RUN_ID") or Path(checkpoint_path).name
    d = Path(POOL_RESULTS_ROOT) / benchmark / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run(run_path, metrics, predictions):
    run_path = Path(run_path)
    with open(run_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(run_path / "predictions.jsonl", "w") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")
    print(f"[eval] wrote {run_path}/metrics.json ({len(predictions)} predictions)")

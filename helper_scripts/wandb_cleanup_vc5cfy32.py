"""Back up wandb run vc5cfy32 (snme/alm-stage2) and report what's polluted.

Run order:
  1. python helper_scripts/wandb_cleanup_vc5cfy32.py --backup
       Pulls all history into a local JSON; prints summary of clean vs polluted.
  2. python helper_scripts/wandb_cleanup_vc5cfy32.py --delete
       After you've stopped training and confirmed the backup, deletes the run.
       This is irreversible — make sure step 1 produced a non-empty JSON.

Polluted = entries where the wandb_step is much larger than expected for the
opt_step (because of a buggy `step=global_opt_step` arg in train_stage2.py
between the 1-node run and the failed first 2-node resume). Clean = entries
that follow the auto-step convention (wandb_step ≈ opt_step / log_every).
"""
import argparse
import json
from pathlib import Path

import wandb


RUN_PATH = "snme/alm-stage2/vc5cfy32"
BACKUP_PATH = Path("/home/sathyae/mclm/evals/vc5cfy32_backup.json")


def backup():
    api = wandb.Api()
    run = api.run(RUN_PATH)
    print(f"Run: {run.name} ({run.id}), state={run.state}")
    print(f"Created: {run.created_at}")
    print(f"Config keys: {list(run.config.keys())[:5]} ...")

    # `scan_history` streams all logged metrics; `keys=None` pulls everything.
    history = list(run.scan_history())
    print(f"\nTotal history rows: {len(history)}")

    # Classify rows: clean (wandb_step < ~600) vs polluted (wandb_step > 5000).
    clean, polluted = [], []
    for r in history:
        ws = r.get("_step", -1)
        gs = r.get("global_opt_step")
        if ws < 1000:
            clean.append(r)
        elif ws >= 5000:
            polluted.append(r)
    print(f"  clean (wandb_step < 1000):    {len(clean)}")
    print(f"  polluted (wandb_step ≥ 5000): {len(polluted)}")

    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BACKUP_PATH, "w") as f:
        json.dump({
            "run_path": RUN_PATH,
            "name": run.name,
            "config": dict(run.config),
            "summary": dict(run.summary._json_dict),
            "history": history,
        }, f, indent=2, default=str)
    size_mb = BACKUP_PATH.stat().st_size / 1e6
    print(f"\nBacked up to {BACKUP_PATH} ({size_mb:.1f} MB)")
    print("\nNext: stop training (kill torchrun on both nodes), then run with --delete to remove the polluted run.")


def delete():
    if not BACKUP_PATH.exists():
        print(f"ERROR: backup not found at {BACKUP_PATH}. Run --backup first.")
        return
    print(f"Backup verified: {BACKUP_PATH} ({BACKUP_PATH.stat().st_size / 1e6:.1f} MB)")
    api = wandb.Api()
    run = api.run(RUN_PATH)
    print(f"Deleting run: {run.name} ({run.id}) — state={run.state}")
    if run.state == "running":
        print("  WARN: run is still in 'running' state. Stop training first.")
        return
    run.delete()
    print(f"Deleted {RUN_PATH}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backup", action="store_true")
    p.add_argument("--delete", action="store_true")
    args = p.parse_args()
    if args.backup:
        backup()
    elif args.delete:
        delete()
    else:
        p.print_help()

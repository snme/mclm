"""Replay the CLEAN portion of vc5cfy32's history into a fresh wandb run.

Reads `evals/vc5cfy32_backup.json` (produced by wandb_cleanup_vc5cfy32.py --backup),
filters to the legitimate 1-node training history (the auto-stepped portion),
and re-logs each row into a new wandb run with `step=global_opt_step` so the
dashboard's wandb-step axis IS opt_step from the start. The corrupted 1:1 step=
rows from the failed first resume are dropped.

Output: prints the new run's id. Use that as `--wandb_run_id <new_id>` on the
next training launch to continue logging into the same clean run.
"""
import json
from pathlib import Path

import wandb


BACKUP_PATH = Path("/home/sathyae/mclm/evals/vc5cfy32_backup.json")
PROJECT = "alm-stage2"
ENTITY = "snme"


def main():
    with open(BACKUP_PATH) as f:
        data = json.load(f)

    config = data["config"]
    history = data["history"]
    print(f"Loaded {len(history)} rows from {BACKUP_PATH}")

    # Keep only auto-stepped rows from the 1-node run AND only opt_steps that
    # match the resume checkpoint, so the resumed training doesn't double-log
    # the overlap range. The 1-node run trained past the saved checkpoint
    # (e.g. ckpt at step=5500 but training continued to ~6440); we drop those.
    RESUME_OPT_STEP = 5500  # match --resume_from_stage2 step=N checkpoint
    clean = [r for r in history
             if r.get("_step", 1e9) < 1000
             and r.get("global_opt_step") is not None
             and r["global_opt_step"] <= RESUME_OPT_STEP]
    clean.sort(key=lambda r: r["global_opt_step"])
    print(f"Replaying {len(clean)} clean rows into a new run.")
    if clean:
        first, last = clean[0]["global_opt_step"], clean[-1]["global_opt_step"]
        print(f"  global_opt_step range: {first} → {last}")

    # Start a brand-new run; same project, same config (so resume_from_stage2 etc.
    # are reflected). Different name and a new auto-id.
    run = wandb.init(
        project=PROJECT, entity=ENTITY, config=config,
        name="stage2_r128_arxivIT_clean", job_type="replay",
    )
    print(f"New run: {run.id}  ({run.url})")

    skipped_keys = {"_step", "_runtime", "_timestamp", "_wandb"}
    for r in clean:
        gs = int(r["global_opt_step"])
        payload = {k: v for k, v in r.items()
                   if k not in skipped_keys and v is not None}
        wandb.log(payload, step=gs)

    wandb.finish()
    print(f"\nDone. Use --wandb_run_id {run.id} on the next training launch "
          f"to continue logging into this clean run.")


if __name__ == "__main__":
    main()

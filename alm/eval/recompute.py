"""Recompute metrics.json from predictions.jsonl using hardened parsers.

Why: r64/12k and r128/6k both leak Qwen3-base markdown-image / URL fallbacks
into eval outputs. The old parsers extracted digits from inside URL hashes, so
those rows counted as "valid" with garbage parsed values (mat2props band_gap
MAE 51 from materialsproject.org/materials/101112 → parsed=101112).

This module re-runs the (now leak-aware) parsers on saved generations, splits
failures into `n_leaked` (Qwen3 fallback) vs `n_parse_fail` (genuine failure),
and recomputes per-bucket metrics on the leak-free, parse-ok subset only.
The original metrics.json is preserved at metrics.json.preimg-fix.

Usage from a script: `recompute_metrics(run_dir, benchmark)`.
"""
import json
import shutil
from pathlib import Path

from parsers import detect_leak, extract_choice, extract_number
from metrics import accuracy, mad_mae_ratio, mae, rmse, weighted_f1


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _bucket_metrics_reg(rows):
    """Common regression bucket: take rows that are non-leaked AND parsable AND
    have a numeric target. Compute MAE / RMSE / MAD:MAE on that subset.
    Returns a metrics dict including n_total, n_leaked, n_parse_fail, n_valid."""
    n_total = len(rows)
    leaks = [r for r in rows if r["leaked"]]
    nonleak = [r for r in rows if not r["leaked"]]
    valid = [r for r in nonleak if r["parsed_new"] is not None and _is_num(r.get("target"))]
    n_leaked = len(leaks)
    n_valid = len(valid)
    n_parse_fail = n_total - n_leaked - n_valid
    out = {
        "n_total": n_total,
        "n_leaked": n_leaked,
        "n_parse_fail": n_parse_fail,
        "n_valid": n_valid,
        "validity_rate": (n_valid / n_total) if n_total else 0.0,
        "leak_rate": (n_leaked / n_total) if n_total else 0.0,
    }
    if valid:
        preds = [v["parsed_new"] for v in valid]
        targets = [float(v["target"]) for v in valid]
        out["mae"] = mae(preds, targets)
        out["rmse"] = rmse(preds, targets)
        out["mad_mae_ratio"] = mad_mae_ratio(preds, targets)
    return out


def _bucket_metrics_cls(rows, target_key="target", parsed_key="parsed_new"):
    """Classification bucket: rows are non-leaked AND have a parsed letter AND
    a non-None target. Computes accuracy + weighted F1."""
    n_total = len(rows)
    leaks = [r for r in rows if r["leaked"]]
    nonleak = [r for r in rows if not r["leaked"]]
    valid = [r for r in nonleak if r[parsed_key] is not None and r.get(target_key) is not None]
    n_leaked = len(leaks)
    n_valid = len(valid)
    n_parse_fail = n_total - n_leaked - n_valid
    out = {
        "n_total": n_total,
        "n_leaked": n_leaked,
        "n_parse_fail": n_parse_fail,
        "n_valid": n_valid,
        "validity_rate": (n_valid / n_total) if n_total else 0.0,
        "leak_rate": (n_leaked / n_total) if n_total else 0.0,
    }
    if valid:
        preds = [v[parsed_key] for v in valid]
        targets = [v[target_key] for v in valid]
        out["accuracy"] = accuracy(preds, targets)
        out["weighted_f1"] = weighted_f1(preds, targets)
    return out


def _is_num(x):
    if x is None:
        return False
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _annotate(row, parsed_new):
    """Tag a row with leaked + parsed_new in-place; return mutated row."""
    row["leaked"] = detect_leak(row.get("generated"))
    row["parsed_new"] = parsed_new
    return row


# ---------- per-benchmark recomputers ----------

def _recompute_llm4mat(rows, old_metrics):
    """Predictions are appended in eval_llm4mat.py's iteration order:
    config-list (mp, jarvis_dft, oqmd, gnome, snumat, hmof, cantor_hea,
    jarvis_qetb, omdb, qmof) × _DATASET_PROPERTIES[config] order. The saved
    metrics.json was written with sort_keys=True so its key order is
    alphabetical and does NOT match the prediction order — slicing by the
    metrics dict's iteration order silently scrambles the buckets. We rebuild
    the canonical (config, prop) sequence here, matching eval_llm4mat.py.

    Some property names recur across configs (bandgap in omdb/oqmd/qmof, lcd
    /pld in hmof/qmof) so we cannot just group by property alone."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from utils import _DATASET_PROPERTIES

    canonical_configs = ["mp", "jarvis_dft", "oqmd", "gnome", "snumat",
                         "hmof", "cantor_hea", "jarvis_qetb", "omdb", "qmof"]
    by_config_old = old_metrics.get("by_config", {})

    new_by_config = {}
    cursor = 0
    for config in canonical_configs:
        if config not in by_config_old:
            continue
        new_by_config[config] = {}
        for prop in _DATASET_PROPERTIES.get(config, []):
            if prop not in by_config_old[config]:
                continue
            n = by_config_old[config][prop].get("n_total", 0)
            slice_rows = rows[cursor : cursor + n]
            cursor += n
            for r in slice_rows:
                _annotate(r, extract_number(r.get("generated")))
            new_by_config[config][prop] = _bucket_metrics_reg(slice_rows)

    if cursor != len(rows):
        print(f"[warn] llm4mat recompute consumed {cursor} rows of {len(rows)}; "
              "some predictions unmatched (different config order or extra props)")

    out = dict(old_metrics)
    out["by_config"] = new_by_config
    return out


def _recompute_by_task_reg(rows, old_metrics):
    """matterchat regression buckets, mat2props, mattext, gnome_fe (single bucket)."""
    by_task = {}
    grouped = {}
    for r in rows:
        key = r.get("task") or r.get("property") or "default"
        grouped.setdefault(key, []).append(r)
    for key, group in grouped.items():
        for r in group:
            _annotate(r, extract_number(r.get("generated")))
        by_task[key] = _bucket_metrics_reg(group)
    return by_task


def _recompute_matterchat(rows, old_metrics):
    """MatterChat: per-task; regression vs classification distinguished by old
    metrics' presence of 'mae' (reg) or 'accuracy' (cls), and matched against
    the original predictions' target letters (rows carry `target_letter` for cls)."""
    grouped = {}
    for r in rows:
        grouped.setdefault(r.get("task", "default"), []).append(r)
    out = {}
    for key, group in grouped.items():
        old = old_metrics.get(key, {})
        is_cls = "accuracy" in old or any("target_letter" in r for r in group)
        if is_cls:
            for r in group:
                # Re-derive choices from the union of old targets so multi-class
                # tasks (A-G crystal_system) still parse correctly.
                tgts = {r.get("target_letter") for r in group if r.get("target_letter")}
                choices = tuple(sorted(tgts)) or ("A", "B", "C", "D")
                _annotate(r, extract_choice(r.get("generated"), choices=choices))
            out[key] = _bucket_metrics_cls(group, target_key="target_letter", parsed_key="parsed_new")
        else:
            for r in group:
                _annotate(r, extract_number(r.get("generated")))
            out[key] = _bucket_metrics_reg(group)
    return out


def _recompute_mat2mcq(rows, old_metrics):
    for r in rows:
        _annotate(r, extract_choice(r.get("generated")))
    n_total = len(rows)
    leaks = [r for r in rows if r["leaked"]]
    valid = [r for r in rows if not r["leaked"] and r["parsed_new"] is not None]
    out = {
        "n_total": n_total,
        "n_leaked": len(leaks),
        "n_valid": len(valid),
        "validity_rate": (len(valid) / n_total) if n_total else 0.0,
        "leak_rate": (len(leaks) / n_total) if n_total else 0.0,
    }
    if valid:
        out["accuracy"] = accuracy([r["parsed_new"] for r in valid],
                                    [r["target"] for r in valid])
    return out


def _recompute_mascqa(rows, old_metrics):
    """Two-mode bucket: mcq (extract_choice → letter accuracy) + numerical (extract_number → MAE)."""
    out = {"n_total": len(rows)}
    mcq_rows = [r for r in rows if r.get("mode") == "mcq"]
    num_rows = [r for r in rows if r.get("mode") == "numerical"]
    out["n_mcq"] = len(mcq_rows)
    out["n_numerical"] = len(num_rows)

    if mcq_rows:
        for r in mcq_rows:
            _annotate(r, extract_choice(r.get("generated")))
        m = _bucket_metrics_cls(mcq_rows)
        out["mcq_accuracy"] = m.get("accuracy")
        out["mcq_n_leaked"] = m["n_leaked"]
        out["mcq_n_valid"] = m["n_valid"]
        out["mcq_leak_rate"] = m["leak_rate"]
        out["mcq_validity_rate"] = m["validity_rate"]

    if num_rows:
        for r in num_rows:
            _annotate(r, extract_number(r.get("generated")))
        m = _bucket_metrics_reg(num_rows)
        out["numerical_mae"] = m.get("mae")
        out["numerical_n_leaked"] = m["n_leaked"]
        out["numerical_n_valid"] = m["n_valid"]
        out["numerical_leak_rate"] = m["leak_rate"]
        out["numerical_validity_rate"] = m["validity_rate"]
    return out


def _recompute_language_retention(rows, old_metrics):
    """3 task subsets (mmlu, gpqa = MCQ; gsm8k = numeric)."""
    grouped = {"mmlu": [], "gpqa": [], "gsm8k": []}
    for r in rows:
        t = r.get("task")
        if t in grouped:
            grouped[t].append(r)
    out = dict(old_metrics)
    for t, group in grouped.items():
        if not group:
            continue
        if t == "gsm8k":
            for r in group:
                _annotate(r, extract_number(r.get("generated")))
            n_total = len(group)
            leaks = [r for r in group if r["leaked"]]
            valid = [r for r in group if not r["leaked"] and r["parsed_new"] is not None]
            n_correct = sum(1 for r in valid
                            if _is_num(r.get("target")) and abs(r["parsed_new"] - float(r["target"])) < 1e-3)
            out[f"{t}_n"] = n_total
            out[f"{t}_n_valid"] = len(valid)
            out[f"{t}_n_leaked"] = len(leaks)
            out[f"{t}_leak_rate"] = (len(leaks) / n_total) if n_total else 0.0
            out[f"{t}_validity_rate"] = (len(valid) / n_total) if n_total else 0.0
            out[f"{t}_accuracy"] = (n_correct / len(valid)) if valid else None
        else:
            for r in group:
                _annotate(r, extract_choice(r.get("generated")))
            m = _bucket_metrics_cls(group)
            out[f"{t}_n_total"] = m["n_total"]
            out[f"{t}_n_valid"] = m["n_valid"]
            out[f"{t}_n_leaked"] = m["n_leaked"]
            out[f"{t}_leak_rate"] = m["leak_rate"]
            out[f"{t}_validity_rate"] = m["validity_rate"]
            out[f"{t}_accuracy"] = m.get("accuracy")
    return out


# ---------- dispatcher ----------

_DISPATCH = {
    "llm4mat":           _recompute_llm4mat,
    "matterchat":        _recompute_matterchat,
    "mat2props":         _recompute_by_task_reg,
    "mattext":           _recompute_by_task_reg,
    "gnome_fe":          _recompute_by_task_reg,
    "mat2mcq":           _recompute_mat2mcq,
    "mascqa":            _recompute_mascqa,
    "language_retention": _recompute_language_retention,
}


def recompute_metrics(run_dir, benchmark):
    """Recompute metrics.json from predictions.jsonl with leak-aware parsers.

    Backs up the original to `metrics.json.preimg-fix` (only on first call;
    repeated calls leave the backup intact). Returns (old_metrics, new_metrics).
    Raises FileNotFoundError if predictions.jsonl is missing.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "predictions.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(f"no predictions.jsonl at {preds_path}")
    old_metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}
    rows = _read_jsonl(preds_path)

    fn = _DISPATCH.get(benchmark)
    if fn is None:
        raise ValueError(f"unknown benchmark {benchmark!r}; known: {list(_DISPATCH)}")

    if benchmark == "mat2props":
        new_by_task = fn(rows, old_metrics)
        new_metrics = new_by_task
    elif benchmark in ("mattext",):
        new_metrics = fn(rows, old_metrics)
    elif benchmark == "gnome_fe":
        new_by_task = fn(rows, old_metrics)
        new_metrics = new_by_task.get("default") or next(iter(new_by_task.values()), {})
    else:
        new_metrics = fn(rows, old_metrics)

    backup = run_dir / "metrics.json.preimg-fix"
    if metrics_path.exists() and not backup.exists():
        shutil.copy(metrics_path, backup)
    with open(metrics_path, "w") as f:
        json.dump(new_metrics, f, indent=2, sort_keys=True)
    return old_metrics, new_metrics

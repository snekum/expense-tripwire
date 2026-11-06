"""Part D: train the v2 probe on UMWP and SQuAD 2.0, choose its layer, and score the test split once.

    uv run python train_probe_v2.py                     # all stages; the test split is scored once
    uv run python train_probe_v2.py --validation-only   # stop before the test split
    uv run python train_probe_v2.py --dry-run --acts DIR   # any activations folder; writes to data/dry_run/

Stages (PLAN.md, Part D, "Choosing the v2 layer"):
  0. Integrity checks: the activations are the Colab v2 run with all checks passed, made from the
     current data/v2/model_inputs.jsonl; one finite row per ID; IDs, labels and splits agree.
  1. Layer choice, on validation only. For each layer 1-36, in both directions: train a probe on one
     dataset's train split, choose C on that dataset's validation split, and score the other
     dataset's validation split (the transfer AUROC). The layer's score is the mean of the two
     transfer AUROCs. Take the best layer, or the earliest layer within 0.01 of it. The same is done
     with shuffled training labels at every layer (a control that should stay near 0.5), and once
     for length and TF-IDF (reference lines; they don't depend on the layer).
  2. Final settings, on validation only. At the chosen layer and at layer 18: probe and shuffled
     control trained on pooled UMWP + SQuAD train, C by pooled validation AUROC, threshold by pooled
     validation balanced accuracy. Same for length and TF-IDF. Also, for the transfer check, each
     method trained on one dataset with C and threshold from that dataset's validation split.
     Everything is written to settings.json before the test split is touched.
  3. The test split, scored once: pooled, per dataset, and transfer (trained on one dataset, tested
     on the other). Refuses to overwrite earlier test results unless --overwrite-test is given,
     and records it when it is.

Methods (logistic regression with L2 regularization, target = should_abstain), as in the pilot:
  probe     the 2,048 activation values at one layer, standardized (scaler fit on train)
  length    word count and character count of the question text, standardized
  tfidf     TF-IDF of the question text (vocabulary fit on train)
  shuffled  the probe pipeline trained on shuffled training labels; evaluated on true labels
Scores are decision values (log-odds), not calibrated probabilities.
"""

import argparse
import json
import os
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

import aya_common as ac

V2 = ac.ROOT / "data" / "v2"
ACTS_DEFAULT = ac.ROOT / "data" / "colab_outputs" / "activations" / "v2_layers0-36_cuda_float16"
N_LAYERS = 36
HIDDEN = 2048
FIXED_LAYER = ac.DEFAULT_LAYER          # 18: the pilot's layer, always reported next to the chosen one
LAYER_TOLERANCE = 0.01
DATASETS = ("umwp", "squad2")
NAMES = {"umwp": "UMWP", "squad2": "SQuAD"}
N_JOBS = max(1, min(6, (os.cpu_count() or 2) - 2))


# ---------------------------------------------------------------- stage 0
def load_and_check(acts_dir: Path, dry_run: bool) -> tuple[pd.DataFrame, Path, dict]:
    acts_path = acts_dir / "activations.npy"
    for f in ("activations.npy", "example_ids.json", "meta.json"):
        if not (acts_dir / f).exists():
            raise SystemExit(f"{acts_dir / f} is missing. Download My Drive/expense-tripwire/outputs/ "
                             f"to data/colab_outputs/ first, or pass --acts.")
    acts = np.load(acts_path, mmap_mode="r")
    ids = json.loads((acts_dir / "example_ids.json").read_text(encoding="utf-8"))
    meta = json.loads((acts_dir / "meta.json").read_text(encoding="utf-8"))
    inputs = pd.read_json(V2 / "model_inputs.jsonl", lines=True)
    labels = pd.read_json(V2 / "labels.jsonl", lines=True)
    summary = json.loads((V2 / "summary.json").read_text(encoding="utf-8"))

    checks = {
        "v2 data passed its own checks (prepare_v2_data.py)": summary["all_checks_passed"],
        "activations: the v2 set, all 36 layers, all checks passed": meta["set"] == "v2"
            and meta["layers_loaded"] == N_LAYERS and meta["all_checks_passed"],
        "activations: the pinned model revision": meta["revision"] == ac.MODEL_REVISION,
        "activations made from the current model_inputs.jsonl": meta["source_sha256"] == ac.sha256(V2 / "model_inputs.jsonl"),
        "every ID appears once": len(ids) == len(set(ids)) == len(inputs) == meta["n"],
        "activation IDs = input IDs = label IDs": set(ids) == set(inputs["example_id"]) == set(labels["example_id"]),
        f"one ({N_LAYERS + 1}, {HIDDEN}) row per ID": acts.shape == (len(ids), N_LAYERS + 1, HIDDEN),
        "all activations finite": all(bool(np.isfinite(acts[i:i + 500]).all()) for i in range(0, len(ids), 500)),
    }
    df = inputs.merge(labels, on=["example_id", "dataset", "split"], how="inner", validate="one_to_one")
    checks["inputs and labels agree on every split"] = len(df) == len(inputs)
    df = df.set_index("example_id").loc[ids].reset_index()      # row i of df <-> row i of acts
    checks["no pair or passage in two splits"] = bool((df.groupby("original_id")["split"].nunique() == 1).all())
    checks["split sizes match prepare_v2_data.py"] = all(
        int(((df["dataset"] == d) & (df["split"] == s)).sum()) == v["rows"]
        for d, info in summary["datasets"].items() for s, v in info["splits"].items())

    print("Stage 0: integrity checks" + (" (dry run)" if dry_run else ""))
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("Integrity checks failed; stopping before any training.")
    return df, acts_path, meta


# ---------------------------------------------------------------- fitting
def choose_C(method: str, X, y_fit, fit_rows, y_val, val_rows):
    """Fit at each C; keep the best validation AUROC (ties keep the smaller C). Returns
    (validation AUROC, C, fitted pipeline, one row per C)."""
    best, rows = None, []
    for C in ac.C_GRID:
        pipe = ac.make_pipeline(method, C).fit(X[fit_rows], y_fit)
        auc = roc_auc_score(y_val, pipe.decision_function(X[val_rows]))
        rows.append({"C": C, "validation_auroc": auc})
        if best is None or auc > best[0]:
            best = (auc, C, pipe)
    return (*best, rows)


def transfer_scores(method: str, X, y, masks: dict, shuffled: dict) -> dict:
    """Train on one dataset (C from its validation split); score the other dataset's validation split."""
    row = {}
    for a, b in (DATASETS, DATASETS[::-1]):
        tr, va, other = masks[(a, "train")], masks[(a, "validation")], masks[(b, "validation")]
        y_fit = shuffled[a] if method == "shuffled" else y[tr]
        auc, C, pipe, _ = choose_C(method, X, y_fit, tr, y[va], va)
        row[f"{a}_within"] = auc
        row[f"{a}_to_{b}"] = roc_auc_score(y[other], pipe.decision_function(X[other]))
        row[f"{a}_C"] = C
    row["transfer_mean"] = (row[f"{DATASETS[0]}_to_{DATASETS[1]}"] + row[f"{DATASETS[1]}_to_{DATASETS[0]}"]) / 2
    return row


def layer_row(acts_path: Path, layer: int, y, masks: dict, shuffled: dict) -> dict:
    """One point of the layer curve: the probe and the shuffled control at this layer."""
    X = np.asarray(np.load(acts_path, mmap_mode="r")[:, layer, :], dtype=np.float32)
    row = {"layer": layer}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for method in ("probe", "shuffled"):
            row.update({f"{method}_{k}": v for k, v in transfer_scores(method, X, y, masks, shuffled).items()})
    row["convergence_warnings"] = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return row


# ---------------------------------------------------------------- reporting helpers
def scored(y, scores, threshold, groups, rng) -> dict:
    return {**ac.metrics(y, scores, threshold), **ac.bootstrap(y, scores, threshold, groups, rng)}


def fmt(r: dict) -> str:
    return (f"{r['auroc']:.3f} ({r['auroc_95ci'][0]:.3f}-{r['auroc_95ci'][1]:.3f}) | "
            f"{r['balanced_accuracy']:.3f} ({r['balanced_accuracy_95ci'][0]:.3f}-{r['balanced_accuracy_95ci'][1]:.3f})")


def curve_markdown(curve: pd.DataFrame, baselines: dict, chosen: int) -> list[str]:
    cols = [f"probe_umwp_within", "probe_squad2_within", "probe_umwp_to_squad2", "probe_squad2_to_umwp",
            "probe_transfer_mean", "shuffled_transfer_mean"]
    md = ["| Layer | Within UMWP | Within SQuAD | UMWP → SQuAD | SQuAD → UMWP | Mean transfer | Shuffled-label mean transfer |",
          "|---|---|---|---|---|---|---|"]
    for _, r in curve.iterrows():
        mark = " (chosen)" if r["layer"] == chosen else " (pilot's layer)" if r["layer"] == FIXED_LAYER else ""
        md.append(f"| {int(r['layer'])}{mark} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    for name, b in baselines.items():
        md.append(f"| {name} (any layer) | {b['umwp_within']:.3f} | {b['squad2_within']:.3f} | "
                  f"{b['umwp_to_squad2']:.3f} | {b['squad2_to_umwp']:.3f} | {b['transfer_mean']:.3f} | |")
    return md


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--acts", type=Path, default=ACTS_DEFAULT, help="the Colab v2 activations folder")
    parser.add_argument("--validation-only", action="store_true", help="stop before the test split")
    parser.add_argument("--overwrite-test", action="store_true", help="score the test split again (recorded)")
    parser.add_argument("--dry-run", action="store_true", help="write to data/dry_run/probe_v2 instead")
    args = parser.parse_args()
    ac.require_equivalence_check()

    base = ac.ROOT / "data" / "dry_run" / "probe_v2" if args.dry_run else ac.ROOT
    out, models = base / "results" / "probe_v2", base / "models" / "probe_v2"
    earlier = out / "test_results.csv"
    if earlier.exists() and not args.validation_only and not args.overwrite_test:
        raise SystemExit(f"{earlier.relative_to(ac.ROOT)} exists: the test split was already scored. "
                         "Use --validation-only, or --overwrite-test (recorded in settings.json).")
    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    started = time.time()

    df, acts_path, meta = load_and_check(args.acts, args.dry_run)
    y = df["should_abstain"].astype(int).to_numpy()
    groups = df["original_id"].to_numpy()
    masks = {(d, s): ((df["dataset"] == d) & (df["split"] == s)).to_numpy()
             for d in DATASETS for s in ("train", "validation", "test")}
    for s in ("train", "validation", "test"):
        masks[("pooled", s)] = (df["split"] == s).to_numpy()
    rng_shuffle = np.random.default_rng(ac.SEED)
    shuffled = {d: rng_shuffle.permutation(y[masks[(d, "train")]]) for d in (*DATASETS, "pooled")}
    text_features = ac.features(df, np.empty((len(df), 0)))

    # ---- stage 1: the layer curve, validation only
    print(f"\nStage 1: layer curve, layers 1-{N_LAYERS}, validation only ({N_JOBS} workers)")
    curve = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(layer_row)(acts_path, k, y, masks, shuffled) for k in range(1, N_LAYERS + 1))
    curve = pd.DataFrame(curve).sort_values("layer").reset_index(drop=True)
    baselines = {m: transfer_scores(m, text_features[m], y, masks, shuffled) for m in ("length", "tfidf")}
    best = float(curve["probe_transfer_mean"].max())
    near = curve.loc[curve["probe_transfer_mean"] >= best - LAYER_TOLERANCE, "layer"].astype(int).tolist()
    chosen_layer = min(near)
    curve.to_csv(out / "layer_curve.csv", index=False)
    print(f"  best mean transfer AUROC {best:.4f} at layer {int(curve.loc[curve['probe_transfer_mean'].idxmax(), 'layer'])}; "
          f"within {LAYER_TOLERANCE}: layers {near}; chosen: layer {chosen_layer}")
    for name, b in baselines.items():
        print(f"  {name}: mean transfer AUROC {b['transfer_mean']:.4f}")
    print(f"  convergence warnings across the curve: {int(curve['convergence_warnings'].sum())}")

    # ---- stage 2: final settings, validation only
    layers = sorted({chosen_layer, FIXED_LAYER})
    print(f"\nStage 2: final probes at layer(s) {layers}, pooled train, choices on pooled validation")
    X, method_layer = dict(text_features), {"length": None, "tfidf": None}
    for k in layers:
        acts_k = np.asarray(np.load(acts_path, mmap_mode="r")[:, k, :], dtype=np.float32)
        X[f"probe_L{k}"] = X[f"shuffled_L{k}"] = acts_k
        method_layer[f"probe_L{k}"] = method_layer[f"shuffled_L{k}"] = k
    methods = [f"probe_L{k}" for k in layers] + ["length", "tfidf"] + [f"shuffled_L{k}" for k in layers]
    kind = lambda m: m.split("_L")[0]       # probe_L18 -> probe

    chosen, fitted, transfer, transfer_pipes, selection_rows = {}, {}, {}, {}, []
    tr, va = masks[("pooled", "train")], masks[("pooled", "validation")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for m in methods:
            y_fit = shuffled["pooled"] if kind(m) == "shuffled" else y[tr]
            auc, C, pipe, rows = choose_C(kind(m), X[m], y_fit, tr, y[va], va)
            threshold, bal = ac.best_threshold(y[va], pipe.decision_function(X[m][va]))
            chosen[m] = {"layer": method_layer[m], "C": C, "threshold": threshold,
                         "validation_auroc": auc, "validation_balanced_accuracy": bal}
            fitted[m] = pipe
            selection_rows += [{"method": m, "trained_on": "pooled", **r} for r in rows]
            joblib.dump(pipe, models / f"{m}.joblib")
            print(f"  {m:<12} C={C:<7g} validation AUROC {auc:.4f}  threshold {threshold:+.3f}  "
                  f"balanced accuracy {bal:.4f}")
            for d in DATASETS:          # for the transfer check: trained on one dataset only
                dtr, dva = masks[(d, "train")], masks[(d, "validation")]
                y_fit = shuffled[d] if kind(m) == "shuffled" else y[dtr]
                auc_d, C_d, pipe_d, rows_d = choose_C(kind(m), X[m], y_fit, dtr, y[dva], dva)
                thr_d, bal_d = ac.best_threshold(y[dva], pipe_d.decision_function(X[m][dva]))
                transfer.setdefault(m, {})[d] = {"C": C_d, "threshold": thr_d, "validation_auroc": auc_d,
                                                 "validation_balanced_accuracy": bal_d}
                transfer_pipes[(m, d)] = pipe_d
                selection_rows += [{"method": m, "trained_on": d, **r} for r in rows_d]
    stage2_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    pd.DataFrame(selection_rows).to_csv(out / "validation_selection.csv", index=False)

    settings = {
        "created_by": "train_probe_v2.py", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dry_run": args.dry_run, "fixed_before_test": True, "seed": ac.SEED, "C_grid": ac.C_GRID,
        "layer_choice": {
            "rule": "for each layer 1-36: train on one dataset's train split, C by its validation AUROC, "
                    "score the other dataset's validation split; layer score = mean of the two transfer AUROCs; "
                    f"best layer, or the earliest within {LAYER_TOLERANCE} of it (PLAN.md Part D, fixed 2026-11-04)",
            "best_mean_transfer_auroc": best, "layers_within_tolerance": near, "chosen_layer": chosen_layer,
            "always_reported_layer": FIXED_LAYER, "curve": "layer_curve.csv",
            "convergence_warnings": int(curve["convergence_warnings"].sum()),
        },
        "chosen": chosen,
        "transfer": transfer,
        "selection_rule": "C by validation AUROC (ties -> smaller C); threshold by validation balanced accuracy",
        "prediction_rule": "predict should_abstain when decision score >= threshold",
        "features": {"probe_L<k>": "2,048 activations at layer k (last prompt token), StandardScaler fit on train",
                     "length": "word count, character count, StandardScaler fit on train",
                     "tfidf": {**ac.TFIDF_SETTINGS, "ngram_range": list(ac.TFIDF_SETTINGS["ngram_range"]),
                               "note": "vocabulary and idf fit on train"},
                     "shuffled_L<k>": "as probe_L<k>, trained on training labels permuted with the seed"},
        "model": "LogisticRegression, L2 (l1_ratio=0), lbfgs, max_iter=5000",
        "scores": "decision function (log-odds); not evaluated as calibrated probabilities",
        "convergence_warnings_stage2": stage2_warnings,
        "bootstrap": {"resamples": ac.N_BOOTSTRAP, "unit": "UMWP pair or SQuAD passage (original_id)", "seed": ac.SEED},
        "data": {"activations": str(args.acts), "activations_sha256": ac.sha256(acts_path),
                 "activations_created_at": meta.get("created_at"), "gpu": meta.get("gpu"),
                 "model": meta["model"], "revision": meta["revision"], "dtype": meta.get("dtype"),
                 "chat_template_sha256": meta.get("chat_template_sha256"),
                 "inputs_sha256": ac.sha256(V2 / "model_inputs.jsonl"), "labels_sha256": ac.sha256(V2 / "labels.jsonl"),
                 "rows": {f"{d}_{s}": int(masks[(d, s)].sum()) for d in (*DATASETS, "pooled")
                          for s in ("train", "validation", "test")}},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"  Settings fixed and saved to {out / 'settings.json'}")

    md = ["# Probe v2: layer choice and results", "",
          "Each probe reads Aya's internal state at the last prompt token and scores how likely the question "
          "can't be answered from what it gives. AUROC is the chance that a randomly picked unanswerable "
          "question scores higher than a randomly picked answerable one (0.5 = guessing, 1.0 = perfect).", "",
          "## Layer choice (validation splits only)", "",
          f"Rule fixed on 2026-11-04: train on one dataset, test on the other, average both directions; best "
          f"layer, or the earliest within {LAYER_TOLERANCE} of it. Best mean transfer AUROC {best:.3f}; layers "
          f"within {LAYER_TOLERANCE}: {near}. **Chosen: layer {chosen_layer}.** Layer {FIXED_LAYER} (the pilot's) "
          "is reported alongside everywhere.", ""]
    md += curve_markdown(curve, baselines, chosen_layer)
    md += ["", "\"Within\" = trained and scored on the same dataset (validation). The shuffled-label column "
           "should stay near 0.5 at every layer; if it doesn't, something leaks."]

    if args.validation_only:
        (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nStopped before the test split (--validation-only). {time.time() - started:.0f} s")
        return

    # ---- stage 3: the test split, once
    print("\nStage 3: single evaluation on the test split")
    rng = np.random.default_rng(ac.SEED)
    predictions = df[["example_id", "dataset", "split", "original_id", "should_abstain"]].copy()
    results = []
    for m in methods:
        scores, thr = fitted[m].decision_function(X[m]), chosen[m]["threshold"]
        predictions[f"{m}_score"] = scores
        predictions[f"{m}_predicts_abstain"] = scores >= thr
        for scope in ("pooled", *DATASETS):
            te = masks[(scope, "test")]
            results.append({"method": m, "layer": method_layer[m], "trained_on": "pooled", "tested_on": scope,
                            "threshold_from": "pooled validation", **scored(y[te], scores[te], thr, groups[te], rng)})
        for a, b in (DATASETS, DATASETS[::-1]):
            te = masks[(b, "test")]
            s = transfer_pipes[(m, a)].decision_function(X[m][te])
            results.append({"method": m, "layer": method_layer[m], "trained_on": a, "tested_on": b,
                            "threshold_from": f"{a} validation",
                            **scored(y[te], s, transfer[m][a]["threshold"], groups[te], rng)})
    predictions.to_csv(out / "predictions.csv", index=False)
    table = pd.DataFrame(results)
    table.to_csv(out / "test_results.csv", index=False)
    if args.overwrite_test:
        settings["test_rescored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def section(title, rows, note):
        lines = ["", f"## {title}", "", "| Method | Layer | Test AUROC (95% CI) | Test balanced accuracy (95% CI) | TN | FP | FN | TP |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['method'].split('_L')[0]} | {r['layer'] if r['layer'] is not None else '-'} | "
                         f"{fmt(r)} | {r['tn']} | {r['fp']} | {r['fn']} | {r['tp']} |")
        return lines + ["", note]

    md += section("Test split, pooled (trained on UMWP + SQuAD)", [r for r in results if r["tested_on"] == "pooled"],
                  f"{int(masks[('pooled', 'test')].sum())} test questions, "
                  f"{int(y[masks[('pooled', 'test')]].sum())} of them unanswerable.")
    for d in DATASETS:
        md += section(f"Test split, {NAMES[d]} only (same pooled probe and threshold)",
                      [r for r in results if r["tested_on"] == d and r["trained_on"] == "pooled"],
                      f"{int(masks[(d, 'test')].sum())} questions.")
    for a, b in (DATASETS, DATASETS[::-1]):
        md += section(f"Transfer: trained on {NAMES[a]} only, tested on {NAMES[b]}",
                      [r for r in results if r["trained_on"] == a],
                      f"Threshold from {NAMES[a]} validation, so balanced accuracy shows whether the threshold "
                      "carries over (in the pilot it didn't).")
    md += ["", "TN/FP/FN/TP: positive = unanswerable. FP = answerable question flagged; FN = unanswerable question "
           "missed. 95% intervals resample whole UMWP pairs or SQuAD passages, 1,000 times."]
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[md.index("## Test split, pooled (trained on UMWP + SQuAD)") - 1:]))
    print(f"\nSaved pipelines to {models} and results to {out}. {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()

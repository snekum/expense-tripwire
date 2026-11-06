"""How well can a probe trained on scrambled labels transfer between datasets, by luck alone?

    uv run python shuffle_spread_v2.py                 # the spread on validation (transfer check)
    uv run python shuffle_spread_v2.py --save-pooled   # 100 pooled scrambled probes per layer, for score_expense.py

Added on 2026-11-05, after train_probe_v2.py's single shuffled control at layer 23 transferred at
0.69 and 0.35 on the test split, far from 0.5. One scrambled run is one draw; this makes 100.

For each seed and each of layers 18 and 23 (the two v2 layers), exactly as train_probe_v2.py's
transfer check does it: scramble one dataset's training labels with the seed, choose C on that
dataset's validation split (true labels), and score the other dataset's validation split. Both
directions. The real probe's transfer AUROC (from layer_curve.csv, same splits) is then compared
with this spread. Validation splits only: no test row is touched, and no setting is changed.

--save-pooled trains 100 scrambled probes per layer the way the final v2 probe was trained:
pooled UMWP + SQuAD train with scrambled labels, C by pooled validation AUROC (true labels). Each
probe is a straight line through the activations, so it's saved as one weight per activation
value plus an offset (models/probe_v2/scrambled_L<k>.npz). score_expense.py scores the expense
claims with them, to show how far from 0.5 luck alone can land there. Nothing is fit on expense data.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import aya_common as ac
import train_probe_v2 as tp

N_SEEDS = 100
LAYERS = (18, 23)
OUT = ac.ROOT / "results" / "probe_v2"


def one_seed(acts_path, layer, seed, y, masks):
    X = np.asarray(np.load(acts_path, mmap_mode="r")[:, layer, :], dtype=np.float32)
    rng = np.random.default_rng(seed)
    shuffled = {d: rng.permutation(y[masks[(d, "train")]]) for d in tp.DATASETS}
    r = tp.transfer_scores("shuffled", X, y, masks, shuffled)
    return {"layer": layer, "seed": seed, **r}


def pooled_seed(acts_path, layer, seed, y, train, val):
    """One scrambled probe trained like the final v2 probe, as weights on the raw activations."""
    X = np.asarray(np.load(acts_path, mmap_mode="r")[:, layer, :], dtype=np.float32)
    y_fit = np.random.default_rng(seed).permutation(y[train])
    auc, C, pipe, _ = tp.choose_C("shuffled", X, y_fit, train, y[val], val)
    scale, logreg = pipe.named_steps["scale"], pipe.named_steps["logreg"]
    w = logreg.coef_[0] / scale.scale_
    b = logreg.intercept_[0] - float((logreg.coef_[0] * scale.mean_ / scale.scale_).sum())
    check = np.abs(X[val][:50] @ w + b - pipe.decision_function(X[val][:50])).max()
    return w.astype(np.float64), b, C, auc, float(check)


def save_pooled() -> None:
    df, acts_path, _ = tp.load_and_check(tp.ACTS_DEFAULT, dry_run=False)
    y = df["should_abstain"].astype(int).to_numpy()
    train, val = (df["split"] == "train").to_numpy(), (df["split"] == "validation").to_numpy()
    out = ac.ROOT / "models" / "probe_v2"
    for k in LAYERS:
        started = time.time()
        res = Parallel(n_jobs=tp.N_JOBS)(delayed(pooled_seed)(acts_path, k, 2000 + s, y, train, val)
                                         for s in range(N_SEEDS))
        W, b, C, auc, check = (np.array(v) for v in zip(*res))
        assert check.max() < 1e-3, f"weights don't reproduce the pipeline (largest difference {check.max()})"
        np.savez(out / f"scrambled_L{k}.npz", weights=W, offsets=b, C=C, validation_auroc=auc,
                 seeds=np.arange(2000, 2000 + N_SEEDS))
        print(f"layer {k}: {N_SEEDS} pooled scrambled probes, validation AUROC median {np.median(auc):.3f} "
              f"(range {auc.min():.3f}-{auc.max():.3f}), {time.time() - started:.0f} s -> {out / f'scrambled_L{k}.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-pooled", action="store_true")
    args = parser.parse_args()
    ac.require_equivalence_check()
    if args.save_pooled:
        save_pooled()
        return
    started = time.time()
    df, acts_path, _ = tp.load_and_check(tp.ACTS_DEFAULT, dry_run=False)
    y = df["should_abstain"].astype(int).to_numpy()
    masks = {(d, s): ((df["dataset"] == d) & (df["split"] == s)).to_numpy()
             for d in tp.DATASETS for s in ("train", "validation")}
    curve = pd.read_csv(OUT / "layer_curve.csv").set_index("layer")

    print(f"\n{N_SEEDS} scrambles x layers {list(LAYERS)}, validation only ({tp.N_JOBS} workers)")
    rows = Parallel(n_jobs=tp.N_JOBS)(delayed(one_seed)(acts_path, k, 1000 + s, y, masks)
                                      for k in LAYERS for s in range(N_SEEDS))
    spread = pd.DataFrame(rows)
    spread.to_csv(OUT / "shuffle_spread.csv", index=False)

    directions = {"umwp_to_squad2": "UMWP → SQuAD", "squad2_to_umwp": "SQuAD → UMWP", "transfer_mean": "mean of both"}
    summary, md = {}, [
        "# Scrambled-label spread (validation only)", "",
        f"{N_SEEDS} probes per layer, each trained on scrambled labels, then scored on the other dataset's "
        "validation questions with the true labels. The real probe's number comes from layer_curve.csv "
        "(same splits, same procedure, true labels). \"Beaten by\" = share of scrambled probes scoring at "
        "least as high as the real one.", "",
        "| Layer | Direction | Real probe | Scrambled: median | middle 90% | highest | Beaten by |",
        "|---|---|---|---|---|---|---|"]
    for k in LAYERS:
        for col, name in directions.items():
            s = spread.loc[spread["layer"] == k, col].to_numpy()
            real = float(curve.loc[k, f"probe_{col}"])
            lo, med, hi = np.percentile(s, [5, 50, 95])
            beaten = float((s >= real).mean())
            summary[f"layer {k}, {name}"] = {"real": round(real, 4), "median": round(float(med), 4),
                                             "p5": round(float(lo), 4), "p95": round(float(hi), 4),
                                             "max": round(float(s.max()), 4), "share_at_or_above_real": beaten}
            md.append(f"| {k} | {name} | {real:.3f} | {med:.3f} | {lo:.3f}-{hi:.3f} | {s.max():.3f} | "
                      f"{beaten:.0%} |")
    (OUT / "shuffle_spread.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "shuffle_spread.json").write_text(json.dumps({
        "created_by": "shuffle_spread_v2.py", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": [1000, 1000 + N_SEEDS - 1], "layers": list(LAYERS), "splits": "train (scrambled) and validation only",
        "added_after": "train_probe_v2.py test results (2026-11-05); changes no setting", "summary": summary},
        indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"\n{time.time() - started:.0f} s")


if __name__ == "__main__":
    main()

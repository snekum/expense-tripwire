"""Part B: build the training data for the v2 probe from two AbstentionBench subsets, UMWP and
SQuAD 2.0, and check it for shortcuts before any activations are collected. CPU only, no model.

    uv run python prepare_v2_data.py                       # stray spaces and ".." cleaned (default)
    uv run python prepare_v2_data.py --keep-upstream-text  # text exactly as upstream

What it does, in order:
  1. Downloads, at pinned revisions with sha256 checks: AbstentionBench's loader code and index
     files, the UMWP data file (GitHub), and the SQuAD 2.0 validation file (Hugging Face).
  2. Rebuilds AbstentionBench's two subsets with the upstream logic (see "Differences" below):
       UMWP:      1,750 answerable problems picked by UMWP_indices_answerable.json, plus each one's
                  unanswerable twin (row i + 2600). 3,500 rows.
       SQuAD 2.0: the 3,500 validation rows picked by subsampling-indices.json, each shown as
                  "Respond to the question using only information given in the context.
                   Context: ... Question: ...".
  3. Keeps only the UMWP pairs whose unanswerable twin is missing information (owner's choice,
     2026-11-04; see UMWP_KEEP). Pairs with impossible numbers or no question are dropped whole.
  4. Caps each dataset at 2,000 rows, 1,000 per label, sampled with seed 42 (UMWP: 1,000 whole
     pairs; SQuAD: 1,000 of each label).
  5. Splits each dataset 60/20/20 with the pilot's grouped splitter (aya_common.assign_splits):
     a UMWP problem and its twin share a split; all questions on one SQuAD passage share a split.
  6. Checks: pairs/passages never cross splits, no question text in two splits, label balance,
     and no v2 row copies a diagnostic question or an expense claim.
  7. Length confound rule (fixed in PLAN.md before any scoring): fit the length baseline on train,
     pick C on validation, report validation AUROC. If it is above 0.75, also mark a
     length-matched subset: word-count deciles (edges from train), labels equalised in each decile.

Differences from upstream (AbstentionBench recipe/abstention_datasets/{umwp,squad}.py at commit
e29184174c69..., identical to data.py on the Hugging Face copy):
  - Revisions are pinned. Upstream fetches UMWP from GitHub `main` and SQuAD from Hugging Face
    `main`. Both files are unchanged since before AbstentionBench was released, so the data match.
  - Plain pandas instead of `datasets`/torch Dataset classes (datasets 5.x won't run the loader).
  - Text cleaning (default on; see clean_text). Upstream text has formatting quirks that track the
    label: in UMWP, ".." and a leading space appear in 15% and 11% of answerable problems and in
    no unanswerable ones; in SQuAD, a leading space starts 7% of unanswerable questions and 0.1%
    of answerable ones. Cleaning removes them. SQuAD passages are left exactly as upstream.
  - UMWP keeps 3 of its 5 unanswerable categories (step 3); AbstentionBench keeps all 5.
  - The 2,000-row cap, linking IDs and splits are ours; AbstentionBench has none of them.

Writes data/v2/{model_inputs.jsonl, labels.jsonl, preview.md, summary.json} and
results/probe_v2/data_summary.md.
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

import aya_common as ac

# ---------------------------------------------------------------- provenance
AB_REPO = "facebook/AbstentionBench"
AB_REVISION = "af06080e2cbfff73f02a7c33135e5567d6f7ce6a"
AB_GITHUB = "facebookresearch/AbstentionBench @ e29184174c69bc95139b8c33dcca09aacab9442a"
AB_FILES = {  # file -> sha256
    "AbstentionBench.py": "800b11a2a93f2145725b47a006fda852266b9fb95ae25800ccf927c34ce8f595",
    "data.py": "eaa0e16a3bc3a78f84bb34ff87d694664b1bdb0526bf8cedbee3bd61b9bd605d",
    "UMWP_indices_answerable.json": "9362b5d655e35c8f31bd0b066cfcf80b7100b93e4894784980721b4a2b7dcc47",
    "subsampling-indices.json": "238ce5eea5e4bf3675c75f6d7dc66487edae0cfc5487710f731c5a02d89b4759",
}
AB_LICENSE = "cc-by-nc-4.0"

# UMWP's data file last changed in commit f5eb429 (2023-10-17); 1b3ae0f is the newest commit.
UMWP_COMMIT = "1b3ae0fae01a6c6f178361c716bbb4f7d7d54c3c"
UMWP_URL = f"https://raw.githubusercontent.com/Yuki-Asuuna/UMWP/{UMWP_COMMIT}/data/StandardDataset.jsonl"
UMWP_SHA256 = "e8840e8383357238a08e9c5028e4758ceceb369e1db31c64678b0f851c9c9e73"
UMWP_LICENSE = "CC-BY-SA-4.0 (data, per the UMWP README; the repo has no LICENSE file)"
UMWP_HALF = 2600   # rows 0..2599 answerable; row i + 2600 is the unanswerable twin of row i
UMWP_CATEGORIES = {1: "Key information missing", 2: "Ambiguous key information",   # upstream CATEGORY_MAP
                   3: "Unrealistic conditions", 4: "Unrelated object", 5: "Question missing"}
# Owner's choice (2026-11-04): keep pairs whose twin lacks information the question needs.
# Dropped: "Unrealistic conditions" (impossible numbers like -5 shoes: nothing is missing, something
# is visibly wrong, which expense variant 2 checks the probe does NOT fire on) and "Question
# missing" (no question at all; nothing like an expense claim, and ~10 words shorter than its twin).
UMWP_KEEP = ("Key information missing", "Ambiguous key information", "Unrelated object")

# The parquet file is unchanged since the 2024-01-04 conversion commit; 3ffb306 only edited the card.
SQUAD_REPO = "rajpurkar/squad_v2"
SQUAD_REVISION = "3ffb306f725f7d2ce8394bc1873b24868140c412"
SQUAD_FILE = "squad_v2/validation-00000-of-00001.parquet"
SQUAD_SHA256 = "0560174ab095c5ac0a8c8dc8da05f1625453c45a77e4ce9cabc6947ddfdd24cb"
SQUAD_LICENSE = "cc-by-sa-4.0"
SQUAD_PREPROMPT = "Respond to the question using only information given in the context."   # verbatim
SQUAD_TEMPLATE = "{preprompt}\nContext: {context}\nQuestion: {question}"                      # verbatim

# ---------------------------------------------------------------- fixed rules (PLAN.md, Part B)
CAP_PER_LABEL = 1000
LENGTH_RULE_AUROC = 0.75
N_DECILES = 10
MAX_COPY_OVERLAP = 0.5
DATASETS = ("umwp", "squad2")

RAW = ac.ROOT / "data" / "raw"
OUT = ac.ROOT / "data" / "v2"
RESULTS = ac.ROOT / "results" / "probe_v2"


# ---------------------------------------------------------------- downloads
def checked(path: Path, expected: str) -> Path:
    got = ac.sha256(path)
    if got != expected:
        raise SystemExit(f"Checksum mismatch for {path}: got {got}, expected {expected}")
    return path


def download_abstentionbench() -> dict[str, Path]:
    paths = {f: checked(Path(hf_hub_download(AB_REPO, f, repo_type="dataset", revision=AB_REVISION,
                                             local_dir=RAW / "abstentionbench")), sha)
             for f, sha in AB_FILES.items()}
    entry = paths["AbstentionBench.py"].read_text(encoding="utf-8")
    for line in ('"squad2": Squad2Dataset()', '"umwp": UMWP('):
        if line not in entry:
            raise SystemExit(f"AbstentionBench.py no longer lists {line}")
    card = hf_hub_download(AB_REPO, "README.md", repo_type="dataset", revision=AB_REVISION,
                           local_dir=RAW / "abstentionbench")
    if f"license: {AB_LICENSE}" not in Path(card).read_text(encoding="utf-8"):
        raise SystemExit("AbstentionBench licence is not the one recorded here")
    return paths


def download_umwp() -> Path:
    path = RAW / "umwp" / "StandardDataset.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UMWP_URL, path)
    return checked(path, UMWP_SHA256)


def download_squad() -> Path:
    return checked(Path(hf_hub_download(SQUAD_REPO, SQUAD_FILE, repo_type="dataset", revision=SQUAD_REVISION,
                                        local_dir=RAW / "squad_v2")), SQUAD_SHA256)


# ---------------------------------------------------------------- upstream logic
def clean_text(text: str) -> str:
    """Remove formatting quirks that track the label (see the module docstring): runs of periods
    ("left..How"), a missing space after a sentence ("shelves.How"), and stray or doubled spaces.
    The same rule runs on every row of both labels, so it can't add a label cue; it only removes
    ones that are there."""
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"(?<=[a-z0-9)%])([.?!])(?=[A-Z])", r"\1 ", text)
    return " ".join(text.split())


def build_umwp(path: Path, indices_path: Path, clean: bool) -> pd.DataFrame:
    """UMWP._load_and_subset_data and UMWP.__getitem__, as plain rows."""
    data = [json.loads(line) for line in path.open(encoding="utf-8")]
    answerable = json.loads(indices_path.read_text(encoding="utf-8"))
    rows = []
    for i in answerable:
        for idx in (i, i + UMWP_HALF):       # upstream order differs (all answerable first); irrelevant here
            item = data[idx]
            should_abstain = not item["answerable"]
            if item["answer"] is None or should_abstain:
                reference = None
            elif isinstance(item["answer"], list):
                reference = [str(item["answer"][0])]
            else:
                reference = [str(item["answer"])]
            rows.append({
                "example_id": f"umwp-{item['id']:04d}",
                "dataset": "umwp",
                "original_id": f"umwp-pair-{data[i]['id']:04d}",
                "question_upstream": item["question"],
                "question": clean_text(item["question"]) if clean else item["question"],
                "should_abstain": should_abstain,
                "reference_answers": reference,
                "category": UMWP_CATEGORIES.get(item["category"]),
                "source": item["source"],
                "upstream_id": item["id"],
            })
    df = pd.DataFrame(rows)
    # Our pair link rests on upstream's claim that row i + 2600 is row i's twin. UMWP's own
    # relevant_ids field says the same; check it.
    twins = df[df["should_abstain"]]
    relevant = {r["id"]: r["relevant_ids"] for r in data}
    linked = twins["upstream_id"].map(relevant)
    expected = twins["original_id"].str.removeprefix("umwp-pair-").astype(int).map(lambda x: [x])
    assert (linked.tolist() == expected.tolist()), "UMWP twin link disagrees with relevant_ids"
    return df


def build_squad(path: Path, subsampling_path: Path, clean: bool) -> pd.DataFrame:
    """Squad2Dataset.__getitem__ on the rows AbstentionBench subsamples."""
    data = pd.read_parquet(path)
    indices = json.loads(subsampling_path.read_text(encoding="utf-8"))["Squad2Dataset"]
    rows = []
    for idx in indices:
        item = data.iloc[idx]
        texts = list(item["answers"]["text"])
        should_abstain = texts == []
        passage = hashlib.sha256(item["context"].encode("utf-8")).hexdigest()[:12]
        bare = clean_text(item["question"]) if clean else item["question"]
        rows.append({
            "example_id": f"squad2-{item['id']}",
            "dataset": "squad2",
            "original_id": f"squad2-passage-{passage}",
            "question_upstream": SQUAD_TEMPLATE.format(preprompt=SQUAD_PREPROMPT, context=item["context"],
                                                       question=item["question"]),
            "question": SQUAD_TEMPLATE.format(preprompt=SQUAD_PREPROMPT, context=item["context"], question=bare),
            "should_abstain": should_abstain,
            "reference_answers": None if should_abstain else sorted(set(texts)),  # upstream: list(set(...))
            "category": None,
            "source": item["title"],
            "upstream_id": item["id"],
            "bare_question": bare,
            "bare_question_upstream": item["question"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cap and split
def cap(df: pd.DataFrame) -> pd.DataFrame:
    """1,000 rows per label, seed 42, drawn before anything is scored. UMWP draws whole pairs."""
    rng = random.Random(ac.SEED)
    if df["dataset"].iat[0] == "umwp":
        pairs = rng.sample(sorted(df["original_id"].unique()), CAP_PER_LABEL)
        return df[df["original_id"].isin(pairs)].reset_index(drop=True)
    keep = []
    for label in (False, True):
        ids = sorted(df.loc[df["should_abstain"] == label, "example_id"])
        keep += rng.sample(ids, CAP_PER_LABEL)
    return df[df["example_id"].isin(keep)].reset_index(drop=True)


def verify(df: pd.DataFrame) -> dict[str, bool]:
    checks = {}
    checks["example IDs unique"] = df["example_id"].is_unique
    checks["1,000 rows per label in each dataset"] = bool(
        (df.groupby(["dataset", "should_abstain"]).size() == CAP_PER_LABEL).all())
    umwp = df[df["dataset"] == "umwp"]
    checks["every UMWP pair complete (one answerable, one not)"] = bool(
        (umwp.groupby("original_id")["should_abstain"].agg(lambda s: sorted(s) == [False, True])).all())
    checks["UMWP unanswerable rows only from the kept categories"] = bool(
        umwp.loc[umwp["should_abstain"], "category"].isin(UMWP_KEEP).all())
    checks["no pair or passage in two splits"] = bool((df.groupby("original_id")["split"].nunique() == 1).all())
    norm = df["question"].map(ac.normalise)
    per_split = {s: set(norm[df["split"] == s]) for s in ac.SPLIT_FRACTIONS}
    checks["no question text in two splits"] = not (
        (per_split["train"] & per_split["validation"]) or (per_split["train"] & per_split["test"])
        or (per_split["validation"] & per_split["test"]))
    if "bare_question" in df:
        sq = df[df["dataset"] == "squad2"]
        bare = sq["bare_question"].map(ac.normalise)
        by_split = {s: set(bare[sq["split"] == s]) for s in ac.SPLIT_FRACTIONS}
        checks["no SQuAD question (without its passage) in two splits"] = not (
            (by_split["train"] & by_split["validation"]) or (by_split["train"] & by_split["test"])
            or (by_split["validation"] & by_split["test"]))
    return checks


# ---------------------------------------------------------------- overlap with other data
def overlap_report(df: pd.DataFrame) -> dict:
    """Largest share of a diagnostic question's (or expense claim's) word triples found in any v2
    row. v2 is later scored frozen on both, so neither may be copied into its training data.
    Also counts UMWP problems that are GSM8K test questions the v1 probe trained on."""
    v2_tri = [ac.trigrams(q) for q in df["question"]]
    targets = {}
    diag = [json.loads(l) for l in (ac.PILOT / "data" / "diagnostic" / "items.jsonl").open(encoding="utf-8")]
    targets["diagnostic questions"] = [r["question"] for r in diag]
    expense = ac.ROOT / "data" / "expense" / "items.jsonl"
    targets["expense claims"] = [json.loads(l)["claim_text"] for l in expense.open(encoding="utf-8")]
    report = {}
    for name, texts in targets.items():
        worst = 0.0
        for t in texts:
            tri = ac.trigrams(t)
            if tri:
                worst = max(worst, max(len(tri & v) / len(tri) for v in v2_tri))
        report[name] = round(worst, 3)

    pilot = [json.loads(l) for l in (ac.PILOT / "data" / "prepared" / "model_inputs.jsonl").open(encoding="utf-8")]
    pilot_norm = {ac.normalise(r["question"]) for r in pilot}
    pilot_tri = [ac.trigrams(r["question"]) for r in pilot]
    gsm = df[(df["dataset"] == "umwp") & (df["source"] == "GSM8K") & ~df["should_abstain"]]
    exact = int(gsm["question"].map(ac.normalise).isin(pilot_norm).sum())
    near = 0
    for q in gsm["question"]:
        tri = ac.trigrams(q)
        if tri and max(len(tri & p) / len(tri) for p in pilot_tri) >= 0.8:
            near += 1
    report["umwp_gsm8k_answerable_rows"] = len(gsm)
    report["of_which_in_pilot_gsm8k_exact"] = exact
    report["of_which_in_pilot_gsm8k_80pct_trigrams"] = near
    return report


# ---------------------------------------------------------------- length confound
def length_baseline(df: pd.DataFrame) -> dict:
    """The pilot's length baseline (words and characters, logistic regression): fit on train for
    each C, keep the C with the best validation AUROC, report that AUROC."""
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    x_tr, x_va = ac.features(tr, None)["length"], ac.features(va, None)["length"]
    y_tr, y_va = tr["should_abstain"].to_numpy(int), va["should_abstain"].to_numpy(int)
    by_c = {}
    for C in ac.C_GRID:
        model = ac.make_pipeline("length", C).fit(x_tr, y_tr)
        by_c[C] = roc_auc_score(y_va, model.predict_proba(x_va)[:, 1])
    best = max(by_c, key=by_c.get)
    words = va["question"].str.split().str.len()
    raw = roc_auc_score(y_va, words)
    return {"validation_auroc": round(by_c[best], 4), "chosen_C": best,
            "validation_auroc_word_count_alone": round(max(raw, 1 - raw), 4),
            "rule_triggered": by_c[best] > LENGTH_RULE_AUROC}


def length_match(df: pd.DataFrame) -> pd.Series:
    """Word-count deciles (edges from train); within each split and decile, keep equal numbers of
    each label by dropping random rows of the larger label (seed 42)."""
    words = df["question"].str.split().str.len()
    edges = np.unique(np.quantile(words[df["split"] == "train"], np.linspace(0, 1, N_DECILES + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    decile = pd.cut(words, edges, labels=False, include_lowest=True)
    keep = pd.Series(False, index=df.index)
    rng = np.random.default_rng(ac.SEED)
    for (_, _), g in df.groupby([df["split"], decile]):
        pos, neg = g.index[g["should_abstain"]], g.index[~g["should_abstain"]]
        n = min(len(pos), len(neg))
        keep[rng.choice(pos, n, replace=False)] = True
        keep[rng.choice(neg, n, replace=False)] = True
    return keep


def length_matched_auroc(df: pd.DataFrame, keep: pd.Series) -> dict:
    sub = df[keep]
    out = {"rows_kept": {s: int((sub["split"] == s).sum()) for s in ac.SPLIT_FRACTIONS}}
    out.update({f"after_matching_{k}": v for k, v in length_baseline(sub).items()
                if k in ("validation_auroc", "validation_auroc_word_count_alone")})
    return out


# ---------------------------------------------------------------- reporting
CUES = {"double period": r"\.\.", "no space after a sentence": r"(?<=[a-z0-9)%])[.?!](?=[A-Z])",
        "leading space": r"^\s", "trailing space": r"\s$", "double space": r"  "}


def cue_rates(df: pd.DataFrame) -> dict:
    """Share of rows of each label showing each formatting quirk, before and after cleaning.
    UMWP: the whole problem. SQuAD: the question only (the passage is shared within a group)."""
    cols = {"umwp": ("question_upstream", "question"), "squad2": ("bare_question_upstream", "bare_question")}
    out = {}
    for name, (before, after) in cols.items():
        d = df[df["dataset"] == name]
        out[name] = {cue: {version: {f"should_abstain={lab}": round(float(g[col].str.contains(p).mean()), 3)
                                     for lab, g in d.groupby("should_abstain")}
                           for version, col in (("upstream", before), ("used", after))}
                     for cue, p in CUES.items()}
    return out


def describe(df: pd.DataFrame) -> dict:
    out = {}
    for name, d in df.groupby("dataset"):
        words = d["question"].str.split().str.len()
        out[name] = {
            "splits": {s: {"rows": int((d["split"] == s).sum()),
                           "should_abstain_true": int(((d["split"] == s) & d["should_abstain"]).sum()),
                           "groups": int(d.loc[d["split"] == s, "original_id"].nunique())}
                       for s in ac.SPLIT_FRACTIONS},
            "words_by_label": {f"should_abstain={lab}": {"mean": round(float(w.mean()), 1),
                                                         "median": float(w.median()),
                                                         "min": int(w.min()), "max": int(w.max())}
                               for lab, w in words.groupby(d["should_abstain"])},
        }
        if name == "umwp":
            out[name]["unanswerable_categories"] = d.loc[d["should_abstain"], "category"].value_counts().to_dict()
            out[name]["sources"] = d.loc[~d["should_abstain"], "source"].value_counts().to_dict()
        else:
            out[name]["articles"] = int(d["source"].nunique())
    return out


def write_preview(df: pd.DataFrame, path: Path) -> None:
    """Six UMWP pairs (training split only) and three SQuAD passages with one question of each label."""
    rng = random.Random(ac.SEED)
    tr = df[df["split"] == "train"]
    lines = ["# v2 data preview (training split only)", ""]
    umwp = tr[tr["dataset"] == "umwp"]
    lines += ["## UMWP: answerable problem and its unanswerable twin", ""]
    for oid in sorted(rng.sample(sorted(umwp["original_id"].unique()), 6)):
        pair = umwp[umwp["original_id"] == oid].set_index("should_abstain")
        twin = pair.loc[True]
        lines += [f"### {oid} ({twin['source']}; twin category: {twin['category']})", "",
                  f"**answerable** ({len(pair.loc[False, 'question'].split())} words)", "",
                  f"> {pair.loc[False, 'question']}", "",
                  f"**unanswerable** ({len(twin['question'].split())} words)", "", f"> {twin['question']}", ""]
    sq = tr[tr["dataset"] == "squad2"]
    both = sq.groupby("original_id")["should_abstain"].nunique()
    lines += ["## SQuAD 2.0: one passage, one answerable and one unanswerable question", ""]
    for oid in sorted(rng.sample(sorted(both[both == 2].index), 3)):
        g = sq[sq["original_id"] == oid]
        context = g["question"].iat[0].split("\nContext: ", 1)[1].split("\nQuestion: ", 1)[0]
        lines += [f"### {oid} ({g['source'].iat[0]})", "", f"> {context}", ""]
        for lab in (False, True):
            r = g[g["should_abstain"] == lab].iloc[0]
            ans = "; ".join(r["reference_answers"]) if r["reference_answers"] else "none"
            lines += [f"- {'unanswerable' if lab else 'answerable'}: *{r['bare_question']}* (answer: {ans})"]
        lines += [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(summary: dict, path: Path) -> None:
    L = ["# v2 probe data: summary for review", "",
         f"Formatting quirks cleaned: **{summary['text_cleaned']}** "
         f"(rows changed: {summary['rows_changed_by_cleaning']})", "",
         "| dataset | split | rows | unanswerable | groups |", "|---|---|---|---|---|"]
    for name, d in summary["datasets"].items():
        for s, v in d["splits"].items():
            L.append(f"| {name} | {s} | {v['rows']} | {v['should_abstain_true']} | {v['groups']} |")
    L += ["", "## Length", "", "| dataset | label | mean words | median | min | max |", "|---|---|---|---|---|---|"]
    for name, d in summary["datasets"].items():
        for lab, w in d["words_by_label"].items():
            L.append(f"| {name} | {lab} | {w['mean']} | {w['median']} | {w['min']} | {w['max']} |")
    L += ["", "| dataset | length baseline, validation AUROC | word count alone | above 0.75? |",
          "|---|---|---|---|"]
    for name, lb in summary["length_baseline"].items():
        L.append(f"| {name} | {lb['validation_auroc']} | {lb['validation_auroc_word_count_alone']} | "
                 f"{'yes' if lb['rule_triggered'] else 'no'} |")
    for name, lm in summary["length_matched"].items():
        L += ["", f"Length-matched subset for {name}: rows kept {lm['rows_kept']}; validation AUROC of the "
                  f"length baseline after matching {lm['after_matching_validation_auroc']}."]
    L += ["", "## Formatting cues (share of rows; SQuAD counts the question only)", "",
          "| dataset | cue | upstream: answerable | upstream: unanswerable | used: answerable | used: unanswerable |",
          "|---|---|---|---|---|---|"]
    for name, cues in summary["format_cues"].items():
        for cue, v in cues.items():
            L.append(f"| {name} | {cue} | {v['upstream']['should_abstain=False']} | {v['upstream']['should_abstain=True']}"
                     f" | {v['used']['should_abstain=False']} | {v['used']['should_abstain=True']} |")
    d = summary["datasets"]["umwp"]
    L += ["", "UMWP unanswerable categories: " + ", ".join(f"{k} {v}" for k, v in d["unanswerable_categories"].items()),
          "", "UMWP sources: " + ", ".join(f"{k} {v}" for k, v in d["sources"].items()), "",
          "## Overlap", ""]
    o = summary["overlap"]
    L += [f"- Largest share of a diagnostic question found in a v2 row: {o['diagnostic questions']}",
          f"- Largest share of an expense claim found in a v2 row: {o['expense claims']}",
          f"- UMWP answerable problems from GSM8K: {o['umwp_gsm8k_answerable_rows']}; of these, "
          f"{o['of_which_in_pilot_gsm8k_exact']} are word-for-word pilot (v1 training) questions and "
          f"{o['of_which_in_pilot_gsm8k_80pct_trigrams']} share at least 80% of their word triples with one.",
          "", "## Checks", ""]
    L += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in summary["checks"].items()]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ac.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--keep-upstream-text", action="store_true",
                        help="use text exactly as upstream, with its label-linked formatting quirks")
    args = parser.parse_args()
    clean = not args.keep_upstream_text
    ac.require_equivalence_check()
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    ab = download_abstentionbench()
    umwp = build_umwp(download_umwp(), ab["UMWP_indices_answerable.json"], clean)
    squad = build_squad(download_squad(), ab["subsampling-indices.json"], clean)
    upstream_sizes = {"umwp": len(umwp), "squad2": len(squad)}
    twin_category = umwp[umwp["should_abstain"]].set_index("original_id")["category"]
    umwp = umwp[umwp["original_id"].map(twin_category).isin(UMWP_KEEP)].reset_index(drop=True)
    after_filter = {"umwp_pairs_before": upstream_sizes["umwp"] // 2, "umwp_pairs_after": len(umwp) // 2,
                    "kept_categories": list(UMWP_KEEP)}

    parts = []
    for d in (umwp, squad):
        d = cap(d)
        d["split"] = ac.assign_splits(d)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    checks = verify(df)
    overlap = overlap_report(df)
    checks[f"no v2 row copies a diagnostic question (largest share {overlap['diagnostic questions']})"] = \
        overlap["diagnostic questions"] <= MAX_COPY_OVERLAP
    checks[f"no v2 row copies an expense claim (largest share {overlap['expense claims']})"] = \
        overlap["expense claims"] <= MAX_COPY_OVERLAP

    length, matched = {}, {}
    df["length_matched"] = False
    for name in DATASETS:
        sub = df[df["dataset"] == name]
        length[name] = length_baseline(sub)
        if length[name]["rule_triggered"]:
            keep = length_match(sub)
            df.loc[sub.index, "length_matched"] = keep
            matched[name] = length_matched_auroc(sub, keep)

    df["n_words"] = df["question"].str.split().str.len()
    df["n_chars"] = df["question"].str.len()
    cues = cue_rates(df)
    summary = {
        "created_by": "prepare_v2_data.py", "git_commit": git_commit(), "seed": ac.SEED,
        "text_cleaned": clean,
        "source": {
            "abstentionbench": {"hf_repo": AB_REPO, "hf_revision": AB_REVISION, "github": AB_GITHUB,
                                "files_sha256": AB_FILES, "license": AB_LICENSE},
            "umwp": {"url": UMWP_URL, "sha256": UMWP_SHA256, "license": UMWP_LICENSE},
            "squad2": {"hf_repo": SQUAD_REPO, "hf_revision": SQUAD_REVISION, "file": SQUAD_FILE,
                       "sha256": SQUAD_SHA256, "license": SQUAD_LICENSE},
        },
        "abstentionbench_subset_sizes": upstream_sizes,
        "umwp_category_filter": after_filter,
        "cap_per_label": CAP_PER_LABEL,
        "datasets": describe(df),
        "length_rule": f"length baseline validation AUROC > {LENGTH_RULE_AUROC} -> also build a length-matched set",
        "length_baseline": length,
        "length_matched": matched,
        "format_cues": cues,
        "rows_changed_by_cleaning": {n: int((d["question"] != d["question_upstream"]).sum())
                                     for n, d in df.groupby("dataset")},
        "overlap": overlap,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    inputs = df[["example_id", "dataset", "split", "question"]]
    label_cols = ["example_id", "dataset", "split", "original_id", "should_abstain", "reference_answers",
                  "category", "source", "upstream_id", "length_matched", "n_words", "n_chars"]
    inputs.to_json(OUT / "model_inputs.jsonl", orient="records", lines=True, force_ascii=False)
    df[label_cols].to_json(OUT / "labels.jsonl", orient="records", lines=True, force_ascii=False)
    write_preview(df, OUT / "preview.md")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(summary, RESULTS / "data_summary.md")

    print((RESULTS / "data_summary.md").read_text(encoding="utf-8"))
    print(f"Wrote {', '.join(p.name for p in sorted(OUT.iterdir()))} to {OUT.relative_to(ac.ROOT)}")
    if not summary["all_checks_passed"]:
        sys.exit("Some checks FAILED; see above.")


if __name__ == "__main__":
    main()

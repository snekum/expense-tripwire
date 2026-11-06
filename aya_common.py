"""Code shared by every script in this project, copied unchanged from the pilot project
(d:\\portfolio\\answerability-probe). Each block names its source file.

Copying (rather than importing across folders) keeps this project self-contained. To make sure
the copies haven't drifted, check_against_pilot.py reruns them against the pilot's saved outputs;
the other scripts refuse to run until that check has passed for the current version of this file.
"""

import ctypes
import hashlib
import json
import os
import platform
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).parent
PILOT = Path(os.environ.get("PILOT_DIR", r"d:\portfolio\answerability-probe"))
VENDOR = ROOT / "vendor" / "pilot_v1"
EQUIVALENCE_RECORD = ROOT / "results" / "equivalence_check.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_equivalence_check() -> None:
    """Stop unless check_against_pilot.py has passed for this exact version of aya_common.py."""
    if not EQUIVALENCE_RECORD.exists():
        raise SystemExit("Run `uv run python check_against_pilot.py` first.")
    record = json.loads(EQUIVALENCE_RECORD.read_text(encoding="utf-8"))
    if not record.get("all_passed") or record.get("aya_common_sha256") != sha256(Path(__file__)):
        raise SystemExit("aya_common.py changed (or the last check failed); rerun check_against_pilot.py.")


# ================================================================ from collect_activations.py
MODEL_ID = "CohereLabs/tiny-aya-global"
MODEL_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
DEFAULT_LAYER = 18          # the model has 36 layers; 18 is the middle
GPU_HEADROOM_GB = 0.6       # GPU memory left free for the forward pass itself


def build_prompt(tokenizer, question: str) -> str:
    """The question as a single user turn, ending where the model's answer would begin.
    The chat template adds Aya's default system preamble and the start-of-sequence token."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(layer: int, requested_device: str):
    """Load layers 1..layer only, then place them on the GPU if they fit, else the CPU.

    Returns tokenizer, model, device, compute dtype, and a sentence saying why."""
    full_config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if not 1 <= layer <= full_config.num_hidden_layers:
        raise SystemExit(f"--layer must be between 1 and {full_config.num_hidden_layers}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "right"   # real tokens keep positions 0..n-1; padding goes after

    # AutoModel has no next-word output layer, which we never use. The config overrides
    # make it build (and load weights for) only the first `layer` layers.
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
        num_hidden_layers=layer, layer_types=full_config.layer_types[:layer],
    )
    model.norm = torch.nn.Identity()   # keep the raw output of the last loaded layer
    model.eval()
    model.requires_grad_(False)        # frozen: only the probe will ever be trained

    layer_gb = sum(p.numel() * p.element_size() for p in model.layers.parameters()) / 1e9
    device, why = "cpu", "no usable GPU found"
    if requested_device == "cpu":
        why = "CPU requested on the command line"
    elif torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        name = torch.cuda.get_device_name(0)
        if layer_gb + GPU_HEADROOM_GB <= free_gb:
            device, why = "cuda", f"{layer} layers ({layer_gb:.1f} GB) fit on {name} ({free_gb:.1f} GB free)"
        else:
            why = f"{layer} layers ({layer_gb:.1f} GB) don't fit on {name} ({free_gb:.1f} GB free)"
    if requested_device == "cuda" and device != "cuda":
        raise SystemExit(f"CUDA requested but unusable: {why}")

    # bfloat16 is the model's own format. GPUs without native bfloat16 (e.g. GTX 16xx, T4)
    # use float16, which has the same size and more precision but a smaller range.
    dtype = torch.bfloat16
    if device == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        dtype = torch.float16
    model.layers.to(device, dtype)
    model.rotary_emb.to(device)        # move only; its frequency table stays in full precision
    return tokenizer, model, device, dtype, why


@torch.inference_mode()
def last_token_states(tokenizer, model, prompts: list[str], device: str, dtype) -> tuple[np.ndarray, list[int]]:
    """Activation at the last non-padding token of each prompt.

    With right padding the last real token is at index (number of real tokens - 1), and
    the causal mask means padding after it can't affect it."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    embeds = model.embed_tokens(enc["input_ids"]).to(device, dtype)   # lookup stays on the CPU
    mask = enc["attention_mask"].to(device)
    out = model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    last = mask.sum(dim=1) - 1
    states = out[torch.arange(len(prompts), device=device), last]
    return states.float().cpu().numpy(), (last + 1).tolist()


def padding_check(tokenizer, model, prompts: list[str], device: str, dtype) -> float:
    """Run the shortest prompt alone and batched next to the longest. The results should
    match apart from tiny rounding differences; a large gap would mean padding leaks in."""
    short, long = min(prompts, key=len), max(prompts, key=len)
    alone, _ = last_token_states(tokenizer, model, [short], device, dtype)
    batched, _ = last_token_states(tokenizer, model, [short, long], device, dtype)
    a, b = alone[0], batched[0]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def keep_awake() -> None:
    """On Windows, stop the PC from going to sleep on idle while this process runs.
    (Closing a laptop lid can still put it to sleep.)"""
    if platform.system() == "Windows":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def chat_template_sha256(tokenizer) -> str:
    """Same text and hash as the pilot saved next to its activations (collect_activations.main)."""
    template = tokenizer.chat_template
    text = template if isinstance(template, str) else json.dumps(template, indent=1, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ================================================================ from train_probe.py
SEED = 42
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]   # smaller C = stronger regularization
N_BOOTSTRAP = 1000
TFIDF_SETTINGS = {"lowercase": True, "ngram_range": (1, 2), "sublinear_tf": True}
METHODS = ["probe", "length", "tfidf", "shuffled"]


def logreg(C: float) -> LogisticRegression:
    # l1_ratio=0 is pure L2 regularization (scikit-learn >= 1.8 deprecates penalty="l2").
    return LogisticRegression(C=C, l1_ratio=0.0, max_iter=5000)


def make_pipeline(method: str, C: float) -> Pipeline:
    if method == "tfidf":
        return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)), ("logreg", logreg(C))])
    return Pipeline([("scale", StandardScaler()), ("logreg", logreg(C))])


def features(df: pd.DataFrame, acts: np.ndarray) -> dict:
    """Inputs for each method. IDs are never features."""
    length = np.c_[df["question"].str.split().str.len(), df["question"].str.len()].astype(float)
    return {"probe": acts, "shuffled": acts, "length": length, "tfidf": df["question"].to_numpy()}


def best_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing balanced accuracy; predict should_abstain when score >= threshold."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    balanced = (tpr + 1 - fpr) / 2
    i = int(np.argmax(balanced))
    return float(thresholds[i]), float(balanced[i])


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auroc": roc_auc_score(y, scores), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap(y, scores, threshold, groups, rng) -> dict:
    """95% intervals, resampling whole problems (both versions together), not single rows."""
    unique = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in unique}
    aurocs, bals = [], []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([rows_of[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        aurocs.append(roc_auc_score(y[idx], scores[idx]))
        bals.append(balanced_accuracy_score(y[idx], (scores[idx] >= threshold).astype(int)))
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]
    return {"auroc_95ci": q(aurocs), "balanced_accuracy_95ci": q(bals)}


# ================================================================ from prepare_data.py
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def normalise(text: str) -> str:
    """Text used for duplicate checks: case-folded, whitespace collapsed."""
    return " ".join(text.casefold().split())


def group_ids(df: pd.DataFrame) -> dict[str, str]:
    """Map each original_id to a group id.

    Both versions of a question start in the same group. If two different questions
    share any question text (e.g. both end in "How much did he pay?"), their groups
    are merged, so that text can't appear in two splits. Uses a small union-find.
    """
    parent = {oid: oid for oid in df["original_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, ids in df.groupby(df["question"].map(normalise))["original_id"]:
        first, *rest = ids.tolist()
        for other in rest:
            parent[find(other)] = find(first)
    return {oid: find(oid) for oid in parent}


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Shuffle the groups (seed 42), then fill train to 60% of rows, validation to 80%,
    and put the rest in test."""
    group_of = df["original_id"].map(group_ids(df))
    sizes = group_of.value_counts().sort_index()  # sort first so the shuffle is reproducible
    groups = sizes.index.tolist()
    random.Random(SEED).shuffle(groups)

    total, filled = len(df), 0
    cutoffs = [("train", SPLIT_FRACTIONS["train"]),
               ("validation", SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"]),
               ("test", 1.0)]
    split_of_group = {}
    for g in groups:
        split = next(name for name, cut in cutoffs if filled < cut * total)
        split_of_group[g] = split
        filled += sizes[g]
    return group_of.map(split_of_group)


# ================================================================ from build_diagnostic.py
def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.:'][a-z0-9]+)*", text.lower())


def trigrams(text: str) -> set:
    w = words_of(text)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}

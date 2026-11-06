"""The equivalence gate: proves the code and files copied from the pilot behave exactly like the
originals. Every other script in this project refuses to run until this has passed.

    uv run python check_against_pilot.py              # all checks (loads 18 layers on the GPU)
    uv run python check_against_pilot.py --no-model   # checks 1, 2, 3 and 5 only; does not unlock

Reads the pilot folder (d:\\portfolio\\answerability-probe, or $PILOT_DIR) without changing it:
  1. Every copied function's source text is identical to the pilot's.
  2. The files in vendor/pilot_v1/ have the same SHA-256 as the pilot's originals.
  3. The copied features + frozen v1 pipelines reproduce the pilot's saved scores and
     predictions for all 2,426 questions.
  4. The copied extraction code reproduces the pilot's saved activations: the 20 smoke-test
     questions and the 48 diagnostic questions (cosine >= 0.9999 on every row).
  5. The copied prompt builder gives the pilot's chat-template hash and final token.
It also checks that no pilot file changed while it ran. Writes results/equivalence_check.json.
"""

import argparse
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import aya_common as ac

COPIED = {"collect_activations": ["build_prompt", "load_model", "last_token_states", "padding_check", "keep_awake"],
          "train_probe": ["logreg", "make_pipeline", "features", "best_threshold", "metrics", "bootstrap"],
          "prepare_data": ["normalise", "group_ids", "assign_splits"],
          "build_diagnostic": ["words_of", "trigrams"]}
VENDORED = {"probe.joblib": "models/milestone4/probe.joblib", "length.joblib": "models/milestone4/length.joblib",
            "tfidf.joblib": "models/milestone4/tfidf.joblib", "shuffled.joblib": "models/milestone4/shuffled.joblib",
            "settings.json": "results/milestone4/settings.json",
            "activation_meta.json": "data/activations/all_layer18_cuda_float16/meta.json",
            "chat_template.jinja": "data/activations/all_layer18_cuda_float16/chat_template.jinja"}
REFERENCE_SETS = {  # name -> (pilot activation folder, where the question text lives)
    "smoke": ("data/activations/smoke_layer18_cuda_float16", "data/prepared/model_inputs.jsonl"),
    "diagnostic": ("data/diagnostic/activations_layer18_cuda_float16", "data/diagnostic/items.jsonl"),
}
MIN_COSINE = 0.9999


def pilot_module(name: str):
    spec = importlib.util.spec_from_file_location(f"pilot_{name}", ac.PILOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ac.PILOT))       # pilot modules import each other by name
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def pilot_fingerprints() -> dict:
    manifest = json.loads((ac.PILOT / "data" / "diagnostic" / "frozen_artifacts.json").read_text(encoding="utf-8"))
    return {p: ac.sha256(ac.PILOT / p) == h for p, h in manifest.items()}


def report(results: dict, name: str, ok: bool, detail: str) -> None:
    results[name] = {"passed": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-model", action="store_true", help="skip the GPU check (does not unlock other scripts)")
    args = parser.parse_args()
    print(f"Pilot folder: {ac.PILOT}")
    results = {}
    before = pilot_fingerprints()
    report(results, "pilot files unchanged before", all(before.values()), f"{sum(before.values())}/{len(before)} match")

    # 1. source text
    differing = []
    for module_name, names in COPIED.items():
        module = pilot_module(module_name)
        differing += [f"{module_name}.{n}" for n in names
                      if inspect.getsource(getattr(module, n)) != inspect.getsource(getattr(ac, n))]
    report(results, "copied functions identical", not differing,
           f"{sum(map(len, COPIED.values()))} functions" + (f"; differ: {differing}" if differing else ""))
    tp, ca = pilot_module("train_probe"), pilot_module("collect_activations")
    constants = {"MODEL_ID": ca.MODEL_ID == ac.MODEL_ID, "MODEL_REVISION": ca.MODEL_REVISION == ac.MODEL_REVISION,
                 "C_GRID": tp.C_GRID == ac.C_GRID, "TFIDF_SETTINGS": tp.TFIDF_SETTINGS == ac.TFIDF_SETTINGS,
                 "SEED": tp.SEED == ac.SEED, "N_BOOTSTRAP": tp.N_BOOTSTRAP == ac.N_BOOTSTRAP}
    report(results, "copied constants identical", all(constants.values()),
           ", ".join(k for k, v in constants.items() if not v) or f"{len(constants)} constants")

    # 2. vendored files
    bad = [v for v, p in VENDORED.items() if ac.sha256(ac.VENDOR / v) != ac.sha256(ac.PILOT / p)]
    report(results, "vendor files identical to pilot", not bad, f"{len(VENDORED)} files" + (f"; differ: {bad}" if bad else ""))

    # 3. scoring with the vendored pipelines and copied features
    act_dir = ac.PILOT / "data" / "activations" / "all_layer18_cuda_float16"
    acts = np.load(act_dir / "activations.npy")
    ids = json.loads((act_dir / "example_ids.json").read_text(encoding="utf-8"))
    inputs = pd.read_json(ac.PILOT / "data" / "prepared" / "model_inputs.jsonl", lines=True).set_index("example_id").loc[ids].reset_index()
    saved = pd.read_csv(ac.PILOT / "results" / "milestone4" / "predictions.csv").set_index("example_id").loc[ids]
    settings = json.loads((ac.VENDOR / "settings.json").read_text(encoding="utf-8"))
    X = ac.features(inputs, acts)
    for m in ac.METHODS:
        s = joblib.load(ac.VENDOR / f"{m}.joblib").decision_function(X[m])
        close = np.allclose(s, saved[f"{m}_score"], rtol=1e-5, atol=1e-5)
        same = bool(((s >= settings["chosen"][m]["threshold"]) == saved[f"{m}_predicts_abstain"]).all())
        report(results, f"v1 {m} reproduces pilot scores", close and same,
               f"largest difference {np.max(np.abs(s - saved[f'{m}_score'])):.1e}, predictions "
               f"{'identical' if same else 'DIFFERENT'} on {len(s)} questions")

    # 5 (tokenizer only) and 4 (model)
    pilot_meta = json.loads((ac.VENDOR / "activation_meta.json").read_text(encoding="utf-8"))
    if args.no_model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ac.MODEL_ID, revision=ac.MODEL_REVISION)
    else:
        tokenizer, model, device, dtype, why = ac.load_model(pilot_meta["layer"], "auto")
        print(f"  Model: {device}, {str(dtype).removeprefix('torch.')} ({why})")
    sample = ac.build_prompt(tokenizer, inputs["question"].iloc[0])
    final = tokenizer.convert_ids_to_tokens(tokenizer(sample, add_special_tokens=False)["input_ids"][-1])
    report(results, "same chat template and final token",
           ac.chat_template_sha256(tokenizer) == pilot_meta["prompt_template"]["chat_template_sha256"]
           and [final] == pilot_meta["final_tokens"], f"final token {final}")

    if not args.no_model:
        setup = (device, str(dtype).removeprefix("torch."))
        report(results, "same device and number format as pilot", setup == (pilot_meta["device"], pilot_meta["dtype"].split()[0]),
               f"{setup[0]}, {setup[1]}")
        for name, (folder, text_file) in REFERENCE_SETS.items():
            ref_dir = ac.PILOT / folder
            ref = np.load(ref_dir / "activations.npy")
            ref_ids = json.loads((ref_dir / "example_ids.json").read_text(encoding="utf-8"))
            ref_meta = json.loads((ref_dir / "meta.json").read_text(encoding="utf-8"))
            text = pd.read_json(ac.PILOT / text_file, lines=True).set_index("example_id")["question"]
            prompts = [ac.build_prompt(tokenizer, text[i]) for i in ref_ids]
            batch = ref_meta["batch_size"]
            mine = np.concatenate([ac.last_token_states(tokenizer, model, prompts[b:b + batch], device, dtype)[0]
                                   for b in range(0, len(prompts), batch)])
            cos = (mine * ref).sum(1) / (np.linalg.norm(mine, axis=1) * np.linalg.norm(ref, axis=1))
            report(results, f"reproduces pilot {name} activations", bool(cos.min() >= MIN_COSINE),
                   f"{len(ref_ids)} rows, lowest cosine {cos.min():.6f}, largest difference {np.abs(mine - ref).max():.2e}")

    after = pilot_fingerprints()
    report(results, "pilot files unchanged after", all(after.values()), f"{sum(after.values())}/{len(after)} match")
    all_passed = all(r["passed"] for r in results.values()) and not args.no_model
    ac.EQUIVALENCE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    ac.EQUIVALENCE_RECORD.write_text(json.dumps({
        "all_passed": all_passed, "model_checks_run": not args.no_model,
        "aya_common_sha256": ac.sha256(ac.ROOT / "aya_common.py"), "pilot_dir": str(ac.PILOT),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": results}, indent=2), encoding="utf-8")
    print(f"\n{'All checks passed; other scripts are unlocked.' if all_passed else 'Not unlocked.'} "
          f"Record: {ac.EQUIVALENCE_RECORD.relative_to(ac.ROOT)}")
    if not all(r["passed"] for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

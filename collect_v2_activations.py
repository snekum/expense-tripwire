"""Part C: save Aya's internal state at every layer, at the last prompt token, for each prompt in
a set. The v2 probe's layer is then chosen on validation data (PLAN.md, "Choosing the v2 layer").

    uv run python collect_v2_activations.py --set reference --layers 18 --same-machine  # laptop
    uv run python collect_v2_activations.py --set reference               # Colab: cross-machine check
    uv run python collect_v2_activations.py --set v2                      # Colab: 4,000 v2 prompts
    uv run python collect_v2_activations.py --set expense                 # Colab: 144 expense prompts
    uv run python collect_v2_activations.py --export-reference            # laptop: copy pilot references

Sets:
  reference  the pilot's 20 smoke and 48 diagnostic questions, whose layer-18 activations the pilot
             saved on the laptop. Their check must pass before the other sets are trusted.
  v2         data/v2/model_inputs.jsonl (UMWP and SQuAD 2.0).
  expense    data/expense/prompts.jsonl (48 claims x 3 wordings), exactly as Aya sees them.
             Refuses to run until the owner's review is complete (review_gate.py).

How it extracts: the model is loaded with the copied aya_common.load_model (same revision,
fp16 on GPUs without bf16, embeddings looked up on the CPU, final norm switched off), with as
many layers as --layers asks for (default all 36). One forward pass returns the output of every
layer. Row i of activations.npy has shape (layers + 1, 2048); entry k is the output of layer k,
and entry 0 is the token embedding. Because the final norm is switched off, the last entry is the
raw output of the last layer, the same as the pilot's code would give, and the script checks that.

Activations are computed in fp16 and saved as fp16, which loses nothing. Work is saved in chunks,
so a Colab disconnect resumes where it stopped.

The reference check:
  - Layer 18 must match the pilot's saved activations: cosine >= 0.9999 on the machine that made
    them (the laptop) and >= 0.999 elsewhere (PLAN.md, Part C).
  - Layers 17 and 19 must match worse than 18, so the layer numbering can't be off by one. (On the
    laptop only 18 layers fit, so layer 19 is checked on Colab.)
  - The last saved layer must equal the model's own final output exactly.
"""

import argparse
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

import aya_common as ac
import review_gate

CHECK_LAYER = 18
SAME_MACHINE_COSINE, CROSS_MACHINE_COSINE = 0.9999, 0.999
CHUNK_ROWS = 128
REFERENCE_DIR = ac.ROOT / "data" / "reference"
REFERENCE_SETS = {  # name -> (pilot activation folder, pilot file holding the question text)
    "smoke": ("data/activations/smoke_layer18_cuda_float16", "data/prepared/model_inputs.jsonl"),
    "diagnostic": ("data/diagnostic/activations_layer18_cuda_float16", "data/diagnostic/items.jsonl"),
}
INPUTS = {"v2": (ac.ROOT / "data" / "v2" / "model_inputs.jsonl", "example_id", "question"),
          "expense": (ac.ROOT / "data" / "expense" / "prompts.jsonl", "prompt_id", "user_message")}


# ---------------------------------------------------------------- extraction
@torch.inference_mode()
def all_layer_states(tokenizer, model, prompts: list[str], device: str, dtype) -> tuple[np.ndarray, list[int], bool]:
    """Like aya_common.last_token_states, but returns every layer's output at the last real token.

    Returns (states of shape (batch, layers + 1, hidden) in fp16, prompt lengths, and whether the
    last hidden state equals the model's final output exactly)."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    embeds = model.embed_tokens(enc["input_ids"]).to(device, dtype)   # lookup stays on the CPU
    mask = enc["attention_mask"].to(device)
    out = model(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)
    last = mask.sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    stacked = torch.stack([h[rows, last] for h in out.hidden_states], dim=1)
    final_matches = bool(torch.equal(out.hidden_states[-1][rows, last], out.last_hidden_state[rows, last]))
    return stacked.to(torch.float16).cpu().numpy(), (last + 1).tolist(), final_matches


def collect(tokenizer, model, device, dtype, ids: list[str], prompts: list[str], out: Path, batch: int) -> dict:
    """Run every prompt, saving chunk by chunk; finished chunks are skipped on a rerun."""
    chunks = out / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    lengths, final_ok, started = [], True, time.time()
    for c0 in range(0, len(prompts), CHUNK_ROWS):
        path = chunks / f"{c0:06d}.npz"
        chunk_ids = ids[c0:c0 + CHUNK_ROWS]
        if path.exists():
            saved = np.load(path)
            if list(saved["ids"]) == chunk_ids:
                lengths += saved["lengths"].tolist()
                continue
        states, lens = [], []
        for b0 in range(c0, min(c0 + CHUNK_ROWS, len(prompts)), batch):
            s, n, ok = all_layer_states(tokenizer, model, prompts[b0:b0 + batch], device, dtype)
            states.append(s)
            lens += n
            final_ok &= ok
        np.savez(path, states=np.concatenate(states), ids=np.array(chunk_ids), lengths=np.array(lens))
        lengths += lens
        done = min(c0 + CHUNK_ROWS, len(prompts))
        print(f"  {done}/{len(prompts)} prompts, {time.time() - started:.0f} s")
    acts = np.concatenate([np.load(chunks / f"{c0:06d}.npz")["states"] for c0 in range(0, len(prompts), CHUNK_ROWS)])
    np.save(out / "activations.npy", acts)
    (out / "example_ids.json").write_text(json.dumps(ids, indent=1), encoding="utf-8")
    shutil.rmtree(chunks)
    return {"prompt_tokens": {"min": int(min(lengths)), "max": int(max(lengths)), "mean": round(float(np.mean(lengths)), 1)},
            "seconds": round(time.time() - started, 1), "last_layer_equals_final_output": final_ok}


# ---------------------------------------------------------------- inputs
def load_set(name: str) -> tuple[list[str], list[str], str]:
    """Returns ids, raw texts (before the chat template), and the sha256 of the source file(s)."""
    if name == "reference":
        ids, texts, shas = [], [], []
        for ref in REFERENCE_SETS:
            folder = reference_folder(ref)
            ref_ids = json.loads((folder / "example_ids.json").read_text(encoding="utf-8"))
            text = pd.read_json(folder / "texts.jsonl", lines=True).set_index("example_id")["question"]
            ids += [f"{ref}:{i}" for i in ref_ids]
            texts += [text[i] for i in ref_ids]
            shas.append(ac.sha256(folder / "texts.jsonl"))
        return ids, texts, ",".join(shas)
    path, id_col, text_col = INPUTS[name]
    df = pd.read_json(path, lines=True)
    return df[id_col].tolist(), df[text_col].tolist(), ac.sha256(path)


def reference_folder(name: str) -> Path:
    folder = REFERENCE_DIR / name
    if not (folder / "activations.npy").exists():
        raise SystemExit(f"{folder} is missing; run --export-reference on the laptop first.")
    return folder


def export_reference() -> None:
    """Copy the pilot's reference activations, their IDs and question texts into data/reference,
    so the check can run on Colab, where the pilot folder doesn't exist."""
    record = {}
    for name, (folder, text_file) in REFERENCE_SETS.items():
        src, dst = ac.PILOT / folder, REFERENCE_DIR / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("activations.npy", "example_ids.json", "meta.json"):
            shutil.copyfile(src / f, dst / f)
            record[f"{name}/{f}"] = ac.sha256(dst / f)
        ids = json.loads((src / "example_ids.json").read_text(encoding="utf-8"))
        text = pd.read_json(ac.PILOT / text_file, lines=True).set_index("example_id")["question"]
        pd.DataFrame({"example_id": ids, "question": [text[i] for i in ids]}).to_json(
            dst / "texts.jsonl", orient="records", lines=True, force_ascii=False)
        record[f"{name}/texts.jsonl"] = ac.sha256(dst / "texts.jsonl")
    (REFERENCE_DIR / "SHA256SUMS.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Copied {len(record)} files to {REFERENCE_DIR.relative_to(ac.ROOT)}")


# ---------------------------------------------------------------- the reference check
def reference_check(acts: np.ndarray, ids: list[str], same_machine: bool) -> dict:
    def cosine(a, b):
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

    need = SAME_MACHINE_COSINE if same_machine else CROSS_MACHINE_COSINE
    sums = json.loads((REFERENCE_DIR / "SHA256SUMS.json").read_text(encoding="utf-8"))
    results = {"reference files unchanged": all(ac.sha256(REFERENCE_DIR / p) == h for p, h in sums.items())}
    detail = {}
    for name in REFERENCE_SETS:
        folder = reference_folder(name)
        ref = np.load(folder / "activations.npy")
        rows = [k for k, i in enumerate(ids) if i.startswith(f"{name}:")]
        mine = acts[rows].astype(np.float32)
        by_layer = {layer: cosine(mine[:, layer], ref) for layer in (CHECK_LAYER - 1, CHECK_LAYER, CHECK_LAYER + 1)
                    if layer < mine.shape[1]}
        low = float(by_layer[CHECK_LAYER].min())
        neighbours = {l: float(c.mean()) for l, c in by_layer.items() if l != CHECK_LAYER}
        detail[name] = {"rows": len(rows), "lowest_cosine_layer18": round(low, 6),
                        "largest_difference_layer18": float(np.abs(mine[:, CHECK_LAYER] - ref).max()),
                        "mean_cosine_neighbour_layers": {l: round(v, 4) for l, v in neighbours.items()}}
        results[f"{name}: layer 18 matches the pilot (lowest cosine {low:.6f} >= {need})"] = low >= need
        results[f"{name}: neighbour layers {sorted(neighbours)} match worse than 18"] = all(
            v < float(by_layer[CHECK_LAYER].mean()) for v in neighbours.values())
    return {"checks": results, "detail": detail, "threshold": need}


# ---------------------------------------------------------------- main
def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ac.ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["reference", "v2", "expense"])
    parser.add_argument("--layers", type=int, default=36, help="load layers 1..N (36 = the whole model)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--out", type=Path, default=ac.ROOT / "data" / "activations",
                        help="parent folder for outputs (on Colab, a Google Drive folder)")
    parser.add_argument("--same-machine", action="store_true",
                        help="this is the laptop that made the pilot's references (stricter threshold)")
    parser.add_argument("--export-reference", action="store_true")
    args = parser.parse_args()

    ac.require_equivalence_check()
    if args.export_reference:
        export_reference()
        return
    if not args.set:
        parser.error("--set is required")
    if args.layers < CHECK_LAYER and args.set == "reference":
        parser.error(f"the reference check needs at least {CHECK_LAYER} layers")

    if args.set == "expense":
        review_gate.require_reviewed()   # Aya doesn't see the claims until the owner has approved them
    ids, texts, source_sha = load_set(args.set)
    ac.keep_awake()
    tokenizer, model, device, dtype, why = ac.load_model(args.layers, args.device)
    print(f"Model: {args.layers} layers on {device}, {str(dtype).removeprefix('torch.')} ({why})")
    prompts = [ac.build_prompt(tokenizer, t) for t in texts]
    final = tokenizer.convert_ids_to_tokens(tokenizer(prompts[0], add_special_tokens=False)["input_ids"][-1])

    out = args.out / f"{args.set}_layers0-{args.layers}_{device}_{str(dtype).removeprefix('torch.')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(prompts)} prompts -> {out}")
    run = collect(tokenizer, model, device, dtype, ids, prompts, out, args.batch)
    padding_cosine = ac.padding_check(tokenizer, model, prompts, device, dtype)

    meta = {
        "created_by": "collect_v2_activations.py", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "set": args.set, "n": len(ids), "source_sha256": source_sha,
        "model": ac.MODEL_ID, "revision": ac.MODEL_REVISION, "layers_loaded": args.layers,
        "saved": "activations.npy[i, k] = output of layer k at the last prompt token (k = 0 is the token embedding); "
                 "final norm switched off; computed and saved in float16",
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": f"{str(dtype).removeprefix('torch.')} for the layers, embeddings looked up on the CPU",
        "batch_size": args.batch, "chat_template_sha256": ac.chat_template_sha256(tokenizer), "final_tokens": [final],
        "padding_check_cosine": round(padding_cosine, 6),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        **run,
    }
    ok = run["last_layer_equals_final_output"] and padding_cosine >= SAME_MACHINE_COSINE
    if args.set == "reference":
        acts = np.load(out / "activations.npy")
        meta["reference_check"] = reference_check(acts, ids, args.same_machine)
        ok &= all(meta["reference_check"]["checks"].values())
    meta["all_checks_passed"] = bool(ok)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Last saved layer equals the model's final output: {run['last_layer_equals_final_output']}")
    print(f"Padding check cosine: {padding_cosine:.6f}")
    if args.set == "reference":
        for name, passed in meta["reference_check"]["checks"].items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print(json.dumps(meta["reference_check"]["detail"], indent=2))
    print(f"{'All checks passed.' if ok else 'CHECKS FAILED.'} Wrote {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

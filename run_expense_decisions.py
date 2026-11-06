"""Part C step 3: Aya's decisions on the 144 expense prompts (Colab, T4, whole model).

    uv run python run_expense_decisions.py --dry-run                 # anywhere: fake model, checks the plumbing
    uv run python run_expense_decisions.py --smoke --out DIR         # Colab: 2 decisions, printed in full
    uv run python run_expense_decisions.py --out DIR                 # Colab: all 384 decisions
    uv run python run_expense_decisions.py --integration-check --out DIR --expense-acts ACTS_DIR

What runs (PLAN.md, "Reliability, from tau-bench pass^k"):
  - greedy: every prompt (48 claims x 3 instruction wordings) = 144 decisions
  - sampled: wording W1 only, 5 samples per claim at temperature 0.7 = 240 decisions
  Each sample k reseeds with SEED + k right before its call, so any one decision can be rerun alone.
  max_new_tokens = 80. Batch size 1, so there's no padding.

During each greedy decision, hooks save Aya's state at the last prompt token at every layer (entry
0 = token embedding, entry k = output of layer k, before the final norm), the same definition as
collect_v2_activations.py. --integration-check then compares these with the --set expense
activations, to show that a gate reading the live decision pass sees the numbers that get scored.

Precision matches the activation code: weights load as bfloat16, then the embeddings, layers, final
norm and output layer move to float16 on GPUs without bfloat16 (the T4). The rotary position table
stays in float32, as in aya_common.load_model.

Checks recorded in meta.json: a greedy rerun of 3 prompts gives identical text; every output is
parsed (strictly, and leniently) and the counts are reported; unparseable outputs are kept verbatim.
Scoring happens later on the laptop (score_expense.py), which imports parse_decision from here.

Refuses to run on real prompts until the owner's review is complete (review_gate.py).
"""

import argparse
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import aya_common as ac
import review_gate

SEED = 42
N_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS = 5, 0.7, 80
SAMPLED_WORDING = "W1"
RERUN_CHECK = 3
ACTIONS = ("APPROVE", "REJECT", "REQUEST_INFO")
INTEGRATION_COSINE = 0.999
STOP_TOKENS = ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>", "<EOS_TOKEN>")


# ---------------------------------------------------------------- parsing (also used by score_expense.py)
def parse_decision(text: str) -> dict:
    """Strict: the reply is exactly 'DECISION: <action>' then 'REASON: <text>', nothing else.
    Lenient: markdown stars and backticks removed, blank lines ignored, and exactly one DECISION
    line with a valid action anywhere. action is None when even the lenient parse fails."""
    lines = [l.strip() for l in text.strip().splitlines()]
    strict = (len(lines) == 2 and re.fullmatch(rf"DECISION: ({'|'.join(ACTIONS)})", lines[0]) is not None
              and re.fullmatch(r"REASON: \S.*", lines[1]) is not None)
    clean = [re.sub(r"[*`]", "", l).strip() for l in lines if l.strip()]
    found = [m.group(1) for l in clean if (m := re.fullmatch(rf"DECISION:\s*({'|'.join(ACTIONS)})\.?", l, re.I))]
    reasons = [m.group(1) for l in clean if (m := re.fullmatch(r"REASON:\s*(.+)", l, re.I))]
    action = found[0].upper() if len(found) == 1 else None
    return {"action": action, "reason": reasons[0] if reasons else None, "strict": strict}


# ---------------------------------------------------------------- jobs
def load_prompts() -> list[dict]:
    path = review_gate.PROMPTS
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def jobs(prompts: list[dict]) -> list[dict]:
    out = [{"key": f"{p['prompt_id']}|greedy", "mode": "greedy", "sample": None, **p} for p in prompts]
    out += [{"key": f"{p['prompt_id']}|s{k}", "mode": "sampled", "sample": k, **p}
            for p in prompts if p["wording"] == SAMPLED_WORDING for k in range(N_SAMPLES)]
    return out


# ---------------------------------------------------------------- the model
class Aya:
    """The whole model for generation, with hooks that keep every layer's state at the last
    prompt token during the first (prompt) step of a greedy decision."""

    def __init__(self, layers: int | None = None):
        """layers: load only the first N layers. Only for checking the hooks on the laptop."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(ac.MODEL_ID, revision=ac.MODEL_REVISION)
        cut = {}
        if layers:
            full = AutoConfig.from_pretrained(ac.MODEL_ID, revision=ac.MODEL_REVISION)
            cut = {"num_hidden_layers": layers, "layer_types": full.layer_types[:layers]}
        model = AutoModelForCausalLM.from_pretrained(ac.MODEL_ID, revision=ac.MODEL_REVISION, dtype=torch.bfloat16, **cut)
        if not torch.cuda.is_available():
            raise SystemExit("This needs the Colab GPU.")
        self.device = "cuda"
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        model.to(self.device)                      # move only; dtypes unchanged so far
        inner = model.model
        for part in (inner.embed_tokens, inner.layers, inner.norm, model.lm_head):
            part.to(self.dtype)                    # the rotary table is left alone, as in aya_common.load_model
        assert inner.rotary_emb.inv_freq.dtype == torch.float32, "the rotary table must stay float32"
        assert model.lm_head.weight.data_ptr() == inner.embed_tokens.weight.data_ptr(), "output layer came untied"
        model.eval()
        model.requires_grad_(False)
        self.model, self.n_layers = model, len(inner.layers)
        # The model's own generation settings list only <EOS_TOKEN> (3). A chat reply ends with
        # <|END_RESPONSE|> and then <|END_OF_TURN_TOKEN|>, so all three stop generation.
        stop = [self.tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS]
        assert self.tokenizer.unk_token_id not in stop, "a stop token isn't in the vocabulary"
        self.eos = sorted(set(stop) | {model.generation_config.eos_token_id})
        common = {"max_new_tokens": MAX_NEW_TOKENS, "eos_token_id": self.eos,
                  "pad_token_id": self.tokenizer.pad_token_id}
        self.greedy_config = GenerationConfig(do_sample=False, **common)
        self.sample_config = GenerationConfig(do_sample=True, temperature=TEMPERATURE, top_k=0, top_p=1.0, **common)

        self.capture, self.captured = False, {}
        inner.embed_tokens.register_forward_hook(self._hook(0))
        for k, layer in enumerate(inner.layers, start=1):
            layer.register_forward_hook(self._hook(k))

    def _hook(self, k: int):
        def save(_module, _inputs, output):
            if self.capture:
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] > 1:                 # the prompt step, not a one-token generation step
                    self.captured[k] = h[0, -1].to(self.torch.float16).cpu().numpy()
        return save

    def generate(self, user_message: str, mode: str, sample: int | None) -> dict:
        torch = self.torch
        prompt = ac.build_prompt(self.tokenizer, user_message)
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        self.captured, self.capture = {}, mode == "greedy"
        if mode == "sampled":
            torch.manual_seed(SEED + sample)
        started = time.time()
        with torch.inference_mode():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                                      generation_config=self.greedy_config if mode == "greedy" else self.sample_config)
        self.capture = False
        new = out[0, ids.shape[1]:].tolist()
        eos = set(self.eos)
        states = None
        if mode == "greedy":
            assert sorted(self.captured) == list(range(self.n_layers + 1)), "a layer's state wasn't captured"
            states = np.stack([self.captured[k] for k in range(self.n_layers + 1)])
        return {"output": self.tokenizer.decode(new, skip_special_tokens=True),
                "output_with_special_tokens": self.tokenizer.decode(new, skip_special_tokens=False),
                "prompt_tokens": ids.shape[1], "new_tokens": len(new),
                "stopped": "end_of_turn" if new and new[-1] in eos else "max_new_tokens",
                "seconds": round(time.time() - started, 2)}, states

    def describe(self) -> dict:
        import torch
        import transformers
        return {"device": self.device, "gpu": torch.cuda.get_device_name(0),
                "dtype": f"{str(self.dtype).removeprefix('torch.')} for embeddings, layers, final norm and output "
                         "layer; rotary table float32; loaded from bfloat16",
                "chat_template_sha256": ac.chat_template_sha256(self.tokenizer),
                "stop_token_ids": self.eos,
                "greedy_config": self.greedy_config.to_diff_dict(), "sample_config": self.sample_config.to_diff_dict(),
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "transformers": transformers.__version__}}


class FakeAya:
    """For --dry-run: no model, canned replies, including ones that must fail to parse."""
    n_layers = 36
    REPLIES = ["DECISION: APPROVE\nREASON: Everything checks out.",
               "DECISION: REQUEST_INFO\nREASON: The attendee count is missing.",
               "**DECISION:** REJECT\nREASON: Alcohol is on the receipt.",
               "I think this should probably be approved."]

    def generate(self, user_message, mode, sample):
        text = self.REPLIES[(len(user_message) + (sample or 0)) % len(self.REPLIES)]   # same input, same reply
        states = np.full((self.n_layers + 1, 4), len(user_message) % 97, dtype=np.float16) if mode == "greedy" else None
        return {"output": text, "output_with_special_tokens": text, "prompt_tokens": 0, "new_tokens": 0,
                "stopped": "end_of_turn", "seconds": 0.0}, states

    def describe(self):
        return {"device": "none (dry run)", "fake_model": True}


# ---------------------------------------------------------------- running
def read_done(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def run_jobs(aya, todo: list[dict], out: Path, label: str) -> None:
    """Run each job unless already saved. A greedy decision's states are written before its line,
    so a saved line always has its states."""
    path, states_dir = out / "decisions.jsonl", out / "greedy_states"
    states_dir.mkdir(parents=True, exist_ok=True)
    done = read_done(path)
    started, ran = time.time(), 0
    with path.open("a", encoding="utf-8") as f:
        for i, job in enumerate(todo):
            state_file = states_dir / f"{job['prompt_id']}.npy"
            if job["key"] in done and (job["mode"] != "greedy" or state_file.exists()):
                continue
            result, states = aya.generate(job["user_message"], job["mode"], job["sample"])
            if states is not None:
                np.save(state_file, states)
            row = {"key": job["key"], "prompt_id": job["prompt_id"], "example_id": job["example_id"],
                   "wording": job["wording"], "mode": job["mode"], "sample": job["sample"],
                   "seed": None if job["mode"] == "greedy" else SEED + job["sample"],
                   **result, "parsed": parse_decision(result["output"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            ran += 1
            if ran % 12 == 0 or i == len(todo) - 1:
                print(f"  {label}: {i + 1}/{len(todo)}, {time.time() - started:.0f} s")


def rerun_check(aya, prompts: list[dict], out: Path) -> dict:
    saved = read_done(out / "decisions.jsonl")
    picks = [prompts[i] for i in np.linspace(0, len(prompts) - 1, RERUN_CHECK).astype(int)]
    same = {}
    for p in picks:
        again, _ = aya.generate(p["user_message"], "greedy", None)
        same[p["prompt_id"]] = again["output"] == saved[f"{p['prompt_id']}|greedy"]["output"]
    return same


def summarise(out: Path) -> dict:
    rows = list(read_done(out / "decisions.jsonl").values())
    summary = {}
    for mode in ("greedy", "sampled"):
        r = [x for x in rows if x["mode"] == mode]
        summary[mode] = {"decisions": len(r), "strict_parse": sum(x["parsed"]["strict"] for x in r),
                         "lenient_parse": sum(x["parsed"]["action"] is not None for x in r),
                         "hit_max_new_tokens": sum(x["stopped"] == "max_new_tokens" for x in r),
                         "actions": {a: sum(x["parsed"]["action"] == a for x in r) for a in (*ACTIONS, None)}}
    return summary


def write_meta(out: Path, aya, prompts_sha: str, extra: dict) -> dict:
    meta = {"created_by": "run_expense_decisions.py", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": ac.MODEL_ID, "revision": ac.MODEL_REVISION, "prompts_sha256": prompts_sha,
            "prompt_template": "aya_common.build_prompt (chat template, one user turn, generation prompt added)",
            "seed": SEED, "sampled_wording": SAMPLED_WORDING, "n_samples": N_SAMPLES, "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS, "batch_size": 1,
            "greedy_states": "greedy_states/<prompt_id>.npy, shape (layers + 1, hidden): entry 0 token embedding, "
                             "entry k output of layer k (before the final norm), last prompt token, float16",
            **aya.describe(), **extra, "summary": summarise(out)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------- integration check
def integration_check(out: Path, acts_dir: Path) -> dict:
    acts = np.load(acts_dir / "activations.npy").astype(np.float32)
    ids = json.loads((acts_dir / "example_ids.json").read_text(encoding="utf-8"))
    live = np.stack([np.load(out / "greedy_states" / f"{i}.npy") for i in ids]).astype(np.float32)
    if live.shape != acts.shape:
        raise SystemExit(f"shapes differ: decision states {live.shape}, activations {acts.shape}")
    cos = (live * acts).sum(2) / (np.linalg.norm(live, axis=2) * np.linalg.norm(acts, axis=2))
    lowest = cos.min(axis=0)
    result = {"rows": len(ids), "lowest_cosine_by_layer": [round(float(c), 6) for c in lowest],
              "threshold": INTEGRATION_COSINE, "passed": bool((lowest[1:] >= INTEGRATION_COSINE).all())}
    (out / "integration_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="parent folder (on Colab, the Drive outputs folder)")
    parser.add_argument("--dry-run", action="store_true", help="fake model, no review needed; checks the plumbing")
    parser.add_argument("--smoke", action="store_true", help="2 greedy decisions, printed in full")
    parser.add_argument("--integration-check", action="store_true")
    parser.add_argument("--expense-acts", type=Path, help="the --set expense activations folder")
    args = parser.parse_args()
    ac.require_equivalence_check()

    if args.dry_run:
        out = (args.out or ac.ROOT / "data" / "dry_run") / "expense_decisions_dry_run"
    else:
        if not args.out:
            parser.error("--out is required")
        out = args.out / ("expense_decisions_smoke" if args.smoke else "expense_decisions")

    if args.integration_check:
        if not args.expense_acts:
            parser.error("--integration-check needs --expense-acts")
        result = integration_check(out, args.expense_acts)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)

    if args.dry_run:
        prompts_sha = ac.sha256(review_gate.PROMPTS)
        aya = FakeAya()
    else:
        prompts_sha = review_gate.require_reviewed()   # Aya doesn't see the claims until the owner has approved them
        ac.keep_awake()
        aya = Aya()
    prompts = load_prompts()
    todo = jobs(prompts)
    if args.smoke:
        todo = [j for j in todo if j["mode"] == "greedy"]
        todo = [todo[0], next(j for j in todo if j["example_id"].endswith("-v3"))]
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} decisions -> {out}")

    run_jobs(aya, [j for j in todo if j["mode"] == "greedy"], out, "greedy")
    run_jobs(aya, [j for j in todo if j["mode"] == "sampled"], out, "sampled")
    extra = {}
    if not args.smoke:
        extra["greedy_rerun_identical"] = rerun_check(aya, prompts, out)
    meta = write_meta(out, aya, prompts_sha, extra)

    if args.smoke:
        for r in read_done(out / "decisions.jsonl").values():
            print(f"\n--- {r['prompt_id']} ({r['new_tokens']} new tokens, stopped at {r['stopped']})\n"
                  f"{r['output_with_special_tokens']}\n--- parsed: {r['parsed']}")
    print(json.dumps(meta["summary"], indent=2))
    ok = all(extra.get("greedy_rerun_identical", {}).values())
    if "greedy_rerun_identical" in extra:
        print(f"Greedy rerun of {RERUN_CHECK} prompts identical: {ok}")
    print(f"Wrote {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

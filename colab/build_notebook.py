"""Writes colab/run_on_colab.ipynb. Edit the cells here, then: uv run python colab/build_notebook.py"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "run_on_colab.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Expense tripwire: Colab session

This notebook runs the parts of the project that need the whole Tiny Aya model, which doesn't fit on
the laptop's 4 GB GPU (PLAN.md, Part C). It needs a **T4 GPU** (Runtime → Change runtime type → T4).

Before you start, on the laptop:

1. `uv run python make_colab_upload.py`
2. Upload `colab/expense_tripwire_upload.zip` to a Google Drive folder called `expense-tripwire`
   (so the path is `My Drive/expense-tripwire/expense_tripwire_upload.zip`).

Run the cells from the top. Everything is saved to `My Drive/expense-tripwire/outputs/`, in chunks, so
if Colab disconnects you can run all cells again and finished work is skipped.

| Step | What | Needs the owner's review of the claims? |
|---|---|---|
| 1 | Cross-machine check: recompute the pilot's 68 reference questions and compare with the laptop | no |
| 2 | v2 activations: 4,000 UMWP and SQuAD prompts, every layer (about 600 MB) | no |
| 3 | Expense activations: 144 expense prompts, every layer | **yes** |
| 4 | Aya's expense decisions: 2 first to read, then all 384, then the integration check | **yes** |
| 5 | Monitor baseline: one yes/no question per claim | **yes** |

Step 1 must pass before anything else is used.
""")

code("""
from google.colab import drive
drive.mount('/content/drive')

import hashlib, json, os, pathlib, shutil, subprocess, sys, zipfile
DRIVE = pathlib.Path('/content/drive/MyDrive/expense-tripwire')
OUT = DRIVE / 'outputs'
ACTS = OUT / 'activations'
PROJECT = pathlib.Path('/content/expense-tripwire')
ZIP = DRIVE / 'expense_tripwire_upload.zip'
assert ZIP.exists(), f'Upload the zip from the laptop to {ZIP} first.'
OUT.mkdir(parents=True, exist_ok=True)


def run(cmd):
    \"\"\"Run a shell command in the project folder, show its output as it comes, stop on failure.\"\"\"
    p = subprocess.Popen(cmd, shell=True, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='')
    if p.wait() != 0:
        raise RuntimeError(f'Failed (exit {p.returncode}): {cmd}')


def finished(set_name):
    \"\"\"True if this set was collected with the whole model and all its checks passed.\"\"\"
    meta = ACTS / f'{set_name}_layers0-36_cuda_float16' / 'meta.json'
    return meta.exists() and json.loads(meta.read_text())['all_checks_passed']
""")

md("""
## Unpack the upload and check every file
""")

code("""
if PROJECT.exists():
    shutil.rmtree(PROJECT)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(PROJECT)
manifest = json.loads((PROJECT / 'MANIFEST.json').read_text())
bad = [f for f, h in manifest['files'].items()
       if hashlib.sha256((PROJECT / f).read_bytes()).hexdigest() != h]
assert not bad, f'These files changed in transit: {bad}'
stamp = manifest['built_at'].replace(':', '').replace('+', '_')
shutil.copy(PROJECT / 'MANIFEST.json', OUT / f'upload_manifest_{stamp}.json')
print(f"{len(manifest['files'])} files match the manifest (zip built {manifest['built_at']}).")
print(f"Model {manifest['model']} at revision {manifest['revision']}")
print('Expense steps:', 'ready' if manifest['expense_ready'] else 'locked until the review is done')
""")

md("""
## GPU and software

The laptop's numbers were made in float16 on a GPU without bfloat16 support. A T4 is the same kind of
GPU, so it computes in float16 too. A different GPU (L4, A100) would switch to bfloat16, which changes
the numbers, so the notebook stops instead.

The software is installed from the project's lock file, so torch, transformers and the rest are the
exact versions the laptop used. That takes a few minutes.
""")

code("""
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print('GPU:', gpu)
assert 'T4' in gpu, 'Switch the runtime to a T4 GPU (Runtime -> Change runtime type).'

import torch as colab_torch   # Colab's own copy, recorded for the record only; the scripts use the locked one
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
run('uv sync --frozen')
run('uv run python -c "import sys, torch, transformers; '
    'print(sys.version.split()[0], torch.__version__, torch.version.cuda, transformers.__version__)"')
env = {'gpu': gpu, 'colab_preinstalled_torch': colab_torch.__version__,
       'locked_versions': subprocess.run('uv run python -c "import torch, transformers, numpy; '
                                         'print(torch.__version__, transformers.__version__, numpy.__version__)"',
                                         shell=True, cwd=PROJECT, capture_output=True, text=True).stdout.strip()}
(OUT / f'colab_env_{stamp}.json').write_text(json.dumps(env, indent=2))
print(env)
""")

md("""
## Log in to Hugging Face

Tiny Aya is gated, so the download needs your Hugging Face token. Type it into the box that appears.
Don't paste it into a code cell, and don't save the notebook with it in.
""")

code("""
from huggingface_hub import notebook_login
notebook_login()
""")

md("""
## Step 1: cross-machine check

Recomputes the pilot's 20 smoke and 48 diagnostic questions with the whole model and compares them with
the laptop's saved numbers. Layer 18 must match with cosine at least 0.999 on every question, and layers
17 and 19 must match worse (which proves the layers are numbered right). If this fails, stop and don't
use anything else from Colab.
""")

code("""
if finished('reference'):
    print('Already done; skipping.')
else:
    run(f'uv run python collect_v2_activations.py --set reference --out "{ACTS}"')
meta = json.loads((ACTS / 'reference_layers0-36_cuda_float16' / 'meta.json').read_text())
for name, ok in meta['reference_check']['checks'].items():
    print('PASS' if ok else 'FAIL', name)
print(json.dumps(meta['reference_check']['detail'], indent=2))
assert meta['all_checks_passed'], 'The cross-machine check failed. Stop here.'
""")

md("""
## Step 2: v2 activations

4,000 prompts (UMWP and SQuAD 2.0), every layer saved, so the v2 layer can be chosen on validation data
later on the laptop. Expect roughly half an hour. Progress is saved every 128 prompts.
""")

code("""
assert finished('reference'), 'Step 1 must pass first.'
if finished('v2'):
    print('Already done; skipping.')
else:
    run(f'uv run python collect_v2_activations.py --set v2 --out "{ACTS}"')
meta = json.loads((ACTS / 'v2_layers0-36_cuda_float16' / 'meta.json').read_text())
print({k: meta[k] for k in ('n', 'prompt_tokens', 'seconds', 'padding_check_cosine', 'all_checks_passed')})
""")

md("""
## Step 3: expense activations (after the owner's review)

The 144 expense prompts (48 claims x 3 wordings), exactly as Aya sees them. This only runs once the
owner has approved all 48 claims and the zip was rebuilt afterwards; the script checks this itself too.
""")

code("""
assert finished('reference'), 'Step 1 must pass first.'
if not manifest['expense_ready']:
    print('Skipped: the review stamp isn\\'t in this upload. Rebuild the zip after the review.')
elif finished('expense'):
    print('Already done; skipping.')
else:
    run(f'uv run python collect_v2_activations.py --set expense --out "{ACTS}"')
""")

md("""
## Step 4a: two decisions first (after the owner's review)

Aya decides two claims (one complete, one with something missing), and both replies are printed in
full, along with how they were read. Read them before going on: if the replies don't follow the
two-line format, stop and tell Claude, since the full run would be wasted.
""")

code("""
def expense_ready():
    if not manifest['expense_ready']:
        print('Skipped: the review stamp isn\\'t in this upload. Rebuild the zip after the review.')
    return manifest['expense_ready']

if expense_ready():
    run(f'uv run python run_expense_decisions.py --smoke --out "{OUT}"')
""")

md("""
## Step 4b: all 384 decisions

144 greedy decisions (48 claims x 3 wordings) and 240 sampled ones (5 per claim, temperature 0.7).
Expect about half an hour. Each decision is saved as it's made. At the end, 3 greedy decisions are
rerun and must come out identical.
""")

code("""
if expense_ready():
    run(f'uv run python run_expense_decisions.py --out "{OUT}"')
""")

md("""
## Step 4c: integration check

Compares Aya's internal state during her real decisions with the step 3 activations, at every layer.
They should match (cosine at least 0.999), which shows a gate reading the live decision would see the
numbers we score.
""")

code("""
if expense_ready():
    assert finished('expense'), 'Step 3 must finish first.'
    run(f'uv run python run_expense_decisions.py --integration-check --out "{OUT}" '
        f'--expense-acts "{ACTS / "expense_layers0-36_cuda_float16"}"')
""")

md("""
## Step 5: monitor baseline

Fresh calls, one per claim, asking Aya whether anything the policy requires is missing. It only runs
once all 384 decisions are saved.
""")

code("""
if expense_ready():
    run(f'uv run python run_monitor_baseline.py --out "{OUT}"')
""")

md("""
## What was saved

Everything is in `My Drive/expense-tripwire/outputs/`. Download that folder to the laptop (for example
with Google Drive for desktop) into `data/colab_outputs/`.
""")

code("""
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        print(f'{p.stat().st_size / 1e6:9.1f} MB  {p.relative_to(OUT)}')
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)

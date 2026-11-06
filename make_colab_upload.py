"""Part C: build the zip that goes to Google Drive for colab/run_on_colab.ipynb.

    uv run python make_colab_upload.py

It holds only what Colab runs: the scripts, the locked dependencies, the equivalence record, the
v2 inputs, the pilot reference copies and the expense prompts. It never holds the hidden worlds,
the oracle labels, the review sheet, the claim source file or the v2 labels, so nothing on Colab
can leak them into a prompt. MANIFEST.json inside the zip lists every file's sha256; the notebook
checks them after unzipping.

The expense steps on Colab need data/expense/review_stamp.json, which review_gate.py writes once
the owner has approved all 48 claims. Without it the zip still works for the cross-machine check
and the v2 activations, and the expense steps refuse to run.
"""

import json
import zipfile
from datetime import datetime, timezone

import aya_common as ac
import review_gate

OUT = ac.ROOT / "colab" / "expense_tripwire_upload.zip"
SCRIPTS = ["aya_common.py", "review_gate.py", "collect_v2_activations.py",
           "run_expense_decisions.py", "run_monitor_baseline.py"]   # the last two once they're built
ALWAYS = ["pyproject.toml", "uv.lock", ".python-version", "results/equivalence_check.json",
          "data/v2/model_inputs.jsonl", "data/expense/prompts.jsonl"]
NEVER = ["data/expense/world.jsonl", "data/expense/items.jsonl", "data/expense/review_sheet.csv",
         "data/v2/labels.jsonl", "expense_families.py", "expense_oracle.py", "build_expense.py"]


def main() -> None:
    ac.require_equivalence_check()
    files = [s for s in SCRIPTS if (ac.ROOT / s).exists()] + ALWAYS
    files += sorted(p.relative_to(ac.ROOT).as_posix() for p in (ac.ROOT / "data" / "reference").rglob("*") if p.is_file())
    missing_scripts = [s for s in SCRIPTS if s not in files]

    try:
        review_gate.require_reviewed()
        files.append("data/expense/review_stamp.json")
        expense_ready, expense_note = True, "review complete: expense steps can run"
    except SystemExit as e:
        expense_ready, expense_note = False, f"expense steps will refuse on Colab ({e})"

    assert not set(files) & set(NEVER), "a file that must stay on the laptop was about to be zipped"
    if "run_monitor_baseline.py" in files:   # its copy of the banned-word patterns must match build_expense.py's
        import build_expense
        import run_monitor_baseline
        assert run_monitor_baseline.BANNED == build_expense.BANNED, "the banned-word patterns have drifted apart"
        run_monitor_baseline.build_monitor_prompts()
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": ac.MODEL_ID, "revision": ac.MODEL_REVISION, "expense_ready": expense_ready,
        "files": {f: ac.sha256(ac.ROOT / f) for f in files},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ac.ROOT / f, f)   # bytes unchanged, so aya_common.py keeps its checked sha256
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        notebook = ac.ROOT / "colab" / "run_on_colab.ipynb"
        if notebook.exists():
            z.write(notebook, "colab/run_on_colab.ipynb")

    print(f"Wrote {OUT.relative_to(ac.ROOT)}: {len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB")
    print(f"  {expense_note}")
    if missing_scripts:
        print(f"  not built yet, so not included: {', '.join(missing_scripts)}")
    print("Upload it to My Drive/expense-tripwire/ and open colab/run_on_colab.ipynb in Colab.")


if __name__ == "__main__":
    main()

"""The owner's review gate for the expense claims (PLAN.md, Part A).

Nothing may run Aya on the expense prompts (activations, decisions, monitor calls) or score them
until the owner has approved all 48 claims and their oracle labels.

How it works:
  - On the laptop, `full_check()` reads data/expense/items.jsonl and review_sheet.csv. Every claim
    must be "approved" in the sheet (any case, since Excel writes TRUE/True differently), and
    items.jsonl must agree, which shows build_expense.py was rerun after the review and the
    approved text is the text that was built.
  - When that passes, `write_stamp()` saves data/expense/review_stamp.json with the sha256 of
    prompts.jsonl. Colab gets only the stamp and prompts.jsonl, never items.jsonl or world.jsonl.
  - `require_reviewed()` is what other scripts call. It checks the stamp matches the current
    prompts.jsonl, and, where items.jsonl exists (the laptop), reruns the full check too.

    uv run python review_gate.py          # show the review state; writes the stamp if complete
"""

import csv
import json
from collections import Counter
from datetime import datetime, timezone

import aya_common as ac

EXPENSE = ac.ROOT / "data" / "expense"
ITEMS, SHEET, PROMPTS = EXPENSE / "items.jsonl", EXPENSE / "review_sheet.csv", EXPENSE / "prompts.jsonl"
STAMP = EXPENSE / "review_stamp.json"
N_CLAIMS = 48


def full_check() -> tuple[bool, list[str], Counter]:
    """Returns (passed, problems, status counts from the sheet)."""
    problems = []
    items = {r["example_id"]: r for r in map(json.loads, ITEMS.open(encoding="utf-8"))}
    with SHEET.open(encoding="utf-8-sig", newline="") as f:
        sheet = {r["example_id"]: r["label_status"].strip().lower() for r in csv.DictReader(f)}
    counts = Counter(sheet.values())
    if len(items) != N_CLAIMS or set(items) != set(sheet):
        problems.append(f"items.jsonl has {len(items)} claims and the sheet {len(sheet)}; they must be the same "
                        f"{N_CLAIMS}. Rerun build_expense.py.")
    not_approved = sorted(i for i, s in sheet.items() if s != "approved")
    if not_approved:
        problems.append(f"{len(not_approved)} claims aren't approved in the sheet: {', '.join(not_approved[:8])}"
                        + (" ..." if len(not_approved) > 8 else ""))
    stale = sorted(i for i in set(items) & set(sheet) if items[i]["label_status"] != sheet[i])
    if stale:
        problems.append(f"the sheet and items.jsonl disagree for {len(stale)} claims ({', '.join(stale[:8])}); "
                        "rerun build_expense.py so it picks up the review.")
    prompt_ids = {json.loads(line)["example_id"] for line in PROMPTS.open(encoding="utf-8")}
    if prompt_ids != set(items):
        problems.append("prompts.jsonl doesn't cover exactly the claims in items.jsonl; rerun build_expense.py.")
    return not problems, problems, counts


def write_stamp() -> dict:
    stamp = {"all_claims_approved": True, "claims": N_CLAIMS, "prompts_sha256": ac.sha256(PROMPTS),
             "items_sha256": ac.sha256(ITEMS), "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    STAMP.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    return stamp


def require_reviewed() -> str:
    """Stop unless the owner's review is complete for the current prompts. Returns their sha256."""
    if ITEMS.exists():
        passed, problems, _ = full_check()
        if not passed:
            raise SystemExit("The expense claims aren't fully reviewed:\n  " + "\n  ".join(problems))
    if not STAMP.exists():
        raise SystemExit("No review stamp. On the laptop, run `uv run python review_gate.py` after the review.")
    stamp = json.loads(STAMP.read_text(encoding="utf-8"))
    current = ac.sha256(PROMPTS)
    if not stamp.get("all_claims_approved") or stamp.get("prompts_sha256") != current:
        raise SystemExit("prompts.jsonl changed since the review was stamped; rerun review_gate.py on the laptop.")
    return current


def main() -> None:
    passed, problems, counts = full_check()
    print("Review sheet: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    if not passed:
        STAMP.unlink(missing_ok=True)
        print("Not complete:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    stamp = write_stamp()
    print(f"All {N_CLAIMS} claims approved. Wrote {STAMP.relative_to(ac.ROOT)} "
          f"(prompts sha256 {stamp['prompts_sha256'][:12]}...)")


if __name__ == "__main__":
    main()

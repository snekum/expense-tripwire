# v2 probe data: summary for review

Formatting quirks cleaned: **True** (rows changed: {'squad2': 145, 'umwp': 842})

| dataset | split | rows | unanswerable | groups |
|---|---|---|---|---|
| squad2 | train | 1201 | 591 | 595 |
| squad2 | validation | 400 | 221 | 184 |
| squad2 | test | 399 | 188 | 204 |
| umwp | train | 1200 | 600 | 600 |
| umwp | validation | 400 | 200 | 200 |
| umwp | test | 400 | 200 | 200 |

## Length

| dataset | label | mean words | median | min | max |
|---|---|---|---|---|---|
| squad2 | should_abstain=False | 150.3 | 135.0 | 48 | 575 |
| squad2 | should_abstain=True | 148.6 | 137.0 | 48 | 564 |
| umwp | should_abstain=False | 40.9 | 37.0 | 14 | 126 |
| umwp | should_abstain=True | 40.1 | 36.0 | 12 | 127 |

| dataset | length baseline, validation AUROC | word count alone | above 0.75? |
|---|---|---|---|
| umwp | 0.5452 | 0.5045 | no |
| squad2 | 0.4855 | 0.5085 | no |

## Formatting cues (share of rows; SQuAD counts the question only)

| dataset | cue | upstream: answerable | upstream: unanswerable | used: answerable | used: unanswerable |
|---|---|---|---|---|---|
| umwp | double period | 0.159 | 0.0 | 0.0 | 0.0 |
| umwp | no space after a sentence | 0.059 | 0.067 | 0.0 | 0.0 |
| umwp | leading space | 0.108 | 0.0 | 0.0 | 0.0 |
| umwp | trailing space | 0.113 | 0.111 | 0.0 | 0.0 |
| umwp | double space | 0.171 | 0.172 | 0.0 | 0.0 |
| squad2 | double period | 0.002 | 0.0 | 0.0 | 0.0 |
| squad2 | no space after a sentence | 0.0 | 0.0 | 0.0 | 0.0 |
| squad2 | leading space | 0.002 | 0.062 | 0.0 | 0.0 |
| squad2 | trailing space | 0.045 | 0.02 | 0.0 | 0.0 |
| squad2 | double space | 0.01 | 0.005 | 0.0 | 0.0 |

UMWP unanswerable categories: Ambiguous key information 571, Key information missing 386, Unrelated object 43

UMWP sources: GSM8K 661, SVAMP 185, MultiArith 108, ASDiv 46

## Overlap

- Largest share of a diagnostic question found in a v2 row: 0.143
- Largest share of an expense claim found in a v2 row: 0.036
- UMWP answerable problems from GSM8K: 661; of these, 0 are word-for-word pilot (v1 training) questions and 0 share at least 80% of their word triples with one.

## Checks

- PASS: example IDs unique
- PASS: 1,000 rows per label in each dataset
- PASS: every UMWP pair complete (one answerable, one not)
- PASS: UMWP unanswerable rows only from the kept categories
- PASS: no pair or passage in two splits
- PASS: no question text in two splits
- PASS: no SQuAD question (without its passage) in two splits
- PASS: no v2 row copies a diagnostic question (largest share 0.143)
- PASS: no v2 row copies an expense claim (largest share 0.036)

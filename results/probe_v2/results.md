# Probe v2: layer choice and results

Each probe reads Aya's internal state at the last prompt token and scores how likely the question can't be answered from what it gives. AUROC is the chance that a randomly picked unanswerable question scores higher than a randomly picked answerable one (0.5 = guessing, 1.0 = perfect).

## Layer choice (validation splits only)

Rule fixed on 2026-11-04: train on one dataset, test on the other, average both directions; best layer, or the earliest within 0.01 of it. Best mean transfer AUROC 0.720; layers within 0.01: [23]. **Chosen: layer 23.** Layer 18 (the pilot's) is reported alongside everywhere.

| Layer | Within UMWP | Within SQuAD | UMWP → SQuAD | SQuAD → UMWP | Mean transfer | Shuffled-label mean transfer |
|---|---|---|---|---|---|---|
| 1 | 0.685 | 0.573 | 0.503 | 0.473 | 0.488 | 0.467 |
| 2 | 0.730 | 0.623 | 0.515 | 0.497 | 0.506 | 0.456 |
| 3 | 0.741 | 0.667 | 0.524 | 0.503 | 0.513 | 0.444 |
| 4 | 0.765 | 0.647 | 0.490 | 0.516 | 0.503 | 0.483 |
| 5 | 0.780 | 0.706 | 0.495 | 0.516 | 0.505 | 0.503 |
| 6 | 0.774 | 0.727 | 0.480 | 0.541 | 0.511 | 0.488 |
| 7 | 0.788 | 0.736 | 0.495 | 0.542 | 0.518 | 0.458 |
| 8 | 0.801 | 0.726 | 0.510 | 0.579 | 0.544 | 0.470 |
| 9 | 0.818 | 0.734 | 0.511 | 0.553 | 0.532 | 0.486 |
| 10 | 0.819 | 0.747 | 0.522 | 0.550 | 0.536 | 0.524 |
| 11 | 0.842 | 0.768 | 0.551 | 0.544 | 0.548 | 0.519 |
| 12 | 0.864 | 0.750 | 0.540 | 0.506 | 0.523 | 0.516 |
| 13 | 0.870 | 0.773 | 0.540 | 0.447 | 0.493 | 0.485 |
| 14 | 0.889 | 0.787 | 0.577 | 0.491 | 0.534 | 0.450 |
| 15 | 0.905 | 0.831 | 0.590 | 0.459 | 0.524 | 0.453 |
| 16 | 0.923 | 0.834 | 0.554 | 0.492 | 0.523 | 0.483 |
| 17 | 0.931 | 0.862 | 0.599 | 0.445 | 0.522 | 0.471 |
| 18 (pilot's layer) | 0.941 | 0.880 | 0.694 | 0.542 | 0.618 | 0.532 |
| 19 | 0.936 | 0.902 | 0.692 | 0.562 | 0.627 | 0.525 |
| 20 | 0.930 | 0.907 | 0.718 | 0.615 | 0.666 | 0.507 |
| 21 | 0.927 | 0.908 | 0.720 | 0.646 | 0.683 | 0.540 |
| 22 | 0.923 | 0.910 | 0.685 | 0.690 | 0.688 | 0.545 |
| 23 (chosen) | 0.921 | 0.904 | 0.704 | 0.736 | 0.720 | 0.551 |
| 24 | 0.912 | 0.896 | 0.696 | 0.721 | 0.709 | 0.513 |
| 25 | 0.903 | 0.896 | 0.659 | 0.701 | 0.680 | 0.506 |
| 26 | 0.897 | 0.889 | 0.623 | 0.706 | 0.665 | 0.511 |
| 27 | 0.898 | 0.887 | 0.613 | 0.684 | 0.648 | 0.475 |
| 28 | 0.904 | 0.880 | 0.624 | 0.690 | 0.657 | 0.475 |
| 29 | 0.898 | 0.877 | 0.632 | 0.682 | 0.657 | 0.484 |
| 30 | 0.899 | 0.869 | 0.638 | 0.683 | 0.660 | 0.477 |
| 31 | 0.901 | 0.869 | 0.618 | 0.678 | 0.648 | 0.483 |
| 32 | 0.891 | 0.870 | 0.605 | 0.696 | 0.651 | 0.495 |
| 33 | 0.884 | 0.874 | 0.594 | 0.696 | 0.645 | 0.492 |
| 34 | 0.887 | 0.870 | 0.575 | 0.705 | 0.640 | 0.504 |
| 35 | 0.882 | 0.868 | 0.541 | 0.705 | 0.623 | 0.508 |
| 36 | 0.878 | 0.865 | 0.568 | 0.688 | 0.628 | 0.501 |
| length (any layer) | 0.545 | 0.486 | 0.554 | 0.500 | 0.527 | |
| tfidf (any layer) | 0.772 | 0.571 | 0.482 | 0.480 | 0.481 | |

"Within" = trained and scored on the same dataset (validation). The shuffled-label column should stay near 0.5 at every layer; if it doesn't, something leaks.

## Test split, pooled (trained on UMWP + SQuAD)

| Method | Layer | Test AUROC (95% CI) | Test balanced accuracy (95% CI) | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| probe | 18 | 0.897 (0.874-0.919) | 0.822 (0.796-0.848) | 348 | 63 | 79 | 309 |
| probe | 23 | 0.900 (0.880-0.919) | 0.816 (0.791-0.841) | 360 | 51 | 95 | 293 |
| length | - | 0.520 (0.494-0.548) | 0.505 (0.496-0.515) | 374 | 37 | 349 | 39 |
| tfidf | - | 0.645 (0.614-0.674) | 0.598 (0.568-0.628) | 291 | 120 | 199 | 189 |
| shuffled | 18 | 0.506 (0.474-0.539) | 0.502 (0.474-0.533) | 156 | 255 | 146 | 242 |
| shuffled | 23 | 0.478 (0.445-0.510) | 0.474 (0.445-0.502) | 241 | 170 | 248 | 140 |

799 test questions, 388 of them unanswerable.

## Test split, UMWP only (same pooled probe and threshold)

| Method | Layer | Test AUROC (95% CI) | Test balanced accuracy (95% CI) | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| probe | 18 | 0.915 (0.890-0.940) | 0.847 (0.815-0.877) | 173 | 27 | 34 | 166 |
| probe | 23 | 0.905 (0.881-0.930) | 0.825 (0.792-0.860) | 177 | 23 | 47 | 153 |
| length | - | 0.510 (0.498-0.522) | 0.505 (0.490-0.520) | 163 | 37 | 161 | 39 |
| tfidf | - | 0.750 (0.718-0.784) | 0.693 (0.657-0.728) | 161 | 39 | 84 | 116 |
| shuffled | 18 | 0.495 (0.463-0.528) | 0.475 (0.440-0.507) | 75 | 125 | 85 | 115 |
| shuffled | 23 | 0.491 (0.459-0.524) | 0.478 (0.445-0.510) | 129 | 71 | 138 | 62 |

400 questions.

## Test split, SQuAD only (same pooled probe and threshold)

| Method | Layer | Test AUROC (95% CI) | Test balanced accuracy (95% CI) | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| probe | 18 | 0.876 (0.838-0.908) | 0.795 (0.753-0.834) | 175 | 36 | 45 | 143 |
| probe | 23 | 0.894 (0.860-0.925) | 0.806 (0.767-0.843) | 183 | 28 | 48 | 140 |
| length | - | 0.513 (0.461-0.563) | 0.500 (0.500-0.500) | 211 | 0 | 188 | 0 |
| tfidf | - | 0.506 (0.458-0.560) | 0.502 (0.457-0.547) | 130 | 81 | 115 | 73 |
| shuffled | 18 | 0.518 (0.470-0.570) | 0.530 (0.485-0.575) | 81 | 130 | 61 | 127 |
| shuffled | 23 | 0.469 (0.417-0.523) | 0.473 (0.425-0.522) | 112 | 99 | 110 | 78 |

399 questions.

## Transfer: trained on UMWP only, tested on SQuAD

| Method | Layer | Test AUROC (95% CI) | Test balanced accuracy (95% CI) | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| probe | 18 | 0.674 (0.620-0.725) | 0.509 (0.495-0.523) | 7 | 204 | 3 | 185 |
| probe | 23 | 0.723 (0.678-0.769) | 0.649 (0.609-0.691) | 173 | 38 | 98 | 90 |
| length | - | 0.480 (0.431-0.529) | 0.491 (0.475-0.504) | 4 | 207 | 7 | 181 |
| tfidf | - | 0.491 (0.444-0.542) | 0.472 (0.429-0.518) | 106 | 105 | 105 | 83 |
| shuffled | 18 | 0.522 (0.467-0.584) | 0.500 (0.500-0.500) | 0 | 211 | 0 | 188 |
| shuffled | 23 | 0.687 (0.631-0.738) | 0.612 (0.574-0.652) | 183 | 28 | 121 | 67 |

Threshold from UMWP validation, so balanced accuracy shows whether the threshold carries over (in the pilot it didn't).

## Transfer: trained on SQuAD only, tested on UMWP

| Method | Layer | Test AUROC (95% CI) | Test balanced accuracy (95% CI) | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| probe | 18 | 0.529 (0.501-0.555) | 0.502 (0.492-0.513) | 4 | 196 | 3 | 197 |
| probe | 23 | 0.725 (0.698-0.757) | 0.588 (0.562-0.618) | 53 | 147 | 18 | 182 |
| length | - | 0.509 (0.496-0.521) | 0.500 (0.500-0.500) | 1 | 199 | 1 | 199 |
| tfidf | - | 0.463 (0.445-0.478) | 0.480 (0.458-0.502) | 100 | 100 | 108 | 92 |
| shuffled | 18 | 0.577 (0.552-0.601) | 0.500 (0.492-0.507) | 196 | 4 | 196 | 4 |
| shuffled | 23 | 0.348 (0.320-0.375) | 0.500 (0.500-0.500) | 1 | 199 | 1 | 199 |

Threshold from SQuAD validation, so balanced accuracy shows whether the threshold carries over (in the pilot it didn't).

TN/FP/FN/TP: positive = unanswerable. FP = answerable question flagged; FN = unanswerable question missed. 95% intervals resample whole UMWP pairs or SQuAD passages, 1,000 times.

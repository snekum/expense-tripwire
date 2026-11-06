# Scrambled-label spread (validation only)

100 probes per layer, each trained on scrambled labels, then scored on the other dataset's validation questions with the true labels. The real probe's number comes from layer_curve.csv (same splits, same procedure, true labels). "Beaten by" = share of scrambled probes scoring at least as high as the real one.

| Layer | Direction | Real probe | Scrambled: median | middle 90% | highest | Beaten by |
|---|---|---|---|---|---|---|
| 18 | UMWP → SQuAD | 0.694 | 0.515 | 0.413-0.607 | 0.638 | 0% |
| 18 | SQuAD → UMWP | 0.542 | 0.525 | 0.405-0.631 | 0.674 | 43% |
| 18 | mean of both | 0.618 | 0.519 | 0.446-0.590 | 0.623 | 1% |
| 23 | UMWP → SQuAD | 0.704 | 0.515 | 0.378-0.691 | 0.773 | 5% |
| 23 | SQuAD → UMWP | 0.736 | 0.528 | 0.398-0.664 | 0.689 | 0% |
| 23 | mean of both | 0.720 | 0.528 | 0.402-0.646 | 0.686 | 0% |

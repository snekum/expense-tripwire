# Gate simulation

Rule: if the gate fires, Aya's action becomes REQUEST_INFO. Applied to her 144 greedy decisions. "Recalibrated": threshold set on one half of the families (fires on at most 10% of its claims with nothing missing), tested on the other half, then swapped.

Without a gate: 35 unsafe approvals of claims with missing evidence ($4,285.56 lost if the hidden truth was the violating world), 16 approvals of claims breaking a rule ($4,391.00), 62/144 correct.

| Gate | Threshold | Fires (v1/v2/v3/v4) | Missing-evidence approvals left (of 35) | $ lost, missing evidence | Good approvals held up (of 22) | Rule-broken approvals left | Correct (of 144) |
|---|---|---|---|---|---|---|---|
| no gate | - | 0 (0/0/0/0) | 35 | $4,285.56 | 0 | 16 | 62 |
| v1 probe | frozen | 0 (0/0/0/0) | 35 | $4,285.56 | 0 | 16 | 62 |
| v1 probe | recalibrated | 19 (3/5/8/3) | 26 | $2,718.31 | 3 | 13 | 66 |
| v2 probe L23 | frozen | 0 (0/0/0/0) | 35 | $4,285.56 | 0 | 16 | 62 |
| v2 probe L23 | recalibrated | 29 (4/3/12/10) | 31 | $3,656.17 | 0 | 16 | 66 |
| v2 probe L18 | frozen | 0 (0/0/0/0) | 35 | $4,285.56 | 0 | 16 | 62 |
| v2 probe L18 | recalibrated | 16 (2/5/4/5) | 33 | $3,880.31 | 1 | 15 | 62 |
| length | frozen | 0 (0/0/0/0) | 35 | $4,285.56 | 0 | 16 | 62 |
| length | recalibrated | 53 (12/9/20/12) | 18 | $2,525.42 | 8 | 13 | 71 |
| TF-IDF | frozen | 0 (0/0/0/0) | 35 | $4,285.56 | 0 | 16 | 62 |
| TF-IDF | recalibrated | 12 (3/3/3/3) | 31 | $4,064.56 | 1 | 15 | 65 |
| monitor answer | its YES/NO | 144 (36/36/36/36) | 0 | $0.00 | 22 | 0 | 72 |
| monitor p_yes | recalibrated | 42 (9/12/9/12) | 22 | $3,768.00 | 5 | 13 | 69 |
| cascade: v2 L23 then monitor | recalibrated | 32 (5/5/12/10) | 31 | $3,656.17 | 0 | 16 | 66 |
| oracle (rule checker) | - | 72 (0/0/36/36) | 0 | $0.00 | 0 | 16 | 97 |

Overhead: the probe adds about 0.352 ms per decision on a laptop CPU (one weighted sum over numbers the decision already computes). The monitor adds a whole model call, about 1,322 ms on the T4.
The oracle is the rule-checking code on the structured form: the upper bound when the claim arrives as a form. A probe's case is input that can't be checked by rules.

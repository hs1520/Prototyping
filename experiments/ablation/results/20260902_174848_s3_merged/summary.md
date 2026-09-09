# Ablation campaign `20260902_174848_s3_merged`

- system: AutonomousDrone (frozen requirements, digest `a2f04214fa1e…`)
- provider: vertex  ·  seeds: [0, 1, 2]  ·  commit: `merged: 2026`
- base pipeline kwargs: `{"max_iterations": 4, "dse_mode": "variation", "verbose": false}`  ·  mcts_iterations: 20

## Gate outcomes (all counted runs; genuine failures stay in the denominator)

| arm | runs | qualified | closure CLOSED | 7/7 role paths | reachability | total tokens |
|---|---|---|---|---|---|---|
| DSE-BILEVEL | 3 | 0/3 | 3/3 | 3/3 | 0.633 ± 0.045 | 171,094 ± 47,419 |
| FULL | 3 | 3/3 | 3/3 | 3/3 | 0.935 ± 0.058 | 170,095 ± 48,105 |
| NO-DETFIX | 3 | 3/3 | 3/3 | 3/3 | 0.935 ± 0.058 | 167,850 ± 42,328 |
| NO-DSE | 3 | 3/3 | 3/3 | 3/3 | 0.935 ± 0.058 | 146,946 ± 24,069 |
| NO-REFINE | 3 | 3/3 | 3/3 | 3/3 | 0.935 ± 0.058 | 146,664 ± 13,979 |
| NO-REPAIR | 3 | 3/3 | 3/3 | 3/3 | 0.935 ± 0.058 | 146,716 ± 13,657 |
| NO-SURGICAL | 3 | 3/3 | 3/3 | 3/3 | 0.935 ± 0.058 | 180,848 ± 64,033 |
| SINGLE-SHOT | 3 | 0/3 | 0/3 | 0/3 | — | — |

## Mechanism ledger (successful runs; mean ± std)

Whether the layer an arm removes was exercised. A zero row in FULL means the matching ablation is uninformative on that trajectory.

| arm | refine iters | Tier-0 fixes | Tier-1 syntax LLM | det. connectivity | surgical accepted | surgical exit passes | closure gaps | closure att./acc. | plan det. adds |
|---|---|---|---|---|---|---|---|---|---|
| DSE-BILEVEL | 2.7 ± 1.2 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 2.3 ± 2.3 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.7 ± 1.2 |
| FULL | 2.0 ± 1.7 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 2.0 ± 3.5 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 |
| NO-DETFIX | 2.0 ± 1.7 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 2.0 ± 3.5 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 |
| NO-DSE | 2.0 ± 1.7 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 1.0 ± 1.7 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 |
| NO-REFINE | 1.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 |
| NO-REPAIR | 1.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 |
| NO-SURGICAL | 1.7 ± 1.2 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 |
| SINGLE-SHOT | — | — | — | — | — | — | — | — / — | — |

## Prefix replay (trajectory-matched arms)

| arm | seed | replayed calls | replayed tokens | live calls | live tokens | total tokens |
|---|---|---|---|---|---|---|
| DSE-BILEVEL | 0 | 7 | 148,981 | 1 | 9,707 | 158,688 |
| DSE-BILEVEL | 1 | 7 | 139,292 | 8 | 84,190 | 223,482 |
| DSE-BILEVEL | 2 | 6 | 121,925 | 1 | 9,186 | 131,111 |
| FULL | 1 | 7 | 139,292 | 14 | 84,615 | 223,907 |
| NO-DETFIX | 0 | 7 | 148,981 | 9 | 9,266 | 158,247 |
| NO-DETFIX | 1 | 7 | 139,292 | 14 | 74,862 | 214,154 |
| NO-DETFIX | 2 | 6 | 121,925 | 9 | 9,224 | 131,149 |
| NO-DSE | 0 | 7 | 148,981 | 0 | 0 | 148,981 |
| NO-DSE | 1 | 7 | 139,292 | 4 | 30,641 | 169,933 |
| NO-DSE | 2 | 6 | 121,925 | 0 | 0 | 121,925 |
| NO-REFINE | 0 | 7 | 148,981 | 9 | 9,311 | 158,292 |
| NO-REFINE | 1 | 7 | 139,292 | 8 | 11,254 | 150,546 |
| NO-REFINE | 2 | 6 | 121,925 | 9 | 9,230 | 131,155 |
| NO-REPAIR | 0 | 7 | 148,981 | 9 | 9,243 | 158,224 |
| NO-REPAIR | 1 | 7 | 139,292 | 8 | 11,007 | 150,299 |
| NO-REPAIR | 2 | 6 | 121,925 | 9 | 9,699 | 131,624 |
| NO-SURGICAL | 0 | 7 | 148,981 | 9 | 9,238 | 158,219 |
| NO-SURGICAL | 1 | 7 | 139,292 | 13 | 113,831 | 253,123 |
| NO-SURGICAL | 2 | 6 | 121,925 | 9 | 9,276 | 131,201 |

## Per-arm aggregate (successful runs)

| arm | ok/failed | final_score | controlled 7-scenario pass rate | reachability | LLM calls | total tokens | wall s |
|---|---|---|---|---|---|---|---|
| DSE-BILEVEL | 3/0 | 0.934 ± 0.002 | 1.000 ± 0.000 | 0.633 ± 0.045 | 10.0 ± 4.4 | 171,094 ± 47,419 | 352 ± 401 |
| FULL | 3/0 | 0.960 ± 0.004 | 1.000 ± 0.000 | 0.935 ± 0.058 | 16.3 ± 4.2 | 170,095 ± 48,105 | 1,497 ± 1,344 |
| NO-DETFIX | 3/0 | 0.962 ± 0.004 | 1.000 ± 0.000 | 0.935 ± 0.058 | 17.3 ± 3.2 | 167,850 ± 42,328 | 284 ± 326 |
| NO-DSE | 3/0 | 0.956 ± 0.004 | 1.000 ± 0.000 | 0.935 ± 0.058 | 8.0 ± 2.6 | 146,946 ± 24,069 | 124 ± 176 |
| NO-REFINE | 3/0 | 0.960 ± 0.004 | 1.000 ± 0.000 | 0.935 ± 0.058 | 15.3 ± 0.6 | 146,664 ± 13,979 | 97 ± 5 |
| NO-REPAIR | 3/0 | 0.960 ± 0.004 | 1.000 ± 0.000 | 0.935 ± 0.058 | 15.3 ± 0.6 | 146,716 ± 13,657 | 110 ± 26 |
| NO-SURGICAL | 3/0 | 0.962 ± 0.004 | 1.000 ± 0.000 | 0.935 ± 0.058 | 17.0 ± 2.6 | 180,848 ± 64,033 | 322 ± 397 |
| SINGLE-SHOT | 0/3 | — | — | — | — | — | — |

## Paired per-seed deltas vs FULL (arm − FULL)

| arm | paired seeds | Δ final_score | Δ controlled pass | Δ LLM calls | Δ total tokens |
|---|---|---|---|---|---|
| DSE-BILEVEL | [0, 1, 2] | -0.026 ([-0.0244, -0.0246, -0.0284]) | +0.000 ([0.0, 0.0, 0.0]) | -6.333 ([-5.0, -6.0, -8.0]) | +999.000 ([3573.0, -425.0, -151.0]) |
| NO-DETFIX | [0, 1, 2] | +0.002 ([0.0, 0.0069, 0.0]) | +0.000 ([0.0, 0.0, 0.0]) | +1.000 ([3.0, 0.0, 0.0]) | -2244.667 ([3132.0, -9753.0, -113.0]) |
| NO-DSE | [0, 1, 2] | -0.004 ([-0.0047, -0.0047, -0.0039]) | +0.000 ([0.0, 0.0, 0.0]) | -8.333 ([-6.0, -10.0, -9.0]) | -23148.333 ([-6134.0, -53974.0, -9337.0]) |
| NO-REFINE | [0, 1, 2] | +0.000 ([0.0, 0.0, 0.0]) | +0.000 ([0.0, 0.0, 0.0]) | -1.000 ([3.0, -6.0, 0.0]) | -23430.333 ([3177.0, -73361.0, -107.0]) |
| NO-REPAIR | [0, 1, 2] | +0.000 ([0.0, 0.0, 0.0]) | +0.000 ([0.0, 0.0, 0.0]) | -1.000 ([3.0, -6.0, 0.0]) | -23379.000 ([3109.0, -73608.0, 362.0]) |
| NO-SURGICAL | [0, 1, 2] | +0.002 ([0.0, 0.0069, 0.0]) | +0.000 ([0.0, 0.0, 0.0]) | +0.667 ([3.0, -1.0, 0.0]) | +10753.000 ([3104.0, 29216.0, -61.0]) |
| SINGLE-SHOT | [] | — | — | — | — |

## Failed runs

- **SINGLE-SHOT** seed 0 (arm failure): RuntimeError: terminal functional closure is not closed on the published model revision: REQ_FUNC_001, REQ_FUNC_007, REQ_FUNC_008 (evidence: /Users/huangsongyi/VSCode/Prototyping/experiments/ablation/results/20260902_133704_s3_prefix_matched/failed_runs/SINGLE-SHOT_seed0.evidence.json)
- **SINGLE-SHOT** seed 1 (arm failure): RuntimeError: [SysML_EXTRACTION_ERROR] 未提取到SysML v2 design. (evidence: /Users/huangsongyi/VSCode/Prototyping/experiments/ablation/results/20260902_165930_s1_fix_arms/failed_runs/SINGLE-SHOT_seed1.evidence.json)
- **SINGLE-SHOT** seed 2 (arm failure): RuntimeError: terminal functional closure is not closed on the published model revision: REQ_FUNC_006, REQ_FUNC_008 (evidence: /Users/huangsongyi/VSCode/Prototyping/experiments/ablation/results/20260902_133704_s3_prefix_matched/failed_runs/SINGLE-SHOT_seed2.evidence.json)

## Reading guide

n per cell is small: means ± std and per-seed deltas are **descriptive**; no significance is claimed. Categorical outcomes (failure counts, gate rejections, LLM-call and token costs) carry the load-bearing comparisons; score deltas are indicative only. A genuine failed run stays in the denominator — failure is data, not noise; an infrastructure failure (rate limit, transport) is reported separately and re-run rather than charged to the arm. final_score is measured at different pipeline stages per arm (DSE arms re-score the enriched terminal snapshot; non-DSE arms keep the generate-phase score), so cross-arm score comparisons must use generate_phase_score; final_score differences between DSE-bearing and DSE-less arms reflect the measurement point and the terminal enrichment layer, not generation quality. Under --prefix-from-baseline every non-FULL arm replays FULL's archived call prefix for the same seed and goes live at the first request the ablated component changes: differences are attributable to the component, not to provider sampling; total tokens include the replayed prefix so effort is comparable with FULL, while wall time covers live calls only.

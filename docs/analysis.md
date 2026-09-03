# Methodology & Analysis

This page describes *why* the project is built the way it is, and what the
analysis performed on it actually shows.

## Why PPAC

Recommenders trained with BPR (Bayesian Personalized Ranking) learn to rank
items a user is likely to prefer — but "likely to prefer" is confounded with
"likely to have been shown/rated at all," so popular items get reinforced
regardless of whether they're genuinely a better match. PPAC (Personal
Popularity Aware Counterfactual) addresses this by decomposing an item's
popularity into two signals instead of one:

- **Global Popularity (GP)** — how popular an item is platform-wide. Treated
  as bias: the model learns to predict it (`global_pred` in
  `PPAC_BPRMF._init_model`) and then subtracts it back out at inference
  time via `beta` (negative weight).
- **Personal Popularity (PP)** — how popular an item is among a user's
  Jaccard-similar neighbors (`cal_local_nov` in `src/metrics.py`). Treated
  as a legitimate personalization signal, not bias: added back in via
  `gamma` (positive weight).

The mechanism is in `PPAC_BPRMF._batch_predict`
(`src/algorithm.py`):

```python
scores = scores * (pred_local * pred_global) + gamma * real_local + beta * real_global
```

Two learned predictor networks (`local_pred`, `global_pred`) are trained to
approximate the *real* local/global popularity from the item's embedding
alone (`_compute_loss`'s `reg_loss` term), so at inference time the model can
gate/adjust every candidate item's score by its predicted popularity
profile — not just the items seen during training.

## Splitting protocol

`PPACEqualExposureSplitter` (`src/splitter.py`) reproduces the PPAC paper's
own released split, reverse-engineered from its reference implementation
(`github.com/Stevenn9981/PPAC`) rather than assumed from the paper text. It's
a two-stage process:

1. **Natural per-user holdout** — each user's interactions are shuffled and
   `test_frac`/`validation_frac` of them are held out; the rest becomes
   `train`. This stage alone determines `train` — nothing added later goes
   back into it.
2. **Equal-exposure resample** — within the held-out test pool (and,
   independently, the validation pool), an item is kept only if the pool has
   at least a *quota* of interactions for it, then downsampled to exactly
   that quota. Items below quota are dropped from evaluation entirely, not
   returned to train.

The quota is expressed as `exposure_frac` — a fraction of the *total user
base* — rather than a fixed interaction count, so it represents the same
relative "how popular does an item need to be to survive evaluation" bar
regardless of dataset size. An earlier version used the paper's fixed count
(`exposure_per_item=75`, from ML-1M's ~6,038 users) directly; on Steam's
~50,000 users that same fixed count was a much *looser* relative bar
(1.5% of users vs. 12.4%), letting in a less consensus-driven tier of items
and silently changing what the fairness evaluation measured. Making the
quota proportional (`exposure_frac`) fixed that.

**Why this matters for evaluation**: a naturally popularity-skewed test set
rewards a model for chasing popularity, since most held-out interactions
*are* on popular items — a debiased model gets penalized for behaving
exactly as intended. The equal-exposure test set removes that confound by
construction, which is why the paper (and this project) uses it for
accuracy numbers too, not just fairness ones.

## Models compared

| Model | Role |
|---|---|
| `BPRMF` | recpack implementation of bprmf is used as baseline |
| `PPAC_BPRMF` | Extends recpack's `BPRMF` with the PPAC global/personal counterfactual adjustment described above. |


## Metrics, and where they disagree

- **NDCG@K / Recall@K** — standard ranking accuracy.
- **Item Gini@K** (`calculate_item_gini`) — inequality of *how many times
  each item appears* across all users' top-K lists, zero-padded so
  never-recommended items still count. 0 = every item recommended equally
  often; higher = more concentrated. Lower is better.
- **Item Coverage@K** (`ItemCoverageK`) — fraction of the catalog that
  receives at least one recommendation. Answers "did debiasing widen
  exposure", which Gini alone doesn't: a handful of dominant items sitting
  on top of a long thin tail of once-recommended items can still count as
  full coverage of that tail. Report the two together for that reason.
  Higher is better.
- **PRU@K / PPRU@K** (`calculate_pru`, `calculate_ppru`) — the PPAC paper's
  own bias metrics: negative Spearman correlation between an item's
  popularity (global for PRU, personal for PPRU) and its *rank within a
  single user's list*. Lower PRU is better (less popularity-driven
  re-ranking); PPRU is not "lower is better" in the same sense, since PPAC
  is deliberately optimizing toward personal popularity as a legitimate
  signal.

These two families measure different things, and runs on this project's
Steam pipeline have shown them diverge: PPAC_BPRMF can post a dramatically
lower PRU@K than plain BPRMF (rank-decorrelation from global popularity
working as intended) while its Item Gini stays roughly flat or even
regresses slightly relative to the baseline. The reason isn't a bug — PRU
measures whether popular items get pushed to better *ranks within a user's
own list*; Gini measures how concentrated exposure is *across the whole
catalog and all users*. A model can stop favoring popular items in rank
order without actually pulling new long-tail items into the recommended set
at all — PPAC's `gamma` term re-weights *which already locally-popular
items* get rank 1 vs. rank 8 for a given user, which doesn't necessarily
change the total pool of items being recommended. **Reporting only one of
these metrics would overstate (or understate) PPAC's actual debiasing
effect** — this project reports both for exactly that reason.

## Current status and limitations

See `paper.tex` for the full results, discussion, and limitations writeup.
One thing worth flagging here so it isn't mistaken for a settled
conclusion:

- `gamma=256`/`beta=-128` (`ppac()` in `src/pipeline.py`) are not the
  output of a hyperparameter search we ran ourselves — they're the default
  values Ning et al. report for PPAC, adopted directly because running
  that search to convergence across multiple seeds/splits wasn't feasible
  within the project's time budget (see `paper.tex`'s Limitations
  section). They fall inside the declared `{128, 256, 512}`/
  `{-128, -512, -1024}` search space and, empirically, work well: PPAC-BPRMF
  improves accuracy, fairness, and coverage simultaneously over BPRMF at
  this configuration. A positive `beta` (an earlier, mis-signed
  configuration) amplifies global-popularity bias rather than removing it
  — see `paper.tex` for that failure mode and its effect on results.

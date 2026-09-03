# recommenderSystems

Debiasing recommendations with **PPAC-BPRMF** — an AI Project  applying the *Personal Popularity Aware Counterfactual* (PPAC)
framework to a BPR matrix-factorization recommender, to reduce popularity
bias without sacrificing accuracy.

## Overview

Recommenders trained on implicit feedback tend to over-recommend
already-popular items ("global popularity bias"), which narrows exposure for
niche items and makes recommendations feel generic. This project asks:

> **RQ1**: To what extent does applying the PPAC framework to
> an MF-BPR recommender reduce popularity bias, measured by the Item Gini
> Index, while preserving recommendation accuracy, measured by NDCG@10 and
> Recall@10, for active Steam users (≥10 interactions)?

> **RQ2**: Does the accuracy–fairness relationship observed under RQ1 hold
> under a naturally popularity-skewed (non-equal-exposure) evaluation
> split, or is it dependent on the choice of evaluation protocol? RQ2
> extends RQ1 by testing whether its conclusion is an artifact of one
> particular evaluation design, rather than a property of PPAC itself —
> see `paper.tex`'s Introduction and Discussion for why that distinction
> matters and how it was tested.

**Dataset**: Steam user-game interactions (binarized playtime). MovieLens-1M
was used earlier in the project to calibrate the splitter against the PPAC
paper's own released split files (see `exposure_frac` in `src/splitter.py`);
it is not part of the Steam research question and the pipeline no longer
runs on it.



## Methodology, in brief

- **Splitting**: `PPACEqualExposureSplitter` (`src/splitter.py`) — a
  two-stage, scale-invariant split reproducing the PPAC paper's own released
  protocol: a natural per-user holdout determines train, then the held-out
  pools are resampled so every surviving item reaches the same relative
  popularity bar. This matters because a naturally popularity-skewed test set
  rewards models for chasing popularity regardless of genuine preference.
  Both protocols are evaluated (`SPLITS` in `src/pipeline.py`): the
  `equal_exposure` resample above, and a `natural` weak-generalization
  holdout that keeps the skew. Stage 1 — and therefore the training set — is
  identical between them, so the two differ only in how the test set is
  built.
- **Models compared**: recpack's own `BPRMF` (the un-debiased BPR-MF
  baseline) vs. `PPAC_BPRMF` (that same `BPRMF` extended with PPAC's
  global/personal counterfactual popularity adjustment). `paper.tex` also
  discusses a one-off "PPAC-BPRMF-Reference" diagnostic (the pre-fix
  variant with saturated `f_PP` gradients, see `PPAC_BPRMF`'s docstring in
  `src/algorithm.py`) — that's a supplementary comparison run ad hoc for
  the paper, not a maintained model in this pipeline.
- **Metrics**: NDCG@K / Recall@K (accuracy); Item Gini@K, Item Coverage@K,
  and mean log-popularity of recommended items (exposure fairness); PRU@K /
  PPRU@K (the PPAC paper's own popularity-bias metrics — negative Spearman
  correlation between an item's popularity and its rank in a user's top-K
  list). All defined in `src/metrics.py`.

## Project structure

```
.
├── data                       # Steam / MovieLens-1M interaction data (gitignored)
├── docs                       # mkdocs site (`uv run mkdocs serve`)
├── metrics                    # saved metrics JSON per pipeline run
├── notebooks                  # exploratory data analysis
├── reports/figures            # generated plots (metrics comparison, exposure diagnostics)
├── score_sheets               # grading rubrics for this course project
├── src                        # pipeline code
│   ├── algorithm.py           # PPAC_BPRMF, extending recpack's BPRMF
│   ├── metrics.py             # Item Gini / Item Coverage / PRU@K / PPRU@K, as functions + recpack Metric wrappers
│   ├── splitter.py            # PPACEqualExposureSplitter (two-stage equal-exposure split)
│   ├── preprocessing.py       # PreprocessBlock (binarization, filtering, id mapping)
│   ├── pipeline.py            # Steam pipeline - CLI entrypoint
│   ├── plotting.py            # all matplotlib-producing functions
│   └── utils.py               # data/reporting helpers used by the pipeline
├── tests                      # pytest suite
├── mkdocs.yml
├── pyproject.toml
├── paper.tex                  # results and discussion
└── research-plan.md           # stakeholders, methodology, evaluation design
```

## Getting started

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Run the pipelines

One run trains and evaluates both algorithms (`BPRMF`, `PPAC_BPRMF`) over
every (split protocol, seed) pair — with the defaults, that's 2 algorithms
× 3 seeds × 2 split protocols. Edit `SEEDS` / `SPLITS` in `src/pipeline.py`
to change which and how many are used; with more than one seed, results
are aggregated into mean ± std per (algorithm, split).

```bash
uv run python -m src.pipeline
```

`--optimize` (hyperparameter search) and `--analyze` (+ comparison
plots/report) only support a single seed and a single split at once —
trim `SEEDS`/`SPLITS` to one entry each first, or they raise:

```bash
uv run python -m src.pipeline --optimize
uv run python -m src.pipeline --analyze
```

### Tests, linting, docs

```bash
uv run pytest              # test suite
uv run ruff check src/ tests/   # lint
uv run mkdocs serve        # docs site
```

A pre-commit hook (`.pre-commit-config.yaml`) runs ruff automatically on
commit.

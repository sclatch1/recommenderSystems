# Games Recommendation System

Research project debiasing a BPR-MF recommender with **PPAC** (Personal
Popularity Aware Counterfactual), evaluated on Steam interaction data.
MovieLens-1M was used earlier in the project to calibrate the splitter
against the PPAC paper's own released split files; it is not part of the
Steam research question and the pipeline no longer runs on it.

## Research questions

> **RQ1**: To what extent does applying the PPAC framework to an MF-BPR recommender
> reduce popularity bias — measured by the Item Gini Index — while
> preserving recommendation accuracy — measured by NDCG@10 and Recall@10 —
> for active Steam users (≥10 interactions)?

> **RQ2**: Does the accuracy–fairness relationship observed under RQ1 hold
> under a naturally popularity-skewed (non-equal-exposure) evaluation
> split, or is it dependent on the choice of evaluation protocol? RQ2
> extends RQ1 by testing whether its conclusion is an artifact of one
> particular evaluation design, rather than a property of PPAC itself.

See `research-plan.md` for the full stakeholder/methodology write-up and
`paper.tex` for results and discussion.

## Project layout

```
.
├── data                 # Steam / MovieLens-1M interaction data
├── docs                 # this site
├── metrics              # saved metrics JSON per run (see save_metrics_incremental)
├── notebooks            # exploratory notebooks
├── reports/figures      # generated plots (metrics comparison, exposure diagnostics)
├── src                  # pipeline code - see Code structure below
├── tests                # pytest suite
├── mkdocs.yml
├── pyproject.toml
├── paper.tex
└── research-plan.md
```

## Code structure (`src/`)

| Module | Owns |
|---|---|
| `algorithm.py` | `PPAC_BPRMF` model implementation (extends recpack's own `BPRMF`) |
| `metrics.py` | Popularity-bias metrics (Item Gini, Item Coverage, PRU@K, PPRU@K) as plain functions, plus recpack `Metric` wrappers for use in a `PipelineBuilder` |
| `splitter.py` | `PPACEqualExposureSplitter` — the two-stage, scale-invariant train/val/test split reproducing the PPAC paper's released protocol |
| `preprocessing.py` | `PreprocessBlock` — binarization, filtering, user/item id mapping |
| `pipeline.py` | Steam pipeline (the project's primary dataset) — CLI entrypoint, hyperparameters, `PipelineBuilder` wiring, analysis/report generation |
| `plotting.py` | All matplotlib-producing functions |
| `utils.py` | Data/reporting helpers used by `pipeline.py` |

## Commands

```bash
# Steam pipeline - trains and evaluates BPRMF and PPAC_BPRMF over every
# (split protocol, seed) pair; edit SEEDS/SPLITS in src/pipeline.py to
# change which/how many are used. With >1 seed, results are aggregated
# into mean +/- std per (algorithm, split) - see Methodology & Analysis.
uv run python -m src.pipeline

# --optimize/--analyze only support a single seed and a single split at
# once - trim SEEDS/SPLITS to one entry each first, or they raise.
uv run python -m src.pipeline --optimize   # + hyperparameter search
uv run python -m src.pipeline --analyze    # + comparison plots/report

# tests / lint
uv run pytest
uv run ruff check src/ tests/

# docs
uv run mkdocs serve
```

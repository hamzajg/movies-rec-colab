# ADR 0001 — Initial architecture

Date: 2026-11-13 (baseline recorded retroactively)
Status: accepted (describes what is built)

## Context

A recommender demo sized for Google Colab free tier: must train on ~100k
ratings quickly, be reproducible, and teach evaluation beyond RMSE.

## Decisions

1. **Jupyter notebook as the entry point, Python modules as the logic.**
   `MoviesRec.ipynb` orchestrates; `MovieLens.py` and
   `RecommenderMetrics.py` hold testable logic. Rationale: notebooks are
   the demo surface, but untested notebook code rots — the split keeps CI
   honest.
2. **Surprise (scikit-surprise) as the only ML framework.** SVD +
   KNNBaseline, `train_test_split`, `dump`/`load`. Rationale: KISS — one
   library covers model, eval helpers, and serialisation.
3. **Dataset committed to the repo.** `ml-latest-small/` (movies +
   ratings CSVs) ships in git. Rationale: Colab clones get data for free;
   CI runs with no download step; dataset is small (~1MB).
4. **`MOVIEREC_USERS` env knob for bounded runs.** CI executes the full
   notebook pipeline on 50 users in ~30s instead of minutes. Rationale:
   end-to-end verification without Colab-scale runtime.
5. **Beyond-accuracy metrics included.** Coverage, diversity, novelty,
   hit rate, ARHR alongside RMSE/MAE. Rationale: rating prediction alone
   doesn't measure feed quality — the metric set is the teaching point.
6. **unittest for tests.** `python -m unittest discover` — no pytest
   dependency. Rationale: minimal deps; matches the simple-script nature.

## Consequences

- Notebook-first means logic duplicated into cells will drift from the
  modules — keep cells thin.
- Committed CSVs mean dataset updates are git diffs; fine at this size.
- Colab-specific cells (`!`, `%`) are intentional — local runs skip them.

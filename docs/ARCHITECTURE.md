# Architecture — movies-rec-colab

Current-state map. Descriptive, not aspirational; see `docs/adr/` for why.

## Pipeline

```
ml-latest-small/ (movies.csv, ratings.csv — committed)
        │
        ▼
MovieLens.py ── loadMovieLensLatestSmall() → ratings/movies dataframes,
   │            id↔name maps, popularity ranks; MOVIEREC_USERS=N subset knob
   ▼
MoviesRec.ipynb (orchestration cells)
   ├─ KNNBaseline item-similarity model → diversity metric input
   ├─ 75/25 train/test split → SVD train
   ├─ evaluate: RMSE, MAE (rating) + leave-one-out top-N:
   │     hit rate, ARHR, cumulative hit rate, rating hit rate,
   │     coverage, diversity, novelty
   ├─ refit on full train → predict anti-test → top-10 per user
   └─ surprise.dump → model.pkl → reload → JSON payload
        │
        ▼
RecommenderMetrics.py ── all metric implementations (unit-tested)
tests/ ── unittest for loader + metrics
```

## Design

- **Notebook orchestrates, modules compute.** Anything testable lives in
  `MovieLens.py` / `RecommenderMetrics.py`; notebook cells wire it together.
- **Surprise-native.** Algorithms (`SVD`, `KNNBaseline`), splitting, dump
  format all come from scikit-surprise — no custom model code.
- **Committed dataset.** `ml-latest-small` ships in-repo so Colab and CI
  need no download; `MOVIEREC_USERS` bounds runtime.
- **Evaluation is first-class.** Both rating accuracy and top-N beyond-
  accuracy metrics (beyond-accuracy: coverage/diversity/novelty) are part
  of the deliverable, not an afterthought.

## Outputs

- `model.pkl` — pickled Surprise model (gitignored, regenerated).
- Sample JSON payload of movie ids + predicted ratings (printed/exported
  by the notebook).

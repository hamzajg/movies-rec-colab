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

## Serving (`scorer.py`)

`scorer.py` wraps the same pipeline in a tiny FastAPI app: on startup it
trains the notebook's SVD config (full trainset, `random_state=10`) on the
committed dataset, precomputes each user's top-50 anti-testset
predictions, and serves:

- `GET /health` — user/rating counts
- `GET /` — endpoint info
- `POST /score` — `{userId, candidateIds?, excludeIds?, limit?}` →
  `[{movieId, title, predictedRating}]` filtered from the precomputed
  top-N (candidate/exclude filtering happens at request time, so the
  precompute stays deep enough — `TOP_N_HEADROOM=50`).

Run: `python -m uvicorn scorer:app --port 8090`. `MOVIEREC_USERS=N`
shrinks training for dev — same knob CI uses. Startup is a real SVD fit
(seconds on 100k ratings); there is no committed model dump — the service
always trains fresh (ADR 0002).

## Outputs

- `model.pkl` — pickled Surprise model (gitignored, regenerated).
- Sample JSON payload of movie ids + predicted ratings (printed/exported
  by the notebook).

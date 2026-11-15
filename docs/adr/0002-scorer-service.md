# ADR 0002 — Notebook pipeline as a FastAPI scorer (`scorer.py`)

## Status

Accepted — implemented on `feat/scorer-service`.

## Context

The queued candidate was "wrap the Surprise model behind a tiny scorer —
movies-surprise-flask already proves the shape". Unlike that repo, this
one commits **no model dump** (`model.pkl` is gitignored; the notebook
trains in minutes on Colab). Options were: (a) commit a trained dump —
adds a binary artifact and a retrain workflow this repo deliberately
doesn't have; (b) train at service startup — seconds on 100k ratings,
zero new artifacts.

## Decision

`scorer.py`: a thin FastAPI wrapper that calls the *same* pipeline pieces
the notebook uses — `MovieLens.loadMovieLensLatestSmall()` →
`build_full_trainset()` → `SVD(random_state=10)` — then precomputes
per-user top-50 anti-testset predictions once, so `/score` is an O(limit)
filter, not a model call.

- Response shape `{movieId, title, predictedRating}` — titles come free
  from `MovieLens.movieID_to_name`.
- `candidateIds`/`excludeIds`/`limit` filters applied per request against
  the precomputed list (`TOP_N_HEADROOM=50` > max `limit=50`).
- `MOVIEREC_USERS` env honored — same dev/CI speed knob as the notebook.
- `create_app(service)` factory keeps tests off the real training path
  for unit tests; the e2e tests train a 20-user slice (fast).
- New deps: `fastapi`, `uvicorn[standard]` — justified: the repo had no
  web layer at all.

## Consequences

- Startup cost = one SVD fit (seconds on latest-small); acceptable for a
  demo scorer, would need a dump + lazy-load if this ever served real
  traffic.
- The notebook and the service can drift (e.g. notebook uses KNNBaseline
  for diversity, service doesn't) — they share `MovieLens.py` but not the
  orchestration. Acceptable: the service documents exactly what it
  trains.
- No auth/rate limiting — consistent with movies-surprise-flask's
  unauthenticated scorer shape; add if exposed beyond localhost.

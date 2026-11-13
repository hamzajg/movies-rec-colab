# AGENTS.md — movies-rec-colab

Guidance for AI agents (and humans) working in this repository.

## Project

A movie recommender built with [Surprise](https://surpriselib.com/) in a
Jupyter notebook (`MoviesRec.ipynb`), sized for Google Colab. Trains an SVD
model on the committed MovieLens latest-small dataset, evaluates rating
accuracy (RMSE/MAE) and top-N quality (hit rate, ARHR, coverage, diversity,
novelty), then exports `model.pkl` + a sample JSON payload.
Stack: Python 3.13 (CI), numpy + scikit-surprise, Jupyter/nbconvert.

## Commands

Verified 2026-11-13 against CI (`.github/workflows/ci.yml`) and README.

| Task | Command |
|---|---|
| Install deps | `pip install -r requirements.txt` (scikit-surprise compiles Cython — takes ~1 min) |
| Unit tests | `python -m unittest discover -s tests -t . -v` |
| Run notebook (local) | `jupyter notebook MoviesRec.ipynb` — run from repo root; skip the Colab-only first cells (`!pip`, `!git clone`, `%cd /content/`) |
| CI notebook smoke | `MOVIEREC_USERS=50 jupyter nbconvert --to notebook --execute MoviesRec.ipynb --output /tmp/out.ipynb` (needs `jupyter nbconvert ipykernel`) |

Dataset `ml-latest-small/` is committed — no download needed. `model.pkl`
is written next to the notebook and gitignored.

## Architecture

- `MoviesRec.ipynb` — the pipeline: load data → popularity ranks → KNNBaseline
  similarities → 75/25 split → SVD train → evaluate → refit → predict →
  `surprise.dump` → reload → JSON payload. Colab-oriented (first cells are
  `!`/`%` magics).
- `MovieLens.py` — dataset loader: resolves `ml-latest-small/` paths
  **relative to the .py file** (works in Colab clones and local checkouts),
  maps ids, computes popularity ranks. `MOVIEREC_USERS=N` env limits to
  first N users (how CI runs fast).
- `RecommenderMetrics.py` — RMSE, MAE, hit rate, ARHR, cumulative hit rate,
  rating hit rate, coverage, diversity, novelty + top-N helper.
- `tests/` — unittest for loader + metrics.
- `ml-latest-small/` — movies.csv + ratings.csv (100k ratings).

## Conventions

- Pipeline logic lives in the two .py modules so it's unit-testable; the
  notebook orchestrates. Keep notebook cells thin — push logic into the
  modules.
- Surprise API throughout (`SVD`, `KNNBaseline`, `train_test_split`,
  `dump`/`load`). No other recommender frameworks.
- Tests are `unittest`, run via discover — no pytest.

## Landmines

- **Default branch is `master`, not `main`.**
- First notebook cells are Colab-specific (`!pip install`, `!git clone`,
  `%cd`) — they fail locally; skip them (documented in README).
- `MovieLens.py` historically hard-coded `/content/...` Colab paths —
  paths are now file-relative; keep it that way.
- `name_to_movieID` has fewer entries than titles — duplicate titles
  collapse (documented behaviour, not a bug).
- `MOVIEREC_USERS` env var is the CI speed knob — full run takes minutes.
- `model.pkl` is generated output — never commit it.

## Agent rules

- Branch from `master`; name branches `feat/*`, `fix/*`, `chore/*`, `docs/*`.
- Docs-only changes still go through a PR; never commit to `master` directly.
- Verify before claiming: run `python -m unittest discover -s tests -t . -v`
  (and the nbconvert smoke with `MOVIEREC_USERS=50` for notebook changes)
  and report the real result.
- Keep changes minimal and consistent with surrounding code. No drive-by
  refactors.
- Update AGENTS.md / docs/ARCHITECTURE.md when you change the pipeline
  stages, file layout, or dataset assumptions.

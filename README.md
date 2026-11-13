# movies-rec-colab

A movie recommender built with [Surprise](https://surpriselib.com/) in a
Jupyter notebook, sized for Google Colab. It trains an SVD model on the
MovieLens *latest-small* dataset, evaluates it on rating accuracy and top-N
quality, and exports the fitted model plus a sample recommendation payload.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamzajg/movies-rec-colab/blob/master/MoviesRec.ipynb)

## Run it

**In Colab:** open the badge above and run the cells top to bottom. The
notebook installs `scikit-surprise`, clones this repository into the Colab
workspace and `cd`s into it.

**Locally:** the first cells are Colab-specific (`!pip install`,
`!git clone`, `%cd /content/`). Skip them and, from a checkout, install the
dependencies and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook MoviesRec.ipynb
```

The dataset is already in `ml-latest-small/`, so no download is needed. The
import cell adds the repository root to `sys.path`; when running locally, run
the notebook from the repository root.

## What the notebook does

1. loads the MovieLens ratings and movies (`MovieLens.loadMovieLensLatestSmall`)
2. computes popularity ranks, for the novelty metric
3. fits a `KNNBaseline` item-similarity model, for the diversity metric
4. splits the data 75/25 and trains an `SVD` model
5. evaluates rating accuracy — RMSE and MAE — and top-N quality with
   leave-one-out: hit rate, average reciprocal hit rank, cumulative hit rate,
   rating hit rate, coverage, diversity and novelty
6. refits on the full training set, predicts the anti-test set, and keeps the
   top 10 per user
7. exports the model with `surprise.dump` to `model.pkl`, reloads it, and
   builds a small JSON payload of movie ids and predicted ratings

## Files

```
MoviesRec.ipynb        the notebook: training, evaluation, export
MovieLens.py           loads ml-latest-small, maps ids, computes popularity ranks
RecommenderMetrics.py  RMSE, MAE, hit rate, ARHR, coverage, diversity, novelty,
                       and the top-N helper
ml-latest-small/       the MovieLens latest-small dataset (movies + ratings)
tests/                 unit tests for the loader and the metrics
requirements.txt       numpy, scikit-surprise
```

## Notes

- `scikit-surprise` compiles Cython extensions, so the install takes a minute.
- The notebook trains on the whole small dataset (100k ratings), which is why
  it fits in a free Colab session.
- `model.pkl` is written next to the notebook and is not tracked.
- `MOVIEREC_USERS=50` limits the notebook to the first 50 users, which is how CI
  runs the whole pipeline in about half a minute instead of minutes.

## License

MIT — see [LICENSE](LICENSE).

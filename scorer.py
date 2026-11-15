"""Tiny FastAPI scorer — the notebook pipeline as a service.

Trains the same SVD model the notebook does (full trainset, random_state=10)
on startup, precomputes each user's top recommendations over the anti-testset,
and serves them over HTTP. This is the "notebook → service" increment: the
model stays Surprise-native, the serving layer is deliberately thin.

Run:  python -m uvicorn scorer:app --port 8090
Env:  MOVIEREC_USERS=N limits training users (dev/CI speed knob, same as CI).
"""

import os
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from surprise import SVD

from MovieLens import MovieLens

TOP_N_HEADROOM = 50  # precompute deeper than the max `limit` we'll serve


def build_service(user_limit: int = 0) -> dict:
    """Train SVD on the full trainset and precompute per-user top-N.

    Returns {"top_n": {userId: [(movieId, est), ...]}, "titles": {...}}.
    """
    ml = MovieLens()
    data = ml.loadMovieLensLatestSmall()

    if user_limit:
        users = sorted({int(r[0]) for r in data.raw_ratings})[:user_limit]
        keep = set(users)
        data.raw_ratings = [r for r in data.raw_ratings if int(r[0]) in keep]

    trainset = data.build_full_trainset()
    algo = SVD(random_state=10)
    algo.fit(trainset)

    predictions = algo.test(trainset.build_anti_testset())
    by_user = defaultdict(list)
    for uid, iid, _true, est, _details in predictions:
        by_user[int(uid)].append((int(iid), est))
    for uid, recs in by_user.items():
        recs.sort(key=lambda x: x[1], reverse=True)
        by_user[uid] = recs[:TOP_N_HEADROOM]

    return {
        "top_n": dict(by_user),
        "titles": {int(k): v for k, v in ml.movieID_to_name.items()},
        "users": len(by_user),
        "ratings": len(data.raw_ratings),
    }


class ScoreRequest(BaseModel):
    userId: int
    candidateIds: list[int] | None = None
    excludeIds: list[int] | None = None
    limit: int = Field(default=10, ge=1, le=50)


def recommend(service: dict, user_id: int, candidate_ids, exclude_ids, limit: int):
    """Filter the user's precomputed top-N by candidates/excludes."""
    recs = service["top_n"].get(user_id, [])
    candidates = set(candidate_ids) if candidate_ids else None
    excludes = set(exclude_ids or [])
    out = []
    for movie_id, est in recs:
        if len(out) >= limit:
            break
        if candidates is not None and movie_id not in candidates:
            continue
        if movie_id in excludes:
            continue
        out.append({
            "movieId": movie_id,
            "title": service["titles"].get(movie_id),
            "predictedRating": round(float(est), 4),
        })
    return out


def create_app(service: dict | None = None) -> FastAPI:
    svc = service if service is not None else build_service(
        user_limit=int(os.environ.get("MOVIEREC_USERS", "0")))
    app = FastAPI(title="movies-rec-colab scorer", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "users": svc["users"], "ratings": svc["ratings"]}

    @app.get("/")
    def info():
        return {
            "service": "movies-rec-colab SVD scorer",
            "endpoints": {"score": "POST /score {userId, candidateIds?, excludeIds?, limit?}"},
        }

    @app.post("/score")
    def score(body: ScoreRequest):
        return {
            "userId": body.userId,
            "recommendations": recommend(
                svc, body.userId, body.candidateIds, body.excludeIds, body.limit),
        }

    return app


# Module-level app for `uvicorn scorer:app` — trains at import (seconds on
# the small dataset; MOVIEREC_USERS shrinks it for dev).
app = create_app()

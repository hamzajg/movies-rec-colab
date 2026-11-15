"""Tests for scorer.py — the notebook pipeline behind FastAPI."""

import unittest

from fastapi.testclient import TestClient

from scorer import TOP_N_HEADROOM, build_service, create_app, recommend


class RecommendTests(unittest.TestCase):
    """Pure filtering logic — no model needed."""

    def setUp(self):
        self.service = {
            "top_n": {7: [(10, 4.5), (20, 4.0), (30, 3.5)]},
            "titles": {10: "A", 20: "B", 30: "C"},
            "users": 1,
            "ratings": 3,
        }

    def test_returns_top_n_with_titles(self):
        recs = recommend(self.service, 7, None, None, 2)
        self.assertEqual([r["movieId"] for r in recs], [10, 20])
        self.assertEqual(recs[0]["title"], "A")

    def test_candidate_filter(self):
        recs = recommend(self.service, 7, [30, 20], None, 10)
        self.assertEqual([r["movieId"] for r in recs], [20, 30])

    def test_exclude_filter(self):
        recs = recommend(self.service, 7, None, [10], 10)
        self.assertEqual([r["movieId"] for r in recs], [20, 30])

    def test_unknown_user_gets_empty(self):
        self.assertEqual(recommend(self.service, 999, None, None, 10), [])


class ScorerServiceTests(unittest.TestCase):
    """End-to-end over a tiny trained slice — needs scikit-surprise."""

    @classmethod
    def setUpClass(cls):
        cls.service = build_service(user_limit=20)
        cls.client = TestClient(create_app(cls.service))

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["users"], 20)

    def test_score_returns_predictions(self):
        user_id = next(iter(self.service["top_n"]))
        body = self.client.post("/score", json={"userId": user_id, "limit": 5}).json()
        recs = body["recommendations"]
        self.assertLessEqual(len(recs), 5)
        self.assertTrue(all("movieId" in r and "predictedRating" in r for r in recs))

    def test_score_respects_candidates(self):
        user_id = next(iter(self.service["top_n"]))
        body = self.client.post(
            "/score",
            json={"userId": user_id, "candidateIds": [1, 2, 3], "limit": 50},
        ).json()
        self.assertTrue(all(r["movieId"] in (1, 2, 3)
                            for r in body["recommendations"]))

    def test_headroom_exceeds_served_limit(self):
        self.assertGreater(TOP_N_HEADROOM, 10)


if __name__ == "__main__":
    unittest.main()

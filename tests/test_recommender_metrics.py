import contextlib
import io
import unittest

import numpy as np
from surprise.prediction_algorithms.predictions import Prediction

from RecommenderMetrics import RecommenderMetrics


def prediction(user, movie, actual, estimated):
    """A Surprise Prediction is (uid, iid, r_ui, est, details)."""
    return Prediction(user, movie, actual, estimated, {})


class AccuracyTests(unittest.TestCase):
    def test_mae_is_the_mean_absolute_error(self):
        predictions = [
            prediction(1, 10, 4.0, 3.0),
            prediction(1, 20, 3.0, 3.0),
        ]

        self.assertAlmostEqual(RecommenderMetrics.MAE(predictions), 0.5)

    def test_rmse_is_the_root_mean_squared_error(self):
        predictions = [
            prediction(1, 10, 4.0, 3.0),
            prediction(1, 20, 3.0, 3.0),
        ]

        self.assertAlmostEqual(RecommenderMetrics.RMSE(predictions), np.sqrt(0.5))

    def test_perfect_predictions_score_zero(self):
        predictions = [
            prediction(1, 10, 4.0, 4.0),
            prediction(2, 20, 2.0, 2.0),
        ]

        self.assertEqual(RecommenderMetrics.MAE(predictions), 0.0)
        self.assertEqual(RecommenderMetrics.RMSE(predictions), 0.0)


class TopNTests(unittest.TestCase):
    def test_get_top_n_sorts_by_estimate_and_keeps_the_highest(self):
        predictions = [
            prediction(1, 10, 4.0, 4.5),
            prediction(1, 20, 3.0, 4.9),
            prediction(1, 30, 3.0, 4.1),
            prediction(2, 10, 5.0, 5.0),
        ]

        top_n = RecommenderMetrics.GetTopN(predictions, n=2)

        self.assertEqual(top_n[1], [(20, 4.9), (10, 4.5)])
        self.assertEqual(top_n[2], [(10, 5.0)])

    def test_get_top_n_drops_predictions_below_the_minimum_rating(self):
        predictions = [
            prediction(1, 10, 4.0, 4.9),
            prediction(1, 20, 4.0, 3.5),
        ]

        top_n = RecommenderMetrics.GetTopN(predictions, n=10, minimumRating=4.0)

        self.assertEqual(top_n[1], [(10, 4.9)])

    def test_get_top_n_truncates_to_n(self):
        predictions = [prediction(1, movie, 4.0, 4.0 + movie / 100) for movie in range(1, 6)]

        top_n = RecommenderMetrics.GetTopN(predictions, n=3)

        self.assertEqual(len(top_n[1]), 3)
        self.assertEqual([movie for movie, _ in top_n[1]], [5, 4, 3])

    def test_get_top_n_keys_are_ints(self):
        predictions = [prediction("7", "42", 4.0, 4.5)]

        top_n = RecommenderMetrics.GetTopN(predictions)

        self.assertEqual(list(top_n.keys()), [7])
        self.assertEqual(top_n[7], [(42, 4.5)])


class HitRateTests(unittest.TestCase):
    def setUp(self):
        self.top_n = {1: [(10, 4.9), (20, 4.5)], 2: [(30, 4.8)]}

    def test_hit_rate_counts_left_out_movies_that_are_in_the_top_n(self):
        left_out = [
            prediction(1, 10, 4.0, 4.0),  # in the list
            prediction(2, 30, 4.0, 4.0),  # in the list
            prediction(2, 40, 4.0, 4.0),  # not listed
            prediction(1, 50, 4.0, 4.0),  # not listed
        ]

        self.assertAlmostEqual(RecommenderMetrics.HitRate(self.top_n, left_out), 0.5)

    def test_hit_rate_is_zero_when_nothing_hits(self):
        left_out = [prediction(1, 99, 4.0, 4.0)]

        self.assertEqual(RecommenderMetrics.HitRate(self.top_n, left_out), 0.0)

    def test_cumulative_hit_rate_ignores_ratings_below_the_cutoff(self):
        left_out = [
            prediction(1, 10, 5.0, 4.0),  # liked, hit
            prediction(2, 40, 2.0, 4.0),  # disliked, not counted
        ]

        self.assertEqual(
            RecommenderMetrics.CumulativeHitRate(self.top_n, left_out, ratingCutoff=4.0),
            1.0,
        )

    def test_cumulative_hit_rate_counts_misses_among_liked_ratings(self):
        left_out = [
            prediction(1, 10, 5.0, 4.0),  # liked, hit
            prediction(1, 60, 5.0, 4.0),  # liked, miss
            prediction(2, 40, 2.0, 4.0),  # disliked, ignored
        ]

        self.assertAlmostEqual(
            RecommenderMetrics.CumulativeHitRate(self.top_n, left_out, ratingCutoff=4.0),
            0.5,
        )

    def test_rating_hit_rate_prints_the_hit_share_for_ratings_that_hit(self):
        left_out = [
            prediction(1, 10, 5.0, 4.0),  # 5.0 hit
            prediction(1, 60, 5.0, 4.0),  # 5.0 miss
            prediction(1, 70, 2.0, 4.0),  # 2.0 miss
        ]

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            result = RecommenderMetrics.RatingHitRate(self.top_n, left_out)

        output = printed.getvalue().strip()

        # The method returns nothing and prints only the ratings that produced a
        # hit, so an actual rating with no hits at all never appears.
        self.assertIsNone(result)
        self.assertEqual(output, "5.0 0.5")
        self.assertNotIn("2.0", output)


class ReciprocalHitRankTests(unittest.TestCase):
    def test_reciprocal_rank_uses_the_position_of_the_hit(self):
        top_n = {1: [(10, 4.9), (20, 4.5)], 2: [(30, 4.8)]}
        left_out = [
            prediction(1, 20, 4.0, 4.0),  # hit at rank 2 -> 1/2
            prediction(2, 30, 4.0, 4.0),  # hit at rank 1 -> 1/1
        ]

        # (0.5 + 1.0) / 2
        self.assertAlmostEqual(RecommenderMetrics.AverageReciprocalHitRank(top_n, left_out), 0.75)

    def test_misses_contribute_nothing_but_still_count_in_the_total(self):
        top_n = {1: [(10, 4.9)]}
        left_out = [
            prediction(1, 10, 4.0, 4.0),  # hit at rank 1 -> 1.0
            prediction(1, 99, 4.0, 4.0),  # miss -> 0
        ]

        self.assertAlmostEqual(RecommenderMetrics.AverageReciprocalHitRank(top_n, left_out), 0.5)


class CoverageTests(unittest.TestCase):
    def test_user_coverage_is_the_share_of_users_with_a_recommendation(self):
        top_n = {1: [(10, 4.9)], 2: [(20, 2.0)]}

        self.assertAlmostEqual(RecommenderMetrics.UserCoverage(top_n, numUsers=4), 0.5)

    def test_user_coverage_respects_the_rating_threshold(self):
        top_n = {1: [(10, 4.9)], 2: [(20, 2.0)]}

        self.assertAlmostEqual(
            RecommenderMetrics.UserCoverage(top_n, numUsers=2, ratingThreshold=4.0),
            0.5,
        )


class NoveltyTests(unittest.TestCase):
    def test_novelty_is_the_average_popularity_rank_of_the_recommendations(self):
        top_n = {1: [(10, 4.9), (20, 4.5)], 2: [(10, 4.8)]}
        rankings = {10: 2, 20: 8}

        # (2 + 8 + 2) / 3
        self.assertAlmostEqual(RecommenderMetrics.Novelty(top_n, rankings), 4.0)


class RecommenderDouble:
    """Stand-in for a fitted Surprise algorithm: 1.0 similarity means identical."""

    def __init__(self, similarities):
        self._similarities = np.array(similarities, dtype=float)
        self.trainset = self

    def compute_similarities(self):
        return self._similarities

    def to_inner_iid(self, raw_id):
        return {"10": 0, "20": 1, "30": 2}[raw_id]


class DiversityTests(unittest.TestCase):
    def test_identical_movies_have_zero_diversity(self):
        top_n = {1: [(10, 4.9), (20, 4.5)]}
        sims = RecommenderDouble([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        self.assertAlmostEqual(RecommenderMetrics.Diversity(top_n, sims), 0.0)

    def test_diversity_is_one_minus_the_mean_similarity(self):
        top_n = {1: [(10, 4.9), (20, 4.5)]}
        sims = RecommenderDouble([[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]])

        self.assertAlmostEqual(RecommenderMetrics.Diversity(top_n, sims), 0.75)

    def test_diversity_averages_over_all_pairs_of_all_users(self):
        top_n = {1: [(10, 4.9), (20, 4.5)], 2: [(10, 4.9), (30, 4.5)]}
        sims = RecommenderDouble([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # pairs: (10,20)=0.5 and (10,30)=0.0 -> mean 0.25
        self.assertAlmostEqual(RecommenderMetrics.Diversity(top_n, sims), 0.75)


if __name__ == "__main__":
    unittest.main()

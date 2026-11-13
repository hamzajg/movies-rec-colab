import collections
import csv
import os
import re
import tempfile
import unittest

from MovieLens import MovieLens

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATINGS_CSV = os.path.join(HERE, "ml-latest-small", "ratings.csv")
MOVIES_CSV = os.path.join(HERE, "ml-latest-small", "movies.csv")
YEAR = re.compile(r"(?:\((\d{4})\))?\s*$")


def read_rows(path):
    with open(path, newline="", encoding="ISO-8859-1") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        return list(reader)


class MovieLensTestCase(unittest.TestCase):
    """Loads the committed dataset once; the CSVs are read fresh for expectations."""

    @classmethod
    def setUpClass(cls):
        cls.ml = MovieLens()
        cls.ratings = cls.ml.loadMovieLensLatestSmall()
        cls.movie_rows = read_rows(MOVIES_CSV)
        cls.rating_rows = read_rows(RATINGS_CSV)

    def setUp(self):
        # getMiseEnScene opens a file this repo does not ship, so each test that
        # touches it points the class at its own fixture and restores it after.
        self._mise_path = MovieLens.miseEnScenePath
        self.addCleanup(setattr, MovieLens, "miseEnScenePath", self._mise_path)


class LoadingTests(MovieLensTestCase):
    def test_every_movie_row_is_loaded(self):
        self.assertEqual(len(self.ml.movieID_to_name), len(self.movie_rows))
        self.assertEqual(len(self.ml.movieID_to_name), 9125)

    def test_movie_names_are_read_from_the_csv(self):
        expected = {int(row[0]): row[1] for row in self.movie_rows}

        self.assertEqual(self.ml.movieID_to_name, expected)

    def test_get_movie_name_and_get_movie_id_round_trip(self):
        self.assertEqual(self.ml.getMovieName(1), "Toy Story (1995)")
        self.assertEqual(self.ml.getMovieID("Toy Story (1995)"), 1)

    def test_unknown_ids_and_names_fall_back_to_their_defaults(self):
        self.assertEqual(self.ml.getMovieName(999999), "")
        self.assertEqual(self.ml.getMovieID("Not A Real Movie (1999)"), 0)

    def test_get_user_ratings_returns_only_that_users_ratings(self):
        expected = [
            (int(row[1]), float(row[2])) for row in self.rating_rows if int(row[0]) == 1
        ]

        self.assertTrue(expected)
        self.assertEqual(self.ml.getUserRatings(1), expected)

    def test_get_user_ratings_is_empty_for_a_user_without_ratings(self):
        self.assertEqual(self.ml.getUserRatings(999999), [])


class PopularityRankTests(MovieLensTestCase):
    def setUp(self):
        super().setUp()
        self.rankings = self.ml.getPopularityRanks()
        self.counts = collections.Counter(int(row[1]) for row in self.rating_rows)

    def test_every_rated_movie_is_ranked(self):
        self.assertEqual(set(self.rankings), set(self.counts))

    def test_ranks_are_a_permutation_from_one_to_the_movie_count(self):
        ranks = sorted(self.rankings.values())

        self.assertEqual(ranks, list(range(1, len(self.counts) + 1)))

    def test_the_most_rated_movie_is_rank_one(self):
        most_rated = max(self.counts, key=self.counts.get)

        self.assertEqual(self.ml.getMovieName(most_rated), "Forrest Gump (1994)")
        self.assertEqual(self.rankings[most_rated], 1)

    def test_more_ratings_means_a_better_rank(self):
        # Ties are ordered arbitrarily by the original implementation, so only
        # pairs with different counts are compared.
        ordered = [movie for movie, _ in self.counts.most_common(50)]

        for better, worse in zip(ordered, ordered[1:]):
            if self.counts[better] > self.counts[worse]:
                self.assertLess(
                    self.rankings[better],
                    self.rankings[worse],
                    f"{better} has more ratings than {worse}",
                )


class GenreTests(MovieLensTestCase):
    def setUp(self):
        super().setUp()
        self.genres = self.ml.getGenres()

    def test_every_movie_has_a_bitfield_of_the_same_width(self):
        widths = {len(bitfield) for bitfield in self.genres.values()}

        self.assertEqual(len(widths), 1)
        self.assertEqual(len(self.genres), len(self.movie_rows))

    def test_bitfield_width_matches_the_number_of_distinct_genres(self):
        distinct = {
            genre for row in self.movie_rows for genre in row[2].split("|")
        }

        width = len(next(iter(self.genres.values())))

        self.assertEqual(width, len(distinct))

    def test_set_bits_match_the_genres_listed_in_the_csv(self):
        for row in self.movie_rows:
            movie_id = int(row[0])
            listed = row[2].split("|")

            self.assertEqual(
                sum(self.genres[movie_id]),
                len(listed),
                f"{movie_id} {self.ml.getMovieName(movie_id)}",
            )

    def test_movies_sharing_a_genre_share_its_bit(self):
        toy_story = self.genres[1]  # Adventure|Animation|Children|Comedy|Fantasy
        another = next(
            movie_id
            for movie_id, bitfield in self.genres.items()
            if movie_id != 1 and bitfield.index(1) >= 0 and bitfield[0] == 1
        )

        self.assertEqual(toy_story[0], 1)
        self.assertEqual(self.genres[another][0], 1)


class YearTests(MovieLensTestCase):
    def setUp(self):
        super().setUp()
        self.years = self.ml.getYears()

    def test_year_is_parsed_from_the_title(self):
        self.assertEqual(self.years[1], 1995)

    def test_titles_without_a_trailing_year_are_skipped(self):
        # "Big Bang Theory, The (2007-)" is a range, not a single year.
        self.assertEqual(self.ml.getMovieName(108548), "Big Bang Theory, The (2007-)")
        self.assertNotIn(108548, self.years)

    def test_every_trailing_year_in_the_csv_is_parsed(self):
        expected = {
            int(row[0]): int(YEAR.search(row[1]).group(1))
            for row in self.movie_rows
            if YEAR.search(row[1]).group(1)
        }

        self.assertEqual(dict(self.years), expected)


class MiseEnSceneTests(MovieLensTestCase):
    def write_fixture(self, header, rows):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_reads_seven_visual_features_per_movie(self):
        header = [
            "movieID", "avgShotLength", "meanColorVariance", "stddevColorVariance",
            "meanMotion", "stddevMotion", "meanLightingKey", "numShots",
        ]
        rows = [
            [1, "4.5", "0.1", "0.02", "0.3", "0.05", "0.7", "120"],
            [2, "6.0", "0.2", "0.03", "0.4", "0.06", "0.8", "200"],
        ]
        MovieLens.miseEnScenePath = self.write_fixture(header, rows)

        mes = MovieLens().getMiseEnScene()

        self.assertEqual(mes[1], [4.5, 0.1, 0.02, 0.3, 0.05, 0.7, 120.0])
        self.assertEqual(mes[2], [6.0, 0.2, 0.03, 0.4, 0.06, 0.8, 200.0])
        self.assertTrue(all(isinstance(value, float) for value in mes[1]))

    def test_missing_feature_file_raises(self):
        MovieLens.miseEnScenePath = os.path.join(HERE, "does-not-exist.csv")

        with self.assertRaises(FileNotFoundError):
            MovieLens().getMiseEnScene()


if __name__ == "__main__":
    unittest.main()

import os
import sys

from surprise import Reader

class MovieLens:
    movieID_to_name = {}
    name_to_movieID = {}
    ratingPath = './ml-latest-small/rating.csv'
    moviesPath = './ml-latest-small/movies.csv'

    def loadMovieLensLatestSmall(self):
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}

        reader = Reader(line_format='user item reating timestamp', sep=',', skipline=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    movieID = int(row[0])
                    movieName = row[1]
                    self.movieID_to_name[movieID] = movieName
                    self.name_to_movieID[movieName] = movieID

        return ratingsDataset

import os

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
        print('loadMovieLensLatestSmall')

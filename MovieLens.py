class MovieLens:

  def loadMovieLensLatestSmall(self):
    movieID_to_name = {}
    name_to_movieID = {}
    ratingPath = './ml-latest-small/rating.csv'
    moviesPath = './ml-latest-small/movies.csv'
    print('loadMovieLensLatestSmall')

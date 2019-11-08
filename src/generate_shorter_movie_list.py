import pandas as pd
import numpy as np
import json
from collections import Counter
import sys
import random

'''
To run this: python generate_shorter_movie_list.py <path_to_csv> <num_of_movies_in_output_csv>
'''
def main():
	if len(sys.argv) < 3:
		assert Error("Not enough input arguments.")

	_, input_csv, num_of_movies = sys.argv
	num_of_movies = int(num_of_movies)

	print("Input CSV={}, num_of_movies={}".format(input_csv, num_of_movies))

	# Read csv + get genre column
	X = np.array(pd.read_csv(input_csv))

	idx_array = list(range(X.shape[0]))
	random.shuffle(idx_array)
	X = X[idx_array[:num_of_movies], :]

	header = ["budget","genres","homepage","id","keywords","original_language","original_title","overview","popularity","production_companies","production_countries","release_date","revenue","runtime","spoken_languages","status","tagline","title","vote_average","vote_count"]

	output_csv = "{}_rand_short_{}.csv".format(input_csv[:-4], num_of_movies)
	print("Output CSV=", output_csv)

	pd.DataFrame(X).to_csv(output_csv, header=header, index=None)

if __name__ == "__main__":
	main()

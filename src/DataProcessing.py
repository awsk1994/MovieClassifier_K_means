'''
	0: Budget: int
	1: Genre: JSON
	2: Homepage: url (string)
	3: id: int (don’t need in distance func?)
	4: keywords: JSON

	5: original_language: conv to 1-hot encoding
	original_title: string
	overview: string
	popularity: double
	production_companies: JSON

	10: production_countries: JSON
	release_date: DD-MM-YY -> convert into some n in range(0, 100)
	revenue: int
	runtime: int
	spoken_languages: JSON

	15: status: String
	tagline: string
	title: string
	vote_average: double
	vote_count: int
'''

import pandas as pd
import json
from GenreUtils import GenreUtils
import numpy as np
from sklearn import preprocessing

class DataProcessing:
	@staticmethod
	def normalize(X):
		normalized_X = (X - min(X))/(max(X) - min(X))
		return normalized_X

	@staticmethod
	def normalize_categorical_data(X, num_unique_keys):
		return X/num_unique_keys

	@staticmethod
	# interested_cols = [(<col_idx>, <col_key_identifier(eg. 'id')>)]
	def add_categorical_data(X, interested_cols):
		hot_encoding_lengths = []
		for interested_col in interested_cols:
			hot_encodings, num_unique_keys = GenreUtils.parse_genre(X[:, interested_col[0]], interested_col[1])
			hot_encodings = DataProcessing.normalize_categorical_data(hot_encodings, num_unique_keys) # Normalize distance by dividing by total num unique keys.
			hot_encoding_lengths.append(hot_encodings.shape[1])
			# print("Adding hot encoding | X shape before add={}, encoding col size={}".format(X.shape, hot_encodings.shape[1]))
			X = np.append(X, np.array(hot_encodings), axis=1) # Add genre 1-hot encoding into X.
		return X, hot_encoding_lengths

	@staticmethod
	def parse_data(X):
		# Add numerical data
		X[:, 8] = DataProcessing.normalize(X[:, 8])        # Normalize popularity
		X[:, 18] = DataProcessing.normalize(X[:, 18])      # Normalize Vote Average

		# Add categorical data
		categorical_data_to_add = []
		categorical_data_to_add.append((1,'id')) # Genre (20 cols)
		# categorical_data_to_add.append((4,'id')) # Keywords
		# categorical_data_to_add.append((9,'id')) # Production Companies (5017 cols)
		# categorical_data_to_add.append((10,'iso_3166_1')) # Production Countries (88 cols)
		categorical_data_to_add.append((14,'iso_639_1')) # spoken_languages (62 cols)
		X, hot_encoding_lengths = DataProcessing.add_categorical_data(X, categorical_data_to_add)

		# Add combination features # TODO: find better term for "combination features"
		# Add profit
		budget = X[:, 0]
		revenue = X[:, 12]
		profit = revenue - budget
		profit = np.transpose(np.array([DataProcessing.normalize(profit)]))
		X = np.append(X, profit, axis=1)

		additional_cols = sum(hot_encoding_lengths) + 1 # +1 because we added 'profit'

		return X, additional_cols

	@staticmethod
	def select_interested_attributes(X, additional_cols=0):
		# Select Certain Attributes.
		X_popularity = X[:, 8]
		X_ave = X[:, 18]

		# Choose columns of interest
		# Numeric Attributes.
		col_of_interest = [X_popularity, X_ave]
		col_of_interest = np.transpose(np.array(col_of_interest))

		# Categorical + Combined Attributes.
		start_idx = 20  # Original X only has 20 columns. Additional columns will start on 20th col.
		end_idx = start_idx + additional_cols
		col_of_interest = np.append(col_of_interest, X[:, start_idx:end_idx], axis=1)

		return col_of_interest

	@staticmethod # DataProcessing.py <path_to_csv> <num k_means> <num one-dimensional kmeans>
	def process_input(argv):
		if len(argv) != 4:
			print("ERROR! Please provide 3 input arguments. Format: python main.py </path/to/movies.csv> <k> <init>")
			return

		# Read argv
		_, path_to_csv, K, init = argv

		K = int(K)

		# Read csv
		X = pd.read_csv(path_to_csv).values

		# Get movie_id
		X_movie_id = X[:, 3]

		# Parse Data.
		X, additional_cols = DataProcessing.parse_data(X)

		# Select Interested Attributes
		X = DataProcessing.select_interested_attributes(X, additional_cols)

		return X, K, init, X_movie_id

if __name__ == "__main__":
	X, k, init = DataProcessing.process_input(["main.py", "../movies.csv", 1, 2])

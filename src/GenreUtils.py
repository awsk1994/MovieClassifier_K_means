import json
import numpy as np

class GenreUtils:
	# Generate a hash of name to id, based on all the genres. {<name>: <id>}
	# TODO: Overkill: only need to find the number of unique genres names. Use set.
	@staticmethod
	def create_genre_name_to_id_hash(raw_genres, key='id'):
		genre_name_to_id_hash = {}
		for raw_genre in raw_genres:
			for genre in json.loads(raw_genre):
				genre_name_to_id_hash[genre['name']] = genre[key]
		return genre_name_to_id_hash

	# Generate a hash of name to the index of 1-hot encoding for given name. {<name}: <1-hot-encoding-idx>}
	@staticmethod
	def create_genre_name_to_1hot_idx_hash(genre_name_to_id_hash, genre_names):
		genre_name_to_1hot_idx_hash = {}
		for i in range(len(genre_name_to_id_hash.keys())):
			genre_name_to_1hot_idx_hash[genre_names[i]] = i
		return genre_name_to_1hot_idx_hash

	# Convert a given data's genre into 1-hot-encoding format.
	@staticmethod
	def convert_to_1hot_encoding(raw_genres, genre_name_to_1hot_idx_hash):
		num_unique_genres = len(genre_name_to_1hot_idx_hash.keys())
		hotencoding = [0] * num_unique_genres
		for genre in json.loads(raw_genres):
			name = genre['name']
			idx = genre_name_to_1hot_idx_hash[name]
			hotencoding[idx] = 1
		return np.array(hotencoding)

	# Calculate similarity between two 1-hot-encodings_str
	@staticmethod
	# eg. x = "100", y = "010"
	def calc_genre1hot_similar(x,y):
		sim = 0
		for i in range(len(x)):
			if int(x[i]) == 1 and int(y[i]) == 1:
				sim += 1
		return sim

	@staticmethod
	def parse_genre(raw_genres, key):
		genre_name_to_id_hash = GenreUtils.create_genre_name_to_id_hash(raw_genres, key)
		genre_names = list(genre_name_to_id_hash.keys())
		genre_name_to_1hot_idx_hash = GenreUtils.create_genre_name_to_1hot_idx_hash(genre_name_to_id_hash, genre_names)

		hot_encodings = []
		for raw_genre in raw_genres:
			hot_encoding = GenreUtils.convert_to_1hot_encoding(raw_genre, genre_name_to_1hot_idx_hash)
			hot_encodings.append(hot_encoding)

		hot_encodings = np.array(hot_encodings) # m x # genre
		return hot_encodings, len(genre_names)
		# return genre_name_to_id_hash, genre_names, genre_name_to_1hot_idx_hash, hot_encodings

def test():
	raw_genres = ['[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]',
	             '[{"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 28, "name": "Action"}]',
	             '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 80, "name": "Crime"}]',
	             '[{"id": 28, "name": "Action"}, {"id": 80, "name": "Crime"}, {"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}]',
	             '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 878, "name": "Science Fiction"}]']

	genre_name_to_id_hash, genre_names, genre_name_to_1hot_idx_hash, genre_hot_encodings_str = GenreUtils.parse_genre(raw_genres)

	exp_genre_name_to_id_hash = {'Action': 28, 'Adventure': 12, 'Fantasy': 14, 'Science Fiction': 878, 'Crime': 80, 'Drama': 18, 'Thriller': 53}
	exp_genre_names = ['Action', 'Adventure', 'Fantasy', 'Science Fiction', 'Crime', 'Drama', 'Thriller']
	exp_genre_name_to_1hot_idx_hash = {'Action': 0, 'Adventure': 1, 'Fantasy': 2, 'Science Fiction': 3, 'Crime': 4, 'Drama': 5, 'Thriller': 6}
	exp_genre_hot_encodings_str = ['1111000', '1110000', '1100100', '1000111', '1101000']

	check_genre_name_to_id_hash = (genre_name_to_id_hash == exp_genre_name_to_id_hash)
	check_genre_names = (genre_names == exp_genre_names)
	check_name_to_1hot_idx_hash = (genre_name_to_1hot_idx_hash == exp_genre_name_to_1hot_idx_hash)
	check_hot_encodings_str = (genre_hot_encodings_str == exp_genre_hot_encodings_str)

	print("Test 1 | No Error = ", check_genre_name_to_id_hash and check_genre_names and check_name_to_1hot_idx_hash)
	print("Test 2 | No Error = ", check_hot_encodings_str)

	test3_1 = GenreUtils.calc_genre1hot_similar(genre_hot_encodings_str[0], genre_hot_encodings_str[1]) == 3
	test3_2 = GenreUtils.calc_genre1hot_similar(genre_hot_encodings_str[1], genre_hot_encodings_str[2]) == 2
	test3_3 = GenreUtils.calc_genre1hot_similar(genre_hot_encodings_str[2], genre_hot_encodings_str[3]) == 2
	test3_4 = GenreUtils.calc_genre1hot_similar(genre_hot_encodings_str[3], genre_hot_encodings_str[4]) == 1

	print("Test 3 | No Error = ", test3_1 and test3_2 and test3_3 and test3_4)

if __name__ == "__main__":
	test()

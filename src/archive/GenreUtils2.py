from sklearn.preprocessing import LabelEncoder
import numpy as np

class GenreUtils2:
	@staticmethod
	def to_one_hots(labels):
		# Function to encode output labels from number indexes
		# e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
		labels = labels.reshape(len(labels))
		n_values = np.max(labels) + 1
		return np.eye(n_values)[np.array(labels, dtype=np.int32)]  # Returns FLOATS

	@staticmethod
	def create_label_name_to_one_hots_hash(data):
		labels = []
		for d in data:
			for genre in d:
				labels.append(genre)
		unique_label_names = list(set(labels))

		le = LabelEncoder()
		unique_label_ids = le.fit_transform(unique_label_names)

		one_hots = GenreUtils2.to_one_hots(unique_label_ids)

		label_name_to_one_hots_hash = {}
		for i in range(len(unique_label_names)):
			label_name = unique_label_names[i]
			label_onehot = one_hots[i]
			label_name_to_one_hots_hash[label_name] = label_onehot

		return label_name_to_one_hots_hash

	@staticmethod
	# in: [[0., 0., 1.], [1., 0., 1.]]
	# out: ['001', 101']
	def one_hot_to_str(one_hot):
		s = ""
		for i in range(one_hot.shape[0]):
			s += str(int(one_hot[i]))
		return s

	@staticmethod
	def find_diff_one_hots_str(x, y):
		sim = 0
		for i in range(len(x)):
			if int(x[i]) == 1 and int(y[i]) == 1:
				sim += 1
		d = len(x) - sim
		# print("x={}, y={}. d={}".format(x,y, d))

		return d # len(x) - dim = difference

	@staticmethod
	def str_to_on_hot(str):
		one_hot = []
		for c in str:
			one_hot_id = int(c)
			one_hot.append(one_hot_id)
		return np.array(one_hot)

	@staticmethod
	def single_one_hot_encoding_convert(x, encoding_bit_size, label_name_to_one_hots_hash):
		one_hot = np.array([0.] * encoding_bit_size)
		for genre in x:
			one_hot += np.array(label_name_to_one_hots_hash[genre])
		return one_hot

	@staticmethod
	def multi_one_hot_encoding_str_convert(X):
		label_name_to_one_hots_hash = GenreUtils2.create_label_name_to_one_hots_hash(X)
		encoding_bit_size = len(label_name_to_one_hots_hash[list(label_name_to_one_hots_hash.keys())[0]])

		one_hots_str = []
		for x in X:
			one_hot = GenreUtils2.single_one_hot_encoding_convert(x, encoding_bit_size, label_name_to_one_hots_hash)
			one_hot_str = GenreUtils2.one_hot_to_str(one_hot)
			one_hots_str.append(one_hot_str)

		return np.array(one_hots_str)

def test():
	data = np.array([["action", "horror"], ["action", "comedy", "horror"], ["comedy"], ["fiction"], ["horror", "fiction"], ["action", "fiction"]])
	one_hots_str = GenreUtils2.multi_one_hot_encoding_str_convert(data)

	print("sim={}. x={}, y={}.".format(GenreUtils2.find_diff_one_hots_str(one_hots_str[0], one_hots_str[1]), one_hots_str[0], one_hots_str[1]))
	print("sim={}. x={}, y={}.".format(GenreUtils2.find_diff_one_hots_str(one_hots_str[1], one_hots_str[2]), one_hots_str[1], one_hots_str[2]))
	print("sim={}. x={}, y={}.".format(GenreUtils2.find_diff_one_hots_str(one_hots_str[5], one_hots_str[3]), one_hots_str[5], one_hots_str[3]))
	print("sim={}. x={}, y={}.".format(GenreUtils2.find_diff_one_hots_str(one_hots_str[2], one_hots_str[4]), one_hots_str[2], one_hots_str[4]))


	print(GenreUtils2.str_to_on_hot('111101'))
if __name__ == "__main__":
	test()
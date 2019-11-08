from KMeanspp import KMeanspp
from KMeans import KMeans
from DataProcessing import  DataProcessing
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_data_opt_k(km, kmpp, x):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(x, km, marker="+", label='K-Means')
	ax1.plot(x, kmpp, marker="+", label='K-Means++')

	plt.title("Distance by K")
	plt.xlabel("K")
	plt.ylabel("Distance")
	plt.legend(loc='upper left');
	plt.show()

def plot_data_opt_k2(x, y, algorithm):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(x, y, marker="+", label='K-Means')
	plt.title("Distance by K using {}".format(algorithm))
	plt.xlabel("K")
	plt.ylabel("Distance")
	plt.legend(loc='upper left');
	plt.show()

def plot_data_opt_k3(km, x):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(x, km, marker="+", label='1d k-means')

	plt.title("Distance by K")
	plt.xlabel("K")
	plt.ylabel("Distance")
	plt.legend(loc='upper left');
	plt.show()

def process_input(argv):
	_, path_to_csv = argv

	# Read csv
	X = pd.read_csv(path_to_csv).values

	# Parse Data.
	X, additional_cols = DataProcessing.parse_data(X)

	# Select Interested Attributes
	X = DataProcessing.select_interested_attributes(X, additional_cols)
	return X

def find_optimum_k():
	print("Start find_optimum_k")

	iter_per_K = 5
	start_range, end_range, step = 1, 40, 2
	km_distance_by_K, kmpp_distance_by_K = np.array([]), np.array([])

	# Process Input.
	X = process_input(sys.argv)

	# Try K=1,3,5 ....  29
	K_range = range(start_range, end_range, step)
	print("Will try K = ", list(K_range))
	for K in K_range:
		print("Trying K = ", K)
		# Run k-means and k-means++ 10 times. Take the average of distance. Append to distance_by_K.
		km_dist, kmpp_dist = np.array([]), np.array([])
		for _ in range(iter_per_K):
			km_clusters, km_centroids, km_distances = KMeans.execute(X, K)
			km_clusters, km_centroids, kmpp_distances = KMeanspp.execute(X, K)
			km_dist = np.append(km_dist, km_distances[-1])
			kmpp_dist = np.append(kmpp_dist, kmpp_distances[-1])

		km_ave_dist, kmpp_ave_dist = np.mean(km_dist), np.mean(kmpp_dist)
		print("Average Distance | K={} | km={}, kmpp = {}".format(K, km_ave_dist, kmpp_ave_dist))
		km_distance_by_K, kmpp_distance_by_K = np.append(km_distance_by_K, km_ave_dist), np.append(kmpp_distance_by_K, kmpp_ave_dist)

	print("Plotting Graph")
	# 2. Plot graph for K for km_distance_by_K, kmpp_distance_by_K to pick best K.
	plot_data_opt_k(km_distance_by_K, kmpp_distance_by_K, list(K_range))
	# plot_data_opt_k2(km_distance_by_K, list(K_range), "K-means")
	# plot_data_opt_k2(kmpp_distance_by_K, list(K_range), "K-means++")

	print("Results:")
	print("K range")
	print(list(K_range))
	print("K-Means Distances:")
	print(km_distance_by_K)
	print("K-Means++ Distances:")
	print(kmpp_distance_by_K)

if __name__ == "__main__":
	find_optimum_k()

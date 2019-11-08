from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from Utils import Utils

class KMeans:
	@staticmethod
	def run_k_means(samples, initial_centroids, n_iter):
		current_centroids = initial_centroids
		clusters = []
		distances = []


		# for _ in range(n_iter):
		# 	clusters = KMeans.find_closest_centroids(samples, current_centroids)
		# 	current_centroids = KMeans.get_centroids(samples, current_centroids.shape[0], clusters)
		# 	distance = KMeans.find_cost(samples, current_centroids, clusters)
		# 	distances.append(distance)

		diff_threshold = 0
		count, distance, old_distance = 0, 0, 0
		while count == 0 or (abs(old_distance - distance) > diff_threshold):
			# print("c = {}".format(count))
			count = count + 1

			old_distance = distance
			clusters = KMeans.find_closest_centroids(samples, current_centroids)
			current_centroids = KMeans.get_centroids(samples, current_centroids.shape[0], clusters)
			distance = KMeans.find_cost(samples, current_centroids, clusters)
			distances.append(distance)

		return clusters, current_centroids, distances

	@staticmethod
	def get_centroids(X, K, clusters):
		D = X[0].shape[0]
		centroids = np.array([[None] * D] * K)

		for d in range(D):          # TODO: can this combine with for-loop below?
			centroids[:, d] = 0

		for d in range(D):
			for x_id in range(X.shape[0]):
				x = X[x_id][d]
				centroids[clusters[x_id], d] += x

			for k in range(K):
				if np.sum(clusters == k) == 0:
					continue
				centroids[k, d] /= np.sum(clusters == k)

		return centroids

	@staticmethod
	def find_closest_centroids(X, centroids):
		labels = []
		for x in X:
			distances_to_centroid = []
			for centroid in centroids:
				x_dim_dist = 0
				D = x.shape[0]
				for d in range(D): # dimensions
					dist_func = Utils.eucledian_distance
					x_dim_dist += dist_func(x[d], centroid[d])
				distances_to_centroid.append(x_dim_dist)
			labels.append(np.argmin(distances_to_centroid))
		return np.array(labels)

	@staticmethod
	def find_cost(X, centroids, clusters):
		total_distances = 0
		scalar = 15
		for x_id in range(X.shape[0]):
			x = X[x_id]
			centroid_idx = clusters[x_id]
			centroid = centroids[centroid_idx]

			x_dim_dist = 0
			D = x.shape[0]
			for d in range(D):  # dimensions
				dist_func = Utils.eucledian_distance
				x_dim_dist += dist_func(x[d], centroid[d])
			total_distances += x_dim_dist
		total_distances = total_distances * scalar
		return total_distances

	@staticmethod
	def initialize_centroids(X, K):
		centroids = []
		for _ in range(K):
			centroid = []
			for _ in range(X.shape[1]):
				centroid.append(random.random())
			centroids.append(centroid)
		return np.array(centroids)

	@staticmethod
	def execute(X, K):
		# print("kmeans | start")
		centroids = KMeans.initialize_centroids(X, K)
		clusters, centroids, k_means_dist = KMeans.run_k_means(X, centroids, n_iter=15)
		# Utils.plot_data(X, centroids, clusters) # Only works if there are 2 dimensions.
		return np.array(clusters), centroids, k_means_dist

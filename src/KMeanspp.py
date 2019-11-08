from KMeans import KMeans
import random
import numpy as np
from Utils import Utils

class KMeanspp(KMeans):
	@staticmethod
	def initialize_centroids(X, K):
		centroids = []

		# 1. Pick random point from X as first centroid.
		random_X_idx = random.randint(0, X.shape[0]-1)
		centroids.append(X[random_X_idx])

		# 2. Until all centroids are found.
		for k in range(1,K):
			# 2a. Compute D(x) for all x. D(x) = distance between x and the nearest center that's chosen.
			DX = []
			for x_idx in range(X.shape[0]):
				x = X[x_idx]
				x_to_all_centroids_distance = []

				for centroid in centroids:
					x_to_centroid_distance = 0

					D = x.shape[0]
					for d in range(D):  # dimensions
						dist_func = Utils.eucledian_distance
						x_to_centroid_distance += dist_func(x[d], centroid[d])

					x_to_all_centroids_distance.append(x_to_centroid_distance)
				DX.append(min(x_to_all_centroids_distance))

			total_DX = sum(DX)

			# 2b. Pick the next cluster from X, with the probability (D(x)^2)/(sum(D(x)))
			rand = random.random()
			cum_dist = 0
			for x_idx in range(X.shape[0]):
				x = X[x_idx]
				cum_dist += DX[x_idx]
				prob = cum_dist / total_DX
				if rand <= prob:
					centroids.append(x) # If guess < prob, we chose this point as our next centroid.
					break
		centroids = np.array(centroids)

		# print("K-means++ | Num of centroids chosen =", len(centroids))
		return centroids

	@staticmethod
	def execute(X, K):
		# Initialize Centroids (Cluster center)
		centroids = KMeanspp.initialize_centroids(X, K)
		clusters, centroids, k_means_dist = KMeans.run_k_means(X, centroids, n_iter=15)
		# Utils.plot_data(X, centroids, clusters)
		# Utils.plot_data_3d(samples, current_centroids, clusters)
		return np.array(clusters), centroids, k_means_dist

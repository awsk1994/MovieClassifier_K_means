from KMeans import KMeans
from KMeanspp import KMeanspp
from PCAHelper import PCAHelper
import numpy as np
from OneDKmeans import OneDKmeans
import sys
from DataProcessing import DataProcessing
import pandas as pd
from Utils import Utils


'''
C0              C1: some matrix
   1  2  3
1[ x  1  0
2[ 1  x  1
3[ 0  1 x

Goal:
 - d = 1 if C0(x) == C0(y) and C1(x) =/= C1(y)
 - d = 1 if C0(x) =/= C0(y) and C1(x) == C1(y)

Algo: sum(!C0 == C1) {only need to track top or lower half of matrix as they duplicate, or divide by 2}
'''

# DisagreementDist.py <path_to_csv> <K_in_kmpp> <K_in_1d>

class DisagreementDist:
	@staticmethod
	def processInput(argv):
		if len(argv) != 4:
			print("ERROR! Please provide 4 input arguments. Format: python main.py </path/to/movies.csv> <k_in_kmpp> <k_in_1d>")
			return

		# Read argv
		_, path_to_csv, K_in_kmpp, K_in_1d = argv

		K_in_kmpp = int(K_in_kmpp)
		K_in_1d = int(K_in_1d)

		# Read csv
		X = pd.read_csv(path_to_csv).values

		# Get movie_id
		X_movie_id = X[:, 3]

		# Parse Data.
		X, additional_cols = DataProcessing.parse_data(X)

		# Select Interested Attributes
		X = DataProcessing.select_interested_attributes(X, additional_cols)

		return X, K_in_kmpp, K_in_1d, X_movie_id

	@staticmethod
	def compute(X, K_in_kmpp, K_in_1d):
		kmpp_clusters, kmpp_centroids, kmpp_distances = KMeanspp.execute(X, K_in_kmpp)
		# print("Ran k-means++. End Distance={:.0f}. Clusters = {}.".format(kmpp_distances[-1],kmpp_clusters))
		print("Ran k-means++. End Distance={:.0f}.".format(kmpp_distances[-1]))
		X = PCAHelper.pca_helper(X, 1)
		X.astype(np.float16)
		oneDK_distances, oneDK_clusters, oneDK_centroids = OneDKmeans(X, K_in_1d).run() # KMeans.execute(X, K)
		# print("Ran 1d k-means. End Distance={:.0f}. Clusters = {}.".format(oneDK_distances[-1], oneDK_clusters))
		print("Ran 1d k-means. End Distance={:.0f}.".format(oneDK_distances[-1]))

		n = X.shape[0]
		c0, c1 = np.array([[None] * n] * n), np.array([[None] * n] * n)

		for i in range(K_in_kmpp):
			for j in range(K_in_1d):
				if i <= j:
					continue
				c0[i][j] = kmpp_clusters[i] == kmpp_clusters[j]
				c1[i][j] = oneDK_clusters[i] == oneDK_clusters[j]

		distance = 0
		for i in range(n):
			for j in range(n):
				if i <= j:
					continue
				if c0[i][j] != c1[i][j]:
					distance += 1

		return distance

def main(argv):
	X, K_in_kmpp, K_in_1d, movie_ids = DisagreementDist.processInput(argv)
	dist = DisagreementDist.compute(X, K_in_kmpp, K_in_1d)
	print("Computed Disagreement Distance = {}. datapoints={}, K_in_kmpp={}. K_in_1d={}".format(dist, X.shape[0], K_in_kmpp, K_in_1d))

if __name__ == "__main__":
	main(sys.argv)

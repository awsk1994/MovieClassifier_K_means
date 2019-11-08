import sys
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from KMeans import KMeans
from KMeanspp import KMeanspp
from Utils import Utils
from DataProcessing import DataProcessing

class PCAHelper:
	@staticmethod
	def parse_data(X):
		# 1. Add total_votes column
		X_vote_ave = X[:, 18]
		X_vote_count = X[:, 19]
		X_total_vote = X_vote_ave * X_vote_count
		X_total_vote = np.transpose(np.array([X_total_vote]))

		X = np.append(X, X_total_vote, axis=1)

		# 2. Filter top 250 using total votes
		X = X[list(reversed(X[:, 20].argsort()))]
		X = X[:250, :]

		# 3. assign cluster label with km, km++
		# 3a. Select Cols of interest
		X = DataProcessing.select_interested_attributes(X)
		return X

	@staticmethod
	def pca_helper(X, n_components):
		pca = PCA(n_components=n_components)
		return pca.fit_transform(X)

	@staticmethod
	def main():
		if len(sys.argv) < 3:
			assert Error("need input argument.")

		_, csv_path, K = sys.argv
		K = int(K)
		X = pd.read_csv(csv_path).values
		X = PCAHelper.parse_data(X)   # Steps 1-5

		# k-means and k-means++ execution
		km_clusters, km_centroids, km_distances = KMeans.execute(X, K)
		kmpp_clusters, kmpp_centroids, kmpp_distances = KMeanspp.execute(X, K)
		print("km dist={}, kmpp dist={}".format(km_distances[-1], kmpp_distances[-1]))

		# pca
		X = PCAHelper.pca_helper(X, 2)

		# plot
		Utils.plot_data2(X, K, km_clusters, title="K-means clustering with PCA", xaxis="First Principal Component", yaxis="Second Principal Component")
		Utils.plot_data2(X, K, kmpp_clusters, title="K-means++ clustering with PCA", xaxis="First Principal Component", yaxis="Second Principal Component")

if __name__ == "__main__":
	PCAHelper.main()
import sys
import csv
import numpy as np
from numpy import genfromtxt
import pandas as pd
from KMeans import KMeans
from KMeanspp import KMeanspp
from PCAHelper import PCAHelper
from DisagreementDist import DisagreementDist
from DataProcessing import DataProcessing
from Utils import Utils
from find_optimum_k import plot_data_opt_k3
from OneDKmeans import OneDKmeans

# python source.py </path/to/movies.csv> <k> <init>
# On executing this command, your program should create a file named output.csv in the same directory.
# 1st arg: path to movies.csv
# 2nd arg: number of clusters (int)
# 3rd arg: "random" (kmeans), "k-means++" or "1d"

def main(argv):
	X, K, init, movie_ids = DataProcessing.process_input(argv)

	if init=="random":
		clusters, centroids, distances = KMeans.execute(X, K)
		print("Ran k-means. Start Distance={:.0f}, End Distance={:.0f}. Clusters = {}.".format(distances[0],
		                                                                                       distances[-1],
		                                                                                       clusters))

		Utils.write_output_csv(clusters, "output.csv", movie_ids)
	elif init=="k-means++":
		clusters, centroids, distances = KMeanspp.execute(X, K)
		print("Ran k-means++. Start Distance={:.0f}, End Distance={:.0f}. Clusters = {}.".format(distances[0],
		                                                                                         distances[-1],
		                                                                                         clusters))
		Utils.write_output_csv(clusters, "output.csv", movie_ids)
	elif init=="1d":
		X = PCAHelper.pca_helper(X, 1)
		X.astype(np.float16)
		distances_by_k, cluster, centroids = OneDKmeans(X, K).run()  # KMeans.execute(X, k)
		print("Ran 1d K-means. Distance={}".format(distances_by_k[-1]))
		# plot_data_opt_k3(distances_by_k, list(range(1,K+1)))
		Utils.write_output_csv(cluster, "output.csv", movie_ids)
	else:
		assert Error("init parameter was not inputted correctly!")

if __name__ == "__main__":
	main(sys.argv)

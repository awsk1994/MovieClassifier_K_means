import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

class Utils:
	@staticmethod
	def eucledian_distance(x,y):
		return (x - y)**2 # L2 distance function <-- need to tweek based on input variable

	@staticmethod
	def sum_of_seq(n):
		return (n * (n-1))/2

	@staticmethod
	def plot_data(X, centroids, clusters = None):
		avail_colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k'])    # TODO: add more colors
		colors = np.resize(avail_colors, len(centroids))

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		if clusters is None:
			ax1.scatter(X[0:10, 0], X[0:10, 1], s=10, c='b', marker="X", label='Dataset')
			ax1.scatter(centroids[:, 0], centroids[:, 1], s=50, c='g', marker="o", label='cluster')
		else:
			for c in range(len(centroids)):
				X_by_c = np.array([X[i, :] for i in range(len(X)) if clusters[i] == c])
				print(X_by_c.shape)
				ax1.scatter(X_by_c[:, 0], X_by_c[:, 1], s=10, c=colors[c], marker="X", label='Dataset', alpha=0.5)
				ax1.scatter(centroids[c, 0], centroids[c, 1], s=30, c=colors[c], marker="o", label='cluster', alpha=1)

		offset = 0.1
		ax1.set_xlim([0-offset, 1+offset])
		ax1.set_ylim([0-offset, 1+offset])

		plt.xlabel("Budget")        # TODO: should be dynamic
		plt.ylabel("Popularity")    # TODO: should be dynamic
		plt.legend(loc='upper left');
		plt.show()

	@staticmethod
	def plot_data2(X, num_centroids, clusters, title=None, xaxis=None, yaxis=None):
		avail_colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  # TODO: add more colors
		colors = np.resize(avail_colors, num_centroids)

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		for c in range(num_centroids):
			X_by_c = np.array([X[i, :] for i in range(len(X)) if clusters[i] == c])
			ax1.scatter(X_by_c[:, 0], X_by_c[:, 1], s=10, c=colors[c], marker="X", label='Dataset', alpha=0.5)

		# offset = 0.1
		# ax1.set_xlim([0-offset, 1+offset])
		# ax1.set_ylim([0-offset, 1+offset])

		if title is not None:
			plt.title(title)
		if xaxis is not None:
			plt.xlabel(xaxis)
		if yaxis is not None:
			plt.ylabel(yaxis)
		plt.show()

	@staticmethod
	def plot_data_3d(X, centroids, clusters):
		avail_colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
		colors = np.resize(avail_colors, len(centroids))

		fig = plt.figure()
		ax1 = Axes3D(fig)

		for c in range(len(centroids)):
			X_by_c = np.array([X[i, :] for i in range(len(X)) if clusters[i] == c])
			ax1.scatter(X_by_c[:, 0], X_by_c[:, 1], s=10, c=colors[c], marker="X")
			ax1.scatter(centroids[c, 0], centroids[c, 1], s=30, c=colors[c], marker="o")
		plt.show()

	@staticmethod
	def write_output_csv(clusters, csv_name, movie_ids):
		output = np.append(np.transpose(np.array([movie_ids])), np.transpose(np.array([clusters])), axis=1)
		pd.DataFrame(output).to_csv(csv_name, header=["id", "label"], index=False)

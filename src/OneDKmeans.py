import numpy as np

# cluster, centroids, distances

class OneDKmeans:
	def __init__(self, data, K):
		self.data = data
		self.K = K

		self.n = len(self.data)
		self.D = np.zeros((self.K, self.n))
		self.B = np.zeros((self.K, self.n))

	def ssd(self, l_idx, r_idx):
		array_len = r_idx - l_idx + 1
		median_idx = l_idx + array_len//2
		d = 0
		for i in range(array_len):
			idx = i + l_idx
			add = abs(self.data[idx] - self.data[median_idx])
			d += add
		return d

	def find_j(self, k, i):
		dists = []
		last_zero_idx = k-1

		for j in range(last_zero_idx, i):
			d = self.D[k-1][j] + self.ssd(j+1, i)
			dists.append(d)

		dists = np.array(dists)
		min_idx = np.argmin(dists)
		min_dist = np.min(dists)
		j = min_idx + last_zero_idx + 1
		return j, min_dist

	def fill_remaining_rows(self):
		for k in range(1,self.K):
			# print("ONEDKMeans | fill row ", k)
			for i in range(self.n):
				if i < k:
					self.B[k][i] = 0
					self.D[k][i] = 0
				else:
					j, min_dist = self.find_j(k, i)
					self.D[k][i] = min_dist
					self.B[k][i] = j

	def calculate_clusters(self, K):
		c = []
		c_r = self.n-1
		for k in range(K-1, -1, -1):
			# print("DKMeans | calculate cluster ", k)
			c_l = int(self.B[k][c_r])
			c.append(self.data[c_l:c_r+1])
			c_r = c_l-1
		return list(reversed(c))

	def create_output(self, cluster):
		# Calculate distance
		distances = int(self.D[len(cluster)-1][self.n-1])

		# cluster and centroids
		final_cluster = []
		centroids = []
		for c_idx, c in enumerate(cluster):
			median_idx = len(c)//2
			centroids.append(c[median_idx])
			for _ in range(len(c)):
				final_cluster.append(c_idx)

		return final_cluster, centroids, distances

	def run(self):
		# print("ONEDKMeans | start")
		for i in range(self.D.shape[1]):
			self.D[0][i] = self.ssd(0, i)   # fill first row.
		# print("ONEDKMeans | fill other rows")

		self.fill_remaining_rows()

		distances_by_k = []
		cluster_by_k = []
		centroids_by_k = []
		for k in range(1, self.K + 1):
			cluster = self.calculate_clusters(k)
			cluster, centroids, distances = self.create_output(cluster)
			distances_by_k.append(distances)
			cluster_by_k.append(cluster)
			centroids_by_k.append(centroids)
		return distances_by_k, cluster_by_k[-1], centroids_by_k[-1]
		# return cluster, centroids, distances

if __name__ == "__main__":
	data = [0,3,4,5,9,30,40,60,100]
	K = 9
	distances_by_k, cluster_by_k, centroids_by_k = OneDKmeans(data, K).run()

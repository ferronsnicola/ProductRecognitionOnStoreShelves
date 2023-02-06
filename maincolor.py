import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def main_colors_difference (c1, c2):
	sum = 0
	i=0
	for i in range(len(c1)):
		for j in range(3):
			sum += ((c1[i][0][j] - c2[i][0][j]) * c1[i][1])**2
	return sum


def get_main_colors(img, k=2):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.reshape((img.shape[0] * img.shape[1], 3))
	clt = KMeans(n_clusters=k)  # cluster number
	clt.fit(img)
	hist = find_histogram(clt)
	
	centroids = []
	for i in range(len(clt.cluster_centers_)):
		centroids.append((clt.cluster_centers_[i], hist[i]))
	
	return sorted(centroids, key=lambda centroid: centroid[1], reverse=True)

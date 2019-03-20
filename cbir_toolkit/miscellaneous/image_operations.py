import numpy as np
import cv2
from matplotlib import pyplot as plt

def quantize_image(img, n_colours=64):
	Z = img.reshape((-1, 3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret, label, center = cv2.kmeans(Z, n_colours, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))

	# according to "Image Indexing Using Color Correlograms" paper
	colors_arr = unique(np.array(res))

	return res2, colors_arr


def unique(a):
	"""
		Temporary function to remove duplicates from input list
		Not necessarily to be used outside of the module
	"""
	order = np.lexsort(a.T)
	a = a[order]
	diff = np.diff(a, axis = 0)
	ui = np.ones(len(a), 'bool')
	ui[1:] = (diff != 0).any(axis = 1)

	return a[ui]


def get_neighbors(X, Y, x, y, dist):
	"""
	Temporary function to find pixel neighbors according to various distances
	Not necessarily to be used outside of the module
	"""
	cn1 = (x + dist, y + dist)
	cn2 = (x + dist, y)
	cn3 = (x + dist, y - dist)
	cn4 = (x, y - dist)
	cn5 = (x - dist, y - dist)
	cn6 = (x - dist, y)
	cn7 = (x - dist, y + dist)
	cn8 = (x, y + dist)

	points = (cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8)
	Cn = []

	for i in points:
		# if isValid(X, Y, i):
		Cn.append(i)

	return Cn

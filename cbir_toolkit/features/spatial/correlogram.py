import numpy as np
import cv2
from cbir_toolkit.miscellaneous.image_operations import get_neighbors, quantize_image


def get_correlogram(image, colours, K):
	"""
	Function to get correlogram of an image
	image: OpenCV image numpy array
	"""

	X, Y, t = image.shape
	colorsPercent = []
	for k in K:

		countColor = 0
		color = [0 for i in range(len(colours))]

		for x in range(0, X, int(round(X / 10))):
			for y in range(0, Y, int(round(Y / 10))):

				Ci = image[x][y]
				Cn = get_neighbors(X, Y, x, y, k)
				for j in Cn:
					Cj = image[j[0]][j[1]]

					for m in range(len(colours)):
						if np.array_equal(colours[m], Ci) and np.array_equal(colours[m], Cj):
							countColor = countColor + 1
							color[m] = color[m] + 1

		for i in range(len(color)):
			if countColor != 0:
				color[i] = float(color[i]) / countColor

		colorsPercent.append(color)

	return colorsPercent


def get_autocorrelogram(image, distance_vector=None, levels=64):
	"""
	Function to return the auto-correlogram features of the input image
	:param image: OpenCV BGR Image (uint8 or float32 matrix) to be operated upon
	:param distance_vector: array_like (default=[1,3,5,7])
		vector representing the different distances in which the color distribution is calculated.
	:param levels: Image colour levels in each channel. Default as 256
	:return: straight vector representing the probabilities of occurrence of 64 quantized colors.
		Its total dimension is 64n X 1; where n is the number of different inf-norm distances
	"""

	if distance_vector is None:
		distance_vector = list(range(1, 9, 2))

	correlogram_vector = []
	X, Y, t = image.shape

	# Quantize the image to a lower level for easier computation
	quantized_image, colours = quantize_image(image, levels)

	return get_correlogram(image, colours, distance_vector)



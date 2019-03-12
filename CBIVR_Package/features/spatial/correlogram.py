try:
	import numpy as np
	import cv2
	from matplotlib import pyplot as plt
except ImportError as error:
	print(error.__class__.__name__ + ": " + error.message)


def get_autocorrelogram(image, distance_vector=None, levels=256):
	"""
	Function to return the auto-correlogram features of the input image
	:param image: OpenCV BGR Image (uint8 or float32 matrix) to be operated upon
	:param distance_vector: vector representing the different distances in which the color distribution is calculated.
	:param levels: Image colour levels in each channel. Default as 256
	:return: straight vector representing the probabilities
		of occurrence of 64 quantized colors. Its total dimension is
	    64n X 1; where n is the number of different inf-norm distances
	"""

	correlogram_vector = []
	(X, Y) = image.shape[:2]

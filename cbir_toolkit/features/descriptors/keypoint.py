import numpy as np
from matplotlib import pyplot as plt
import cv2


def get_HOG_descriptors(image, hog=None, window_stride=(8, 8), image_resize_dims=(1024, 1024)):
	"""
	Function to get the Histogram of Oriented Gradient features of an image
	:param image: OpenCV RGB Image of ndarray like
	:param hog: Object of type cv2.HOGDescriptor
	:param window_stride: int array_like of length 2
		Window size of local histogram computation
	:param image_resize_dims: int array_like of length 2
		Image dimensions to resize the image to, if required
	:return: ndarray_like
		HOG feature descriptor array
	"""

	if not hog:
		hog = cv2.HOGDescriptor()

	image = cv2.resize(image, image_resize_dims, interpolation=cv2.INTER_AREA)
	descp = hog.compute(image, winStride=window_stride, padding=None)

	if descp is None:
		descp = []
	else:
		descp = descp.ravel()

	return descp


def get_SIFT_features(image, image_size=(1024, 1024), plot_and_return_keypoints=False):
	"""
	Function to get Scale Invariant Feature Transform of an image
	:param image: OpenCV image of ndarray_like
	:param image_size: array_like
		Image dimensions to resize image to for maintaining uniformity in features
	:param plot_and_return_keypoints: Boolean, optional
	:return: matrix_like (and OpenCV image ndarray_like)
		Returns descriptors along with an image with kepypoints drawn depending on the boolean value of plot_and_return_keypoints
	"""
	sift = cv2.xfeatures2d.SIFT_create()
	image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
	keypoints, descriptors = sift.detectAndCompute(image, None)

	if plot_and_return_keypoints:
		keypoint_image = cv2.drawKeypoints(image, keypoints, None)

		plt.imshow(keypoint_image)
		plt.title("Image with SIFT keypoints")
		plt.show()

		return keypoint_image, descriptors
	else:
		return descriptors


def get_SURF_features(image, image_size=(1024, 1024), plot_and_return_keypoints=False):
	"""
	Function to get Speeded-Up Robust Features of an image
	:param image: OpenCV image of ndarray_like
	:param image_size: array_like
		Image dimensions to resize image to for maintaining uniformity in features
	:param plot_and_return_keypoints: Boolean, optional
	:return: matrix_like (and OpenCV image ndarray_like)
		Returns descriptors along with an image with kepypoints drawn depending on the boolean value of plot_and_return_keypoints
	"""
	surf = cv2.xfeatures2d.SURF_create()
	image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
	keypoints, descriptors = surf.detectAndCompute(image, None)

	if plot_and_return_keypoints:
		keypoint_image = cv2.drawKeypoints(image, keypoints, None)

		plt.imshow(keypoint_image)
		plt.title("Image with SURF keypoints")
		plt.show()

		return keypoint_image, descriptors
	else:
		return descriptors


def get_ORB_features(image, num_features=500, scaleFactor=1.2, nlevels=8, score_type=None, image_size=(1024, 1024), plot_and_return_keypoints=False):
	"""
		Function to get  of an image
		:param image: OpenCV image of ndarray_like
		:param num_features: int, optional (Default = 500)
			The maximum number of features to retain
		:param scaleFactor: float, optional (Default = 1.2)
			Pyramid decimation ratio, greater than 1.
			scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous,
			but such a big scale factor will degrade feature matching scores dramatically.
			On the other hand, too close to 1 scale factor will mean that to cover certain scale range
			you will need more pyramid levels and so the speed will suffer.
		:param nlevels: int, optional(Default = 8)
			The number of pyramid levels. The smallest level will have linear size equal to
			input_image_linear_size/pow(scaleFactor, nlevels).
		:param score_type: (Default: HARRIS_SCORE)
			Algorithm to be used to compute rank of the features
		:param image_size: array_like
			Image dimensions to resize image to for maintaining uniformity in features
		:param plot_and_return_keypoints: Boolean, optional
		:return: matrix_like (and OpenCV image ndarray_like)
			Returns descriptors along with an image with kepypoints drawn depending on the boolean value of plot_and_return_keypoints
		"""
	if score_type == None:
		orb = cv2.ORB_create(num_features, scaleFactor, scoreType=cv2.ORB_HARRIS_SCORE)
	else:
		orb = cv2.ORB_create(num_features, scaleFactor, scoreType=score_type)
	image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
	keypoints, descriptors = orb.detectAndCompute(image, None)

	if plot_and_return_keypoints:
		keypoint_image = cv2.drawKeypoints(image, keypoints, None)

		plt.imshow(keypoint_image)
		plt.title("Image with SURF keypoints")
		plt.show()

		return keypoint_image, descriptors
	else:
		return descriptors

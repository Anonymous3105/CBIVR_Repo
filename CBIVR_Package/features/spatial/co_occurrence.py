try:
	import numpy as np
	import cv2
	from skimage.feature import texture
except ImportError as error:
	print(error.__class__.__name__ + ": " + error.message)


def get_GLCM_features(image, distances=[0], angles=None, levels=256, symmetric=True, normed=True, features=None):
	"""
	Function to return features extracted from the gray level co-occurrence matrix of an image
	:param image: OpenCV numpy array_like of uint8
	:param distances: array_like
		List of pixel pair distance offsets
	:param angles: array_like
		List of pixel pair angles in radians.
	:param levels: int, optional
		The input image should contain integers in [0, levels-1],
		where levels indicate the number of grey-levels counted (typically 256 for an 8-bit image). Default= 256.
	:param symmetric: bool, optional
		If True, the output matrix P[:, :, d, theta] is symmetric.
		This is accomplished by ignoring the order of value pairs,
		so both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. Default= False.
	:param normed: bool, optional
		If True, normalize each matrix P[:, :, d, theta] by dividing by
		the total number of accumulated co-occurrences for the given offset.
		The elements of the resulting matrix sum to 1. Default= False.
	:param features: array_like
		The list of desired features that can be extracted from GLCM matrix.
		Accepted values for array elements include:
			"energy", "contrast", "homogeneity", "ASM", "dissimilarity", "correlation", "entropy"
	:return: GLCM feature dictionary containing feature names as keys and the corresponding feature values
		Features included - Energy, Contrast, Homogeneity, Entropy, ASM, Dissimilarity, Correlation
	"""

	if angles is None:
		angles = [0, np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4]

	if features is None:
		features = ["energy", "contrast", "homogeneity", "ASM", "dissimilarity", "correlation", "entropy"]
	else:
		accepted_features = ["energy", "contrast", "homogeneity", "ASM", "dissimilarity", "correlation", "entropy"]
		for f in features:
			if f not in accepted_features:
				raise Exception("Feature " + f + "is not accepted in the set of features")
				return None

	image_glcm = texture.greycomatrix(image, distances, angles, levels=levels, symmetric=symmetric, normed=normed)

	output_features = dict()
	for feature in features:
		if feature == "entropy":
			entropy = np.zeros((1, 4))
			for i in range(image_glcm.shape[0]):
				for j in range(image_glcm.shape[1]):
					entropy -= image_glcm[i, j] * np.ma.log(image_glcm[i, j])
			output_features[feature] = entropy
		else:
			output_features[feature] = texture.greycoprops(image_glcm, feature)

	return output_features


def get_LBP(image, P, R, method=None):
	"""
	Function to get Local Binary Pattern of an image based on given parameters
	:param image: OpenCV array_like of uint8
	:param P: int
		Number of circularly symmetric neighbour set points (quantization of the angular space).
	:param R: float
		Radius of circle (spatial resolution of the operator).
	:param method: {‘default’, ‘ror’, ‘uniform’, ‘var’}
		Method to determine the pattern.
			‘default’: original local binary pattern which is gray scale but not rotation invariant.
			‘ror’: extension of default implementation which is gray scale and rotation invariant.
			‘uniform’: improved rotation invariance with uniform patterns and
				finer quantization of the angular space which is gray scale and rotation invariant.
			‘var’: rotation invariant variance measures of the contrast of local image texture
				which is rotation but not gray scale invariant.
	:return: LBP image array_like
	"""

	return texture.local_binary_pattern(image, P, R, method)

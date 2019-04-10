import numpy as np
from skimage.feature import texture as sktex
import cv2
import mahotas.features as mfeats
import mahotas.texture as mtex


def get_GLCM_features(image, distances=(0), angles=None, levels=256, symmetric=True, normed=True, features=None):
	"""
	Function to return features extracted from the gray level co-occurrence matrix of an image
	:param image: OpenCV numpy array_like of uint8
	:param distances: array_like, object
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

	image_glcm = sktex.greycomatrix(image, distances, angles, levels=levels, symmetric=symmetric, normed=normed)

	output_features = dict()
	for feature in features:
		if feature == "entropy":
			entropy = np.zeros((1, 4))
			for i in range(image_glcm.shape[0]):
				for j in range(image_glcm.shape[1]):
					entropy -= image_glcm[i, j] * np.ma.log(image_glcm[i, j])
			output_features[feature] = entropy
		else:
			output_features[feature] = sktex.greycoprops(image_glcm, feature)

	return output_features


def get_LBP(image, points, radius, method="default"):
	"""
	Function to get Local Binary Pattern of an image based on given parameters
	:param image: OpenCV array_like of uint8
	:param points: int
		Number of circularly symmetric neighbour set points (quantization of the angular space).
	:param radius: float
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

	Example: get_LBP(image, radius=8, points=6)
	"""
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return mfeats.lbp(image, points, radius, method)


def get_cooccurrence_matrix(image, direction, symmetric=False, distance=1):
	"""
	Computes and returns the gray-level co-occurrence matrix
	:param image: ndarray of uint8
	:param direction: int
		Direction of computation of GLCM
			{ 0:horizontal(default), 1:diagonal(leading), 2:vertical, 3:diagonal(counterdiagonal) }
	:param symmetric: boolean, optional (Default=False)
		Whether to return a symetric matrix or not
	:param distance: int, optional (Default=1)
		Distance between pixel pairs
	:return: co-occurence matrix of uint8
	"""
	if image.dtype != np.dtype("int64"):
		image = np.int64(image)

	if len(image.shape) == 2 and not (0 <= direction < 4):
		raise ValueError('mahotas.texture.cooccurence: `direction` {0} is not in range(4).'.format(direction))
	elif len(image.shape) == 3 and not (0 <= direction < 13):
		raise ValueError('mahotas.texture.cooccurence: `direction` {0} is not in range(13).'.format(direction))
	elif len(image.shape) not in (2, 3):
		raise ValueError('mahotas.texture.cooccurence: cannot handle images of %s dimensions.' % len(image.shape))

	return mtex.cooccurence(image, direction, output=None, symmetric=symmetric, distance=distance)


def get_haralick_features(image, ignore_zeros=False, get_14th_feature=False, distance=1):
	"""
	Function to return Haralick features for given cooccurrence matrices
	:param image: OpenCV image ndaaray of dtype int64
	:param ignore_zeros: boolean, optional (Default: False)
		Can be used to have the function ignore any zero-valued pixels (as background).
		If there are no-nonzero neighbour pairs in all directions, an exception is raised.
	:param get_14th_feature: boolean, optional (Default: False)
	:param distance: int, optional (Default: 1)
		Distance between pixel pairs
	:return: ndarray of np.double
		A 4x13 or 4x14 feature vector (one row per direction) if `f` is 2D, 13x13 or 13x14 if it is 3D.
		The exact number of features depends on the value of "compute_14th_feature"
	"""

	return mfeats.haralick(image,
							ignore_zeros=ignore_zeros,
							compute_14th_feature=get_14th_feature,
							distance=distance
	)

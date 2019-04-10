import numpy as np


def get_cosine_similarity(d1, d2):
	"""
		Function to find the cosine similarity value between 2 vectors
		Parameters:
			 d1, d2: numpy ndarray
	"""
	try:
		return np.round(np.sum(d1.T.dot(d2) / (np.sqrt(d1.T.dot(d1) * np.sqrt(d2.T.dot(d2))))), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_pearson_correlation_coefficient(d1, d2):
	"""
		Function to find the cosine similarity value between 2 vectors
		Parameters:
			d1, d2: numpy ndarray
	"""
	try:
		return np.fabs(np.average(np.corrcoef(d1, d2)))
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def hist_intersection(image1, image2):
	"""
	Function for calculating histogram intersection between two image
	A, B:
		numpy ndarray for image
	"""
	min_matrix = np.where(image1 >= image2, image2, 0) + np.where(image1 < image2, image1, 0)
	the_min = min_matrix / float(min(np.sum(image1.ravel()), np.sum(image2.ravel())))

	return sum(the_min.ravel())


def jaccard_similarity(vec1, vec2):
	"""
	Function to compute the Jaccard similarity between two boolean vectors
	:param vec1: numpy ndarray_like
	:param vec2: numpy ndarray_like
	"""

	if vec1.shape != vec2.shape:
		raise ValueError("Shape mismatch: vec1 and vec2 must have the same shape.")

	intersection = np.logical_and(vec1, vec2)
	union = np.logical_or(vec1, vec2)
	return intersection.sum() / float(union.sum())


similarity_metric_mapper = {
	"cosine": get_cosine_similarity,
	"pearson": get_pearson_correlation_coefficient,
	"hist": hist_intersection,
	"jaccard": jaccard_similarity

}


def get_similarity(vec1, vec2, method="cosine", addn_params={}):
	"""
	:param vec1: ndarray_like
		First vector in the computation
	:param vec2: ndarray_like
	:param method: string (Default: cosine)
	:param addn_params: dict like, optional
		Additional parameters to be used in special cases
	:return: float
		The distance between the 2 vectors
	"""
	if method not in similarity_metric_mapper.keys():
		raise KeyError("Method not found in the list of defined metrics")
	else:
		similarity_function = similarity_metric_mapper[method]
		return similarity_function(vec1, vec2)



def get_metric_choices():
	return similarity_metric_mapper.keys()
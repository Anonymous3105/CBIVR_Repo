import numpy as np


def get_sum_absolute_difference(d1, d2):
	"""
		Function to find the sum of absolute differences between 2 vectors
		Parameters:
			 d1, d2: numpy ndarray

	"""
	try:
		return np.round(np.sum(np.fabs(d1) - np.fabs(d2)), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_sum_absolute_square_difference(d1, d2):
	"""
		Function to find the sum of squares of absolute differences between 2 vectors
		Parameters:
			 d1, d2: numpy ndarray
	"""
	try:
		return np.round(np.sum((np.fabs(d1) - np.fabs(d2)) ** 2), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_euclidean_distance(d1, d2):
	"""
		Function to find the Euclidean distance between 2 vectors
		Parameters:
			 d1, d2: numpy ndarray
	"""
	try:
		return np.round(np.sqrt(np.sum((d1 - d2) ** 2)), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_city_block_distance(d1, d2):
	"""
		Function to find the City-Block or Manhattan distance between 2 vectors
		Parameters:
			 d1, d2: numpy ndarray
	"""
	try:
		return np.round(np.sum(np.fabs(d1 - d2)), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_canberra_distance(d1, d2):
	"""
		Function to find the Canberra distance between 2 vectors
		Parameters:
			 d1, d2: numpy ndarray
	"""
	try:
		return np.round(np.sum(np.fabs(d1 - d2) / (np.fabs(d1) + np.fabs(d2))), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_maximum_value_distance(d1, d2):
	"""
		Function to find the Maximum value distance between 2 vectors
			Parameters:
				 d1, d2: numpy ndarray

	"""
	try:
		return np.round(np.sum(np.fabs(d1 - d2)), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_minkowski_distance(d1, d2, p=2):
	"""
		Function to find the Minkowski distance between 2 vectors
			Parameters:
				 :param d1: numpy ndarray
				 :param d2: numpy ndarray
				 :param p: order of the norm of the difference

	"""
	try:
		return np.round(np.power(np.sum(np.power(np.fabs(d1 - d2), p)), 1 / p), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_chi_square_distance(d1, d2):
	"""
		Function to find the Chi-squared distance between 2 vectors
			Parameters:
			d1, d2: numpy ndarray


	"""
	try:
		return np.round(np.sum((d1 - d2) ** 2 / (d1 + d2)), 4)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_hamming_distance(d1, d2):
	"""
		Function to find the hamming distance between 2 vectors
			Parameters:
				d1, d2: binary numpy ndarray
	"""
	try:
		return np.average(d1 != d2)
	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


def get_wasserstein_distance(u_values, v_values, p=2):
	"""
		Function to find the wasserstein or earth mover's distance between 2 vectors
			Parameters:
				u_values, v_values: numpy array_like
					Values observed in the (empirical) distribution.
				p: order of the norm of the difference
				:param u_values: numpy array_like
				:param v_values: numpy array_like
				:param p: order of the norm of the difference
	"""
	try:
		u_sorter = np.argsort(u_values)
		v_sorter = np.argsort(v_values)

		all_values = np.concatenate((u_values, v_values))
		all_values.sort(kind='mergesort')

		# Compute the differences between pairs of successive values of u and v.
		deltas = np.diff(all_values)

		# Get the respective positions of the values of u and v among the values of both distributions.
		u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
		v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

		u_cdf = u_cdf_indices / u_values.size
		v_cdf = v_cdf_indices / v_values.size

		return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p), deltas)), 1 / p)

	except Exception as exception:
		# Output unexpected Exceptions.
		print(exception, False)
		print(exception.__class__.__name__ + ": " + str(exception))


distance_metric_mapper = {
		"euclidean": get_euclidean_distance,
		"city-block": get_city_block_distance,
		"canberra": get_canberra_distance,
		"maximum-value-distance": get_maximum_value_distance,
		"minkowski": get_minkowski_distance,
		"chi-square": get_chi_square_distance,
		"hamming": get_hamming_distance,
		"wasserstein": get_wasserstein_distance
}


def get_distance(vec1, vec2, method="euclidean", addn_params=None):
	"""
	Wrapper funtion to get the distance between two vectors using the given method
	:param vec1: ndarray_like
		First vector in the computation
	:param vec2: ndarray_like
	:param method: string (Default: euclidean)
		Distance metric function to invoke.
		Can be any one of "euclidean", "city-block", "canberra", "maximum-value-distance",
						"minkowski", "chi-square", "hamming", "wasserstein"
	:param addn_params: dict like, optional
		Additional parameters to be used in special cases
	:return: float
		The distance between the 2 vectors
	"""

	if addn_params is None:
		addn_params = {'p': 2}
	if method not in distance_metric_mapper.keys():
		raise KeyError("Method not found in the list of defined metrics")
	else:
		distance_function = distance_metric_mapper[method]
		if method in ["minkowski", "wasserstein"]:
			return distance_function(vec1, vec2, addn_params['p'])
		else:
			return distance_function(vec1, vec2)


def get_metric_choices():
	return distance_metric_mapper.keys()
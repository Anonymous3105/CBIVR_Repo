try:
	import numpy as np
except ImportError as error:
	print(error.__class__.__name__ + ": " + error.message)
	raise ImportError("Python module Numpy not installed")


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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)


def get_chi_squared_distance(d1, d2):
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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)

try:
	import numpy as np
except ImportError as error:
	print(error.__class__.__name__ + ": " + error.message)
	raise ImportError("Python module Numpy not installed")


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
		print(exception.__class__.__name__ + ": " + exception.message)


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
		print(exception.__class__.__name__ + ": " + exception.message)

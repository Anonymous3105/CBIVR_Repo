import numpy as np
import cv2


def compare_histogram(hist1, hist2, method="chisqr"):
	"""
	Function to compare histograms of 2 images using the given algorithm
	:param hist1: ndarray_like
		First histogram to be compared
	:param hist2: ndarray_like
		Second compared histogram of the same size as hist1
	:param method: string
		Comparison algorithm. (Default: "chisqr")
		Can be any of
			"correlation" -> Correlation
			"chisqr" -> Chi-Square
			"chisqr_alt" -> Alternate Chi-Square
			"intersection" -> Intersection
			"bhattacharya" -> Bhattacharya Distance
			"kl_div" -> Kullback-Leibler divergence
	:return: Return value of the algorithm
	"""

	dist_metric = {
		"correlation": cv2.CV_COMP_CORREL,
		"chisqr": cv2.CV_COMP_CHISQR,
		"chisqr_alt": cv2.CV_COMP_CHISQR_ALT,
		"intersection": cv2.CV_COMP_INTERSECT,
		"bhattacharya": cv2.CV_COMP_BHATTACHARYYA,
		"kl_div": cv2.CV_COMP_KL_DIV
	}

	return cv2.compareHist(hist1, hist2, dist_metric[method])

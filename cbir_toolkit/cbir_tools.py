import cv2
import numpy as np

from glob import glob
from os.path import isdir, expanduser

from cbir_toolkit.metrics.distances import get_metric_choices as dis_choices, get_distance
from cbir_toolkit.metrics.similarity import get_metric_choices as sim_choices, get_similarity


distance_metrics = dis_choices()
similarity_metrics = sim_choices()


def get_metric_function(method):
	if method in distance_metrics:
		return get_distance, "asc"
	elif method in similarity_metrics:
		return get_similarity, "desc"
	else:
		raise KeyError(
			"Given metric does not match known metrics. Choose any of: " + (distance_metrics + similarity_metrics))


def load_images_from_dir(files_dir):
	"""
	Function to load images from a directory of files
	:param files_dir: string
		Relative or absolute path of the image directory
	:return: dict like
		Dictionary with image filenames as keys and image OpenCV arrays as values.
	"""

	if not isdir(expanduser(files_dir)):
		raise NotADirectoryError("Given directory does not exists.")
	else:
		images = {}
		global_dir_path = expanduser(files_dir)

		image_file_extensions = ["jpg", "jpeg", "png", "bmp", "gif", "tiff"]
		files = []

		for ext in image_file_extensions:
			files.extend(glob(global_dir_path + "/*." + ext))

		for imagePath in files:
			filename = imagePath[imagePath.rfind("/") + 1:]
			image = cv2.imread(imagePath)
			images[filename] = image

		return images


class Ranker:

	def __init__(self, index, num_of_result=None, method="euclidean", ranking_order=None, addn_params=None):
		"""
		Initializing function
		:param index: dict like
			A dictionary with key as filename of the image and values as corresponding feature vector as a numpy array.
		:param num_of_result: int like, optional
			The number of output feature images to be given after ranking. If None all image are returned
		:param method: string, optional
			The method to be used to rank the images
		:param ranking_order: string, optional
			The order of ranking. Can be 'asc', 'desc' or None, where the function will choose on the basis of the metric
		:param addn_params: dict like, optional
			Additional parameters to be used in defined metrics. Like 'p" is minkowski distance metric
		:returns
			A list of 2 lists: First being the sorted order of
		"""

		assert type(index) == dict, "Index needs to be of dictionary type."

		if addn_params is None:
			addn_params = {"p": 2}
		self.index = index
		self.num_of_result = num_of_result
		self.metric = method
		self.metric_category_function, default_order = get_metric_function(method)
		self.ranking_order = ranking_order if ranking_order else default_order
		self.addn_params = addn_params

	def rank(self, query_feature_vector):
		computed_values = {}

		for image in self.index:
			image_feature_vector = self.index[image]
			computed_values[image] = self.metric_category_function(image_feature_vector, query_feature_vector, self.metric, self.addn_params)

		desc_flag = True if self.ranking_order == "desc" else False
		sorted_images = sorted(self.index.keys(), key=lambda k: computed_values[k], reverse=desc_flag)

		if self.num_of_result and self.num_of_result <= len(sorted_images):
			sorted_images = sorted_images[:self.num_of_result]

		return sorted_images, [computed_values[image] for image in sorted_images]

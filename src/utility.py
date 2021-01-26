import numpy as np
import exception as e


def accuracy(preds, y):
	if preds.shape != y.shape:
		print(f"*Warning! preds.shape != y.shape ({preds.shape} != {y.shape}) This may result in an unexpected accuracy*")
	total = len(preds)
	diff = preds - y
	correct = len(diff[diff == 0])
	acc = (correct / total) * 100
	return correct, total, acc


def add_bias_col(mat):
	return np.append(mat, np.ones([mat.shape[0], 1]), axis=1)


def one_hot(vector, classes=0):
	if len(vector.shape) < 2:
		raise e.DimensionError(f"One hot expects a 2d vector but a 1d vector was provided ({vector.shape})")
	if vector.shape[1] != 1:
		raise e.DimensionError(f"One hot expects a vector of shape (n, 1) but a vector of shape ({vector.shape}) was provided")
	
	max_val = vector.max()
	if classes == 0:
		classes = max_val+1
	if classes <= max_val:
		raise OneHotClassError(classes, max_val)
	one_hot = np.zeros([vector.shape[0], classes])
	cols = vector.reshape(1, -1)
	cols = cols.astype(np.int32)
	one_hot[np.arange(len(vector)), cols] = 1
	return one_hot



class OneHotClassError(Exception):
	def __init__(self, classes, max_val):
		self.classes = classes
		self.max_val = max_val

	def __str__(self):
		return (f"Possible classes in a one hot vector must be at least equal to max+1 ({self.max_val+1} in this case) "
				f"of the vector (classes={self.classes} | max value={self.max_val})")

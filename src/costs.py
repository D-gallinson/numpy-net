import abc
import numpy as np
import utility as ut
import exception as ex


class Cost():
	def __init__(self):
		pass

	@abc.abstractmethod
	def cost(self, predictions, Y):
		pass

	@abc.abstractmethod
	def derivative(self, predictions, Y):
		pass



class CrossEntropy(Cost):
	def __init__(self):
		pass

	def __call__(self, predictions, Y):
		return self.cost(predictions, Y)

	def cost(self, predictions, Y):
		y_one_hot = ut.one_hot(Y)
		n = len(predictions)
		truth_mat = np.where(y_one_hot == 1, True, False)
		ground_truth_preds = predictions[truth_mat]
		return -(1 / n) * np.sum(np.log(ground_truth_preds))

	def derivative(self, predictions, Y):
		y_one_hot = ut.one_hot(Y)
		n = len(predictions)
		grads = np.zeros(y_one_hot.shape)
		truth_mat = np.where(y_one_hot == 1, True, False)
		grads[:, :] = -(1 / n) * (1 / predictions[truth_mat]).reshape(-1, 1)
		return grads



class MSE(Cost):
	def __init__(self):
		pass

	def __call__(self, predictions, Y):
		return self.cost(predictions, Y)

	def cost(self, predictions, Y):
		n = len(predictions)
		sum_squares = np.sum((predictions - Y) ** 2)
		avg = sum_squares / n
		return avg

	def derivative(self, predictions, Y):
		n = len(predictions)
		return (2/n) * (predictions - Y)



class NLLLoss(Cost):
	def __init__(self, reduction="mean"):
		if reduction not in ["mean", "sum", "none"]:
			raise ex.ParameterError(f"NLLLoss does not have reduction={reduction}. Try sum, mean, or none.")
		self.reduction = reduction

	def __call__(self, predictions, y):
		return self.cost(predictions, y)

	def cost(self, predictions, y):
		if len(y.shape) > 1:
			raise ex.DimensionError(f"Expected y to be 1d but a 2d y was received: {y.shape}")
		cost = -predictions[np.arange(len(predictions)), y]
		if self.reduction == "mean":
			return np.sum(cost) / len(cost)
		elif self.reduction == "sum":
			return np.sum(cost)
		else:
			return cost

	def derivative(self, predictions, y):
		mult = 1
		y = y.reshape(-1, 1)
		if self.reduction == "mean":
			mult = 1/len(predictions)
		grads = ut.one_hot(y, predictions.shape[1])
		return grads * -mult



class NegativeLogLikelihood():
	def __init__(self):
		pass

	def __call__(self, predictions, Y):
		return self.cost(predictions, Y)

	def cost(self, predictions, Y):
		n = len(predictions)
		neg_log = -(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
		class_sum = np.sum(neg_log, axis=1)
		return np.sum(class_sum) / n

	def derivative(self, predictions, Y):
		n = len(predictions)
		deriv = -((Y / predictions) - ((1 - Y) / (1 - predictions)))
		deriv_sum = np.sum(deriv, axis=1).reshape(-1, 1)
		return (1/n) * deriv_sum

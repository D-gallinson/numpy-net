import abc
import numpy as np
import utility as ut
import exception as ex

class Activation():
	def __init__(self):
		self.multivariate = False

	def __str__(self):
		return self.function

	def __eq__(self, x):
		return self.function.lower() == x.lower()

	def __ne__(self, x):
		return self.function.lower() != x.lower()

	@abc.abstractmethod
	def forward(self, x):
		pass

	@abc.abstractmethod
	def derivative(self, x):
		pass



class LogSoftmax(Activation):
	def __init__(self, dim=1):
		self.function = "LogSoftmax"
		self.multivariate = True
		self.dim = dim

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		max_vals = np.max(x, axis=self.dim)
		if self.dim == 1:
			max_vals = max_vals.reshape(-1, 1)
		exp_sum = np.sum(np.exp(x - max_vals), axis=self.dim)
		if self.dim == 1:
			exp_sum = exp_sum.reshape(-1, 1)
		log_sum_exp = max_vals + np.log(exp_sum)
		return x - log_sum_exp

	def derivative(self, x, loss_grad):
		if self.dim == 0:
			x = x.T
		grads = np.zeros(x.shape)
		soft = Softmax(dim=1)
		for i in range(len(x)):
			x_i = x[i].reshape(1, -1)
			softmax = soft(x_i)
			tiled = np.tile(softmax, x_i.shape[1]).reshape(x_i.shape[1], -1)
			submat = np.diagflat(np.ones(x.shape[1]))
			deriv = submat - tiled
			grads[i] = loss_grad[i].dot(deriv)
		if self.dim == 0:
			grads = grads.T
		return grads



class Naught(Activation):
	def __init__(self):
		self.function = "Naught"

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		return x

	def derivative(self, x):
		return 1



class ReLU(Activation):
	def __init__(self, inplace=False):
		self.inplace = inplace
		self.function = "ReLU"

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		if not self.inplace:
			result = np.copy(x)
			result[result < 0] = 0
		else:
			x[x < 0] = 0
			result = x
		return result

	def derivative(self, x):
		result = np.copy(x)
		result[result > 0] = 1
		result[result <= 0] = 0
		return result



class Sigmoid(Activation):
	def __init__(self):
		self.function = "Sigmoid"

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		return (1 / (1+ np.exp(-x)))

	def derivative(self, x):
		sigmoid = self.forward(x)
		return sigmoid * (1 - sigmoid)


#!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!
class Softmax(Activation):
	def __init__(self, dim=1):
		self.function = "Softmax"
		self.multivariate = True
		self.dim = dim

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		max_vals = np.max(x, axis=self.dim)
		if self.dim == 1:
			max_vals = max_vals.reshape(-1, 1)
		numer = np.exp(x - max_vals)
		denom = np.sum(numer, axis=self.dim)
		if self.dim == 1:
			denom = denom.reshape(-1, 1)
		return numer / denom

	#THIS NEEDS TO BE FINISHED. EVERYTHING BEFORE THE FOR LOOP IS POTENTIALLY USABLE
	def derivative(self, x):
		if self.dim == 0:
			raw_shape = [x.shape[0], x.shape[1]*x.shape[0]]
		else:
			raw_shape = [x.shape[1]*x.shape[0], x.shape[0]]
		grads = np.zeros(raw_shape)
		softmax = self.forward(x)
		for row in range(len(grads)):
			soft_row = softmax[row, :][np.newaxis]
			row_deriv = np.diagflat(soft_row) - (soft_row.T.dot(soft_row))
			true_grad = row_deriv[truth_mat[row, :]]
			grads[row] = true_grad
		return grads



"""
IMPLEMENT THESE

def __tanh(self, vec):
	return np.tanh(vec)
"""


activation_dict = {
	"logsoftmax": LogSoftmax,
	"naught": Naught,
	"relu": ReLU,
	"sigmoid": Sigmoid,
	"softmax": Softmax
}



class ActivationFunctionDNEError(Exception):
	def __init__(self, dne_func):
		self.dne_func = dne_func

	def __str__(self):
		return f"The activation function \"{self.dne_func}\" does not exist"

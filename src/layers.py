import numpy as np
import utility as ut


class Layer():
	pass



class LinearFC(Layer):
	def __init__(self, in_features=0, out_features=0, load=np.array([]), bias=True, init_func="he", bias_init="zeros", const_val=0.01):
		rows = in_features
		cols = out_features
		self.name = "LinearFC"
		self.bias = bias
		if load.size == 0:
			if bias:
				weights = self.__handle_init(init_func, rows, cols)
				bias = self.__handle_bias_init(bias_init, cols, val=const_val)
				self.parameter_matrix = np.concatenate([weights, bias])
				rows += 1
			else:
				self.parameter_matrix = self.__handle_init(init_func, rows, cols)
			self.shape = (rows, cols)
		else:
			self.parameter_matrix = load
			self.shape = load.shape


	def __call__(self, input):
		return self.forward(input)


	def __getitem__(self, index):
		return self.parameter_matrix[index]


	def __str__(self):
		return f"{self.parameter_matrix}"


	def alter_param(self, alteration, row, col, save_old=True):
		if save_old:
			self.old_param = np.copy(self.parameter_matrix)
		self.parameter_matrix[row, col] = alteration


	def forward(self, x):
		#Convenience check for inputting numbers. Not necessary, remove if needed
		if not isinstance(x, np.ndarray):
			x = np.array([[x]])
		if self.bias:
			x = ut.add_bias_col(x)
		return x.dot(self.parameter_matrix)


	def get_weight(self):
		if self.bias:
			return self.parameter_matrix[:-1, :]
		return self.parameter_matrix


	def get_bias(self):
		if self.bias:
			return self.parameter_matrix[-1, :]
		return np.array([])


	def restore_param(self):
		try:
			self.parameter_matrix = self.old_param
		except AttributeError:
			print("Restore failed. No parameter matrix in memory.")


	def str_dim(self):
		string = f"{self.shape[0]} x {self.shape[1]}"
		if not self.bias:
			string += "[NO BIAS]"
		return string


	def str_params(self):
		bias = self.get_bias()
		weights = self.get_weights()
		string = ""
		string += f"--WEIGHTS--\n{weights}\n\n"
		if self.bias:
			string += f"--BIAS--\n{bias}"

		return string


	def update_params(self, grad, alpha):
		self.parameter_matrix -= alpha * grad


	def __handle_init(self, init_func, rows, cols):
		if init_func == "he":
			return self.__init_he(rows, cols)
		elif init_func == "ones":
			return self.__init_ones(rows, cols)
		elif init_func == "rand":
			return self.__init_rand(rows, cols)
		elif init_func == "randpos":
			return self.__init_randpos(rows, cols)
		elif init_func == "xavier":
			return self.__init_xavier(rows, cols)
		elif init_func == "zeros":
			return self.__init_zeros(rows, cols)


	def __handle_bias_init(self, init_func, cols, val=0.01):
		if init_func == "const":
			return self.__init_const(val, 1, cols)
		if init_func == "ones":
			return self.__init_ones(1, cols)
		if init_func == "zeros":
			return self.__init_zeros(1, cols)


	def __init_const(self, val, rows, cols):
		return np.full([rows, cols], val)


	def __init_he(self, rows, cols):
		return np.random.randn(rows, cols) * np.sqrt(2./rows)


	def __init_ones(self, rows, cols):
		return np.ones([rows, cols])


	def __init_rand(self, rows, cols):
		return np.random.randn(rows, cols)


	def __init_randpos(self, rows, cols):
		return np.random.rand(rows, cols)


	def __init_xavier(self, rows, cols):
		return np.random.uniform(-1, 1, [rows, cols]) * np.sqrt(6./(rows+cols))


	def __init_zeros(self, rows, cols):
		return np.zeros([rows, cols])


layer_dict = {
	"linearfc": LinearFC
}

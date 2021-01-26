import numpy as np
import utility as ut


class Backprop():
	def __init__(self, X, Y, network, cost):
		self.X = X
		self.Y = Y
		self.network = network
		self.cost = cost
		self.grads = []

		self.cost_tracking = False
		self.cost_printing = False


	def __str__(self):
		if not self.grads:
			return "Grad empty"
		string = ""
		for grad in self.grads:
			string += f"{grad}\n\n"
		return string


	def backprop(self):
		backprop_mat = np.array([])

		#Iterate backwards, stop when the first (element 0) item in z_list/a_list is reached
		for i in range(self.network.size-1, -1, -1):
			#-------------------------------SETUP LAYER VARS-------------------------------
			"""
			Network structure: [[layer, activation], [layer, activation], [layer, activation]].
			Thus, for a given complete_layer, complete[0] is always a layer and complete[1] is always an activation
			"""
			predictions = self.network(self.X)
			current_activation = self.network.network[i][1]
			prev_layer = self.network.network[i-1][0]
			current_z = self.network.z_list[i+1]


			if prev_layer.bias:
				prev_a = ut.add_bias_col(self.network.a_list[i])
			else:
				prev_a = self.network.a_list[i]

			#-------------------------------------------------------------------------------

			#----------------------------------BACKPROP PHASE-------------------------------			
			if backprop_mat.size == 0:
				backprop_mat = self.__output_layer_backprop(predictions, current_z, current_activation)
			else:
				next_layer = self.network.network[i+1][0]
				weights = next_layer.get_weight().T
				backprop_mat = self.__hidden_layer_backprop(backprop_mat, weights, current_z, current_activation)
			#-------------------------------------------------------------------------------

			#----------------------------------GRADIENT PHASE-------------------------------
			grad_cols = []
			for col in backprop_mat.T:
				col = col[np.newaxis].T
				dcost_dw = col * prev_a
				grad_sum = np.sum(dcost_dw, axis=0)
				grad_cols.append(grad_sum[np.newaxis].T)
			grad_mat = np.concatenate(grad_cols, axis=1)
			self.grads.insert(0, grad_mat)
			#-------------------------------------------------------------------------------


	def step(self, alpha=0.01, max_grad=10):
		max_grad = max_grad / alpha
		for i in range(len(self.grads)):
			current_grad = self.grads[i]
			if np.isnan(current_grad).any():
				raise InvalidGradError("nan", i+1, len(self.grads))
			max_grads = current_grad[current_grad > max_grad]
			if max_grads.size > 0:
				print(f"*WARNING! Found {max_grads.size} large grads!*")
				print(max_grads)
			current_layer = self.network.network[i][0]
			current_layer.update_params(current_grad, alpha)

		#Cost tracking. Not relevant to actual gradient updating
		if self.cost_printing:
			print(f"Cost at epoch {self.epoch}: {self.cost(self.network(self.X), self.Y)}")
			self.epoch += 1
		if self.cost_tracking:
			new_cost = self.cost(self.network(self.X), self.Y)
			self.__check_cost(new_cost)
			self.cost_mem[1, self.cost_mem_i] = new_cost
			self.cost_mem_i += 1


	def clear_grads(self):
		self.grads = []


	def train(self, epochs=100, alpha=0.01, end_cost=True):
		for epoch in range(epochs):
			self.backprop()
			self.step(alpha)
			self.clear_grads()
		if end_cost:
			print(f"Cost after {epochs} epochs: {self.cost(self.network(self.X), self.Y)}")


	"""
	Used to help choose a learning rate. Should not be used during actual training when
	the number of epochs may be high, as the cost will (unnecessarily) be calculated after
	each gradient step. This should also be used with caution. For many networks, a cost
	that occassionally increases is acceptable. This is especially applicable for SGD
	(when the batch size=1).

	This should be used wherever the netowrk is run! Use case (assume Sequential and Backprop are set up as net and back respectively):
	
	back.track_cost(epochs=epochs)
	try: 
		for epoch in range(epochs):
			back.backprop()
			back.step()
			back.clear_grads()
	except bp.IncreasingCostError as e:
		print(e)
	"""
	def track_cost(self, flag=True, epochs=0):
		self.cost_tracking = flag
		self.cost_mem = np.array([np.arange(1, epochs+1), np.zeros([epochs])])
		self.cost_mem_i = 0


	def load_data(self, X, Y):
		self.X = X
		self.Y = Y


	def print_cost(self, flag=True):
		self.cost_printing = flag
		self.epoch = 1


	def print_largest_grads(self):
		i = 1
		print(f"Showing max grad for each individual neuron within a given layer")
		for grad in self.grads:
			max_neuron_grads = grad.max(axis=0)
			print(f"-----Layer {i} ({grad.shape[1]} Neurons)-----")
			print(max_neuron_grads)
			print()
			i += 1


	def __output_layer_backprop(self, predictions, current_z, current_activation):
		dcost_da = self.cost.derivative(predictions, self.Y)
		if not current_activation.multivariate:
			da_dz = current_activation.derivative(current_z)
			dcost_dz = dcost_da * da_dz
		else:
			dcost_dz = current_activation.derivative(current_z, dcost_da)
		return dcost_dz


	def __hidden_layer_backprop(self, backprop_mat, weights, current_z, current_activation):
		dcost_da = backprop_mat.dot(weights)
		da_dz = current_activation.derivative(current_z)
		dcost_dz = dcost_da * da_dz
		return dcost_dz


	def __check_cost(self, new_cost):
		current_mem = self.cost_mem[:2, :self.cost_mem_i]
		if current_mem.size > 0:
			smaller_prev_costs = current_mem[:2, current_mem[1] < new_cost]
			if smaller_prev_costs.size != 0:
				raise IncreasingCostError([self.cost_mem_i+1, new_cost], smaller_prev_costs[:, 0])



class IncreasingCostError(Exception):
	def __init__(self, new_cost, old_cost):
		self.new_cost = new_cost
		self.old_cost = old_cost

	def __str__(self):
		error_msg = f"Cost increased!\n"
		error_msg += f"Cost at epoch {int(self.old_cost[0])}: {self.old_cost[1]}\n"
		error_msg += f"Cost at epoch {self.new_cost[0]}: {self.new_cost[1]} <- this is the current cost"
		return error_msg



class InvalidGradError(Exception):
	def __init__(self, invalid_val, i, total):
		self.invalid_val = invalid_val
		self.i = i
		self.total = total

	def __str__(self):
		return f"Invalid value in grad matrix {self.i} of {self.total}: {self.invalid_val}"

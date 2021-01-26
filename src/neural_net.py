import types
import numpy as np
import activations as ac
import layers as ly


"""
--INPUT--
Takes as an argument an arbitrary number of individual network layers. This is modeled after PyTorch's Sequential class
and works in the same way - supply the class with the desired network operations in sequence. A limitation of this is that
a single layer may only have one activation function applied to it - a sigmoid and ReLU cannot be used on different 
neurons of the same layer. Of course, I don't believe this is really ever done but it could be interesting for exploratory
purposes. If no activation is desired for a layer, simply do not add an activation object after a given layer.

EXAMPLE:
Architecture: input (5 neurons, bias) -> h1 (500 neurons, no bias) -> ReLU -> h2 (25 neurons, bias) -> ReLU -> output (10 neurons, bias not possible) -> Sigmoid
Implementation: As a note, assume the input x is a (n, 5) matrix

net = Sequential(
	layers.LinearFC(5, 500),
	activations.ReLU(),
	layers.LinearFC(500, 25, bias=False),
	activations.ReLU(),
	layers.LinearFC(25, 10),
	activations.Sigmoid()
)

As a note, the actual number of neurons in a layer is defined by the column value (or the out_features parameter of LinearFC()). The row value (in_feautres)
must be equivalent to the prior layer's out_features. Furthermore, if a bias is to be omitted, the LinearFC() object AHEAD of the omitting layer should
set bias=False. The above example shows this: the network has 4 layers - an input layer, 2 hidden layers, and an output layer. Thus, it has 3 LinearFC()
objects (the number of LinearFC() objects is always layers-1). The second layer is to omit a bias. The structure of the network's parameter matrices
are as follows:
---Parameters 1---
Between input layer and hidden layer 1 (layers 1 and 2)

---Parameters 2--
Between hidden layer 1 and hidden layer 2 (layers 2 and 3)

---Parameters 3---
Between hidden layer 2 and output layer (layers 3 and 4)

Parameters 2 lies ahead of layer 2 and behind layer 3; because we are to omit the bias of layer 2, this is the layer where bias=False
"""
class Sequential():
	def __init__(self, *layers):
		if isinstance(layers[0], str):
			self.load(layers[0])
		elif isinstance(layers[0], types.GeneratorType):
			self.network = self.__load_pytorch(layers[0])
		else:
			self.network = self.__gen_architecture(layers)
		self.z_list = []
		self.a_list = []
		self.forward_pass_flag = False
		self.tracking = True


	def __call__(self, x):
		return self.forward(x)


	def __getitem__(self, i):
		return self.network[i]


	def __iter__(self):
		self.index = 0
		return self


	def __next__(self):
		if self.index >= len(self.network):
			raise StopIteration
		item = self.network[self.index]
		self.index += 1
		return item


	def __str__(self):
		architecture = ""
		i = 0
		input_neurons = self.network[0][0].shape[0]
		input_bias = self.network[0][0].bias
		input_neurons = input_neurons if not input_bias else input_neurons-1
		architecture += f"Input: {input_neurons} {'neurons' if input_neurons>1 else 'neuron'}"
		if input_bias:
			architecture += ", 1 bias --> "
		else:
			architecture += " --> "

		for complete_layer in self:
			layer = complete_layer[0]
			activation = complete_layer[1]
			neurons = layer.shape[1]
			if i < len(self.network) - 1:
				bias = self.network[i+1][0].bias
				layer_type = "Hidden"
				arrow = " --> "
			else:
				layer_type = "Output"
				bias = False
				arrow = ""
			architecture += f"{layer_type}: {neurons} {'neurons' if neurons > 1 else 'neuron'}"
			if bias:
				architecture += f", 1 bias"
			architecture += f" [{activation}]{arrow}"
			i += 1
		return architecture


	def add_layer(self, layer):
		if issubclass(type(layer), ac.Activation):
			last_layer = self.network[-1]
			if last_layer[1].function != "Naught":
				raise ArchitectureError((f"Attempted to add an activation layer onto an activaton function. "
										f"Previous layer: {last_layer[1].function}() | adding layer: {layer.function}()"))
			self.network[-1][1] = layer
		else:
			complete_layer = [layer, ac.Naught()]
			self.network.append(complete_layer)


	def get_params(self):
		params = [layer[0].parameter_matrix for layer in self.network]
		return params


	def forward(self, x):
		self.forward_pass_flag = True
		self.z_list = [np.array([[1]])]
		self.a_list = [x]

		for layer in self.network:
			linear = layer[0]
			activation = layer[1]

			x = linear(x)
			if self.tracking:
				self.z_list.append(x)
			x = activation(x)
			if self.tracking:
				self.a_list.append(x)
		return x


	def grads(self, flag=True):
		self.tracking = flag


	def output_layer(self, x, layer):
		if layer > self.size:
			print("Layer is larger than the size of the network!")
		for i in range(layer):
			linear = self.network[i][0]
			activation = self.network[i][1]
			x = linear(x)
			x = activation(x)
		return x


	def pred_classes(self, x, is_inputs=True):
		if is_inputs:
			preds = self.forward(x)
		else:
			preds = x
		return np.argmax(preds, axis=1)


	def print_arch_raw(self):
		architecture = ""
		i = 1
		for complete_layer in self.network:
			layer = complete_layer[0]
			activation = complete_layer[1].function
			architecture += f"Linear({layer.str_dim()}) -> {activation}"
			if i < len(self.network):
				architecture += " -> "
			i += 1

		print(architecture)


	def print_neuron_vals(self):
		if not self.a_list:
			print("Please run a forward pass with grad tracking on (grads())!")
		else:
			print("-----Z LIST-----")
			for z in self.z_list:
				print(f"{z}\n")
			print("\n\n-----A LIST-----")
			for a in self.a_list:
				print(f"{a}\n")


	def print_params(self, format=False):
		i = 1
		for complete_layer in self.network:
			layer = complete_layer[0]
			if format:
				print(f"*****Parameters {i} (between layers {i} and {i+1})*****")
				print(layer.str_params())
				print()
				i += 1
			else:
				print(layer)
				print()


	"""
	Prints the max value within a column. This corresponds to the max weight going into a specific neuron
	"""
	def print_largest_params(self):
		i = 1
		print(f"Showing max weights for each individual neuron within a given layer")
		for complete_layer in self.network:
			layer = complete_layer[0]
			max_neuron_weights = layer.parameter_matrix.max(axis=0)
			print(f"-----Layer {i} ({layer.shape[1]} Neurons)-----")
			print(max_neuron_weights)
			print()
			i += 1


	def save(self, filename):
		param_dict = {}
		layer_list = []
		activation_list = []
		for i in range(self.size):
			key = f"arr{i}"
			layer = self.network[i][0]
			activation = self.network[i][1]
			if layer.bias:
				key = "b_" + key
			param_dict[key] = layer.parameter_matrix
			layer_list.append(layer.name.lower())
			activation_list.append(activation.function.lower())
		np.savez(f"{filename}_params.npz", **param_dict)
		with open(f"{filename}_architecture.txt", "w") as f:
			layer_arch_str = self.__write_arch(layer_list)
			activation_arch_str = self.__write_arch(activation_list)
			f.write(layer_arch_str)
			f.write("\n")
			f.write(activation_arch_str)

		print(f"File saved as {filename}_params.npz and {filename}_architecture.txt")


	def load(self, filename):
		network = []
		param_load = np.load(f"{filename}_params.npz")
		with open(f"{filename}_architecture.txt", 'r') as f:
			layer_load = f.readline().rstrip()
			activation_load = f.readline()
		layer_load = layer_load.split(",")
		activation_load = activation_load.split(",")

		for i in range(len(param_load)):
			key = param_load.files[i]
			current_param = param_load[key]
			current_layer = ly.layer_dict[layer_load[i]]
			current_activation = ac.activation_dict[activation_load[i]]
			if "b" in key:
				bias = True
			else:
				bias = False
			complete_layer = [current_layer(load=current_param, bias=bias), current_activation()]
			network.append(complete_layer)
		self.network = network



	"""
	Combines the methods __check_architecture() and complete_layers()
	"""
	def __gen_architecture(self, layers):
		verified_net = self.__check_architecture(layers)
		return self.__complete_layers(verified_net)


	"""
	This ensures that the network has the structure: [layer, activation, layer, activation, layer, activation].
	Basically, it ensures that the network always starts with a layer (or throws an exception), that the network
	always alternates between layers and activations, and that it ends with an activation. It also ensures that
	there are never 2 activations in a row (or it throws an exception). If there are 2 or more layers in a row,
	it inserts a Layer.Naught() object. I consider a "complete layer" to be a layer that includes its activation function.
	"""
	def __check_architecture(self, layers):
		last_layer = None
		network = []

		if issubclass(type(layers[0]), ac.Activation):
			raise ArchitectureError(f"Attempted to add a {layers[0].function}() as the first layer.")

		for layer in layers:
			if issubclass(type(last_layer), ly.Layer) and issubclass(type(layer), ly.Layer):
				network.append(ac.Naught())
			elif issubclass(type(last_layer), ac.Activation) and issubclass(type(layer), ac.Activation):
				raise ArchitectureError((f"Attempted to stack two activation layers. Previous layer: "
										 f"{last_layer.function}() | current layer: {layer.function}()"))
			network.append(layer)
			last_layer = layer

		if not issubclass(type(layers[-1]), ac.Activation):
			network.append(ac.Naught())

		return network


	"""
	I consider a "complete layer" to be a layer that includes its activation function. This method generates a network
	of the structure: [[layer, activation], [layer, activation], [layer, activation]].
	It must be run after __check_architecture(self, layers)
	"""
	def __complete_layers(self, layers):
		network = []
		for i in range(0, len(layers), 2):
			complete_layer = [layers[i], layers[i+1]]
			network.append(complete_layer)

		self.size = len(network)
		return network


	def __write_arch(self, arch_list):
		string = ""
		length = len(arch_list)
		for i in range(length):
			to_write = arch_list[i]
			if i != length - 1:
				to_write += ","
			string += to_write
		return string


	"""
	This expects gen to be obtained from pytorch.modules() NOT pytorch.parameters()
	Can only accept nn.Sequential (creating a custom Net(nn.Module) will break this)
	"""
	def __load_pytorch(self, gen):
		architecture = next(gen)
		first_type = str(type(architecture[0]))
		if "Tensor" in first_type:
			raise PytorchLoadError(f"Pytorch layer of type {first_type} detected. Ensure net.modules() was used and not net.parameters()")

		layers = []
		for layer in architecture:
			layer_type = str(type(layer))
			if "activation" in layer_type:
				start = layer_type.rfind(".") + 1
				activation_func = layer_type[start:-2].lower()
				if activation_func in ac.activation_dict.keys():
					layers.append(ac.activation_dict[activation_func]())
				else:
					raise ac.ActivationFunctionDNEError(activation_func)
			elif "Linear" in layer_type:
				weight = np.array(layer.weight.detach()).T
				bias = np.array(layer.bias.detach()).reshape(1, -1)
				parameters = np.concatenate([weight, bias], axis=0)
				layers.append(ly.LinearFC(load=parameters))
			else:
				msg = (f"Could not convert Pytorch architecture to my library. Encountered Pytorch layer of unknown type: {str(layer)}"
					   f"is of type: {layer_type}")
				raise PytorchLoadError(msg)
		return self.__gen_architecture(layers)



class ArchitectureError(Exception):
	def __init__(self, msg):
		self.msg = msg

	def __str__(self):
		return f"There is a problem with the network architecture. {self.msg}"



class PytorchLoadError(Exception):
	def __init__(self, msg):
		self.msg = msg

	def __str__(self):
		return self.msg

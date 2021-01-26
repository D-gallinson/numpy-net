class DimensionError(Exception):
	def __init__(self, problem_str):
		self.problem = problem_str

	def __str__(self):
		return f"Problem with matrix dimension. {self.problem}"



class ParameterError(Exception):
	def __init__(self, problem_str):
		self.problem = problem_str

	def __str__(self):
		return f"Problem with parameter input. {self.problem}"
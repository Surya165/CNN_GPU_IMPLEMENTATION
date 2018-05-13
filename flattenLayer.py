import numpy
class Flatten:
	def __init__(self):
		self.name = "flatten"
		self.isCompiled  = False
	def printLayer(self):
		print("Flatten Layer")
		if self.isCompiled:
			print(str(self.shape))

	def compile(self,previousLayerShape):
		self.number = previousLayerShape[0]
		self.matrix = numpy.zeros(previousLayerShape[1:])
		self.matrix = numpy.resize(self.matrix,(numpy.product(list(previousLayerShape[1:])),))
		self.shape = tuple([self.number]+list(self.matrix.shape))
		self.isCompiled = True

		return self

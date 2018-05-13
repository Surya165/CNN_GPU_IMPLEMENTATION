import numpy
class FlatElement:
	def __init__(self,previousLayerShape):
		self.matrix = numpy.zeros(previousLayerShape)
		self.shape = (   numpy.product(list(previousLayerShape))   ,   )
		self.matrix = numpy.resize(self.matrix,self.shape)
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
		self.FlatElements = []
		for i in range(self.number):
			self.FlatElements.append(FlatElement(previousLayerShape[1:]))
		self.shape = tuple([self.number]+list(self.FlatElements[0].shape))
		self.isCompiled = True

		return self.shape 

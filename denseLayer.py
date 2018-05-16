import numpy

class Dense:
	def __init__(self,number):
		self.name = "dense"
		self.number = number
		self.activationMatrix = []
	def printLayer(self):
		print("Dense Layer. Number of Neurons = " + str(self.number))
		print(str(self.weightBuffer.shape))

	def compile(self,previousLayerShape,cl):
		shape = previousLayerShape
		self.previousLayerShape = previousLayerShape

		nOld = shape[0]


		self.shape = tuple([self.number])
		self.outputBuffer = numpy.zeros(self.shape)
		self.weightBuffer = numpy.random.rand(self.number,nOld)
		self.isCompiled = True

		return self.shape
	def forwardPropagate():
		pass

	def backwardPropagate():
		pass

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
		shape = previousLayerShape
		self.previousLayerShape = previousLayerShape

		mOld = shape[0]
		nOld = shape[1]
		pOld = shape[2]

		nNew = nOld
		pNew = pOld

		self.shape = tuple([nNew*pNew])
		self.outputBuffer = numpy.zeros(self.shape)
		self.isCompiled = True
		return self.shape


	def forwardPropagate():
		pass

	def backwardPropagate():
		pass

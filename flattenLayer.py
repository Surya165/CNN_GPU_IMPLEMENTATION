import numpy


class Flatten:
	def __init__(self):
		self.name = "flatten"
		self.isCompiled  = False
	def printLayer(self):
		print("Flatten Layer")
		if self.isCompiled:
			print(str(self.shape))

	def compile(self,previousLayerShape,cl):
		shape = previousLayerShape
		self.previousLayerShape = previousLayerShape

		mOld = shape[0]
		nOld = shape[1]
		pOld = shape[2]

		nNew = nOld
		pNew = pOld

		self.shape = tuple([nNew*pNew])
		self.outputMatrix = numpy.zeros(self.shape)
		self.isCompiled = True


		self.outputBuffer = cl.getBuffer(self.outputMatrix,"READ_WRITE")
		self.inputShapeBuffer = numpy.asarray(self.previousLayerShape)
		self.inputShapeBuffer = cl.getBuffer(self.inputShapeBuffer,"READ_ONLY")
		return self.shape


	def forwardPropagate():
		pass

	def backwardPropagate():
		pass

	def getAttributeList(self):
		return(self.inputShapeBuffer,self.outputBuffer)

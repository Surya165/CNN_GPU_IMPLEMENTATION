import numpy

class ConvolutionalLayer:

	def __init__(self,number,kernelShape):
		self.name = "convLayer"
		self.number = number
		self.kernelShape = kernelShape
		self.isCompiled = False

	def printLayer(self):
		print("Convolutional Layer.")
		if self.isCompiled:
			print(str(self.shape))
			print(str(self.weightBuffer.shape))

	def compile(self,previousLayerShape):
		self.previousLayerShape = previousLayerShape
		self.isCompiled = True
		shape = previousLayerShape
		if len(shape) != 3:
			print("Error: The input buffer for a ConvolutionalLayer should be a three dimensional image")
		mOld = shape[0]
		nOld = shape[1]
		pOld = shape[2]

		mNew = self.number
		nNew = nOld - self.kernelShape[0] + 1
		pNew = pOld - self.kernelShape[1] + 1



		self.shape = (mNew,nNew,pNew)
		self.outputBuffer = numpy.zeros(self.shape)
		self.weightBuffer = numpy.zeros((mNew,self.kernelShape[0],self.kernelShape[1],mOld))
		return self.shape

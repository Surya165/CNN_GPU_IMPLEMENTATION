import numpy
from cl import CL
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
			print(str(self.weightMatrix.shape))
	def compile(self,previousLayerShape,cl):
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
		self.outputMatrix = numpy.random.rand(mNew,nNew,pNew)
		self.weightMatrix = numpy.random.rand(mNew,self.kernelShape[0],self.kernelShape[1],mOld)
		self.biasMatrix = numpy.random.rand(mNEw,nNEw,pNew)

		self.weightShapeBuffer = numpy.asarray(list(self.weightMatrix.shape))
		self.weightShapeBuffer = cl.getBuffer(self.weightShapeBuffer,"READ_ONLY")

		self.weightBuffer = cl.getBuffer(self.weightMatrix,"READ_WRITE")
		self.outputBuffer = cl.getBuffer(self.outputMatrix,"READ_WRITE")
		self.biasBuffer = cl.biasBuffer(self.biasMatrix,"READ_WRITE")




		self.inputShapeBuffer = numpy.asarray(self.previousLayerShape)
		self.inputShapeBuffer = cl.getBuffer(self.inputShapeBuffer,"READ_ONLY")
		return self.shape

	def getAttributeList(self):
		return (self.inputShapeBuffer,self.outputBuffer,\
		self.weightBuffer,self.weightShapeBuffer,self.biasBuffer)

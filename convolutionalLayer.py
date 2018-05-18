import numpy
from cl import CL
from time import time
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
		self.outputMatrix = numpy.random.normal(0,1.0,(mNew,nNew,pNew))
		self.weightMatrix = numpy.random.normal(0,1.0,(mNew,self.kernelShape[0],self.kernelShape[1],mOld))
		self.biasMatrix = numpy.random.normal(0,1.0,(mNew))

		self.weightShapeBuffer = numpy.asarray(list(self.weightMatrix.shape))
		self.weightShapeBuffer = cl.getBuffer(self.weightShapeBuffer,"READ_ONLY")

		self.weightBuffer = cl.getBuffer(self.weightMatrix,"READ_WRITE")
		self.outputBuffer = cl.getBuffer(self.outputMatrix,"READ_WRITE")
		self.biasBuffer = cl.getBuffer(self.biasMatrix,"READ_WRITE")

		self.program = cl.getProgram("kernels/conv.cl")




		self.inputShapeBuffer = numpy.asarray(self.previousLayerShape)
		self.inputShapeBuffer = cl.getBuffer(self.inputShapeBuffer,"READ_ONLY")
		return self.shape

	def getAttributeList(self,cl):
		outputMatrix = cl.getFilterMapImages(self.outputBuffer,self.outputMatrix.shape,"float")
		weightMatrix = cl.getFilterMapImages(self.weightBuffer,self.weightMatrix.shape,"float")
		biasMatrix = cl.getFilterMapImages(self.biasBuffer,self.biasMatrix.shape,"float")
		return (self.previousLayerShape,outputMatrix,\
		weightMatrix,self.weightMatrix.shape,biasMatrix)




	def forwardPropagate(self,inputBuffer,cl):
		globalSize = self.shape
		t1 = time()
		self.program.forwardPropagate(\
		cl.commandQueue,\
		globalSize,\
		None,\
		inputBuffer,\
		self.inputShapeBuffer,\
		self.outputBuffer,\
		self.weightBuffer,\
		self.weightShapeBuffer,\
		self.biasBuffer\
		).wait()

		t2=time()

		#cl.clear([inputBuffer,inputShapeBuffer])
		#print("Time for convLayer is "+str(round((t2-t1)*100000)/100)+"ms")
		return self.outputBuffer

	def backwardPropagate(self,errorBuffer,cl,lambdaValue,etaValue,count,previousOutputBuffer):
		globalSize = self.weightMatrix.shape[0:2]
		nextErrorBuffer = numpy.zeros(self.previousLayerShape,dtype=numpy.float64)
		nextErrorBuffer = cl.getBuffer(nextErrorBuffer,"READ_WRITE")
		t1 = time()
		biasBuffer = self.biasBuffer
		self.previousOutputBuffer = previousOutputBuffer
		self.program.backwardPropagate\
		(\
		cl.commandQueue,\
		globalSize,\
		None,\
		errorBuffer,\
		self.weightBuffer,\
		self.weightShapeBuffer,\
		biasBuffer,\
		self.outputBuffer,\
		etaValue,\
		nextErrorBuffer,\
		previousOutputBuffer,\
		self.inputShapeBuffer\
		).wait()
		self.biasBuffer = biasBuffer
		return nextErrorBuffer

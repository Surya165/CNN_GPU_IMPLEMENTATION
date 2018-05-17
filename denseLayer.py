import numpy
from time import time

class Dense:
	def __init__(self,number):
		self.name = "dense"
		self.number = number
	def printLayer(self):
		print("Dense Layer. Number of Neurons = " + str(self.number))
		print(str(self.weightMatrix.shape))

	def compile(self,previousLayerShape,cl):
		shape = previousLayerShape
		self.program = cl.getProgram("kernels/dense.cl")
		self.previousLayerShape = previousLayerShape

		nOld = shape[0]


		self.shape = tuple([self.number])
		self.outputMatrix = numpy.zeros(self.shape)
		self.weightMatrix = numpy.random.normal(0,1.0,(self.number,nOld))
		self.biasMatrix = numpy.random.normal(0.0,1.0,(self.number))
		self.weightShapeMatrix = numpy.asarray(list(self.weightMatrix.shape))

		self.weightBuffer = cl.getBuffer(self.weightMatrix,"READ_WRITE")
		self.weightShapeBuffer = cl.getBuffer(self.weightShapeMatrix,"READ_WRITE")
		self.outputBuffer = cl.getBuffer(self.outputMatrix,"READ_WRITE")
		self.biasBuffer = cl.getBuffer(self.biasMatrix,"READ_WRITE")
		self.isCompiled = True

		return self.shape


	def forwardPropagate(self,inputBuffer,cl):
		globalSize = self.shape
		t1 = time()
		self.program.forwardPropagate(\
		cl.commandQueue,\
		globalSize,\
		None,\
		inputBuffer,\
		self.weightBuffer,\
		self.weightShapeBuffer,\
		self.biasBuffer,\
		self.outputBuffer\
		).wait()
		t2 = time()
		print("Time for dense is "+str(round((t2-t1)*100000)/100)+"ms")
		return self.outputBuffer


	def backwardPropagate(self,errorBuffer,cl,lambdaValue,etaValue,isOuterLayer):
		if isOuterLayer:
			errorBufferShape = layer.number

		globalSize = layer.weightMatrix.shape
		t1 = time()
		self.program.backwardPropagate\
		(\
		cl.commandQueue,\
		globalSize,\
		errorBuffer,\
		self.weightBuffer,\
		self.weightShapeBuffer,\
		self.outputBuffer\
		)

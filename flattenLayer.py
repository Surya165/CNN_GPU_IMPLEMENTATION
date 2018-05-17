import numpy
from time import time

class Flatten:
	def __init__(self):
		self.name = "flatten"
		self.isCompiled  = False
	def printLayer(self):
		print("Flatten Layer")
		if self.isCompiled:
			print(str(self.shape))

	def compile(self,previousLayerShape,cl):
		self.program = cl.getProgram("kernels/flatten.cl")
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


	def forwardPropagate(self,inputBuffer,cl):
		globalSize = self.shape
		t1 = time()
		self.program.forwardPropagate(\
		cl.commandQueue,\
		globalSize,\
		None,\
		inputBuffer,\
		self.inputShapeBuffer,\
		self.outputBuffer\
		).wait()
		t2 = time()
		print("Time for flatten is "+str(round((t2-t1)*100000)/100)+"ms")
		#cl.clear([inputBuffer])
		return self.outputBuffer

	def backwardPropagate(self,errorBuffer,cl,lambdaValue,etaValue,count):
		globalSize = self.previousLayerShape
		nextErrorBuffer = cl.getBuffer(numpy.zeros(self.previousLayerShape,dtype=numpy.float64),"READ_WRITE")
		self.program.backwardPropagate\
		(\
		cl.commandQueue,\
		globalSize,\
		None,\
		errorBuffer,\
		nextErrorBuffer\
		).wait()
		return nextErrorBuffer

	def getAttributeList(self):
		return(self.inputShapeBuffer,self.outputBuffer)

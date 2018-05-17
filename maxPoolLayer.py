
import numpy
from time import time


class MaxPoolLayer:

	def __init__(self,kernelShape):
		self.name = "maxPool"
		self.kernelShape = kernelShape
		self.isCompiled = False





	def compile(self,previousLayerShape,cl):
		self.previousLayerShape = previousLayerShape
		self.program = cl.getProgram("kernels/maxPool.cl")

		self.isCompiled = True
		shape = previousLayerShape

		mOld = shape[0]
		nOld = shape[1]
		pOld = shape[2]

		mNew = mOld
		nNew = nOld - self.kernelShape[0] + 1
		pNew = pOld - self.kernelShape[0] + 1

		self.shape = (mNew,nNew,pNew)
		self.outputMatrix = numpy.zeros(self.shape)
		self.outputBuffer = cl.getBuffer(self.outputMatrix,"READ_WRITE")

		self.kernelShapeBuffer = numpy.asarray(self.kernelShape)
		self.kernelShapeBuffer = cl.getBuffer(self.kernelShapeBuffer,"READ_ONLY")
		return self.shape


	def printLayer(self):
		print("MaxPoolLayer" )
		if self.isCompiled:
			print(self.shape)
	def getAttributeList(self):
		return (self.outputBuffer,self.kernelShapeBuffer)


	def forwardPropagate(self,inputBuffer,cl):
		globalSize = self.shape
		t1 = time()
		self.program.forwardPropagate(\
		cl.commandQueue,\
		globalSize,\
		None,\
		inputBuffer,\
		self.outputBuffer,\
		self.kernelShapeBuffer\
		).wait()
		t2 = time()
		print("Time for maxPool is "+str(round((t2-t1)*100000)/100)+"ms")
		return self.outputBuffer

from cl import CL
from time import time
import numpy
class Trainer:
	def __init__(self,dataset=None,numberOfEpochs=1,miniBatchSize=1):
		if len(dataset.shape) == 3:
			self.inputImage = dataset
			return
		training_data,validation_data,testing_data = dataset
		self.training_data = training_data
		self.validation_data = validation_data
		self.testing_data = testing_data
		self.numberOfEpochs = numberOfEpochs
		self.miniBatchSize = miniBatchSize
	def train(self,network):
		#temp Draft
		self.forwardPropagate(self.inputImage,network)

	def forwardPropagate(self,inputImage,network):
		print("Setting Up GPU")
		t1 = time()
		cl = CL()
		t2 = time()
		print("Time taken for setting the GPU is "+ str(t2-t1))
		kernelFile = "kernels/forwardPropagate.cl"
		print("Building forwardPropagate Program")
		program = cl.getProgram(kernelFile)
		buffer = cl.getBuffer(inputImage,"READ_ONLY")
		for layer in network.layerStack:
			self.runKernelForwardPropagate(program,layer,buffer,cl)


		#pass

	def runKernelForwardPropagate(self,program,layer,inputBuffer,cl):


		if(layer.name == "convLayer"):
			weightBuffer = cl.getBuffer(layer.weightBuffer,"READ_ONLY")
			outputBuffer = cl.getBuffer(layer.outputBuffer,"READ_WRITE")
			globalSize = layer.shape
			weightShapeBuffer = numpy.asarray(list(layer.weightBuffer.shape))
			weightShapeBuffer = cl.getBuffer(weightShapeBuffer,"READ_ONLY")
			inputShapeBuffer = layer.previousLayerShape
			inputShapeBuffer = numpy.asarray(list(inputShapeBuffer))
			inputShapeBuffer = cl.getBuffer(inputShapeBuffer,"READ_ONLY")
			program.convLayer(cl.commandQueue,globalSize,None,inputBuffer,outputBuffer,weightBuffer,weightShapeBuffer).wait()



		if(layer.name == "maxPool"):
			outputBuffer = cl.getBuffer(layer.outputBuffer,"READ_WRITE")
			globalSize = layer.shape
			program.maxPool(cl.commandQueue,globalSize,None,inputBuffer,outputBuffer).wait()


		if(layer.name == "flatten"):
			outputBuffer = cl.getBuffer(layer.outputBuffer,"READ_WRITE")
			globalSize = layer.shape
			program.flatten(cl.commandQueue,globalSize,None,inputBuffer,outputBuffer).wait()


		if(layer.name == "dense"):
			globalSize = layer.shape
			outputBuffer = cl.getBuffer(layer.outputBuffer,"READ_WRITE")
			weightBuffer = cl.getBuffer(layer.weightBuffer,"READ_ONLY")
			weightShapeBuffer = numpy.asarray(list(layer.weightBuffer.shape))
			weightShapeBuffer = cl.getBuffer(weightShapeBuffer,"READ_ONLY")
			program.dense(cl.commandQueue,globalSize,None,inputBuffer,outputBuffer,weightShapeBuffer).wait()

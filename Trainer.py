from cl import CL
from time import time
import cv2 as cv
import numpy
class Trainer:
	def __init__(self,cl,dataset=None,numberOfEpochs=1,miniBatchSize=1):
		self.cl = cl

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
		t1 = time()
		cl = self.cl
		inputBuffer = self.cl.getBuffer(self.inputImage,"READ_ONLY")
		t2 = time()
		print("Time taken for creation of the buffer is "+ str(round((t2-t1)*100000)/100)+"ms")
		images = []
		t1 = time()
		for count,layer in enumerate(network.layerStack):
			outputBuffer = layer.forwardPropagate(inputBuffer,self.cl)
			inputBuffer = outputBuffer
		t2 = time()
		forwardPropagateTime = t2-t1

		errorBuffer = outputBuffer

		t1 = time()
		lambdaValue = 0
		etaValue = 0.001

		"""
		for count, layer in enumerate(network.layerStack[::-1]):
			if layer.name != "dense":
				break
			errorBuffer = layer.backwardPropagate(errorBuffer,self.cl,lambdaValue,etaValue)
		t2 = time()
		"""
		backwardPropagateTime = t2-t1


		totalTime = backwardPropagateTime + forwardPropagateTime



		#output = numpy.zeros((layer.number,),dtype=numpy.float64)
		#cl.enqueue_read_buffer(cl.commandQueue,outputBuffer,output)
		output = cl.getFilterMapImages(outputBuffer,(layer.number,),"float")

		maxOutput = max(output)
		for i in range(len(output)):
			if output[i] == maxOutput:
				print(i)
				break


		self.numberOfEpochs = 300
		self.miniBatchSize = 10
		numberOfImages = 7000/self.miniBatchSize
		time2 =int(round((totalTime)*numberOfImages*self.numberOfEpochs))
		timeInHrs = round(int(round(time2/60))/60)
		time2 = time2 % 3600
		timeInMins = round(time2/60)
		timeInSeconds = round(time2%60)

		totalTime = str(timeInHrs)+"Hrs "+str(timeInMins)+"Mins " + str(timeInSeconds) + "s"
		print("Estimated Time for "+str(self.numberOfEpochs)+" Epochs is " +totalTime)

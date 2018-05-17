from cl import CL
from time import time
import cv2 as cv
import numpy
from math import log
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


		output = cl.getFilterMapImages(outputBuffer,(layer.number,),"float")
		print(output)
		desiredOutput = [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]

		errorBuffer =numpy.zeros((layer.number,),dtype=numpy.float64)

		for i in range(len(desiredOutput)):
			a = float(desiredOutput[i])
			y = float(output[i])
			errorBuffer[i] = y*log(abs(a+0.00001)) + (1-y)*log(abs(a+0.00001))

		errorBuffer = cl.getBuffer(errorBuffer,"READ_WRITE")


		errorBuffer = outputBuffer

		t1 = time()
		lambdaValue = 0
		etaValue = 0.001



		for count, layer in enumerate(network.layerStack[::-1]):
			if layer.name != "dense" and layer.name != "flatten":
				break
			errorBuffer = layer.backwardPropagate(errorBuffer,self.cl,lambdaValue,etaValue,count)
		t2 = time()


		backwardPropagateTime = t2-t1


		totalTime = backwardPropagateTime + forwardPropagateTime



		#output = numpy.zeros((layer.number,),dtype=numpy.float64)
		#cl.enqueue_read_buffer(cl.commandQueue,outputBuffer,output)


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

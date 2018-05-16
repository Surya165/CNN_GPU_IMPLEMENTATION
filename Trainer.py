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
		inputBuffer = self.cl.getBuffer(self.inputImage,"READ_ONLY")
		t2 = time()
		print("Time taken for creation of the buffer is "+ str(round((t2-t1)*100000)/100)+"ms")
		images = []
		t1 = time()
		for count,layer in enumerate(network.layerStack):
			outputBuffer = layer.forwardPropagate(inputBuffer,self.cl)
			inputBuffer = outputBuffer
		t2 = time()
		numberOfEpochs = 30
		numberOfImages = 1400
		time2 =int(round((t2-t1)*numberOfImages*numberOfEpochs))
		timeInMins = round(time2/60)
		timeInSeconds = round(time2%60)
		totalTime = str(timeInMins)+"Mins " + str(timeInSeconds) + "s"
		print("Estimated Time for 30 Epochs is " +totalTime)

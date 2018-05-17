from cl import CL
from time import time
import cv2 as cv
import numpy
from math import log
import pyopencl
import sys
from random import shuffle
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
def delete_last_lines(n=1):
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)
class Trainer:
	def __init__(self,cl,dataset,numberOfEpochs=1,miniBatchSize=1):
		self.cl = cl

		"""
		if len(dataset.shape) == 3:
			self.inputImage = dataset
			return
		"""
		training_data,validation_data,testing_data = dataset
		self.training_data = training_data
		self.validation_data = validation_data
		self.testing_data = testing_data
		self.numberOfEpochs = numberOfEpochs
		self.miniBatchSize = miniBatchSize

	def train(self,network):
		numberOfImages = self.training_data[1].shape[0]
		labels = self.training_data[1]
		images = self.training_data[0]
		accuracy = 0
		n  = int(round(numberOfImages/100))
		x = [[i] for i in range(n)]

		for epoch in range(self.numberOfEpochs):
			accuracy = 0
			print("Shuffling Dataset")
			shuffle(x)
			print("Shuffling Completed")
			for i in range(n):
				#print(labels[i])
				accuracy += self.miniBatch(network,images[x[i]],labels[x[i]])
				percentageCompleted = round((float(i)/float(n))*100)
				delete_last_lines()
				print(str(percentageCompleted)+"%")
			print(int(round((float(accuracy)/float(n))*100)))
			print("\n")

		#self.miniBatch(network,self.inputImage)
		network.saveModel("Model.pkl")

	def miniBatch(self,network,inputImage,actualImageNumber):
		t1 = time()
		cl = self.cl
		inputImageBuffer = self.cl.getBuffer(inputImage,"READ_ONLY")


		etaValue = 0.01
		found = False
		desiredOutput = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
		for i in range(10):
			if i == actualImageNumber:
				desiredOutput[i] = 1.0
			else:
				desiredOutput[i] = 0.0
		for num in range(5):
			inputBuffer = inputImageBuffer
			for count,layer in enumerate(network.layerStack):
				outputBuffer = layer.forwardPropagate(inputBuffer,self.cl)
				inputBuffer = outputBuffer


			output = cl.getFilterMapImages(outputBuffer,(layer.number,),"float")
			maxOutput = max(output)
			accuracy = 0
			for i in range(len(output)):
				if output[i] == maxOutput:
					if i == actualImageNumber:
						accuracy = 1
						return accuracy
						found = True
					break
			if found:
				break

			#print(output)


			errorBuffer =numpy.zeros((layer.number,),dtype=numpy.float64)

			for i in range(len(desiredOutput)):
				a = float(desiredOutput[i])
				y = float(output[i])
				#print(log(abs(y)))
				if a == 0.0 or 1 - a == 0.0:
					errorBuffer[i] = y*log(abs(a+0.1)) - (1-y)*log(abs(1-a+0.1))
				else:
					errorBuffer[i] = y*log(abs(a)) - (1-y)*log(abs(1-a))

			#print(errorBuffer[5])
			errorBuffer = cl.getBuffer(errorBuffer,"READ_WRITE")







			layers = network.layerStack
			for count, layer in enumerate(network.layerStack[::-1]):
				if(count + 1 != len(layers)):
					previousOutputBuffer = layers[count+1].outputBuffer
				else:
					previousOutputBuffer = inputImageBuffer
				errorBuffer = layer.backwardPropagate(errorBuffer,self.cl,0,etaValue,count,previousOutputBuffer)





		return accuracy

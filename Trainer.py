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

		cl = self.cl
		flags = pyopencl.mem_flags.COPY_HOST_PTR | pyopencl.mem_flags.READ_WRITE
		numberOfImages = self.training_data[1].shape[0]
		labels = self.training_data[1]
		images = self.training_data[0]
		self.etaValue = numpy.zeros((1,),dtype=numpy.float32)
		self.etaValue[0] = 7.0
		self.etaValueBuffer = pyopencl.Buffer(cl.context,flags,hostbuf=self.etaValue)
		n  = int(round(numberOfImages/100))
		n = 100
		x = [[i] for i in range(n)]

		accuracyList = []
		for epoch in range(self.numberOfEpochs):
			accuracy = 0
			#print("Shuffling Dataset")
			shuffle(x)
			#print("Shuffling Completed")
			for i in range(n):
				#print(labels[i])
				accuracy += self.miniBatch(network,images[x[i]],labels[x[i]],True)
				percentageCompleted = round((float(i)/float(n))*100)
				delete_last_lines()
				print(str(percentageCompleted)+"%")
			accuracyList.append(accuracy)
			print("Train accuracy is: "+str(int(round((float(accuracy)/float(n))*100)))+"%")
			"""
			#########################
			cl = self.cl
			flags = pyopencl.mem_flags.COPY_HOST_PTR | pyopencl.mem_flags.READ_WRITE
			numberOfImages = self.training_data[1].shape[0]
			labels = self.training_data[1]
			images = self.training_data[0]
			n = 100
			for i in range(n):
				#print(labels[i])
				accuracy += self.miniBatch(network,images[x[i]],labels[x[i]],False)
				percentageCompleted = round((float(i)/float(n))*100)

				print(str(percentageCompleted)+"%")
				delete_last_lines()
			testAccuracy = str(int(round((float(accuracy)/float(n))*100)))
			print("Test accuracy is: " + testAccuracy+"%\n")
			#########################
			"""



			self.etaValue[0] *= 0.85
			self.etaValueBuffer = pyopencl.Buffer(cl.context,flags,hostbuf=self.etaValue)
			if accuracy == max(accuracyList):
				network.saveModel("Model.pkl")


		#self.miniBatch(network,self.inputImage)


	def miniBatch(self,network,inputImage,actualImageNumber,isTrain):
		t1 = time()
		cl = self.cl
		inputImageBuffer = self.cl.getBuffer(inputImage,"READ_ONLY")



		found = False
		desiredOutput = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
		for i in range(10):
			if i == actualImageNumber:
				desiredOutput[i] = 1.0
			else:
				desiredOutput[i] = 0.0

		if isTrain:
			blah = 30
		else:
			blah = 1
		for num in range(blah):
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
			if not isTrain:
				return accuracy
				break
			#if found:
			#	break

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
				errorBuffer = layer.backwardPropagate(errorBuffer,self.cl,0,self.etaValueBuffer,count,previousOutputBuffer)
			#self.etaValue = pyopencl.enqueue_read_buffer(self.cl.commandQueue,self.etaValueBuffer,self.etaValue).wait()
			#print(self.etaValue[0])




			return accuracy

		def getTestAccuracy(self,network):
			cl = self.cl
			flags = pyopencl.mem_flags.COPY_HOST_PTR | pyopencl.mem_flags.READ_WRITE
			numberOfImages = self.training_data[1].shape[0]
			labels = self.training_data[1]
			images = self.training_data[0]
			n = 100
			for i in range(n):
				#print(labels[i])
				accuracy += self.miniBatch(network,images[x[i]],labels[x[i]],False)
				percentageCompleted = round((float(i)/float(n))*100)
				delete_last_lines()
				print(str(percentageCompleted)+"%")
			return str(int(round((float(accuracy)/float(n))*100)))

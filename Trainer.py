from cl import CL
from time import time
import cv2 as cv
import numpy
from forwardPropagateLibrary import forwardPropagate
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
		forwardPropagate(self.inputImage,network,self.cl)
		t2 = time()
		

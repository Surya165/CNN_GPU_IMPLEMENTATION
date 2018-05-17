import random
import numpy
from Trainer import Trainer
from time import time
from cl import CL

class ConvolutionalNeuralNetwork:
	def __init__(self):
		self.layerStack = []
		self.allowedLayers = ['convLayer','flatten','maxPool','dense']
		print("Setting GPU")
		t1 = time()
		self.cl = CL()
		t2 = time()
		print("Time taken to set the GPU is "+str(round((t2-t1)*100000)/100)+"ms")


	def addLayer(self,layer):
		self.layerStack.append(layer)


	def printModel(self):
		for layer in self.layerStack:
			layer.printLayer()

	def compile(self,input):
		t1 = time()
		for count, layer in enumerate(self.layerStack):
			output = layer.compile(input,self.cl)
			input = output
		t2 = time()
		print("Time Taken to compile is "+str(round((t2-t1)*100000)/100)+"ms")
		

	def train(self,dataset=None,numberOfEpochs=1,miniBatchSize=1):
		trainer = Trainer(self.cl,dataset,numberOfEpochs=1,miniBatchSize=1)
		trainer.train(self)

import random
import numpy
from convolutionalLayer import ConvolutionalLayer
from maxPoolLayer import MaxPoolLayer
from denseLayer import Dense
from flattenLayer import Flatten
from compiler import Compiler
from random import shuffle
import _pickle as pkl
from Trainer import Trainer

class ConvolutionalNeuralNetwork:
	def __init__(self):
		self.layerStack = []
		self.allowedLayers = ['convLayer','flatten','maxPool','dense']



	def addLayer(self,layer):
		self.layerStack.append(layer)


	def printModel(self):
		for layer in self.layerStack:
			layer.printLayer()

	def compile(self,input):
		comp = Compiler(input)
		comp.compile(self)
		"""
		for count, layer in enumerate(self.layerStack):
			if count == 0:
				input = tuple([len(input)] + list(input))
			output = layer.compile(input)
			input = output

		return input"""

	def train(self,dataset=None,numberOfEpochs=1,miniBatchSize=1):
		trainer = Trainer(dataset,numberOfEpochs=1,miniBatchSize=1)
		trainer.train(self)






def main():
	model = ConvolutionalNeuralNetwork()
	model.addLayer(ConvolutionalLayer(30,(5,5)))
	model.addLayer(MaxPoolLayer((5,5)))
	model.addLayer(ConvolutionalLayer(100,(3,3)))
	model.addLayer(MaxPoolLayer((3,3)))
	model.addLayer(Flatten())
	model.addLayer(Dense(30))
	model.addLayer(Dense(100))

	f = open("trainingImage.pkl","rb")
	inputImage = pkl.load(f,encoding="latin1")
	model.compile(inputImage.shape)
	print("##########\n\n")
	model.printModel()
	model.train(inputImage,numberOfEpochs=1,miniBatchSize=1)
main()

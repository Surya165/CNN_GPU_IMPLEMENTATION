import random
import numpy
from convolutionalLayer import ConvolutionalLayer
from maxPoolLayer import MaxPoolLayer
from denseLayer import Dense
from flattenLayer import Flatten
from compiler import Compiler


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



def main():
	model = ConvolutionalNeuralNetwork()
	model.addLayer(ConvolutionalLayer(30,(5,5)))
	model.addLayer(MaxPoolLayer((5,5)))
	model.addLayer(Flatten())
	model.addLayer(Dense(30))
	model.printModel()

	inputImageShape = (28,28)
	model.compile(inputImageShape)
	print("##########\n\n")
	model.printModel()
main()

from convolutionalLayer import ConvolutionalLayer
from flattenLayer import Flatten
from denseLayer import Dense
from maxPoolLayer import MaxPoolLayer

class Compiler:
	def __init__(self,input):
		self.input = input

	def compile(self,network):
		for count, layer in enumerate(network.layerStack):
			self.input = self.getProcessedInput(layer)
			print(self.input)
			output = layer.compile(self.input)
			self.input = output

	def getLayerType(self,input):
		if isinstance(input,ConvolutionalLayer):
			return "convLayer"
		if isinstance(input,Dense):
			return "dense"
		if isinstance(input,MaxPoolLayer):
			return "maxPool"
		if isinstance(input,Flatten):
			return "flatten"
		if isinstance(input,tuple):
			return "tuple"
		return "undefined"

	def getProcessedInput(self,layer):
		previousLayerType = self.getLayerType(self.input)
		presentLayerType = self.getLayerType(layer)

		print("presentLayerType: "+presentLayerType)
		print("previousLayerType: "+previousLayerType)

		if presentLayerType == "maxPool":
			if previousLayerType == "convLayer":
				self.input = self.input.shape

			if previousLayerType == "maxPool":
				self.input = self.input.shape

		if presentLayerType == "convLayer":
			if previousLayerType == "tuple":
				self.input = tuple([layer.number]+list(self.input))
			if previousLayerType == "convLayer":
				self.input = self.input.shape
			if previousLayerType == "maxPool":
				self.input = input.shape
		if presentLayerType == "flatten":
			if previousLayerType == "maxPool":
				self.input =tuple([self.input.number]+list(self.input.shape))


		if presentLayerType =="dense":
			if previousLayerType == "flatten":
				self.input = self.input.shape
				
		return self.input

from convolutionalLayer import ConvolutionalLayer
from flattenLayer import Flatten
from denseLayer import Dense
from maxPoolLayer import MaxPoolLayer

class Compiler:
	def __init__(self,input):
		self.input = input

	def compile(self,network):
		for count, layer in enumerate(network.layerStack):
			#self.input = self.getProcessedInput(layer)
			if count == 0:
				self.input = tuple([layer.number] + list(self.input))
			output = layer.compile(self.input)
			self.input = output

		return self.input

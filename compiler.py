from convolutionalLayer import ConvolutionalLayer
from flattenLayer import Flatten
from denseLayer import Dense
from maxPoolLayer import MaxPoolLayer
from cl import CL

class Compiler:
	def __init__(self,input):
		self.input = input


	def compile(self,network,cl):
		for count, layer in enumerate(network.layerStack):
			output = layer.compile(self.input,cl)
			self.input = output

		return self.input

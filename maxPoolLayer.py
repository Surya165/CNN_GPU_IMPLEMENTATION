
import numpy

class FeatureMap:

	def __init__(self,kernelShape,previousLayerShape):
		self.shape = self.getShape(kernelShape,previousLayerShape)
		self.activationMatrix = numpy.zeros(self.shape)

	def getShape(self,kernelShape,previousLayerShape):

		shape = []
		for count,index in enumerate(previousLayerShape):
			shape.append(index - kernelShape[count] + 1)
		shape = tuple(shape)
		return shape


class MaxPoolLayer:

	def __init__(self,kernelShape):
		self.name = "maxPool"
		self.kernelShape = kernelShape
		self.isCompiled = False
		self.FeatureMap = []




	def compile(self,previousLayerShape):

		self.isCompiled = True
		self.FeatureMaps = []
		self.number = previousLayerShape[0]
		for i in range(self.number):
			self.FeatureMaps.append(FeatureMap(self.kernelShape,previousLayerShape[1:]))
		self.shape = self.FeatureMaps[0].shape
		return self


	def printLayer(self):
		print("maxPool Layer, kernelShape is " + str(self.kernelShape) )
		if self.isCompiled:
			print("shape of the activationMatrix is " + str(self.shape))

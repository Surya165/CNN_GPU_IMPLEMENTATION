import numpy



class FeatureMap:

	def __init__(self,kernelShape,previousLayerShape):
		self.kernelShape = kernelShape
		self.shape = self.getShape(previousLayerShape)
		self.weightMatrix = numpy.zeros(self.kernelShape)
		self.activationMatrix = numpy.zeros(self.shape)
		self.randomInitialise()

	def printWeightMatrix(self):
		print(self.weightMatrix)

	def getShape(self,previousLayerShape):
		shape = []
		for count,index in enumerate(previousLayerShape):
			shape.append(index - self.kernelShape[count] + 1)
		shape = tuple(shape)
		return shape

	def randomInitialise(self):
		mean = 0
		variance = 0.1
		totalNumberOfWeights = 1
		for index in self.kernelShape:
			totalNumberOfWeights *= index
		weightList = numpy.resize(self.weightMatrix,(totalNumberOfWeights,))
		for i in range(totalNumberOfWeights):
			weight = numpy.random.normal(mean,variance)
			weightList[i] = weight
		weightMatrix = numpy.resize(weightList,self.kernelShape)
		self.weightMatrix = weightMatrix
		#self.printWeightMatrix()


class ConvolutionalLayer:

	def __init__(self,number,kernelShape):
		self.name = "convLayer"
		self.number = number
		self.kernelShape = kernelShape
		self.isCompiled = False

	def printLayer(self):
		print("Convolutional Layer.")
		if self.isCompiled:
			print(str(self.shape))

	def compile(self,previousLayerShape):
		self.isCompiled = True
		self.FeatureMaps = []
		for i in range(self.number):
			self.FeatureMaps.append(FeatureMap(self.kernelShape,previousLayerShape[1:]))
		self.shape = tuple([self.number]+list(self.FeatureMaps[0].shape))
		return self

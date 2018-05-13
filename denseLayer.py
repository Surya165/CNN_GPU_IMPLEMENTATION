import numpy
class WeightMatrix:
	def __init__(self,previousLayerShape):
		self.weightMatrix = numpy.zeros(previousLayerShape)
		self.randomInitialise()
	def randomInitialise(self):
		mean = 0
		variance = 0.1
		totalNumberOfWeights = 1
		for index in self.weightMatrix.shape:
			totalNumberOfWeights *= index
		weightList = numpy.resize(self.weightMatrix,(totalNumberOfWeights,))
		for i in range(totalNumberOfWeights):
			weight = numpy.random.normal(mean,variance)
			weightList[i] = weight
		weightMatrix = numpy.resize(weightList,self.weightMatrix.shape)
		self.weightMatrix = weightMatrix
		pass
class Dense:
	def __init__(self,number):
		self.name = "dense"
		self.number = number
		self.activationMatrix = []
	def printLayer(self):
		print("Dense Layer. Number of Neurons = " + str(self.number))

	def compile(self,previousLayerShape):
		self.WeightMatrixList = []
		for i in range(self.number):
			self.WeightMatrixList.append(WeightMatrix(previousLayerShape))
		return self.number 

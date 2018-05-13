class Dense:
	def __init__(self,number):
		self.name = "dense"
		self.number = number
	def printLayer(self):
		print("Dense Layer. Number of Neurons = " + str(self.number))

	def compile(self,previousLayerShape):
		pass

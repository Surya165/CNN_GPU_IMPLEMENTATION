from convolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from convolutionalLayer import ConvolutionalLayer
from maxPoolLayer import MaxPoolLayer
from denseLayer import Dense
from flattenLayer import Flatten
import _pickle as pkl
import gzip

from time import sleep

def main():

	f = gzip.open('mnist.pkl.gz', 'rb')
	tr,va,te= pkl.load(f,encoding="latin1")
	f.close()
	data = tuple([tr,va,te])

	print("Waiting for three seconds")
	sleep(3)

	model = ConvolutionalNeuralNetwork()
	model.addLayer(ConvolutionalLayer(100,(3,3)))
	model.addLayer(MaxPoolLayer((3,3)))
	#model.addLayer(ConvolutionalLayer(100,(5,5)))
	#model.addLayer(MaxPoolLayer((5,5)))
	model.addLayer(Flatten())
	model.addLayer(Dense(30))
	model.addLayer(Dense(10))

	f = open("trainingImage.pkl","rb")
	inputImage = pkl.load(f,encoding="latin1")
	model.compile(inputImage.shape)
	print("##########\n\n")
	#model.printModel()
	numberOfEpochs = 1
	miniBatchSize = 1
	model.train(data,numberOfEpochs,miniBatchSize)



main()

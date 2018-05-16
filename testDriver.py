from convolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from convolutionalLayer import ConvolutionalLayer
from maxPoolLayer import MaxPoolLayer
from denseLayer import Dense
from flattenLayer import Flatten
import _pickle as pkl

from time import sleep

def main():

	print("Waiting for three seconds")
	sleep(3)
	model = ConvolutionalNeuralNetwork()
	model.addLayer(ConvolutionalLayer(30,(3,3)))
	model.addLayer(MaxPoolLayer((3,3)))
	model.addLayer(ConvolutionalLayer(100,(5,5)))
	model.addLayer(MaxPoolLayer((5,5)))
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

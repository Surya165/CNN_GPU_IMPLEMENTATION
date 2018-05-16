import numpy
import _pickle as pkl
import gzip
f = gzip.open('mnist.pkl.gz','rb')
training_data,validation_data,test_data = pkl.load(f,encoding="latin1")
f.close()
trainingImage = numpy.resize(training_data[0][0],(1,28,28))
print(trainingImage.shape)
f = open('trainingImage.pkl','wb')
pkl.dump(trainingImage,f)

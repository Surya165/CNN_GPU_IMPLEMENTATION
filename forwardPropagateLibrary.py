from cl import CL
from time import time
import cv2 as cv
import numpy
from time import sleep
import pyopencl
def forwardPropagate(inputImage,network,cl):

	kernelFiles = ["kernels/forwardPropagateConv.cl","kernels/forwardPropagateMaxPool.cl","kernels/forwardPropagateFlatten.cl"\
	,"kernels/forwardPropagateDense.cl"]

	print("Building forwardPropagate Program")

	programs = []

	t1 = time()
	for kernelFile in kernelFiles:
		programs.append(cl.getProgram(kernelFile))
	t2 = time()
	print("Time taken to build programs = "+str(round((t2-t1)*100000)/100)+"ms")

	buffer = cl.getBuffer(inputImage,"READ_ONLY")
	images = []
	t1 = time()
	for count,layer in enumerate(network.layerStack):
		if layer.name != "convLayer" and layer.name != "maxPool" and layer.name != "flatten":
			break
		buffer = runKernelForwardPropagate(programs,layer,buffer,cl,count)
	t2 = time()
	print("Estimated Time for each Epoch is " +str(round((t2-t1)*1400))+"s")



	#pass

def runKernelForwardPropagate(programs,layer,inputBuffer,cl,count):


	if(layer.name == "convLayer"):
		program = programs[0]
		inputShapeBuffer,outputBuffer,weightBuffer,weightShapeBuffer,biasBuffer = layer.getAttributeList()
		globalSize = layer.shape
		t1=time()
		program.convLayer(cl.commandQueue,globalSize,None,inputBuffer\
		,inputShapeBuffer,outputBuffer,\
		weightBuffer,weightShapeBuffer,biasBuffer).wait()
		t2=time()
		#cl.clear([inputBuffer,inputShapeBuffer])
		print("Time for convLayer is "+str(round((t2-t1)*100000)/100)+"ms")






	if(layer.name == "maxPool"):
		program = programs[1]
		outputBuffer,kernelShape = layer.getAttributeList()
		globalSize = layer.shape
		t1 = time()
		program.maxPool(cl.commandQueue,globalSize,None,inputBuffer,outputBuffer,kernelShape).wait()
		t2 = time()
		print("Time for maxPool is "+str(round((t2-t1)*100000)/100)+"ms")
		#cl.clear([inputBuffer,kernelShape])


	if(layer.name == "flatten"):
		program = programs[2]
		inputShapeBuffer,outputBuffer = layer.getAttributeList()
		globalSize = layer.shape
		t1 = time()
		program.flatten(cl.commandQueue,globalSize,None,inputBuffer,inputShapeBuffer,outputBuffer).wait()
		t2 = time()
		print("Time for flatten is "+str(round((t2-t1)*100000)/100)+"ms")
		#cl.clear([inputBuffer])


	if(layer.name == "dense"):
		program = programs[3]
		globalSize = layer.shape
		outputBuffer = cl.getBuffer(layer.outputBuffer,"READ_WRITE")
		weightBuffer = cl.getBuffer(layer.weightBuffer,"READ_ONLY")
		weightShapeBuffer = numpy.asarray(layer.weightBuffer.shape)
		weightShapeBuffer = cl.getBuffer(weightShapeBuffer,"READ_ONLY")
		program.dense(cl.commandQueue,globalSize,None,inputBuffer,outputBuffer,weightShapeBuffer).wait()

	return outputBuffer

def printBuffer(outputBuffer,shape,cl,type):
	if type == "int":
		image = numpy.zeros(shape,dtype=numpy.uint32)
	if type=="double":
		image = numpy.zeros(shape,dtype=numpy.float64)
	pyopencl.enqueue_read_buffer(cl.commandQueue,outputBuffer,image)
	print(image)

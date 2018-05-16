import time
import pyopencl as cl
import numpy
import math
import _pickle as pkl


kernelStr = "kernel void add(global float *a,global float *b, global float *c)\
{\
	int gid = get_global_id(0);\
	int gid2 = get_global_id(1);\
	int gid3 = get_global_id(2);\
	int gidSize1 = get_global_size(0);\
	if(gid == 0)\
	{\
		a[gid] = 0;\
	}\
	else\
	{\
		a[gid] = gidSize1;\
	}\
	if(gid2 == 0)\
	{\
		b[gid2] = 0;\
	}\
	else\
	{\
		b[gid2] = gid2;\
	}\
	if(gid3 == 0)\
	{\
		c[gid3] = 0 ;\
	}\
	else{c[gid3] = gid3;}\
}\
"

A = 30
def CreateContext():
	platforms = cl.get_platforms()
	devices = platforms[0].get_devices(cl.device_type.GPU)
	if len(devices) == 0:
		print("Wait for some time. The GPU is clearing the cache of the last program")
		exit(0)
	context = cl.Context([devices[0]])
	return context,devices[0]

t1 = time.time()
a = 0
#for i in range(A):
#	a = a + 1
t2 = time.time()
timeWithoutGPU = t2-t1

a = numpy.zeros((5,),dtype=numpy.float32)
b = numpy.zeros((5,),dtype=numpy.float32)
c = numpy.zeros((5,),dtype=numpy.float32)

context,device = CreateContext()
commandQueue = cl.CommandQueue(context,device)
program = cl.Program(context,kernelStr)
program.build(devices=[device])
mf = cl.mem_flags
a2 = a
b2 = b
c2 = c
a = cl.Buffer(context,mf.READ_WRITE,a.nbytes)
b = cl.Buffer(context,mf.READ_WRITE,b.nbytes)
c = cl.Buffer(context,mf.READ_WRITE,c.nbytes)
t3 = time.time()
print("Time taken for setting GPU is "+str(t3-t2))
program.add(commandQueue,(5,5,5),None,a,b,c)
cl.enqueue_read_buffer(commandQueue,a,a2).wait()
cl.enqueue_read_buffer(commandQueue,b,b2).wait()
cl.enqueue_read_buffer(commandQueue,c,c2).wait()
t4 = time.time()
timeWithGPU = t4-t3
print("The time gain is "+ str(round(timeWithoutGPU/(timeWithGPU))))
print("Time taken is : "+str(timeWithGPU))
print(a2,b2,c2)


image = pkl.load(open("trainingImage.pkl","rb"),encoding="latin1")
buffer = cl.Buffer(context,cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR,hostbuf=image)
cl.enqueue_read_buffer(commandQueue, buffer, image)
print(image)

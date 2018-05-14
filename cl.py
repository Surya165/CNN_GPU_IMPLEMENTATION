import pyopencl as cl
import numpy
class CL:
	def __init__(self):
		self.context , self.device = self.createContextAndDevice()
		self.commandQueue = cl.CommandQueue(self.context,self.device)

	def createContextAndDevice(self):
		platforms = cl.get_platforms();
		if len(platforms) == 0:
			print ("Failed to find any OpenCL platforms.")
			return None

	    # Next, create an OpenCL context on the first platform.  Attempt to
	    # create a GPU-based context, and if that fails, try to create
	    # a CPU-based context.
		devices = platforms[0].get_devices(cl.device_type.GPU)
		if len(devices) == 0:
			print ("Could not find GPU device, trying CPU...")
			devices = platforms[0].get_devices(cl.device_type.CPU)
			if len(devices) == 0:
				print ("Could not find OpenCL GPU or CPU device.")
				return None

	    # Create a context using the first devic
		context = cl.Context([devices[0]])
		return context, devices[0]

	def CreateProgram(self,context, device, fileName):
		kernelFile = open(fileName, 'r')
		kernelStr = kernelFile.read()

	    # Load the program source
		program = cl.Program(context, kernelStr)

	    # Build the program and check for errors
		program.build(devices=[device])
		return program
	def getProgram(self,kernelFile):
		self.program = self.CreateProgram(self.context,self.device,kernelFile)
		return self.program

	def getBuffer(self,input,mem_flag):
		inputSize = numpy.product(list(input.shape))
		input = numpy.resize(input,(inputSize,))
		if(mem_flag == "READ_WRITE"):
			flag = cl.mem_flags.READ_WRITE
		if(mem_flag == "READ_ONLY"):
			flag = cl.mem_flags.READ_ONLY
		if(mem_flag == "WRITE_ONLY"):
			flag = cl.mem_flags.WRITE_ONLY
		input = cl.Buffer(self.context,flag,input.nbytes)
		return input

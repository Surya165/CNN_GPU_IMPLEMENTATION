#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void forwardPropagate(
		global double *inputBuffer,
		global double *weightBuffer,
		global int *weightShapeBuffer,
		global double *biasBuffer,
		global double * outputBuffer)
{


	int numberOfNeurons = get_global_size(0);
	int numberOfInputNeurons = get_global_size(1);

	int outputIndex = get_global_id(0);
	int inputIndex = get_global_id(1);
	int weightBufferIndex = outputIndex*numberOfInputNeurons+inputIndex;

	double weight = weightBuffer[weightBufferIndex];
	double bias = biasBuffer[outputIndex];
	outputBuffer[outputIndex] += inputBuffer[inputIndex]*weight;
	if(inputIndex == 0)
	{
		outputBuffer[outputIndex] += bias;
		outputBuffer[outputIndex] = 1/(1+(double)exp(-1*outputBuffer[outputIndex]));
	}
}


/*
kernel void backwardPropagate
(
	global double *errorBuffer,
	global double *weightBuffer,
	global int *weightShapeBuffer,
	global double outputBuffer
)
{

}
*/

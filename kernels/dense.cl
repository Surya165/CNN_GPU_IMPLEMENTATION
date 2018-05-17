#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void forwardPropagate(
		global double *inputBuffer,
		global double *weightBuffer,
		global int *weightShapeBuffer,
		global double *biasBuffer,
		global double *outputBuffer
		)
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




kernel void backwardPropagate
(
	global double *errorBuffer,
	global double *weightBuffer,
	global int *weightShapeBuffer,
	global double *biasBuffer,
	global double *outputBuffer,
	global int *trainingParams,
	global double *nextErrorBuffer,
	global double *previousOutputBuffer
)
{


	int epochCount = trainingParams[2] / 100;

	float eta = 9.0;


	int numberOfNeurons = get_global_size(0);
	int numberOfInputNeurons = get_global_size(1);

	int outputIndex = get_global_id(0);
	int inputIndex = get_global_id(1);
	int weightIndex = outputIndex*numberOfInputNeurons+inputIndex;






	//based on the equation errj = oj(1-oj)(ei),deltawij = eta*errij*oi,deltabias = eta*errj
	double Oj = outputBuffer[outputIndex];
	double Ei = errorBuffer[outputIndex];

	double Oi = previousOutputBuffer[inputIndex];
	double Ej = Oj * (1-Oj) *Ei;
	if(inputIndex == 0)
	{
		double deltaBias = eta*Ej;
		biasBuffer[outputIndex] += deltaBias;
	}
	weightBuffer[weightIndex] += eta*Ej*Oi;


	nextErrorBuffer[inputIndex] += eta*Ej*Oi;



}

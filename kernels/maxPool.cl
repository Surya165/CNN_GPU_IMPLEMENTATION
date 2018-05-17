#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void forwardPropagate
(
	global double* inputBuffer,
	global double* outputBuffer,
	global int * kernelShape,
	global int* maxIndexBuffer
)
{

	int mNew = get_global_id(0);
	int pNew = get_global_id(1);
	int nNew = get_global_id(2);

	int mNewSize = get_global_size(0);
	int nNewSize = get_global_size(1);
	int pNewSize = get_global_size(2);


	int mOldSize = mNewSize;
	int nOldSize = nNewSize + kernelShape[0] - 1;
	int pOldSize = pNewSize + kernelShape[1] - 1;

	int i;
	int j;



	int totalRowsInFeatureMap = get_global_size(1);
	int totalColsInFeatureMap = get_global_size(2);

	int index = mNew * totalRowsInFeatureMap* totalColsInFeatureMap + nNew * totalColsInFeatureMap + pNew;
	int input = 0;
	int inputIndex1D = 0;


	int max = 0;
	for ( i = 0; i < kernelShape[0]; i ++)
	{
		for ( j = 0; j < kernelShape[1]; j ++)
		{

			inputIndex1D = mNew*nOldSize*pOldSize+(nNew+i)*pOldSize+pNew+j;

			input = inputBuffer[inputIndex1D];
			if(max < input)
			{
				max = input;
				maxIndexBuffer[index*2] = i;
				maxIndexBuffer[index*2+1] = j;
			}

		}
	}
	outputBuffer[index] = max;


}


kernel void backwardPropagate
(
	global double* errorBuffer,
	global double* nextErrorBuffer,
	global int*  maxIndexBuffer,
	global int*previousLayerShape
)
{
	int filterMapNumber = get_global_id(0);
	int i = get_global_id(1);
	int j = get_global_id(2);

	int numberOfRows = get_global_size(1);
	int numberOfCols = get_global_size(2);

	int numberOfRowsInPreviousLayer = previousLayerShape[1];
	int numberOfColsInPreviousLayer = previousLayerShape[2];

	int outputIndex = filterMapNumber*numberOfRows*numberOfCols+i*numberOfCols+j;
	int maxIndexI = maxIndexBuffer[2*outputIndex];
	int maxIndexJ = maxIndexBuffer[2*outputIndex+1];
	nextErrorBuffer[filterMapNumber*numberOfRowsInPreviousLayer*numberOfColsInPreviousLayer+
	maxIndexI*numberOfColsInPreviousLayer+maxIndexJ
	] = errorBuffer[outputIndex];


}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void forwardPropagate
(
	global double *inputBuffer,
	global int *inputBufferShape,
	global double * outputBuffer,
	global double *weightBuffer,
	global int *weightShapeBuffer,
	global double  *biasBuffer
)
{
	int i,j,k;

	int mNew = get_global_id(0);
	int nNew = get_global_id(1);
	int pNew = get_global_id(2);


	int mOld = weightShapeBuffer[3];
	int nWeights = weightShapeBuffer[1];
	int pWeights = weightShapeBuffer[2];

	int mOldSize = inputBufferShape[0];
	int nOldSize = inputBufferShape[1];
	int pOldSize = inputBufferShape[2];

	int totalRowsInFeatureMap = get_global_size(1);
	int totalColsInFeatureMap = get_global_size(2);

	int index = mNew * totalRowsInFeatureMap* totalColsInFeatureMap + nNew * totalColsInFeatureMap + pNew;




 	int weightIndex = 0;
	int weight = 0;
	int inputIndex3D[3] = {0,0,0};
	int inputIndex1D = 0;
	int input = 0;
	int weightedInput = 0;




	for( i = 0; i < nWeights; i ++)
	{
		for( j = 0; j < pWeights; j ++)
		{

			for ( k = 0; k < mOld; k ++)
			{

				weightIndex = mNew*nWeights*pWeights*mOld + i * pWeights*mOld + j * mOld + k;
				weight = weightBuffer[weightIndex];
				inputIndex3D[0] = k;
				inputIndex3D[1] = i + nNew;
				inputIndex3D[2] = j + pNew;

				inputIndex1D = inputIndex3D[0]*nOldSize*pOldSize+inputIndex3D[1]*pOldSize+inputIndex3D[2];

				input = inputBuffer[inputIndex1D];
				outputBuffer[index] = inputIndex1D;
				break;

				weightedInput += input*weight;
				outputBuffer[index] += weightedInput;


			}
		}

	}





/*

  if(outputBuffer[index]<0)
	{
		outputBuffer[index] = 0;
	}
	else
	{}
	*/
	outputBuffer[index] = outputBuffer[index] - biasBuffer[index];
	outputBuffer[index] = 1/(1+(double)exp(-1*outputBuffer[index]));
	//outputBuffer[index] = index;









}



kernel void backwardPropagate
(
	global double *errorBuffer,
	global double *weightBuffer,
	global int *weightShapeBuffer,
	global double *biasBuffer,
	global double *outputBuffer,
	global int* trainingParams,
	global double *nextErrorBuffer,
	global double *previousOutputBuffer,
	global int* previousLayerShape
)
{
	int filterMapNumber = get_global_id(0);
	int i = get_global_id(1);
	int j = get_global_id(2);


	int epochCount = trainingParams[2] / 100;
	float etaValue = 9.0;



	int numberOfRowsInKernel = weightShapeBuffer[1];
	int numberOfColsInKernel = weightShapeBuffer[2];

	int outputIndex = filterMapNumber*numberOfRowsInKernel*numberOfColsInKernel+i*numberOfColsInKernel+j;
	int inputNumberOfRows;
	int inputNumberOfCols;

	inputNumberOfRows = previousLayerShape[1];
	inputNumberOfCols = previousLayerShape[2];

	int inputIndexI;
	int inputIndexJ;

	//only considering the kernel at point (0,0)
	inputIndexI = i;
	inputIndexJ =j;

	double Oj = outputBuffer[outputIndex];
	double Ek = errorBuffer[outputIndex];
	double Ej = Oj*(1-Oj)*Ek;
	double Oi;
	int inputIndex;
	for ( int inputFilterMapNumber = 0; inputFilterMapNumber < previousLayerShape[0]; inputFilterMapNumber ++)
	{
		inputIndex = inputFilterMapNumber * inputNumberOfRows * inputNumberOfCols + inputIndex*inputNumberOfCols+j;
		Oi = previousOutputBuffer[inputIndex];
		weightBuffer[outputIndex] += etaValue*Ej*Oi;

	}
	if(i == 0 && j == 0)
	{
		biasBuffer[filterMapNumber] += etaValue*Ej;
		biasBuffer[filterMapNumber] = 0;
	}

	biasBuffer[filterMapNumber] = 0;

}

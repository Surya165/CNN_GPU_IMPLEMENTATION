#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void convLayer( __global float *inputBuffer,global int *inputBufferShape,__global float* outputBuffer,__global float *weightBuffer,__global int *weightShapeBuffer)
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
				inputIndex3D[0] = k + mNew;
				inputIndex3D[1] = i + nNew;
				inputIndex3D[2] = i + pNew;

				inputIndex1D = inputIndex3D[0]*nOldSize*pOldSize+inputIndex3D[1]*pOldSize+inputIndex3D[2];

				input = inputBuffer[inputIndex1D];

				weightedInput += input*weight;

			}
		}
	}
	float temp = -1 * weightedInput;
  temp = exp(temp);
	temp += 1;
	temp = 1/ temp;
	outputBuffer[index] = temp;

}






kernel void maxPool(global float* inputBuffer, global float* outputBuffer)
{

	int mNew = get_global_id(0);
	int pNew = get_global_id(1);
	int nNew = get_global_id(2);
}






kernel void flatten(global float *inputBuffer,global float * outputBuffer)
{

}





kernel void dense(global float*inputBuffer, global float * outputBuffer, global int*weightShapeBuffer)
{

}

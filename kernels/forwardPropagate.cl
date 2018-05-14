kernel void convLayer( __global float *inputBuffer,__global float* outputBuffer,__global float *weightBuffer,__global int *weightShapeBuffer)
{
	int i,j,k;

	int mNew = get_global_id(0);
	int nNew = get_global_id(1);
	int pNew = get_global_id(2);

	int mOld = weightShapeBuffer[3];
	int nWeights = weightShapeBuffer[1];
	int pWeights = weightShapeBuffer[2];

	int totalRowsInFeatureMap = get_global_size(1);
	int totalColsInFeatureMap = get_global_size(2);

	int index = mNew * totalRowsInFeatureMap* totalColsInFeatureMap + nNew * totalColsInFeatureMap + pNew;
 	int weightIndex = 0;
	int weight = 0;
	for( i = 0; i < nWeights; i ++)
	{
		for( j = 0; j < pWeights; j ++)
		{
			for ( k = 0; k < mOld; k ++)
			{
				weightIndex = mNew*nWeights*pWeights*mOld + i * pWeights*mOld + j * mOld + k;
				weight = weightBuffer[weightIndex];
			}
		}
	}

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

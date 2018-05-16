kernel void maxPool(global float* inputBuffer, global float* outputBuffer, global int * kernelShape)
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
			}

		}
	}
	outputBuffer[index] = max;
	outputBuffer[index] = inputBuffer[index];


}

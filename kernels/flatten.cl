#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void forwardPropagate(global double *inputBuffer,global int*inputShapeBuffer,global double * outputBuffer)
{

	int gid = get_global_id(0);
	int number = inputShapeBuffer[0];
	int rows = inputShapeBuffer[1];
	int cols = inputShapeBuffer[2];
	int index = gid;

	outputBuffer[gid] = 0;
	int mOld = 0;
	int pOld = 0;
	int k = 0;

	int mNew = gid / rows;
	int nNew = gid % rows;

	int previousIndex = 0;
	for (int i = 0; i < number; i ++)
	{
		previousIndex = number*rows*cols+mNew*cols+nNew;
		outputBuffer[index] = inputBuffer[previousIndex];
	}



}


kernel void backwardPropagate
(
	global double *errorBuffer,
	global double *nextErrorBuffer
)
{

	int inputFilterMapNumber = get_global_id(0);
	int i = get_global_id(1);
	int j = get_global_id(2);

	int numberOfRows = get_global_size(1);
	int numberOfCols = get_global_size(2);
	int numberOfInputFilterMaps = get_global_size(0);

	int nextErrorBufferIndex = inputFilterMapNumber*numberOfRows*numberOfCols+i*numberOfCols+j;
	int errorBufferIndex = i*numberOfRows+j;
	nextErrorBuffer[nextErrorBufferIndex] = errorBuffer[errorBufferIndex]/numberOfInputFilterMaps;

}

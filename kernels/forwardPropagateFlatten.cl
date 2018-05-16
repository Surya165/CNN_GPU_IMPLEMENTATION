#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void flatten(global double *inputBuffer,global int*inputShapeBuffer,global double * outputBuffer)
{

	int gid = get_global_id(0);
	int number = inputShapeBuffer[0];
	int rows = inputShapeBuffer[1];
	int cols = inputShapeBuffer[2];
	int index = gid;

	outputBuffer[gid] = 0;
	int mOld = 0;
	int nOld = 0;
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

//This function finds the change in the X or Y from the originally guessed centeral position using the profile we created
__global__ void FindDelta(int *x, int cross, int numofimages, int numofbeads, cufftComplex *AvgXProfile, float *xout, bool *beadstatus)
{
	int ImageID = threadIdx.x + blockIdx.x * blockDim.x;
	int BeadID = threadIdx.y + blockIdx.y * blockDim.y;
if ((ImageID < numofimages) && (BeadID < numofbeads) && beadstatus[BeadID])
{
	int Xmax = 0;
	for(int i = 0; i < cross; i++)
	{
		//we can probably remove this normilization in the end since finding delta is not dependant on this
		AvgXProfile[i+(ImageID*cross*numofbeads)+(BeadID*cross)].x /= cross*numofimages;

		//Store the index of the max value in Xmax and Ymax
		//NOTE XMAX IS DISPLACEMENT FROM TID*CROSS
		if(AvgXProfile[Xmax+(ImageID*cross*numofbeads)+(BeadID*cross)].x < AvgXProfile[i+(ImageID*cross*numofbeads)+(BeadID*cross)].x)
			Xmax = i;
	}

	float solutionX[3] = {0};

	//Determines size of subarray, you will need to adjust this if you want more or less elements.
	int size = 3;
	float *subinX = new float[size*2];

	//These will be our subarrays to pass into quadfit
	for(int i=0; i < size; i++)
	{
		subinX[i*2+0] = (int)((Xmax - (size/2))+i);
		subinX[i*2+1] = AvgXProfile[(int)((ImageID*cross*numofbeads)+(BeadID*cross) + ((Xmax - (size/2))+i))].x;
	}

	QuadFit(size, subinX, solutionX);


	xout[((ImageID*numofbeads) + BeadID)] = x[BeadID] + (-(solutionX[1]/(2*solutionX[0])) - (cross/2))/2;


	delete[] subinX;

} else if ((ImageID < numofimages) && (BeadID < numofbeads) && !beadstatus[BeadID])
{
	xout[((ImageID*numofbeads) + BeadID)] = 0;
}
}
//this will make a profile of the bead using a + cross along the x and y axis at the approximate center of the bead

__global__ void AvgYCrossAndPrep(int *x, int *y, int cross, int crossthickness, int numofimages, int numofbeads, int numrow, int numcol, unsigned char *image, cufftComplex *AvgYProfile, bool *beadstatus)
{
	int ImageID = threadIdx.x + blockIdx.x * blockDim.x;
	int BeadID = threadIdx.y + blockIdx.y * blockDim.y;
if ((ImageID < numofimages) && (BeadID < numofbeads) && beadstatus[BeadID])
{
	int imagesize = numrow * numcol;

	//int crossthickness = 10; //Using 10px Cross Thickness // Now inputting cross thickness externally
	//In the Original Code they use cross * .6 and cross * 1.2, why not just cross * .5 and cross?
	//Starting from the center, find the top left corner of subarray.
	int RowStart = (int)(y[BeadID] - (cross * .5));
	int XSliceStart = (int)(x[BeadID] - (crossthickness * .5));
	
	//Sum Slices Along X -> AvgYProfile
	//-Notice i and j flipped in loops, but i still means Rows and j means Cols
	for(int i = 0; i < cross; i++)
	{
		AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x = 0;
		AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].y = 0;
		for (int j = 0; j <crossthickness; j++)
		{
			AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x += image[(ImageID*imagesize)+((i+RowStart) * numcol) + (j+XSliceStart)];
		}
		AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x /= crossthickness;
	}

	//This brings the graph down to zero by finding the average height under the first and last quarter of the graph, 
	//and then subtracting all points by this amount.
	double Yelvfix =0;

	double *CosWindow = new double[cross];
	for (int i=0; i < ceil((float)cross/2); i++)
	{
		CosWindow[i] = (cos(-PI + i*(2*PI/cross)) + 1)/2;
		CosWindow[cross-1-i] = CosWindow[i];
	}

	//Adds all the elevation from the first and last quarter
	for(int i = 0; i < cross/4; i++)
	{
		Yelvfix += AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x + AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+cross - 1 - i].x;
	}
	//Finds Average Elevation
	Yelvfix /= (cross/2);
	//Subtracks all values by the average elevation
	for(int i = 0; i < cross; i++)
	{
		AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x = (AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x - Yelvfix) * CosWindow[i];
	}
	
	delete[] CosWindow;
} else if((ImageID < numofimages) && (BeadID < numofbeads) && !beadstatus[BeadID])
{
	for(int i = 0; i < cross; i++)
	{
		AvgYProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x = 0;
	}
}
}
//this will make a profile of the bead using a + cross along the x and y axis at the approximate center of the bead

__global__ void AvgXCrossAndPrep(int *x, int *y, int cross, int crossthickness, int numofimages, int numofbeads, int numrow, int numcol, unsigned char *image, cufftComplex *AvgXProfile, bool *beadstatus)
{
	int ImageID = threadIdx.x + blockIdx.x * blockDim.x;
	int BeadID = threadIdx.y + blockIdx.y * blockDim.y;
if ((ImageID < numofimages) && (BeadID < numofbeads) && beadstatus[BeadID])
{
	int imagesize = numrow * numcol;

	//int crossthickness = 10; //Using 10px Cross Thickness // Now inputting cross thickness externally
	//In the Original Code they use cross * .6 and cross * 1.2, why not just cross * .5 and cross?
	//Starting from the center, find the top left corner of subarray.
	int ColStart = (int)(x[BeadID] - (cross * .5));
	int YSliceStart = (int)(y[BeadID] - (crossthickness * .5));

	//Sum Slices Along Y -> AvgXProfile
	for(int j = 0; j < cross; j++)
	{
		AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+j].x = 0;
		AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+j].y = 0;
		for (int i = 0; i < crossthickness; i++)
		{
			AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+j].x += image[(ImageID*imagesize)+((i+YSliceStart) * numcol) + (j+ColStart)];
		}
		//Finds the Average
		AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+j].x /= crossthickness;
	}
	

	//This brings the graph down to zero by finding the average height under the first and last quarter of the graph, 
	//and then subtracting all points by this amount.
	double Xelvfix = 0;

	double *CosWindow = new double[cross];
	for (int i=0; i < ceil((float)cross/2); i++)
	{
		CosWindow[i] = (cos(-PI + i*(2*PI/cross)) + 1)/2;
		CosWindow[cross-1-i] = CosWindow[i];
	}

	//Adds all the elevation from the first and last quarter
	for(int i = 0; i < cross/4; i++)
	{
		Xelvfix += AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x + AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+cross - 1 - i].x;
	}
	//Finds Average Elevation
	Xelvfix /= (cross/2);
	//Subtracks all values by the average elevation
	for(int i = 0; i < cross; i++)
	{
		AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x = (AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x - Xelvfix) * CosWindow[i];
	}
	
	delete[] CosWindow;
} else if((ImageID < numofimages) && (BeadID < numofbeads) && !beadstatus[BeadID])
{
	for(int i = 0; i < cross; i++)
	{
		AvgXProfile[(ImageID*numofbeads*cross)+(BeadID*cross)+i].x = 0;
	}
}
}
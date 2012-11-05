//This function serves to check the results of the tracking are within reasonable bounds.
//Obviously if numbers go negative or something simliar, the tracking program goofed, and that bead should stop being tracked.
__global__ void CheckBead (int numofimages, int numofbeads, int numrows, int numcols, int cross, int numofslices, bool* beadstatus, float* xout, float* yout, float* zout)
{
	int ImageID = threadIdx.x + blockIdx.x * blockDim.x;
	int BeadID = threadIdx.y + blockIdx.y * blockDim.y;
if ((ImageID < numofimages) && (BeadID < numofbeads) && beadstatus[BeadID])
{
	if(	   (numcols - (cross/2)) < xout[(ImageID*numofbeads)+BeadID]		//If the center of the bead is farther than cross/2 from the edge,
			|| (cross/2) > xout[(ImageID*numofbeads)+BeadID]				//of the frame then we should stop because our tracking program relies
			|| (numrows - (cross/2)) < yout[(ImageID*numofbeads)+BeadID]	//data that will not exist!
			|| (cross/2) > yout[(ImageID*numofbeads)+BeadID]				//
			|| 1 > zout[(ImageID*numofbeads)+BeadID]						//Obvoiusly the Z position cannot be negative
			|| (numofslices - 1) < zout[(ImageID*numofbeads)+BeadID] )		//If the bead is too high, we cannot accurately say its poition since the quadratic fit
																			//depends  on the surrounding profiles both above and below.
	{
		//Set Bead as a Bad Bead
		beadstatus[BeadID] = false;
	}
}
}
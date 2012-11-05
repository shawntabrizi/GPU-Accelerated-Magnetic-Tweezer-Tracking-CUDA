//This program will analyze 'all' pixels of our subimage, and create a radial profile of the intensity
//'all' is in quotes because we actually only care about pixels that will form full and complete circles thus corners will be ignored to a certain extent


//NOTE we have a bit of discrepency in the last values for the radial profile
//also we might want to include a check for acessing image data out of bounds.
__global__ void CalculateRadialProfile2(int cross, int numofimages, int numofbeads, int numcols, int numrows, float *x, float *y, cufftComplex *radprofileout, unsigned char *image, bool *beadstatus, float *radinprofile)
{
	int ImageID = threadIdx.x + blockIdx.x * blockDim.x;
	int BeadID = threadIdx.y + blockIdx.y * blockDim.y;
if ((ImageID < numofimages) && (BeadID < numofbeads) && beadstatus[BeadID])
{
	for(int i = 0; i < (cross/2); i++)
	{
		//divide the intensity profile by the total weight to normalize the profile
		if (radinprofile[i*2+1] > 0)
			radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1+i].x = (radinprofile[((ImageID*numofbeads*cross) + BeadID*cross)+i*2]/radinprofile[((ImageID*numofbeads*cross) + BeadID*cross)+(i*2)+1]);
		else
			radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1+i].x = 0;
		radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1-i].x = radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1+i].x;
		radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1+i].y = 0;
		radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1-i].y = 0;
	}
} else if ((ImageID < numofimages) && (BeadID < numofbeads) && !beadstatus[BeadID])
{
	for(int i = 0; i < (cross/2); i++)
	{
		//divide the intensity profile by the total weight to normalize the profile
		radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1+i].x = 0;
		radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1-i].x = 0;
		radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1+i].y = 0;
		radprofileout[((ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)))+(cross/2)-1-i].y = 0;
	}
}
}
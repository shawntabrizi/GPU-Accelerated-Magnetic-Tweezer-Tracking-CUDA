//This program will analyze 'all' pixels of our subimage, and create a radial profile of the intensity
//'all' is in quotes because we actually only care about pixels that will form full and complete circles thus corners will be ignored to a certain extent


//NOTE we have a bit of discrepency in the last values for the radial profile
//also we might want to include a check for acessing image data out of bounds.
__global__ void CalculateRadialProfile1(int cross, int numofimages, int numofbeads, int numcols, int numrows, float *x, float *y, cufftComplex *radprofileout, unsigned char *image, bool *beadstatus, float *radinprofile)
{
	int ImageID = threadIdx.x + blockIdx.x * blockDim.x;
	int BeadID = threadIdx.y + blockIdx.y * blockDim.y;
	int ElementID = threadIdx.z;
if ((ImageID < numofimages) && (BeadID < numofbeads) && beadstatus[BeadID] && (ElementID < cross))
{
	//This is where our radial profile will be stored
	//radinprofile[radius][0] = sum of intensity
	//radinprofile[radius][1] = sum of partial weight
	radinprofile[((ImageID*numofbeads*cross) + BeadID*cross)+ElementID]=0;
	__syncthreads();

	//we start analyzing at cross/2 before the central x and y position
	//ceil rounds up to next highest integer
	int xstart = (int)(ceil(x[((ImageID*numofbeads) + BeadID)]) - (cross/2));
	int ystart = (int)(ceil(y[((ImageID*numofbeads) + BeadID)]) - (cross/2));

	float xfrac = x[((ImageID*numofbeads) + BeadID)] - ceil(x[((ImageID*numofbeads) + BeadID)]);
	float yfrac = y[((ImageID*numofbeads) + BeadID)] - ceil(y[((ImageID*numofbeads) + BeadID)]);


		for(int j = 0; j < cross; j++)
		{
			//note that we must subtract the center location, cross/2 from the index to calulate radius from the central location
			float radius = sqrt((pow(((cross/2)+yfrac-ElementID),2) + pow(((cross/2)+xfrac-j),2)));
			//skip points on the corners that do not form complete circles
			// and skip points in the center that is within forget radius
			//if (forgetradius < radius && radius < (cross/2)) DOES SKIPPING THE FORGET RADIUS AFFECT ANYTHING?
			if (radius < (cross/2))
			{
				float radfloor;
				float radfrac = modf(radius, &radfloor);
				//Add a weighted intensity profile and the weighting value
				//Atomic Add is the safe way to add using multiple threads
				atomicAdd(radinprofile+((ImageID*numofbeads*cross) + BeadID*cross)+(int)radfloor*2+0, ((1-radfrac) * image[ImageID*(numcols*numrows) + ((ystart+ElementID) * numcols) + (xstart+j)]));
				atomicAdd(radinprofile+((ImageID*numofbeads*cross) + BeadID*cross)+(int)radfloor*2+1, (1-radfrac));
				//this fixes an error when radius is 63.X and then it tried to assign a value to radinprofile[64][0] which doesnt exist
				if ((radfloor + 1) < (cross/2))
				{
					atomicAdd(radinprofile+((ImageID*numofbeads*cross) + BeadID*cross)+(int)(radfloor + 1)*2+0, (radfrac * image[ImageID*(numcols*numrows) + ((ystart+ElementID) * numcols) + (xstart+j)]));
					atomicAdd(radinprofile+((ImageID*numofbeads*cross) + BeadID*cross)+(int)(radfloor + 1)*2+1, radfrac);
				}
			}
		}
	
	
} else if ((ImageID < numofimages) && (BeadID < numofbeads) && !beadstatus[BeadID] && (ElementID < cross))
{
	//Do Nothing
}
}
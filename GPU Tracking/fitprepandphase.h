//this finds the least squares fit of the prepped I(r) to find the index of the closest slice


__global__ void FitPrepandPhase (int cross, int forgetradius, int numofimages, int numofbeads, int RealRow, int RealCol, int AmpRow, int AmpCol, int CorkRow, int CorkCol, cufftComplex* radinprofile, float *Real, float *amplitudes, cufftComplex *corkscrews, float *zout, bool* beadstatus)
{
	int ImageID = threadIdx.x + blockIdx.x * blockDim.x;
	int BeadID = threadIdx.y + blockIdx.y * blockDim.y;
if ((ImageID < numofimages) && (BeadID < numofbeads) && beadstatus[BeadID])
{
	float *prep = new float[((cross/2)-forgetradius)];
	cufftComplex *corkscrew = new cufftComplex[((cross/2)-forgetradius)];
	for(int i=0; i < ((cross/2)-forgetradius); i++)
	{
		prep[i] = radinprofile[(ImageID*numofbeads*(cross-1))+(BeadID*(cross-1))+ (((cross/2)-forgetradius)-1)-i].x/(cross-1);
		corkscrew[i].x = radinprofile[(ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)) + (((cross/2)-forgetradius)-1)-i].x/(cross-1);
		corkscrew[i].y = -(radinprofile[(ImageID*numofbeads*(cross-1))+(BeadID*(cross-1)) + (((cross/2)-forgetradius)-1)-i].y/(cross-1));
	}
	
	int index=0;
	float *Arr = new float[RealRow];
	for (int i=0; i < RealRow; i++)
	{
		Arr[i]=0;
		for (int j=0; j < RealCol; j++)
		{
			Arr[i] += pow((prep[j] - Real[(BeadID * RealRow * RealCol) + ((i * RealCol) + j)]), 2);
		}
		if(Arr[i] < Arr[index])
			index = i;
	}
	delete[] Arr;
	delete[] prep;
	/* Merge 2 functions
}

//Using Complex numbers:
//complex in polar form is x = r * e(i* theta)
//abs(x) = r;
//arg(x) = theta;
__device__ void PhaseInNBHD (int cross, int forgetradius, int bestfit, float *phases, Array2dHandle amplitudes, complex<float> *radinprofile, CArray2dHandle corkscrews)
{ */
	int size = 5;
	float *phases = new float[size];
	int indexstart = (int)(index - (size/2));
	int length = ((cross/2)-forgetradius);
	float temp;

	for(int i=0; i < size; i++)
	{
		float sum1=0, sum2=0;
		for(int j=0; j < length; j++)
		{
			float a = corkscrew[j].x;
			float b = corkscrew[j].y;
			float c = corkscrews[(BeadID * CorkRow * CorkCol) + (((i+indexstart) * CorkCol) + j)].x;
			float d = corkscrews[(BeadID * CorkRow * CorkCol) + (((i+indexstart) * CorkCol) + j)].y;
			float e = amplitudes[(BeadID * AmpRow * AmpCol) + (((i+indexstart) * AmpCol) + j)];
			temp = sqrt(a*a + b*b) * e;
			sum1 += temp;
			sum2 += temp * atan((b*c - a*d)/(a*c + b*d));
		}
		phases[i]=sum2/sum1;
		//phases[i] = i;
	}

	delete[] corkscrew;


	/*Merge 2 functions
}

//This takes a descrete value for the best fit cal image index, and then uses a quadratic fit
//ong the phase information to find a non integer more accurate value for the best fit z height

__device__ float QuadFitPhase (int bestfit, float *phases)
{
*/
	float *data = new float[size*2];
	float solution[3] = {0};
	for(int i=0; i < size; i++)
	{
		//note that phases is the domain
		data[i*2+0] = phases[i];
		data[i*2+1] = (i - floor((float)size/2));
	}

	QuadFit(size, data, solution);
	zout[(ImageID*numofbeads)+BeadID] = (index + solution[2]);
	delete[] phases;
	delete[] data;
}
else if((ImageID < numofimages) && (BeadID < numofbeads) && !beadstatus[BeadID])
{
	zout[(ImageID*numofbeads)+BeadID] = 0;
}
}
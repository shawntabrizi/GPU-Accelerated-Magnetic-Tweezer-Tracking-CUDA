//cosine bandpass application

__global__ void CosBandPass (int cross, int numofimages, int numofbeads, cufftComplex *radinprofile, float *cosband, int CosBandLength)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < numofimages*numofbeads*(cross-1))
{
	//int imagenum = tid/(numofbeads*(cross-1));
	int beadnum = floor((float)(tid%(numofbeads*(cross-1)))/(cross-1));
	radinprofile[tid].x = radinprofile[tid].x * cosband[beadnum*CosBandLength+tid%(cross-1)];
	radinprofile[tid].y = radinprofile[tid].y * cosband[beadnum*CosBandLength+tid%(cross-1)];	
	/*if ((tid%(cross-1)) < cross/4)
		{
			radinprofile[tid].x = radinprofile[tid].x * (float)(cos((tid-cross/8)*((2*PI)/(cross/4 + 4)))+1)/2;
			radinprofile[tid].y = radinprofile[tid].y * (float)(cos((tid-cross/8)*((2*PI)/(cross/4 + 4)))+1)/2;
		} else
		{
			radinprofile[tid].x = 0;
			radinprofile[tid].y = 0;
		}
		*/
	tid += blockDim.x * gridDim.x;
}
}

//This program preps the radial profile so that calculations can be made on it
__host__ void PrepRadInProf (int maxthreadsperblock, int cross, int forgetradius, int numofimages, int numofbeads, cufftComplex *radinprofile, float *cosband, int CosBandLength)
{
	cufftHandle plan;
	cufftPlan1d (&plan, (cross-1), CUFFT_C2C, numofimages*numofbeads);


	//forward transform into hilbert space
	cufftExecC2C(plan, radinprofile, radinprofile, CUFFT_FORWARD);

	//Apply the cosine bandpass to the entire array
	CosBandPass<<<(1+((cross-1)*numofimages*numofbeads)/maxthreadsperblock), maxthreadsperblock>>>(cross, numofimages, numofbeads, radinprofile, cosband, CosBandLength);
	
	//reverse transform back
	cufftExecC2C(plan, radinprofile, radinprofile, CUFFT_INVERSE);

	cufftDestroy ( plan ) ;

}
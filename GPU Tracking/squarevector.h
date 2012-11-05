__global__ void SquareVector( int cross, int numofimages, int numofbeads, cufftComplex *a)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < cross * numofimages * numofbeads)
	{
		
		float tempREAL = a[tid].x;
		a[tid].x = a[tid].x * a[tid].x - (a[tid].y *a[tid].y);
		a[tid].y = 2*tempREAL*a[tid].y;
		tid += blockDim.x * gridDim.x;
		
	}
}
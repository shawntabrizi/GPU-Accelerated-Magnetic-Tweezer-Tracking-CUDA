void Configure(){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int maxthreadsperblock = deviceProp.maxThreadsPerBlock;
	//We want threadx/thready to be as close to numimages/numbeads without threadx * thready > maxthreads
	/*float threadxfrac = (float)NUMOFIMAGES/NUMOFBEADS;
	float threadyfrac = (float)NUMOFBEADS/NUMOFIMAGES;
	if (threadxfrac > maxthreadsperblock)
	{
		threadxfrac = maxthreadsperblock;
		threadyfrac = (float)1/maxthreadsperblock;
	} else if (threadyfrac > maxthreadsperblock)
	{
		threadyfrac = 1024;
		threadxfrac = (float)1/maxthreadsperblock;
	}
	int threadx = 1+(int)(sqrt(threadxfrac * maxthreadsperblock)-1);
	int thready = 1+(int)(sqrt(threadyfrac * maxthreadsperblock)-1);

	for(int i=10; i > 0; i--)
	{
		while ((threadx*(thready+i) < 1025) || ((threadx+i)*thready < 1025))
		{
			//this is implicit high bead trackin bias, flip the if statements for multiplexing tracking bias
			if(threadx*(thready+i) < 1025)
			{
				thready += i;
			}
			if((threadx+i)*thready < 1025)
			{
				threadx += i;
			}
			
		}
	}
	*/
	(*test)->val[1] = 512;
	(*test)->val[2] = 2;
	(*test)->val[0] = maxthreadsperblock;

	//Max threads per block:  1024
	dim3 threadsPerBlock( 16, 2  );
	//Make enough blocks such that every single bead has its own thread
	dim3 numBlocks( (1 + (NUMOFIMAGES-1)/threadsPerBlock.x), (1 + (NUMOFBEADS-1)/threadsPerBlock.y));
}

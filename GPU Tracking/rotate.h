//This function rotates a subsection of an array.
//I use functions defined and written for the CPU, and modified them to be used for the GPU.
//http://www.cplusplus.com/reference/algorithm/rotate/

template <class T>
__device__ void swaping ( T& a, T& b )
{
  T c(a); a=b; b=c;
}

template <class ForwardIterator>
__device__ void rotation ( ForwardIterator first, ForwardIterator middle,
                ForwardIterator last )
{
  ForwardIterator next = middle;
  while (first!=next)
  {
    swaping (*first++,*next++);
    if (next==last) next=middle;
    else if (first == middle) middle=next;
  }
}

__global__ void Rotate (int numofimages, int numofbeads, int cross, cufftComplex* AvgProfile)
{
	int ImageID = threadIdx.x + blockIdx.x * blockDim.x;
	int BeadID = threadIdx.y + blockIdx.y * blockDim.y;
	if ((ImageID < numofimages) && (BeadID < numofbeads))
	{
		rotation(AvgProfile+((ImageID*numofbeads*cross)+(BeadID*cross)), AvgProfile+((ImageID*numofbeads*cross)+(BeadID*cross))+cross/2, AvgProfile+((ImageID*numofbeads*cross)+(BeadID*cross))+cross);
	}

}
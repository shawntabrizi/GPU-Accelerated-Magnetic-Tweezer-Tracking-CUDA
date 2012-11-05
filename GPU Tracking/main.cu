#define PI 3.14159265
#define DllExport __declspec(dllexport)
//We must use this so that data is packaged correctly for LabView
#pragma pack(1)
//Only Necessary for the standalone EXE File
#include <iostream>
#include <fstream>
using namespace std;
//Necessary Includes for Math
#include <cuda.h>
#include <cufft.h>
#include <complex>
#include <math.h>
#include <algorithm>
//XY Tracking Header Files
#include "datatypes.h"
#include "avgycrossandprep.h"
#include "avgxcrossandprep.h"
#include "squarevector.h"
#include "finddelta.h"
#include "rotate.h"
//Z Tracking Header Files
#include "calculateradialprofile1.h"
#include "calculateradialprofile2.h"
#include "prepradprofile.h"
#include "fitprepandphase.h"
//Other
#include "checkbead.h"
#include "quadfit.h"

DllExport int GPUTracking (Array1dIntHandle x, Array1dIntHandle y, int cross, int crossthickness, ImageHandle image, Array2dHandle xout, Array2dHandle yout, Array2dHandle zout, ArrayClusterHandle calibration, BoolArrayHandle BeadStatus, char* text, Array1dHandle test)
{

	cudaError_t error;
	size_t free;
	size_t total;

	int NUMOFBEADS = (*x)->length;
	int NUMOFIMAGES = (*image)->length[0];
	int NUMROWS = (*image)->length[1];
	int NUMCOLS = (*image)->length[2];

//TODO: Create a function that automatically optimizes the parmeters below based off the CUDA Deivice Properties
	//Must take into account the Number of Beads, Threads, Max Number of Threads, Max Registers per Block, etc...
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int maxthreadsperblock = deviceProp.maxThreadsPerBlock;
	(*test)->val[0] = maxthreadsperblock;

	//Depending on your GPU, you might have a limited number of threads per block you can allocate.
	//Take a look at the function above that outputs the Max Threads Per Block.
	//You will also have to adjust this function for the experiment you are performing.
	//The first dimension is directly related to the number of images, and the second dimension is related to number of beads.
	//If you are performing an experiment with only 2 beads, you should not have the second dimension larger than 2 or you will waste threads.
	//Similarly with number of images. Remember you must have: DIM1 * DIM2 < Max Number of Threads
	dim3 threadsPerBlock( 32, 2  );
	//Make enough blocks such that every single bead has its own thread
	dim3 numBlocks( (1 + (NUMOFIMAGES-1)/threadsPerBlock.x), (1 + (NUMOFBEADS-1)/threadsPerBlock.y));

	//Calculate Radial Profile is independently optimized to perform better, since it is the slowest of all the functions.
	//Notice this now uses 3 dimensions, and the 3rd dimension should ALWAYS remain "cross"
	//Similarly with the above optimization, DIM1 and DIM2 are directly related to Num of Images and Threads.
	//However you must now have DIM1 * DIM2 * Cross < MaxNumberOfThreads
	dim3 threadsRadial( 2, 2, cross);
	dim3 blocksRadial( (1 + (NUMOFIMAGES-1)/threadsRadial.x), (1 + (NUMOFBEADS-1)/threadsRadial.y));

//Start XY Tracking
	unsigned char *devimage;
	float *devxout, *devyout;
	bool *devbeadstatus;
	cufftComplex *devAvgXProfile, *devAvgYProfile;
	int *devxin, *devyin;
	
	cudaMalloc( (void**)&devimage, sizeof(unsigned char) * NUMOFIMAGES*NUMCOLS*NUMROWS);
	cudaMalloc( (void**)&devbeadstatus, sizeof(bool) * NUMOFBEADS);
	cudaMalloc( (void**)&devxin, sizeof(int) * NUMOFBEADS);
	cudaMalloc( (void**)&devyin, sizeof(int) * NUMOFBEADS);

	cudaMalloc( (void**)&devxout, sizeof(float)* NUMOFIMAGES* NUMOFBEADS );
	cudaMalloc( (void**)&devyout, sizeof(float)* NUMOFIMAGES* NUMOFBEADS );
	cudaMalloc( (void**)&devAvgXProfile, sizeof(cufftComplex) * cross * NUMOFIMAGES * NUMOFBEADS);
	cudaMalloc( (void**)&devAvgYProfile, sizeof(cufftComplex) * cross * NUMOFIMAGES * NUMOFBEADS);
	
	cudaMemcpy( devimage, (*image)->val, sizeof(unsigned char) * NUMOFIMAGES*NUMCOLS*NUMROWS, cudaMemcpyHostToDevice);
	cudaMemcpy( devbeadstatus, (*BeadStatus)->val, sizeof(bool) * NUMOFBEADS, cudaMemcpyHostToDevice);
	cudaMemcpy( devxin, (*x)->val, NUMOFBEADS * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( devyin, (*y)->val, NUMOFBEADS * sizeof(int), cudaMemcpyHostToDevice );
	
	//We can probably be clever and make 1 function that handles both the X and Y Profile
	//These two functions can run in parellel with one another, since their outputs are completely indpendent, however we do not have it set up like this currently.
	//There will be a minimal speed increase by parellelizing these two functions.
	AvgXCrossAndPrep<<<numBlocks,threadsPerBlock>>>(devxin, devyin, cross, crossthickness, NUMOFIMAGES, NUMOFBEADS, NUMROWS, NUMCOLS, devimage, devAvgXProfile, devbeadstatus);
	AvgYCrossAndPrep<<<numBlocks,threadsPerBlock>>>(devxin, devyin, cross, crossthickness, NUMOFIMAGES, NUMOFBEADS, NUMROWS, NUMCOLS, devimage, devAvgYProfile, devbeadstatus);
	
	//start fft
	cufftHandle plan;
	cufftPlan1d (&plan, cross, CUFFT_C2C, NUMOFIMAGES*NUMOFBEADS);
	
	cufftExecC2C(plan, devAvgXProfile, devAvgXProfile, CUFFT_FORWARD);
	cufftExecC2C(plan, devAvgYProfile, devAvgYProfile, CUFFT_FORWARD);
	
	//Note that this function uses as many threads as possible to perform this task, not the grid defined above.
	SquareVector<<<(1+(cross*NUMOFIMAGES*NUMOFBEADS)/maxthreadsperblock), maxthreadsperblock>>>(cross, NUMOFIMAGES, NUMOFBEADS, devAvgXProfile);
	SquareVector<<<(1+(cross*NUMOFIMAGES*NUMOFBEADS)/maxthreadsperblock), maxthreadsperblock>>>(cross, NUMOFIMAGES, NUMOFBEADS, devAvgYProfile);
	
	cufftExecC2C(plan, devAvgXProfile, devAvgXProfile, CUFFT_INVERSE);
	cufftExecC2C(plan, devAvgYProfile, devAvgYProfile, CUFFT_INVERSE);

	cufftDestroy ( plan ) ;
	//end fft
	
	Rotate<<<numBlocks,threadsPerBlock>>>(NUMOFIMAGES, NUMOFBEADS, cross, devAvgXProfile);
	Rotate<<<numBlocks,threadsPerBlock>>>(NUMOFIMAGES, NUMOFBEADS, cross, devAvgYProfile);
	
	FindDelta<<<numBlocks,threadsPerBlock>>>(devyin, cross, NUMOFIMAGES, NUMOFBEADS, devAvgYProfile, devyout, devbeadstatus);
	FindDelta<<<numBlocks,threadsPerBlock>>>(devxin, cross, NUMOFIMAGES, NUMOFBEADS, devAvgXProfile, devxout, devbeadstatus);

	cudaMemcpy( (*xout)->val, devxout, NUMOFIMAGES*NUMOFBEADS*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( (*yout)->val, devyout, NUMOFIMAGES*NUMOFBEADS*sizeof(float), cudaMemcpyDeviceToHost);
//End XY Tracking

//Start Z Tracking
	int AmpRow = (*(*calibration)->cluster[0].amplitude)->length[0];
	int AmpCol = (*(*calibration)->cluster[0].amplitude)->length[1];
	int CorkRow = (*(*calibration)->cluster[0].corkscrews)->length[0];
	int CorkCol = (*(*calibration)->cluster[0].corkscrews)->length[1];
	int RealRow = (*(*calibration)->cluster[0].Real)->length[0];
	int RealCol = (*(*calibration)->cluster[0].Real)->length[1];
	int CosBandLength = (*(*calibration)->cluster[0].cosband)->length;
	int forgetradius = (*calibration)->cluster[0].forgetradius;

	cufftComplex *devradprofileout;
	float *devzout, *devcosband;
	cudaMalloc( (void**)&devcosband, sizeof(float) * NUMOFBEADS * CosBandLength);
	cudaMalloc( (void**)&devradprofileout, sizeof(cufftComplex) * NUMOFBEADS * (cross-1) * NUMOFIMAGES);
	cudaMalloc( (void**)&devzout, sizeof(float) * NUMOFBEADS * NUMOFIMAGES);

	cufftComplex *devcorkscrews;
	float *devReal, *devamplitudes;

	cudaMalloc( (void**)&devcorkscrews, sizeof(cufftComplex) * NUMOFBEADS * CorkRow * CorkCol);
	cudaMalloc( (void**)&devReal, sizeof(float) * NUMOFBEADS * RealRow * RealCol);
	cudaMalloc( (void**)&devamplitudes, sizeof(float) * NUMOFBEADS * AmpRow * AmpCol);
	for (int i=0; i< NUMOFBEADS; i++)
	{
		cudaMemcpy( &devcorkscrews[i*CorkRow * CorkCol], (*(*calibration)->cluster[i].corkscrews)->val, sizeof(cufftComplex)*CorkRow*CorkCol, cudaMemcpyHostToDevice );
		cudaMemcpy( &devReal[i*RealRow * RealCol], (*(*calibration)->cluster[i].Real)->val, sizeof(float) * RealRow * RealCol, cudaMemcpyHostToDevice );
		cudaMemcpy( &devamplitudes[i*AmpRow*AmpCol], (*(*calibration)->cluster[i].amplitude)->val, sizeof(float) * AmpRow * AmpCol, cudaMemcpyHostToDevice );
		cudaMemcpy( &devcosband[i*CosBandLength], (*(*calibration)->cluster[i].cosband)->val, sizeof(float) * CosBandLength, cudaMemcpyHostToDevice);
	}
//Calculate Radial Profile is one of the most important, and most time consuming functions
//We have attempted to optimize it significantly, and have succeeded in making it run significantly faster, yet futher work  can still be done.
	float *devradinprofile;
	cudaMalloc( (void**)&devradinprofile, sizeof(float)* NUMOFIMAGES* NUMOFBEADS* cross);

	//Note that Calculate Radial Profile 1 uses a unique grid to make blocks and threads
	CalculateRadialProfile1<<<blocksRadial,threadsRadial>>>(cross, NUMOFIMAGES, NUMOFBEADS, NUMCOLS, NUMROWS, devxout, devyout, devradprofileout, devimage, devbeadstatus, devradinprofile);
	CalculateRadialProfile2<<<numBlocks,threadsPerBlock>>>(cross, NUMOFIMAGES, NUMOFBEADS, NUMCOLS, NUMROWS, devxout, devyout, devradprofileout, devimage, devbeadstatus, devradinprofile);
	
	cudaFree ( devradinprofile );

	PrepRadInProf(maxthreadsperblock, cross, forgetradius, NUMOFIMAGES, NUMOFBEADS, devradprofileout, devcosband, CosBandLength);	

	FitPrepandPhase<<<numBlocks,threadsPerBlock>>>(cross, forgetradius, NUMOFIMAGES, NUMOFBEADS, RealRow, RealCol, AmpRow, AmpCol, CorkRow, CorkCol, devradprofileout, devReal, devamplitudes, devcorkscrews, devzout, devbeadstatus);
	
	cudaMemcpy( (*zout)->val, devzout, NUMOFIMAGES*NUMOFBEADS*sizeof(float), cudaMemcpyDeviceToHost);	
//End Z Tracking

	CheckBead<<<numBlocks,threadsPerBlock>>>(NUMOFIMAGES, NUMOFBEADS, NUMROWS, NUMCOLS, cross, RealRow, devbeadstatus, devxout, devyout, devzout);
	cudaMemcpy( (*BeadStatus)->val, devbeadstatus, NUMOFBEADS*sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree ( devAvgYProfile ) ;
	cudaFree ( devAvgXProfile ) ;
	cudaFree ( devcosband ) ;
	cudaFree ( devamplitudes );
	cudaFree ( devReal );
	cudaFree ( devcorkscrews );
	cudaFree ( devzout ) ;
	cudaFree ( devradprofileout );
	cudaFree ( devimage ) ;
	cudaFree ( devyout ) ;
	cudaFree ( devxout ) ;
	cudaFree ( devyin ) ;
	cudaFree ( devxin ) ;
	cudaFree ( devbeadstatus ) ;

	error = cudaGetLastError();
	if(error!=cudaSuccess) {
		memcpy(text, cudaGetErrorString(error), strlen(cudaGetErrorString(error)));
	} else
	{
		memcpy(text, "No_Errors", 9);
	}
	return 0;
}

//This Whole Function allows us to run the tracking program independent of LabView.
//Use our Labview Program to Output the tracking data into a file, and the modify the filename below.
int main()
{
	Array1dIntHandle x = new Array1dInt*;
	Array1dIntHandle y = new Array1dInt*;
	int cross;
	int crossthickness;
	ImageHandle image = new Image*;
	Array2dHandle xout = new Array2d*;
	Array2dHandle yout = new Array2d*;
	Array2dHandle zout = new Array2d*;
	ArrayClusterHandle calibration = new ArrayCluster*;
	BoolArrayHandle BeadStatus = new BoolArray*;
	char* text = new char[1000];
	Array1dHandle test = new Array1d*;

	ifstream file ("TESTDATAINNOTEXT.dat", ios::in|ios::binary);

	if(file.is_open())
	{	
		int tempsize[3] = {0};
		file.read((char *)&tempsize[0], sizeof(int));
		(*x) = (Array1dInt*)malloc(sizeof(int) + sizeof(int)*tempsize[0]);
		(*x)->length = tempsize[0];
		file.read((char *)&(*x)->val[0], sizeof(int)*tempsize[0]);

		file.read((char *)&tempsize[0], sizeof(int));
		(*y) = (Array1dInt*)malloc(sizeof(int) + sizeof(int)*tempsize[0]);
		(*y)->length = tempsize[0];
		file.read((char *)&(*y)->val[0], sizeof(int)*tempsize[0]);

		file.read((char *)&cross, sizeof(int));
		file.read((char *)&crossthickness, sizeof(int));

		file.read((char *)&tempsize[0], sizeof(int)*3);
		(*image) = (Image*)malloc(sizeof(int)* 3 + sizeof(unsigned char)*tempsize[0]*tempsize[1]*tempsize[2]);
		for(int i=0; i < 3; i++)
			(*image)->length[i] = tempsize[i];
		file.read((char *)&(*image)->val[0], sizeof(unsigned char)*tempsize[0]*tempsize[1]*tempsize[2]);

		file.read((char *)&tempsize[0], sizeof(int)*2);
		(*xout) = (Array2d*)malloc(sizeof(int)* 2 + sizeof(float)*tempsize[0]*tempsize[1]);
		for(int i=0; i < 2; i++)
			(*xout)->length[i] = tempsize[i];
		file.read((char *)&(*xout)->val[0], sizeof(float)*tempsize[0]*tempsize[1]);

		file.read((char *)&tempsize[0], sizeof(int)*2);
		(*yout) = (Array2d*)malloc(sizeof(int)* 2 + sizeof(float)*tempsize[0]*tempsize[1]);
		for(int i=0; i < 2; i++)
			(*yout)->length[i] = tempsize[i];
		file.read((char *)&(*yout)->val[0], sizeof(float)*tempsize[0]*tempsize[1]);

		file.read((char *)&tempsize[0], sizeof(int)*2);
		(*zout) = (Array2d*)malloc(sizeof(int)* 2 + sizeof(float)*tempsize[0]*tempsize[1]);
		for(int i=0; i < 2; i++)
			(*zout)->length[i] = tempsize[i];
		file.read((char *)&(*zout)->val[0], sizeof(float)*tempsize[0]*tempsize[1]);
		
		//CLUSTER START
		file.read((char *)&tempsize[0], sizeof(int));
		(*calibration) = (ArrayCluster*)malloc(sizeof(int) + sizeof(Cluster) * tempsize[0]);
		(*calibration)->length = tempsize[0];
		for (int i=0; i< tempsize[0]; i++)
		{	
			int temp[2] = {0};
			//forgetradius
			file.read((char *)&(*calibration)->cluster[i].forgetradius, sizeof(int));
			//zstep
			file.read((char *)&(*calibration)->cluster[i].zstep, sizeof(float));
			//cosband
			(*calibration)->cluster[i].cosband = new Array1d*;
			file.read((char *)&temp[0], sizeof(int));
			(*(*calibration)->cluster[i].cosband) = (Array1d*)malloc(sizeof(int) + sizeof(float) * temp[0]);
			(*(*calibration)->cluster[i].cosband)->length = temp[0];
			file.read((char *)&(*(*calibration)->cluster[i].cosband)->val[0], sizeof(float) * temp[0]);
			//Amplitude
			(*calibration)->cluster[i].amplitude = new Array2d*;
			file.read((char *)&temp[0], sizeof(int)*2);
			(*(*calibration)->cluster[i].amplitude) = (Array2d*)malloc(sizeof(int)*2 + sizeof(float)*temp[0]*temp[1]);
			for(int j=0; j < 2; j++)
				(*(*calibration)->cluster[i].amplitude)->length[j] = temp[j];
			file.read((char *)&(*(*calibration)->cluster[i].amplitude)->val[0], sizeof(float) * temp[0]*temp[1]);
			//corkscrews
			(*calibration)->cluster[i].corkscrews = new CArray2d*;
			file.read((char *)&temp[0], sizeof(int)*2);
			(*(*calibration)->cluster[i].corkscrews) = (CArray2d*)malloc(sizeof(int)*2 + sizeof(complex<float>)*temp[0]*temp[1]);
			for(int j=0; j < 2; j++)
				(*(*calibration)->cluster[i].corkscrews)->length[j] = temp[j];
			file.read((char *)&(*(*calibration)->cluster[i].corkscrews)->val[0], sizeof(complex<float>) * temp[0]*temp[1]);
			//Real
			(*calibration)->cluster[i].Real = new Array2d*;
			file.read((char *)&temp[0], sizeof(int)*2);
			(*(*calibration)->cluster[i].Real) = (Array2d*)malloc(sizeof(int)*2 + sizeof(float)*temp[0]*temp[1]);
			for(int j=0; j < 2; j++)
				(*(*calibration)->cluster[i].Real)->length[j] = temp[j];
			file.read((char *)&(*(*calibration)->cluster[i].Real)->val[0], sizeof(float) * temp[0]*temp[1]);
		}
		//CLUSTER END

		file.read((char *)&tempsize[0], sizeof(int));
		(*BeadStatus) = (BoolArray*)malloc(sizeof(int) + sizeof(bool)*tempsize[0]);
		(*BeadStatus)->length = tempsize[0];
		file.read((char *)&(*BeadStatus)->val[0], sizeof(bool)*tempsize[0]);

		//file.read((char *)&text[0], sizeof(char)*1000);

		file.read((char *)&tempsize[0], sizeof(int));
		cout << tempsize[0] << endl;
		(*test) = (Array1d*)malloc(sizeof(int) + sizeof(float)*tempsize[0]);
		(*test)->length = tempsize[0];
		file.read((char *)&(*test)->val[0], sizeof(float)*tempsize[0]);
		
		cout << file.tellg() << endl;
		file.seekg(0, ios_base::end);
		cout << file.tellg() << endl << endl;

		GPUTracking (x, y, cross, crossthickness, image, xout, yout, zout, calibration, BeadStatus,text,test);
		
		for (int i=0; i < (*xout)->length[0] * (*xout)->length[1]; i++)
		{
			cout << i/2;
			if (i%2 == 0)
				cout << "a";
			else
				cout << "b";
			cout << ": "<< (*xout)->val[i] << " | " << (*yout)->val[i] << " | " << (*zout)->val[i] << endl;
		}
		cout << text << endl;
		

		//free them all!
		free((*x));
		free((*y));
		free((*image));
		free((*xout));
		free((*yout));
		free((*zout));
		free((*calibration));
		free((*BeadStatus));
		free((*test));
		file.close();
	} else
	{
		cout << "File Did Not Open!" << endl;
	}

	delete x;
	delete y;
	delete image;
	delete xout;
	delete yout;
	delete zout;
	delete calibration;
	delete BeadStatus;
	delete text;
	delete test;

	return 0;
}
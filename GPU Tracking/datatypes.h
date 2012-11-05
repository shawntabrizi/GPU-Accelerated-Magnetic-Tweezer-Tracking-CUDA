//Defines 1d Array Handle for INT
typedef struct {
	int length;
	int val[1];
		//to access value at a row => val[row]
} Array1dInt, **Array1dIntHandle;

//Defines 1d Array Handle
typedef struct {
	int length;
	float val[1];
		//to access value at a row => val[row]
} Array1d, **Array1dHandle;

//Defines 2d Array Handle
typedef struct {
	int length[2];
		//length[0] = number of rows
		//length[1] = number of columns
	float val[1];
		//to access value at a row/col => val[(row * length[1]) + col)]
} Array2d, **Array2dHandle;

//Defines 3d Array Handle
typedef struct {
	int length[3];
		//length[0] = number of pages
		//length[1] = number of rows
		//length[2] = number of columns
	unsigned char val[1];
		//to access value at a row/col => val[(row * length[1]) + col)]
} Image, **ImageHandle;

//Defines 1d Complex Array Handle
typedef struct {
	int length;
	complex<float> val[1];
		//to access value at a row => val[row]
} CArray1d, **CArray1dHandle;

//Defines Upto 2d Complex Array Handle
typedef struct {
	int length[2]; //increase the index to handle bigger array sizes
		//length[0] = number of rows
		//length[1] = number of columns
	complex<float> val[1];
		//to access value at a row/col => val[(row * length[1]) + col)]
} CArray2d, **CArray2dHandle;

//Defines 1d Boolean Array for Bead is Good
typedef struct {
	int length;
	bool val[1];
		//to access value at a row => val[row]
} BoolArray, **BoolArrayHandle;

//defines a calibration cluster
typedef struct {
	int forgetradius;
	float zstep;
	Array1dHandle cosband;
	Array2dHandle amplitude;
	CArray2dHandle corkscrews;
	Array2dHandle Real;
} Cluster;

typedef struct {
	int length;
	Cluster cluster[1];
} ArrayCluster, **ArrayClusterHandle;
// standard
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "/usr/local/MATLAB/R2013a/extern/include/tmwtypes.h"
#include <assert.h>

// GDAL
#include "gdal.h"
#include "cpl_conv.h" 		/* for CPLMalloc() */
#include "cpl_string.h"
#include "gdal_priv.h"

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// TYPE of DATA???????
#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define tid  (tx + blockDim.x*( ty + blockDim.y  *( bx + gridDim.x*by)))

//#define uintXX_t uint32_T

#define europe 1
//	I/-
#if europe
	//	**EUROPE**
	const char *iFil1	= "/home/giuliano/git/cuda/change-map-mat/data/g100_00.tif";
	const char *iFil2	= "/home/giuliano/git/cuda/change-map-mat/data/g100_06.tif";
	const char *roiFil	= "/home/giuliano/git/cuda/change-map-mat/data/clc_roi-eur_100m.tif";
	//	-/O
	const char *oFil	= "/home/giuliano/git/cuda/change-map-mat/data/ch06-00_L3-eur_100m.tif";
	const char *chFil	= "/home/giuliano/git/cuda/change-map-mat/data/chMat_06-00_L3-eur.txt";
#define			nclasses  45
#else
	//	**ITALY**
	const char *iFil1	= "/home/giuliano/git/cuda/change-map-mat/data/clc2000_L1_100m.tif";
	const char *iFil2	= "/home/giuliano/git/cuda/change-map-mat/data/clc2006_L1_100m.tif";
	const char *roiFil	= "/home/giuliano/git/cuda/change-map-mat/data/clc_roi-ita_100m.tif";
	//	-/O
	const char *oFil	= "/home/giuliano/git/cuda/change-map-mat/data/ch06-00_L1-ita_100m.tif";
	const char *chFil	= "/home/giuliano/git/cuda/change-map-mat/data/chMat_06-00_L1-ita.txt";
#define			nclasses  6
#endif

/*
 INPUTS:
    iMap1:      map of OLDER year;
    iMap2:      map of NEWER year;
    nX,nY:		WIDTH & HEIGHT of Map's without padding;
	pitch:		number of pixels of Map's with padding
	nclasses:	number of classes to be counted;
 OUTPUTS:
    chMat:      change matrix 	---> storing the accountancy of classes change-of-state;
    oMap:       change map 		---> storing the differences between iMap1 and iMap2 (code:2211);
*/
__global__ void changemap( const uint8_T *iMap1,const uint8_T *iMap2,const uint8_T *roiMap,
                           uint32_T nX, uint32_T nY, uint32_T pitch, uint32_T nClasses,
                           uint32_T *chMat, uint32_T *oMap )
{
	uint32_T iCl1,iCl2,iroi,i_row,x,k;
	if (tid<nY){
        /* i: offset to move the pointer tile-by-tile on the input/output maps
              (iMap1,iMap2,oMap): this is helpful to assign each tile to a thread.
        */
		i_row = tid * pitch;// ***pitch!!
        /* k: offset to move the pointer tile-by-tile on the output change
              matrix (chMat):
        */
		k = tid*nClasses*nClasses;
        /*  Now that "i" is defined (that is the pointer to the first element
            of any row==tile), I have to offset along the y direction (see the
            for loop on x below).
        */
		// The pointer along the x direction is straightforward:
		for( x=0; x<nX; x++ ){
			iCl1 			= iMap1 [i_row+x];
			iCl2 			= iMap2 [i_row+x];
			iroi 			= roiMap[i_row+x]; // 0:out-of-roi; 1:in-roi
			oMap[i_row+x] 	= ( iCl1 + iCl2*100 )*iroi;
			chMat[ (k + (iCl1+iCl2*nClasses))*iroi ]++;
		}
	}
}

__global__ void changemat(uint32_T *chMat, uint32_T classes_2, uint32_T ntiles)
{
/*
 DESCRIPTION:
 	This is a "reduce" kernel
 INPUTS:
    classes_2:  dimX * dimY of cross matrix;
    ntiles:     to be splitted between blocks of threads and grid;
 OUTPUTS:
    chMat:      change matrix sum considering all tiles --> only the first
                tile/layer is the output (i.e. chMat(:,:,1) );
*/
	uint32_T k,j,i,ii,nt,isodd;
    nt      = ntiles;
    /*  ntiles/2 (SHIFT A DESTRA) --> lavora sui binari(bit!!), look at the
        example below:
            decimal nt=7        ---> binary 111
            operation nt>>1     ---> binary 11      ---> decimal i=3
    */
    i       = nt>>1;
	/*  now:
            decimal i=3         ---> binary 11
            operation i<<1      ---> binary 110     ---> decimal ii=6
    */
    ii      = i<<1;
    /*  if nt is odd then ii is not equal to nt (e.g. nt=7,ii=6), otherwise
        it does (e.g. nt=6, ii=6).
    */
    isodd   = (ii!=nt);
    while(i>0) {
        // k:       is the pointer to the first element of any tile.
		k=tid*classes_2;
		if (tid<i) {
            // j:   is the pointer that moves within each k tile.
			for(j=0;j<classes_2;j++) {
                /*  It sums - in k tile - k tile with the k+ntiles/2
                    ( remember that ntiles/2 is expressed by i (after i>>1) )
                    At the end of while loop the change signatures from all
                    pixels of the iMap1 & iMap2 are summed in the first tile.
                */
				chMat[k+j]=chMat[k+j]+chMat[k+j+i*classes_2];
                /*
                    If the number of tiles at current while loop is odd,
                    then the first thread has to add to its k tile (i.e. the
                    first tile) the tile excluded from summation by the
                    previous line of code.
                */
				if ((tid==0)&&(isodd)) {
                    //printf("dispari! i=%d ii=%d nt=%d\n",i,ii,nt);
					chMat[k+j]=chMat[k+j]+chMat[k+j+2*i*classes_2];
               	}
			}
		}

        /*  It steps forward to the next while loop only when all threads
            have finished:
        */
        __syncthreads();
        /* Wait all writes in global memory*/
        //__threadfence();

        /*  Here, the next 4 lines repeat within while loop what was done
            the first time outside. Go there for description.
        */
        nt  = i;
		i   = i>>1;
        ii  = i<<1;
        isodd=(ii!=nt);
	}
}

void save_chmat_on_HDD(	uint32_T nC,uint32_T *host_chMat ){

	FILE *fid ;
	fid = fopen(chFil,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",chFil); exit(EXIT_FAILURE); }
	for(int i=0;i<nC;i++){
		for(int j=0;j<nC;j++){
			fprintf(fid, "%9d ", host_chMat[j+i*nC]);
		}fprintf(fid,"\n");
	} fclose(fid);
}

void save_omap_on_HDD(	const char *RefSyst, double *iGeoTransform, GDALDriverH hDriver,
						unsigned int nX, unsigned int nY, uint32_T *host_oMap){

	/*
	 * 	DECLARATIONS
	 */
	// options for GDALCreate:
	char 				**papszOptions = NULL;
	GDALDatasetH		oMap;
	GDALRasterBandH		oBand;

	// CREATE DATASET on HDD
	//	-options: tiling with block 512x512
	papszOptions = CSLSetNameValue( papszOptions, "TILED", "YES");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKXSIZE", "512");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKYSIZE", "512");
	//	-instantiate GRID
	oMap = GDALCreate( hDriver, oFil, nX, nY, 1, GDT_UInt32, papszOptions );
	//	-set projection
	GDALSetProjection( 	 oMap, RefSyst );
	//	-set geospatial transformation
	GDALSetGeoTransform( oMap, iGeoTransform );
	//	-band
	oBand = GDALGetRasterBand( oMap, 1 );
	//	-write to HDD
	GDALRasterIO(oBand, GF_Write, 0, 0, nX, nY, host_oMap, nX, nY, GDT_UInt32, 0, 0);

	GDALClose( oMap );
	CSLDestroy( papszOptions );
}

void write_generic_array_on_HDD(uint32_T *mat, uint32_T NC, uint32_T NR, const char *FileName, const char *printFormat ){

	// printFormat = "%9d "
	FILE *fid ;
	fid = fopen(FileName,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",FileName); exit(EXIT_FAILURE); }

	for(int r=0; r<NR; r++){
		for (int c=0; c<NC; c++){
			fprintf(fid, "%d ", mat[c+r*NC]);
		}fprintf(fid,"\n");
	} fclose(fid);
}

void write_uint8T_array_on_HDD(uint8_T *mat, uint32_T NC, uint32_T NR, const char *FileName, const char *printFormat ){

	// printFormat = "%9d "
	FILE *fid ;
	fid = fopen(FileName,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",FileName); exit(EXIT_FAILURE); }

	for(int r=0; r<NR; r++){
		for (int c=0; c<NC; c++){
			fprintf(fid, "%d ", mat[c+r*NC]);
		}fprintf(fid,"\n");
	} fclose(fid);
}

void run_kernels_single_gpu( int gpuDev, unsigned int nX, unsigned int nY, uint32_T nC,
							 GDALDatasetH iMap1, GDALDatasetH iMap2, GDALDatasetH roiMap,
							 const char *RefSyst, double *iGeoTransform, GDALDriverH hDriver ){

	/*
	 * 	DECLARATIONS
	 */
	GDALRasterBandH		iBand1,iBand2, roiBand;
	cudaEvent_t 		start, stop;
	uint8_T				*dev_iMap1, *dev_iMap2;
	uint32_T			*dev_oMap, *dev_chMat;
	uint8_T				*dev_roiMap;
	float 				elapsed_time_ms = 0.0f;
	unsigned int 		BLOCKSIZE, GRIDSIZE;
	cudaDeviceProp		devProp;

	/*
	 * 	GPU device
	 */
	// find out required memory:
	double				iMaps_bytes		= nY*nX*sizeof(uint8_T  );
	double				oMap_bytes		= nY*nX*sizeof(uint32_T );
	double				chMat_bytes		= nC*nC*nY*sizeof(uint32_T);
	// query GPU characteristics for selected GPU:
	cudaGetDeviceProperties(&devProp, gpuDev);
	printf("Texture alignment:\t\t%zu\t\t[el]\n", devProp.textureAlignment);
	printf("Concurrent copy and execute:\t%d\n", devProp.deviceOverlap);
	double reqMem		= (iMaps_bytes*3 + oMap_bytes + chMat_bytes)/pow(1024,3);
	double avGlobMem	= (double)devProp.totalGlobalMem/pow(1024,3);
	printf("Global memory on GPU[%d]:\t%.2f\t\t[Gbytes]\n", gpuDev,avGlobMem );
	printf("Required memory on GPU:\t\t%.2f\t\t[Gbytes]\n", reqMem);
	printf("Available memory:\t\t%.2f\t\t[Gbytes]\n", avGlobMem-reqMem);
	if(avGlobMem < 0) {
		printf("Error: Available Global Memory on gpu %s #%d not sufficient!\n",devProp.name, gpuDev );
		exit(EXIT_FAILURE);
	}
	cudaSetDevice(gpuDev); //	-select GPU (a multi-gpu management is possible with "async" calls of memcopies and kernels):

	/*
	 * 	Allocate on RAM
	 */
	uint8_T  			*host_iMap1		= (uint8_T  *) CPLMalloc( iMaps_bytes );
	uint8_T  			*host_iMap2		= (uint8_T  *) CPLMalloc( iMaps_bytes );
	uint8_T  			*host_roiMap	= (uint8_T  *) CPLMalloc( iMaps_bytes );
	uint32_T 			*host_oMap		= (uint32_T *) CPLMalloc( oMap_bytes  );
	uint32_T 			*host_chMat		= (uint32_T *) CPLMalloc( chMat_bytes );
/*	assert( cudaMallocHost( (void**)&host_iMap1, iMaps_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_iMap2, iMaps_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_oMap,   oMap_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_chMat, chMat_bytes ) == 0 );
*/

	/*
	 * 	GDAL I/O
	 */
	// get BANDS:
	iBand1				= GDALGetRasterBand( iMap1,  GDALGetRasterCount( iMap1  ) ); // GDALGetRasterCount( iMap1 ) is equal to 1.
	iBand2				= GDALGetRasterBand( iMap2,  GDALGetRasterCount( iMap2  ) );
	roiBand				= GDALGetRasterBand( roiMap, GDALGetRasterCount( roiMap ) );
	// read data from HDD:
	GDALRasterIO( iBand1,  GF_Read, 0, 0, nX, nY, host_iMap1,  nX, nY, GDT_Byte, 0, 0 );
	GDALRasterIO( iBand2,  GF_Read, 0, 0, nX, nY, host_iMap2,  nX, nY, GDT_Byte, 0, 0 );
	GDALRasterIO( roiBand, GF_Read, 0, 0, nX, nY, host_roiMap, nX, nY, GDT_Byte, 0, 0 );

	/*
	 * 	KERNELS LAUNCH
	 */
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
	//	-Allocate arrays on GPU device:
	assert( cudaMalloc((void **)&dev_iMap1, iMaps_bytes) == 0 );
	assert( cudaMalloc((void **)&dev_iMap2, iMaps_bytes) == 0 );
	assert( cudaMalloc((void **)&dev_roiMap,iMaps_bytes) == 0 );
	assert( cudaMalloc((void **)&dev_oMap,  oMap_bytes)  == 0 );
	assert( cudaMalloc((void **)&dev_chMat, chMat_bytes) == 0 );
	assert( cudaMemset(dev_chMat,0,(size_t) chMat_bytes) == 0 ); //IMPORTANT!!
/*	//This allocates the 2D array of size nX*nY in device memory and returns pitch
 	assert( cudaMallocPitch( (void**)&dev_iMap1, &iPitch1, iWIDTH_bytes, nY) == 0 );
	assert( cudaMallocPitch( (void**)&dev_iMap2, &iPitch2, iWIDTH_bytes, nY) == 0 );
	assert( cudaMallocPitch( (void**)&dev_oMap,  &oPitch,  oWIDTH_bytes, nY) == 0 );
 	printf("iPitch1:\t%zu\niPitch2:\t%zu\noPitch:\t\t%zu\n",iPitch1,iPitch2,oPitch);
 */
	//	-copy: RAM--->GPU
	assert( cudaMemcpy(dev_iMap1, host_iMap1, iMaps_bytes,cudaMemcpyHostToDevice) == 0 );
	assert( cudaMemcpy(dev_iMap2, host_iMap2, iMaps_bytes,cudaMemcpyHostToDevice) == 0 );
	assert( cudaMemcpy(dev_roiMap,host_roiMap,iMaps_bytes,cudaMemcpyHostToDevice) == 0 );
/*	assert( cudaMemcpyAsync(dev_iMap1+offset, host_iMap1+offset, iMaps_bytes/nStreams, cudaMemcpyHostToDevice, stream[i]) == 0 );
	assert( cudaMemcpyAsync(dev_iMap2+offset, host_iMap2+offset, iMaps_bytes/nStreams, cudaMemcpyHostToDevice, stream[i]) == 0 );
*/
/*	assert( cudaMemcpy2D(dev_iMap1,iPitch1,host_iMap1,(size_t) iWIDTH_bytes,(size_t) iWIDTH_bytes,(size_t) nY, cudaMemcpyHostToDevice) == 0 );
	assert( cudaMemcpy2D(dev_iMap2,iPitch2,host_iMap2,(size_t) iWIDTH_bytes,(size_t) iWIDTH_bytes,(size_t) nY, cudaMemcpyHostToDevice) == 0 );
*/
//	if(iPitch1 != iPitch2){ printf("Error: the pitch of the two input iMap's are different!\n"); exit(EXIT_FAILURE); }
	//
	// -kernel config:
	BLOCKSIZE         	= floor(sqrt( devProp.maxThreadsPerBlock ));
	GRIDSIZE			= 1 + (nY/(BLOCKSIZE*BLOCKSIZE));
	dim3 block(BLOCKSIZE,BLOCKSIZE,1);
	dim3 grid(GRIDSIZE,1,1);
	uint32_T pitch 		= nX;//(uint32_T)iPitch1/sizeof(uint8_T);

	changemap<<<grid,block>>>(	dev_iMap1,dev_iMap2, dev_roiMap,
								nX,nY,pitch,nC,
								dev_chMat,dev_oMap );
	cudaDeviceSynchronize();

	changemat<<<grid,block>>>( dev_chMat, nC*nC, nY );
	cudaDeviceSynchronize();

	// gather outputs:
	//	-copy: GPU--->RAM
	assert( cudaMemcpy(host_oMap,dev_oMap,oMap_bytes,cudaMemcpyDeviceToHost) == 0 );
//	assert( cudaMemcpy2D(host_oMap,(size_t) oWIDTH_bytes,dev_oMap,oPitch,(size_t) oWIDTH_bytes,(size_t) nY, cudaMemcpyDeviceToHost) == 0 );
	assert( cudaMemcpy(host_chMat,dev_chMat,chMat_bytes/nY,cudaMemcpyDeviceToHost) == 0 );
	//
	// CUDA timer:
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	printf("Elapsed time {H2D,k1,k2,D2H}:\t%1.3f\t\t[s]\n\n",elapsed_time_ms/(float)1000.00);

	save_chmat_on_HDD( nC, host_chMat );

	save_omap_on_HDD( RefSyst, iGeoTransform, hDriver, nX, nY, host_oMap);

	/*
	 * 	FREE
	 */
	// GDAL
	VSIFree( host_iMap1 );
	VSIFree( host_iMap2 );
	VSIFree( host_oMap );
	VSIFree( host_chMat );
	// CUDA free:
	cudaFree(dev_iMap1);
	cudaFree(dev_iMap2);
	cudaFree(dev_oMap);
	cudaFree(dev_chMat);
	// C free:
/*	cudaFreeHost(host_iMap1);
	cudaFreeHost(host_iMap2);
	cudaFreeHost(host_oMap);
	cudaFreeHost(host_chMat);
*/
}

void run_kernels_multi_gpu(	unsigned int nX, unsigned int nY, uint32_T nC,
							GDALDatasetH iMap1, GDALDatasetH iMap2, GDALDatasetH roiMap,
							const char *RefSyst, double *iGeoTransform, GDALDriverH hDriver ){

	/*
	 * 	DECLARATIONS
	 */
	GDALRasterBandH		iBand1,iBand2, roiBand;
	cudaEvent_t 		start, stop;
	float 				elapsed_time_ms = 0.0f;
	unsigned int 		BLOCKSIZE, GRIDSIZE;
	cudaDeviceProp		devProp;
	int					devCount = 0;
	uint8_T  			*host_iMap1, *host_iMap2, *host_roiMap;
	uint32_T 			*host_oMap, *host_chMat;
	uint32_T			stream_offset_I, stream_offset_O;
	uint32_T			gpu_offset_I, gpu_offset_O;
	uint32_T			offset_I, offset_O;
	int					i, j, nGPUs, gpu, nStreams, iS, nLaunches;

	/*
	 * 	GPU device
	 */
	// find out required memory:
	double				iMaps_bytes		= nY*nX*sizeof(uint8_T  );
	double				oMap_bytes		= nY*nX*sizeof(uint32_T );
	double				chMat_bytes		= nC*nC*nY*sizeof(uint32_T);
	double				chMat_bytes_host= nC*nC*sizeof(uint32_T);
	double				reqMem			= (iMaps_bytes*3 + oMap_bytes + chMat_bytes)/pow(1024,3);
	double				resMem 			= reqMem;
	// gpu count:
	cudaGetDeviceCount(&devCount);
	// query GPU characteristics for all GPUs:
	printf("Required memory:\t\t%.2f\t\t[Gbytes]\n", reqMem);
	for(i=0;i<devCount;i++){
		nGPUs = i+1;
		cudaGetDeviceProperties(&devProp, i);
		double avGlobMem	= (double)devProp.totalGlobalMem/pow(1024,3);
		printf("Global memory on GPU[%d]:\t%.2f\t\t[Gbytes]\n", i,avGlobMem );
		resMem = resMem-avGlobMem;
		printf("Residual memory:\t\t%.2f\t\t[Gbytes]\n", resMem);
		if(resMem<0){ break; }
	}
	nStreams = 2+ (int)reqMem / (int)(reqMem-resMem);
	nLaunches = nGPUs * nStreams;
	printf("Number of launches (%dxGPU):\t%d\n", nGPUs, nLaunches );

	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
	/*
	 * 	Bytes allocation
	 */
/*	if( (iMaps_bytes % nLaunches !=0) | (oMap_bytes % nLaunches!=0) | (chMat_bytes % nLaunches!=0) ){
		printf("Error: allocation in bytes is not integer when divided by %d!\n",nLaunches);
	}*/
	double				iMaps_by_chunk	= iMaps_bytes / nLaunches;
	double				oMap_by_chunk	= oMap_bytes  / nLaunches;
	double				chMat_by_chunk	= chMat_bytes / nLaunches;
	/*
	 * 	Pointers on GPUs
	 */
	uint8_T				*dev_iMap1, *dev_iMap2, *dev_roiMap;
	uint32_T			*dev_oMap,  *dev_chMat;
	/*
	 * 	Allocate page-locked memory on RAM
	 */
	assert( cudaMallocHost( (void**)&host_iMap1, iMaps_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_iMap2, iMaps_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_roiMap,iMaps_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_oMap,   oMap_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_chMat, chMat_bytes_host*nLaunches ) == 0 );
	/*
	 * 	GDAL I/O
	 */
	// get BANDS:
	iBand1				= GDALGetRasterBand( iMap1,  GDALGetRasterCount( iMap1  ) ); // GDALGetRasterCount( iMap1 ) is equal to 1.
	iBand2				= GDALGetRasterBand( iMap2,  GDALGetRasterCount( iMap2  ) );
	roiBand				= GDALGetRasterBand( roiMap, GDALGetRasterCount( roiMap ) );
	// read data from HDD:
	GDALRasterIO( iBand1,  GF_Read, 0, 0, nX, nY, host_iMap1,  nX, nY, GDT_Byte, 0, 0 );
	GDALRasterIO( iBand2,  GF_Read, 0, 0, nX, nY, host_iMap2,  nX, nY, GDT_Byte, 0, 0 );
	GDALRasterIO( roiBand, GF_Read, 0, 0, nX, nY, host_roiMap, nX, nY, GDT_Byte, 0, 0 );
	/*
	 * 	Create streams:
	 */
	//
	cudaStream_t stream[nLaunches];
	for (i=0; i<nLaunches; i++) { assert( cudaStreamCreate(&stream[i]) == 0); }
	/*
	 * 	Loop on synchronous streams (not asynchronous!!)
	 */
	for (j=0; j<nStreams; i++) {
		stream_offset_I		= (nX*nY)			* j/nStreams;
		stream_offset_O 	= (nC*nC*nLaunches)	* j/nStreams;

		//	-select GPU (a multi-gpu management is possible with "async" calls of memcopies and kernels):
		for(gpu=0;gpu<nGPUs;gpu++){
			cudaSetDevice(gpu);
			gpu_offset_I	= (nX*nY/nStreams)	* gpu/nGPUs;
			gpu_offset_O 	= (nC*nC*nGPUs)		* gpu/nGPUs;
			iS 				= gpu + j*nStreams; // i-stream
			offset_I		= gpu_offset_I + stream_offset_I;
			offset_O		= gpu_offset_O + stream_offset_O;
			//	-Allocate arrays on GPU device:
			assert( cudaMalloc((void **)&dev_iMap1, iMaps_by_chunk ) == 0 );
			assert( cudaMalloc((void **)&dev_iMap2, iMaps_by_chunk ) == 0 );
			assert( cudaMalloc((void **)&dev_roiMap,iMaps_by_chunk ) == 0 );
			assert( cudaMalloc((void **)&dev_oMap,  oMap_by_chunk  ) == 0 );
			assert( cudaMalloc((void **)&dev_chMat, chMat_by_chunk ) == 0 );
			assert( cudaMemset(dev_chMat,0,(size_t) chMat_by_chunk ) == 0 ); //IMPORTANT!!
		/*	//This allocates the 2D array of size nX*nY in device memory and returns pitch
			assert( cudaMallocPitch( (void**)&dev_iMap1, &iPitch1, iWIDTH_bytes, nY) == 0 );
			assert( cudaMallocPitch( (void**)&dev_iMap2, &iPitch2, iWIDTH_bytes, nY) == 0 );
			assert( cudaMallocPitch( (void**)&dev_oMap,  &oPitch,  oWIDTH_bytes, nY) == 0 );
			printf("iPitch1:\t%zu\niPitch2:\t%zu\noPitch:\t\t%zu\n",iPitch1,iPitch2,oPitch);
		 */

			//	-copy: RAM--->GPU
			assert( cudaMemcpyAsync(dev_iMap1 , host_iMap1 +offset_I, iMaps_by_chunk, cudaMemcpyHostToDevice, stream[iS]) == 0 );
			assert( cudaMemcpyAsync(dev_iMap2 , host_iMap2 +offset_I, iMaps_by_chunk, cudaMemcpyHostToDevice, stream[iS]) == 0 );
			assert( cudaMemcpyAsync(dev_roiMap, host_roiMap+offset_I, iMaps_by_chunk, cudaMemcpyHostToDevice, stream[iS]) == 0 );
		/*	assert( cudaMemcpy2D(dev_iMap1,iPitch1,host_iMap1,(size_t) iWIDTH_bytes,(size_t) iWIDTH_bytes,(size_t) nY, cudaMemcpyHostToDevice) == 0 );
			assert( cudaMemcpy2D(dev_iMap2,iPitch2,host_iMap2,(size_t) iWIDTH_bytes,(size_t) iWIDTH_bytes,(size_t) nY, cudaMemcpyHostToDevice) == 0 );
			if(iPitch1 != iPitch2){ printf("Error: the pitch of the two input iMap's are different!\n"); exit(EXIT_FAILURE); }
		*/
			/*
			 * 	KERNELS LAUNCH
			 */
			// -kernel config:
			BLOCKSIZE         	= floor(sqrt( devProp.maxThreadsPerBlock ));
			GRIDSIZE			= 1 + (nY/(BLOCKSIZE*BLOCKSIZE))/nLaunches;
			dim3 block(BLOCKSIZE,BLOCKSIZE,1);
			dim3 grid(GRIDSIZE,1,1);
			uint32_T pitch 		= nX;//(uint32_T)iPitch1/sizeof(uint8_T);
			// first kernel:
			changemap<<<grid,block,0,stream[iS]>>>(	dev_iMap1,dev_iMap2, dev_roiMap,
										nX,nY/nLaunches,pitch,nC,
										dev_chMat,dev_oMap ); //cudaDeviceSynchronize();
			assert( cudaMemcpyAsync(host_oMap+offset_I,dev_oMap,oMap_by_chunk,cudaMemcpyDeviceToHost, stream[iS]) == 0 );
			// second kernel:
			changemat<<<grid,block,0,stream[iS]>>>( dev_chMat, nC*nC, nY/nLaunches ); //cudaDeviceSynchronize();
			assert( cudaMemcpyAsync(host_chMat+offset_O,dev_chMat,chMat_bytes_host,cudaMemcpyDeviceToHost, stream[iS]) == 0 );

			cudaDeviceSynchronize();

			// CUDA free:
			cudaFree(dev_iMap1);
			cudaFree(dev_iMap2);
			cudaFree(dev_roiMap);
			cudaFree(dev_oMap);
			cudaFree(dev_chMat);
		}
	}

	// CUDA timer:
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	printf("Elapsed time {H2D,k1,k2,D2H}:\t%1.3f\t\t[s]\n\n",elapsed_time_ms/(float)1000.00);

	save_chmat_on_HDD( nC, host_chMat );

	save_omap_on_HDD( RefSyst, iGeoTransform, hDriver, nX, nY, host_oMap);

	/*
	 * 	FREE
	 */
	// GDAL
	VSIFree( host_iMap1 );
	VSIFree( host_iMap2 );
	VSIFree( host_oMap );
	VSIFree( host_chMat );
	// C free:
/*	cudaFreeHost(host_iMap1);
	cudaFreeHost(host_iMap2);
	cudaFreeHost(host_oMap);
	cudaFreeHost(host_chMat);
*/
}

void run_kernels_multi_gpu_dev(	unsigned int nX, unsigned int nY, uint32_T nC,
								GDALDatasetH iMap1, GDALDatasetH iMap2, GDALDatasetH roiMap,
								const char *RefSyst, double *iGeoTransform, GDALDriverH hDriver ){

	/*
	 * 	DECLARATIONS
	 */
	GDALRasterBandH		iBand1,iBand2, roiBand, oBand;
	GDALDatasetH		oMap;
	float 				elapsed_time_ms = 0.0f;
	unsigned int 		BLOCKSIZE, GRIDSIZE;
	cudaDeviceProp		devProp;
	int					devCount = 0;
	uint8_T  			*host_iMap1, *host_iMap2, *host_roiMap;
	uint32_T 			*host_oMap, *host_chMat;
	uint8_T				*dev_iMap1, *dev_iMap2, *dev_roiMap;
	uint32_T			*dev_oMap,  *dev_chMat;
	uint32_T			i, j, nGPUs, gpu, nStreams, iS, nLaunches;
	uint32_T 			HEIGHT_CHUNK_i;
	/*
	 * 	GPU device
	 */
	// find out required memory:
	double				iMaps_bytes		= nY*nX*sizeof(uint8_T  );
	double				oMap_bytes		= nY*nX*sizeof(uint32_T );
	double				chMat_bytes		= nY*nC*nC*sizeof(uint32_T);
	double				chMat_bytes_host= nC*nC*sizeof(uint32_T);
	double				reqMem			= (iMaps_bytes*3 + oMap_bytes + chMat_bytes)/pow(1024,3);
	double				resMem 			= reqMem;
	// gpu count:
	cudaGetDeviceCount(&devCount);
	/*
	 * 	to calculate the net time of using more GPUs I fix to one GPU:
	 *	devCount = 1;
	 */
	// query GPU characteristics for all GPUs:
	printf("Required memory:\t\t%.2f\t\t[Gbytes]\n", reqMem);
	for(i=0;i<devCount;i++){
		nGPUs = i+1;
		cudaGetDeviceProperties(&devProp, i);
		double avGlobMem	= (double)devProp.totalGlobalMem/pow(1024,3);
		printf("Global memory on GPU[%d]:\t%.2f\t\t[Gbytes]\n", i,avGlobMem );
		resMem = resMem-avGlobMem;
		printf("Residual memory:\t\t%.2f\t\t[Gbytes]\n", resMem);
		if(resMem<0){ break; }
	}
	/*
	 * 	to force Italy on 2 GPUs and 3 Streams:
	 * 	nStreams = 2+ 1; nGPUs = 2;
	 */
	nStreams = 2+ (int)reqMem / (int)(reqMem-resMem);
	nLaunches = nGPUs * nStreams;
	printf("Number of launches (%dxGPU):\t%d\n", nGPUs, nLaunches );
	uint32_T 		CHUNK_LEN 			= (uint32_T) (nY / nLaunches);

	// allocate the Final changeMatrix:
	uint32_T 		*host_chMat_sum		= (uint32_T *) CPLMalloc( chMat_bytes_host );
	for(int r=0;r<nC;r++) for(int c=0;c<nC;c++) host_chMat_sum[r+c*nC] = 0;

	/*
	 * 	GDAL I/O for iMaps:
	 */
	// get BANDS:
	iBand1				= GDALGetRasterBand( iMap1,  GDALGetRasterCount( iMap1  ) ); // GDALGetRasterCount( iMap1 ) is equal to 1.
	iBand2				= GDALGetRasterBand( iMap2,  GDALGetRasterCount( iMap2  ) );
	roiBand				= GDALGetRasterBand( roiMap, GDALGetRasterCount( roiMap ) );
	/*
	 * 	CREATE oMap on HDD:
	 */
	// options for GDALCreate:
	char 				**papszOptions = NULL;
	// CREATE DATASET on HDD
	//	-options: tiling with block 512x512
	papszOptions = CSLSetNameValue( papszOptions, "TILED", "YES");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKXSIZE", "512");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKYSIZE", "512");
	//	-instantiate GRID
	oMap = GDALCreate( hDriver, oFil, nX, nY, 1, GDT_UInt32, papszOptions );
	//	-set projection
	GDALSetProjection( 	 oMap, RefSyst );
	//	-set geospatial transformation
	GDALSetGeoTransform( oMap, iGeoTransform );
	//	-band
	oBand = GDALGetRasterBand( oMap, 1 );
	/*
	 * 	Create streams:
	 */
	//
	cudaStream_t stream[nLaunches];
	/*
	 * 	Loop on synchronous streams (not asynchronous!!)
	 */
	for (j=0; j<nStreams; j++) {
		//	-select GPU (a multi-gpu management is possible with "async" calls of memcopies and kernels):
		for(gpu=0;gpu<nGPUs;gpu++){
			assert( cudaSetDevice(gpu) == 0 );
			iS 				= gpu + j*nGPUs;
			cudaEvent_t 		start, stop, k1_s, k1_e, k2_s, k2_e;
			assert( cudaEventCreate( &start )	 	== 0 );
			assert( cudaEventCreate( &stop )  		== 0 );
			assert( cudaEventCreate( &k1_s )	 	== 0 );
			assert( cudaEventCreate( &k1_e )  		== 0 );
			assert( cudaEventCreate( &k2_s )	 	== 0 );
			assert( cudaEventCreate( &k2_e )  		== 0 );
			assert( cudaEventRecord( start, 0 )		== 0 );
			assert( cudaStreamCreate(&stream[iS])	== 0 );
			printf("Stream[%d], gpu[%d], Launch[%d]\t",j,gpu,iS);
			/*
			 * 	Bytes allocation
			 */
			if(iS<nLaunches-1){	HEIGHT_CHUNK_i 	= CHUNK_LEN;}
			else{				HEIGHT_CHUNK_i 	= nY - (CHUNK_LEN * (nLaunches-1));}
			double				iMaps_by_chunk	= HEIGHT_CHUNK_i  * nX 		* sizeof(uint8_T);
			double				oMap_by_chunk	= HEIGHT_CHUNK_i  * nX 		* sizeof(uint32_T);
			double				chMat_by_chunk	= HEIGHT_CHUNK_i  * nC * nC * sizeof(uint32_T);
			printf("Size[%dx%d], ",HEIGHT_CHUNK_i,nX);
			printf("Offset[%dx%d]\t",CHUNK_LEN*iS,0);
			/*
			 * 	Allocate page-locked memory on RAM
			 */
			assert( cudaMallocHost( (void**)&host_iMap1, iMaps_by_chunk   ) == 0 );
			assert( cudaMallocHost( (void**)&host_iMap2, iMaps_by_chunk   ) == 0 );
			assert( cudaMallocHost( (void**)&host_roiMap,iMaps_by_chunk   ) == 0 );
			assert( cudaMallocHost( (void**)&host_oMap,   oMap_by_chunk   ) == 0 );
			assert( cudaMallocHost( (void**)&host_chMat, chMat_bytes_host ) == 0 );
			/*
			 * 	GDAL I/O
			 */
			// read data from HDD:
			GDALRasterIO( iBand1,  GF_Read, 0, CHUNK_LEN*iS, nX, HEIGHT_CHUNK_i, host_iMap1,  nX, HEIGHT_CHUNK_i, GDT_Byte, 0, 0 );
			GDALRasterIO( iBand2,  GF_Read, 0, CHUNK_LEN*iS, nX, HEIGHT_CHUNK_i, host_iMap2,  nX, HEIGHT_CHUNK_i, GDT_Byte, 0, 0 );
			GDALRasterIO( roiBand, GF_Read, 0, CHUNK_LEN*iS, nX, HEIGHT_CHUNK_i, host_roiMap, nX, HEIGHT_CHUNK_i, GDT_Byte, 0, 0 );
			//	-Allocate arrays on GPU device:
			assert( cudaMalloc((void **)&dev_iMap1, iMaps_by_chunk ) == 0 );
			assert( cudaMalloc((void **)&dev_iMap2, iMaps_by_chunk ) == 0 );
			assert( cudaMalloc((void **)&dev_roiMap,iMaps_by_chunk ) == 0 );
			assert( cudaMalloc((void **)&dev_oMap,  oMap_by_chunk  ) == 0 );
			assert( cudaMalloc((void **)&dev_chMat, chMat_by_chunk ) == 0 );
			assert( cudaMemset(dev_chMat,0,(size_t) chMat_by_chunk ) == 0 ); //IMPORTANT!!
		/*	//This allocates the 2D array of size nX*nY in device memory and returns pitch
			assert( cudaMallocPitch( (void**)&dev_iMap1, &iPitch1, iWIDTH_bytes, nY) == 0 );
			assert( cudaMallocPitch( (void**)&dev_iMap2, &iPitch2, iWIDTH_bytes, nY) == 0 );
			assert( cudaMallocPitch( (void**)&dev_oMap,  &oPitch,  oWIDTH_bytes, nY) == 0 );
			printf("iPitch1:\t%zu\niPitch2:\t%zu\noPitch:\t\t%zu\n",iPitch1,iPitch2,oPitch);
		 */
			//	-copy: RAM--->GPU
			assert( cudaMemcpyAsync(dev_iMap1 , host_iMap1,  iMaps_by_chunk, cudaMemcpyHostToDevice, stream[iS]) == 0 );
			assert( cudaMemcpyAsync(dev_iMap2 , host_iMap2,  iMaps_by_chunk, cudaMemcpyHostToDevice, stream[iS]) == 0 );
			assert( cudaMemcpyAsync(dev_roiMap, host_roiMap, iMaps_by_chunk, cudaMemcpyHostToDevice, stream[iS]) == 0 );
		/*	assert( cudaMemcpy2D(dev_iMap1,iPitch1,host_iMap1,(size_t) iWIDTH_bytes,(size_t) iWIDTH_bytes,(size_t) nY, cudaMemcpyHostToDevice) == 0 );
			assert( cudaMemcpy2D(dev_iMap2,iPitch2,host_iMap2,(size_t) iWIDTH_bytes,(size_t) iWIDTH_bytes,(size_t) nY, cudaMemcpyHostToDevice) == 0 );
			if(iPitch1 != iPitch2){ printf("Error: the pitch of the two input iMap's are different!\n"); exit(EXIT_FAILURE); }
		*/
			/*
			 * 	KERNELS LAUNCH
			 */
			// -kernel config:
			BLOCKSIZE         	= floor(sqrt( devProp.maxThreadsPerBlock ));
			GRIDSIZE			= 1 + ( HEIGHT_CHUNK_i/(BLOCKSIZE*BLOCKSIZE) );
			dim3 block(BLOCKSIZE,BLOCKSIZE,1);
			dim3 grid(GRIDSIZE,1,1);
			uint32_T pitch 		= nX;//(uint32_T)iPitch1/sizeof(uint8_T);
			// first kernel:
			assert( cudaEventRecord( k1_s, 0 )		== 0 );
			changemap<<<grid,block,0,stream[iS]>>>(	dev_iMap1,dev_iMap2, dev_roiMap,
										nX,HEIGHT_CHUNK_i,pitch,nC,
										dev_chMat,dev_oMap ); //cudaDeviceSynchronize();
			assert( cudaEventRecord( k1_e, 0 )		== 0 );
			assert( cudaMemcpyAsync(host_oMap,dev_oMap,oMap_by_chunk,cudaMemcpyDeviceToHost, stream[iS]) == 0 );
			//cudaDeviceSynchronize();
			// second kernel:
			assert( cudaEventRecord( k2_s, 0 )		== 0 );
			changemat<<<grid,block,0,stream[iS]>>>( dev_chMat, nC*nC, HEIGHT_CHUNK_i ); //cudaDeviceSynchronize();
			assert( cudaEventRecord( k2_e, 0 )		== 0 );
			assert( cudaMemcpyAsync(host_chMat,dev_chMat,chMat_bytes_host,cudaMemcpyDeviceToHost, stream[iS]) == 0 );
			cudaDeviceSynchronize();

			/*
			 * 	write CHUNK in oMap filename
			 */
			//	-write to HDD
			GDALRasterIO(oBand, GF_Write, 0, CHUNK_LEN*iS, nX, HEIGHT_CHUNK_i, host_oMap, nX, HEIGHT_CHUNK_i, GDT_UInt32, 0, 0);
/*			char Fname[255], pFormat[4];
			sprintf(Fname,"/home/giuliano/git/cuda/change-map-mat/data/oMap_chunk_#%d",iS);
			sprintf(pFormat,"%s","%5d ");
			write_generic_array_on_HDD( host_oMap,  nX,  HEIGHT_CHUNK_i, Fname, pFormat );
			sprintf(Fname,"/home/giuliano/git/cuda/change-map-mat/data/iMap1_chunk_#%d",iS);
			write_uint8T_array_on_HDD( host_iMap1,  nX,  HEIGHT_CHUNK_i, Fname, pFormat );
*/			/*
			 * 	sum chmat from current CHUNK
			 */
/*			sprintf(Fname,"%s%d","/home/giuliano/git/cuda/change-map-mat/data/chMat_chunk_#",iS);
			sprintf(pFormat,"%s","%9d ");
			write_generic_array_on_HDD( host_chMat,  nC,  nC, Fname, pFormat );
*/			for(int r=0;r<nC;r++) for(int c=0;c<nC;c++) host_chMat_sum[r+c*nC] += host_chMat[r+c*nC];

			// CUDA timer:
			assert( cudaEventRecord( stop, 0 ) == 0 );
			assert( cudaEventSynchronize( stop ) == 0 );
			assert( cudaEventElapsedTime( &elapsed_time_ms, start, stop ) == 0 );
			printf("Elapsed time=[%1.3fs], ",elapsed_time_ms/(float)1000.00);
			assert( cudaEventElapsedTime( &elapsed_time_ms, k1_s, k1_e ) == 0 );
			printf("k1[changemap] time=[%1.3fs], ",elapsed_time_ms/(float)1000.00);
			assert( cudaEventElapsedTime( &elapsed_time_ms, k2_s, k2_e ) == 0 );
			printf("k2[changemat] time=%1.3f [s]\n",elapsed_time_ms/(float)1000.00);

			/*
			 * 	FREE
			 */
			// CUDA Malloc Host free:
			assert( cudaFreeHost( host_iMap1 ) 		== 0 );
			assert( cudaFreeHost( host_iMap2 ) 		== 0 );
			assert( cudaFreeHost( host_roiMap) 		== 0 );
			assert( cudaFreeHost( host_oMap  ) 		== 0 );
			assert( cudaFreeHost( host_chMat ) 		== 0 );
			// CUDA Malloc free:
			assert( cudaFree(dev_iMap1 ) 			== 0 );
			assert( cudaFree(dev_iMap2 ) 			== 0 );
			assert( cudaFree(dev_roiMap) 			== 0 );
			assert( cudaFree(dev_oMap  ) 			== 0 );
			assert( cudaFree(dev_chMat ) 			== 0 );
			// Destroy events:
			assert( cudaEventDestroy( start )	 	== 0 );
			assert( cudaEventDestroy( stop )  		== 0 );
			assert( cudaEventDestroy( k1_s )	 	== 0 );
			assert( cudaEventDestroy( k1_e )  		== 0 );
			assert( cudaEventDestroy( k2_s )	 	== 0 );
			assert( cudaEventDestroy( k2_e )  		== 0 );
		}
	}

	save_chmat_on_HDD( nC, host_chMat_sum );

	GDALClose( oMap );
	CSLDestroy( papszOptions );
	VSIFree( host_chMat_sum );
	// C free:
/*	cudaFreeHost(host_iMap1);
	cudaFreeHost(host_iMap2);
	cudaFreeHost(host_oMap);
	cudaFreeHost(host_chMat);
*/
}

int main(int argc, char **argv)
{
	/*
	 * 	Based on GDAL
	 * 		In order to compile goto:
	 * 		Project->Properties->Settings->Tool Settings
	 * 		->NVCC Linker
	 * 			->Libraries(-l)	->add "gdal" (without virgolette)
	 * 			-> (-L)			->add /usr/lib
	 * 		->NVCC Compiler
	 * 			->Includes(-I)	-> /usr/include/gdal/
	 * 		see also this link: http://stackoverflow.com/questions/16148814/how-to-link-to-cublas-library-in-eclipse-nsight
	 */

	/*CUTTED CODE:
		char iFil1[255];
		char iFil2[255];
		const char *legend_level = "L1";
		const char *resolution = "100m";
		sprintf(iFil1,"/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/clc2000_%s_%sm",legend_level,resolution);

		//--
		if( GDALGetOverviewCount(hBand) > 0 ) printf( "Band has %d overviews.\n", GDALGetOverviewCount(hBand));
		if( GDALGetRasterColorTable( hBand ) != NULL ) printf( "Band has a color table with %d entries.\n",
			GDALGetColorEntryCount( GDALGetRasterColorTable(hBand)) );
		//--

		int[] overviewlist 	= {2,4,8,16,32};
		dataset.BuildOverviews(overviewlist);
	*/

	GDALDatasetH		iMap1,iMap2, roiMap;
	GDALDriverH			hDriver;
/*	uint8_T  			*host_iMap1, *host_iMap2;
	uint32_T 			*host_oMap, *host_chMat;
*/
//	unsigned int		deltaX=0,deltaY=0;
	// size of iMap's: [nX * nY]
	unsigned int 		nX, nY;
	// GPU device
	int					devCount;
	int					gpuSel;
	cudaDeviceProp		devProp;
//	size_t				iPitch1, iPitch2, oPitch;
	// get geo-referencing from iMaps
	double        		iGeoTransform[6];

	/*
	 * 	ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.

	/*
	 * 	SET VARIABLES (I should pass it to the main function as input)
	 */
	// open iMap's
	iMap1 				= GDALOpen( iFil1, GA_ReadOnly );
	iMap2 				= GDALOpen( iFil2, GA_ReadOnly );
	roiMap 				= GDALOpen( roiFil, GA_ReadOnly );
	if( iMap1 == NULL || iMap2 == NULL ){ printf("Error: cannot load input grids!\n"); exit(EXIT_FAILURE);}
	// driver of iMap's
	hDriver = GDALGetDatasetDriver( iMap1 );
	if( hDriver!=GDALGetDatasetDriver( iMap2 ) ){ printf("Error: the driver of iMaps are different!"); exit(EXIT_FAILURE); }
	// Spatial Reference System:
	if( GDALGetProjectionRef( iMap1 ) == NULL || GDALGetProjectionRef( iMap2 ) == NULL ){
		printf("Error: one or more input layers miss spatial reference system!");
		exit(EXIT_FAILURE);
	}
	printf( "Projection is:\t\t\t%s\n", GDALGetProjectionRef( iMap1 ) );
	GDALGetGeoTransform( iMap1, iGeoTransform );

	// get size
	nX = GDALGetRasterXSize( iMap1 );
	nY = GDALGetRasterYSize( iMap1 );
	if( nX!=GDALGetRasterXSize( iMap2  ) || nY!=GDALGetRasterYSize( iMap2  ) ||
		nX!=GDALGetRasterXSize( roiMap ) || nY!=GDALGetRasterYSize( roiMap ) ){
		printf("Error: the input iMaps have different SIZE!\n");
		exit(EXIT_FAILURE);
	}
	printf("Size of iMaps:\t\t\t%dx%d\t[el]\n",nY,nX);
	if(GDALGetRasterCount( iMap1 )>1 || GDALGetRasterCount( iMap1 )!=GDALGetRasterCount( iMap2 )){
		printf("Error: iMaps have more than 1 band! [not allowed]");
		exit(EXIT_FAILURE);
	}

	//------------ single/multiple GPUs ------------//
	// find out required memory:
	unsigned long long int	iMaps_bytes		= nY*nX*sizeof(uint8_T  );
	unsigned long long int	oMap_bytes		= nY*nX*sizeof(uint32_T );
	unsigned long long int	chMat_bytes		= nclasses*nclasses*nY*sizeof(uint32_T);
	unsigned long long int	reqMem			= (iMaps_bytes*3 + oMap_bytes + chMat_bytes);
	bool 					found 			= false;
	// gpu count:
	cudaGetDeviceCount(&devCount);
	// check which GPU can run, otherwise switch to multi-GPU process!
	for(int i=0;i<devCount;i++){
		cudaGetDeviceProperties(&devProp, i);
		if( (unsigned long long int)devProp.totalGlobalMem > reqMem ){
			// I find out that I must use GPU i:
			gpuSel	= i;
			found	= true;
			break;
		}
	}

	if( found ){
		run_kernels_single_gpu( gpuSel, nX, nY, nclasses, iMap1, iMap2, roiMap, GDALGetProjectionRef( iMap1 ), iGeoTransform, hDriver );
		/*
		 * 	Force running multi-gpu for Italy:
		 * 	run_kernels_multi_gpu_dev( nX, nY, nclasses, iMap1, iMap2, roiMap, GDALGetProjectionRef( iMap1 ), iGeoTransform, hDriver );
		 */
	}
	else{
		//run_kernels_multi_gpu( nX, nY, nclasses, iMap1, iMap2, roiMap, GDALGetProjectionRef( iMap1 ), iGeoTransform, hDriver );
		run_kernels_multi_gpu_dev( nX, nY, nclasses, iMap1, iMap2, roiMap, GDALGetProjectionRef( iMap1 ), iGeoTransform, hDriver );
	}

	//------------ KILL ------------//
	// GDAL free/close:
	GDALClose( iMap1 );
	GDALClose( iMap2 );
	// Destroy context
	assert( cudaDeviceReset() == cudaSuccess );

	printf("\n\nFinished!!\n");

	return 0;
}

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

#define KRED  "\x1B[31m"
//#define uintXX_t uint32_T

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
                           uint32_T nX, uint32_T nY, uint32_T pitch, uint32_T nclasses,
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
		k = tid*nclasses*nclasses;
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
			chMat[ (k + (iCl1+iCl2*nclasses))*iroi ]++;
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

	GDALDatasetH		iMap1,iMap2, oMap, roiMap;
	GDALRasterBandH		iBand1,iBand2, oBand, roiBand;
	GDALDriverH			hDriver;
	uint8_T				*dev_iMap1, *dev_iMap2;
	uint32_T			*dev_oMap, *dev_chMat;
	uint8_T				*dev_roiMap;
/*	uint8_T  			*host_iMap1, *host_iMap2;
	uint32_T 			*host_oMap, *host_chMat;
*/	uint32_T 			nclasses;//tiledimX, tiledimY, ntilesX, ntilesY;
//	unsigned int		deltaX=0,deltaY=0;
	// cuda timer:
	float 				elapsed_time_ms = 0.0f;
	cudaEvent_t 		start, stop;
	// size of iMap's: [nX * nY]
	unsigned int 		nX, nY;
	// block & grid kernel size:
	unsigned int 		BLOCKSIZE, GRIDSIZE;
	// GPU device
	int					devCount;
	int					gpuSel;
	cudaDeviceProp		devProp;
//	size_t				iPitch1, iPitch2, oPitch;
	// options for GDALCreate:
	char 				**papszOptions = NULL;
	// get geo-referencing from iMaps
	double        		iGeoTransform[6];

	/*
	 * 	ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	/*
	 * 	SET VARIABLES (I should pass it to the main function as input)
	 */
	gpuSel		= 0;
	nclasses	= 5 +1;
	// filename of maps:
	//	I/-
	//	**ITALY**
	const char *iFil1	= "/home/giuliano/git/cuda/change-map-mat/data/clc2000_L1_100m.tif";
	const char *iFil2	= "/home/giuliano/git/cuda/change-map-mat/data/clc2006_L1_100m.tif";
	const char *roiFil	= "/home/giuliano/git/cuda/change-map-mat/data/clc_roi-ita_100m.tif";
	//	**EUROPE**
/*	const char *iFil1	= "/home/giuliano/git/cuda/change-map-mat/data/g100_00.tif";
	const char *iFil2	= "/home/giuliano/git/cuda/change-map-mat/data/g100_06.tif";
	const char *roiFil	= "/home/giuliano/git/cuda/change-map-mat/data/clc_roi-eur_100m.tif";
*/	//	-/O
	const char *oFil	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/ch00-06_L1_100m";
	//	**chMat**
	//	-/O
	const char *chFil	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/chMat.txt";

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
	printf("Size of iMaps:\t\t\t%dx%d\t[el]\n",nX,nY);
	if(GDALGetRasterCount( iMap1 )>1 || GDALGetRasterCount( iMap1 )!=GDALGetRasterCount( iMap2 )){
		printf("Error: iMaps have more than 1 band! [not allowed]");
		exit(EXIT_FAILURE);
	}
	iBand1				= GDALGetRasterBand( iMap1,  GDALGetRasterCount( iMap1  ) ); // GDALGetRasterCount( iMap1 ) is equal to 1.
	iBand2				= GDALGetRasterBand( iMap2,  GDALGetRasterCount( iMap2  ) );
	roiBand				= GDALGetRasterBand( roiMap, GDALGetRasterCount( roiMap ) );

	// allocate on RAM
//	uint32_T			iWIDTH_bytes= 	 nX*sizeof(uint8_T  );
//	uint32_T			oWIDTH_bytes= 	 nX*sizeof(uint32_T );
	double				iMaps_bytes		= nY*nX*sizeof(uint8_T  );
	double				oMap_bytes		= nY*nX*sizeof(uint32_T );
	double				chMat_bytes		= nclasses*nclasses*nY*sizeof(uint32_T);
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

	//------------ gpu-devices ------------//
	cudaGetDeviceCount(&devCount);
	cudaGetDeviceProperties(&devProp, gpuSel);
	printf("Texture alignment:\t\t%zu\t\t[el]\n", devProp.textureAlignment);
	printf("Concurrent copy and execute:\t%d\n", devProp.deviceOverlap);
	double reqMem		= (iMaps_bytes*3 + oMap_bytes + chMat_bytes)/pow(1024,3);
	double avGlobMem	= (double)devProp.totalGlobalMem/pow(1024,3);
	printf("Global memory on GPU[%d]:\t%.2f\t\t[Gbytes]\n", gpuSel,avGlobMem );
	printf("Required memory on GPU:\t\t%.2f\t\t[Gbytes]\n", reqMem);
	printf("Available memory:\t\t%.2f\t\t[Gbytes]\n", avGlobMem-reqMem);

	if(avGlobMem < 0) {
		printf("Error: Available Global Memory on gpu %s #%d not sufficient!\n",devProp.name, gpuSel );
		exit(EXIT_FAILURE);
	}

	// read from HDD
	GDALRasterIO( iBand1,  GF_Read, 0, 0, nX, nY, host_iMap1,  nX, nY, GDT_Byte, 0, 0 );
	GDALRasterIO( iBand2,  GF_Read, 0, 0, nX, nY, host_iMap2,  nX, nY, GDT_Byte, 0, 0 );
	GDALRasterIO( roiBand, GF_Read, 0, 0, nX, nY, host_roiMap, nX, nY, GDT_Byte, 0, 0 );

	//------------ KERNELS ------------//
	cudaEventRecord( start, 0 );
	//
	//	-select GPU (a multi-gpu management is possible with "async" calls of memcopies and kernels):
	cudaSetDevice(0);
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
	/*	-kernel call 	[changemap]
		const uint8_T *iMap1
		const uint8_T *iMap2
		uint32_T tiledimX
		uint32_T tiledimY
		uint32_T ntilesX
		uint32_T ntilesY
		uint32_T chMat_dim
		uint32_T *chMat
		uint32_T *oMap
	*/
	changemap<<<grid,block>>>(	dev_iMap1,dev_iMap2, dev_roiMap,
								nX,nY,pitch,nclasses,
								dev_chMat,dev_oMap );
	cudaDeviceSynchronize();
	//
	// -kernel config:
	/*	--call kernel 	[changemat]
	 * 	(uint32_T *chMat, uint32_T chMat_dim, uint32_T ntiles)
	 */
	changemat<<<grid,block>>>( dev_chMat, nclasses*nclasses, nY );
	cudaDeviceSynchronize();
	//
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

	//------------ OUTPUT changematrix ------------//
	// ...to be completed
	FILE *fid ;
	fid = fopen(chFil,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",chFil); exit(1); }
	for(int i=0;i<nclasses;i++){
		for(int j=0;j<nclasses;j++){
			fprintf(fid, "%8d ", host_chMat[j+i*nclasses]);
		}fprintf(fid,"\n");
	} fclose(fid);

	//------------ OUTPUT changemap ------------//
	// CREATE DATASET on HDD
	//	-options: tiling with block 512x512
	papszOptions = CSLSetNameValue( papszOptions, "TILED", "YES");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKXSIZE", "512");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKYSIZE", "512");
	//	-instantiate GRID
	oMap = GDALCreate( hDriver, oFil, nX, nY, 1, GDT_UInt32, papszOptions );
	//	-set projection
	GDALSetProjection( 	 oMap, GDALGetProjectionRef( iMap1 ) );
	//	-set geospatial transformation
	GDALSetGeoTransform( oMap, iGeoTransform );
	//	-band
	oBand = GDALGetRasterBand( oMap, 1 );
	//	-write to HDD
	GDALRasterIO(oBand, GF_Write, 0, 0, nX, nY, host_oMap, nX, nY, GDT_UInt32, 0, 0);

	//------------ KILL ------------//
	// GDAL free/close:
	GDALClose( iMap1 );
	GDALClose( iMap2 );
	GDALClose( oMap );
	CSLDestroy( papszOptions );
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
*/	// Destroy context
	assert( cudaDeviceReset() == cudaSuccess );

	printf("\n\nFinished!!\n");

	return 0;
}

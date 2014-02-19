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
__global__ void changemap( const uint8_T *iMap1,const uint8_T *iMap2,
                           uint32_T nX, uint32_T nY, uint32_T pitch, uint32_T nclasses,
                           uint32_T *chMat, uint32_T *oMap )
{
	uint32_T iCl1,iCl2,i_row,x,k;
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
			iCl1 			= iMap1[i_row+x];
			iCl2 			= iMap2[i_row+x];
			oMap[i_row+x] 	= ( iCl1 + iCl2*100 );
			chMat[ k + (iCl1+iCl2*nclasses) ]++;
		}
	}
}


__global__ void changemap_shmem( 	const uint8_T *iMap1,const uint8_T *iMap2,
                           	   	   uint32_T nX, uint32_T nY, uint32_T pitch, uint32_T nclasses,
                           	   	   uint32_T *chMat, uint32_T *oMap )
{
	__shared__ uint8_T s_iMap1[1024];
	__shared__ uint8_T s_iMap2[1024];
	__shared__ uint8_T s_oMap_[1024];
	__shared__ uint8_T s_chMat[36];

	uint32_T iCl1,iCl2,i_row,x,k;

	// Tile ID: (of size 1024x1)
	uint32_T Tid =  tx;
	// Map ID:
	uint32_T Mid = (tx + blockDim.x*( ty + blockDim.y  *( bx + gridDim.x*by)));
	// iTile within GRID:
	uint32_T Gid =  bx + gridDim.x*by;

	if( Gid<(gridDim.x*gridDim.y) ) {

		s_iMap1[Tid] = iMap1[Mid]; syncthreads();
		s_iMap2[Tid] = iMap2[Mid]; syncthreads();
		s_oMap_[Tid] = oMap [Mid]; syncthreads();

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
			iCl1 			= iMap1[i_row+x];
			iCl2 			= iMap2[i_row+x];
			oMap[i_row+x] 	= ( iCl1 + iCl2*100 );
			chMat[ k + (iCl1+iCl2*nclasses) ]++;
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

	GDALDatasetH		iMap1,iMap2, oMap;
	GDALRasterBandH		iBand1,iBand2, oBand;
	GDALDriverH			hDriver;
	uint8_T				*dev_iMap1, *dev_iMap2;
	uint32_T			*dev_oMap, *dev_chMat;
	uint8_T  			*host_iMap1, *host_iMap2;
	uint32_T 			*host_oMap, *host_chMat;
	uint32_T 			nclasses;//tiledimX, tiledimY, ntilesX, ntilesY;
//	unsigned int		deltaX=0,deltaY=0;
	// size of iMap's: [nX * nY]
	unsigned int 		nX, nY;
	// block & grid kernel size:
	unsigned int 		BLOCKSIZE, GRIDSIZE;
	// GPU device
	int					devCount;
	cudaDeviceProp		devProp;
//	size_t				iPitch1, iPitch2, oPitch;
	// options for GDALCreate:
	char 				**papszOptions = NULL;
	// get geo-referencing from iMaps
	double        		iGeoTransform[6];

	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.

	// SET VARIABLES (I should pass it to the main function as input)
	nclasses	= 5 +1;

	// filename of maps:
	const char *iFil1 = "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/clc2000_L1_100m";
	const char *iFil2 = "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/clc2006_L1_100m";
	const char *oFil  = "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/ch00-06_L1_100m";
	const char *chFil = "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/chMat.txt";

	// open iMap's
	iMap1 = GDALOpen( iFil1, GA_ReadOnly );
	iMap2 = GDALOpen( iFil2, GA_ReadOnly );
	if( iMap1 == NULL || iMap2 == NULL ){ printf("Error: cannot load input grids!\n"); return 1;}
	// driver of iMap's
	hDriver = GDALGetDatasetDriver( iMap1 );
	if( hDriver!=GDALGetDatasetDriver( iMap2 ) ){ printf("Error: the driver of iMaps are different!"); return 1; }
	// Spatial Reference System:
	if( GDALGetProjectionRef( iMap1 ) == NULL || GDALGetProjectionRef( iMap2 ) == NULL ){
		printf("Error: one or more input layers miss spatial reference system!");
		return 1;
	}
	printf( "Projection is:\n\n `%s'\n\n", GDALGetProjectionRef( iMap1 ) );
	GDALGetGeoTransform( iMap1, iGeoTransform );

	// get size
	nX = GDALGetRasterXSize( iMap1 );
	nY = GDALGetRasterYSize( iMap1 );
	if( nX!=GDALGetRasterXSize( iMap2 ) || nY!=GDALGetRasterYSize( iMap2 ) ){
		printf("Error: the input iMaps have different SIZE!\n");
		return 1;// 1 ==> error
	}
	if(GDALGetRasterCount( iMap1 )>1 || GDALGetRasterCount( iMap1 )!=GDALGetRasterCount( iMap2 )){
		printf("Error: iMaps have more than 1 band! [not allowed]");
		return 1;
	}
	iBand1 = GDALGetRasterBand( iMap1, GDALGetRasterCount( iMap1 ) ); // GDALGetRasterCount( iMap1 ) is equal to 1.
	iBand2 = GDALGetRasterBand( iMap2, GDALGetRasterCount( iMap2 ) );

	// allocate on RAM
//	uint32_T			iWIDTH_bytes= 	 nX*sizeof(uint8_T  );
//	uint32_T			oWIDTH_bytes= 	 nX*sizeof(uint32_T );
	uint32_T			iMaps_bytes = nY*nX*sizeof(uint8_T  );
	uint32_T			oMap_bytes  = nY*nX*sizeof(uint32_T );
	uint32_T			chMat_bytes = nclasses*nclasses*nY*sizeof(uint32_T);
/*	uint8_T  			*host_iMap1 = (uint8_T  *) CPLMalloc( iMaps_bytes );
	uint8_T  			*host_iMap2 = (uint8_T  *) CPLMalloc( iMaps_bytes );
	uint32_T 			*host_oMap 	= (uint32_T *) CPLMalloc( oMap_bytes  );
	uint32_T 			*host_chMat = (uint32_T *) CPLMalloc( chMat_bytes );
*/
	assert( cudaMallocHost( (void**)&host_iMap1, iMaps_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_iMap2, iMaps_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_oMap,   oMap_bytes ) == 0 );
	assert( cudaMallocHost( (void**)&host_chMat, chMat_bytes ) == 0 );

	// read from HDD
	GDALRasterIO( iBand1, GF_Read, 0, 0, nX, nY,
				  host_iMap1, nX, nY, GDT_Byte, 0, 0 );
	GDALRasterIO( iBand2, GF_Read, 0, 0, nX, nY,
				  host_iMap2, nX, nY, GDT_Byte, 0, 0 );

	//------------ gpu-devices ------------//
	cudaGetDeviceCount(&devCount);
	cudaGetDeviceProperties(&devProp, devCount-1);
	printf("Texture alignment:\t\t%zu\n", devProp.textureAlignment);
	printf("Concurrent copy and execute:\t%d\n", devProp.deviceOverlap);

	//------------ KERNELS ------------//
	//
	//	-select GPU (a multi-gpu management is possible with "async" calls of memcopies and kernels):
	cudaSetDevice(0);
	//	-Allocate arrays on GPU device:
	assert( cudaMalloc((void **)&dev_iMap1,iMaps_bytes) == 0 );
	assert( cudaMalloc((void **)&dev_iMap2,iMaps_bytes) == 0 );
	assert( cudaMalloc((void **)&dev_oMap,  oMap_bytes) == 0 );
	assert( cudaMalloc((void **)&dev_chMat,chMat_bytes) == 0 );
	assert( cudaMemset(dev_chMat,0,(size_t)chMat_bytes) == 0 ); //IMPORTANT!!

/*	//This allocates the 2D array of size nX*nY in device memory and returns pitch
 	assert( cudaMallocPitch( (void**)&dev_iMap1, &iPitch1, iWIDTH_bytes, nY) == 0 );
	assert( cudaMallocPitch( (void**)&dev_iMap2, &iPitch2, iWIDTH_bytes, nY) == 0 );
	assert( cudaMallocPitch( (void**)&dev_oMap,  &oPitch,  oWIDTH_bytes, nY) == 0 );
 	printf("iPitch1:\t%zu\niPitch2:\t%zu\noPitch:\t\t%zu\n",iPitch1,iPitch2,oPitch);
 */
	int nStreams = 1, i=0,j=0;
	//	-create stream:
	cudaStream_t stream[nStreams];
	for (i=1; i<=nStreams; i++) { assert( cudaStreamCreate(&stream[i]) == 0); }
	for (i=1; i<=nStreams; i++) {
		j = i-1;
	//	-offset:
		uint32_T offset  = (nX*nY) * j/nStreams;
		uint32_T offset2 = nclasses*nclasses*nY * j/nStreams;
	//	-copy: RAM--->GPU
//		assert( cudaMemcpy(dev_iMap1,host_iMap1,iMaps_bytes,cudaMemcpyHostToDevice) == 0 );
//		assert( cudaMemcpy(dev_iMap2,host_iMap2,iMaps_bytes,cudaMemcpyHostToDevice) == 0 );
		assert( cudaMemcpyAsync(dev_iMap1+offset, host_iMap1+offset, iMaps_bytes/nStreams, cudaMemcpyHostToDevice, stream[i]) == 0 );
		assert( cudaMemcpyAsync(dev_iMap2+offset, host_iMap2+offset, iMaps_bytes/nStreams, cudaMemcpyHostToDevice, stream[i]) == 0 );
/*	assert( cudaMemcpy2D(dev_iMap1,iPitch1,host_iMap1,(size_t) iWIDTH_bytes,(size_t) iWIDTH_bytes,(size_t) nY, cudaMemcpyHostToDevice) == 0 );
	assert( cudaMemcpy2D(dev_iMap2,iPitch2,host_iMap2,(size_t) iWIDTH_bytes,(size_t) iWIDTH_bytes,(size_t) nY, cudaMemcpyHostToDevice) == 0 );
*/
//	if(iPitch1 != iPitch2){ printf("Error: the pitch of the two input iMap's are different!\n"); return 1; }
	//
	// -kernel config:
		BLOCKSIZE         	= floor(sqrt( devProp.maxThreadsPerBlock ));
		GRIDSIZE			= 1 + (nY/(BLOCKSIZE*BLOCKSIZE))/nStreams;
		dim3 block(BLOCKSIZE,BLOCKSIZE,1);
		dim3 grid(GRIDSIZE,1,1);
		uint32_T pitch = nX;//(uint32_T)iPitch1/sizeof(uint8_T);
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
		changemap<<<grid,block,0,stream[i]>>>(	dev_iMap1+offset,dev_iMap2+offset,
									nX,nY/nStreams,pitch,nclasses,
									dev_chMat+offset2,dev_oMap+offset );
		assert( cudaMemcpyAsync(host_oMap+offset,dev_oMap+offset,oMap_bytes/nStreams,cudaMemcpyDeviceToHost, stream[i]) == 0 );
	}
//	cudaDeviceSynchronize();
	//
	// -kernel config:
	BLOCKSIZE         	= floor(sqrt( devProp.maxThreadsPerBlock ));
	GRIDSIZE			= 1 + (nY/(BLOCKSIZE*BLOCKSIZE));
	dim3 block(BLOCKSIZE,BLOCKSIZE,1);
	dim3 grid(GRIDSIZE,1,1);
	/*	--call kernel 	[changemat]
	 * 	(uint32_T *chMat, uint32_T chMat_dim, uint32_T ntiles)
	 */
	changemat<<<grid,block>>>( dev_chMat, nclasses*nclasses, nY );
	cudaDeviceSynchronize();

	//------------ OUTPUT changematrix ------------//
	// ...to be completed
	printf("to-be-saved:\n\t%s...\n",chFil);
	assert( cudaMemcpy(host_chMat,dev_chMat,chMat_bytes/nY,cudaMemcpyDeviceToHost) == 0 );
	FILE *fid ;
	fid = fopen(chFil,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",chFil); exit(1); }
	for(int i=1;i<nclasses;i++){
		for(int j=1;j<nclasses;j++){
			fprintf(fid, "%8d ", host_chMat[j+i*nclasses]);
		}fprintf(fid,"\n");
	} fclose(fid);

	//------------ OUTPUT changemap ------------//
	// CREATE DATASET on HDD
	//	-copy: GPU--->RAM
//	assert( cudaMemcpy(host_oMap,dev_oMap,oMap_bytes,cudaMemcpyDeviceToHost) == 0 );
//	assert( cudaMemcpy2D(host_oMap,(size_t) oWIDTH_bytes,dev_oMap,oPitch,(size_t) oWIDTH_bytes,(size_t) nY, cudaMemcpyDeviceToHost) == 0 );
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
//	VSIFree( host_iMap1 );
//	VSIFree( host_iMap2 );
	// CUDA free:
	cudaFree(dev_iMap1);
	cudaFree(dev_iMap2);
	cudaFree(dev_oMap);
	cudaFree(dev_chMat);
	// C free:
	cudaFreeHost(host_iMap1);
	cudaFreeHost(host_iMap2);
	cudaFreeHost(host_oMap);
	cudaFreeHost(host_chMat);
	// Destroy context
	assert( cudaDeviceReset() == cudaSuccess );

	return 0;
}

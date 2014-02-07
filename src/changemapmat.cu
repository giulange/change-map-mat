// standard
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "/usr/local/MATLAB/R2013a/extern/include/tmwtypes.h"
#include <stdio.h>

#include "gdal.h"
#include "cpl_conv.h" /* for CPLMalloc() */
#include "cpl_string.h"
#include "gdal_priv.h"

#define cuda 1
#define pinned 0

#if (cuda)
	#include <cuda.h>
	#include <cuda_runtime.h>
/*
	#if CUDA_VERSION==5000
        	#include <helper_cuda.h>
	#else
	        #include <cutil_inline.h>
        	#define gpuGetMaxGflopsDeviceId cutGetMaxGflopsDeviceId
			#define _ConvertSMVer2Cores(major,minor) ((major << 4) + minor)
	#endif
*/
	// TYPE of DATA???????
	#define tx threadIdx.x
	#define ty threadIdx.y
	#define bx blockIdx.x
	#define by blockIdx.y
	#define tid  (tx + blockDim.x*( ty + blockDim.y  *( bx + gridDim.x*by)))
#endif

//#define uintXX_t uint32_T

__global__ void changemap( const uint8_T *iMap1,const uint8_T *iMap2,
                           uint32_T tiledimX, uint32_T tiledimY,
                           uint32_T ntilesX, uint32_T ntilesY,
                           uint32_T chMat_dim,
                           uint32_T *chMat, uint32_T *oMap )
{
/*
 INPUTS:
    iMap1:      map of OLDER year;
    iMap2:      map of NEWER year;
    tiledim:    number of pixels given to one thread;
    ntiles:     to be splitted between blocks of threads and grid;
    chMat_dim:  dimension of cross matrix;
 OUTPUTS:
    chMat:      change matrix;
    oMap:       map storing difference between iMap1 and iMap2;
*/
	uint32_T i,x,y,yo,k;
	if (tid<ntilesX*ntilesY){
        /* i: offset to move the pointer tile-by-tile on the input/output maps
              (iMap1,iMap2,oMap): this is helpful to assign each tile to a thread.
        */
		i = tiledimX * ( ntilesX*tiledimY*(tid/ntilesX) + tid%ntilesX );
        /* k: offset to move the pointer tile-by-tile on the output change
              matrix (chMat):
        */
		k = tid*chMat_dim*chMat_dim;
        /*  Now that "i" is defined (that is the pointer to the first element
            of any tile), I have to offset along x and y directions (see the
            two for loops on x and y).
        */
        for(y=0;y<tiledimY;y++){
            /*  Since movement of pointer along y must consider the extent of
                ntilesX*tiledimX for "carriage return", I use the yo variable:
            */
            yo = y*ntilesX*tiledimX;
            // The pointer along the x direction is straightforward:
            for(x=0;x<tiledimX;x++) {
                oMap[i+x+yo]=(uint32_T)iMap1[i+x+yo]+(uint32_T)(iMap2[i+x+yo])*100;
                chMat[k+(uint32_T)(iMap2[i+x+yo])+(uint32_T)(iMap1[i+x+yo])*chMat_dim]++;
            }
        }
	}
}

__global__ void changemat(uint32_T *chMat, uint32_T chMat_dim, uint32_T ntiles)
{
/*
 DESCRIPTION:
 	This is a "reduce" kernel
 INPUTS:
    chMat_dim:  dimX * dimY of cross matrix;
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
		k=tid*chMat_dim;
		if (tid<i) {
            // j:   is the pointer that moves within each k tile.
			for(j=0;j<chMat_dim;j++) {
                /*  It sums - in k tile - k tile with the k+ntiles/2
                    ( remember that ntiles/2 is expressed by i (after i>>1) )
                    At the end of while loop the change signatures from all
                    pixels of the iMap1 & iMap2 are summed in the first tile.
                */
                		chMat[k+j]=chMat[k+j]+chMat[k+j+i*chMat_dim];
                /*
                    If the number of tiles at current while loop is odd,
                    then the first thread has to add to its k tile (i.e. the
                    first tile) the tile excluded from summation by the
                    previous line of code.
                */
                		if ((tid==0)&&(isodd)) {
                    //printf("dispari! i=%d ii=%d nt=%d\n",i,ii,nt);
                    			chMat[k+j]=chMat[k+j]+chMat[k+j+2*i*chMat_dim];
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
	*/

	GDALDatasetH	iMap1,iMap2, oMap;
	GDALRasterBandH	iBand1,iBand2, oBand;
	GDALDriverH		hDriver;

	GDALAllRegister();

	// size of iMap's: [nX * nY]
	unsigned int nX, nY;
	// options for GDALCreate:
	char **papszOptions = NULL;
	// get georeferencing from iMaps
	double        iGeoTransform[6];
	// filename of maps:
	const char *iFil1 = "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/clc2000_L1_100m";
	const char *iFil2 = "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/clc2006_L1_100m";
	const char *oFil  = "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/ch00-06_L1_100m";

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
	}
	iBand1 = GDALGetRasterBand( iMap1, GDALGetRasterCount( iMap1 ) ); // GDALGetRasterCount( iMap1 ) is equal to 1.
	iBand2 = GDALGetRasterBand( iMap2, GDALGetRasterCount( iMap2 ) );

	/*
	GDALGetBlockSize( hBand, &nBlockXSize, &nBlockYSize );
	printf( "Block=%dx%d Type=%s, ColorInterp=%s\n", nBlockXSize, nBlockYSize, GDALGetDataTypeName(GDALGetRasterDataType(hBand)),
			GDALGetColorInterpretationName( GDALGetRasterColorInterpretation(hBand)) );
	if( GDALGetOverviewCount(hBand) > 0 ) printf( "Band has %d overviews.\n", GDALGetOverviewCount(hBand));
	if( GDALGetRasterColorTable( hBand ) != NULL ) printf( "Band has a color table with %d entries.\n",
			GDALGetColorEntryCount( GDALGetRasterColorTable(hBand)) );
	*/


	// allocate on RAM
	float *host_iMap1 = (float *) CPLMalloc(sizeof(float)*nX*nY);
	float *host_iMap2 = (float *) CPLMalloc(sizeof(float)*nX*nY);
	// read from HDD
	GDALRasterIO( iBand1, GF_Read, 0, 0, nX, nY,
				  host_iMap1, nX, nY, GDT_Float32, 0, 0 );
	GDALRasterIO( iBand2, GF_Read, 0, 0, nX, nY,
				  host_iMap2, nX, nY, GDT_Float32, 0, 0 );

	// kernels...




	//------------ OUTPUT MAP ------------//
	// create DATASET on HDD
	papszOptions = CSLSetNameValue( papszOptions, "TILED", "YES");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKXSIZE", "512");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKYSIZE", "512");
	oMap = GDALCreate( hDriver, oFil, nX, nY, 1, GDT_Float32, papszOptions );

	// set projection
	GDALSetProjection( 	 oMap, GDALGetProjectionRef( iMap1 ) );
	GDALSetGeoTransform( oMap, iGeoTransform );

	// band
	oBand = GDALGetRasterBand( oMap, 1 );

	// write to HDD
	GDALRasterIO(oBand, GF_Write, 0, 0, nX, nY, host_iMap1, nX, nY, GDT_Float32, 0, 0);


	//------------ FREE ------------//
	GDALClose( iMap1 );
	GDALClose( iMap2 );
	GDALClose( oMap );
	CSLDestroy( papszOptions );
	VSIFree( host_iMap1 );
	VSIFree( host_iMap2 );

	/*
	gdal.AllRegister();

	Dataset hDataset;
	double[] adfGeoTransform = new double[6];
	String iFilename	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/rasterize/clc2000_L3_100m.tif";
	hDataset = gdal.Open(iFilename, gdalconstConstants.GA_ReadOnly);
	String prj=hDataset.GetProjectionRef();
	hDataset.GetGeoTransform(adfGeoTransform);
	//SpatialReference hSRS;

	String filename		= "/home/giuliano/imp_trial.tif";
	String[] options	= {"TILED=yes","BLOCKXSIZE=512","BLOCKYSIZE=512"};
	Dataset dataset 	= null;
	Driver driver 		= null;
	Band band 			= null;
	int[] overviewlist 	= {2,4,8,16,32};
	driver				= gdal.GetDriverByName("GTiff");
	dataset 			= driver.Create(filename, imWidth, imHeght, 1, gdalconst.GDT_Int32, options);
	dataset.SetProjection(prj);
	dataset.SetGeoTransform(adfGeoTransform);
	dataset.BuildOverviews(overviewlist);
	band 				= dataset.GetRasterBand(1);
	band.WriteRaster(0, 0, imWidth, imHeght, host_oMap);
*/

/* INFO
	info0       = geotiffinfo([Fldr,'clc2000_',Corine_level,'_',cellsize,'m']);
	info6       = geotiffinfo([Fldr,'clc2006_',Corine_level,'_',cellsize,'m']);
*/

/* DATA-TYPES
	iMap1                   = uint8(C0);
	iMap2                   = uint8(C6);
	tiledim                 = uint32(tiledim);
	ntiles                  = uint32(ntiles);
	chMat_dim               = uint32(chMat_dim);
	chMat                   = zeros([chMat_dim,chMat_dim,prod(ntiles)],'uint32');
	oMap                    = zeros(size(C0),'uint32');
*/

/* KERNEL-CONF
	blockDim            = floor(sqrt(k.MaxThreadsPerBlock))+0;
	k.ThreadBlockSize   = [blockDim,blockDim,1];
	GRIDSIZE            = 1+ceil( sqrt(prod(ntiles)/(blockDim^2)) );
	k.GridSize          = [GRIDSIZE, GRIDSIZE, 1];
	% RUN:
	[chMat,oMap]        = feval(k, iMap1,iMap2,tiledim(1),tiledim(2),ntiles(1),ntiles(2),chMat_dim,chMat,oMap);
	oMap                = gather(oMap);
*/

/* KERNEL-CONF
	k2.ThreadBlockSize  = [blockDim,blockDim,1];
	k2.GridSize         = [GRIDSIZE, GRIDSIZE, 1];
	chMat_all           = feval( k2, chMat,chMat_dim*chMat_dim,prod(ntiles) );
	chMat_sum           = gather( chMat_all(1:chMat_dim,1:chMat_dim, 1) );
*/

	return 0;
}

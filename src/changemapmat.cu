#define cuda 1
#define pinned 0

#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "/usr/local/MATLAB/R2013a/extern/include/tmwtypes.h"
#include <stdio.h>

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
/*
	From "gprism_inference_cudakernel_BUFFER.cu":
	unsigned long int location = (gridDim.x*blockIdx.y + blockIdx.x) *
        (blockDim.x*blockDim.y) + (blockDim.x*threadIdx.y + threadIdx.x);
*/
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

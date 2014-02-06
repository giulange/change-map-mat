change-map-mat
==============

Change map/matrix computation with CUDA for WebGIS tools

DESCRIPTION:
The core operation consists in the calculation of an aggregated map of
changes in all classes between maps of the state of classes in two
different time snapshots.

The whole CUDA implementation is made of two kernels:

	1) changemap
			It computes the final output oMap and the intermediate 
			output chMat which is processed in the next kernel.
			The interface follows:
				const uint8_T *iMap1
				const uint8_T *iMap2
				uint32_T tiledimX
				uint32_T tiledimY
				uint32_T ntilesX
				uint32_T ntilesY
				uint32_T chMat_dim
				uint32_T *chMat
				uint32_T *oMap

	2) changemat
			The first kernel computes the chMat 3-D array in which each tile
			is accounted in a singular plane of that 3-D array.
			This is a reduction kernel function in which the 3-D chMat is 
			reduced along the third dimension producing the chMat 2-D output 
			of this function.
			The interface follows:
				uint32_T *chMat
				uint32_T chMat_dim
				uint32_T ntiles

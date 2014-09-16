/**
 * Self-Organizing Maps on a cluster
 *  Copyright (C) 2013 Peter Wittek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <string>

using namespace std;

#ifndef SOMOCLU_H
#define SOMOCLU_H

#if HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_MPI         
#include <mpi.h> 
#endif


#define DENSE_CPU 0
#define DENSE_GPU 1
#define SPARSE_CPU 2

/// The neighbor_fuct value below which we consider 
/// the impact zero for a given node in the map
#define NEIGHBOR_THRESHOLD 0.05

#define FLOAT_T float

/// Sparse structures and routines
struct svm_node
{
	int index;
	FLOAT_T value;
};

/// Core data structures
struct core_data
{
	FLOAT_T *codebook;
	int *globalBmus;
	FLOAT_T *uMatrix;
	int codebook_size;
	int globalBmus_size;
	int uMatrix_size;

        int *global2ndBmus;
};


FLOAT_T euclideanDistanceOnToroidMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y, const unsigned int nSomX, const unsigned int nSomY);
FLOAT_T euclideanDistanceOnPlanarMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y); 
FLOAT_T getWeight(FLOAT_T distance, FLOAT_T radius, FLOAT_T scaling);
int saveCodebook(string cbFileName, FLOAT_T *codebook, 
                unsigned int nSomX, unsigned int nSomY, unsigned int nDimensions);
FLOAT_T *calculateUMatrix(FLOAT_T *codebook, unsigned int nSomX,
             unsigned int nSomY, unsigned int nDimensions, string mapType);
int saveUMatrix(string fname, FLOAT_T *uMatrix, unsigned int nSomX, 
              unsigned int nSomY);
int saveBmus(string filename, int *bmus, unsigned int nSomX, 
             unsigned int nSomY, unsigned int nVectors);              
//void printMatrix(FLOAT_T *A, int nRows, int nCols);
FLOAT_T *readMatrix(const string inFilename, 
                  unsigned int &nRows, unsigned int &nCols);
void readSparseMatrixDimensions(const string filename, unsigned int &nRows, 
                            unsigned int &nColumns);
svm_node** readSparseMatrixChunk(const string filename, unsigned int nRows, 
                                 unsigned int nRowsToRead, 
                                 unsigned int rowOffset);
core_data trainOneEpoch(int itask, FLOAT_T *data, svm_node **sparseData,
           core_data coreData, unsigned int nEpoch, unsigned int currentEpoch,
           bool enableCalculatingUMatrix,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           unsigned int nVectorsPerRank,
           unsigned int radius0, unsigned int radiusN,
           string radiusCooling,
           FLOAT_T scale0, FLOAT_T scaleN,
           string scaleCooling,
           unsigned int kernelType, string mapType, int);                             
void train(int itask, FLOAT_T *data, svm_node **sparseData, 
           unsigned int nSomX, unsigned int nSomY, 
           unsigned int nDimensions, unsigned int nVectors, 
           unsigned int nVectorsPerRank, unsigned int nEpoch, 
           unsigned int radius0, unsigned int radiusN, 
           string radiusCooling,
           FLOAT_T scale0, FLOAT_T scaleN,
           string scaleCooling,
           string outPrefix, unsigned int snapshots, 
           unsigned int kernelType, string mapType,
           string initialCodebookFilename, int , int);
void trainOneEpochDenseCPU(int itask, FLOAT_T *data, FLOAT_T *numerator, 
                           FLOAT_T *denominator, FLOAT_T *codebook, 
                           unsigned int nSomX, unsigned int nSomY, 
                           unsigned int nDimensions, unsigned int nVectors,
                           unsigned int nVectorsPerRank, FLOAT_T radius, 
                           FLOAT_T scale, string mapType, int *globalBmus);
void trainOneEpochSparseCPU(int itask, svm_node **sparseData, FLOAT_T *numerator, 
                           FLOAT_T *denominator, FLOAT_T *codebook, 
                           unsigned int nSomX, unsigned int nSomY, 
                           unsigned int nDimensions, unsigned int nVectors,
                           unsigned int nVectorsPerRank, FLOAT_T radius, 
                           FLOAT_T scale, string mapType, int *globalBmus, int *global2ndBmus);

////
//from sparseCpuKernels
void get_bmu_coord(FLOAT_T* codebook, svm_node **sparseData,
                   unsigned int nSomY, unsigned int nSomX,
                   unsigned int nDimensions, int* coords, int* coords2, unsigned int n);


///



void initializeCodebook(unsigned int seed, FLOAT_T *codebook, unsigned int nSomX,
                        unsigned int nSomY, unsigned int nDimensions);


extern "C" {
#ifdef CUDA
void setDevice(int commRank, int commSize);
void freeGpu();
void initializeGpu(FLOAT_T *hostData, int nVectorsPerRank, int nDimensions, int nSomX, int nSomY);
void trainOneEpochDenseGPU(int itask, FLOAT_T *data, FLOAT_T *numerator, 
                           FLOAT_T *denominator, FLOAT_T *codebook, 
                           unsigned int nSomX, unsigned int nSomY, 
                           unsigned int nDimensions, unsigned int nVectors,
                           unsigned int nVectorsPerRank, FLOAT_T radius,
                           FLOAT_T scale, string mapType, int *globalBmus, int);
#endif                           
void my_abort(int err);
}
#endif

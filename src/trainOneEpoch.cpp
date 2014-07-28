#include"somoclu.h"
#include<cmath>
#include<stdlib.h>

FLOAT_T linearCooling(FLOAT_T start, FLOAT_T end, FLOAT_T nEpoch, FLOAT_T epoch) {
  FLOAT_T diff = (start - end) / (nEpoch-1);
  return start - (epoch * diff);
}

FLOAT_T exponentialCooling(FLOAT_T start, FLOAT_T end, FLOAT_T nEpoch, FLOAT_T epoch) {
  FLOAT_T diff = 0;
  if (end == 0.0)
  {
      diff = -log(0.1) / nEpoch;
  }
  else
  {
      diff = -log(end / start) / nEpoch;
  }
  return start * exp(-epoch * diff);
}


/** Initialize SOM codebook with random values
 * @param seed - random seed
 * @param codebook - the codebook to fill in
 * @param nSomX - dimensions of SOM map in the currentEpoch direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

void initializeCodebook(unsigned int seed, FLOAT_T *codebook, unsigned int nSomX,
                        unsigned int nSomY, unsigned int nDimensions)
{
    ///
    /// Fill initial random weights
    ///
    srand(seed);
    for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
            for (unsigned int d = 0; d < nDimensions; d++) {
                int w = 0xFFF & rand();
                //w -= 0x800; //non-zero values
                codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d] = (FLOAT_T)w / 4096.0f;
            }
        }
    }
}

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
           unsigned int kernelType, string mapType){

    FLOAT_T N = (FLOAT_T)nEpoch;
    FLOAT_T *numerator;
    FLOAT_T *denominator;
    FLOAT_T scale = scale0;
    FLOAT_T radius = radius0;
    if (itask == 0) {
        numerator = new FLOAT_T[nSomY*nSomX*nDimensions];
        denominator = new FLOAT_T[nSomY*nSomX];
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                denominator[som_y*nSomX + som_x] = 0.0;
                for (unsigned int d = 0; d < nDimensions; d++) {
                    numerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] = 0.0;
                }
            }
        }

        if (radiusCooling == "linear") {
          radius = linearCooling(FLOAT_T(radius0), radiusN, N, currentEpoch);
        } else {
          radius = exponentialCooling(radius0, radiusN, N, currentEpoch);
        }
        if (scaleCooling == "linear") {
          scale = linearCooling(scale0, scaleN, N, currentEpoch);
        } else {
          scale = exponentialCooling(scale0, scaleN, N, currentEpoch);
        }
//        cout << "Epoch: " << currentEpoch << " Radius: " << radius << endl;
    }
#ifdef HAVE_MPI
    MPI_Bcast(&radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coreData.codebook, nSomY*nSomX*nDimensions, MPI_FLOAT,
              0, MPI_COMM_WORLD);
#endif

    /// 1. Each task fills localNumerator and localDenominator
    /// 2. MPI_reduce sums up each tasks localNumerator and localDenominator to the root's
    ///    numerator and denominator.
    switch (kernelType) {
    default:
    case DENSE_CPU:
        trainOneEpochDenseCPU(itask, data, numerator, denominator,
                              coreData.codebook, nSomX, nSomY, nDimensions,
                              nVectors, nVectorsPerRank, radius, scale,
                              mapType, coreData.globalBmus);
        break;
#ifdef CUDA
    case DENSE_GPU:
        trainOneEpochDenseGPU(itask, data, numerator, denominator,
                              coreData.codebook, nSomX, nSomY, nDimensions,
                              nVectors, nVectorsPerRank, radius, scale,
                              mapType, coreData.globalBmus);
        break;
#endif
    case SPARSE_CPU:
        trainOneEpochSparseCPU(itask, sparseData, numerator, denominator,
                               coreData.codebook, nSomX, nSomY, nDimensions,
                               nVectors, nVectorsPerRank, radius, scale,
                               mapType, coreData.globalBmus,coreData.global2ndBmus);
        break;
    }

    /// 3. Update codebook using numerator and denominator
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (itask == 0) {
        #pragma omp parallel for
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                FLOAT_T denom = denominator[som_y*nSomX + som_x];
                for (unsigned int d = 0; d < nDimensions; d++) {
                    FLOAT_T newWeight = numerator[som_y*nSomX*nDimensions
                                                + som_x*nDimensions + d] / denom;
                    if (newWeight > 0.0) {
                        coreData.codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d] = newWeight;
                    }
                }
            }
        }
    }
    if (enableCalculatingUMatrix) {
        coreData.uMatrix = calculateUMatrix(coreData.codebook, nSomX, nSomY, nDimensions, mapType);
    }
    if (itask == 0) {
        delete [] numerator;
        delete [] denominator;
    }
    return coreData;
}

#ifndef SOMOCLUWRAP_H
#define SOMOCLUWRAP_H
#include"somoclu.h"
#include<string>


using namespace std;

void trainWrapper(FLOAT_T *data, int data_length,
                  unsigned int nEpoch,
                  unsigned int nSomX, unsigned int nSomY,
                  unsigned int nDimensions, unsigned int nVectors,
                  unsigned int radius0, unsigned int radiusN,
                  string radiusCooling,
                  FLOAT_T scale0, FLOAT_T scaleN,
                  string scaleCooling, unsigned int snapshots,
                  unsigned int kernelType, string mapType,
                  string initialCodebookFilename,
                  FLOAT_T* codebook, int codebook_size,
                  int* globalBmus, int globalBmus_size,
                  FLOAT_T* uMatrix, int uMatrix_size);

void trainWrapperR(FLOAT_T *data, int data_length,
                  unsigned int nEpoch,
                  unsigned int nSomX, unsigned int nSomY,
                  unsigned int nDimensions, unsigned int nVectors,
                  unsigned int radius0, unsigned int radiusN,
                  string radiusCooling,
                  FLOAT_T scale0, FLOAT_T scaleN,
                  string scaleCooling, unsigned int snapshots,
                  unsigned int kernelType, string mapType,
                  FLOAT_T *codebook, int codebook_size,
                  int *globalBmus, int globalBmus_size,
                  FLOAT_T *uMatrix, int uMatrix_size);

#endif // SOMOCLUWRAP_H

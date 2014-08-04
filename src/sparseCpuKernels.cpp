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

#include <cmath>
#include "somoclu.h"

#ifdef HAVE_R
#include <Rconfig.h>
#endif

#include <limits>

/** Distance b/w a feature vector and a weight vector
 * = Euclidean
 * @param som_y
 * @param som_x
 * @param r - row number in the input feature file
  */

FLOAT_T get_distance(FLOAT_T* codebook, svm_node **sparseData,
                   unsigned int som_y, unsigned int som_x, unsigned int nSomX,
                   unsigned int nDimensions, unsigned int r)
{
    FLOAT_T distance = 0.0f;
    unsigned int j=0;
    for ( unsigned int d=0; d < nDimensions; d++ ) {
        if ( (int) d == sparseData[r][j].index ) {
            distance += (codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d]-
                         sparseData[r][j].value) *
                        (codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d]-
                         sparseData[r][j].value);
            ++j;
        } else {
            distance += codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d]*
                        codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d];
        }
    }
    return distance;
}

/** Get node coords for the best matching unit (BMU)
 * @param coords - BMU coords
 * @param n - row num in the input feature file
 */
void get_bmu_coord(FLOAT_T* codebook, svm_node **sparseData,
                   unsigned int nSomY, unsigned int nSomX,
                   unsigned int nDimensions, int* coords, int* coords2, unsigned int n)
{
    FLOAT_T mindist = std::numeric_limits<FLOAT_T>::max();
    FLOAT_T mindist2 = std::numeric_limits<FLOAT_T>::max();
    FLOAT_T dist = 0.0f;

    /// Check nSomX * nSomY nodes one by one and compute the distance
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
            dist = get_distance(codebook, sparseData, som_y, som_x, nSomX,
                                nDimensions, n);
            if (dist < mindist) {
                //keep 2nd closest
                coords2[0] = coords[0];
                coords2[1] = coords[1];
                mindist2 = mindist;

                mindist = dist; 
                coords[0] = som_x;
                coords[1] = som_y;
            } else if (dist < mindist2) {
                mindist2 = dist;
                //keep 2nd closest
                coords2[0] = som_x;
                coords2[1] = som_y;
            }

        }
    }
}

void trainOneEpochSparseCPU(int itask, svm_node **sparseData, FLOAT_T *numerator,
                            FLOAT_T *denominator, FLOAT_T *codebook,
                            unsigned int nSomX, unsigned int nSomY,
                            unsigned int nDimensions, unsigned int nVectors,
                            unsigned int nVectorsPerRank, FLOAT_T radius,
                            FLOAT_T scale, string mapType, int *globalBmus, int *global2ndBmus)
{
    int p1[2] = {0, 0};
    int *bmus = new int[nVectorsPerRank*2];
    
    int p2[2] = {0, 0};
    int *bmus2nd = new int[nVectorsPerRank*2];

#ifdef _OPENMP
    #pragma omp parallel default(shared) private(p1,p2)
#endif
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for (unsigned int n = 0; n < nVectorsPerRank; n++) {
            if (itask*nVectorsPerRank+n<nVectors) {
                /// get the best matching unit
                get_bmu_coord(codebook, sparseData, nSomY, nSomX,
                              nDimensions, p1, p2, n);
                bmus[2*n] = p1[0]; bmus[2*n+1] = p1[1];
                bmus2nd[2*n] = p2[0]; bmus2nd[2*n+1] = p2[1];
            }
        }
    }

    FLOAT_T *localNumerator = new FLOAT_T[nSomY*nSomX*nDimensions];
    FLOAT_T *localDenominator = new FLOAT_T[nSomY*nSomX];
#ifdef _OPENMP
    #pragma omp parallel default(shared)
#endif
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                localDenominator[som_y*nSomX + som_x] = 0.0;
                for (unsigned int d = 0; d < nDimensions; d++)
                    localNumerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] = 0.0;
            }
        }

    /// Accumulate denoms and numers
#ifdef _OPENMP
        #pragma omp for
#endif
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                for (unsigned int n = 0; n < nVectorsPerRank; n++) {
                    if (itask*nVectorsPerRank+n<nVectors) {
                        FLOAT_T dist = 0.0f;
                        if (mapType == "planar") {
                            dist = euclideanDistanceOnPlanarMap(som_x, som_y, bmus[2*n], bmus[2*n+1]);
                        } else if (mapType == "toroid") {
                            dist = euclideanDistanceOnToroidMap(som_x, som_y, bmus[2*n], bmus[2*n+1], nSomX, nSomY);
                        }
                        FLOAT_T neighbor_fuct = getWeight(dist, radius, scale);
                        unsigned int j=0;
                        while ( sparseData[n][j].index!=-1 ) {
                            localNumerator[som_y*nSomX*nDimensions +
                                           som_x*nDimensions +
                                           sparseData[n][j].index] +=
                                               1.0f * neighbor_fuct * sparseData[n][j].value;
                            j++;
                        }
                        localDenominator[som_y*nSomX + som_x] += neighbor_fuct;
                    }
                }
            }
        }
    }
#ifdef HAVE_MPI
    MPI_Reduce(localNumerator, numerator,
               nSomY*nSomX*nDimensions, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localDenominator, denominator,
               nSomY*nSomX, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(bmus, nVectorsPerRank*2, MPI_INT, globalBmus, nVectorsPerRank*2, MPI_INT, 0, MPI_COMM_WORLD);
#else
    for (unsigned int i=0; i < nSomY*nSomX*nDimensions; ++i) {
        numerator[i] = localNumerator[i];
    }
    for (unsigned int i=0; i < nSomY*nSomX; ++i) {
        denominator[i] = localDenominator[i];
    }
    for (unsigned int i=0; i < 2*nVectorsPerRank; ++i) {
      globalBmus[i]=bmus[i];
    }
    for (unsigned int i=0; i < 2*nVectorsPerRank; ++i) {
      global2ndBmus[i]=bmus2nd[i];
    } 
 
#endif
    delete [] bmus;
    delete [] bmus2nd;
    delete [] localNumerator;
    delete [] localDenominator;
}

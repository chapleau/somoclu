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
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip> 

#include "somoclu.h"

using namespace std;

/** Initialize SOM codebook with random values
 * @param seed - random seed
 * @param codebook - the codebook to fill in
 * @param nSomX - dimensions of SOM map in the currentEpoch direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

//void initializeCodebook(unsigned int seed, FLOAT_T *codebook, unsigned int nSomX,
//                        unsigned int nSomY, unsigned int nDimensions)
//{
//    ///
//    /// Fill initial random weights
//    ///
//    srand(seed);
//    for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
//        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
//            for (unsigned int d = 0; d < nDimensions; d++) {
//                int w = 0xFFF & rand();
//                w -= 0x800;
//                codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d] = (FLOAT_T)w / 4096.0f;
//            }
//        }
//    }
//}


/** Main training loop
 * @param itask - number of work items
 * @param kv
 * @param ptr
 */

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
           string initialCodebookFilename, int seed, int gpu_thread_num)
{
    ///
    /// Codebook
    ///
    core_data coreData;
    coreData.codebook = new FLOAT_T[nSomY*nSomX*nDimensions];
    coreData.globalBmus = NULL;
    coreData.global2ndBmus = NULL;
    coreData.uMatrix = NULL;    
    if (itask == 0) {
        coreData.globalBmus = new int[nVectorsPerRank*int(ceil(nVectors/(double)nVectorsPerRank))*2];
        coreData.global2ndBmus = new int[nVectorsPerRank*int(ceil(nVectors/(double)nVectorsPerRank))*2];
        if (initialCodebookFilename.empty()){
            initializeCodebook(seed, coreData.codebook, nSomX, nSomY, nDimensions);
        } else {
            unsigned int nSomXY = 0;
            unsigned int tmpNDimensions = 0;
            delete [] coreData.codebook;
            coreData.codebook = readMatrix(initialCodebookFilename, nSomXY, tmpNDimensions);
            if (tmpNDimensions != nDimensions) {
                cerr << "Dimension of initial codebook does not match data!\n";
                my_abort(5);
            } else if (nSomXY / nSomY != nSomX) {
                cerr << "Dimension of initial codebook does not match specified SOM grid!\n";
                my_abort(6);
            }
            cout << "Read initial codebook: " << initialCodebookFilename << "\n";
        }
    }
    ///
    /// Parameters for SOM
    ///
    if (radius0 == 0) {
        unsigned int minDim = min(nSomX, nSomY);
        radius0 = minDim / 2.0f;              /// init radius for updating neighbors
    }
    if (radiusN == 0) {
        radiusN = 1;
    }
    if (scale0 == 0) {
      scale0 = 1.0;
    }
        
    unsigned int currentEpoch = 0;             /// 0...nEpoch-1
    
    ///
    /// Training
    ///
#ifdef HAVE_MPI    
    double training_time = MPI_Wtime();
#endif    

    while ( currentEpoch < nEpoch ) {

#ifdef HAVE_MPI      
        double epoch_time = MPI_Wtime();
#endif        

        coreData = trainOneEpoch(itask, data, sparseData,
                                 coreData, nEpoch, currentEpoch,
                                 snapshots > 0,
                                 nSomX, nSomY,
                                 nDimensions, nVectors,
                                 nVectorsPerRank,
                                 radius0, radiusN,
                                 radiusCooling,
                                 scale0, scaleN,
                                 scaleCooling,
                                 kernelType, mapType, gpu_thread_num);

        if (snapshots > 0 && itask == 0) {
            cout << "Saving interim U-Matrix..." << endl;
            stringstream sstm;
            sstm << outPrefix << "." << currentEpoch + 1;
            saveUMatrix(sstm.str() + string(".umx"), coreData.uMatrix, nSomX, nSomY);
            if (snapshots == 2){
                saveBmus(sstm.str() + string(".bm"), coreData.globalBmus, nSomX, nSomY, nVectors); 
                saveCodebook(sstm.str() + string(".wts"), coreData.codebook, nSomX, nSomY, nDimensions);                
            }
        }
        currentEpoch++;

       if ( (currentEpoch != nEpoch) && (currentEpoch % (nEpoch/100+1) != 0) ){}
       else{
              FLOAT_T ratio  =  currentEpoch/(FLOAT_T)nEpoch;
              int   c      =  ratio * 50 + 1;
              cerr << std::setw(7) << (int)(ratio*100) << "% [";
              for (int x=0; x<c; x++) cout << "=";
              for (int x=c; x<50; x++) cout << " ";
              cerr << "]\r"; 
       }  

#ifdef HAVE_MPI        
        if (itask == 0) {
            epoch_time = MPI_Wtime() - epoch_time;
            cerr << "Epoch Time: " << epoch_time << endl;
            if ( (currentEpoch != nEpoch) && (currentEpoch % (nEpoch/100+1) != 0) ){}
            else{
              FLOAT_T ratio  =  currentEpoch/(FLOAT_T)nEpoch;
              int   c      =  ratio * 50 + 1;
              cout << std::setw(7) << (int)(ratio*100) << "% [";
              for (int x=0; x<c; x++) cout << "=";
              for (int x=c; x<50; x++) cout << " ";
              cout << "]\n" << flush; 
            }  
        } 
#endif        
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    training_time = MPI_Wtime() - training_time;
    if (itask == 0) {
        cerr << "Total training Time: " << training_time << endl;
    }
#endif
    ///
    /// Save SOM map and u-mat
    ///
    if (itask == 0) {
        ///
        /// Save U-mat
        ///
        coreData.uMatrix = calculateUMatrix(coreData.codebook, nSomX, nSomY, nDimensions, mapType);
        int ret =  saveUMatrix(outPrefix + string(".umx"), coreData.uMatrix, nSomX, nSomY);        
        if (ret < 0)
            cout << "    Failed to save u-matrix. !" << endl;
        else {
            cout << "    Done!" << endl;
        }
        saveBmus(outPrefix + string(".bm"), coreData.globalBmus, nSomX, nSomY, nVectors); 
        if (kernelType == SPARSE_CPU) {
               saveBmus(outPrefix + string("_2nd.bm"), coreData.global2ndBmus, nSomX, nSomY, nVectors);

               //re-calculate BMUs
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
                   get_bmu_coord(coreData.codebook, sparseData, nSomY, nSomX,
                                nDimensions, p1, p2, n);
                   bmus[2*n] = p1[0]; bmus[2*n+1] = p1[1];
                   bmus2nd[2*n] = p2[0]; bmus2nd[2*n+1] = p2[1];
                  }
                 }
                }

                //save
                saveBmus(outPrefix + string("_after.bm"), bmus, nSomX, nSomY, nVectors);
                saveBmus(outPrefix + string("_2nd_after.bm"), bmus2nd, nSomX, nSomY, nVectors);
                delete [] bmus;
                delete [] bmus2nd;
        }
        ///
        /// Save codebook
        ///
        saveCodebook(outPrefix + string(".wts"), coreData.codebook, nSomX, nSomY, nDimensions);
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    delete [] coreData.codebook;
    delete [] coreData.globalBmus;
    delete [] coreData.global2ndBmus;
    delete [] coreData.uMatrix;
}

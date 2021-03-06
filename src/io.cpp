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

#include <cstdlib>
#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>

#include "somoclu.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace std;

/** Save a SOM codebook
 * @param cbFileName - name of the file to save
 * @param codebook - the codebook to save
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */
int saveCodebook(string cbFilename, FLOAT_T *codebook, unsigned int nSomX, unsigned int nSomY, unsigned int nDimensions)
{
    FILE* file = fopen(cbFilename.c_str(), "wt");
    cout << "    Saving Codebook " << cbFilename << endl;
    fprintf(file, "%%%d %d\n", nSomY, nSomX);
    fprintf(file, "%%%d\n", nDimensions);
    if (file!=0) {
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                for (unsigned int d = 0; d < nDimensions; d++) {
                    fprintf(file, "%0.10f ", codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d]);
                }
                fprintf(file, "\n");
            }
        }
        fclose(file);
        return 0;
    } else {
        return 1;
    }
}

/** Save best matching units
 * @param filename - name of the file to save
 * @param bmus - the best matching units to save
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nVectors - the number of vectors
 */
int saveBmus(string filename, int *bmus, unsigned int nSomX, unsigned int nSomY, unsigned int nVectors)
{
    FILE* file = fopen(filename.c_str(), "wt");
    cout << "    Saving best matching units " << filename << endl;
    fprintf(file, "%%%d %d\n", nSomY, nSomX);
    fprintf(file, "%%%d\n", nVectors);
    if (file!=0) {
        for (unsigned int i = 0; i < nVectors; ++i) {
            // ESOM Tools swaps x and y!
            fprintf(file, "%d %d %d\n", i, bmus[2*i+1], bmus[2*i]);
        }
        fclose(file);
        return 0;
    } else {
        return 1;
    }
}


/** Save u-matrix
 * @param fname
 * @param codebook - the codebook to save
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

int saveUMatrix(string fname, FLOAT_T *uMatrix, unsigned int nSomX,
             unsigned int nSomY)
{

    FILE* fp = fopen(fname.c_str(), "wt");
    fprintf(fp, "%%");
    fprintf(fp, "%d %d", nSomY, nSomX);
    fprintf(fp, "\n");
    if (fp != 0) {
        for (unsigned int som_y1 = 0; som_y1 < nSomY; som_y1++) {
            for (unsigned int som_x1 = 0; som_x1 < nSomX; som_x1++) {
                fprintf(fp, " %f", uMatrix[som_y1*nSomX+som_x1]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        return 0;
    }
    else
        return -2;
}

void process_line_mat_dim(const std::string& line, unsigned int &nRows, unsigned int &nColumns) {

  FLOAT_T tmp;
  if (line.substr(0,1) == "#") return;
  std::istringstream iss(line);
  if (nRows == 0) {
     while (iss >> tmp) nColumns++;
  }
  nRows++;

}

void getMatrixDimensions(string inFilename, unsigned int &nRows, unsigned int &nColumns) 
{
    ifstream file;
    bool gzipped = inFilename.compare(inFilename.size()-2, 2, "gz") == 0;
    if (gzipped) file.open(inFilename.c_str(), std::ios_base::in | std::ios_base::binary);
    else file.open(inFilename.c_str());

    if (file.is_open()) {
      string line;
      if (gzipped) {
        try {
          boost::iostreams::filtering_istream in;
          in.push(boost::iostreams::gzip_decompressor());
          in.push(file);
          while(getline(in,line)) process_line_mat_dim(line, nRows, nColumns); 
        }
        catch(const boost::iostreams::gzip_error& e) {
          std::cerr << e.what() << '\n';
          my_abort(-1);
        }
      }
      else {
        while(getline(file,line)) process_line_mat_dim(line, nRows, nColumns);
      }         
      file.close();
    } else {
        std::cerr << "Input file could not be opened!\n";
        my_abort(-1);
    }
}

unsigned int *readLrnHeader(string inFilename, unsigned int &nRows, unsigned int &nColumns)
{
    ifstream file;
    file.open(inFilename.c_str());
    string line;
    unsigned int currentColumn = 0;
    unsigned int nAllColumns = 0;
    unsigned int *columnMap = NULL;
    while(getline(file,line)) {
        if (line.substr(0,1) == "#") {
            continue;
        }
        if (line.substr(0,1) == "%") {
            std::istringstream iss(line.substr(1,line.length()));
            if (nRows == 0) {
                iss >> nRows;
            } else if (nAllColumns == 0) {
                iss >> nAllColumns;
            } else if (columnMap == NULL) {
                columnMap = new unsigned int[nAllColumns];
                unsigned int itmp = 0;
                currentColumn = 0;
                while (iss >> itmp) {
                    columnMap[currentColumn++] = itmp;
                    if (itmp == 1) {
                        ++nColumns;
                    }
                }
            } else {
                break;
            }
        }
    }
    file.close();
    return columnMap;
}

unsigned int *readWtsHeader(string inFilename, unsigned int &nRows, unsigned int &nColumns)
{
    ifstream file;
    file.open(inFilename.c_str());
    string line;
    while(getline(file,line)) {
        if (line.substr(0,1) == "#") {
            continue;
        }
        if (line.substr(0,1) == "%") {
            std::istringstream iss(line.substr(1,line.length()));
            if (nRows == 0) {
                iss >> nRows;
                unsigned int nSomY = 0;
                iss >> nSomY;
                nRows = nRows*nSomY;
            } else if (nColumns == 0) {
                iss >> nColumns;
            } else {
                break;
            }
        }
    }
    file.close();
    unsigned int *columnMap = new unsigned int[nColumns];
    for (unsigned int i = 0; i < nColumns; ++i) {
        columnMap[i] = 1;
    }    
    return columnMap;
}


void process_line_mat(const std::string& line, FLOAT_T *& data, unsigned int *& columnMap, unsigned int& j, unsigned int& currentColumn, unsigned int &nRows, unsigned int &nColumns) {

  FLOAT_T tmp;

  if ( (line.substr(0,1) == "#") | (line.substr(0,1) == "%") ) return;
        
  if (data == NULL) data = new FLOAT_T[nRows*nColumns];
        
  std::istringstream iss(line);
  currentColumn = 0;
  while (iss >> tmp) {
     if (columnMap[currentColumn++] != 1) continue;
     data[j++] = tmp;
  }

}

/** Reads a matrix
 * @param inFilename
 * @param nRows - returns the number of rows
 * @param nColumns - returns the number of columns
 * @return the matrix
 */
FLOAT_T *readMatrix(string inFilename, unsigned int &nRows, unsigned int &nColumns)
{
    FLOAT_T *data = NULL;
    unsigned int *columnMap = NULL;
    if (inFilename.compare(inFilename.size()-3, 3, "lrn") == 0) {
        columnMap = readLrnHeader(inFilename, nRows, nColumns);
    } else if (inFilename.compare(inFilename.size()-3, 3, "wts") == 0) {
        columnMap = readWtsHeader(inFilename, nRows, nColumns);
    } else {
        getMatrixDimensions(inFilename, nRows, nColumns);
        columnMap = new unsigned int[nColumns];
        for (unsigned int i = 0; i < nColumns; ++i) {
            columnMap[i] = 1;
        }
    }

    ifstream file;
    bool gzipped = inFilename.compare(inFilename.size()-2, 2, "gz") == 0;
    if (gzipped) file.open(inFilename.c_str(), std::ios_base::in | std::ios_base::binary);
    else file.open(inFilename.c_str());

    string line;
    unsigned int j = 0;
    unsigned int currentColumn = 0;

    if (gzipped) {
        try {
          boost::iostreams::filtering_istream in;
          in.push(boost::iostreams::gzip_decompressor());
          in.push(file);
          while(getline(in,line)) process_line_mat(line, data, columnMap, j, currentColumn, nRows, nColumns);
        }
        catch(const boost::iostreams::gzip_error& e) {
          std::cerr << e.what() << '\n';
          my_abort(-1);
        }
    }
    else {
      while(getline(file,line)) process_line_mat(line, data, columnMap, j, currentColumn, nRows, nColumns);
    }    
    
    file.close();
    delete [] columnMap;
    return data;
}

void readSparseMatrixDimensions(string filename, unsigned int &nRows,
                                unsigned int &nColumns) {
    ifstream file;
    file.open(filename.c_str());
    if (file.is_open()) {
        string line;
        int max_index=-1;
        while(getline(file,line)) {
            if (line.substr(0,1) == "#") {
                continue;
            }
            stringstream linestream(line);
            string value;
            int dummy_index;
            while(getline(linestream,value,' ')) {
                int separator=value.find(":");
                istringstream myStream(value.substr(0,separator));
                myStream >> dummy_index;
                if(dummy_index > max_index) {
                    max_index = dummy_index;
                }
            }
            ++nRows;
        }
        nColumns=max_index+1;
        file.close();
    } else {
        std::cerr << "Input file could not be opened!\n";
        my_abort(-1);
    }
}

svm_node** readSparseMatrixChunk(string filename, unsigned int nRows,
                                 unsigned int nRowsToRead,
                                 unsigned int rowOffset) {
    ifstream file;
    file.open(filename.c_str());
    string line;
    for (unsigned int i=0; i<rowOffset; i++) {
        getline(file, line);
    }
    if (rowOffset+nRowsToRead >= nRows) {
        nRowsToRead = nRows-rowOffset;
    }
    svm_node **x_matrix = new svm_node *[nRowsToRead];
    for(unsigned int i=0; i<nRowsToRead; i++) {
        getline(file, line);
        if (line.substr(0,1) == "#") {
            --i;
            continue;
        }
        stringstream tmplinestream(line);
        string value;
        int elements = 0;
        while(getline(tmplinestream,value,' ')) {
            elements++;
        }
        elements++; // To account for the closing dummy node in the row
        x_matrix[i] = new svm_node[elements];
        stringstream linestream(line);
        int j=0;
        while(getline(linestream,value,' ')) {
            int separator=value.find(":");
            istringstream myStream(value.substr(0,separator));
            myStream >> x_matrix[i][j].index;
            istringstream myStream2(value.substr(separator+1));
            myStream2 >> x_matrix[i][j].value;
            j++;
        }
        x_matrix[i][j].index = -1;
    }
    file.close();
    return x_matrix;
}

/** Shut down MPI cleanly if something goes wrong
 * @param err - error code to print
 */
void my_abort(int err)
{
    cerr << "Aborted\n";
#ifdef HAVE_MPI    
    MPI_Abort(MPI_COMM_WORLD, err);
#else
    exit(err);
#endif
}

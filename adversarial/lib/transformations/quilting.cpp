#include "quilting.h"
#include <cmath>
#include <limits>
#include <map>
#include <vector>
#include "findseam.h"

void generatePatches(
    float* result,
    float* img,
    unsigned int imgH,
    unsigned int imgW,
    unsigned int patchSize,
    unsigned int overlap) {
  int n = 0;

  for (int y = 0; y < imgH - patchSize; y += patchSize - overlap) {
    for (int x = 0; x < imgW - patchSize; x += patchSize - overlap) {
      for (int c = 0; c < 3; c++) {
        for (int j = 0; j < patchSize; j++) {
          for (int i = 0; i < patchSize; i++) {
            result
                [n * 3 * patchSize * patchSize + c * patchSize * patchSize +
                 j * patchSize + i] =
                    img[c * imgH * imgW + (y + j) * imgW + (x + i)];
          }
        }
      }
      n++;
    }
  }
}

// "from" nodes and "to" nodes
using graphLattice = std::pair<std::vector<int>, std::vector<int>>;

std::map<std::pair<unsigned int, unsigned int>, graphLattice> cache;

graphLattice _getFourLattice(unsigned int h, unsigned int w, bool useCache) {
  if (useCache) {
    auto iter = cache.find(std::make_pair(h, w));
    if (iter != cache.end()) {
      return iter->second;
    }
  }

  std::vector<int> from, to;

  // right
  for (int j = 0; j < h; j++) {
    for (int i = 0; i < w - 1; i++) {
      from.push_back(j * w + i);
      to.push_back(j * w + (i + 1));
    }
  }

  // left
  for (int j = 0; j < h; j++) {
    for (int i = 1; i < w; i++) {
      from.push_back(j * w + i);
      to.push_back(j * w + (i - 1));
    }
  }

  // down
  for (int j = 0; j < h - 1; j++) {
    for (int i = 0; i < w; i++) {
      from.push_back(j * w + i);
      to.push_back((j + 1) * w + i);
    }
  }

  // up
  for (int j = 1; j < h; j++) {
    for (int i = 0; i < w; i++) {
      from.push_back(j * w + i);
      to.push_back((j - 1) * w + i);
    }
  }

  graphLattice result = std::make_pair(from, to);

  if (useCache) {
    cache[std::make_pair(h, w)] = result;
  }

  return result;
}

void _findSeam(
    int* result,
    float* im1,
    float* im2,
    unsigned int patchSize,
    unsigned int* mask) {
  graphLattice graph = _getFourLattice(patchSize, patchSize, true);
  std::vector<int> from = graph.first;
  std::vector<int> to = graph.second;
  int edgeNum = 4 * patchSize * patchSize - 2 * (patchSize + patchSize);

  float* values = new float[edgeNum];
  for (int i = 0; i < edgeNum; i++) {
    values[i] = 0;
  }

  for (int c = 0; c < 3; c++) {
    for (int i = 0; i < edgeNum; i++) {
      values[i] += fabs(
          im2[c * patchSize * patchSize + to[i]] -
          im1[c * patchSize * patchSize + from[i]]);
    }
  }

  int nodeNum = patchSize * patchSize;
  float* tvalues = new float[nodeNum * 2];
  for (int i = 0; i < nodeNum * 2; i++) {
    tvalues[i] = 0;
  }

  for (int j = 0; j < patchSize; j++) {
    for (int i = 0; i < patchSize; i++) {
      for (int c = 0; c < 2; c++) {
        if (mask[j * patchSize + i] == c + 1) {
          tvalues[(j * patchSize + i) * 2 + c] =
              std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  findseam(nodeNum, edgeNum, from.data(), to.data(), values, tvalues, result);
  delete[] values;
  delete[] tvalues;
}

void stitch(
    float* result,
    float* im1,
    float* im2,
    unsigned int patchSize,
    unsigned int overlap,
    unsigned int y,
    unsigned int x) {
  unsigned int* mask = new unsigned int[patchSize * patchSize];

  for (int j = 0; j < patchSize; j++) {
    for (int i = 0; i < patchSize; i++) {
      mask[j * patchSize + i] = 2;
    }
  }

  if (y > 0) {
    for (int j = 0; j < overlap; j++) {
      for (int i = 0; i < patchSize; i++) {
        mask[j * patchSize + i] = 0;
      }
    }
  }

  if (x > 0) {
    for (int j = 0; j < patchSize; j++) {
      for (int i = 0; i < overlap; i++) {
        mask[j * patchSize + i] = 0;
      }
    }
  }

  int* seamMask = new int[patchSize * patchSize];
  _findSeam(seamMask, im1, im2, patchSize, mask);

  int offset;
  for (int c = 0; c < 3; c++) {
    for (int j = 0; j < patchSize; j++) {
      for (int i = 0; i < patchSize; i++) {
        offset = c * patchSize * patchSize + j * patchSize + i;
        result[offset] =
            (seamMask[j * patchSize + i] == 1) ? im2[offset] : im1[offset];
      }
    }
  }
  delete [] mask;
  delete [] seamMask;
}

void generateQuiltedImages(
    float* result,
    long* neighbors,
    float* patchDict,
    unsigned int imgH,
    unsigned int imgW,
    unsigned int patchSize,
    unsigned int overlap,
    bool graphcut) {
  int n = 0;
  for (int y = 0; y < imgH - patchSize; y += patchSize - overlap) {
    for (int x = 0; x < imgW - patchSize; x += patchSize - overlap) {
      if (neighbors[n] != -1) {
        if (graphcut) {
          float* patch = new float[3 * patchSize * patchSize];
          for (int c = 0; c < 3; c++) {
            for (int j = 0; j < patchSize; j++) {
              for (int i = 0; i < patchSize; i++) {
                patch[c * patchSize * patchSize + j * patchSize + i] =
                    result[c * imgH * imgW + (y + j) * imgW + (x + i)];
              }
            }
          }

          float* stitched = new float[3 * patchSize * patchSize];
          float* matched =
              patchDict + (neighbors[n] * 3 * patchSize * patchSize);
          stitch(stitched, patch, matched, patchSize, overlap, y, x);
          for (int c = 0; c < 3; c++) {
            for (int j = 0; j < patchSize; j++) {
              for (int i = 0; i < patchSize; i++) {
                result[c * imgH * imgW + (y + j) * imgW + (x + i)] =
                    stitched[c * patchSize * patchSize + j * patchSize + i];
              }
            }
          }
          delete[] patch;
          delete[] stitched;
        } else {
          for (int c = 0; c < 3; c++) {
            for (int j = 0; j < patchSize; j++) {
              for (int i = 0; i < patchSize; i++) {
                result[c * imgH * imgW + (y + j) * imgW + (x + i)] = patchDict
                    [neighbors[n] * 3 * patchSize * patchSize +
                     c * patchSize * patchSize + j * patchSize + i];
              }
            }
          }
        }
      }

      n++;
    }
  }
}

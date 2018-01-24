#pragma once

#ifdef __cplusplus
extern "C" {
#endif

double findseam(
    int numnodes, // number of nodes
    int numedges, // number of edges
    int* from, // from indices
    int* to, // to indices
    float* values, // values on edges
    float* tvalues, // values for terminal edges
    int* labels // memory in which to write the labels
    );

#ifdef __cplusplus
}
#endif

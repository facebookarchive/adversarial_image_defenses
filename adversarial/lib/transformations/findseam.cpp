#include "findseam.h"
#include "graph.h"

double findseam(
    int numnodes, // number of nodes
    int numedges, // number of edges
    int* from, // from indices
    int* to, // to indices
    float* values, // values on edges
    float* tvalues, // values for terminal edges
    int* labels // memory in which to write the labels
    ) {
  // initialize graph:
  Graph<float, float, float>* g =
      new Graph<float, float, float>(numnodes, numedges);
  g->add_node(numnodes);

  // add edges:
  for (unsigned int i = 0; i < numedges; i++) {
    g->add_edge(from[i], to[i], values[i], 0.0f);
  }

  // add terminal nodes:
  for (unsigned int i = 0; i < numnodes; i++) {
    g->add_tweights(i, tvalues[i * 2], tvalues[i * 2 + 1]);
  }

  // run maxflow algorithm:
  double flow = g->maxflow();
  for (unsigned int i = 0; i < numnodes; i++) {
    labels[i] = g->what_segment(i);
  }

  // return results:
  delete g;
  return flow;
}

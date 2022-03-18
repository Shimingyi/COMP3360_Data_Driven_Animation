#include "trimesh.h"
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

using namespace std;

struct vertex_idx_wrapper_t {
  vertex_idx_wrapper_t() : vert(-1) {}

  int vert;
};

struct vert_less {
  bool operator()(const vertex_idx_wrapper_t &lhs,
                  const vertex_idx_wrapper_t &rhs) const {
    // handle any size mesh
    if (lhs.vert != rhs.vert)
      return (lhs.vert < rhs.vert);
    return false;
  }
};

trimesh_t::trimesh_t() {}

void trimesh_t::load(const char *filename) {
  ifstream inf;
  inf.open(filename, ios_base::in);
  if (!inf.is_open()) {
    cerr << "[!] Failed to load file: " << filename << endl;
  }

  vertices.clear();
  triangles.clear();

  const char *delims = " \n\r";
  const unsigned int CHARACTER_COUNT = 500;
  char line[CHARACTER_COUNT] = {0};

  std::vector<glm::vec3> verts;

  std::map<vertex_idx_wrapper_t, int, vert_less> uniqueverts;
  unsigned int vert_count = 0;

  while (inf.good()) {
    memset((void *)line, 0, CHARACTER_COUNT);
    inf.getline(line, CHARACTER_COUNT);
    if (inf.eof())
      break;

    char *token = strtok(line, delims);
    if (token == NULL || token[0] == '#' || token[0] == '$')
      continue;

    // verts look like:
    //	v float float float
    if (strcmp(token, "v") == 0) {
      float x = 0, y = 0, z = 0, w = 1;
      sscanf(line + 2, "%f %f %f %f", &x, &y, &z, &w);
      verts.push_back(glm::vec3(x / w, y / w, z / w));
    }
    // keep track of smoothing groups
    // s [number|off]
    else if (strcmp(token, "s") == 0) {
    }

    // faces start with:
    //	f
    else if (strcmp(token, "f") == 0) {

      std::vector<int> vindices;
      std::vector<int> nindices;
      std::vector<int> tindices;

      // fill out a triangle from the line, it could have 3 or 4 edges
      char *lineptr = line + 2;
      while (lineptr[0] != 0) {
        while (lineptr[0] == ' ')
          ++lineptr;

        int vi = 0, ni = 0, ti = 0;
        if (sscanf(lineptr, "%d/%d/%d", &vi, &ni, &ti) == 3) {
          vindices.push_back(vi - 1);
        } else if (sscanf(lineptr, "%d//%d", &vi, &ni) == 2) {
          vindices.push_back(vi - 1);
        } else if (sscanf(lineptr, "%d/%d", &vi, &ti) == 2) {
          vindices.push_back(vi - 1);
        } else if (sscanf(lineptr, "%d", &vi) == 1) {
          vindices.push_back(vi - 1);
        }

        while (lineptr[0] != ' ' && lineptr[0] != 0)
          ++lineptr;
      }

      // being that some exporters can export either 3 or 4 sided polygon's
      // convert what ever was exported into triangles
      for (size_t i = 1; i < vindices.size() - 1; ++i) {
        // Face face;
        vertex_idx_wrapper_t tri;

        tri.vert = vindices[0];

        if (uniqueverts.count(tri) == 0)
          uniqueverts[tri] = vert_count++;
        // face.a = uniqueverts[tri];
        triangles.push_back(uniqueverts[tri]);
        tri.vert = vindices[i];

        if (uniqueverts.count(tri) == 0)
          uniqueverts[tri] = vert_count++;
        // face.b = uniqueverts[tri];
        triangles.push_back(uniqueverts[tri]);
        tri.vert = vindices[i + 1];

        if (uniqueverts.count(tri) == 0)
          uniqueverts[tri] = vert_count++;
        // face.c = uniqueverts[tri];
        triangles.push_back(uniqueverts[tri]);
      }
    }
  }
  inf.close();

  // use resize instead of reserve because we'll be indexing in random
  // locations.
  vertices.resize(vert_count);

  std::map<vertex_idx_wrapper_t, int, vert_less>::iterator iter;
  for (iter = uniqueverts.begin(); iter != uniqueverts.end(); ++iter) {
    vertices[iter->second] = verts[iter->first.vert];
  }
}

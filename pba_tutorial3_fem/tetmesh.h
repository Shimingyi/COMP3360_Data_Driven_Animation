#ifndef __TETMESH__
#define __TETMESH__

#include <fstream> // std::ifstream
#include <algorithm> // std::sort
#include <string> // std::string 
#include <vector> // std::vector
#include <assert.h> // assert

#include <glm/glm.hpp>

struct tetmesh_t {

    tetmesh_t()
    {
        
    }

    ~tetmesh_t()
    {
    }

    void load(std::string fpath_basename /*e.g. "foo/bar/name_without_ext" */)
    {
        this->load_vertex_data(fpath_basename + ".node");
        this->load_tetrahedron_data(fpath_basename + ".ele");
        this->load_boundary_face_data(fpath_basename + ".face");
    }

    std::vector<glm::vec3> vertices;
    std::vector<int> tetrahedra;
    std::vector<int> boundaryFaces;

    void load_vertex_data(const std::string& fpath)
    {
        printf("load: %s\n", fpath.c_str());
        std::ifstream in(fpath.c_str());

        // Read the first line containing structural information
        int vertexCount;
        in >> vertexCount;

        int dimensions;
        in >> dimensions;

        int attributes;
        in >> attributes;

        int boundaryMarkers;
        in >> boundaryMarkers;

        // If we're getting something else than 3 dimensions, fail
        if (dimensions != 3) {
            throw 1;
        }

        this->vertices.resize(vertexCount);

        int pointIndex;
        float x, y, z;
        bool zeroIndexed = false;

        for (int i = 0; i < vertexCount; i++) {
            in >> pointIndex;

            in >> x;
            in >> y;
            in >> z;

            if (pointIndex == 0) {
                zeroIndexed = true;
            }

            size_t index = pointIndex - (zeroIndexed ? 0 : 1);
            this->vertices[index] = glm::vec3(x, y, z);
        }
    }

    struct tetrahedron_t {
        int v1, v2, v3, v4;
    };

    static bool cmpTetrahedron(const tetrahedron_t& a, const tetrahedron_t& b)
    {
        return (a.v1 + a.v2 + a.v3 + a.v4) < (b.v1 + b.v2 + b.v3 + b.v4);
    }

    void load_tetrahedron_data(const std::string& fpath)
    {
        printf("load: %s\n", fpath.c_str());
        std::ifstream in(fpath.c_str());
        
        int tetrahedraCount;
        in >> tetrahedraCount;

        int nodesPerTet;
        in >> nodesPerTet;

        if (nodesPerTet != 4) {
            throw 1;
        }

        int regionAttribute;
        in >> regionAttribute;

        bool zeroIndexed = false;

        std::vector<tetrahedron_t> tets;

        int tetrahedraIndex, region, v1, v2, v3, v4;
        for (int i = 0; i < tetrahedraCount; i++) {
            in >> tetrahedraIndex;

            in >> v1;
            in >> v2;
            in >> v3;
            in >> v4;

            if (v1 == 0 || v2 == 0 || v3 == 0 || v4 == 0) {
                zeroIndexed = true;
            }
            
            tetrahedron_t tet;
            tet.v1 = v4;
            tet.v2 = v2;
            tet.v3 = v3;
            tet.v4 = v1;
            tets.push_back(tet);
            /*this->tetrahedra.push_back(v1);
        this->tetrahedra.push_back(v2);
        this->tetrahedra.push_back(v3);
        this->tetrahedra.push_back(v4);*/

            if (regionAttribute) {
                in >> region;
            }
        }

        assert(tetrahedraCount == (int)tets.size());

        std::sort(tets.begin(), tets.end(), cmpTetrahedron);

        for (int i =0; i < tetrahedraCount; ++i) {
            this->tetrahedra.push_back(tets[i].v1);
            this->tetrahedra.push_back(tets[i].v2);
            this->tetrahedra.push_back(tets[i].v3);
            this->tetrahedra.push_back(tets[i].v4);
        }

        if (!zeroIndexed) {
            for (int i = 0; i < tetrahedraCount * 4; i++) {
                this->tetrahedra[i] -= 1;
            }
        }
    }

    void load_boundary_face_data(std::string fpath)
    {
        printf("load: %s\n", fpath.c_str());
        std::ifstream in(fpath.c_str());

        int faceCount;
        in >> faceCount;

        bool zeroIndexed = false;

        int boundaryMarker;
        in >> boundaryMarker;

        int faceIndex, boundary, v1, v2, v3;
        for (int i = 0; i < faceCount; i++) {
            in >> faceIndex;

            in >> v1;
            in >> v2;
            in >> v3;

            this->boundaryFaces.push_back(v1);
            this->boundaryFaces.push_back(v2);
            this->boundaryFaces.push_back(v3);

            if (v1 == 0 || v2 == 0 || v3 == 0) {
                zeroIndexed = true;
            }

            if (boundaryMarker) {
                in >> boundary;
            }
        }

        if (!zeroIndexed) {
            for (int i = 0; i < faceCount * 3; i++) {
                this->boundaryFaces[i] -= 1;
            }
        }
    }
};

#endif
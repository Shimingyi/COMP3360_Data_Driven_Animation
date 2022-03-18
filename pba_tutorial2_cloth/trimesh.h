#ifndef _TRIMESH_H
#define _TRIMESH_H

#include <glm/glm.hpp>
#include <vector>

class trimesh_t {
public:
  trimesh_t();

  void load(const char *filename);

  int get_index_count() const { return (int)triangles.size(); }

  int get_vertex_count() const { return (int)vertices.size(); }

  void set_triangles(const std::vector<unsigned int> &f) { triangles = f; }

  void set_vertices(const std::vector<glm::vec3> &v) { vertices = v; }

  const std::vector<unsigned int> &get_indices() const { return triangles; }

  const std::vector<glm::vec3> &get_vertices() const { return vertices; }

  inline const glm::vec3 &get_vertex(int i) const { return vertices[i]; }

  // volume of triangle mesh + its volume center of mass
  float get_volume() const {
    float vol = 0;
    int num_triangles = get_index_count() / 3;

    for (int i = 0; i < num_triangles; ++i) {
      int v0 = triangles[(i * 3) + 0];
      int v1 = triangles[(i * 3) + 1];
      int v2 = triangles[(i * 3) + 2];

      const glm::vec3 &a = vertices[v0];
      const glm::vec3 &b = vertices[v1];
      const glm::vec3 &c = vertices[v2];

      float v = glm::dot(a, glm::cross(b, c)) / 6.0;
      vol += v;
    }

    return vol;
  }

private:
  std::vector<unsigned int> triangles;
  std::vector<glm::vec3> vertices;
};
#endif // _TRIMESH_H

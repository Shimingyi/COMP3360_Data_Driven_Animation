
#include "gl_utils.h"

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <set>

#include "camera.h"
#include "trimesh.h"

camera_t camera(glm::vec3(0.0f, 2.0f, 5.0f));
float lastX = g_gl_width / 2.0f;
float lastY = g_gl_height / 2.0f;
bool firstMouse = true;
float dt = 0.0f; // time between current frame and last frame
float scene_scale = 16;
trimesh_t object_mesh;
GLuint object_vao;
GLuint object_vbo;
GLuint object_ibo;
trimesh_t plane_mesh;
GLuint plane_vao;
GLuint plane_vbo;
GLuint plane_ibo;
std::vector<glm::vec3> grid_vertices;
std::vector<GLuint> grid_indices;
GLuint grid_vao;
GLuint grid_vbo;
GLuint grid_ibo;
GLuint shader_programme;
int model_mat_location;
int view_mat_location;
int proj_mat_location;
int colour_location;

struct ordered_pair : std::pair<int, int> {
  ordered_pair(int a, int b)
      : std::pair<int, int>(a < b ? a : b, a < b ? b : a) {}
};

struct node_t {
  // state
  glm::dvec3 x; // position
  glm::dvec3 v; // velocity
  // constants
  double m; // mass
  // auxilliary quantity
  glm::dvec3 force;
};

struct spring_t {
  int p0;
  int p1;
  double ks; // spring stiffness
  double kd; // damping coefficient
  double r;  // rest length
  // auxilliary quantity
  glm::dvec3 force;
};

std::vector<spring_t> springs;
std::vector<node_t> nodes;
std::vector<int> constrained_nodes;

void init_physical_object(std::vector<node_t> &nodes_,
                          std::vector<spring_t> &springs_,
                          const trimesh_t &mesh,
                          const glm::vec3 &initial_position,
                          const glm::mat4 &initial_orientation) {

  // init nodes
  // ==========
  nodes_.resize(mesh.get_vertex_count());

  const glm::mat4 T = glm::translate(glm::mat4(1.0), initial_position);
  const glm::mat4 &R = initial_orientation;

  for (int i = 0; i < mesh.get_vertex_count(); ++i) {
    node_t &n = nodes_.at(i);

    const glm::vec3 &vertex = mesh.get_vertex(i);
    n.x = glm::dvec3(T * R * glm::dvec4(vertex, 1.0));
    // TODO: init the rest of node's data
  }

  printf("nodes=%d\n", (int)nodes_.size());
  

  // init springs
  // ============

  const unsigned int *triptr = &mesh.get_indices()[0];
  std::set<ordered_pair> edges;

  for (int i = 0; i < mesh.get_index_count() / 3; ++i) { // for each triangle
    for (int j = 0; j < 3; ++j) {
      const unsigned int i0 = triptr[i * 3 + j + 0];
      const unsigned int i1 = triptr[i * 3 + (j + 1) % 3];
      edges.insert(ordered_pair(i0, i1));
    }
  }

  springs_.resize(edges.size());

  for (std::set<ordered_pair>::const_iterator it = edges.cbegin();
       it != edges.cend(); ++it) {
    const int idx = std::distance(edges.cbegin(), it);
    spring_t &s = springs_[idx];
    // TODO: init spring
  }

  printf("springs=%d\n", (int)springs_.size());
}

void update_physics(float dt) {
  // TODO: implement this
}

void draw_scene();
void destroy();
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void process_input(GLFWwindow *window);

int main() {
  restart_gl_log();

  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // glfw window creation
  // --------------------
  g_window =
      glfwCreateWindow(g_gl_width, g_gl_height, PROJECT_NAME_STR, NULL, NULL);

  if (g_window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(g_window);
  glfwSetFramebufferSizeCallback(g_window, framebuffer_size_callback);
  glfwSetCursorPosCallback(g_window, mouse_callback);
  glfwSetScrollCallback(g_window, scroll_callback);

  // NOTE: comment this out if you wish to have control over your mouse
  // other GLFW will capture our mouse
  glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // get version info
  const GLubyte *renderer = glGetString(GL_RENDERER); // get renderer string
  const GLubyte *version = glGetString(GL_VERSION);   // version as a string

  printf("Renderer: %s\n", renderer);
  printf("OpenGL version supported %s\n", version);
  gl_log("renderer: %s\nversion: %s\n", renderer, version);

  // configure global opengl state
  // -----------------------------
  glEnable(GL_DEPTH_TEST); // enable depth-testing
  glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"
  // glEnable(GL_CULL_FACE); // cull face
  // glCullFace(GL_BACK);    // cull back face
  glFrontFace(GL_CCW); // set counter-clock-wise vertex order to mean the front
  glClearColor(0.0, 0.0, 0.0, 1.0); // grey background to help spot mistakes
  glViewport(0, 0, g_gl_width, g_gl_height);

  // load and setup shaders
  // ----------------------
  shader_programme = create_programme_from_files(
      PROJECT_ROOT_DIR "/test_vs.glsl", PROJECT_ROOT_DIR "/test_fs.glsl");
  model_mat_location = glGetUniformLocation(shader_programme, "model");
  view_mat_location = glGetUniformLocation(shader_programme, "view");
  proj_mat_location = glGetUniformLocation(shader_programme, "proj");
  colour_location = glGetUniformLocation(shader_programme, "colour");

  // setup ground plane mesh
  // -----------------------
  float s = scene_scale / 2; // scale of environment
  plane_mesh.set_vertices({
      {-s, 0, s}, // 0
      {s, 0, s},  // 1
      {s, 0, -s}, // 2
      {-s, 0, -s} // 3
  });
  plane_mesh.set_triangles({0, 1, 3, 1, 2, 3});

  glGenVertexArrays(1, &plane_vao);
  glBindVertexArray(plane_vao);
  {
    glGenBuffers(1, &plane_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, plane_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 3 * plane_mesh.get_vertex_count() * sizeof(GLfloat),
                 &plane_mesh.get_vertices()[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &plane_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 plane_mesh.get_index_count() * sizeof(GLuint),
                 &plane_mesh.get_indices()[0], GL_STATIC_DRAW);
  }
  glBindVertexArray(0);

  // setup grid lines along ground mesh
  // ----------------------------------

  int slices = 16;
  for (int j = 0; j <= slices; ++j) {
    for (int i = 0; i <= slices; ++i) {
      float x = (float)i / (float)slices;
      float y = 0;
      float z = (float)j / (float)slices;
      grid_vertices.push_back(glm::vec3((x - 0.5) * 2, y, (z - 0.5) * 2));
    }
  }

  for (int j = 0; j < slices; ++j) {
    for (int i = 0; i < slices; ++i) {

      GLuint row1 = j * (slices + 1);
      GLuint row2 = (j + 1) * (slices + 1);

      grid_indices.insert(grid_indices.end(),
                          {row1 + i, row1 + i + 1, row1 + i + 1, row2 + i + 1});
      grid_indices.insert(grid_indices.end(),
                          {row2 + i + 1, row2 + i, row2 + i, row1 + i});
    }
  }

  {
    glGenVertexArrays(1, &grid_vao);
    glBindVertexArray(grid_vao);

    glGenBuffers(1, &grid_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, grid_vbo);
    glBufferData(GL_ARRAY_BUFFER, grid_vertices.size() * sizeof(float) * 3,
                 &grid_vertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glGenBuffers(1, &grid_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, grid_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, grid_indices.size() * sizeof(GLuint),
                 &grid_indices.data()[0], GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
  }

  // load and setup mesh used for simulation
  // ---------------------------------------
  bool high_res = false;
  
  if(high_res) {
    object_mesh.load(PROJECT_ROOT_DIR "/grid20x20.obj");
    constrained_nodes = {380, 399}; // top left and top right
  }
  else{
    object_mesh.load(PROJECT_ROOT_DIR "/grid10x10.obj");
    constrained_nodes = {99, 91}; // top left and top right
  }
  glGenVertexArrays(1, &object_vao);
  glBindVertexArray(object_vao);
  {
    glGenBuffers(1, &object_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, object_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 3 * object_mesh.get_vertex_count() * sizeof(GLfloat),
                 &object_mesh.get_vertices()[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &object_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 object_mesh.get_index_count() * sizeof(GLuint),
                 &object_mesh.get_indices()[0], GL_STATIC_DRAW);
  }
  glBindVertexArray(0);

  const glm::vec3 initial_position(0.f, 4.f, 0.f);
  const glm::mat4 initial_orientation =
      glm::rotate(glm::mat4(1.f), glm::radians(00.f), glm::vec3(1.f, 0.f, 0.f));

  init_physical_object(nodes, springs, object_mesh, initial_position,
                       initial_orientation);

  // timing

  float lastFrame = 0.0f;
  /*-------------------------------MAIN LOOP-------------------------------*/
  while (!glfwWindowShouldClose(g_window)) {
    // per-frame time logic
    // --------------------
    float currentFrame = static_cast<float>(glfwGetTime());
    dt = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // input
    // -----
    process_input(g_window);

    // if (frame_counter < 1)
    update_physics(/*1.0/60.0*/ dt);

    // render
    // ------
    _update_fps_counter(g_window);

    draw_scene();

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved
    // etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(g_window);
    glfwPollEvents();
    frame_counter += 1;
  }

  destroy();

  return 0;
}

void destroy() {
  glDeleteBuffers(1, &object_vbo);
  glDeleteBuffers(1, &object_ibo);
  glDeleteVertexArrays(1, &object_vao);

  glDeleteBuffers(1, &plane_vbo);
  glDeleteBuffers(1, &plane_ibo);
  glDeleteVertexArrays(1, &plane_vao);

  glDeleteBuffers(1, &grid_vbo);
  glDeleteBuffers(1, &grid_ibo);
  glDeleteVertexArrays(1, &grid_vao);

  glDeleteProgram(shader_programme);

  // close GL context and any other GLFW resources
  glfwTerminate();
}

void draw_scene() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(shader_programme);

  const float near = 0.1f;  // clipping plane
  const float far = 100.0f; // clipping plane
  // float fovy = 67.0f;                                    // 67 degrees
  const float aspect = (float)g_gl_width / (float)g_gl_height; // aspect ratio
  const glm::mat4 projection =
      glm::perspective(glm::radians(camera.m_zoom), aspect, near, far);
  glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE,
                     glm::value_ptr(projection));

  const glm::mat4 view = camera.get_view_matrix();
  glUniformMatrix4fv(view_mat_location, 1, GL_FALSE, glm::value_ptr(view));

  // draw ground plane
  {
    glBindVertexArray(plane_vao);
    const glm::mat4 model_mat = glm::mat4(1.0); // at origin
    glUniformMatrix4fv(model_mat_location, 1, GL_FALSE,
                       glm::value_ptr(model_mat));

    const glm::vec4 colour(0.3f, 0.3f, 0.3f, 0.3f);
    glUniform4fv(colour_location, 1, glm::value_ptr(colour));
    glDrawElements(GL_TRIANGLES, plane_mesh.get_index_count(), GL_UNSIGNED_INT,
                   0);
    glBindVertexArray(0);
  }

  // draw grid
  {
    glBindVertexArray(grid_vao);
    const float s = scene_scale / 2;
    glm::mat4 model_mat = glm::scale(
        glm::mat4(1.0), glm::vec3(s, s, s)); // at origin and scale to match env
    model_mat = translate(model_mat, glm::vec3(0, 1e-3, 0));
    glUniformMatrix4fv(model_mat_location, 1, GL_FALSE,
                       glm::value_ptr(model_mat));

    const glm::vec4 colour(0.0f, 0.0f, 0.0f, 0.3f);
    glUniform4fv(colour_location, 1, glm::value_ptr(colour));
    glDrawElements(GL_LINES, grid_indices.size(), GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
  }

  // draw sim mesh objects
  {
    glBindVertexArray(object_vao);

    // update vertices buffer
    // ======================
    { // copy update cloth vertex positions from CPU RAM to the GPU
      glBindBuffer(GL_ARRAY_BUFFER, object_vbo);

      void *ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
      static std::vector<float> tmp(object_mesh.get_vertex_count() * 3);
      for (int i = 0; i < nodes.size(); ++i) {
        const node_t &n = nodes[i];
        for (int j = 0; j < 3; j++) {
          tmp[i * 3 + j] = n.x[j];
        }
      }

      memcpy(ptr, &tmp[0], nodes.size() * 3 * sizeof(float));

      glUnmapBuffer(GL_ARRAY_BUFFER);
    }

    const glm::mat4 rotation(1.0);
    const glm::mat4 translation(1.0);
    const glm::mat4 model_mat = translation * rotation;
    glUniformMatrix4fv(model_mat_location, 1, GL_FALSE,
                       glm::value_ptr(model_mat));
    glm::vec4 colour(1.0f, 0.0f, 0.0f, 0.0f);
    glUniform4fv(colour_location, 1, glm::value_ptr(colour));
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, object_mesh.get_index_count(), GL_UNSIGNED_INT,
                   0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    colour = glm::vec4(0.1f, 0.1f, 0.1f, 0.1f);
    glUniform4fv(colour_location, 1, glm::value_ptr(colour));
    glDrawElements(GL_TRIANGLES, object_mesh.get_index_count(), GL_UNSIGNED_INT,
                   0);

    glBindVertexArray(0);
  }
}

// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(g_window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.process_keyboard_input(FORWARD, dt);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.process_keyboard_input(BACKWARD, dt);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.process_keyboard_input(LEFT, dt);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.process_keyboard_input(RIGHT, dt);
}

// glfw: whenever the g_window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn) {
  float xpos = static_cast<float>(xposIn);
  float ypos = static_cast<float>(yposIn);

  if (firstMouse) {
    lastX = xpos;
    lastY = ypos;
    firstMouse = false;
  }

  float xoffset = xpos - lastX;
  float yoffset =
      lastY - ypos; // reversed since y-coordinates go from bottom to top

  lastX = xpos;
  lastY = ypos;

  camera.process_mouse_motion(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  camera.process_mouse_scroll(static_cast<float>(yoffset));
}

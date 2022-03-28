
#include "gl_utils.h"

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <map>
#include <set>

#include "camera.h"
#include "tetmesh.h"
#include "trimesh.h"

#include "svd3.h"

const float flp_epsilon = 1e-4f;

camera_t camera(glm::vec3(0.0f, 2.0f, 5.0f));
float lastX = g_gl_width / 2.0f;
float lastY = g_gl_height / 2.0f;
bool firstMouse = true;
float dt = 0.0f; // time between current frame and last frame
float scene_scale = 16;
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

std::map<int, int> tmesh_to_smesh_vmap;
std::vector<int> fixed_nodes;
std::vector<int> pulled_and_pushed_nodes;

/*
    Set this to true or false to apply pull or pull boundary conditions (~using constant velocity)

    true --> pull the cylinder
    false --> compress cylinder (by pushing)
*/
bool apply_pull_force = true;

enum MaterialModel
{
    LINEAR = 0,
    ST_VK,
    COROTATIONAL
};

struct finite_element_t
{
    // ... TODO
};

struct node_t
{
    glm::vec3 pos; // current position
    // TODO ...
    glm::vec3 force;
    float mass;
};

struct material_t
{
    float lambda;
    float mu;
};

struct simulation_mesh_t
{
    tetmesh_t tetmesh; // tetrahedron mesh (as loaded from file).
    material_t material_parameters;
    std::vector<node_t> nodes;
    std::vector<finite_element_t> finite_elements;
} sim_mesh;

// mini helper function to compute PD
void polar_decomposition(const glm::mat3 &m, glm::mat3 &R, glm::mat3 &S);

float trace(const glm::mat3 &m);

void normalize_mesh(tetmesh_t &tetmesh, const glm::vec3 &translation, const glm::vec3 &scale);

void init_physical_object(simulation_mesh_t &sm)
{
    using namespace glm;

    //
    // set material parameters
    //
    sm.material_parameters.lambda = 30000.45f;
    sm.material_parameters.mu = 10000.f;

    //
    // Create the simulation nodes of our finite e mesh
    //
    const int node_count = sm.tetmesh.vertices.size();

    float min_x = 1e10;
    float max_x = -1e10;

    for (int i = 0; i < node_count; ++i)
    {

        node_t node;
        //
        // TODO: initialize each node of your finite element mesh here.
        // e.g start by uncommenting the following line and then run the code.
        //
        // node.pos = sm.tetmesh.vertices[i];

        

        sm.nodes.push_back(node);

        //
        // do not change this
        //
        min_x = glm::min(min_x, node.pos.x);
        max_x = glm::max(max_x, node.pos.x);

    } // for (int i = 0; i < node_count; ++i) {

    //
    // set constrained vertices
    //
    for (int i = 0; i < node_count; ++i)
    {
        const node_t &node = sm.nodes.at(i);

        if (node.pos.x == min_x)
        {
            fixed_nodes.push_back(i);
        }
        else if (node.pos.x == max_x)
        {
            pulled_and_pushed_nodes.push_back(i);
        }
    }

    printf("fixed nodes=%d\n", (int)fixed_nodes.size());
    printf("pulled and pushed nodes=%d\n", (int)pulled_and_pushed_nodes.size());

    assert(fixed_nodes.size() >= 3);

    //
    // TODO: Compute constant variables of each finite element
    //

    const int element_count = sm.tetmesh.tetrahedra.size() / 4;
    assert(element_count >= 1);

    for (int i = 0; i < element_count; ++i)
    {
        const int base_index = i * 4;
        finite_element_t e;

        // This array represents a list of N*4 integers, where N is the number of
        // tetrahedra (i.e. elements). The first tetrahedron has its vertices (indices)
        // at tet[0] .. tet[3], the second tetrahedron at tet[4]... tet[7] and so on.
        const std::vector<int> &tet = sm.tetmesh.tetrahedra;
        const vec3 &Xi = sm.nodes[tet[base_index + 0]].pos;
        // const vec3 &Xj = ...

        sm.finite_elements.push_back(e);

    } // for (int i = 0; i < element_count; ++i) {
}

void update_physics(float dt)
{
    // compute elastic forces (per node)

    const int node_count = sim_mesh.nodes.size();

    //
    // reset all vertex forces
    //
    for (int i = 0; i < node_count; ++i)
    {
        sim_mesh.nodes[i].force = glm::vec3(0, 0, 0);
    }

    //
    // compute elastic forces
    //

    const int element_count = sim_mesh.finite_elements.size();
    const float mu = sim_mesh.material_parameters.mu;
    const float lambda = sim_mesh.material_parameters.lambda;

    //
    // TODO: compute elastic forces
    //

    for (int i = 0; i < element_count; ++i)
    { // for each finite e

        finite_element_t &e = sim_mesh.finite_elements[i]; // reference to current e
        const int base_index = i * 4; // the starting index of [the vertex indices which define the current e]

        const std::vector<int> &tet = sim_mesh.tetmesh.tetrahedra;
        const glm::vec3 &xi = sim_mesh.nodes[tet[base_index + 0]].pos;
        // const glm::vec3 &xj = ...

        glm::mat3 P(1.0);

        int model = MaterialModel::LINEAR;
        switch (model)
        {
        case MaterialModel::LINEAR: {
        }
        break;
        // ...
        default:
            break;
        }

        glm::mat3 H(1.0);
        // ...

        sim_mesh.nodes[tet[base_index + 0]].force += H[0];
        // sim_mesh.nodes[tet[base_index + 1]].force += ...
    } // for (int i = 0; i < element_count; ++i) {

    for (int i = 0; i < node_count; ++i)
    {
        node_t &node = sim_mesh.nodes[i];

        // Numerical integration
        bool is_fixed = std::find(fixed_nodes.begin(), fixed_nodes.end(), i) != fixed_nodes.end();

        bool is_pulled_and_pushed_node = std::find(pulled_and_pushed_nodes.begin(), pulled_and_pushed_nodes.end(), i) !=
                                         pulled_and_pushed_nodes.end();

        bool is_free_dof = !is_fixed && !is_pulled_and_pushed_node;
        if (is_free_dof)
        {
            //
            // TODO: update free vertex positions (DOFs) using numerical integration here
            //

            // Example verlet integration:
            // x(t + dt)    = 2x(t) - x(t - dt) + ddx(t)dt^2
            //              = x(t) + [x(t)  - x(t - dt)] + (f(t)/m)dt^2
        }
        else
        {
            if (is_pulled_and_pushed_node)
            {
                float upper = 2.631568;
                float lower = 0.210527f;

                if (apply_pull_force) // pull (tensile loading)
                {
                    node.pos += glm::vec3(0.5f, 0.f, 0.f) * dt;
                    node.pos.x = glm::clamp(node.pos.x, -1e10f, upper);
                }
                else // push (compressive loading)
                {
                    node.pos += glm::vec3(-0.25f, 0.f, 0.f) * dt;
                    node.pos.x = glm::clamp(node.pos.x, lower, 1e10f);
                }
            }
        }

        const float s = scene_scale / 2;
        const glm::vec3 cmin(-s, 0, -s);
        const glm::vec3 cmax(s, 5, s);
        node.pos = glm::clamp(node.pos, cmin, cmax); // simple collision detection with invisible walls
    }
}

void draw_scene();
void destroy();
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void process_input(GLFWwindow *window);

std::map<int, int> get_tmesh_to_smesh_vmap(const simulation_mesh_t &sm)
{
    assert(sm.tetmesh.vertices.size() >= 4);

    std::map<int, int> tmesh_to_smesh_vmap;
    const int boundary_face_count = (int)sm.tetmesh.boundaryFaces.size() / 3;

    assert(boundary_face_count >= 4);

    for (int i = 0; i < boundary_face_count; ++i)
    { // for each tetrahedron

        int face_1st_vertex_idx = i * 3;

        for (int j = 0; j < 3; ++j)
        { // for each vertex of face
            int vertex_index = sm.tetmesh.boundaryFaces[face_1st_vertex_idx + j];
            if (tmesh_to_smesh_vmap.find(vertex_index) == tmesh_to_smesh_vmap.cend())
            {
                tmesh_to_smesh_vmap[vertex_index] = tmesh_to_smesh_vmap.size();
            }
        }
    }
    return tmesh_to_smesh_vmap;
}

std::vector<glm::vec3> get_boundary_vertex_array(const std::map<int, int> &tmesh_to_smesh_vmap)
{
    std::vector<glm::vec3> out(tmesh_to_smesh_vmap.size());

    const int boundary_face_count = (int)sim_mesh.tetmesh.boundaryFaces.size() / 3;

    for (int i = 0; i < boundary_face_count; ++i)
    { // for each tetrahedron
        const int face_1st_vertex_idx = i * 3;

        for (int j = 0; j < 3; ++j)
        { // for each vertex of face
            const int tmesh_index = sim_mesh.tetmesh.boundaryFaces[face_1st_vertex_idx + j];
            const int smesh_index = tmesh_to_smesh_vmap.at(tmesh_index);
            out[smesh_index] = sim_mesh.tetmesh.vertices[tmesh_index];
        }
    }

    return out;
}

std::vector<int> get_boundary_vertex_array_face_indices(const std::map<int, int> &tmesh_to_smesh_vmap)
{
    std::vector<int> out;
    const int boundary_face_count = (int)sim_mesh.tetmesh.boundaryFaces.size() / 3;

    assert(boundary_face_count >= 4);

    for (int i = 0; i < boundary_face_count; ++i)
    { // for each tetrahedron

        int face_1st_vertex_idx = i * 3;

        for (int j = 0; j < 3; ++j)
        { // for each vertex of face
            int tmesh_index = sim_mesh.tetmesh.boundaryFaces[face_1st_vertex_idx + j];
            int smesh_index = tmesh_to_smesh_vmap.at(tmesh_index);
            out.push_back(smesh_index);
        }
    }

    return out;
}

int main(int argc, char *argv[])
{
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
    g_window = glfwCreateWindow(g_gl_width, g_gl_height, PROJECT_NAME_STR, NULL, NULL);

    if (g_window == NULL)
    {
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
    // glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
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
    glDepthFunc(GL_LESS);    // depth-testing interprets a smaller value as "closer"
    // glEnable(GL_CULL_FACE); // cull face
    // glCullFace(GL_BACK);    // cull back face
    glFrontFace(GL_CCW);              // set counter-clock-wise vertex order to mean the front
    glClearColor(0.0, 0.0, 0.0, 1.0); // grey background to help spot mistakes
    glViewport(0, 0, g_gl_width, g_gl_height);

    // load and setup shaders
    // ----------------------
    shader_programme = create_programme_from_files(PROJECT_ROOT_DIR "/test_vs.glsl", PROJECT_ROOT_DIR "/test_fs.glsl");
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
        glBufferData(GL_ARRAY_BUFFER, 3 * plane_mesh.get_vertex_count() * sizeof(GLfloat),
                     &plane_mesh.get_vertices()[0], GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(0);

        glGenBuffers(1, &plane_ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, plane_mesh.get_index_count() * sizeof(GLuint),
                     &plane_mesh.get_indices()[0], GL_STATIC_DRAW);
    }
    glBindVertexArray(0);

    // setup grid lines along ground mesh
    // ----------------------------------

    int slices = 16;
    for (int j = 0; j <= slices; ++j)
    {
        for (int i = 0; i <= slices; ++i)
        {
            float x = (float)i / (float)slices;
            float y = 0;
            float z = (float)j / (float)slices;
            grid_vertices.push_back(glm::vec3((x - 0.5) * 2, y, (z - 0.5) * 2));
        }
    }

    for (int j = 0; j < slices; ++j)
    {
        for (int i = 0; i < slices; ++i)
        {

            GLuint row1 = j * (slices + 1);
            GLuint row2 = (j + 1) * (slices + 1);

            grid_indices.insert(grid_indices.end(), {row1 + i, row1 + i + 1, row1 + i + 1, row2 + i + 1});
            grid_indices.insert(grid_indices.end(), {row2 + i + 1, row2 + i, row2 + i, row1 + i});
        }
    }

    {
        glGenVertexArrays(1, &grid_vao);
        glBindVertexArray(grid_vao);

        glGenBuffers(1, &grid_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, grid_vbo);
        glBufferData(GL_ARRAY_BUFFER, grid_vertices.size() * sizeof(float) * 3, &grid_vertices[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        glGenBuffers(1, &grid_ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, grid_ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, grid_indices.size() * sizeof(GLuint), &grid_indices.data()[0],
                     GL_STATIC_DRAW);

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    // load and setup mesh used for simulation
    // ---------------------------------------

    // load tetmesh files
    sim_mesh.tetmesh.load(PROJECT_ROOT_DIR "/cylinder.1");

    tmesh_to_smesh_vmap = get_tmesh_to_smesh_vmap(sim_mesh);

    // rescale the input tetmesh vertices according to our needs
    const glm::vec3 local_mesh_translation(.0f, 2.f, .0f);
    const glm::vec3 local_mesh_scale(1.f, 1.f, 1.f);
    normalize_mesh(sim_mesh.tetmesh, local_mesh_translation, local_mesh_scale);

    // array of boundary vertex positions that we will use to render the
    // surface of our deformable object
    const std::vector<glm::vec3> boundary_vertex_array = get_boundary_vertex_array(tmesh_to_smesh_vmap);
    const std::vector<int> boundary_vertex_array_face_indices =
        get_boundary_vertex_array_face_indices(tmesh_to_smesh_vmap);

    glGenVertexArrays(1, &object_vao);
    glBindVertexArray(object_vao);
    {
        glGenBuffers(1, &object_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, object_vbo);
        glBufferData(GL_ARRAY_BUFFER, 3 * boundary_vertex_array.size() * sizeof(GLfloat), &boundary_vertex_array[0],
                     GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(0);

        glGenBuffers(1, &object_ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object_ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, boundary_vertex_array_face_indices.size() * sizeof(GLuint),
                     &boundary_vertex_array_face_indices[0], GL_STATIC_DRAW);
    }
    glBindVertexArray(0);

    init_physical_object(sim_mesh);

    // timing

    float lastFrame = 0.0f;
    /*-------------------------------MAIN LOOP-------------------------------*/
    while (!glfwWindowShouldClose(g_window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        dt = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        process_input(g_window);

        const float timestep_size = 1.0 / 1000;
        const int steps_per_frame = 16;

        for (int i = 0; i < steps_per_frame; ++i)
        {
            update_physics(timestep_size);
        }

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

void destroy()
{
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

void draw_scene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader_programme);

    const float near_plane = 0.1f;  // clipping plane
    const float far_plane = 100.0f; // clipping plane

    // float fovy = 67.0f;                                    // 67 degrees
    const float aspect = (float)g_gl_width / (float)g_gl_height; // aspect ratio
    const glm::mat4 projection = glm::perspective(glm::radians(camera.m_zoom), aspect, near_plane, far_plane);
    glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, glm::value_ptr(projection));

    const glm::mat4 view = camera.get_view_matrix();
    glUniformMatrix4fv(view_mat_location, 1, GL_FALSE, glm::value_ptr(view));

    // draw ground plane
    {
        glBindVertexArray(plane_vao);
        const glm::mat4 model_mat = glm::mat4(1.0); // at origin
        glUniformMatrix4fv(model_mat_location, 1, GL_FALSE, glm::value_ptr(model_mat));

        const glm::vec4 colour(0.3f, 0.3f, 0.3f, 0.3f);
        glUniform4fv(colour_location, 1, glm::value_ptr(colour));
        glDrawElements(GL_TRIANGLES, plane_mesh.get_index_count(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    // draw grid
    {
        glBindVertexArray(grid_vao);
        const float s = scene_scale / 2;
        glm::mat4 model_mat = glm::scale(glm::mat4(1.0), glm::vec3(s, s, s)); // at origin and scale to match env
        model_mat = translate(model_mat, glm::vec3(0, 1e-3, 0));
        glUniformMatrix4fv(model_mat_location, 1, GL_FALSE, glm::value_ptr(model_mat));

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

        { // update boundary vertex positions from CPU RAM to the GPU
            glBindBuffer(GL_ARRAY_BUFFER, object_vbo);

            void *ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
            // rest space vertex positions (from tet mesh).
            std::vector<glm::vec3> boundary_vertex_array(tmesh_to_smesh_vmap.size());

            for (std::map<int, int>::const_iterator i = tmesh_to_smesh_vmap.cbegin(); i != tmesh_to_smesh_vmap.cend();
                 ++i)
            {
                const int tmesh_vidx = i->first;
                const int smesh_vidx = i->second;
                const node_t &n = sim_mesh.nodes[tmesh_vidx];
                const glm::vec3 &pos = n.pos;
                boundary_vertex_array[smesh_vidx] = pos;
            }

            memcpy(ptr, &boundary_vertex_array[0], boundary_vertex_array.size() * 3 * sizeof(float));

            glUnmapBuffer(GL_ARRAY_BUFFER);
        }

        glm::mat4 model_mat = glm::scale(glm::mat4(1.0), glm::vec3(1.001f));
        glUniformMatrix4fv(model_mat_location, 1, GL_FALSE, glm::value_ptr(model_mat));
        glm::vec4 colour(1.0f, 0.0f, 0.0f, 0.0f);
        glUniform4fv(colour_location, 1, glm::value_ptr(colour));
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, sim_mesh.tetmesh.boundaryFaces.size(), GL_UNSIGNED_INT, 0);

        model_mat = glm::mat4(1.0);
        glUniformMatrix4fv(model_mat_location, 1, GL_FALSE, glm::value_ptr(model_mat));
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        colour = glm::vec4(0.1f, 0.1f, 0.1f, 0.1f);
        glUniform4fv(colour_location, 1, glm::value_ptr(colour));
        glDrawElements(GL_TRIANGLES, sim_mesh.tetmesh.boundaryFaces.size(), GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);
    }
}

// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void process_input(GLFWwindow *window)
{
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
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width
    // and height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.process_mouse_motion(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    camera.process_mouse_scroll(static_cast<float>(yoffset));
}

// mini helper function to compute PD
void polar_decomposition(const glm::mat3 &m, glm::mat3 &R, glm::mat3 &S)
{
    pd(m[0][0], m[1][0], m[2][0], // row 0
       m[0][1], m[1][1], m[2][1], // row 1
       m[0][2], m[1][2], m[2][2], // row 2
       // output R
       R[0][0], R[1][0], R[2][0], // row 0
       R[0][1], R[1][1], R[2][1], // row 1
       R[0][2], R[1][2], R[2][2], // row 2
       // output S
       S[0][0], S[1][0], S[2][0], // row 0
       S[0][1], S[1][1], S[2][1], // row 1
       S[0][2], S[1][2], S[2][2]);
}

float trace(const glm::mat3 &m)
{
    return m[0][0] + m[1][1] + m[2][2];
}

void normalize_mesh(tetmesh_t &tetmesh, const glm::vec3 &translation, const glm::vec3 &scale)
{
    glm::vec3 minimum(FLT_MAX);
    glm::vec3 maximum(FLT_MIN);

    for (int i = 0; i < (int)tetmesh.vertices.size(); ++i)
    {
        const glm::vec3 &v = tetmesh.vertices[i];

        if (v.x < minimum.x)
        {
            minimum.x = v.x;
        }
        else if (v.x > maximum.x)
        {
            maximum.x = v.x;
        }

        if (v.y < minimum.y)
        {
            minimum.y = v.y;
        }
        else if (v.y > maximum.y)
        {
            maximum.y = v.y;
        }

        if (v.z < minimum.z)
        {
            minimum.z = v.z;
        }
        else if (v.z > maximum.z)
        {
            maximum.z = v.z;
        }
    }

    glm::vec3 offset((maximum + minimum) / 2.0f);
    glm::vec3 span(std::fabs(maximum.x - minimum.x), std::fabs(maximum.y - minimum.y),
                   std::fabs(maximum.z - minimum.z));

    float maxSpan = std::max(span.x, std::max(span.y, span.z));
    float normalize_scale = 2.0f / maxSpan;

    for (int i = 0; i < (int)tetmesh.vertices.size(); ++i)
    {
        glm::vec3 &v = tetmesh.vertices[i];
        v -= offset;
        v *= scale * normalize_scale;
        v += translation;
    }
}
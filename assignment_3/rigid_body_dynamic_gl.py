import taichi as ti
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time
# Set up Taichi
ti.init(arch=ti.cpu, debug=True)


# Function to read an OBJ file
def read_obj_file(file_path, scale=1.0):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            elements = line.split()
            if elements[0] == 'v':
                vertices.append([scale * float(e) for e in elements[1:]])
            elif elements[0] == 'f':
                faces.append([int(e.split('/')[0]) - 1 for e in elements[1:]])

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


# Read an OBJ file, feel free to change the file path to use your own mesh
file_path = "sphere.obj"
vertices_np, faces_np = read_obj_file(file_path, scale=0.3)

faces = ti.field(dtype=ti.i32, shape=faces_np.shape)
# Particle state
particle_vertices = ti.Vector.field(3, dtype=ti.f32, shape=vertices_np.shape[0])
particle_origin_vertices = ti.Vector.field(3, dtype=ti.f32, shape=vertices_np.shape[0])
particle_velocities = ti.Vector.field(3, dtype=ti.f32, shape=vertices_np.shape[0])
particle_force = ti.Vector.field(3, dtype=float, shape=vertices_np.shape[0])

# Body state
# IMPORTANT: if you wish to access the following fields in the kernel function, use field_name[None]
body_cm_position = ti.Vector.field(3, dtype=float, shape=())
body_origin_cm_position = ti.Vector.field(3, dtype=float, shape=())
body_velocity = ti.Vector.field(3, dtype=float, shape=())
body_angular_velocity = ti.Vector.field(3, dtype=float, shape=())
body_rotation = ti.Matrix.field(3, 3, dtype=float, shape=())
body_rotation_quaternion = ti.Vector.field(4, dtype=float, shape=())
body_angular_momentum = ti.Vector.field(3, dtype=float, shape=())
body_origin_inverse_inertia = ti.Matrix.field(3, 3, dtype=float, shape=())
body_mass = ti.field(float, shape=())

# Simulation parameters, feel free to change them
# We assume all particles have the same mass
particle_mass = 1
initial_velocity = ti.Vector([0.0, 0.0, 0.0])
initial_angular_velocity = ti.Vector([0.0, 0, 0.0])
gravity = ti.Vector([0.0, -9.8, 0.0])
# stiffness of the collision
collision_stiffness = 1e4
velocity_damping_stiffness = 1e3
friction_stiffness = 0.1
# simulation integration time step
dt = 1e-3

# Initialize the fields
# Copy the vertices and faces numpy data to Taichi Fields
particle_vertices.from_numpy(vertices_np)
particle_origin_vertices.from_numpy(vertices_np)
faces.from_numpy(faces_np)

# Indices field for rendering
indices = ti.field(int, shape=3 * faces_np.shape[0])
for i in range(faces_np.shape[0]):
    indices[3 * i] = faces[i, 0]
    indices[3 * i + 1] = faces[i, 1]
    indices[3 * i + 2] = faces[i, 2]


@ti.kernel
def initial():
    # Initialize the body and particle state

    # Compute the center of mass and mass of the body
    for i in ti.grouped(particle_vertices):
        body_mass[None] += particle_mass
        body_cm_position[None] += particle_mass * particle_vertices[i]
    body_cm_position[None] /= body_mass[None]
    body_origin_cm_position[None] = body_cm_position[None]

    # Compute the inertia of the body
    inertia = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    # TODO 1: Compute the inertia tensor of the body
    # Hint: You can use the function ti.Matrix.outer_product to compute v*v^T
    # Hint: You can use the function ti.Matrix.dot to compute v^T*v
    # Hint: You can use the function ti.Matrix.identity(float, 3) to get a 3x3 identity matrix
    for i in ti.grouped(particle_vertices):
        # inertia += particle_mass * ((particle_vertices[i] - body_cm_position[None]).dot(
        pass

    # Compute the inverse inertia of the body and store it in the field
    body_origin_inverse_inertia[None] = inertia.inverse()

    # Initialize the particle velocities
    for i in ti.grouped(particle_vertices):
        particle_velocities[i] = initial_velocity

    # Initialize the body state
    body_velocity[None] = initial_velocity
    body_angular_velocity[None] = initial_angular_velocity
    body_angular_momentum[None] = inertia @ initial_angular_velocity

    # Initialize the rotation matrix and quaternion
    body_rotation[None] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    body_rotation_quaternion[None] = ti.Vector([1.0, 0.0, 0.0, 0.0])


initial()


# quaternion multiplication, this is used to update the rotation quaternion
@ti.func
def quaternion_multiplication(p: ti.template(), q: ti.template()) -> ti.template():
    return ti.Vector([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                      p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                      p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                      p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]])


# quaternion to rotation matrix
@ti.func
def quaternion_to_matrix(q: ti.template()) -> ti.template():
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    return ti.Matrix([[qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                      [2 * (qx * qy + qw * qz), (qw * qw - qx * qx + qy * qy - qz * qz), 2 * (qy * qz - qw * qx)],
                      [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz]])


@ti.kernel
def substep():
    # computer the force on each particle
    for i in ti.grouped(particle_vertices):
        # TODO 2: gravity
        # particle_force[i] =

        # Collision force, we use a spring model to simulate the collision
        if particle_vertices[i][1] < -1:
            f_collision = collision_stiffness * (-1 - particle_vertices[i][1])
            particle_force[i] += ti.Vector([0, f_collision, 0])
        if particle_vertices[i][0] < -1:
            f_collision = collision_stiffness * (-1 - particle_vertices[i][0])
            particle_force[i] += ti.Vector([f_collision, 0, 0])
        if particle_vertices[i][0] > 1:
            f_collision = collision_stiffness * (1 - particle_vertices[i][0])
            particle_force[i] += ti.Vector([f_collision, 0, 0])
        if particle_vertices[i][2] < -1:
            f_collision = collision_stiffness * (-1 - particle_vertices[i][2])
            particle_force[i] += ti.Vector([0, 0, f_collision])
        if particle_vertices[i][2] > 1:
            f_collision = collision_stiffness * (1 - particle_vertices[i][2])
            particle_force[i] += ti.Vector([0, 0, f_collision])

    # computer the force for rigid body
    body_force = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.grouped(particle_vertices):
        # TODO 3: compute the force for rigid body
        # body_force +=
        pass

    # computer the torque for rigid body
    body_torque = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.grouped(particle_vertices):
        # TODO 4: compute the torque for rigid body
        # Hint: use ti.math.cross(v1, v2) to compute the cross product
        # torque +=
        pass

    # update the rigid body
    # TODO 5: update the center of mass position and velocity
    # body_cm_position[None] +=
    # body_velocity[None] +=

    # TODO 6: update the rotation quaternion
    # d_q = 0.5 * quaternion_multiplication(ti.Vector([0, ?, ?, ?]), body_rotation_quaternion[None])
    # body_rotation_quaternion[None] +=

    # normalize the quaternion to avoid numerical error
    body_rotation_quaternion[None] /= body_rotation_quaternion[None].norm()
    body_rotation[None] = quaternion_to_matrix(body_rotation_quaternion[None])

    # TODO 7: update, the angular momentum, inertia tensor and angular velocity
    # body_angular_momentum[None] =
    # body_inverse_inertia_body[None] =
    # body_angular_velocity[None] =

    # update the particles
    for i in ti.grouped(particle_vertices):
        ri = body_rotation[None] @ (particle_origin_vertices[i] - body_origin_cm_position[None])
        particle_vertices[i] = ri + body_cm_position[None]
        particle_velocities[i] = body_velocity[None] + ti.math.cross(body_angular_velocity[None], ri)


# GUI stuff
# draw a cube frame
frame_vertices = ti.Vector.field(3, dtype=float, shape=24)
vertices_list = [
    [-1, -1, 0], [1, -1, 0], [1, -1, 0], [1, 1, 0],
    [1, 1, 0], [-1, 1, 0], [-1, 1, 0], [-1, -1, 0],
    [-1, -1, 1], [1, -1, 1], [1, -1, 1], [1, 1, 1],
    [1, 1, 1], [-1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    [-1, -1, 0], [-1, -1, 1], [1, -1, 0], [1, -1, 1],
    [1, 1, 0], [1, 1, 1], [-1, 1, 0], [-1, 1, 1]
]
for i in range(len(vertices_list)):
    frame_vertices[i] = ti.Vector(vertices_list[i])

# rendering frame rate is 1/60
substeps = int(1 / 60 // dt)
current_t = 0.0


np_vertex_positions = particle_vertices.to_numpy()


def draw_mesh():
    glBegin(GL_LINES)
    for face in faces_np:
        for i in range(3):
            glColor3f(0, 0, 0)  # Set line color to black
            glVertex3fv(np_vertex_positions[face[i]])
            glVertex3fv(np_vertex_positions[face[(i + 1) % 3]])
    glEnd()

def draw_cube_frame():
    glColor3f(1, 0, 0)  # Set line color to red
    glBegin(GL_LINES)
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for e in edges:
        glVertex3fv(vertices[e[0]])
        glVertex3fv(vertices[e[1]])
    glEnd()

def update_mesh():
    global np_vertex_positions, current_t
    for i in range(substeps):
        substep()
        current_t += dt
    np_vertex_positions = particle_vertices.to_numpy()


# Initialize GLFW
if not glfw.init():
    raise RuntimeError('GLFW initialization failed')

# Create a GLFW window
window = glfw.create_window(1024, 1024, 'Rigid Body Simulation', None, None)
if not window:
    glfw.terminate()
    raise RuntimeError('GLFW window creation failed')

# Make the window's context current
glfw.make_context_current(window)
glfw.swap_interval(1)  # Enable vsync

# Set the perspective and view
glMatrixMode(GL_PROJECTION)
gluPerspective(45, 800 / 600, 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glTranslatef(0.0, 0.0, -3)

# Set the shading model and enable depth test
glShadeModel(GL_SMOOTH)
glEnable(GL_DEPTH_TEST)
glClearColor(1, 1, 1, 1)
# Main loop

# Variables for FPS calculation
frame_count = 0
fps_refresh_rate = 1.0  # Refresh the FPS count every second
previous_time = time.time()

while not glfw.window_should_close(window):
    current_time = time.time()
    frame_count += 1

    # Calculate and display the FPS
    if current_time - previous_time >= fps_refresh_rate:
        fps = frame_count / (current_time - previous_time)
        glfw.set_window_title(window, f'Rigid Body Simulation - FPS: {fps:.2f}')
        frame_count = 0
        previous_time = current_time
    # Clear the screen and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Update and draw the mesh
    update_mesh()
    draw_mesh()
    draw_cube_frame()

    # Swap buffers
    glfw.swap_buffers(window)

    # Poll for events
    glfw.poll_events()

# Terminate GLFW
glfw.terminate()
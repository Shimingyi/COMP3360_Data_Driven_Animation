import taichi as ti

ti.init(arch=ti.cpu)  # Alternatively, ti.init(arch=ti.cpu)
# ti.init(debug=True)

ball_radius = 0.2
# Use a 1D field for storing the position of the ball center
# The only element in the field is a 3-dimentional floating-point vector
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
# Place the ball center at the original point
ball_center[0] = [0, -1, 0]

n = 32
quad_size = 1.0 / n
dt = 4e-3 / n
substeps = int(1 / 60 // dt)

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
is_fixed = ti.field(dtype=bool, shape=(n, n))


@ti.kernel
def initialize_mass_points():
    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5, 0, j * quad_size - 0.5
        ]
        v[i, j] = [0, 0, 0]
        # is_fixed[i, j] = (i == 0 and j == n - 1) or (i == n - 1 and j == n - 1)
        is_fixed[i, j] = False


spring_offsets = []

for i in range(-2, 3):
    for j in range(-2, 3):
        if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
            spring_offsets.append(ti.Vector([i, j]))
print(spring_offsets)
gravity = ti.Vector([0, -9.8, 0])
spring_stiffness = 3e3
damping_stiffness = 50


@ti.kernel
def substep():
    # gravity
    for i in ti.grouped(x):
        v[i] += gravity * dt
    # internal springs

    for i in ti.grouped(x):
        force = ti.Vector([0, 0, 0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # spring force
                force += -spring_stiffness * d * (current_dist / original_dist - 1)
                # damping force
                force += -damping_stiffness * v_ij.dot(d) * d / current_dist
        v[i] += force * dt

    for i in ti.grouped(x):
        if is_fixed[i]:
            v[i] = ti.Vector([0, 0, 0])

    for i in ti.grouped(x):
        offset_to_centre = x[i] - ball_center[0]
        if offset_to_centre.norm() < ball_radius:
            normal = offset_to_centre.normalized()
            v[i] -= min(0, v[i].dot(normal)) * normal
        x[i] += v[i] * dt


num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


window = ti.ui.Window("Cloth Simulation", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0

initialize_mass_points()


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j


initialize_mesh_indices()

while window.running:
    # if current_t > 5:
    #     # Reset
    #     initialize_mass_points()
    #     current_t = 0

    for i in range(substeps):
        substep()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True,
               show_wireframe=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()

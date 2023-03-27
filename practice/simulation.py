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


@ti.kernel
def initialize_mass_points():
    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5, j * quad_size - 0.5, 0
        ]
        v[i, j] = [0, 0, 0]


@ti.kernel
def substep():
    # TODO
    pass


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
    scene.particles(ball_center, radius=ball_radius * 1.6, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()

import taichi as ti
import numpy as np


ti.init(arch=ti.cpu, debug=False)


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


def init_edges(vertices, faces):
    # Generate vertex index pairs for edges, assuming triangular faces.
    edges = np.hstack([faces, np.roll(faces, -1, axis=1)]).reshape(-1, 2)

    # Ensure each edge is ordered (smaller index first).
    edges = np.sort(edges, axis=1)

    # Remove duplicate edges.
    edges = np.unique(edges, axis=0)

    # Compute edge lengths.
    edge_lengths = np.sqrt(np.sum(np.square(vertices[edges[:, 0]] - vertices[edges[:, 1]]), -1))

    return edges, edge_lengths


# Read an OBJ file, feel free to change the file path to use your own mesh
file_path = "cloth.obj"
vertices_np, faces_np = read_obj_file(file_path, scale=1)
edges_np, edge_lengths_np = init_edges(vertices_np, faces_np)

indices = ti.field(int, shape=3 * faces_np.shape[0])
indices.from_numpy(faces_np.flatten())


num_verts = vertices_np.shape[0]
num_edges = edges_np.shape[0]

# Simulation parameters
fps = 60
subsample = 1
t = 1.0 / fps
damping = 0.99
alpha = 0.2
gravity = ti.Vector([0, -9.8, 0])

# Collision body
c = ti.Vector.field(3, dtype=ti.f32, shape=1)
c.from_numpy(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
r = 0.5

prev_X = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)  # Previous position
X = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)  # Current position
E = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)  # Edge indices
L = ti.field(dtype=ti.f32, shape=num_edges)               # Rest edge length
sum_X = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
sum_N = ti.field(dtype=ti.i32, shape=num_verts)

# Copy data to Taichi fields
X.from_numpy(vertices_np)
prev_X.from_numpy(vertices_np)
E.from_numpy(edges_np)
L.from_numpy(edge_lengths_np)


@ti.kernel
def strain_limiting():
    sum_X.fill(0.0)
    sum_N.fill(0)
    # TODO 1: Implement the projection function
    # There will be two loops. The first iterates over all edges to compute the correction vector
    # The second iterates over all vertices and linearly blends the vectors
    # Variables you can use: X (vertex position), E (edges, i.e., indices of two vertices each edge connects),
    # L (edge length), sum_X, sum_N, alpha (blend factor)
    # There is no need to update velocity here. You only need to update the position.
    # To fix the corner of the cloth, you need to keep the vertex 0 and vertex 1 fixed.
    pass

@ti.kernel
def collision_handling():
    # TODO 2: Implement collision handling
    # when the cloth vertex falls within the collision sphere, push it back to the sphere surface
    # Simply find the closest point on the sphere surface and move the vertex to that point
    # Variables you can use: X (vertex position), c (collision sphere position), r (collision sphere radius)
    # To fix the corner of the cloth, you need to keep the vertex 0 and vertex 1 fixed.
    pass


@ti.kernel
def update():
    for i in range(num_verts):
        if i != 0 and i != 1:
            v = (X[i] - prev_X[i]) / t
            prev_X[i] = X[i]
            # TODO 3
            # 1. Update the velocity (a. multiply by damping, b. add gravity)
            # 2. Update the position X using the velocity
            # To fix the corner of the cloth, you need to keep the vertex 0 and vertex 1 fixed.
            pass


def substep():
    update()
    for _ in range(32):
        strain_limiting()
    collision_handling()


headless = False  # if this is True, no window will show, video will be saved locally
window = ti.ui.Window("PBD Model", (800, 800), fps_limit=fps // subsample, vsync=True,
                      show_window=not headless)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(-4, 0, -2)
camera.lookat(0, 0, 0.2)
camera.up(0, 1, 0)

if headless:
    video_manager = ti.tools.VideoManager(output_dir='.', framerate=fps // subsample)
else:
    video_manager = None

idx = 0
while window.running:
    for _ in range(subsample):
        substep()

    scene.set_camera(camera)
    scene.point_light(pos=(-4, 0, -2), color=(1, 1, 1))
    scene.ambient_light((0.2, 0.2, 0.2))

    # Since the cloth's resolution is low,
    # here we make the sphere visually smaller than the actual collision sphere to prevent intersection
    scene.particles(c, radius=r-0.02, color=(0, 1, 1))
    scene.mesh(X, indices=indices, color=(1, 0.5, 0.7), show_wireframe=False, two_sided=True)

    canvas.scene(scene)

    idx += 1

    if headless:
        img = window.get_image_buffer_as_numpy()
        video_manager.write_frame(img)
        if idx >= 600:
            break
    else:
        window.show()

if headless:
    video_manager.make_video(gif=False, mp4=True)

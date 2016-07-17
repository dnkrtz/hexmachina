import meshpy
import numpy as np
import trimesh

from .helpers import compute_normal, compute_avg

###############################
## Generate tetrahedral mesh ##
###############################

tri_mesh = trimesh.load_mesh('./tests/data/icosphere.stl')

# Define MeshPy options
opt = meshpy.tet.Options(switches='pq', edgesout=True, facesout=True, neighout=True)

# Generate mesh
mesh_info = meshpy.tet.MeshInfo()
mesh_info.set_points(tri_mesh.vertices)
faces = [list(map(lambda x: int(x), i)) for i in tri_mesh.faces]
mesh_info.set_facets(faces)
tet_mesh = meshpy.tet.build(mesh_info, opt, max_volume=10)

# Output tetrahedral mesh
tet_mesh.write_vtk("./tests/data/test.vtk")


###########################################
## Compute cross-field surface curvature ##
###########################################

surface_faces = []
for i, tet in enumerate(tet_mesh.elements):
    neighbors = list(tet_mesh.neighbors[i])
    # TODO(aidan) This overlooks case of multiple surface faces...
    try:
        non_surface_vtx = tet[neighbors.index(-1)]
        tet_cpy = tet.copy()
        tet_cpy.remove(non_surface_vtx)
        surface_faces.append(tet_cpy)
    except ValueError:
    	continue

normals = [compute_normal(face) for face in surface_faces]
avg = [compute_avg(face) for face in surface_faces]


# The cubical chiral symmetry group of permutations.
rotational_symmetries = [
    # Identity
    np.matrix([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]),
    # 90 degree 4-fold rotations
    np.matrix([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]]),
    np.matrix([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]]),
    np.matrix([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]]),
    np.matrix([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]]),
    # 180 degree 4-fold rotations
    np.matrix([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]]),
    np.matrix([[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]]),
    np.matrix([[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]]),
    # 120 degree 3-fold rotations
    np.matrix([[ 0,  1,  0], [ 0,  0, -1], [-1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [ 1,  0,  0], [ 0, -1,  0]]),
    np.matrix([[ 0,  0,  1], [-1,  0,  0], [ 0, -1,  0]]),
    np.matrix([[ 0, -1,  0], [ 0,  0, -1], [ 1,  0,  0]]),
    np.matrix([[ 0,  0,  1], [ 1,  0,  0], [ 0,  1,  0]]),
    np.matrix([[ 0,  1,  0], [ 0,  0,  1], [ 1,  0,  0]]),
    np.matrix([[ 0, -1,  0], [ 0,  0,  1], [-1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [-1,  0,  0], [ 0,  1,  0]]),
    # 180 degree 2-fold rotations
    np.matrix([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]]),
    np.matrix([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]]),
    np.matrix([[-1,  0,  0], [ 0,  0, -1], [ 0, -1,  0]]),
    np.matrix([[-1,  0,  0], [ 0,  0,  1], [ 0,  1,  0]]),
    np.matrix([[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]]),
    np.matrix([[ 0,  0, -1], [ 0, -1,  0], [=1,  0,  0]]),
]

# Compute face normal
def compute_normal(face):
    return np.cross(np.array(tet_mesh.points[face[1]]) - np.array(tet_mesh.points[face[0]]),
                    np.array(tet_mesh.points[face[2]]) - np.array(tet_mesh.points[face[0]]))

# Compute face center
def compute_avg(face):
    return (np.array(tet_mesh.points[face[0]])
          + np.array(tet_mesh.points[face[1]])
          + np.array(tet_mesh.points[face[2]])) / 3
"""Exact terrain queries on the rough road (height-map patch), through the compiled helper of terrain/.

The helper answers height, normal and friction from the triangle mesh of the patch, in double
precision. Outside the mesh it falls back to the Chrono terrain. Flat motions do not need it.
"""
import ctypes as ct

import numpy as np
from scipy.spatial.transform import Rotation
import pychrono as chrono
import pychrono.vehicle as veh

import config

_LIB = []


def _library():
    if not _LIB:
        lib = ct.CDLL(str(config.TERRAIN_LIBRARY))
        lib.install_exact.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double), ct.c_size_t,
                                      ct.POINTER(ct.c_int32), ct.c_size_t, ct.c_double]
        lib.install_exact.restype = ct.c_void_p
        lib.exact_error.argtypes = []
        lib.exact_error.restype = ct.c_char_p
        _LIB.append(lib)
    return _LIB[0]


def _vec(x):
    return np.array([x.x, x.y, x.z], float)


def _rot(q):
    return Rotation.from_quat([q.e1, q.e2, q.e3, q.e0])


def mesh(rigid):
    """Vertices (world frame) and triangles of the single height-map patch of a Chrono rigid terrain."""
    body = list(rigid.GetPatches())[0].GetGroundBody()
    visual = body.GetVisualModel()
    shapes = [(chrono.CastToChTriangleMeshShape(visual.GetShape(i)), visual.GetShapeFrame(i)) for i in range(visual.GetNumShapes())]
    shapes = [(s.GetMesh(), f) for s, f in shapes if s]
    assert len(shapes) == 1
    m, frame = shapes[0]
    vertices = np.array([_vec(v) for v in m.getCoordsVertices()])
    faces = np.array([[f.x, f.y, f.z] for f in m.getIndicesVertexes()], dtype=np.int32)
    vertices = _rot(body.GetRot()).apply(_rot(frame.GetRot()).apply(vertices) + _vec(frame.GetPos())) + _vec(body.GetPos())
    return vertices, faces


class RoughRoad:
    """A Chrono terrain object whose queries are answered by the helper; mu is the friction it reports."""

    def __init__(self, mesh_data, mu, rigid):
        self.mesh = mesh_data
        vertices = np.ascontiguousarray(mesh_data[0], dtype=np.float64)
        faces = np.ascontiguousarray(mesh_data[1], dtype=np.int32)
        self.terrain = veh.ChTerrain()
        self.rigid = rigid
        handle = _library().install_exact(ct.c_void_p(int(self.terrain.this)), ct.c_void_p(int(rigid.this)),
                                          vertices.ctypes.data_as(ct.POINTER(ct.c_double)), len(vertices),
                                          faces.ctypes.data_as(ct.POINTER(ct.c_int32)), len(faces), float(mu))
        if not handle:
            raise RuntimeError(_library().exact_error().decode())

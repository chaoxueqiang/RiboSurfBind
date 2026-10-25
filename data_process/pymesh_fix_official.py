#!/usr/bin/env python3
"""Run GeoBind's released PyMesh-0.3 mesh repair in its Python-3.7 ABI.

This file intentionally preserves the operation order and constants from
zpliulab/GeoBind commit f8e336b8239fecf655d5a306ae56f887b1cf146e,
``compute_surface/fixmesh.py``.  It is a process boundary only: NumPy arrays
are exchanged with the current MVRBind environment through NPZ files.
"""

import argparse

import numpy as np
import pymesh
from sklearn.neighbors import KDTree


def fix_mesh(mesh, vertex_info, resolution, detail="normal"):
    # The bbox-derived values are retained because they occur in the released
    # function, although GeoBind then explicitly overrides target_len.
    bbox_min, bbox_max = mesh.bbox
    diagonal_length = np.linalg.norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diagonal_length * 5e-3
    elif detail == "high":
        target_len = diagonal_length * 2.5e-3
    elif detail == "low":
        target_len = diagonal_length * 1e-2
    target_len = resolution

    mesh_new, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)
    mesh_new, _ = pymesh.remove_degenerated_triangles(mesh_new, 100)
    mesh_new, _ = pymesh.split_long_edges(mesh_new, target_len)
    number_vertices = mesh.num_vertices
    count = 0
    while True:
        mesh_new, _ = pymesh.collapse_short_edges(mesh_new, 1e-6)
        mesh_new, _ = pymesh.collapse_short_edges(
            mesh_new, target_len, preserve_feature=True
        )
        mesh_new, _ = pymesh.remove_obtuse_triangles(mesh_new, 150.0, 100)
        # Preserve the released-code comparison (mesh, not mesh_new).
        if mesh.num_vertices == number_vertices:
            break
        number_vertices = mesh.num_vertices
        count += 1
        if count > 10:
            break

    mesh_new = pymesh.resolve_self_intersection(mesh_new)
    mesh_new, _ = pymesh.remove_duplicated_faces(mesh_new)
    mesh_new = pymesh.compute_outer_hull(mesh_new)
    mesh_new, _ = pymesh.remove_duplicated_faces(mesh_new)
    mesh_new, _ = pymesh.remove_obtuse_triangles(mesh_new, 179.0, 5)
    mesh_new, _ = pymesh.remove_isolated_vertices(mesh_new)
    mesh_new, _ = pymesh.remove_duplicated_vertices(mesh_new, 0.001)

    tree = KDTree(mesh.vertices)
    new_vertex_info = []
    for vertex in mesh_new.vertices:
        _, position = tree.query(vertex[None, :])
        new_vertex_info.append(vertex_info[position[0, 0]])
    return mesh_new, new_vertex_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as payload:
        vertices = np.asarray(payload["vertices"], dtype=float)
        faces = np.asarray(payload["faces"], dtype=int)
        source_ids = (
            np.asarray(payload["source_ids"]).astype(str)
            if "source_ids" in payload
            else np.arange(vertices.shape[0]).astype(str)
        )
    if source_ids.shape != (vertices.shape[0],):
        raise ValueError("source_ids must contain one MSMS source atom ID per raw vertex")
    mesh = pymesh.form_mesh(vertices, faces)
    # Released GeoBind behavior: each repaired vertex inherits the vert_info
    # of its nearest raw MSMS surface vertex.
    fixed, fixed_source_ids = fix_mesh(mesh, source_ids.tolist(), 1.2)
    fixed.add_attribute("vertex_mean_curvature")
    mean_curvature = fixed.get_attribute("vertex_mean_curvature")
    fixed.add_attribute("vertex_gaussian_curvature")
    gaussian_curvature = fixed.get_attribute("vertex_gaussian_curvature")
    np.savez_compressed(
        args.output,
        vertices=np.asarray(fixed.vertices, dtype=np.float64),
        faces=np.asarray(fixed.faces, dtype=np.int64),
        mean_curvature=np.asarray(mean_curvature, dtype=np.float64),
        gaussian_curvature=np.asarray(gaussian_curvature, dtype=np.float64),
        source_ids=np.asarray(fixed_source_ids, dtype=str),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate ligand-free RNA solvent-excluded surfaces with GeoBind's pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from Bio.PDB import PDBParser
from sklearn.neighbors import KDTree

from data_process.geometry import (
    ATOM_REGIONS,
    BASE_ORDER,
    PDB_NUCLEOTIDE_MAP,
    PHOSPHATE_ATOMS,
    RNA_BASE_ATOMS,
    _align_record,
    _chain_records,
)
from data_process.chemistry import (
    BACKBONE_ACCEPTORS,
    CANONICAL_ACCEPTORS,
    CANONICAL_DONORS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/raw/custom/surface_manifest"
DEFAULT_OUTPUT = ROOT / "data/raw/custom/surface"
PYMESH_HELPER = ROOT / "data_process/pymesh_fix_official.py"
OFFICIAL_MSMS_RADII = {
    "N": 1.54,
    "O": 1.40,
    "C": 1.74,
    "H": 1.20,
    "S": 1.80,
    "P": 1.80,
    "Z": 1.39,
    "X": 0.77,
}
ELEMENT_ORDER = ("C", "N", "O", "P", "S", "H", "other")
ATOM_NAME_VOCAB = (
    "unknown",
    "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    "N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9",
    "N2", "O6", "O2", "N4", "O4", "H", "S", "SE",
)
ATOM_NAME_TO_INDEX = {name: index for index, name in enumerate(ATOM_NAME_VOCAB)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def aligned_samples(manifest: Path):
    residue_rows = read_rows(manifest / "residue_mapping.tsv")
    by_record = {}
    for row in residue_rows:
        by_record.setdefault(row["sample_id"], []).append(row)
    for rows in by_record.values():
        rows.sort(key=lambda row: int(row["sequence_index"]))
    for sample in read_rows(manifest / "samples.tsv"):
        record_id = sample["sample_id"]
        expected = "".join(row["sequence_base"] for row in by_record[record_id])
        pdb_path = Path(sample["pdb_path"])
        model_index = int(sample["model_index"])
        chain_id = sample["chain_id"]
        mapping_rows = by_record[record_id]
        has_multichain_mapping = all(
            row.get("pdb_chain_id", "") != ""
            and row.get("pdb_residue_number", "") != ""
            for row in mapping_rows
        )
        if has_multichain_mapping:
            residue_lookup = {}
            for chain_record in _chain_records(pdb_path, model_index):
                for identifier, atoms in zip(
                    chain_record["residue_identifiers"], chain_record["residue_atoms"]
                ):
                    key = (
                        str(identifier["chain"]),
                        int(identifier["component_number"]),
                        str(identifier["insertion_code"] or ""),
                    )
                    if key in residue_lookup:
                        raise ValueError(f"Duplicate PDB residue identity for {record_id}: {key}")
                    residue_lookup[key] = (identifier, atoms)
            identifiers, atoms = [], []
            recovered = []
            for row in mapping_rows:
                key = (
                    row["pdb_chain_id"],
                    int(row["pdb_residue_number"]),
                    row.get("pdb_insertion_code", ""),
                )
                if key not in residue_lookup:
                    raise ValueError(f"Missing explicit multi-chain residue for {record_id}: {key}")
                identifier, residue_atoms = residue_lookup[key]
                identifiers.append(identifier)
                atoms.append(residue_atoms)
                recovered.append(PDB_NUCLEOTIDE_MAP.get(str(identifier["component_id"]).upper(), "N"))
            recovered_sequence = "".join(recovered).upper().replace("T", "U")
            if recovered_sequence != expected.upper().replace("T", "U"):
                raise ValueError(f"Explicit multi-chain mapping changes sequence for {record_id}")
            yield sample, expected, identifiers, atoms, "explicit_manifest_multichain_residue_ids"
            continue
        candidates = [
            record for record in _chain_records(pdb_path, model_index)
            if str(record["chain_id"]) == chain_id
        ]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous structure mapping for {record_id}: {len(candidates)}")
        record = candidates[0]
        has_explicit_indices = all(row.get("pdb_sequence_index", "") != "" for row in mapping_rows)
        if has_explicit_indices:
            pdb_indices = [int(row["pdb_sequence_index"]) for row in mapping_rows]
            if len(pdb_indices) != len(expected) or len(set(pdb_indices)) != len(pdb_indices):
                raise ValueError(f"Invalid explicit PDB residue mapping for {record_id}")
            if any(index < 0 or index >= len(record["sequence"]) for index in pdb_indices):
                raise ValueError(f"Out-of-range explicit PDB residue mapping for {record_id}")
            recovered = "".join(str(record["sequence"])[index] for index in pdb_indices)
            if recovered.upper().replace("T", "U") != expected.upper().replace("T", "U"):
                raise ValueError(f"Explicit PDB residue mapping changes sequence for {record_id}")
            atoms = [record["residue_atoms"][index] for index in pdb_indices]
            identifiers = [record["residue_identifiers"][index] for index in pdb_indices]
            alignment_status = "explicit_manifest_pdb_sequence_indices"
        else:
            _, alignment_status, atoms = _align_record(
                record, expected, f"{pdb_path} model={model_index} chain={chain_id}"
            )
            identifiers = list(record["residue_identifiers"])
            if len(record["sequence"]) == len(expected) + 1:
                identifiers = identifiers[:-1]
        if len(atoms) != len(expected) or len(identifiers) != len(expected):
            raise ValueError(f"Aligned residue length mismatch for {record_id}")
        yield sample, expected, identifiers, atoms, alignment_status


def pdb_atom_field(name: str) -> str:
    name = name.strip().replace("*", "'")[:4]
    # PDB columns 13-16: one-letter elements carry a leading blank.  Reduce
    # uses this alignment to identify nucleotide atom templates.
    return name if len(name) == 4 else f" {name:<3s}"


def write_rna_receptor(path: Path, identifiers, residue_atoms):
    lines = []
    serial = 1
    for identifier, atoms in zip(identifiers, residue_atoms):
        chain = str(identifier["chain"] or " ")[:1]
        residue_number = int(identifier["component_number"])
        insertion = str(identifier["insertion_code"] or " ")[:1]
        residue_name = str(identifier["component_id"] or "UNK").upper()[:3]
        for atom in atoms:
            coordinate = np.asarray(atom["coordinate"], dtype=float)
            element = str(atom.get("element") or atom["name"][:1]).upper()[:2]
            lines.append(
                f"ATOM  {serial:5d} {pdb_atom_field(str(atom['name']))} "
                f"{residue_name:>3s} {chain}{residue_number:4d}{insertion}   "
                f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}{coordinate[2]:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="ascii")


def run_checked(arguments, *, stdout_path=None):
    completed = subprocess.run(arguments, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(map(str, arguments))}\n"
            + completed.stderr.decode("utf-8", errors="replace")
        )
    if stdout_path is not None:
        stdout_path.write_bytes(completed.stdout)
    return completed


def protonate(reduce_bin: Path, receptor: Path, output: Path, work: Path):
    trimmed = work / "trimmed.pdb"
    run_checked([str(reduce_bin), "-Trim", str(receptor)], stdout_path=trimmed)
    # GeoBind's released protonate.py invokes Reduce with -HIS on the trimmed
    # structure; default Reduce behavior adds the hydrogens.
    run_checked([str(reduce_bin), "-HIS", str(trimmed)], stdout_path=output)


def extract_xyzrn(protonated: Path, xyzrn_path: Path, residue_key_to_index):
    structure = PDBParser(QUIET=True).get_structure("rna", str(protonated))
    lines, atoms = [], []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        if residue.get_id()[0] != " ":
            continue
        name = atom.get_name().strip().upper().replace("*", "'")
        atom_type = name[:1]
        if atom_type not in OFFICIAL_MSMS_RADII:
            continue
        chain = residue.get_parent().get_id()
        number = int(residue.get_id()[1])
        insertion = residue.get_id()[2].strip()
        key = (str(chain), number, insertion)
        if key not in residue_key_to_index:
            raise ValueError(f"Reduce emitted an unmapped RNA residue: {key}")
        residue_name = residue.get_resname().strip().upper()
        coordinate = np.asarray(atom.get_coord(), dtype=np.float32)
        element = (atom.element or atom_type).strip().upper()
        owner = residue_key_to_index[key]
        base = PDB_NUCLEOTIDE_MAP.get(residue_name)
        full_id = f"{chain}_{number}_{insertion or 'x'}_{residue_name}_{name}_{atom_type}"
        lines.append(
            f"{coordinate[0]:.06f} {coordinate[1]:.06f} {coordinate[2]:.06f} "
            f"{OFFICIAL_MSMS_RADII[atom_type]:.06f} 1 {full_id}\n"
        )
        atoms.append(
            {
                "coordinate": coordinate,
                # This exact identifier is emitted in xyzrn and returned by
                # MSMS in the .vert source-residue field.
                "source_id": full_id,
                "name": name,
                "element": element,
                "owner": owner,
                "base": base,
                "residue_name": residue_name,
            }
        )
    if not atoms:
        raise ValueError("No RNA atoms remained after Reduce/xyzrn extraction")
    xyzrn_path.write_text("".join(lines), encoding="ascii")
    return atoms


def read_msms(root: Path):
    vertex_lines = (Path(str(root) + ".vert")).read_text().rstrip().splitlines()
    vertex_count = int(vertex_lines[2].split()[0])
    vertex_fields = [line.split() for line in vertex_lines[3:]]
    vertices = np.asarray(
        [[float(value) for value in fields[:3]] for fields in vertex_fields],
        dtype=np.float64,
    )
    if vertices.shape != (vertex_count, 3):
        raise ValueError(f"MSMS vertex count mismatch: {vertices.shape} vs {vertex_count}")
    # GeoBind's released read_msms.py stores fields[9] as vert_info, then
    # transfers it to each repaired vertex in fix_mesh.
    if any(len(fields) <= 9 for fields in vertex_fields):
        raise ValueError("MSMS .vert lacks GeoBind-compatible source atom fields")
    source_ids = np.asarray([fields[9] for fields in vertex_fields], dtype=str)
    face_lines = (Path(str(root) + ".face")).read_text().rstrip().splitlines()
    face_count = int(face_lines[2].split()[0])
    faces = np.asarray(
        [[int(value) - 1 for value in line.split()[:3]] for line in face_lines[3:]],
        dtype=np.int64,
    )
    if faces.shape != (face_count, 3):
        raise ValueError(f"MSMS face count mismatch: {faces.shape} vs {face_count}")
    return vertices, faces, source_ids


def read_msms_all_components(root: Path):
    """Read and merge MSMS' primary and ``_N`` component files.

    MSMS writes disconnected components to separate ``root_N.vert/face``
    files when ``-all_components`` is enabled.  Face indices are shifted on
    concatenation; coordinates and source-atom identifiers are unchanged.
    """
    roots = [root]
    numbered = []
    for path in root.parent.glob(root.name + "_*.vert"):
        suffix = path.stem[len(root.name) + 1:]
        if suffix.isdigit():
            numbered.append((int(suffix), root.parent / path.stem))
    roots.extend(component_root for _, component_root in sorted(numbered))
    vertices_parts, faces_parts, source_parts = [], [], []
    offset = 0
    for component_root in roots:
        vertices, faces, source_ids = read_msms(component_root)
        vertices_parts.append(vertices)
        faces_parts.append(faces + offset)
        source_parts.append(source_ids)
        offset += len(vertices)
    return (
        np.concatenate(vertices_parts, axis=0),
        np.concatenate(faces_parts, axis=0),
        np.concatenate(source_parts, axis=0),
        len(roots),
    )


def compute_normals(vertices, faces):
    # Exact NumPy translation used by GeoBind compute_surface/compute_normal.py.
    points = vertices.T
    triangles = faces.T
    face_normals = np.cross(
        (points[:, triangles[1]] - points[:, triangles[0]]).T,
        (points[:, triangles[2]] - points[:, triangles[0]]).T,
    ).T
    lengths = np.sqrt(np.sum(face_normals ** 2, axis=0))
    lengths[lengths < 1e-6] = 1.0
    face_normals /= lengths[None]
    normals = np.zeros((3, points.shape[1]), dtype=np.float64)
    for face_index in range(triangles.shape[1]):
        for vertex_index in triangles[:, face_index]:
            normals[:, vertex_index] += face_normals[:, face_index]
    lengths = np.sqrt(np.sum(normals ** 2, axis=0))
    lengths[lengths < 1e-6] = 1.0
    normals /= lengths[None]
    centered = points - np.mean(points, axis=0)[None]
    orientation = np.sum(centered * normals, axis=0)
    if np.sum(orientation > 0) < np.sum(orientation < 0):
        normals = -normals
    return normals.T


def board_feature(vertices, normals):
    tree = KDTree(vertices)
    neighbors = tree.query_radius(vertices, r=12.0)
    board = []
    for index, selected in enumerate(neighbors):
        # Preserve GeoBind's released element-wise mean exactly.
        board.append(np.mean((vertices[selected] - vertices[index]) * normals[index]))
    board = np.asarray(board, dtype=np.float64)
    spread = float(np.max(board) - np.min(board))
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError("GeoBind BOARD orientation scalar is degenerate")
    return ((board - np.min(board)) / spread)[:, None]


def tangent_frames(vertices, normals, board):
    from pykeops.numpy import LazyTensor

    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    sign = (2 * (z >= 0)) - 1.0
    coefficient = -1.0 / (sign + z)
    product = x * y * coefficient
    uv = np.stack(
        (
            1 + sign * x * x * coefficient,
            sign * product,
            -sign * x,
            product,
            sign + y * y * coefficient,
            -y,
        ),
        axis=-1,
    ).reshape((-1, 2, 3))
    points = vertices / 12.0
    weights_j = LazyTensor(board[:, 0].reshape(1, -1, 1))
    x_i = LazyTensor(points[:, None, :])
    x_j = LazyTensor(points[None, :, :])
    n_i = LazyTensor(normals[:, None, :])
    n_j = LazyTensor(normals[None, :, :])
    uv_i = LazyTensor(uv.reshape((-1, 1, 6)))
    window = (-((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)).exp()
    local = uv_i.matvecmult(x_j - x_i)
    orientation = (window * weights_j * local).sum(dim=1) + 1e-5
    orientation /= np.linalg.norm(orientation, axis=1, keepdims=True)
    ex, ey = orientation[:, 0:1], orientation[:, 1:2]
    u, v = uv[:, 0], uv[:, 1]
    tangent = np.stack((ex * u + ey * v, -ey * u + ex * v), axis=1)
    return np.concatenate((normals[:, None], tangent), axis=1)


def atom_region(name, base):
    if name in PHOSPHATE_ATOMS:
        return "phosphate"
    if "'" in name or "*" in name:
        return "sugar"
    if base is not None and name in RNA_BASE_ATOMS.get(base, ()):
        return "base"
    return "other"


def atom_name_index(name):
    canonical = name.upper().replace("*", "'")
    if canonical.startswith("H"):
        canonical = "H"
    return ATOM_NAME_TO_INDEX.get(canonical, 0)


def point_chemistry(vertices, curvature, atoms, source_atom_ids):
    """Attach chemistry through inherited GeoBind MSMS source atom IDs."""
    source_atom_ids = np.asarray(source_atom_ids).astype(str)
    if source_atom_ids.shape != (vertices.shape[0],):
        raise ValueError(
            "Each repaired surface vertex must have exactly one inherited MSMS source atom"
        )
    atoms_by_source = {str(atom["source_id"]): atom for atom in atoms}
    if len(atoms_by_source) != len(atoms):
        raise ValueError("Duplicate xyzrn source atom IDs")
    features, names, elements, owners, name_indices = [], [], [], [], []
    source_distances = []
    for point_index, source_atom_id in enumerate(source_atom_ids):
        try:
            atom = atoms_by_source[str(source_atom_id)]
        except KeyError as error:
            raise ValueError(
                f"MSMS source atom {source_atom_id!r} is absent from protonated xyzrn"
            ) from error
        element = atom["element"] if atom["element"] in ELEMENT_ORDER[:-1] else "other"
        base = atom["base"] if atom["base"] in BASE_ORDER else None
        region = atom_region(atom["name"], base)
        donors = CANONICAL_DONORS.get(base, ())
        acceptors = set(CANONICAL_ACCEPTORS.get(base, ())) | set(BACKBONE_ACCEPTORS)
        strong_donor = atom["name"] in donors and not atom["name"].startswith("C")
        weak_donor = atom["name"] in donors and atom["name"].startswith("C")
        acceptor = atom["name"] in acceptors
        row = (
            [float(element == value) for value in ELEMENT_ORDER]
            + [float(base == value) for value in BASE_ORDER]
            + [float(base is None)]
            + [float(region == value) for value in ATOM_REGIONS]
            + [float(strong_donor), float(weak_donor), float(acceptor)]
            + [
                float(np.linalg.norm(vertices[point_index] - atom["coordinate"])) / 4.0,
                float(curvature[point_index]),
            ]
        )
        if len(row) != 21:
            raise AssertionError(f"RNA surface chemistry dimension changed: {len(row)}")
        features.append(row)
        names.append(atom["name"])
        elements.append(atom["element"])
        owners.append(atom["owner"])
        name_indices.append(atom_name_index(atom["name"]))
        source_distances.append(float(np.linalg.norm(vertices[point_index] - atom["coordinate"])))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(name_indices, dtype=np.int64),
        np.asarray(owners, dtype=np.int64),
        np.asarray(names),
        np.asarray(elements),
        np.asarray(source_distances, dtype=np.float32),
    )


def process_sample(sample, expected, identifiers, residue_atoms, alignment_status, args):
    record_id = sample["sample_id"]
    residue_keys = [
        (
            str(item["chain"]),
            int(item["component_number"]),
            str(item["insertion_code"] or ""),
        )
        for item in identifiers
    ]
    if len(set(residue_keys)) != len(residue_keys):
        raise ValueError(f"Duplicate aligned residue keys for {record_id}")
    residue_key_to_index = {key: index for index, key in enumerate(residue_keys)}
    with tempfile.TemporaryDirectory(prefix=f"surface_{record_id}_") as temporary:
        work = Path(temporary)
        receptor = work / "rna_receptor.pdb"
        protonated = work / "rna_protonated.pdb"
        xyzrn = work / "rna.xyzrn"
        msms_root = work / "rna_msms"
        write_rna_receptor(receptor, identifiers, residue_atoms)
        protonate(args.reduce_bin, receptor, protonated, work)
        atoms = extract_xyzrn(protonated, xyzrn, residue_key_to_index)
        msms_command = [
                str(args.msms_bin),
                "-density", "3", "-hdensity", "3", "-probe", "1.5",
                "-if", str(xyzrn), "-of", str(msms_root), "-af", str(msms_root),
            ]
        if args.msms_all_components:
            msms_command.append("-all_components")
        completed = run_checked(msms_command)
        if args.msms_all_components:
            vertices, faces, source_ids, msms_component_count = read_msms_all_components(msms_root)
        else:
            vertices, faces, source_ids = read_msms(msms_root)
            msms_component_count = 1
        raw_mesh = work / "raw_mesh.npz"
        fixed_mesh = work / "fixed_mesh.npz"
        np.savez_compressed(
            raw_mesh,
            vertices=vertices,
            faces=faces,
            source_ids=source_ids,
        )
        run_checked(
            [
                str(args.surface_python), str(PYMESH_HELPER),
                "--input", str(raw_mesh), "--output", str(fixed_mesh),
            ]
        )
        with np.load(fixed_mesh, allow_pickle=False) as payload:
            fixed_vertices = np.asarray(payload["vertices"], dtype=np.float64)
            fixed_faces = np.asarray(payload["faces"], dtype=np.int64)
            mean_curvature = np.asarray(payload["mean_curvature"], dtype=np.float64)
            gaussian_curvature = np.asarray(payload["gaussian_curvature"], dtype=np.float64)
            fixed_source_ids = np.asarray(payload["source_ids"]).astype(str)

    normals = compute_normals(fixed_vertices, fixed_faces)
    radicand = mean_curvature ** 2 - gaussian_curvature
    radicand[radicand <= 0] = 1e-8
    principal_1 = mean_curvature + np.sqrt(radicand)
    principal_2 = mean_curvature - np.sqrt(radicand)
    denominator = principal_1 - principal_2
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError(f"Degenerate PyMesh curvature for {record_id}")
    curvature = np.arctan((principal_1 + principal_2) / denominator) * (2.0 / np.pi)
    board = board_feature(fixed_vertices, normals)
    frames = tangent_frames(fixed_vertices, normals, board)
    chemistry, name_indices, owners, names, elements, source_distance = point_chemistry(
        fixed_vertices, curvature, atoms, fixed_source_ids
    )
    node_count = len(expected)
    point_counts = np.bincount(owners, minlength=node_count)
    output = args.output / f"{record_id}.npz"
    np.savez_compressed(
        output,
        surface_xyz=fixed_vertices.astype(np.float32),
        surface_faces=fixed_faces,
        surface_normal=normals.astype(np.float32),
        surface_nuv=frames.astype(np.float32),
        curvature=curvature.astype(np.float32),
        board=board.astype(np.float32),
        chemical_features=chemistry,
        atom_name_index=name_indices,
        source_atom_id=fixed_source_ids,
        source_atom_name=names,
        source_atom_element=elements,
        source_atom_distance=source_distance,
        surface_atom_assignment=np.asarray("msms_source_atom"),
        owner_residue=owners,
        residue_keys=np.asarray([f"{c}:{n}:{i}" for c, n, i in residue_keys]),
        node_count=np.asarray(node_count, dtype=np.int64),
        ligand_coordinate_used=np.asarray(False),
    )
    return {
        "split": sample["split"],
        "sample_id": record_id,
        "alignment_status": alignment_status,
        "node_count": node_count,
        "protonated_atom_count": len(atoms),
        "surface_point_count": int(fixed_vertices.shape[0]),
        "surface_face_count": int(fixed_faces.shape[0]),
        "residues_with_surface": int(np.sum(point_counts > 0)),
        "residue_surface_coverage": float(np.mean(point_counts > 0)),
        "points_per_residue_min": int(np.min(point_counts)),
        "points_per_residue_max": int(np.max(point_counts)),
        "points_per_residue_mean": float(np.mean(point_counts)),
        "source_atom_distance_max": float(np.max(source_distance)),
        "msms_terminated_normally": "MSMS terminated normally" in completed.stdout.decode(
            "utf-8", errors="replace"
        ),
        "msms_all_components": bool(args.msms_all_components),
        "msms_component_count": int(msms_component_count),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--msms-bin", type=Path, required=True)
    parser.add_argument(
        "--msms-all-components",
        action="store_true",
        help="Request every disconnected MSMS surface component (needed for some multi-chain records).",
    )
    parser.add_argument("--reduce-bin", type=Path, required=True)
    parser.add_argument("--surface-python", type=Path, required=True)
    parser.add_argument("--record-id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for executable in (args.msms_bin, args.reduce_bin, args.surface_python):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for sample, expected, identifiers, atoms, alignment_status in aligned_samples(args.manifest):
        if args.record_id and sample["sample_id"] != args.record_id:
            continue
        cache_path = args.output / f"{sample['sample_id']}.npz"
        if cache_path.exists() and not args.overwrite:
            continue
        report = process_sample(
            sample, expected, identifiers, atoms, alignment_status, args
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    manifest_samples = read_rows(args.manifest / "samples.tsv")
    cached_reports = []
    missing_records = []
    split_summary = {}
    for sample in manifest_samples:
        cache_path = args.output / f"{sample['sample_id']}.npz"
        if not cache_path.is_file():
            missing_records.append(sample["sample_id"])
            continue
        with np.load(cache_path, allow_pickle=False) as payload:
            xyz = np.asarray(payload["surface_xyz"])
            faces = np.asarray(payload["surface_faces"])
            owner = np.asarray(payload["owner_residue"])
            node_count = int(np.asarray(payload["node_count"]).item())
            ligand_used = bool(np.asarray(payload["ligand_coordinate_used"]).item())
            assignment = (
                str(np.asarray(payload["surface_atom_assignment"]).item())
                if "surface_atom_assignment" in payload
                else "legacy_nearest_atom"
            )
        counts = np.bincount(owner, minlength=node_count)
        cached_reports.append(
            {
                "split": sample["split"],
                "sample_id": sample["sample_id"],
                "node_count": node_count,
                "surface_point_count": int(xyz.shape[0]),
                "surface_face_count": int(faces.shape[0]),
                "residues_with_surface": int(np.sum(counts > 0)),
                "residue_surface_coverage": float(np.mean(counts > 0)),
                "ligand_coordinate_used": ligand_used,
                "surface_atom_assignment": assignment,
            }
        )
        summary = split_summary.setdefault(
            sample["split"],
            {"rna_count": 0, "residue_count": 0, "surface_points": 0},
        )
        summary["rna_count"] += 1
        summary["residue_count"] += node_count
        summary["surface_points"] += int(xyz.shape[0])
    full_pass = (
        len(cached_reports) == len(manifest_samples)
        and not missing_records
        and all(not row["ligand_coordinate_used"] for row in cached_reports)
        and all(row["residue_surface_coverage"] > 0 for row in cached_reports)
        and all(row["surface_atom_assignment"] == "msms_source_atom" for row in cached_reports)
    )
    audit = {
        "status": "PASS" if full_pass else "PARTIAL",
        "geobind_commit": "f8e336b8239fecf655d5a306ae56f887b1cf146e",
        "pipeline": "Reduce -Trim/-HIS -> MSMS density=3 hdensity=3 probe=1.5 with .vert source atom IDs -> PyMesh 0.3 fix_mesh resolution=1.2 inheriting nearest raw-MSMS vertex source IDs -> GeoBind normals/curvature/BOARD frame",
        "surface_atom_assignment": "msms_source_atom",
        "ligand_coordinates_used": False,
        "msms_binary": str(args.msms_bin.resolve()),
        "msms_sha256": sha256(args.msms_bin),
        "reduce_binary": str(args.reduce_bin.resolve()),
        "reduce_sha256": sha256(args.reduce_bin),
        "surface_python": str(args.surface_python.resolve()),
        "atom_name_vocabulary": ATOM_NAME_VOCAB,
        "chemical_feature_dim": 21,
        "record_count": len(cached_reports),
        "expected_record_count": len(manifest_samples),
        "missing_records": missing_records,
        "splits": split_summary,
        "cached_records": cached_reports,
        "records_generated_this_run": reports,
    }
    (args.output / "SURFACE_CACHE_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

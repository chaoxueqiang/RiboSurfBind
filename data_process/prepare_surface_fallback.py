#!/usr/bin/env python3
"""Build one multichain Surface cache by processing chains separately.

This is a narrow fallback for records on which MSMS ``-all_components``
segfaults.  Each chain uses the unchanged Surface pipeline; resulting point
clouds are concatenated and local owners are mapped back to the official
global node indices.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


ARRAY_KEYS = (
    "surface_xyz", "surface_normal", "surface_nuv", "curvature", "board",
    "chemical_features", "atom_name_index", "source_atom_id",
    "source_atom_name", "source_atom_element", "source_atom_distance",
)


def rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, values: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=values[0].keys(), delimiter="\t")
        writer.writeheader(); writer.writerows(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record-id", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--surface-python", required=True)
    p.add_argument("--msms-bin", type=Path, required=True)
    p.add_argument("--reduce-bin", type=Path, required=True)
    args = p.parse_args()

    samples = {r["sample_id"]: r for r in rows(args.manifest / "samples.tsv")}
    source_rows = [r for r in rows(args.manifest / "residue_mapping.tsv")
                   if r["sample_id"] == args.record_id]
    sample = samples[args.record_id]
    source_rows.sort(key=lambda r: int(r["sequence_index"]))
    chains = []
    for value in source_rows:
        if value["pdb_chain_id"] not in chains:
            chains.append(value["pdb_chain_id"])

    # Preserve inter-chain occlusion for chains that are in atomic contact.
    # Disjoint connected groups are processed independently only because the
    # legacy MSMS binary crashes when asked for all disconnected components.
    structure = PDBParser(QUIET=True).get_structure(
        args.record_id, Path(sample["pdb_path"])
    )[int(sample["model_index"])]
    coordinates = {}
    for chain in structure:
        values = [atom.coord.astype(np.float64) for residue in chain
                  if residue.id[0] == " " for atom in residue
                  if (atom.element or atom.name[:1]).upper() != "H"]
        if chain.id in chains and values:
            coordinates[chain.id] = np.asarray(values)
    parent = {chain: chain for chain in chains}
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]; value = parent[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        if left != right: parent[right] = left
    contact_cutoff = 3.6
    for i, left in enumerate(chains):
        for right in chains[i + 1:]:
            best = float("inf")
            a, b = coordinates[left], coordinates[right]
            for start in range(0, len(a), 500):
                best = min(best, float(np.sqrt(
                    ((a[start:start + 500, None] - b[None]) ** 2).sum(axis=2)
                ).min()))
            if best <= contact_cutoff:
                union(left, right)
    grouped = []
    for chain in chains:
        root = find(chain)
        match = next((group for group in grouped if find(group[0]) == root), None)
        if match is None: grouped.append([chain])
        else: match.append(chain)

    parts = []
    with tempfile.TemporaryDirectory(prefix=f"surface_chains_{args.record_id}_") as td:
        work = Path(td)
        for group_number, group in enumerate(grouped):
            selected = [r for r in source_rows if r["pdb_chain_id"] in group]
            global_indices = [int(r["sequence_index"]) for r in selected]
            local_id = f"{args.record_id}__group{group_number}"
            local_rows = []
            for local_index, row in enumerate(selected):
                copied = dict(row); copied["sample_id"] = local_id
                copied["sequence_index"] = str(local_index)
                local_rows.append(copied)
            manifest = work / f"manifest_{group_number}"; manifest.mkdir()
            local_sample = dict(sample); local_sample["sample_id"] = local_id
            write_tsv(manifest / "samples.tsv", [local_sample])
            write_tsv(manifest / "residue_mapping.tsv", local_rows)
            out = work / f"out_{group_number}"; out.mkdir()
            subprocess.run([
                args.python, "-u", str(args.prepare), "--manifest", str(manifest),
                "--output", str(out), "--msms-bin", str(args.msms_bin),
                "--reduce-bin", str(args.reduce_bin), "--surface-python", args.surface_python,
                "--record-id", local_id,
            ], check=True)
            with np.load(out / f"{local_id}.npz", allow_pickle=False) as z:
                part = {key: z[key] for key in z.files}
            part["owner_residue"] = np.asarray(
                [global_indices[int(i)] for i in part["owner_residue"]], dtype=np.int64
            )
            parts.append(part)

    merged = {key: np.concatenate([part[key] for part in parts], axis=0)
              for key in ARRAY_KEYS}
    faces, offset = [], 0
    for part in parts:
        faces.append(part["surface_faces"] + offset)
        offset += len(part["surface_xyz"])
    merged["surface_faces"] = np.concatenate(faces, axis=0)
    merged["owner_residue"] = np.concatenate([p["owner_residue"] for p in parts])
    merged["residue_keys"] = np.asarray([
        f"{r['pdb_chain_id']}:{r['pdb_residue_number']}:{r.get('pdb_insertion_code', '')}"
        for r in source_rows
    ])
    merged["node_count"] = np.asarray(len(source_rows), dtype=np.int64)
    merged["ligand_coordinate_used"] = np.asarray(False)
    merged["surface_atom_assignment"] = np.asarray("msms_source_atom")
    merged["multichain_component_policy"] = np.asarray("contact_group_fallback_after_msms_all_components_failure")
    merged["chain_contact_cutoff_angstrom"] = np.asarray(contact_cutoff, dtype=np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{args.record_id}.npz", **merged)
    owner = merged["owner_residue"]
    missing = set(range(len(source_rows))) - set(np.unique(owner).tolist())
    if missing:
        raise RuntimeError(f"{args.record_id}: per-chain fallback still misses {sorted(missing)}")
    print(f"{args.record_id}: CONTACT_GROUP_SURFACE_FALLBACK=PASS groups={grouped} points={len(owner)}")


if __name__ == "__main__":
    main()

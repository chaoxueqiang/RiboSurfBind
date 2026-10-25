"""RNA residue coordinates, heavy atoms and strict sequence alignment."""
import gzip
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser

PDB_NUCLEOTIDE_MAP = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "T": "U",
    "DA": "A",
    "DC": "C",
    "DG": "G",
    "DT": "U",
    "DU": "U",
    "RA": "A",
    "RC": "C",
    "RG": "G",
    "RU": "U",
    "PSU": "U",
    "H2U": "U",
    "5MU": "U",
    "4SU": "U",
    "OMU": "U",
    "M2G": "G",
    "7MG": "G",
    "OMG": "G",
    "1MG": "G",
    "2MG": "G",
    "YG": "G",
    "I": "A",
    "1MA": "A",
    "M2A": "A",
    "RIA": "A",
    "OMC": "C",
    "5MC": "C",
    "ADE": "A",
    "CYT": "C",
    "GUA": "G",
    "URI": "U",
    "U37": "U",
    "A23": "A",
    "IU": "A",
    "GTP": "G",
}

RNA_BASE_ATOMS = {
    "A": ("N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"),
    "G": ("N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"),
    "C": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "U": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
}

ATOM_REGIONS = ("base", "sugar", "phosphate", "other")

BASE_ORDER = ("A", "U", "C", "G")

PHOSPHATE_ATOMS = {
    "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
}


def _mean_coordinate(coordinates: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack(coordinates, axis=0).mean(axis=0).astype(np.float32)


def _residue_anchor(residue, base: str) -> np.ndarray:
    base_atoms = {
        atom_name.upper()
        for atom_name in RNA_BASE_ATOMS.get(base, ())
    }
    coordinates = [
        np.asarray(atom.get_coord(), dtype=np.float32)
        for atom in residue.get_atoms()
        if atom.get_name().strip().upper() in base_atoms
    ]
    if len(coordinates) >= 3:
        return _mean_coordinate(coordinates)

    coordinates = []
    for atom in residue.get_atoms():
        atom_name = atom.get_name().strip().upper()
        element = (atom.element or "").strip().upper()
        if element != "H" and not atom_name.startswith("H"):
            coordinates.append(
                np.asarray(atom.get_coord(), dtype=np.float32)
            )
    if len(coordinates) >= 3:
        return _mean_coordinate(coordinates)
    # HARIBOSS contains two experimentally deposited residues with only one
    # heavy atom. Preserve the node and use its available heavy-atom
    # centroid, keeping the existing residue coordinate convention unchanged.
    if coordinates:
        return _mean_coordinate(coordinates)
    raise ValueError(f"RNA residue {residue.get_resname()} has no usable atoms")

def _rna_heavy_atoms(residue, base: str) -> List[Dict[str, object]]:
    atoms = []
    for atom in residue.get_atoms():
        atom_name = atom.get_name().strip().upper()
        element = (atom.element or "").strip().upper()
        if not element:
            element = atom_name[:1]
        if element == "H" or atom_name.startswith("H"):
            continue
        atom_charge = atom.get_charge() if hasattr(atom, "get_charge") else None
        atoms.append(
            {
                "name": atom_name,
                "element": element,
                "base": base,
                # Retain the experimental residue identity for the optional
                # v2 chemistry mapper.  Existing v1/full-atom consumers ignore
                # this metadata, so their numeric tensors are unchanged.
                "residue_name": residue.get_resname().strip().upper(),
                # PQR inputs may provide partial charge.  Ordinary PDB/mmCIF
                # often do not; v2 carries an explicit validity mask and then
                # falls back only to the canonical phosphate net charge.
                "charge": (
                    float(atom_charge) if atom_charge is not None else None
                ),
                "coordinate": np.asarray(
                    atom.get_coord(), dtype=np.float32
                ),
            }
        )
    if not atoms:
        raise ValueError(
            f"RNA residue {residue.get_resname()} has no heavy atoms"
        )
    return atoms


def _chain_records(
    pdb_path: Path,
    model_index: int,
) -> List[Dict[str, object]]:
    pdb_path = Path(pdb_path)
    suffixes = [suffix.lower() for suffix in pdb_path.suffixes]
    is_cif = ".cif" in suffixes or ".mmcif" in suffixes
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    if suffixes and suffixes[-1] == ".gz":
        with gzip.open(pdb_path, "rt", encoding="utf-8") as handle:
            structure = parser.get_structure("mvrbind", handle)
    else:
        structure = parser.get_structure("mvrbind", str(pdb_path))
    models = list(structure)
    if model_index >= len(models):
        raise ValueError(
            f"{pdb_path} has {len(models)} models, requested {model_index}"
        )
    records = []
    for chain in models[model_index]:
        sequence = []
        coordinates = []
        residue_atoms = []
        residue_identifiers = []
        for residue in chain:
            base = PDB_NUCLEOTIDE_MAP.get(
                residue.get_resname().strip().upper()
            )
            if base is None:
                continue
            sequence.append(base)
            anchor = _residue_anchor(residue, base)
            coordinates.append(anchor)
            residue_atoms.append(_rna_heavy_atoms(residue, base))
            residue_identifiers.append(
                {
                    "chain": str(chain.id),
                    "component_id": residue.get_resname().strip().upper(),
                    "component_number": int(residue.id[1]),
                    "insertion_code": str(residue.id[2]).strip() or None,
                    "model": max(
                        1,
                        int(
                            getattr(
                                models[model_index],
                                "serial_num",
                                model_index + 1,
                            )
                        ),
                    ),
                    "alt_id": next(
                        (
                            str(atom.get_altloc()).strip()
                            for atom in residue.get_atoms()
                            if str(atom.get_altloc()).strip()
                        ),
                        None,
                    ),
                    "symmetry": "1_555",
                }
            )
        if sequence:
            records.append(
                {
                    "chain_id": str(chain.id),
                    "sequence": "".join(sequence).replace("T", "U"),
                    "coordinates": coordinates,
                    "residue_atoms": residue_atoms,
                    "residue_identifiers": residue_identifiers,
                }
            )
    return records

def _align_record(
    record: Dict[str, object],
    expected_sequence: str,
    source: str,
) -> Tuple[
    torch.Tensor,
    str,
    List[List[Dict[str, object]]],
]:
    observed = str(record["sequence"])
    expected = expected_sequence.replace("T", "U")
    coordinates = list(record["coordinates"])
    residue_atoms = list(record["residue_atoms"])

    if observed == expected:
        status = "exact"
    elif len(observed) == len(expected):
        mismatches = [
            index
            for index, (left, right) in enumerate(zip(observed, expected))
            if left != right
        ]
        if mismatches != [0]:
            raise ValueError(
                f"Unapproved sequence mismatch for {source}: {mismatches}"
            )
        status = "explicit_terminal_substitution_index0"
    elif len(observed) == len(expected) + 1:
        without_last = observed[:-1]
        mismatches = [
            index
            for index, (left, right) in enumerate(
                zip(without_last, expected)
            )
            if left != right
        ]
        if mismatches not in ([], [0]):
            raise ValueError(
                f"Unapproved terminal alignment for {source}: {mismatches}"
            )
        coordinates = coordinates[:-1]
        residue_atoms = residue_atoms[:-1]
        status = "explicit_extra_terminal_residue_removed"
    else:
        raise ValueError(
            f"Sequence length mismatch for {source}: "
            f"observed={len(observed)}, expected={len(expected)}"
        )

    position = torch.tensor(
        np.asarray(coordinates, dtype=np.float32),
        dtype=torch.float32,
    )
    if position.shape != (len(expected), 3):
        raise ValueError(
            f"Coordinate shape mismatch for {source}: {tuple(position.shape)}"
        )
    if len(residue_atoms) != len(expected):
        raise ValueError(
            f"Atom/residue shape mismatch for {source}: "
            f"{len(residue_atoms)} != {len(expected)}"
        )
    return (
        position,
        status,
        residue_atoms,
    )

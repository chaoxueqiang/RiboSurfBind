"""Source-preserving numerical kernels; see THIRD_PARTY_NOTICES.md."""

CANONICAL_DONORS = {
    "A": ("N6", "C2", "C8", "O2'"),
    "G": ("N1", "N2", "C8", "O2'"),
    "C": ("N4", "C5", "C6", "O2'"),
    "U": ("N3", "C5", "C6", "O2'"),
}

CANONICAL_ACCEPTORS = {
    "A": ("N1", "N3", "N7", "O2'"),
    "G": ("N3", "O6", "N7", "O2'"),
    "C": ("O2", "N3", "O2'"),
    "U": ("O2", "O4", "O2'"),
}

BACKBONE_ACCEPTORS = ("O4'", "O3'", "O5'", "OP1", "OP2")

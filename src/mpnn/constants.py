from typing import Final

AA_ALPHABET: Final[str] = "ACDEFGHIKLMNPQRSTVWYX"
CHAIN_ALPHABET: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
VOCAB_SIZE: Final[int] = len(AA_ALPHABET)

# Amino-acid alphabets and mappings
AA_1: Final[list[str]] = list(AA_ALPHABET)
AA_3: Final[list[str]] = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]
AA_1_TO_N: Final[dict[str, int]] = {a: n for n, a in enumerate(AA_1)}
AA_3_TO_N: Final[dict[str, int]] = {a: n for n, a in enumerate(AA_3)}
AA_N_TO_1: Final[dict[int, str]] = {n: a for n, a in enumerate(AA_1)}
AA_1_TO_3: Final[dict[str, str]] = {a: b for a, b in zip(AA_1, AA_3)}
AA_3_TO_1: Final[dict[str, str]] = {b: a for a, b in zip(AA_1, AA_3)}

# Atom sets
BACKBONE_MAINCHAIN_ATOMS: Final[list[str]] = ["N", "CA", "C"]
BACKBONE_ATOMS: Final[list[str]] = ["N", "CA", "C", "O"]
CA_ATOMS: Final[list[str]] = ["CA"]
BACKBONE_CB_ATOMS: Final[list[str]] = ["N", "CA", "C", "O", "CB"]

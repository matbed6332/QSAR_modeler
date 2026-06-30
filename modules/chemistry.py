"""RDKit helpers for rendering structures from SMILES."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Draw


def rdkit_available() -> bool:
    return True


@lru_cache(maxsize=512)
def smiles_to_png_bytes(smiles: str, width: int = 320, height: int = 240) -> tuple[bytes | None, str | None]:
    """Render a SMILES string to PNG bytes with RDKit."""

    if not smiles or str(smiles).strip().lower() in {"nan", "none", "<na>"}:
        return None, "No SMILES is available for this sample."

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, "RDKit could not parse this SMILES."

    image = Draw.MolToImage(mol, size=(width, height))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), None

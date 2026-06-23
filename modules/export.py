"""Download and report-export utilities."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from modules.plots import fig_to_bytes


def safe_file_stem(value: object, fallback: str = "qsar_model") -> str:
    """Return a filesystem-friendly stem for exported files."""

    stem = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    stem = "_".join(part for part in stem.split("_") if part).strip("._-")
    return stem or fallback


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframes_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31].replace("/", "-").replace("\\", "-") or "Sheet"
            pd.DataFrame(df).to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.getvalue()


def figures_to_zip_bytes(figures: dict[str, object], fmt: str = "png") -> bytes:
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, fig in figures.items():
            safe_name = safe_file_stem(str(name).lower())
            archive.writestr(f"{safe_name}.{fmt}", fig_to_bytes(fig, fmt=fmt))
    output.seek(0)
    return output.getvalue()


def list_to_frame(values: list[str], column: str = "descriptor") -> pd.DataFrame:
    return pd.DataFrame({column: values})


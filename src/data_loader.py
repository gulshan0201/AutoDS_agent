from pathlib import Path
import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".parquet"
}


def load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".csv":
        df = pd.read_csv(path)

    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(path)

    elif extension == ".parquet":
        df = pd.read_parquet(path)

    else:
        raise ValueError("Unsupported dataset format.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df
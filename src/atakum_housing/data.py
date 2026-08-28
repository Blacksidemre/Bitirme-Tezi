"""Data loading, validation and listing-level de-duplication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Fiyat (TL)"
LISTING_ID = "İlan No"
LISTING_DATE = "İlan Tarihi"

NUMERIC_FEATURES = [
    "Brüt m²",
    "Net m²",
    "Banyo Sayısı",
    "Bina Yaşı Ortalama",
    "Bulunduğu Kat (Dönüştürülmüş)",
    "Oda Sayısı Numeric",
    "Kat Sayısı Numeric",
    "Aidat (TL) Numeric",
    "İlan Günü",
]

CATEGORICAL_FEATURES = [
    "Mahalle",
    "Isıtma",
    "Mutfak",
    "Balkon",
    "Asansör",
    "Otopark",
    "Eşyalı",
    "Kullanım Durumu",
    "Site İçerisinde",
    "Krediye Uygun",
    "Tapu Durumu",
    "Kimden",
    "Takas",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TURKISH_MONTHS = {
    "Ocak": "January",
    "Şubat": "February",
    "Mart": "March",
    "Nisan": "April",
    "Mayıs": "May",
    "Haziran": "June",
    "Temmuz": "July",
    "Ağustos": "August",
    "Eylül": "September",
    "Ekim": "October",
    "Kasım": "November",
    "Aralık": "December",
}


@dataclass(frozen=True)
class DataAudit:
    """Counts used to document the complete sample lineage."""

    reported_raw_rows: int
    previously_removed_rows: int
    repository_rows: int
    exact_duplicate_rows: int
    malformed_rows: int
    structurally_valid_rows: int
    unique_listing_ids: int
    repeated_snapshot_rows: int
    listings_with_price_change: int
    latest_snapshot_rows: int
    gross_below_net_rows: int
    date_min: str
    date_max: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass
class PreparedData:
    """Validated snapshots and the latest observation for every listing."""

    snapshots: pd.DataFrame
    latest: pd.DataFrame
    rejected: pd.DataFrame
    audit: DataAudit


def parse_turkish_dates(values: pd.Series) -> pd.Series:
    """Parse Turkish long-form dates and native Excel date values."""

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    native_mask = values.map(lambda value: isinstance(value, (date, datetime, pd.Timestamp)))
    if native_mask.any():
        parsed.loc[native_mask] = pd.to_datetime(values.loc[native_mask], errors="coerce")

    text = values.astype("string").str.strip()
    for turkish, english in TURKISH_MONTHS.items():
        text = text.str.replace(turkish, english, regex=False)

    text_dates = pd.to_datetime(text, format="%d %B %Y", errors="coerce")
    return parsed.fillna(text_dates)


def normalize_listing_ids(values: pd.Series) -> pd.Series:
    """Return stable string identifiers without Excel's trailing decimal."""

    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _numeric_aidat(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype("string")
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_dataset(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Load the project workbook without mutating the source file."""

    return pd.read_excel(Path(path), sheet_name=sheet_name, engine="openpyxl")


def prepare_dataset(
    raw: pd.DataFrame,
    *,
    reported_raw_rows: int = 2_836,
) -> PreparedData:
    """Validate records and retain one latest snapshot per listing for modelling.

    The repository workbook is already the thesis' cleaned dataset. No second
    outlier-deletion pass is applied here. Only structurally malformed records are
    rejected, and repeated listing snapshots are resolved by keeping the latest date.
    """

    required = {
        TARGET,
        LISTING_ID,
        LISTING_DATE,
        "Mahalle",
        "Brüt m²",
        "Net m²",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Eksik zorunlu sütunlar: {', '.join(missing)}")

    data = raw.copy()
    data["_Kaynak Satır"] = np.arange(2, len(data) + 2)
    data["İlan Kimliği"] = normalize_listing_ids(data[LISTING_ID])
    data["İlan Tarihi Parsed"] = parse_turkish_dates(data[LISTING_DATE])

    numeric_columns = [
        TARGET,
        "Brüt m²",
        "Net m²",
        "Banyo Sayısı",
        "Bina Yaşı Ortalama",
        "Bulunduğu Kat (Dönüştürülmüş)",
        "Oda Sayısı Numeric",
        "Kat Sayısı Numeric",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["Aidat (TL) Numeric"] = _numeric_aidat(data["Aidat (TL)"])

    for column in CATEGORICAL_FEATURES:
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA})
            .fillna("Belirtilmemiş")
        )

    reasons = pd.Series("", index=data.index, dtype="string")
    valid_id = data["İlan Kimliği"].str.fullmatch(r"\d{8,12}", na=False)
    reasons = reasons.mask(~valid_id, reasons + "geçersiz ilan kimliği; ")
    reasons = reasons.mask(data["İlan Tarihi Parsed"].isna(), reasons + "geçersiz ilan tarihi; ")
    reasons = reasons.mask(data[TARGET].isna() | data[TARGET].le(0), reasons + "geçersiz fiyat; ")
    reasons = reasons.mask(
        data["Brüt m²"].isna() | data["Brüt m²"].le(0),
        reasons + "geçersiz brüt alan; ",
    )
    reasons = reasons.mask(
        data["Net m²"].isna() | data["Net m²"].le(0),
        reasons + "geçersiz net alan; ",
    )
    invalid = reasons.ne("")
    data["Reddetme Nedeni"] = reasons.str.removesuffix("; ")

    rejected = data.loc[invalid].copy()
    snapshots = data.loc[~invalid].copy()
    snapshots["İlan Günü"] = (
        snapshots["İlan Tarihi Parsed"] - snapshots["İlan Tarihi Parsed"].min()
    ).dt.days.astype(float)
    snapshots["Brüt m² Başına Fiyat"] = snapshots[TARGET] / snapshots["Brüt m²"]

    snapshots = snapshots.sort_values(["İlan Tarihi Parsed", "_Kaynak Satır"], kind="stable")
    latest = snapshots.drop_duplicates("İlan Kimliği", keep="last").copy()

    repeated_snapshot_rows = int(snapshots["İlan Kimliği"].duplicated(keep=False).sum())
    listings_with_price_change = int(
        snapshots.groupby("İlan Kimliği")[TARGET].nunique().gt(1).sum()
    )

    audit = DataAudit(
        reported_raw_rows=reported_raw_rows,
        previously_removed_rows=max(reported_raw_rows - len(raw), 0),
        repository_rows=len(raw),
        exact_duplicate_rows=int(raw.duplicated().sum()),
        malformed_rows=len(rejected),
        structurally_valid_rows=len(snapshots),
        unique_listing_ids=int(snapshots["İlan Kimliği"].nunique()),
        repeated_snapshot_rows=repeated_snapshot_rows,
        listings_with_price_change=listings_with_price_change,
        latest_snapshot_rows=len(latest),
        gross_below_net_rows=int((snapshots["Brüt m²"] < snapshots["Net m²"]).sum()),
        date_min=snapshots["İlan Tarihi Parsed"].min().date().isoformat(),
        date_max=snapshots["İlan Tarihi Parsed"].max().date().isoformat(),
    )

    return PreparedData(
        snapshots=snapshots.reset_index(drop=True),
        latest=latest.reset_index(drop=True),
        rejected=rejected.reset_index(drop=True),
        audit=audit,
    )

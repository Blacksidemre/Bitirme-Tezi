from __future__ import annotations

from datetime import datetime

import pandas as pd

from atakum_housing.data import parse_turkish_dates, prepare_dataset


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "Fiyat (TL)": 2_000_000,
        "İlan No": 123456789,
        "İlan Tarihi": "1 Nisan 2025",
        "Mahalle": "Atakent",
        "Brüt m²": 120,
        "Net m²": 105,
        "Banyo Sayısı": 1,
        "Bina Yaşı Ortalama": 5,
        "Bulunduğu Kat (Dönüştürülmüş)": 3,
        "Oda Sayısı Numeric": 3,
        "Kat Sayısı Numeric": 8,
        "Aidat (TL)": "750 TL",
        "Isıtma": "Kombi",
        "Mutfak": "Kapalı",
        "Balkon": "Var",
        "Asansör": "Var",
        "Otopark": "Açık",
        "Eşyalı": "Hayır",
        "Kullanım Durumu": "Boş",
        "Site İçerisinde": "Evet",
        "Krediye Uygun": "Evet",
        "Tapu Durumu": "Kat Mülkiyetli",
        "Kimden": "Emlak Ofisinden",
        "Takas": "Hayır",
    }
    row.update(overrides)
    return row


def test_parse_turkish_dates_accepts_text_and_native_values() -> None:
    parsed = parse_turkish_dates(pd.Series(["19 Mayıs 2025", datetime(2025, 4, 1), "geçersiz"]))

    assert parsed.iloc[0] == pd.Timestamp("2025-05-19")
    assert parsed.iloc[1] == pd.Timestamp("2025-04-01")
    assert pd.isna(parsed.iloc[2])


def test_prepare_dataset_keeps_latest_listing_and_documents_lineage() -> None:
    first = _row()
    duplicate = first.copy()
    latest = _row(**{"İlan Tarihi": "2 Nisan 2025", "Fiyat (TL)": 2_200_000})
    second_listing = _row(**{"İlan No": 987654321, "Mahalle": "Denizevleri"})
    malformed = _row(**{"İlan No": datetime(2025, 4, 3), "İlan Tarihi": "Satılık Daire"})
    prepared = prepare_dataset(
        pd.DataFrame([first, duplicate, latest, second_listing, malformed]),
        reported_raw_rows=10,
    )

    assert prepared.audit.repository_rows == 5
    assert prepared.audit.previously_removed_rows == 5
    assert prepared.audit.exact_duplicate_rows == 1
    assert prepared.audit.malformed_rows == 1
    assert prepared.audit.structurally_valid_rows == 4
    assert prepared.audit.unique_listing_ids == 2
    assert prepared.audit.repeated_snapshot_rows == 3
    assert prepared.audit.listings_with_price_change == 1
    assert prepared.audit.latest_snapshot_rows == 2
    selected = prepared.latest.loc[
        prepared.latest["İlan Kimliği"].eq("123456789"), "Fiyat (TL)"
    ].item()
    assert selected == 2_200_000
    assert "geçersiz ilan kimliği" in prepared.rejected["Reddetme Nedeni"].item()

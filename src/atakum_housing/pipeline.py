"""End-to-end reproducible analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import load_dataset, prepare_dataset
from .modeling import evaluate_models
from .statistics import (
    binary_group_tests,
    descriptive_statistics,
    hedonic_regression,
    neighborhood_kruskal,
    neighborhood_summary,
    spearman_correlation,
)
from .visuals import create_all_figures


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"JSON'a dönüştürülemeyen tür: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _fmt_tl(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") + " TL"


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _build_markdown_summary(summary: dict[str, Any]) -> str:
    audit = summary["data_audit"]
    model = summary["best_model"]
    test = model["locked_test"]
    return f"""# Analiz Özeti

Bu rapor, `veriseti.xlsx` dosyasından otomatik olarak üretilmiştir.

## Veri kapsamı

- Çalışmada bildirilen ham örneklem: **{_fmt_int(audit["reported_raw_rows"])}** kayıt
- Repodaki temizlenmiş veri: **{_fmt_int(audit["repository_rows"])}** kayıt
- Yapısal olarak geçerli görüntü: **{_fmt_int(audit["structurally_valid_rows"])}** kayıt
- Tekil ilanların son görüntüsü: **{_fmt_int(audit["latest_snapshot_rows"])}** ilan
- İncelenen tarih aralığı: **{audit["date_min"]} - {audit["date_max"]}**

## Tahmin modeli

- Seçilen model: **{model["name"]}**
- Kilitli test MAE: **{_fmt_tl(test["test_mae"])}**
- Kilitli test RMSE: **{_fmt_tl(test["test_rmse"])}**
- Kilitli test R²: **{test["test_r2"]:.3f}**
- MAE için bootstrap %95 güven aralığı:
  **{_fmt_tl(model["mae_ci_95_low"])} - {_fmt_tl(model["mae_ci_95_high"])}**

## Yorum sınırı

Sonuçlar gerçekleşen satış bedellerini değil, veri setindeki ilan fiyatlarını açıklar.
Model çıktıları otomatik ekspertiz veya yatırım tavsiyesi değildir. Aynı ilanların farklı
tarihlerdeki görüntüleri ana model değerlendirmesinde tekilleştirilmiştir.
"""


def run_pipeline(
    data_path: str | Path = "veriseti.xlsx",
    output_dir: str | Path = "outputs/latest",
    *,
    reported_raw_rows: int = 2_836,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run cleaning, statistics, modelling, visualisation and artifact export."""

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    for directory in (output_dir, tables_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(data_path)
    prepared = prepare_dataset(raw, reported_raw_rows=reported_raw_rows)
    latest = prepared.latest

    descriptive = descriptive_statistics(latest)
    neighborhoods = neighborhood_summary(latest)
    correlation, correlation_p = spearman_correlation(latest)
    binary_tests = binary_group_tests(latest)
    kruskal = neighborhood_kruskal(latest)
    hedonic = hedonic_regression(latest)
    evaluation = evaluate_models(
        latest,
        prepared.snapshots,
        random_state=random_state,
    )

    prepared.rejected.to_csv(tables_dir / "reddedilen_kayitlar.csv", index=False)
    latest.to_csv(tables_dir / "tekil_son_ilanlar.csv", index=False)
    descriptive.to_csv(tables_dir / "tanimlayici_istatistikler.csv", index=False)
    neighborhoods.to_csv(tables_dir / "mahalle_ozeti.csv", index=False)
    correlation.to_csv(tables_dir / "spearman_korelasyon.csv")
    correlation_p.to_csv(tables_dir / "spearman_p_degerleri.csv")
    binary_tests.to_csv(tables_dir / "ikili_grup_testleri.csv", index=False)
    hedonic.coefficients.to_csv(tables_dir / "hedonik_katsayilar_hc3.csv", index=False)
    evaluation.comparison.to_csv(tables_dir / "model_karsilastirma.csv", index=False)
    evaluation.predictions.to_csv(tables_dir / "test_tahminleri.csv", index=False)
    evaluation.feature_importance.to_csv(tables_dir / "ozellik_onemi.csv", index=False)
    evaluation.sensitivity.to_csv(tables_dir / "protokol_duyarliligi.csv", index=False)

    _write_json(output_dir / "veri_denetimi.json", prepared.audit.to_dict())
    _write_json(output_dir / "kruskal_wallis.json", kruskal)
    _write_json(output_dir / "hedonik_model_ozeti.json", hedonic.summary)
    _write_json(output_dir / "model_bolme_ozeti.json", evaluation.split_summary)
    joblib.dump(evaluation.best_estimator, models_dir / "en_iyi_model.joblib", compress=3)

    figure_paths = create_all_figures(
        data=latest,
        audit=prepared.audit,
        neighborhood=neighborhoods,
        correlation=correlation,
        model_comparison=evaluation.comparison,
        predictions=evaluation.predictions,
        feature_importance=evaluation.feature_importance,
        sensitivity=evaluation.sensitivity,
        best_model_name=evaluation.best_model_name,
        output_dir=figures_dir,
    )

    best_row = evaluation.comparison.loc[
        evaluation.comparison["model"].eq(evaluation.best_model_name)
    ].iloc[0]
    summary = {
        "data_audit": prepared.audit.to_dict(),
        "price": {
            "mean": float(latest["Fiyat (TL)"].mean()),
            "median": float(latest["Fiyat (TL)"].median()),
            "minimum": float(latest["Fiyat (TL)"].min()),
            "maximum": float(latest["Fiyat (TL)"].max()),
            "median_price_per_gross_m2": float(latest["Brüt m² Başına Fiyat"].median()),
        },
        "best_model": {
            "name": evaluation.best_model_name,
            "locked_test": {
                "test_mae": float(best_row["test_mae"]),
                "test_rmse": float(best_row["test_rmse"]),
                "test_r2": float(best_row["test_r2"]),
                "test_mape": float(best_row["test_mape"]),
            },
            "cross_validation": {
                "cv_mae_mean": float(best_row["cv_mae_mean"]),
                "cv_mae_std": float(best_row["cv_mae_std"]),
                "cv_r2_mean": float(best_row["cv_r2_mean"]),
                "cv_r2_std": float(best_row["cv_r2_std"]),
            },
            "mae_ci_95_low": evaluation.split_summary["mae_ci_95_low"],
            "mae_ci_95_high": evaluation.split_summary["mae_ci_95_high"],
        },
        "neighborhood_test": kruskal,
        "hedonic_model": hedonic.summary,
        "outputs": {
            "tables": str(tables_dir),
            "figures": [str(path) for path in figure_paths],
            "model": str(models_dir / "en_iyi_model.joblib"),
        },
    }
    _write_json(output_dir / "analiz_ozeti.json", summary)
    (output_dir / "ANALIZ_OZETI.md").write_text(_build_markdown_summary(summary), encoding="utf-8")
    return summary

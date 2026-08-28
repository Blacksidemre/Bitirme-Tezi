"""Publication-ready static figures shared by the thesis and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import TARGET, DataAudit

NAVY = "#102A43"
BLUE = "#2F80ED"
TEAL = "#12B3A8"
AMBER = "#F2B84B"
RED = "#D64545"
SLATE = "#627D98"
LIGHT = "#EAF2F8"


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": "#BCCCDC",
            "grid.color": "#D9E2EC",
            "grid.alpha": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_data_flow(audit: DataAudit, path: Path) -> None:
    stages = [
        "Ham çalışma\nörneklemi",
        "Repo temiz\nverisi",
        "Yapısal geçerli\nkayıt",
        "Tekil son\nilan görüntüsü",
    ]
    values = [
        audit.reported_raw_rows,
        audit.repository_rows,
        audit.structurally_valid_rows,
        audit.latest_snapshot_rows,
    ]
    colors = [NAVY, BLUE, TEAL, AMBER]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1], height=0.56)
    for bar, value in zip(bars, values[::-1], strict=True):
        ax.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}".replace(",", "."),
            va="center",
            fontweight="bold",
            color=NAVY,
        )
    ax.set_title("Örneklem akışı ve analiz birimi", loc="left")
    ax.set_xlabel("Kayıt sayısı")
    ax.set_xlim(0, max(values) * 1.16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.text(
        0.01,
        -0.01,
        "Not: Repo verisi daha önce temizlenmiş çalışma setidir; modelleme aynı ilanın "
        "tekrarlarından etkilenmemesi için son görüntüyü kullanır.",
        fontsize=9,
        color=SLATE,
    )
    _save(fig, path)


def plot_price_distribution(data: pd.DataFrame, path: Path) -> None:
    prices_million = data[TARGET] / 1_000_000
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.histplot(prices_million, bins=28, kde=True, color=BLUE, ax=axes[0])
    axes[0].axvline(prices_million.median(), color=RED, linestyle="--", linewidth=2)
    axes[0].set_title("İlan fiyatı dağılımı", loc="left")
    axes[0].set_xlabel("Fiyat (milyon TL)")
    axes[0].set_ylabel("İlan sayısı")

    sns.boxplot(x=prices_million, color=TEAL, width=0.38, ax=axes[1])
    axes[1].set_title("Merkez ve yayılım", loc="left")
    axes[1].set_xlabel("Fiyat (milyon TL)")
    axes[1].set_ylabel("")
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Atakum satılık konut ilanlarının fiyat profili", x=0.06, ha="left", y=1.03)
    _save(fig, path)


def plot_neighborhoods(summary: pd.DataFrame, path: Path, top_n: int = 12) -> None:
    view = summary.nlargest(top_n, "listing_count").sort_values("median_price")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    bars = ax.barh(view["Mahalle"], view["median_price"] / 1_000_000, color=TEAL)
    for bar, count in zip(bars, view["listing_count"], strict=True):
        ax.text(
            bar.get_width() + 0.04,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(count)}",
            va="center",
            fontsize=8.5,
            color=SLATE,
        )
    ax.set_title("Yeterli gözleme sahip başlıca mahallelerde medyan fiyat", loc="left")
    ax.set_xlabel("Medyan ilan fiyatı (milyon TL)")
    ax.set_ylabel("")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, view["median_price"].max() / 1_000_000 * 1.18)
    _save(fig, path)


def plot_room_price_profiles(data: pd.DataFrame, path: Path) -> None:
    counts = data["Oda Sayısı"].value_counts()
    eligible = counts.loc[counts >= 10].index
    frame = data.loc[data["Oda Sayısı"].isin(eligible)].copy()
    order = frame.groupby("Oda Sayısı")[TARGET].median().sort_values().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.boxplot(
        data=frame,
        x="Oda Sayısı",
        y=TARGET,
        order=order,
        color=TEAL,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_title("Oda tipine göre ilan fiyatı", loc="left")
    axes[0].set_xlabel("Oda tipi")
    axes[0].set_ylabel("İlan fiyatı (TL)")

    per_m2 = frame.groupby("Oda Sayısı")["Brüt m² Başına Fiyat"].median().reindex(order)
    axes[1].bar(per_m2.index, per_m2.values / 1_000, color=BLUE)
    axes[1].set_title("Oda tipine göre medyan brüt m² fiyatı", loc="left")
    axes[1].set_xlabel("Oda tipi")
    axes[1].set_ylabel("Bin TL / brüt m²")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_correlation(correlation: pd.DataFrame, path: Path) -> None:
    labels = {
        TARGET: "Fiyat",
        "Brüt m²": "Brüt m²",
        "Net m²": "Net m²",
        "Banyo Sayısı": "Banyo",
        "Bina Yaşı Ortalama": "Bina yaşı",
        "Bulunduğu Kat (Dönüştürülmüş)": "Bul. kat",
        "Oda Sayısı Numeric": "Oda",
        "Kat Sayısı Numeric": "Kat sayısı",
        "Aidat (TL) Numeric": "Aidat",
    }
    view = correlation.rename(index=labels, columns=labels)
    mask = np.triu(np.ones_like(view, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9.3, 7.4))
    sns.heatmap(
        view,
        mask=mask,
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.6,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.78},
        ax=ax,
    )
    ax.set_title("Sayısal değişkenler arasındaki sıra korelasyonları", loc="left")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, path)


def plot_model_performance(comparison: pd.DataFrame, path: Path) -> None:
    view = comparison.sort_values("cv_mae_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    axes[0].barh(view["model"], view["cv_mae_mean"] / 1_000, color=BLUE)
    axes[0].errorbar(
        view["cv_mae_mean"] / 1_000,
        np.arange(len(view)),
        xerr=view["cv_mae_std"] / 1_000,
        fmt="none",
        ecolor=NAVY,
        capsize=3,
    )
    axes[0].set_title("5 katlı çapraz doğrulama MAE", loc="left")
    axes[0].set_xlabel("Ortalama mutlak hata (bin TL, düşük daha iyi)")
    axes[0].set_ylabel("")

    axes[1].barh(view["model"], view["test_r2"], color=TEAL)
    axes[1].axvline(0, color=NAVY, linewidth=0.8)
    axes[1].set_title("Kilitli test seti R²", loc="left")
    axes[1].set_xlabel("R² (yüksek daha iyi)")
    axes[1].set_ylabel("")
    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="y", visible=False)
    fig.suptitle("Model karşılaştırması", x=0.05, ha="left", y=1.02)
    _save(fig, path)


def plot_actual_vs_predicted(predictions: pd.DataFrame, model_name: str, path: Path) -> None:
    view = predictions.loc[predictions["protocol"].eq("locked_random_holdout")]
    actual = view[TARGET] / 1_000_000
    predicted = view["prediction"] / 1_000_000
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(actual, predicted, alpha=0.55, s=34, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.plot([lower, upper], [lower, upper], color=RED, linestyle="--", linewidth=2)
    ax.set_title(f"Gerçek ve tahmin edilen fiyatlar\n{model_name}", loc="left")
    ax.set_xlabel("Gerçek ilan fiyatı (milyon TL)")
    ax.set_ylabel("Tahmin edilen fiyat (milyon TL)")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    _save(fig, path)


def plot_residuals(predictions: pd.DataFrame, path: Path) -> None:
    view = predictions.loc[predictions["protocol"].eq("locked_random_holdout")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(
        view["prediction"] / 1_000_000,
        view["residual"] / 1_000,
        alpha=0.55,
        s=30,
        color=TEAL,
    )
    axes[0].axhline(0, color=RED, linestyle="--")
    axes[0].set_title("Tahmine karşı artıklar", loc="left")
    axes[0].set_xlabel("Tahmin (milyon TL)")
    axes[0].set_ylabel("Artık (bin TL)")

    sns.histplot(view["residual"] / 1_000, bins=24, kde=True, color=BLUE, ax=axes[1])
    axes[1].axvline(0, color=RED, linestyle="--")
    axes[1].set_title("Artık dağılımı", loc="left")
    axes[1].set_xlabel("Artık (bin TL)")
    axes[1].set_ylabel("İlan sayısı")
    _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path, top_n: int = 12) -> None:
    view = importance.head(top_n).sort_values("mae_increase_mean")
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    ax.barh(view["feature"], view["mae_increase_mean"] / 1_000, color=AMBER)
    ax.errorbar(
        view["mae_increase_mean"] / 1_000,
        np.arange(len(view)),
        xerr=view["mae_increase_std"] / 1_000,
        fmt="none",
        ecolor=NAVY,
        capsize=3,
    )
    ax.set_title("Permütasyon önemine göre model girdileri", loc="left")
    ax.set_xlabel("Karıştırıldığında MAE artışı (bin TL)")
    ax.set_ylabel("")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_protocol_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    labels = {
        "Satır bazlı rastgele bölme (sızıntı riski)": "Satır bazlı\n(sızıntı riski)",
        "Tekil ilanlar: kilitli rastgele test": "Tekil ilan\nkilitli test",
        "Kronolojik sağlamlık kontrolü": "Kronolojik\nsağlamlık testi",
        "İlan kimliği gruplu bölme": "İlan kimliği\ngruplu test",
    }
    view = sensitivity.assign(
        chart_label=sensitivity["protocol"].map(labels).fillna(sensitivity["protocol"])
    ).sort_values("mae")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    axes[0].barh(view["chart_label"], view["mae"] / 1_000, color=[BLUE, TEAL, AMBER, RED])
    axes[0].set_title("Değerlendirme protokolüne göre MAE", loc="left")
    axes[0].set_xlabel("MAE (bin TL)")
    axes[0].set_ylabel("")
    axes[1].barh(view["chart_label"], view["r2"], color=[BLUE, TEAL, AMBER, RED])
    axes[1].set_title("Değerlendirme protokolüne göre R²", loc="left")
    axes[1].set_xlabel("R²")
    axes[1].set_ylabel("")
    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="y", visible=False)
    fig.suptitle("Veri sızıntısı ve zaman dayanıklılığı kontrolü", x=0.04, ha="left", y=1.03)
    _save(fig, path)


def create_all_figures(
    *,
    data: pd.DataFrame,
    audit: DataAudit,
    neighborhood: pd.DataFrame,
    correlation: pd.DataFrame,
    model_comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    sensitivity: pd.DataFrame,
    best_model_name: str,
    output_dir: Path,
) -> list[Path]:
    configure_style()
    figures = {
        "01_veri_akisi.png": lambda path: plot_data_flow(audit, path),
        "02_fiyat_dagilimi.png": lambda path: plot_price_distribution(data, path),
        "03_mahalle_medyan_fiyat.png": lambda path: plot_neighborhoods(neighborhood, path),
        "04_oda_fiyat_profili.png": lambda path: plot_room_price_profiles(data, path),
        "05_korelasyon_isiharitasi.png": lambda path: plot_correlation(correlation, path),
        "06_model_performansi.png": lambda path: plot_model_performance(model_comparison, path),
        "07_gercek_tahmin.png": lambda path: plot_actual_vs_predicted(
            predictions, best_model_name, path
        ),
        "08_artik_analizi.png": lambda path: plot_residuals(predictions, path),
        "09_ozellik_onemi.png": lambda path: plot_feature_importance(feature_importance, path),
        "10_protokol_duyarliligi.png": lambda path: plot_protocol_sensitivity(sensitivity, path),
    }
    paths: list[Path] = []
    for filename, builder in figures.items():
        path = output_dir / filename
        builder(path)
        paths.append(path)
    return paths

"""Descriptive and inferential statistics for the thesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .data import TARGET

DESCRIPTIVE_COLUMNS = [
    TARGET,
    "Brüt m²",
    "Net m²",
    "Banyo Sayısı",
    "Bina Yaşı Ortalama",
    "Bulunduğu Kat (Dönüştürülmüş)",
    "Oda Sayısı Numeric",
    "Kat Sayısı Numeric",
    "Aidat (TL) Numeric",
    "Brüt m² Başına Fiyat",
]


@dataclass(frozen=True)
class HedonicResult:
    coefficients: pd.DataFrame
    summary: dict[str, float | int]


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, auditable summary for the numeric variables."""

    columns = [column for column in DESCRIPTIVE_COLUMNS if column in data]
    summary = data[columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary["missing"] = data[columns].isna().sum()
    summary["variance"] = data[columns].var()
    summary["iqr"] = summary["75%"] - summary["25%"]
    summary["skewness"] = data[columns].skew()
    return (
        summary[
            [
                "count",
                "missing",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "variance",
                "iqr",
                "skewness",
            ]
        ]
        .rename_axis("variable")
        .reset_index()
    )


def neighborhood_summary(data: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """Summarize neighborhoods with enough observations for stable comparisons."""

    grouped = data.groupby("Mahalle", observed=True)
    result = grouped.agg(
        listing_count=(TARGET, "size"),
        mean_price=(TARGET, "mean"),
        median_price=(TARGET, "median"),
        price_q1=(TARGET, lambda values: values.quantile(0.25)),
        price_q3=(TARGET, lambda values: values.quantile(0.75)),
        median_price_per_gross_m2=("Brüt m² Başına Fiyat", "median"),
        median_gross_m2=("Brüt m²", "median"),
    )
    result["price_iqr"] = result["price_q3"] - result["price_q1"]
    return (
        result.loc[result["listing_count"] >= min_count]
        .sort_values("median_price", ascending=False)
        .reset_index()
    )


def spearman_correlation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rank correlations and two-sided p-values."""

    columns = [
        TARGET,
        "Brüt m²",
        "Net m²",
        "Banyo Sayısı",
        "Bina Yaşı Ortalama",
        "Bulunduğu Kat (Dönüştürülmüş)",
        "Oda Sayısı Numeric",
        "Kat Sayısı Numeric",
        "Aidat (TL) Numeric",
    ]
    frame = data[columns].apply(pd.to_numeric, errors="coerce")
    correlation = frame.corr(method="spearman")

    p_values = pd.DataFrame(np.nan, index=columns, columns=columns)
    for left_index, left in enumerate(columns):
        for right_index, right in enumerate(columns):
            if right_index < left_index:
                p_values.loc[left, right] = p_values.loc[right, left]
                continue
            pair = frame[[left, right]].dropna()
            if len(pair) < 3:
                continue
            if left == right:
                p_value = 0.0
            else:
                p_value = float(stats.spearmanr(pair[left], pair[right]).pvalue)
            p_values.loc[left, right] = p_value
            p_values.loc[right, left] = p_value
    return correlation, p_values


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction without an extra dependency."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min((total - rank) * values[index], 1.0)
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def binary_group_tests(data: pd.DataFrame) -> pd.DataFrame:
    """Run robust two-group price comparisons with effect sizes."""

    comparisons = [
        ("Site İçerisinde", "Evet", "Hayır"),
        ("Eşyalı", "Evet", "Hayır"),
        ("Balkon", "Var", "Yok"),
        ("Asansör", "Var", "Yok"),
        ("Krediye Uygun", "Evet", "Hayır"),
        ("Takas", "Evet", "Hayır"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for variable, group_a, group_b in comparisons:
        sample_a = data.loc[data[variable].eq(group_a), TARGET].dropna().to_numpy()
        sample_b = data.loc[data[variable].eq(group_b), TARGET].dropna().to_numpy()
        if len(sample_a) < 5 or len(sample_b) < 5:
            continue
        result = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        rank_biserial = (2.0 * result.statistic / (len(sample_a) * len(sample_b))) - 1.0
        rows.append(
            {
                "variable": variable,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(sample_a),
                "n_b": len(sample_b),
                "median_a": float(np.median(sample_a)),
                "median_b": float(np.median(sample_b)),
                "median_difference": float(np.median(sample_a) - np.median(sample_b)),
                "u_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "rank_biserial": float(rank_biserial),
            }
        )

    output = pd.DataFrame(rows)
    if not output.empty:
        output["p_value_holm"] = _holm_adjust(output["p_value"].tolist())
        output["significant_0_05"] = output["p_value_holm"].lt(0.05)
    return output


def neighborhood_kruskal(data: pd.DataFrame, min_count: int = 20) -> dict[str, float | int]:
    """Test whether price distributions differ across sufficiently large neighborhoods."""

    eligible = data.groupby("Mahalle").filter(lambda group: len(group) >= min_count)
    groups = [group[TARGET].to_numpy() for _, group in eligible.groupby("Mahalle")]
    if len(groups) < 2:
        return {
            "group_count": len(groups),
            "n": len(eligible),
            "h_statistic": np.nan,
            "p_value": np.nan,
            "epsilon_squared": np.nan,
        }

    result = stats.kruskal(*groups)
    n = sum(len(group) for group in groups)
    k = len(groups)
    epsilon_squared = max((float(result.statistic) - k + 1) / (n - k), 0.0)
    return {
        "group_count": k,
        "n": n,
        "h_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "epsilon_squared": float(epsilon_squared),
    }


def _collapse_rare(values: pd.Series, min_count: int) -> pd.Series:
    text = values.astype("string").fillna("Belirtilmemiş")
    common = text.value_counts().loc[lambda counts: counts >= min_count].index
    return text.where(text.isin(common), "Diğer")


def hedonic_regression(data: pd.DataFrame) -> HedonicResult:
    """Estimate a log-price hedonic model with HC3 robust standard errors."""

    frame = pd.DataFrame(
        {
            "ln_brut_m2": np.log(data["Brüt m²"].clip(lower=1)),
            "bina_yasi": data["Bina Yaşı Ortalama"],
            "banyo_sayisi": data["Banyo Sayısı"],
            "oda_sayisi": data["Oda Sayısı Numeric"],
            "bulundugu_kat": data["Bulunduğu Kat (Dönüştürülmüş)"],
            "kat_sayisi": data["Kat Sayısı Numeric"],
            "ilan_gunu": data["İlan Günü"],
            "mahalle": _collapse_rare(data["Mahalle"], 20),
            "site": data["Site İçerisinde"],
            "asansor": data["Asansör"],
            "otopark": _collapse_rare(data["Otopark"], 20),
            "isitma": _collapse_rare(data["Isıtma"], 20),
            "kimden": _collapse_rare(data["Kimden"], 20),
        }
    )
    y = np.log(data[TARGET].clip(lower=1)).astype(float)
    frame = pd.get_dummies(
        frame,
        columns=["mahalle", "site", "asansor", "otopark", "isitma", "kimden"],
        drop_first=True,
        dtype=float,
    )
    frame = frame.apply(pd.to_numeric, errors="coerce")
    complete = frame.notna().all(axis=1) & y.notna()
    frame = frame.loc[complete]
    y = y.loc[complete].to_numpy(dtype=float)

    x = frame.to_numpy(dtype=float)
    terms = ["const"] + frame.columns.tolist()
    x = np.column_stack([np.ones(len(x)), x])

    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residuals = y - fitted
    n, p = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    leverage = np.clip(leverage, 0.0, 0.999999)
    scaled_residuals = residuals / (1.0 - leverage)
    meat = x.T @ ((scaled_residuals**2)[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    t_values = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    degrees_of_freedom = max(n - p, 1)
    p_values = 2 * stats.t.sf(np.abs(t_values), degrees_of_freedom)
    critical = stats.t.ppf(0.975, degrees_of_freedom)

    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    residual_sum_squares = float(np.sum(residuals**2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / max(n - p, 1)

    coefficients = pd.DataFrame(
        {
            "term": terms,
            "coefficient": beta,
            "std_error_hc3": standard_errors,
            "t_value": t_values,
            "p_value": p_values,
            "ci_95_low": beta - critical * standard_errors,
            "ci_95_high": beta + critical * standard_errors,
        }
    )
    coefficients["percent_change_per_unit"] = np.expm1(coefficients["coefficient"]) * 100
    coefficients.loc[coefficients["term"].eq("ln_brut_m2"), "percent_change_per_unit"] = np.nan
    coefficients.loc[coefficients["term"].eq("const"), "percent_change_per_unit"] = np.nan

    summary = {
        "n": n,
        "parameter_count": p,
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "rmse_log": float(np.sqrt(np.mean(residuals**2))),
        "condition_number": float(np.linalg.cond(x)),
    }
    return HedonicResult(coefficients=coefficients, summary=summary)

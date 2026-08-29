"""Interactive portfolio dashboard for the Atakum housing-price thesis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atakum_housing.data import MODEL_FEATURES, TARGET, load_dataset, prepare_dataset
from atakum_housing.modeling import fit_dashboard_model

st.set_page_config(
    page_title="Atakum Konut Analitiği",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #102a43;
        --muted: #627d98;
        --blue: #2f80ed;
        --teal: #12b3a8;
        --amber: #f2b84b;
        --paper: #f5f8fb;
      }
      .stApp { background: var(--paper); color: var(--ink); }
      [data-testid="stSidebar"] { background: #0f2740; }
      [data-testid="stSidebar"] * { color: #eef6ff !important; }
      [data-testid="stSidebar"] .stMultiSelect span,
      [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
        color: #102a43 !important;
      }
      .block-container { padding-top: 1.2rem; max-width: 1500px; }
      .hero {
        border-radius: 22px;
        padding: 26px 30px;
        color: white;
        background:
          radial-gradient(circle at 90% 20%, rgba(18,179,168,.33), transparent 28%),
          linear-gradient(120deg, #0f2740, #164c7e 62%, #126f78);
        box-shadow: 0 14px 40px rgba(16,42,67,.16);
        margin-bottom: 18px;
      }
      .hero-kicker { letter-spacing: .14em; text-transform: uppercase; font-size: .76rem; opacity: .78; }
      .hero h1 { font-size: clamp(1.8rem, 3vw, 3.2rem); margin: 8px 0 6px; line-height: 1.02; }
      .hero p { max-width: 850px; opacity: .9; margin: 0; }
      .mini-note {
        border-left: 4px solid var(--amber);
        background: #fff8e7;
        color: #5c4510;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0 18px;
      }
      [data-testid="stMetric"] {
        background: white;
        border: 1px solid #d9e2ec;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 5px 18px rgba(16,42,67,.06);
      }
      [data-testid="stMetricLabel"] { color: var(--muted); }
      [data-testid="stMetricValue"] { color: var(--ink); }
      [data-testid="stTabs"] button { font-weight: 700; }
      .section-label { color: var(--muted); font-size: .82rem; letter-spacing: .08em; text-transform: uppercase; }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_project_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prepared = prepare_dataset(load_dataset(ROOT / "veriseti.xlsx"))
    return (
        prepared.latest,
        prepared.snapshots,
        prepared.rejected,
        prepared.audit.to_dict(),
    )


@st.cache_resource(show_spinner="Tahmin modeli hazırlanıyor...")
def load_prediction_model(latest: pd.DataFrame):
    return fit_dashboard_model(latest)


@st.cache_data(show_spinner=False)
def load_output_table(filename: str) -> pd.DataFrame:
    path = ROOT / "outputs" / "latest" / "tables" / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    path = ROOT / "outputs" / "latest" / "analiz_ozeti.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def tr_number(value: float, decimals: int = 0) -> str:
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def tr_tl(value: float, compact: bool = False) -> str:
    if compact and abs(value) >= 1_000_000:
        return f"{tr_number(value / 1_000_000, 2)} Mn TL"
    if compact and abs(value) >= 1_000:
        return f"{tr_number(value / 1_000, 0)} Bin TL"
    return f"{tr_number(value, 0)} TL"


def style_figure(figure: go.Figure, height: int = 420) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=58, b=22),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color="#102a43"),
        title_font=dict(size=18),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    figure.update_xaxes(gridcolor="#e8eef4", zerolinecolor="#bcccdc")
    figure.update_yaxes(gridcolor="#e8eef4", zerolinecolor="#bcccdc")
    return figure


latest, snapshots, rejected, audit = load_project_data()
summary = load_summary()

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Bitirme projesi • Etkileşimli analiz</div>
      <h1>Atakum Konut Analitiği</h1>
      <p>Temizlenmiş ilan verisini mahalle, fiyat, oda tipi ve yapı özellikleri üzerinden
      keşfedin; model performansını inceleyin ve tek bir konut için senaryo tahmini üretin.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Filtreler")
    st.caption("Filtreler Piyasa Görünümü sekmesini günceller.")
    selected_neighborhoods = st.multiselect(
        "Mahalle",
        options=sorted(latest["Mahalle"].unique()),
        placeholder="Tüm mahalleler",
    )
    selected_rooms = st.multiselect(
        "Oda tipi",
        options=sorted(latest["Oda Sayısı"].unique()),
        placeholder="Tüm oda tipleri",
    )
    site_filter = st.selectbox("Site durumu", ["Tümü", "Evet", "Hayır"])
    minimum_price = int(latest[TARGET].min() // 100_000 * 100_000)
    maximum_price = int(np.ceil(latest[TARGET].max() / 100_000) * 100_000)
    selected_price = st.slider(
        "Fiyat aralığı (TL)",
        minimum_price,
        maximum_price,
        (minimum_price, maximum_price),
        step=100_000,
        format="%d TL",
    )
    gross_range = st.slider(
        "Brüt alan (m²)",
        int(latest["Brüt m²"].min()),
        int(latest["Brüt m²"].max()),
        (int(latest["Brüt m²"].min()), int(latest["Brüt m²"].max())),
    )
    st.divider()
    st.caption(
        f"Veri dönemi: {audit['date_min']} - {audit['date_max']}  ·  "
        f"Tekil ilan: {tr_number(audit['latest_snapshot_rows'])}"
    )

filtered = latest.loc[
    latest[TARGET].between(*selected_price) & latest["Brüt m²"].between(*gross_range)
].copy()
if selected_neighborhoods:
    filtered = filtered.loc[filtered["Mahalle"].isin(selected_neighborhoods)]
if selected_rooms:
    filtered = filtered.loc[filtered["Oda Sayısı"].isin(selected_rooms)]
if site_filter != "Tümü":
    filtered = filtered.loc[filtered["Site İçerisinde"].eq(site_filter)]

if filtered.empty:
    st.warning("Bu filtrelerle eşleşen ilan bulunamadı. Filtreleri genişletin.")
    st.stop()

tabs = st.tabs(["Piyasa Görünümü", "Model Laboratuvarı", "Fiyat Simülatörü", "Veri Kalitesi"])

with tabs[0]:
    st.markdown(
        '<div class="section-label">Filtrelenmiş piyasa özeti</div>', unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)
    coverage = len(filtered) / len(latest) * 100
    col1.metric(
        "İlan sayısı",
        tr_number(len(filtered)),
        delta=f"Toplamın %{tr_number(coverage)}'i",
        delta_color="off",
    )
    col2.metric("Medyan fiyat", tr_tl(filtered[TARGET].median(), compact=True))
    col3.metric(
        "Medyan brüt m² fiyatı",
        tr_tl(filtered["Brüt m² Başına Fiyat"].median()),
    )
    col4.metric("Medyan brüt alan", f"{tr_number(filtered['Brüt m²'].median())} m²")

    left, right = st.columns([1.05, 0.95])
    with left:
        histogram = px.histogram(
            filtered,
            x=TARGET,
            nbins=28,
            marginal="box",
            title="İlan fiyatı dağılımı",
            color_discrete_sequence=["#2f80ed"],
            labels={TARGET: "İlan fiyatı (TL)", "count": "İlan sayısı"},
        )
        histogram.update_layout(showlegend=False)
        st.plotly_chart(style_figure(histogram), width="stretch")
    with right:
        neighborhood = (
            filtered.groupby("Mahalle")
            .agg(medyan=(TARGET, "median"), ilan=(TARGET, "size"))
            .loc[lambda frame: frame["ilan"] >= 5]
            .nlargest(12, "ilan")
            .sort_values("medyan")
            .reset_index()
        )
        bar = px.bar(
            neighborhood,
            x="medyan",
            y="Mahalle",
            orientation="h",
            text="ilan",
            title="Mahalle bazında medyan fiyat",
            color_discrete_sequence=["#12b3a8"],
            labels={"medyan": "Medyan fiyat (TL)", "Mahalle": ""},
        )
        bar.update_traces(texttemplate="n=%{text}", textposition="outside", cliponaxis=False)
        st.plotly_chart(style_figure(bar), width="stretch")

    left, right = st.columns([1.25, 0.75])
    with left:
        color_counts = filtered["Mahalle"].value_counts()
        top_neighborhoods = set(color_counts.head(8).index)
        scatter_data = filtered.assign(
            MahalleGrubu=filtered["Mahalle"].where(
                filtered["Mahalle"].isin(top_neighborhoods), "Diğer"
            )
        )
        scatter = px.scatter(
            scatter_data,
            x="Brüt m²",
            y=TARGET,
            color="MahalleGrubu",
            hover_data=["Mahalle", "Oda Sayısı", "Bina Yaşı Ortalama"],
            opacity=0.62,
            title="Brüt alan ile ilan fiyatı ilişkisi",
            labels={TARGET: "İlan fiyatı (TL)", "MahalleGrubu": "Mahalle"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(style_figure(scatter, 470), width="stretch")
    with right:
        room_summary = (
            filtered.groupby("Oda Sayısı")
            .agg(medyan=(TARGET, "median"), ilan=(TARGET, "size"))
            .loc[lambda frame: frame["ilan"] >= 5]
            .sort_values("medyan")
            .reset_index()
        )
        room_bar = px.bar(
            room_summary,
            x="Oda Sayısı",
            y="medyan",
            text="ilan",
            title="Oda tipine göre medyan fiyat",
            color_discrete_sequence=["#f2b84b"],
            labels={"medyan": "Medyan fiyat (TL)", "Oda Sayısı": "Oda tipi"},
        )
        room_bar.update_traces(texttemplate="n=%{text}", textposition="outside")
        st.plotly_chart(style_figure(room_bar, 470), width="stretch")

    export_columns = [
        "Mahalle",
        "İlan Tarihi Parsed",
        TARGET,
        "Brüt m²",
        "Net m²",
        "Oda Sayısı",
        "Bina Yaşı Ortalama",
        "Site İçerisinde",
    ]
    st.download_button(
        "Filtrelenmiş veriyi CSV indir",
        data=filtered[export_columns].to_csv(index=False).encode("utf-8-sig"),
        file_name="atakum_filtrelenmis_ilanlar.csv",
        mime="text/csv",
    )

with tabs[1]:
    st.markdown(
        '<div class="section-label">Model performansı ve güvenilirlik</div>', unsafe_allow_html=True
    )
    comparison = load_output_table("model_karsilastirma.csv")
    importance = load_output_table("ozellik_onemi.csv")
    protocols = load_output_table("protokol_duyarliligi.csv")

    if comparison.empty:
        st.info("Model tabloları için önce `python analiz_icin_kodlar.py` komutunu çalıştırın.")
    else:
        best = comparison.sort_values("cv_mae_mean").iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Seçilen model", str(best["model"]))
        c2.metric("Test MAE", tr_tl(best["test_mae"], compact=True))
        c3.metric("Test R²", f"{best['test_r2']:.3f}")
        c4.metric("Test MAPE", f"%{best['test_mape'] * 100:.1f}")

        left, right = st.columns(2)
        with left:
            model_bar = px.bar(
                comparison.sort_values("cv_mae_mean", ascending=False),
                x="cv_mae_mean",
                y="model",
                orientation="h",
                error_x="cv_mae_std",
                title="5 katlı çapraz doğrulama MAE",
                color_discrete_sequence=["#2f80ed"],
                labels={"cv_mae_mean": "MAE (TL, düşük daha iyi)", "model": ""},
            )
            st.plotly_chart(style_figure(model_bar), width="stretch")
        with right:
            r2_bar = px.bar(
                comparison.sort_values("test_r2"),
                x="test_r2",
                y="model",
                orientation="h",
                title="Kilitli test R²",
                color_discrete_sequence=["#12b3a8"],
                labels={"test_r2": "R² (yüksek daha iyi)", "model": ""},
            )
            st.plotly_chart(style_figure(r2_bar), width="stretch")

        left, right = st.columns([0.9, 1.1])
        with left:
            if not importance.empty:
                importance_chart = px.bar(
                    importance.head(12).sort_values("mae_increase_mean"),
                    x="mae_increase_mean",
                    y="feature",
                    orientation="h",
                    error_x="mae_increase_std",
                    title="Permütasyon özelliği önemi",
                    color_discrete_sequence=["#f2b84b"],
                    labels={
                        "mae_increase_mean": "Karıştırıldığında MAE artışı (TL)",
                        "feature": "",
                    },
                )
                st.plotly_chart(style_figure(importance_chart, 480), width="stretch")
        with right:
            if not protocols.empty:
                protocol_chart = go.Figure()
                protocol_chart.add_trace(
                    go.Bar(
                        x=protocols["protocol"],
                        y=protocols["r2"],
                        name="R²",
                        marker_color="#12b3a8",
                    )
                )
                protocol_chart.update_layout(title="Değerlendirme protokolü duyarlılığı")
                protocol_chart.update_xaxes(tickangle=-22)
                st.plotly_chart(style_figure(protocol_chart, 480), width="stretch")

        st.markdown(
            """
            <div class="mini-note"><strong>Neden eski %92 başarı doğrudan kullanılmadı?</strong><br>
            Aynı ilan kimliği farklı tarihlerde hem eğitim hem test tarafına düşerse model ilanı
            kısmen hatırlayabilir. Ana değerlendirme tekil ilanların son görüntüsünde yapılır;
            ayrıca ilan-kimliği gruplu ve kronolojik kontroller raporlanır.</div>
            """,
            unsafe_allow_html=True,
        )

with tabs[2]:
    st.markdown(
        '<div class="section-label">Bonus araç • Senaryo bazlı tahmin</div>', unsafe_allow_html=True
    )
    st.subheader("Tek konut için ilan fiyatı simülatörü")
    st.caption("Girdi değerleri değiştirilerek modelin senaryo tahmini görülebilir.")

    with st.form("prediction_form"):
        row1 = st.columns(4)
        neighborhood_value = row1[0].selectbox("Mahalle", sorted(latest["Mahalle"].unique()))
        gross_value = row1[1].number_input("Brüt m²", 30, 350, int(latest["Brüt m²"].median()))
        net_value = row1[2].number_input("Net m²", 20, 320, int(latest["Net m²"].median()))
        room_value = row1[3].selectbox("Oda tipi", sorted(latest["Oda Sayısı"].unique()))

        row2 = st.columns(4)
        bathroom_value = row2[0].number_input(
            "Banyo sayısı", 1, 5, int(latest["Banyo Sayısı"].median())
        )
        age_value = row2[1].number_input(
            "Bina yaşı", 0.0, 60.0, float(latest["Bina Yaşı Ortalama"].median()), 0.5
        )
        floor_value = row2[2].number_input(
            "Bulunduğu kat", -3, 30, int(latest["Bulunduğu Kat (Dönüştürülmüş)"].median())
        )
        total_floor_value = row2[3].number_input(
            "Bina kat sayısı", 1, 40, int(latest["Kat Sayısı Numeric"].median())
        )

        row3 = st.columns(4)
        site_value = row3[0].selectbox("Site içinde", sorted(latest["Site İçerisinde"].unique()))
        heating_value = row3[1].selectbox("Isıtma", sorted(latest["Isıtma"].unique()))
        parking_value = row3[2].selectbox("Otopark", sorted(latest["Otopark"].unique()))
        fee_value = row3[3].number_input(
            "Aidat (TL)", 0, 20_000, int(latest["Aidat (TL) Numeric"].median())
        )

        with st.expander("Diğer özellikler"):
            row4 = st.columns(4)
            kitchen_value = row4[0].selectbox("Mutfak", sorted(latest["Mutfak"].unique()))
            balcony_value = row4[1].selectbox("Balkon", sorted(latest["Balkon"].unique()))
            elevator_value = row4[2].selectbox("Asansör", sorted(latest["Asansör"].unique()))
            furnished_value = row4[3].selectbox("Eşyalı", sorted(latest["Eşyalı"].unique()))
            row5 = st.columns(4)
            usage_value = row5[0].selectbox(
                "Kullanım durumu", sorted(latest["Kullanım Durumu"].unique())
            )
            credit_value = row5[1].selectbox(
                "Krediye uygun", sorted(latest["Krediye Uygun"].unique())
            )
            title_value = row5[2].selectbox("Tapu durumu", sorted(latest["Tapu Durumu"].unique()))
            seller_value = row5[3].selectbox("Kimden", sorted(latest["Kimden"].unique()))
            trade_value = st.selectbox("Takas", sorted(latest["Takas"].unique()))

        submitted = st.form_submit_button("Tahmini hesapla", type="primary", width="stretch")

    if submitted:
        room_numeric = float(
            latest.loc[latest["Oda Sayısı"].eq(room_value), "Oda Sayısı Numeric"].median()
        )
        input_row = pd.DataFrame(
            [
                {
                    "Brüt m²": gross_value,
                    "Net m²": net_value,
                    "Banyo Sayısı": bathroom_value,
                    "Bina Yaşı Ortalama": age_value,
                    "Bulunduğu Kat (Dönüştürülmüş)": floor_value,
                    "Oda Sayısı Numeric": room_numeric,
                    "Kat Sayısı Numeric": total_floor_value,
                    "Aidat (TL) Numeric": fee_value,
                    "İlan Günü": float(latest["İlan Günü"].max()),
                    "Mahalle": neighborhood_value,
                    "Isıtma": heating_value,
                    "Mutfak": kitchen_value,
                    "Balkon": balcony_value,
                    "Asansör": elevator_value,
                    "Otopark": parking_value,
                    "Eşyalı": furnished_value,
                    "Kullanım Durumu": usage_value,
                    "Site İçerisinde": site_value,
                    "Krediye Uygun": credit_value,
                    "Tapu Durumu": title_value,
                    "Kimden": seller_value,
                    "Takas": trade_value,
                }
            ],
            columns=MODEL_FEATURES,
        )
        model = load_prediction_model(latest)
        prediction = float(model.predict(input_row)[0])
        reported_mae = summary.get("best_model", {}).get("locked_test", {}).get("test_mae", 485_000)
        similar = latest.loc[
            latest["Mahalle"].eq(neighborhood_value)
            & latest["Oda Sayısı"].eq(room_value)
            & latest["Brüt m²"].between(gross_value * 0.8, gross_value * 1.2)
        ]

        a, b, c = st.columns(3)
        a.metric("Model tahmini", tr_tl(prediction, compact=True))
        b.metric(
            "Test MAE referans bandı",
            f"± {tr_tl(float(reported_mae), compact=True)}",
        )
        c.metric(
            "Benzer ilan medyanı",
            tr_tl(similar[TARGET].median(), compact=True) if len(similar) else "Yetersiz örnek",
            delta=f"n={len(similar)}",
        )
        st.markdown(
            """
            <div class="mini-note"><strong>Kullanım sınırı:</strong> Bu çıktı ilan fiyatı
            senaryosudur; resmi ekspertiz, satış garantisi veya yatırım tavsiyesi değildir.
            “± MAE” bireysel tahmin aralığı değil, kilitli testteki tipik mutlak hata için
            referanstır.</div>
            """,
            unsafe_allow_html=True,
        )

with tabs[3]:
    st.markdown(
        '<div class="section-label">İzlenebilir veri hazırlama</div>', unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ham örneklem", tr_number(audit["reported_raw_rows"]))
    c2.metric(
        "Repo verisi",
        tr_number(audit["repository_rows"]),
        delta=f"-{audit['previously_removed_rows']} ön temizlik",
    )
    c3.metric("Yapısal hatalı", tr_number(audit["malformed_rows"]))
    c4.metric("Tekil son ilan", tr_number(audit["latest_snapshot_rows"]))

    flow = pd.DataFrame(
        {
            "Aşama": ["Ham örneklem", "Repo temiz veri", "Yapısal geçerli", "Tekil son ilan"],
            "Kayıt": [
                audit["reported_raw_rows"],
                audit["repository_rows"],
                audit["structurally_valid_rows"],
                audit["latest_snapshot_rows"],
            ],
        }
    )
    left, right = st.columns([1.15, 0.85])
    with left:
        funnel = px.funnel(
            flow,
            x="Kayıt",
            y="Aşama",
            title="Örneklem akışı",
            color_discrete_sequence=["#2f80ed"],
        )
        st.plotly_chart(style_figure(funnel, 430), width="stretch")
    with right:
        quality = pd.DataFrame(
            {
                "Kontrol": [
                    "Birebir tekrar satır",
                    "Tekrarlanan ilan görüntüsü",
                    "Fiyatı değişen ilan",
                    "Brüt alanı net alandan küçük",
                ],
                "Sayı": [
                    audit["exact_duplicate_rows"],
                    audit["repeated_snapshot_rows"],
                    audit["listings_with_price_change"],
                    audit["gross_below_net_rows"],
                ],
            }
        )
        quality_bar = px.bar(
            quality.sort_values("Sayı"),
            x="Sayı",
            y="Kontrol",
            orientation="h",
            title="Veri kalite kontrolleri",
            color_discrete_sequence=["#f2b84b"],
        )
        st.plotly_chart(style_figure(quality_bar, 430), width="stretch")

    with st.expander("Yapısal olarak reddedilen kayıtları göster"):
        columns = ["_Kaynak Satır", TARGET, "Mahalle", "İlan No", "İlan Tarihi", "Reddetme Nedeni"]
        rejected_display = rejected[columns].copy()
        rejected_display["İlan No"] = rejected_display["İlan No"].astype(str)
        rejected_display["İlan Tarihi"] = rejected_display["İlan Tarihi"].astype(str)
        st.dataframe(rejected_display, width="stretch", hide_index=True)

st.markdown(
    """
    <div class="mini-note"><strong>Veri tanımı:</strong> Panel gerçekleşmiş tapu satışlarını
    değil, 2024-2025 döneminde derlenen satılık konut ilan fiyatlarını gösterir. Filtreli
    küçük gruplardaki sonuçlar temkinli yorumlanmalıdır.</div>
    """,
    unsafe_allow_html=True,
)

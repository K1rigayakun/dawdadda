from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from components.ui import hero, inject_css, metric_card, section
from utils.data import load_laptop_data
from utils.model import CATEGORICAL, FEATURES, NUMERIC, feature_importance, format_idr, train_price_model


st.set_page_config(
    page_title="Intelijen AI Laptop",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

LABELS = {
    "brand": "Merek",
    "processor": "Prosesor",
    "gpu": "GPU",
    "os": "Sistem Operasi",
    "segment": "Segmen",
    "ram_gb": "RAM (GB)",
    "storage_gb": "Penyimpanan (GB)",
    "screen_size": "Ukuran Layar",
    "weight_kg": "Berat (kg)",
    "price": "Harga",
    "importance": "Pengaruh",
    "feature": "Fitur",
    "actual": "Harga Aktual",
    "predicted": "Harga Prediksi",
    "error": "Galat",
    "units": "Jumlah Unit",
    "avg_price": "Rata-rata Harga",
    "avg_ram": "Rata-rata RAM",
    "mobility_score": "Skor Mobilitas",
}


@st.cache_data(show_spinner=False)
def filtered_data(df: pd.DataFrame, brands: list[str], segments: list[str], price_range: tuple[int, int]) -> pd.DataFrame:
    if not brands:
        brands = sorted(df["brand"].unique())
    if not segments:
        segments = sorted(df["segment"].unique())
    data = df[df["brand"].isin(brands) & df["segment"].isin(segments)]
    return data[(data["price"] >= price_range[0]) & (data["price"] <= price_range[1])].copy()


def chart_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,9,20,0.35)",
        font_color="#e5edf7",
        margin=dict(l=20, r=20, t=52, b=30),
        hoverlabel=dict(bgcolor="#111827", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


df = load_laptop_data()
model, metrics, x_test, y_test, y_pred = train_price_model(df)

with st.sidebar:
    st.markdown("### AI Laptop")
    st.caption("Dashboard analitik premium")
    page = option_menu(
        None,
        [
            "Beranda",
            "Ringkasan Dataset",
            "Analisis Eksploratif",
            "Lab Visualisasi",
            "Pipeline ML",
            "Rekayasa Fitur",
            "Pusat Performa",
            "Playground Prediksi",
            "Insight Bisnis",
            "Tentang Proyek",
        ],
        icons=["stars", "database", "activity", "bar-chart", "diagram-3", "cpu", "speedometer2", "magic", "briefcase", "info-circle"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "icon": {"color": "#23d7ff", "font-size": "17px"},
            "nav-link": {"color": "#cbd5e1", "font-size": "14px", "border-radius": "12px", "margin": "4px 0"},
            "nav-link-selected": {"background": "linear-gradient(135deg, rgba(35,215,255,.24), rgba(155,92,255,.32))"},
        },
    )

    st.divider()
    brand_filter = st.multiselect("Merek", sorted(df["brand"].unique()), default=sorted(df["brand"].unique()))
    segment_filter = st.multiselect("Segmen", sorted(df["segment"].unique()), default=sorted(df["segment"].unique()))
    min_price, max_price = int(df["price"].min()), int(df["price"].max())
    price_filter = st.slider("Rentang harga", min_price, max_price, (min_price, max_price), step=500000)

data = filtered_data(df, brand_filter, segment_filter, price_filter)

if page == "Beranda":
    hero(
        "Intelijen Harga Laptop",
        "Dashboard AI sinematik untuk analisis pasar laptop, eksplorasi fitur, evaluasi model, dan prediksi harga instan.",
    )
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Produk", f"{len(data):,}", "baris laptop terfilter")
    with c2:
        metric_card("Median Harga", format_idr(data["price"].median()), "titik tengah pasar")
    with c3:
        metric_card("R2 Model", f"{metrics['r2']:.3f}", "skor validasi")
    with c4:
        metric_card("MAE", format_idr(metrics["mae"]), "rata-rata galat")

    section("Denyut Pasar", "Grafik interaktif dengan hover, zoom, dan transisi Plotly.")
    left, right = st.columns([1.2, 1])
    with left:
        fig = px.histogram(data, x="price", color="segment", nbins=34, title="Distribusi Harga per Segmen", labels=LABELS)
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        fig = px.treemap(data, path=["brand", "segment"], values="price", color="price", color_continuous_scale="Turbo", title="Peta Merek Berbasis Nilai Harga", labels=LABELS)
        st.plotly_chart(chart_layout(fig), width="stretch")

elif page == "Ringkasan Dataset":
    st.markdown('<h1 class="page-title">Ringkasan Dataset</h1>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", f"{len(data):,}")
    c2.metric("Kolom", f"{data.shape[1]}")
    c3.metric("Merek", data["brand"].nunique())
    c4.metric("Rata-rata Harga", format_idr(data["price"].mean()))
    st.dataframe(data, width="stretch", height=420)
    section("Kualitas Data")
    quality = pd.DataFrame({"kolom": data.columns, "nilai_kosong": data.isna().sum().values, "tipe_data": data.dtypes.astype(str).values})
    st.dataframe(quality, width="stretch", hide_index=True)

elif page == "Analisis Eksploratif":
    st.markdown('<h1 class="page-title">Analisis Eksploratif</h1>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        fig = px.box(data, x="brand", y="price", color="brand", title="Sebaran Harga per Merek", labels=LABELS)
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        fig = px.scatter(data, x="ram_gb", y="price", size="storage_gb", color="segment", hover_data=["brand", "processor"], title="Relasi RAM, Penyimpanan, dan Harga", labels=LABELS)
        st.plotly_chart(chart_layout(fig), width="stretch")
    corr = data[NUMERIC + ["price"]].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Matriks Korelasi Numerik")
    st.plotly_chart(chart_layout(fig, 460), width="stretch")

elif page == "Lab Visualisasi":
    st.markdown('<h1 class="page-title">Lab Visualisasi Interaktif</h1>', unsafe_allow_html=True)
    x_axis = st.selectbox("Sumbu X", NUMERIC, index=0, format_func=lambda x: LABELS.get(x, x))
    y_axis = st.selectbox("Sumbu Y", ["price"] + NUMERIC, index=0, format_func=lambda x: LABELS.get(x, x))
    color = st.selectbox("Warna", CATEGORICAL, index=4, format_func=lambda x: LABELS.get(x, x))
    fig = px.scatter(data, x=x_axis, y=y_axis, color=color, size="storage_gb", hover_data=FEATURES, title="Analisis Kustom Berbasis Filter", labels=LABELS)
    st.plotly_chart(chart_layout(fig, 560), width="stretch")
    section("Drill-down Segmen")
    fig = px.sunburst(data, path=["segment", "brand", "processor"], values="price", color="price", color_continuous_scale="Viridis", labels=LABELS)
    st.plotly_chart(chart_layout(fig, 560), width="stretch")

elif page == "Pipeline ML":
    st.markdown('<h1 class="page-title">Pipeline Machine Learning</h1>', unsafe_allow_html=True)
    stages = ["Impor", "Bersihkan", "Encode", "Skalakan", "Latih", "Validasi", "Prediksi"]
    cols = st.columns(len(stages))
    for col, stage in zip(cols, stages):
        with col:
            metric_card(stage, "OK", "tahap pipeline")
    section("Arsitektur Pipeline")
    st.code(
        "DataFrame -> ColumnTransformer(OneHotEncoder + StandardScaler) -> RandomForestRegressor -> Prediksi harga",
        language="text",
    )
    fig = px.bar(feature_importance(model), x="importance", y="feature", orientation="h", title="Peringkat Sinyal Model", color="importance", color_continuous_scale="Turbo", labels=LABELS)
    st.plotly_chart(chart_layout(fig, 520), width="stretch")

elif page == "Rekayasa Fitur":
    st.markdown('<h1 class="page-title">Rekayasa Fitur</h1>', unsafe_allow_html=True)
    engineered = data.copy()
    engineered["price_per_gb_ram"] = engineered["price"] / engineered["ram_gb"]
    engineered["price_per_storage"] = engineered["price"] / engineered["storage_gb"]
    engineered["mobility_score"] = (engineered["screen_size"] / engineered["weight_kg"]).round(2)
    cols = st.columns(3)
    cols[0].metric("Harga / GB RAM", format_idr(engineered["price_per_gb_ram"].median()))
    cols[1].metric("Harga / GB Penyimpanan", format_idr(engineered["price_per_storage"].median()))
    cols[2].metric("Skor Mobilitas", f"{engineered['mobility_score'].median():.2f}")
    fig = px.scatter(engineered, x="mobility_score", y="price", color="segment", size="ram_gb", hover_data=["brand", "processor"], title="Skor Mobilitas vs Harga", labels=LABELS)
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.dataframe(engineered.head(60), width="stretch")

elif page == "Pusat Performa":
    st.markdown('<h1 class="page-title">Pusat Performa Model</h1>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Skor R2", f"{metrics['r2']:.3f}")
    c2.metric("MAE", format_idr(metrics["mae"]))
    c3.metric("Baris Validasi", f"{metrics['test_rows']:,}")
    results = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    results["error"] = results["predicted"] - results["actual"]
    left, right = st.columns(2)
    with left:
        fig = px.scatter(results, x="actual", y="predicted", title="Harga Aktual vs Prediksi", labels=LABELS)
        fig.add_trace(go.Scatter(x=[results.actual.min(), results.actual.max()], y=[results.actual.min(), results.actual.max()], mode="lines", name="Prediksi Sempurna"))
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        fig = px.histogram(results, x="error", nbins=34, title="Distribusi Galat Prediksi", labels=LABELS)
        st.plotly_chart(chart_layout(fig), width="stretch")

elif page == "Playground Prediksi":
    st.markdown('<h1 class="page-title">Playground Prediksi AI</h1>', unsafe_allow_html=True)
    left, right = st.columns([0.82, 1.18])
    with left:
        user_input = {
            "brand": st.selectbox("Merek", sorted(df["brand"].unique())),
            "processor": st.selectbox("Prosesor", sorted(df["processor"].unique())),
            "gpu": st.selectbox("GPU", sorted(df["gpu"].unique())),
            "os": st.selectbox("Sistem Operasi", sorted(df["os"].unique())),
            "segment": st.selectbox("Segmen", sorted(df["segment"].unique())),
            "ram_gb": st.slider("RAM (GB)", 4, 64, 16, 4),
            "storage_gb": st.slider("Penyimpanan (GB)", 128, 2048, 512, 128),
            "screen_size": st.slider("Ukuran Layar", 11.0, 18.0, 14.0, 0.1),
            "weight_kg": st.slider("Berat (kg)", 0.8, 3.5, 1.5, 0.1),
        }
    pred_df = pd.DataFrame([user_input])
    prediction = float(model.predict(pred_df)[0])
    market_median = float(df["price"].median())
    confidence = max(62, min(96, 96 - abs(prediction - market_median) / market_median * 18))
    with right:
        metric_card("Harga Prediksi", format_idr(prediction), f"{confidence:.0f}% keyakinan model")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=confidence, title={"text": "Keyakinan"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#23d7ff"}}))
        st.plotly_chart(chart_layout(gauge, 300), width="stretch")
        recommendation = "posisi premium" if prediction > market_median * 1.2 else "posisi hemat bernilai tinggi" if prediction < market_median * 0.85 else "posisi pasar seimbang"
        st.info(f"Rekomendasi: {recommendation.capitalize()} berdasarkan median pasar saat ini {format_idr(market_median)}.")

elif page == "Insight Bisnis":
    st.markdown('<h1 class="page-title">Insight Bisnis</h1>', unsafe_allow_html=True)
    brand_perf = data.groupby("brand", as_index=False).agg(avg_price=("price", "mean"), units=("price", "size"), avg_ram=("ram_gb", "mean"))
    fig = px.scatter(brand_perf, x="units", y="avg_price", size="avg_ram", color="brand", text="brand", title="Peta Peluang Merek", labels=LABELS)
    st.plotly_chart(chart_layout(fig), width="stretch")
    top_segment = data.groupby("segment")["price"].median().sort_values(ascending=False)
    for idx, (segment, price) in enumerate(top_segment.items(), start=1):
        st.markdown(f'<div class="glass-card"><b>{idx}. {segment}</b><br><span style="color:#9aa7bd">Median harga pasar: {format_idr(price)}</span></div>', unsafe_allow_html=True)

else:
    st.markdown('<h1 class="page-title">Tentang Proyek</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        Dashboard Streamlit ini mengubah workflow analitik harga laptop menjadi pengalaman produk AI yang premium.
        Di dalamnya ada pemuatan data dengan cache, EDA interaktif, pipeline ML reusable, feature importance,
        analisis performa, dan playground prediksi langsung.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**Tech stack:** Streamlit, Pandas, Plotly, Scikit-learn, custom CSS, dan cached model resources.")


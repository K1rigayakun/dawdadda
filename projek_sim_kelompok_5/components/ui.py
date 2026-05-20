from __future__ import annotations

from pathlib import Path

import streamlit as st


def inject_css() -> None:
    css_path = Path("assets/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "Intelijen AI Laptop") -> None:
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div style="margin-top:26px">
            <span class="pill">Mesin prediksi langsung</span>
            <span class="pill">Lab EDA interaktif</span>
            <span class="pill">UI Streamlit premium</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="glass-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div style="color:#9aa7bd;margin-top:4px">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)

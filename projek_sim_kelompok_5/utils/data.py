from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


DATA_CANDIDATES = (
    "laptop_price.csv",
    "laptop_data.csv",
    "laptops.csv",
    "data/laptop_data.csv",
    "data/laptops.csv",
    "dataset/laptop_data.csv",
    "dataset/laptops.csv",
)


@st.cache_data(show_spinner=False)
def load_laptop_data() -> pd.DataFrame:
    for candidate in DATA_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return normalize_dataset(read_csv_safely(path))
    return generate_synthetic_laptops()


def read_csv_safely(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", encoding_errors="replace")


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [c.strip().lower().replace(" ", "_") for c in normalized.columns]

    aliases = {
        "brand": ["company", "manufacturer", "make"],
        "price": ["price_in_euros", "price_euros", "price_idr", "harga", "price"],
        "ram_gb": ["ram", "ram_gb"],
        "storage_gb": ["storage", "memory", "storage_gb"],
        "weight_kg": ["weight", "weight_kg"],
        "processor": ["cpu", "processor"],
        "gpu": ["gpu", "graphics"],
        "screen_size": ["inches", "screen_size"],
        "os": ["opsys", "os"],
        "segment": ["typename", "type_name", "segment"],
    }
    for target, sources in aliases.items():
        if target not in normalized:
            for source in sources:
                if source in normalized:
                    normalized[target] = normalized[source]
                    break

    if "ram_gb" in normalized:
        normalized["ram_gb"] = normalized["ram_gb"].astype(str).str.extract(r"(\d+)").astype(float)
    if "storage_gb" in normalized:
        storage = normalized["storage_gb"].astype(str)
        values = storage.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
        normalized["storage_gb"] = np.where(storage.str.contains("TB", case=False, na=False), values * 1024, values)
    if "weight_kg" in normalized:
        normalized["weight_kg"] = normalized["weight_kg"].astype(str).str.replace("kg", "", case=False).astype(float)
    if "price" in normalized:
        normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
        if normalized["price"].median(skipna=True) < 10000:
            normalized["price"] = normalized["price"] * 17000

    required = generate_synthetic_laptops(320)
    for col in required.columns:
        if col not in normalized:
            normalized[col] = required[col].sample(len(normalized), replace=True, random_state=7).to_numpy()

    return normalized.dropna(subset=["price"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def generate_synthetic_laptops(n: int = 950) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    brands = np.array(["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Razer"])
    processors = np.array(["Intel i5", "Intel i7", "Intel i9", "Ryzen 5", "Ryzen 7", "Apple M2", "Apple M3"])
    gpu = np.array(["Integrated", "RTX 3050", "RTX 4060", "RTX 4070", "Radeon", "Apple GPU"])
    os = np.array(["Windows", "macOS", "Linux", "ChromeOS"])
    segments = np.array(["Creator", "Gaming", "Business", "Student", "Ultrabook", "Workstation"])

    brand = rng.choice(brands, n, p=[0.13, 0.16, 0.15, 0.17, 0.15, 0.1, 0.09, 0.05])
    ram = rng.choice([8, 16, 24, 32, 64], n, p=[0.25, 0.42, 0.08, 0.2, 0.05])
    storage = rng.choice([256, 512, 1024, 2048], n, p=[0.19, 0.44, 0.3, 0.07])
    screen = rng.choice([13.3, 14.0, 15.6, 16.0, 17.3], n, p=[0.18, 0.27, 0.36, 0.14, 0.05])
    weight = np.round(rng.normal(1.65, 0.38, n).clip(0.9, 3.2), 2)
    cpu = rng.choice(processors, n)
    graphics = rng.choice(gpu, n, p=[0.45, 0.16, 0.17, 0.08, 0.08, 0.06])
    opsys = np.where(np.isin(cpu, ["Apple M2", "Apple M3"]) | (brand == "Apple"), "macOS", rng.choice(os, n, p=[0.82, 0.02, 0.08, 0.08]))
    segment = rng.choice(segments, n)

    brand_premium = pd.Series(brand).map({"Apple": 9000000, "Razer": 7500000, "MSI": 4200000, "Dell": 2400000}).fillna(1200000).to_numpy()
    cpu_premium = pd.Series(cpu).map({"Intel i9": 8500000, "Intel i7": 4300000, "Ryzen 7": 3800000, "Apple M3": 7800000, "Apple M2": 5600000}).fillna(1600000).to_numpy()
    gpu_premium = pd.Series(graphics).map({"RTX 4070": 9500000, "RTX 4060": 6500000, "RTX 3050": 3600000, "Apple GPU": 4800000}).fillna(900000).to_numpy()
    segment_premium = pd.Series(segment).map({"Workstation": 5200000, "Gaming": 3900000, "Creator": 3100000, "Ultrabook": 2200000}).fillna(700000).to_numpy()
    price = (
        5200000
        + ram * 430000
        + storage * 9500
        + screen * 310000
        - weight * 280000
        + brand_premium
        + cpu_premium
        + gpu_premium
        + segment_premium
        + rng.normal(0, 1900000, n)
    ).clip(4800000, 72000000)

    return pd.DataFrame(
        {
            "brand": brand,
            "processor": cpu,
            "gpu": graphics,
            "os": opsys,
            "segment": segment,
            "ram_gb": ram,
            "storage_gb": storage,
            "screen_size": screen,
            "weight_kg": weight,
            "price": price.round(0).astype(int),
        }
    )

# Dashboard Intelijen AI Laptop

Premium Streamlit dashboard untuk analisis harga laptop, EDA interaktif, pipeline machine learning, dan simulator prediksi harga.

## Fitur

- Landing page cinematic dengan glassmorphism dan motion CSS.
- Dataset overview, EDA, visualization lab, feature engineering, model center, dan prediction playground.
- Plotly interactive charts dengan hover, zoom, filtering, dan drill-down.
- Model prediksi harga berbasis `RandomForestRegressor`.
- Auto-load dataset CSV jika tersedia, fallback ke synthetic laptop dataset.

## Struktur

```text
app.py
assets/styles.css
components/ui.py
utils/data.py
utils/model.py
requirements.txt
```

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opsional: letakkan dataset di salah satu path berikut agar dipakai otomatis:

- `laptop_price.csv`
- `laptop_data.csv`
- `laptops.csv`
- `data/laptop_data.csv`
- `data/laptops.csv`
- `dataset/laptop_data.csv`
- `dataset/laptops.csv`

Kolom umum seperti `Company`, `Ram`, `Memory`, `Weight`, `Cpu`, `Gpu`, `OpSys`, dan `Price` akan dinormalisasi otomatis.

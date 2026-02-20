# 🏨 Hotel Price Forecasting Models

Repositori ini berisi kumpulan skrip dan pemodelan untuk memprediksi harga kamar hotel di berbagai Online Travel Agency (OTA) menggunakan pendekatan Statistik (*Time-Series*) dan *Machine Learning*.

## 📖 Latar Belakang
Fluktuasi harga kamar hotel di OTA sangat dinamis dan dipengaruhi oleh banyak faktor seperti musim liburan, hari libur akhir pekan, dan tren pasar. Proyek ini bertujuan untuk membangun model prediktif yang dapat mempelajari pola historis harga kamar hotel dari waktu ke waktu dan memproyeksikan harganya di masa depan. 

Dengan prediksi yang akurat beserta rentang *Confidence Interval*-nya, pelaku bisnis perhotelan (atau tim *Revenue Management*) dapat membuat keputusan yang lebih tepat terkait strategi harga.

## 🚀 Fitur & Algoritma Model

Proyek ini menyediakan dua pendekatan pemodelan utama (terdapat di dalam `predict_price_clean.py`):

1. **Model ARIMA (Class `PriceForecast`)**
   - Menggunakan algoritma **SARIMA** (*Seasonal AutoRegressive Integrated Moving Average*).
   - Dioptimasi menggunakan transformasi Logaritma (`np.log1p`) untuk menstabilkan variansi harga yang fluktuatif ekstrem.
   - Menggunakan teknik imputasi **Forward Fill (`ffill`)** yang menyesuaikan sifat alami perubahan harga (berbentuk *step function*).
   - Menghasilkan prediksi lengkap dengan *99% Confidence Interval* secara statistik.

2. **Model Random Forest (Class `PriceForecastRevamped`)**
   - Menggunakan algoritma **Random Forest Regressor** (*Ensemble Machine Learning*).
   - Melakukan *Feature Engineering* otomatis dari data tanggal (ekstraksi `day_of_week`, `is_weekend`, `day_of_month`, dll).
   - Menghitung rentang *95% Confidence Interval* berdasarkan perhitungan Standar Deviasi dari estimasi pohon (*trees*) di dalam *forest*.
   - Lebih tangguh dalam menangani data harga historis yang memiliki pola non-linear.

---

## 💻 Prasyarat Sistem

Pastikan environment Anda memenuhi spesifikasi berikut:
* **Python**: Versi **3.12.12** (Direkomendasikan).
* **Library**: Semua *dependencies* sudah terdaftar di `requirements.txt`.

## 🛠️ Cara Instalasi (How to Run)

1. **Clone Repositori**
   ```bash
    git clone [https://github.com/oscar-sinaga/predict_price_hotel.git](https://github.com/oscar-sinaga/predict_price_hotel.git)
    cd predict_price_hotel
   ```
2. **Buat Virtual Environment (Opsional namun direkomendasikan)**
   ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    venv\Scripts\activate     # Untuk Windows
   ```

3. **Install Dependencies**
   ```bash
    pip install -r requirements.txt
   ```


# 📊 Panduan Penggunaan

## A. Penggunaan via Google Colab / Jupyter Notebook (Metode Interaktif)

Jika Anda menggunakan **Google Colab**, Anda dapat menjalankan model secara interaktif tanpa perlu mengubah banyak kode secara manual.

### Langkah-langkah:

1. Upload file data historis (`.csv` atau `.xlsx`) ke environment Colab.
2. Buat cell baru.
3. Import class model.
4. Inisialisasi data.
5. Jalankan evaluasi dan prediksi.

Gunakan method berikut:

- `.evaluate_model()` → untuk mendapatkan metrik evaluasi seperti **MAPE**
- `.model_predict()` → untuk melakukan proyeksi harga ke depan
- `.plot_prediction()` → untuk menampilkan grafik hasil prediksi

---

## B. Penggunaan Dasar via Script Python

Anda dapat langsung menggunakan class di dalam script Python Anda seperti berikut:

```python
import pandas as pd
from predict_price_clean import PriceForecastRevamped

# 1. Baca Data
df = pd.read_csv('histories_room.csv')

# 2. Inisiasi Model Random Forest
model_rf = PriceForecastRevamped()

# 3. Masukkan Data ke Model
model_rf.initialize_data(
    df['Date'],
    df['Price'],
    df['Hotel Name'],
    df['Room Name'],
    df['OTA']
)

# 4. Filter Kombinasi Hotel yang Ingin Diprediksi
model_rf.read_data(
    'Yogyakarta Marriott Hotel',
    'Deluxe, Guest room, 2 Twin/Single Bed(s), City view',
    'agoda'
)

# 5. Prediksi 7 Hari ke Depan
hasil_prediksi = model_rf.model_predict(days_pred=7)
print(hasil_prediksi)

# 6. Tampilkan Grafik
model_rf.plot_prediction()
```

# 📁 Struktur Repositori

- **`predict_price_clean.py`**  
  Berisi source code utama dari:
  - `PriceForecast` → Model berbasis **ARIMA (Time-Series)**
  - `PriceForecastRevamped` → Model berbasis **Random Forest (Machine Learning)**

- **`experiments.ipynb`**  
  Notebook Jupyter yang digunakan untuk:
  - Tahap riset dan eksplorasi data  
  - Pengujian metrik performa  
  - Visualisasi dan plotting eksperimental  

- **`requirements.txt`**  
  Daftar library Python beserta versi yang dibutuhkan untuk menjalankan proyek.

- **`histories_room_ota2.xlsx` / `histories_room.csv`** *(Opsional)*  
  Dataset mentah contoh yang berisi riwayat pencatatan harga kamar hotel.

---

Created and Maintained by **Oscar Sinaga**


# Predictive Maintenance Motor Industri berbasis IoT dan Machine Learning

Studi kasus ini diturunkan dari materi kuliah tamu *Introduction to Machine Learning*
(Informatic Practitioner Lecture Series 2026, Dr. Taniza Tajudin, UiTM Kedah).
Semua konsep inti materi diterapkan pada data sensor IoT: alur kerja ML, supervised
(klasifikasi dan regresi), unsupervised (clustering, PCA, deteksi anomali), pengecekan
overfitting, dan deployment (inference).

## Skenario
Sebuah pabrik memasang sensor IoT (mis. ESP32) pada motor/pompa. Setiap sensor mengirim
data secara berkala. Tujuannya mendeteksi kerusakan lebih awal agar perawatan dapat
dijadwalkan sebelum mesin berhenti mendadak.

| Sensor | Satuan |
|---|---|
| suhu | °C |
| getaran | mm/s |
| arus | A |
| rpm | putaran/menit |
| tekanan | bar |

Label status: `normal`, `overheat`, `bearing_aus`, `beban_berlebih`.

> Data pada repositori ini **simulasi** (`src/generate_data.py`), sengaja dibuat memuat
> missing value dan duplikat. Untuk data nyata, ganti `data/sensor_motor.csv` dengan
> hasil pembacaan perangkat Anda dengan nama kolom yang sama.

## Pemetaan Materi ke Proyek
| Materi | Implementasi |
|---|---|
| Alur kerja ML (kumpul, preprocessing, split, latih, evaluasi, deploy) | `src/main.py` bagian 1 sampai 7 |
| Preprocessing (missing value, duplikat, normalisasi) | `SimpleImputer`, `drop_duplicates`, `StandardScaler` |
| Split 80:20 | `train_test_split(test_size=0.2, stratify=y)` |
| Supervised: klasifikasi | Decision Tree vs Random Forest |
| Supervised: regresi | Linear Regression memprediksi suhu |
| Unsupervised: clustering | K-Means dengan silhouette score |
| Reduksi dimensi | PCA untuk visualisasi 2D |
| Deteksi anomali | Isolation Forest |
| Overfitting | Selisih akurasi train vs test dan 5-fold cross-validation |
| Deploy / inference | `joblib` menyimpan model, lalu prediksi data sensor baru |

## Cara Menjalankan
```bash
git clone <url-repo-anda>
cd iot-ml-predictive-maintenance
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py    # membuat data simulasi
python src/main.py             # menjalankan seluruh pipeline
```
Keluaran: ringkasan di terminal, grafik di `results/`, model di `models/`.

## Hasil (data simulasi, seed 42)
- **Klasifikasi:** Random Forest akurasi test 0,980 (cv5 0,983); Decision Tree 0,973.
  Akurasi train 1,000 dengan selisih sekitar 0,02 menandakan overfitting ringan,
  sebagaimana diingatkan di materi bahwa akurasi yang terlalu sempurna perlu dicurigai.
- **Regresi:** MAE sekitar 5,2 °C dengan R² sekitar 0,05. Sensor lain kurang menjelaskan
  suhu pada data simulasi ini, jadi model regresi ini lemah. Ini contoh bahwa model yang
  dijalankan belum tentu baik; fitur atau jenis model perlu dievaluasi ulang.
- **K-Means:** silhouette tertinggi pada k=2 (0,503), sedangkan k=4 (0,478) sejalan
  dengan jumlah kondisi mesin.
- **Isolation Forest:** tanpa label, mampu menangkap sekitar 86% kondisi rusak.
- **Inference:** empat sampel sensor baru terklasifikasi sesuai kondisi yang dirancang.

## Keterbatasan
- Data simulasi, sehingga hasil lebih rapi dari kondisi lapangan.
- Belum ada integrasi perangkat keras; alur pengiriman data (MQTT/HTTP) hanya disarankan.
- Perlu validasi pada data nyata sebelum dipakai untuk keputusan perawatan.

## Pengembangan Lanjutan
- Ambil data langsung dari ESP32/Arduino lewat MQTT.
- Uji deep learning (mis. LSTM) sebagai pembanding, dengan menjadikan model konvensional
  sebagai tolok ukur.
- Tambahkan explainable AI (SHAP) dan dasbor pemantauan.

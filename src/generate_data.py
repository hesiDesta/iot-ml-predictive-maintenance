"""Simulasi data sensor IoT pada motor/pompa industri.

Sensor: suhu (C), getaran (mm/s), arus (A), kecepatan putar (RPM), tekanan (bar).
Label status: 0 = normal, 1 = overheat, 2 = bearing aus, 3 = beban berlebih.
Data disimulasikan agar proyek dapat dijalankan tanpa perangkat keras.
Untuk data asli, ganti dengan hasil pembacaan ESP32/Arduino (CSV/MQTT).
"""
import numpy as np
import pandas as pd

LABELS = {0: "normal", 1: "overheat", 2: "bearing_aus", 3: "beban_berlebih"}


def generate(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    status = rng.choice([0, 1, 2, 3], size=n, p=[0.70, 0.10, 0.10, 0.10])

    suhu = rng.normal(55, 4, n)
    getaran = rng.normal(2.0, 0.4, n)
    arus = rng.normal(10, 1.0, n)
    rpm = rng.normal(1450, 25, n)
    tekanan = rng.normal(5.0, 0.3, n)

    m = status == 1  # overheat
    suhu[m] += rng.normal(22, 5, m.sum())
    arus[m] += rng.normal(1.5, 0.6, m.sum())
    m = status == 2  # bearing aus
    getaran[m] += rng.normal(3.5, 0.9, m.sum())
    suhu[m] += rng.normal(6, 3, m.sum())
    m = status == 3  # beban berlebih
    arus[m] += rng.normal(4.5, 1.0, m.sum())
    rpm[m] -= rng.normal(70, 20, m.sum())
    tekanan[m] += rng.normal(1.2, 0.4, m.sum())

    df = pd.DataFrame({
        "suhu": suhu, "getaran": getaran, "arus": arus,
        "rpm": rpm, "tekanan": tekanan, "status": status,
    })

    # Masalah data nyata: missing value & duplikat (sesuai materi preprocessing)
    idx = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[idx, "suhu"] = np.nan
    idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[idx, "getaran"] = np.nan
    df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)
    return df.round(3)


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/sensor_motor.csv", index=False)
    print("Tersimpan: data/sensor_motor.csv", out.shape)

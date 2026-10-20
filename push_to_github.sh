#!/usr/bin/env bash
# Ganti URL dengan repositori kosong milik Anda di GitHub.
set -e
git init
git add .
git commit -m "Studi kasus IoT predictive maintenance dengan machine learning"
git branch -M main
git remote add origin https://github.com/USERNAME/iot-ml-predictive-maintenance.git
git push -u origin main

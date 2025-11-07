# Prediksi Pemasaran Bank

Proyek ini mengimplementasikan dan mengevaluasi tiga model machine learning (K-Nearest Neighbors, Decision Tree, dan Naive Bayes) untuk memprediksi apakah seorang klien akan berlangganan deposito berjangka dalam kampanye pemasaran bank.

## Dataset

Dataset yang digunakan adalah dataset "Bank Marketing" dari UCI Machine Learning Repository. Dataset ini berisi data dari kampanye pemasaran langsung sebuah institusi perbankan di Portugal.

- **File:** `datasets/bank-full.csv`

## Instalasi

Untuk menyiapkan env dan menjalankan proyek, ikuti langkah-langkah berikut:

1.  **Clone repositori ini:**
    ```bash
    git clone https://github.com/n0tavaliduser/bank-marketing-prediction.git
    cd bank-marketing-prediction
    ```

2.  **Buat dan aktifkan env virtual** (disarankan):
    ```bash
    # Untuk conda
    conda create --name <ENV-NAME> python=3.8
    conda activate <ENV-NAME>

    # Untuk venv
    python -m venv venv
    source venv/bin/activate  # Di Windows, gunakan `venv\Scripts\activate`
    ```

3.  **Instal dependensi yang diperlukan:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Instal proyek dalam mode yang dapat diedit (editable mode):**
    Langkah ini akan membuat perintah `run_prediction` yang dapat dijalankan dari baris perintah.
    ```bash
    pip install -e .
    ```

5.  **Jalankan pipeline:**
    Perintah ini akan mengeksekusi keseluruhan pipeline: pra-pemrosesan data, pelatihan model, evaluasi, dan pembuatan semua file output (confusion matrix, skor, dan plot).
    ```bash
    run_prediction
    ```
    Sebagai alternatif, Anda dapat menjalankan skrip utama sebagai modul dari direktori root proyek:
    ```bash
    python -m src.main
    ```

## Hasil

Tabel berikut merangkum kinerja setiap model di berbagai konfigurasi rasio pemisahan data (train-test split) dan validasi silang (k-fold cross-validation). "Akurasi Split" adalah akurasi pada set pengujian, dan "Rata-rata Akurasi CV" adalah akurasi rata-rata dari lipatan validasi silang.

| Model         | Rasio Split | K-Fold | Akurasi Split | Rata-rata Akurasi CV |
|---------------|-------------|--------|---------------|----------------------|
| knn           | 0.7         | 5      | 0.8946        | 0.8360               |
| knn           | 0.7         | 10     | 0.8946        | 0.8518               |
| knn           | 0.8         | 5      | 0.8946        | 0.8394               |
| knn           | 0.8         | 10     | 0.8947        | 0.8491               |
| knn           | 0.9         | 5      | 0.8914        | 0.8360               |
| knn           | 0.9         | 10     | 0.8919        | 0.8537               |
| decision_tree | 0.7         | 5      | 0.8736        | 0.5702               |
| decision_tree | 0.7         | 10     | 0.8736        | 0.6297               |
| decision_tree | 0.8         | 5      | 0.8732        | 0.5702               |
| decision_tree | 0.8         | 10     | 0.8732        | 0.6297               |
| decision_tree | 0.9         | 5      | 0.8678        | 0.5702               |
| decision_tree | 0.9         | 10     | 0.8678        | 0.6297               |
| naive_bayes   | 0.7         | 5      | 0.8278        | 0.7885               |
| naive_bayes   | 0.7         | 10     | 0.8278        | 0.8047               |
| naive_bayes   | 0.8         | 5      | 0.8248        | 0.7885               |
| naive_bayes   | 0.8         | 10     | 0.8248        | 0.8047               |
| naive_bayes   | 0.9         | 5      | 0.8189        | 0.7885               |
| naive_bayes   | 0.9         | 10     | 0.8189        | 0.8047               |
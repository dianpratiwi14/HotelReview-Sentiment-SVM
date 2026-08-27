# Sentiment Analysis of Hotel Reviews Using Support Vector Machine

Laporan Tugas UAS Mata Kuliah Pembelajaran Mesin — analisis sentimen ulasan hotel menggunakan pendekatan _machine learning_, dengan **Support Vector Machine (SVM)** sebagai model utama dan Naive Bayes serta Logistic Regression sebagai pembanding.

## 📋 Daftar Isi

- [Deskripsi Project](#-deskripsi-project)
- [Tujuan](#-tujuan)
- [Dataset](#-dataset)
- [Metodologi](#-metodologi)
- [Struktur Repository](#-struktur-repository)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil](#-hasil)
- [Referensi](#-referensi)
- [Kontributor](#-kontributor)

## 📖 Deskripsi Project

Project ini merupakan tugas akhir (UAS) mata kuliah Pembelajaran Mesin yang bertujuan membangun sistem analisis sentimen otomatis terhadap ulasan hotel. Ulasan diklasifikasikan ke dalam dua kelas: **positif** dan **negatif**, menggunakan algoritma Support Vector Machine (SVM) yang dibandingkan performanya dengan Naive Bayes dan Logistic Regression.

## 🎯 Tujuan

1. Mengembangkan model analisis sentimen menggunakan Support Vector Machine (SVM) untuk mengklasifikasikan ulasan hotel secara otomatis.
2. Mempermudah calon pengunjung hotel dalam mengevaluasi dan memilih hotel berdasarkan ulasan online.
3. Menguji keandalan dan akurasi metode SVM dalam analisis sentimen ulasan hotel.

## 📊 Dataset

Dataset yang digunakan adalah **[Trip Advisor Hotel Reviews](https://www.kaggle.com/code/qusaybtoush1990/trip-advisor-hotel-reviews)** yang diambil dari Kaggle.

| Detail       | Keterangan                                         |
| ------------ | -------------------------------------------------- |
| Jumlah baris | 20.491 ulasan                                      |
| Jumlah kolom | 2 (teks ulasan & rating)                           |
| Rating       | Skala 1–5                                          |
| Label        | 1 = Positif (rating ≥ 3), 0 = Negatif (rating < 3) |

> Dataset tersedia di folder [`dataset/`](./dataset) atau dapat diunduh langsung dari Kaggle.

## 🔬 Metodologi

Alur penelitian mengikuti tahapan berikut:

### 1. Collecting Data

Mengambil data ulasan hotel dari Kaggle (20.491 baris).

### 2. Data Preprocessing

- **Standardization** — menghapus karakter non-alfabet, mengubah teks menjadi huruf kecil.
- **Stop Words Removal** — menghapus kata-kata umum yang tidak berkontribusi pada makna.
- **Stemming** — mengubah kata ke bentuk dasarnya.
- **Labeling** — rating ≥ 3 → label 1 (positif), rating < 3 → label 0 (negatif).

### 3. Representasi Data

Menggunakan **TF-IDF (Term Frequency–Inverse Document Frequency)** untuk mengubah teks menjadi vektor numerik.

### 4. Pembagian Data

Data dibagi menjadi data training dan testing. Untuk mendapatkan performa terbaik, dilakukan eksperimen dengan beberapa skema rasio pembagian data serta beberapa nilai `random_state` pada tiap skema:

| Skema   | Test Size | Rasio Train : Test |
| ------- | --------- | ------------------ |
| Skema 1 | 0.1936    | ≈ 80.6% : 19.4%    |
| Skema 2 | 0.20      | 80% : 20%          |
| Skema 3 | 0.25      | 75% : 25%          |
| Skema 4 | 0.30      | 70% : 30%          |

Hasil terbaik secara konsisten diperoleh pada **Skema 1 (test size = 0.1936)**, sehingga skema ini digunakan sebagai acuan hasil akhir pada bagian [Hasil](#-hasil).

### 5. Pelatihan Model

Tiga model dilatih dan dibandingkan:

- **Naive Bayes** — model probabilistik berbasis Teorema Bayes.
- **Support Vector Machine (SVM)** — mencari hyperplane optimal dengan margin terbesar (model utama).
- **Logistic Regression** — model regresi untuk klasifikasi biner.

Pada percobaan awal (text size = 0.2, random state = 10), kernel **linear** menghasilkan F1-score (0.76) dan recall (0.71) yang sedikit lebih tinggi dibanding kernel RBF, sehingga kernel linear dipilih sebagai kernel SVM utama untuk percobaan selanjutnya.

### 6. Evaluasi Model

Evaluasi menggunakan **confusion matrix**, dengan metrik: **akurasi, presisi, recall, dan F1-score**.

### 7. Perbandingan Hasil

Model dengan performa terbaik dipilih berdasarkan keseimbangan keempat metrik evaluasi di atas.

## 📁 Struktur Repository

```
HotelReview-Sentiment-SVM/
│
├── dataset/
│   └── hotel.csv
│
├── notebook/
│   └── sentiment_analysis_svm.ipynb
│
├── report/
│   ├── 20_ProjectUASFinal.pdf
│   └── paper_referensi.pdf
│
├── images/
│   └── metodologi_penelitian.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Instalasi

1. Clone repository ini:

   ```bash
   git clone https://github.com/dianpratiwi14/HotelReview-Sentiment-SVM.git
   cd HotelReview-Sentiment-SVM
   ```

2. (Opsional) Buat virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Cara Menjalankan

1. Pastikan dataset (`tripadvisor_hotel_reviews.csv`) berada di folder `dataset/`.
2. Buka notebook dengan Jupyter:
   ```bash
   jupyter notebook notebook/sentiment_analysis_svm.ipynb
   ```
3. Jalankan seluruh cell secara berurutan (Run All) untuk melihat proses preprocessing, training, hingga evaluasi model.

## 📈 Hasil

### Perbandingan Kernel SVM

Percobaan awal (text size = 0.2, random state = 10) membandingkan tiga kernel SVM. Kernel **linear** menghasilkan F1-score dan recall yang sedikit lebih tinggi dibanding kernel **RBF**, sehingga dipilih sebagai kernel utama. Kernel **sigmoid** juga diuji sebagai pembanding.

### Analisa Masalah

Pada percobaan awal dengan satu kombinasi `random_state` dan `test_size` tetap, hasil Naive Bayes tampak jauh lebih rendah secara tidak proporsional dibanding SVM dan Logistic Regression. Peneliti mengidentifikasi bahwa nilai `random_state` dan `test_size` yang digunakan dapat memengaruhi hasil evaluasi, sehingga diperlukan percobaan ulang dengan variasi kombinasi keduanya untuk memastikan hasil yang lebih representatif.

### Solusi: Percobaan dengan Berbagai Random State

Percobaan pertama dilakukan dengan memvariasikan `random_state` (0, 10, 42) pada `test_size` tetap, namun belum menunjukkan perbedaan akurasi yang signifikan antar `random_state`.

### Solusi: Percobaan dengan Berbagai Text Size & Random State

Percobaan lanjutan memvariasikan **text size** (0.1936, 0.2, 0.25, 0.3) **dan** `random_state` (0, 10, 42) untuk setiap model:

| Model               | Text Size  | Random State 0 | Random State 10 | Random State 42 |
| ------------------- | ---------- | -------------- | --------------- | --------------- |
| **SVM**             | **0.1936** | **0.94**       | 0.93            | 0.93            |
| SVM                 | 0.2        | 0.93           | 0.93            | 0.93            |
| SVM                 | 0.25       | 0.93           | 0.93            | 0.93            |
| SVM                 | 0.3        | 0.93           | 0.93            | 0.93            |
| Logistic Regression | 0.1936     | 0.93           | 0.93            | 0.93            |
| Logistic Regression | 0.2        | 0.93           | 0.93            | 0.93            |
| Logistic Regression | 0.25       | 0.93           | 0.93            | 0.93            |
| Logistic Regression | 0.3        | 0.93           | 0.93            | 0.93            |
| Naive Bayes         | 0.1936     | 0.66           | 0.66            | 0.66            |
| Naive Bayes         | 0.2        | 0.67           | 0.66            | 0.66            |
| Naive Bayes         | 0.25       | 0.67           | 0.66            | 0.66            |
| Naive Bayes         | 0.3        | 0.66           | 0.66            | 0.66            |

### Kesimpulan

Dari seluruh percobaan, model **SVM dengan kernel linear, text size = 0.1936, dan random_state = 0** menghasilkan akurasi tertinggi yaitu **0.94 (94%)**, mengungguli Logistic Regression (0.93) dan Naive Bayes (0.66–0.67) secara konsisten di seluruh kombinasi skema yang diuji.

Detail classification report untuk model terbaik (SVM, text size = 0.1936, random_state = 0):

| Kelas            | Precision | Recall   | F1-Score | Support  |
| ---------------- | --------- | -------- | -------- | -------- |
| 0 (Negatif)      | 0.83      | 0.74     | 0.78     | 620      |
| 1 (Positif)      | 0.95      | 0.97     | 0.96     | 3348     |
| **Weighted Avg** | **0.93**  | **0.94** | **0.93** | **3968** |

## 📚 Referensi

- Dataset: [Trip Advisor Hotel Reviews - Kaggle](https://www.kaggle.com/code/qusaybtoush1990/trip-advisor-hotel-reviews)
- Paper referensi tersedia di folder [`report/`](./report)

## 👥 Kontributor

Tugas ini dikerjakan secara berkelompok untuk memenuhi UAS Mata Kuliah Pembelajaran Mesin.

**Kelompok 20**

1. Filzah Syakirah
2. Dian Pratiwi

---

_Laporan ini disusun untuk memenuhi Tugas UAS Mata Kuliah Pembelajaran Mesin._

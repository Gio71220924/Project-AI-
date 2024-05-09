import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

# Baca data dari file JSON
with open('data.json', 'r') as file:
    data = json.load(file)

# Persiapan data
features = []  # untuk menyimpan fitur (mata kuliah)
labels = []    # untuk menyimpan label (profil lulusan)

# Iterasi melalui setiap entri dalam data
for alumni in data['alumni']:
    # Masukkan nilai mata kuliah ke dalam list features
    features.append(list(alumni['mata_kuliah'].values()))
    # Masukkan profil lulusan ke dalam list labels
    labels.append(alumni['profil_lulusan'])

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Cetak jumlah data latih dan data uji
print("Jumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))

# Cari fitur-fitur yang memiliki setidaknya satu nilai yang diamati dalam dataset latih
fitur_ada = np.any(X_train, axis=0)
fitur_ada_index = np.where(fitur_ada)[0]

# Filter data latih dan uji hanya untuk fitur-fitur yang ada nilai yang diamatinya
X_train_filtered = np.array(X_train)[:, fitur_ada_index]
X_test_filtered = np.array(X_test)[:, fitur_ada_index]

# Inisialisasi model Naive Bayes Gaussian
model = GaussianNB()

# Latih model menggunakan data latih yang telah difilter
model.fit(X_train_filtered, y_train)

# Lakukan prediksi menggunakan data uji yang telah difilter
y_pred = model.predict(X_test_filtered)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi prediksi:", accuracy)

# Baca data uji dari file JSON
with open('uji.json', 'r') as file:
    data_uji = json.load(file)

# Persiapan data mahasiswa untuk prediksi
mahasiswa_baru = []

# Iterasi melalui setiap entri dalam data uji
for mahasiswa in data_uji['mahasiswa']:
    # Inisialisasi nilai mata kuliah untuk mahasiswa baru
    nilai_mata_kuliah = [mahasiswa['mata_kuliah'].get(mk, np.nan) for mk in mahasiswa['mata_kuliah'].keys()]

    # Hanya ambil nilai untuk fitur-fitur yang ada nilai yang diamatinya dalam dataset latih
    nilai_mata_kuliah_filtered = [nilai_mata_kuliah[i] for i in fitur_ada_index]

    # Masukkan nilai mata kuliah ke dalam list mahasiswa_baru
    mahasiswa_baru.append(nilai_mata_kuliah_filtered)

# Ubah menjadi array numpy
mahasiswa_baru = np.array(mahasiswa_baru)

# Inisialisasi imputer dengan strategi mean
imputer = SimpleImputer(strategy='mean')

# Imputasi nilai-nilai yang kosong pada data uji
mahasiswa_baru_imputed = imputer.fit_transform(mahasiswa_baru)

# Lakukan prediksi profil lulusan untuk mahasiswa baru yang telah diimputasi
prediksi_profil_imputed = model.predict(mahasiswa_baru_imputed)

# Tampilkan hasil prediksi
print("Hasil Prediksi Profil Lulusan Mahasiswa Baru:")
for i, mahasiswa in enumerate(data_uji['mahasiswa']):
    print(f"Nama: {mahasiswa['nama']}, Tahun Masuk: {mahasiswa['tahun_masuk']}, Prediksi Profil: {prediksi_profil_imputed[i]}")

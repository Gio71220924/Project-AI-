import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

# Baca data dari file JSON
with open('databaru.json', 'r') as file:
    data = json.load(file)

# Persiapan data
features = []  # untuk menyimpan fitur (mata kuliah)
labels = []    # untuk menyimpan label (profil lulusan)

# Cari semua mata kuliah yang ada dalam dataset
all_mata_kuliah = set()
for alumni in data['alumni']:
    all_mata_kuliah.update(alumni['mata_kuliah'].keys())

# Buat list dari semua mata kuliah
all_mata_kuliah = list(all_mata_kuliah)

# Iterasi melalui setiap entri dalam data
for alumni in data['alumni']:
    # Inisialisasi nilai mata kuliah dengan NaN
    nilai_mata_kuliah = [np.nan] * len(all_mata_kuliah)
    for i, mk in enumerate(all_mata_kuliah):
        if mk in alumni['mata_kuliah']:
            nilai_mata_kuliah[i] = alumni['mata_kuliah'][mk]
    features.append(nilai_mata_kuliah)
    labels.append(alumni['profil_lulusan'])

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)

# Cetak jumlah data latih dan data uji
print("Jumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))

# Inisialisasi imputer dengan strategi mean
imputer = SimpleImputer(strategy='mean')

# Imputasi nilai-nilai yang kosong pada data latih dan uji
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Inisialisasi model Naive Bayes Gaussian
model = GaussianNB()

# Latih model menggunakan data latih yang telah diimputasi
model.fit(X_train_imputed, y_train)

# Lakukan prediksi menggunakan data uji yang telah diimputasi
y_pred = model.predict(X_test_imputed)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi prediksi:", accuracy * 100, "%")

# Baca data uji dari file JSON
with open('uji.json', 'r') as file:
    data_uji = json.load(file)

# Persiapan data mahasiswa untuk prediksi
mahasiswa_baru = []

# Iterasi melalui setiap entri dalam data uji
for mahasiswa in data_uji['mahasiswa']:
    # Inisialisasi nilai mata kuliah dengan NaN
    nilai_mata_kuliah = [np.nan] * len(all_mata_kuliah)
    for i, mk in enumerate(all_mata_kuliah):
        if mk in mahasiswa['mata_kuliah']:
            nilai_mata_kuliah[i] = mahasiswa['mata_kuliah'][mk]
    mahasiswa_baru.append(nilai_mata_kuliah)

# Ubah menjadi array numpy
mahasiswa_baru = np.array(mahasiswa_baru)

# Imputasi nilai-nilai yang kosong pada data mahasiswa baru
mahasiswa_baru_imputed = imputer.transform(mahasiswa_baru)


# Imputasi nilai-nilai yang kosong pada data mahasiswa baru
mahasiswa_baru_imputed = imputer.transform(mahasiswa_baru)

# Lakukan prediksi probabilitas untuk setiap profil lulusan pada data mahasiswa baru yang telah diimputasi
probabilities = model.predict_proba(mahasiswa_baru_imputed)

# Tampilkan hasil prediksi
print("Hasil Prediksi Profil Lulusan Mahasiswa Baru:")
for i, mahasiswa in enumerate(data_uji['mahasiswa']):
    print(f"Nama: {mahasiswa['nama']}, Tahun Masuk: {mahasiswa['tahun_masuk']}")
    print("Prediksi Profil Lulusan dan Persentase:")
    for j, prob in enumerate(probabilities[i]):
        profile = model.classes_[j]
        print(f"- {profile}: {prob * 100:.2f}%")


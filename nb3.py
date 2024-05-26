import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Baca data dari file JSON
with open('databaru.json', 'r') as file:
    data = json.load(file)

# Persiapan data
features = []  # untuk menyimpan fitur (mata kuliah)
labels = []    # untuk menyimpan label (profil lulusan)
all_subjects = set()

# Iterasi melalui setiap entri dalam data
for alumni in data['alumni']:
    mata_kuliah = alumni['mata_kuliah']
    all_subjects.update(mata_kuliah.keys())
    features.append(mata_kuliah)
    labels.append(alumni['profil_lulusan'])

all_subjects = list(all_subjects)

# Buat array fitur yang seragam
features_array = []

for mata_kuliah in features:
    row = [mata_kuliah.get(subject, np.nan) for subject in all_subjects]
    features_array.append(row)

features_array = np.array(features_array)

# Penskalaan data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)

# Imputasi nilai-nilai yang hilang
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_scaled)

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(features_imputed, labels, test_size=0.4, random_state=42)

# Inisialisasi model Naive Bayes Gaussian
model = GaussianNB()

# Latih model menggunakan data latih yang telah diimputasi
model.fit(X_train, y_train)

# Lakukan prediksi menggunakan data uji yang telah diimputasi
y_pred = model.predict(X_test)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi prediksi:", accuracy*100, "%")

# Evaluasi menggunakan cross-validation
scores = cross_val_score(model, features_imputed, labels, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean cross-validation accuracy:", scores.mean())

# Baca data uji dari file JSON
with open('uji.json', 'r') as file:
    data_uji = json.load(file)

# Persiapan data mahasiswa untuk prediksi
mahasiswa_baru = []

# Iterasi melalui setiap entri dalam data uji
for mahasiswa in data_uji['mahasiswa']:
    # Inisialisasi nilai mata kuliah untuk mahasiswa baru
    nilai_mata_kuliah = [mahasiswa['mata_kuliah'].get(subject, np.nan) for subject in all_subjects]
    mahasiswa_baru.append(nilai_mata_kuliah)

# Ubah menjadi array numpy
mahasiswa_baru = np.array(mahasiswa_baru)

# Penskalaan dan imputasi data uji
mahasiswa_baru_scaled = scaler.transform(mahasiswa_baru)
mahasiswa_baru_imputed = imputer.transform(mahasiswa_baru_scaled)

# Lakukan prediksi profil lulusan untuk mahasiswa baru yang telah diimputasi
prediksi_profil_imputed = model.predict(mahasiswa_baru_imputed)

# Tampilkan hasil prediksi
print("Hasil Prediksi Profil Lulusan Mahasiswa Baru:")
for i, mahasiswa in enumerate(data_uji['mahasiswa']):
    print(f"Nama: {mahasiswa['nama']}, Tahun Masuk: {mahasiswa['tahun_masuk']}, Prediksi Profil: {prediksi_profil_imputed[i]}")

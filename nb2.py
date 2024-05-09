import json
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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

# Hitung komponen utama dari data latih
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Inisialisasi model Naive Bayes Gaussian
model = GaussianNB()

# Latih model menggunakan data latih yang telah ditransformasi
model.fit(X_train_pca, y_train)

# Lakukan prediksi menggunakan data uji yang telah ditransformasi
y_pred = model.predict(X_test_pca)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi prediksi setelah reduksi dimensi:", accuracy)

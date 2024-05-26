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

# Fungsi untuk melatih dan menguji model dengan parameter yang diberikan
def evaluate_model(test_size, random_state):
    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

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
    return accuracy

# Coba beberapa kombinasi test_size dan random_state
test_sizes = [0.2, 0.3, 0.4, 0.5]
random_states = [42, 0, 7, 21]

best_accuracy = 0
best_params = {}

results = []

for test_size in test_sizes:
    for random_state in random_states:
        accuracy = evaluate_model(test_size, random_state)
        results.append((test_size, random_state, accuracy))
        print(f"Test Size: {test_size}, Random State: {random_state}, Accuracy: {accuracy*100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'test_size': test_size, 'random_state': random_state}

print("Best Parameters:")
print(best_params)
print(f"Best Accuracy: {best_accuracy*100:.2f}%")

# Tampilkan semua hasil
print("\nAll Results:")
for result in results:
    print(f"Test Size: {result[0]}, Random State: {result[1]}, Accuracy: {result[2]*100:.2f}%")

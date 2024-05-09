import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import json

# 1. Baca dataset dari file JSON
with open('alumni.json', 'r') as file:
    data = json.load(file)

# 2. Konversi data ke DataFrame Pandas
df = pd.DataFrame(data['alumni'])

# 3. Pisahkan kolom target 'profil_lulusan'
y = df['profil_lulusan']

# 4. Hapus kolom 'profil_lulusan' dari DataFrame
df.drop(columns=['profil_lulusan'], inplace=True)

# 5. Konversi fitur kategorikal menjadi numerik (misalnya, nama mahasiswa)
df_encoded = pd.get_dummies(df, columns=['nama'])

# 6. Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# 7. Inisialisasi dan latih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 8. Lakukan prediksi pada data uji
y_pred = model.predict(X_test)

# 9. Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model Naive Bayes:", accuracy)

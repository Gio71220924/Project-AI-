import json

# 1. Membaca data dari file JSON
with open('alumni.json', 'r') as file:
    data_alumni = json.load(file)

# 2. Membaca data mahasiswa baru
def baca_data_mahasiswa(nama_file):
    with open(nama_file, 'r') as file:
        data = json.load(file)
    return data

# 3. Memperhitungkan kesesuaian profil dengan metode AI (contoh sederhana)
def hitung_kesesuaian(mahasiswa, alumni):
    kesesuaian = 0
    for mata_kuliah, nilai in mahasiswa['KHS'].items():
        if mata_kuliah in alumni['mata_kuliah']:
            kesesuaian += (nilai / 100) * (alumni['mata_kuliah'][mata_kuliah] / 100)
    return kesesuaian

# 4. Memperkirakan profil lulusan yang paling cocok
def prediksi_profil_lulusan(mahasiswa, data_alumni):
    skor_max = -1
    profil_terbaik = None
    for alumni in data_alumni["alumni"]:
        skor_kesesuaian = hitung_kesesuaian(mahasiswa, alumni)
        if skor_kesesuaian > skor_max:
            skor_max = skor_kesesuaian
            profil_terbaik = alumni['profil_lulusan']
    return profil_terbaik

# 5. Menampilkan hasil prediksi
def tampilkan_hasil_prediksi(nama_mahasiswa, profil_prediksi):
    print(f"Berdasarkan data, prediksi untuk mahasiswa {nama_mahasiswa}:")
    print(f"Profil lulusan yang paling cocok: {profil_prediksi}")

# Baca data mahasiswa baru
data_mahasiswa = baca_data_mahasiswa('mahasiswa_baru.json')

# Lakukan prediksi untuk setiap mahasiswa baru
for mahasiswa in data_mahasiswa["mahasiswa"]:
    profil_prediksi = prediksi_profil_lulusan(mahasiswa, data_alumni)
    tampilkan_hasil_prediksi(mahasiswa['nama'], profil_prediksi)

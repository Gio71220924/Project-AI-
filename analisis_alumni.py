import json
from statistics import median
from collections import Counter
import csv

# 1. Membaca data dari file JSON
with open('databaru.json', 'r') as file:
    data = json.load(file)

# 2. Menghitung jumlah alumni untuk setiap profil lulusan
profil_count = {}
for alumni in data["alumni"]:
    if 'profil_lulusan' in alumni:
        profil = alumni['profil_lulusan']
        if profil in profil_count:
            profil_count[profil] += 1
        else:
            profil_count[profil] = 1
    else:
        print("Kunci 'profil_lulusan' tidak ditemukan untuk salah satu entri alumni.")

# 3. Menghitung total alumni
total_alumni = sum(profil_count.values())

# 4. Menghitung persentase alumni untuk setiap profil lulusan
if profil_count:  # Memeriksa apakah ada data alumni
    profil_percentage = {profil: (count / total_alumni) * 100 for profil, count in profil_count.items()}

    # 5. Menemukan profil lulusan dengan persentase tertinggi
    max_percentage_profile = max(profil_percentage, key=profil_percentage.get)
    max_percentage = profil_percentage[max_percentage_profile]

    # Menampilkan hasil profil lulusan dengan minat alumni paling tinggi
    print("Profil lulusan dengan minat alumni paling tinggi:")
    print(f"Profil: {max_percentage_profile}")
    print(f"Persentase: {max_percentage:.2f}%")

    # 6. Menampilkan persentase untuk semua profil lulusan
    print("\nPersentase semua profil lulusan:")
    for profil, percentage in profil_percentage.items():
        print(f"Profil: {profil}, Persentase: {percentage:.2f}%")
else:
    print("Data alumni kosong. Tidak dapat menghitung minat alumni.")

# 7. Menghitung rata-rata IPK dari data mata kuliah
total_ipk = 0
jumlah_mahasiswa = 0
for alumni in data["alumni"]:
    if 'ipk' in alumni:
        total_ipk += alumni['ipk']
        jumlah_mahasiswa += 1

if jumlah_mahasiswa > 0:
    rata_ipk = total_ipk / jumlah_mahasiswa
    print(f"\nRata-rata IPK dari semua mahasiswa: {rata_ipk:.2f}")
    print(f"Median IPK dari semua mahasiswa: {median([alumni['ipk'] for alumni in data['alumni']]):.2f}")

    # Modus IPK dari semua mahasiswa
    ipk_counter = Counter([alumni['ipk'] for alumni in data['alumni']])
    modus_ipk = ipk_counter.most_common(1)[0][0]
    print(f"Modus IPK dari semua mahasiswa: {modus_ipk:.2f}")
else:
    print("Tidak ada data IPK yang tersedia.")

# 8. Menampilkan daftar mata kuliah yang diambil oleh setiap alumni
# print("\nDaftar mata kuliah yang diambil oleh setiap alumni:")
# for alumni in data["alumni"]:
#     print(f"Nama: {alumni['nama']}, Mata Kuliah: {list(alumni['mata_kuliah'].keys())}")

# 9. Menghitung rata-rata nilai mata kuliah untuk setiap profil lulusan
nilai_mata_kuliah = {}
for alumni in data["alumni"]:
    profil = alumni['profil_lulusan']
    if profil not in nilai_mata_kuliah:
        nilai_mata_kuliah[profil] = []
    for nilai in alumni['mata_kuliah'].values():
        nilai_mata_kuliah[profil].append(nilai)

print("\nRata-rata nilai mata kuliah untuk setiap profil lulusan:")
for profil, nilai in nilai_mata_kuliah.items():
    rata_nilai = sum(nilai) / len(nilai)
    print(f"Profil: {profil}, Rata-rata Nilai: {rata_nilai:.2f}")

# 10. Menyimpan hasil analisis ke file CSV
with open('hasil_analisis.csv', 'w', newline='') as csvfile:
    fieldnames = ['profil_lulusan', 'jumlah_alumni', 'persentase', 'rata_ipk']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for profil, count in profil_count.items():
        persentase = profil_percentage[profil]
        writer.writerow({'profil_lulusan': profil, 'jumlah_alumni': count, 'persentase': persentase, 'rata_ipk': rata_ipk})



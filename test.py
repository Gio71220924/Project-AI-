from naive_bayes import *

def main() -> None:
    dataset : list = get_dataset()
    seperated_dataset : list = seperate_based_on_class(dataset, "profil_lulusan")
    dataprediski : dict = get_data_prediksi("kasus/mahasiswa1.json")

    for mahasiswa in dataprediski:
        datauji = mahasiswa["mata_kuliah"]
        hasil = do_naive_bayes(seperated_dataset, datauji)
        
        max_val = hasil["UI/UX"]
        kunci = "UI/UX"
        for i, val in hasil.items():
            if max_val < val:
                max_val = val
                kunci = i
            print(f"{i}\t : {val:.2f}")
        
        print(f"Mahasiswa: {mahasiswa['nama']}, Prediksi: {kunci}")

if __name__ == "__main__":
    main()
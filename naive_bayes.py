import json

def print_cantik(hashmap) -> None:
    pretty_dict = json.dumps(hashmap, indent=4)
    print(pretty_dict)

## Mengambil dataset dari newdata.json
def get_dataset() -> list:
    with open("data/newdata.json", "r") as f:
        data = f.read()
        kamus_x = json.loads(data)
    kamus = []
    for alumni in kamus_x["alumni"]:
        alumni["mata_kuliah"]["profil_lulusan"] = alumni["profil_lulusan"]
        kamus.append(alumni["mata_kuliah"])
    return kamus

## Mengambil data yang akan diprediksi dari uji.json
def get_data_prediksi(name_data) -> dict:
    with open(name_data, "r") as f:
        data = f.read()
        field = json.loads(data)
    return field['mahasiswa']
    
    # with open(name_data, "r") as f:
    #     data = f.read()
    #     field = json.loads(data)
    # mahasiswa = field['mahasiswa'][0]
    # return mahasiswa["mata_kuliah"]

## Memisahkan dataset berdasarkan kelas profil_lulusan
def seperate_based_on_class(data: list, target: str) -> dict:
    class_seperated = {}
    for x in data:
        if x[target] not in class_seperated:
            class_seperated[x[target]] = [x]
        else:
            class_seperated[x[target]].append(x)
    return class_seperated

## Menghitung probabilitas
def get_probability(dataset: dict, feature: str, value: int = None, given: int = None) -> float:
    total = 1
    len_all = 0
    for target, data in dataset.items():
        len_all += len(data)
        if given and target != given:
            continue
        for case in data:
            try:
                if int(case[feature]) == int(value):
                    total += 1
            except:
                continue
    probability = (total / len_all) * 100
    return probability

## Mencari prior probability untuk setiap fitur dan kelas
def get_prior_probability(dataset: dict, target) -> float:
    len_all = 0
    for key, data in dataset.items():
        len_all += len(data)
    return len(dataset[target]) / len_all

## Naive Bayes
def do_naive_bayes(dataset: dict, features: dict) -> dict:
    probabilities = {}
    for target in dataset:
        probabilities[target] = get_prior_probability(dataset, target)
        for feature, value in features.items():
            probabilities[target] *= get_probability(dataset, feature, value, target)
    return probabilities
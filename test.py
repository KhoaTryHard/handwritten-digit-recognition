# Kiem tra so luong mau EMNIST.
# Cach 1: Dung tensorflow_datasets INFO file da co tren disk (neu da tung download)
import json, pathlib, os

emnist_dir = pathlib.Path(os.path.expanduser("~")) / "tensorflow_datasets" / "emnist" / "digits"
candidates = sorted(emnist_dir.glob("*/dataset_info.json")) if emnist_dir.exists() else []

if candidates:
    info = json.loads(candidates[-1].read_text())
    splits = {s["name"]: s["statistics"]["numExamples"] for s in info["splits"]}
    print(f"EMNIST digits (from local cache): {splits}")
else:
    print("EMNIST chưa được download về máy — dùng con số từ nguồn chính thức bên dưới")

# Cach 2: fetch thang tu TF Datasets catalog (khong can cai gi ca)
import urllib.request
url = "https://storage.googleapis.com/tfds-data/dataset_info/emnist/digits/3.0.0/dataset_info.json"
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        info = json.loads(r.read())
    splits = {s["name"]: s["statistics"]["numExamples"] for s in info["splits"]}
    print(f"EMNIST digits (tu TF catalog): {splits}")
except Exception as e:
    print(f"Khong fetch duoc: {e}")
    print("Con so chinh thuc (tu GitHub TF datasets): train=240000, test=40000")

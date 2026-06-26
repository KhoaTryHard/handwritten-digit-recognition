# Huong Dan Project

## Code Style
- Uu tien `project_paths.project_file()` va `project_paths.project_path()` cho duong dan tinh tu repository; khong hardcode absolute path.
- Giu root-level script gon. Dua logic tai su dung vao `digit_pipeline/` va compose tu cac entrypoint.
- Dung constant cap module o dau script cho cac gia tri cau hinh nhu batch size, epochs, paths va thresholds.
- Trong cac script import TensorFlow, dat `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"` truoc khi import TensorFlow.
- Tuan theo flow anh grayscale 28x28 va preprocessing float32 da chuan hoa hien co trong `digit_pipeline/data.py` va `digit_pipeline/preprocessing.py`.

## Kien Truc
- `digit_pipeline/` chua cac helper dung chung cho data loading, preprocessing, model, training va evaluation.
- Root script cai dat stage workflow va analysis entrypoint da mo ta trong [README.md](../README.md).
- `models/` luu artifact `.keras`, va `reports/` luu CSV da export cung anh confusion-pair.

## Build Va Test
- Khong co build step rieng.
- Chay script lien quan tu repository root de validate thay doi, theo workflow trong [README.md](../README.md).
- Voi thay doi fine-tuning, dung cac stage script theo thu tu: `train_mnist_base.py`, `finetune_emnist.py`, `split_personal_data.py`, `convert_personal_data_to_28.py`, `finetune_personal_data.py`, `finetune_legacy_data.py`.

## Quy Uoc
- Thu muc dataset phai duoc to chuc theo ten lop: `0/` den `9/`.
- Validation data co the nap tu thu muc `val/` rieng; neu thu muc nay rong, code fallback sang automatic train/validation split voi seed co dinh.
- So lop output cua model phai khop voi so lop cua dataset.
- Neu path, ten artifact hoac workflow step thay doi, cap nhat README thay vi lap lai toan bo giai thich o day.

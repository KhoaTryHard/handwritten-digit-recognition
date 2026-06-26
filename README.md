# Nhan Dang Chu So Viet Tay

Pipeline nhan dang chu so viet tay dua tren CNN, duoc xay dung voi MNIST, EMNIST va bo du lieu ca nhan 28x28.

## Tong Quan

Du an nay train bo phan lop chu so theo ba stage:

1. Train CNN nen tren MNIST.
2. Fine-tune model nen tren EMNIST digits.
3. Fine-tune tiep tren bo du lieu chu so viet tay ca nhan da chuyen ve dinh dang 28x28 thong nhat.

Model cuoi duoc dung cho prediction, evaluation, cac script phan tich va app demo local trong repository nay.

## Training Pipeline

### Stage 1: Train tren MNIST

Script:
- `python train_mnist_base.py`

Output:
- `models/stage_01_mnist_base.keras`

### Stage 2: Fine-tune tren EMNIST

Script:
- `python finetune_emnist.py`

Output:
- `models/stage_02_emnist_finetuned.keras`

### Stage 3: Fine-tune tren du lieu ca nhan

Script:
- `python finetune_personal_data.py`

Input dataset:
- `my_digits_28/train`
- `my_digits_28/val`

Output:
- `models/stage_03_final.keras`

## Cau Truc Repository

```text
.
|- digit_pipeline/              Helper dung chung cho preprocessing, training, data va evaluation
|- models/                      Artifact model chuan da luu cho workflow 3 stage
|- my_digits_new/               Dataset chu so viet tay ca nhan dang raw
|- my_digits_28/                Dataset ca nhan da chuyen ve 28x28
|- reports/                     Bao cao va ket qua export phan tich
|- train_mnist_base.py          Training stage 1
|- finetune_emnist.py           Training stage 2
|- finetune_personal_data.py    Training stage 3
|- split_personal_data.py       Tach du lieu ca nhan raw thanh train/val
|- convert_personal_data_to_28.py
|- predict_digit_app.py         App demo local co trinh chon anh
|- predict_digit_image.py       Du doan mot anh don le
|- analyze_confusion_matrix.py
|- export_misclassified_csv.py
|- export_confusion_pair_images.py
|- preview_mnist_samples.py
|- requirements.txt
`- README.md
```

## Huong Dan File

Mo ta nhanh cong dung cua cac tep chinh o thu muc goc cua du an.

- `project_paths.py`: Cung cap cac helper nho de tao duong dan tinh tu thu muc goc cua project.
- `analyze_confusion_matrix.py`: Chay model cuoi tren tap validation, in confusion matrix, bao cao accuracy theo tung lop va liet ke cac cap nham lan xuat hien nhieu nhat.
- `convert_personal_data_to_28.py`: Chuyen anh chu so viet tay ca nhan thanh validation dataset 28x28 giong MNIST voi nguong preprocessing da cau hinh.
- `export_confusion_pair_images.py`: Xuat anh cua mot cap nham lan true-to-predicted cu the de co the xem loi truc quan.
- `export_misclassified_csv.py`: Xuat toan bo mau validation bi du doan sai ra file CSV kem duong dan file, label, probability va top-3 prediction.
- `finetune_emnist.py`: Fine-tune model MNIST stage 1 tren EMNIST digits voi augmentation va luu `models/stage_02_emnist_finetuned.keras`.
- `finetune_personal_data.py`: Fine-tune model stage 2 tren `my_digits_28/train` va `my_digits_28/val` de tao model cuoi phu hop voi net viet ca nhan.
- `predict_digit_app.py`: Mo app desktop local de chon anh, xem preview sau preprocessing va xem ket qua prediction truc quan.
- `predict_digit_image.py`: Chay cung inference pipeline o dang script, lay anh tu bien `IMAGE_PATH` da cau hinh hoac tu duong dan CLI.
- `preview_mnist_samples.py`: Hien thi anh mau MNIST cua mot chu so duoc chon de ho tro so sanh truc quan nhanh.
- `project_report_sections.md`: Luu cac phan bao cao nhap ve phuong phap nghien cuu va quy trinh xay dung he thong.
- `README.md`: Tai lieu chinh cua du an, gom tong quan, buoc cai dat, workflow va command su dung.
- `requirements.txt`: Liet ke cac dependency Python can cho training, prediction, visualization va analysis script.
- `split_personal_data.py`: Chuyen mot phan dataset ca nhan raw tu `my_digits_new/train` sang `my_digits_new/val` de tao validation split.
- `train_mnist_base.py`: Train CNN nen tren MNIST tu dau va luu `models/stage_01_mnist_base.keras`.

## Cai Dat

Tao va kich hoat virtual environment, sau do cai dat dependency:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependency chinh:
- TensorFlow
- TensorFlow Datasets
- NumPy
- Pillow
- Pandas
- Matplotlib

## Chuan Bi Du Lieu

### Tach du lieu ca nhan raw

Dung script nay de tach anh viet tay raw thanh tap train va validation:

```powershell
python split_personal_data.py
```

### Chuyen anh ca nhan ve 28x28

Chuyen anh ca nhan thanh dataset 28x28 giong MNIST:

```powershell
python convert_personal_data_to_28.py
```

## Su Dung

### Train toan bo pipeline

Chay cac script theo thu tu nay:

```powershell
python train_mnist_base.py
python finetune_emnist.py
python split_personal_data.py
python convert_personal_data_to_28.py
python finetune_personal_data.py
```

### App demo local (khuyen nghi)

```powershell
python predict_digit_app.py
```

Luong demo nhanh:

1. Kich hoat virtual environment va bao dam `models/stage_03_final.keras` ton tai.
2. Chay `python predict_digit_app.py`.
3. Bam `Choose Image`, chon file anh dau vao va doi app hien thi anh goc, preview 28x28 da xu ly, chu so du doan, confidence va cac probability cao nhat.

Dinh dang anh ho tro: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`.

### Du doan mot anh don le

Neu muon dung script thay vi GUI app, chay:

```powershell
python predict_digit_image.py
```

Dat `IMAGE_PATH` trong `predict_digit_image.py` truoc khi chay, hoac truyen duong dan anh qua CLI:

```powershell
python predict_digit_image.py path/to/image.png
```

### Phan tich va truc quan hoa

```powershell
python analyze_confusion_matrix.py
python export_misclassified_csv.py
python export_confusion_pair_images.py
python preview_mnist_samples.py
```

## Artifact Model

- `models/stage_01_mnist_base.keras`: model nen da train tren MNIST
- `models/stage_02_emnist_finetuned.keras`: model da thich nghi voi EMNIST digits
- `models/stage_03_final.keras`: model cuoi dung cho prediction va evaluation

Repository chu y chi giu ba output stage chuan o tren. Cac file `.keras` thu nghiem them trong `models/` duoc `models/.gitignore` bo qua de tranh nham lan voi workflow chinh.

## Cau Hinh Quan Trong

Nen xem cac file nay truoc khi chay thi nghiem:

- `predict_digit_app.py`: `MODEL_PATH`, `PREPROCESS_THRESHOLD`, `TTA_SAMPLES`
- `predict_digit_image.py`: `MODEL_PATH`, `IMAGE_PATH`, `PREPROCESS_THRESHOLD`, `TTA_SAMPLES`
- `convert_personal_data_to_28.py`: `SOURCE_DIR`, `DESTINATION_DIR`, `THRESHOLD`
- `split_personal_data.py`: `TRAIN_DIR`, `VAL_DIR`, `VAL_RATIO`, `SEED`
- `finetune_personal_data.py`: dataset paths, batch size, epochs, learning rate

## Mapping Script Lich Su

| Ten cu | File hien tai |
| --- | --- |
| `Day2.py` | `train_mnist_base.py` |
| `finetune_emnist_digits.py` | `finetune_emnist.py` |
| `split_train_val.py` | `split_personal_data.py` |
| `convert_to_28.py` | `convert_personal_data_to_28.py` |
| `finetune_mystyle.py` | `finetune_personal_data.py` |
| `trainAIDigit.py` | `predict_digit_image.py` |
| `confusion_matrix.py` | `analyze_confusion_matrix.py` |
| `save_csv.py` | `export_misclassified_csv.py` |
| `xuatAnhCapSai.py` | `export_confusion_pair_images.py` |
| `xemAnhDaTrain.py` | `preview_mnist_samples.py` |

## Ghi Chu

- `my_digits_28` la dataset ca nhan 28x28 chuan duoc dung trong workflow hien tai.
- `my_digits_28_new` da duoc merge vao `my_digits_28`.
- `models/.gitignore` bo qua cac model export khong chuan de chi ba artifact stage chinh duoc track.
- Cac thu muc tam thoi nhu `.venv/`, `.idea/` va `__pycache__/` khong nen commit.

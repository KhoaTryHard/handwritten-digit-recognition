# Bao cao 3 muc cho repo nhan dien chu so viet tay

## 1. Mo hinh de xuat

### 1.1. Kien truc duoc dung o model cuoi

Artifact `models/stage_03_final.keras` hien dang luu mot CNN nho gon cho bai toan phan lop 10 chu so. Neu dem theo chuoi layer xu ly chinh trong artifact, model cuoi gom 8 layer:

`Conv2D(32, kernel_size=3x3, activation=relu, padding=valid)` -> `MaxPooling2D(2x2)` -> `Conv2D(64, 3x3, relu, valid)` -> `MaxPooling2D(2x2)` -> `Conv2D(64, 3x3, relu, valid)` -> `Flatten` -> `Dense(64, relu)` -> `Dense(10, softmax)`

Bang tom tat:

| Artifact | Kien truc | Regularization | Tong tham so* |
| --- | --- | --- | ---: |
| `stage_03_final.keras` | 3 `Conv2D` + 2 `MaxPooling2D` + `Flatten` + 2 `Dense` | Khong co `BatchNormalization`, khong co `Dropout` | 93,322 |
| `stage_02_emnist_finetuned.keras` | Giong `stage_03_final.keras` | Khong co `BatchNormalization`, khong co `Dropout` | 93,322 |
| `stage_01_mnist_base.keras` | `InputLayer` + 5 `Conv2D` + 5 `BatchNormalization` + 5 `ReLU` + 2 `MaxPooling2D` + 3 `Dropout` + `GlobalAveragePooling2D` + 2 `Dense` | `Dropout(0.25, 0.30, 0.35)` va `BatchNormalization` | 134,954 |

\* Bao cao nay dung `model.count_params()` de thong nhat cach dem. `model.summary()` cua file `.keras` con hien thi them optimizer state, nen tong hien thi trong summary lon hon con so bang nay.

### 1.2. Bien the qua 3 stage

**Stage 1 - baseline MNIST (`stage_01_mnist_base.keras`)**

- Block 1: `Conv2D(32, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- Block 2: `Conv2D(32, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- `MaxPooling2D(2x2)` -> `Dropout(0.25)`
- Block 3: `Conv2D(64, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- Block 4: `Conv2D(64, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- `MaxPooling2D(2x2)` -> `Dropout(0.30)`
- Block 5: `Conv2D(96, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- `GlobalAveragePooling2D` -> `Dense(128, relu)` -> `Dropout(0.35)` -> `Dense(10, softmax)`

**Stage 2 - fine-tune tren EMNIST (`stage_02_emnist_finetuned.keras`)**

- Kien truc artifact hien co giong model cuoi:
  `Conv2D(32, 3x3, relu)` -> `MaxPooling2D` -> `Conv2D(64, 3x3, relu)` -> `MaxPooling2D` -> `Conv2D(64, 3x3, relu)` -> `Flatten` -> `Dense(64, relu)` -> `Dense(10, softmax)`

**Stage 3 - model cuoi (`stage_03_final.keras`)**

- Kien truc artifact giong stage 2, thong so 93,322.
- Dau vao anh xam `28x28x1`, dau ra `Dense(10, softmax)` cho 10 lop `0-9`.

### 1.3. Ly do chon kien truc va diem cai tien

- Baseline stage 1 duoc xay dung theo huong CNN co `BatchNormalization` va `Dropout` de hoc dac trung tren MNIST, giup mo hinh on dinh hon va giam overfitting tren bo du lieu chuan.
- Stage 2 va stage 3 khong cho thay thay doi kien truc trong artifact hien luu; cai tien chinh den tu fine-tune theo mien du lieu, lan luot tren `EMNIST digits` va bo du lieu ca nhan `my_digits_28`.
- Nghia la, theo artifact thuc te trong repo, su cai thien o model cuoi khong duoc chung minh la do tang do phuc tap kien truc, ma chu yeu do thich nghi trong so voi du lieu gan bai toan muc tieu hon.
- Can luu y co do lech giua artifact stage 2/3 va builder dang dinh nghia trong [`digit_pipeline/models/cnn.py`](../digit_pipeline/models/cnn.py): builder hien tai mo ta mang lon hon, gan voi `stage_01_mnist_base.keras`, nhung `stage_02_emnist_finetuned.keras` va `stage_03_final.keras` hien dang la CNN 8 layer nho hon. Bao cao nay uu tien artifact thuc te da luu trong `models/`.

## 2. Cau hinh huan luyen

### 2.1. Dataset va tien xu ly

| Nguon du lieu | Cach nap | Tien xu ly co trong repo |
| --- | --- | --- |
| MNIST | `tf.keras.datasets.mnist.load_data()` | Chuyen sang `float32`, chuan hoa ve `[0, 1]`, neu anh co rank 2 thi bo sung channel de thanh `28x28x1` |
| EMNIST digits | `tfds.load("emnist/digits", split=["train", "test"], as_supervised=True)` | Chuan hoa ve `[0, 1]`, bo sung channel neu can, augmentation manh hon o stage 2 |
| `my_digits_28/train`, `my_digits_28/val` | `tf.keras.utils.image_dataset_from_directory(..., color_mode="grayscale", image_size=(28, 28), label_mode="int")` | Anh xam `28x28`, chuan hoa ve `[0, 1]`, giu nhan dang so nguyen (`label_mode="int"`) |

Ghi chu ve nhan:

- Repo hien tai **khong one-hot hoa nhan** trong pipeline huan luyen.
- Nhan duoc giu dang so nguyen va hoc voi `SparseCategoricalCrossentropy`.

Ghi chu ve du lieu ca nhan:

- Script tach du lieu goc: `split_personal_data.py`
  - `train_dir = my_digits_new/train`
  - `val_dir = my_digits_new/val`
  - `val_ratio = 0.2`
  - `seed = 42`
- Script chuyen anh ve dang MNIST-like: `convert_personal_data_to_28.py`
  - `SOURCE_DIR = my_digits_new/val`
  - `DESTINATION_DIR = my_digits_28/val`
  - `THRESHOLD = 0.22`
- Trong `digit_pipeline/preprocessing/images.py`, anh viet tay duoc:
  - dua ve grayscale,
  - dao cuc neu can de dua net chu so thanh vung sang tren nen toi,
  - tao mask bang nguong `threshold`,
  - dilate/erode,
  - giu lai thanh phan lien thong lon,
  - cat vung chu so,
  - resize vao khung `20x20`,
  - dat vao canvas `28x28`,
  - can giua theo tam khoi luong,
  - lam muot nhe bang `GaussianBlur(radius=0.4)`,
  - chuan hoa thanh tensor `1x28x28x1`.

Quy mo du lieu ca nhan dang co trong repo:

| Tap | So anh |
| --- | ---: |
| `my_digits_28/train` | 17,481 |
| `my_digits_28/val` | 4,369 |

### 2.2. Hyper-parameters theo tung stage

| Stage | Script | Dau vao / Dau ra | Dataset | Batch size | Epochs | Optimizer | Learning rate | Loss / Metric | Augmentation | Callback / Scheduler |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 1 | `train_mnist_base.py` | Tao `models/stage_01_mnist_base.keras` | MNIST train / MNIST test | 64 | 5 | Adam | 1e-3 | `SparseCategoricalCrossentropy`, `accuracy` | Khong co augmentation train rieng | Khong cai dat callback rieng |
| 2 | `finetune_emnist.py` | `stage_01_mnist_base.keras` -> `models/stage_02_emnist_finetuned.keras` | EMNIST digits train / test | 128 | 5 | Adam | 1e-4 | `SparseCategoricalCrossentropy`, `accuracy` | `RandomRotation(0.10)`, `RandomTranslation(0.15, 0.15)`, `RandomZoom(0.12)`, dao mau ngau nhien, speckle noise | `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` |
| 3 | `finetune_personal_data.py` | `stage_02_emnist_finetuned.keras` -> `models/stage_03_final.keras` | `my_digits_28/train`, `my_digits_28/val` | 64 | 40 | Adam | 1e-4 | `SparseCategoricalCrossentropy`, `accuracy` | `RandomRotation(0.08)`, `RandomTranslation(0.10, 0.10)`, `RandomZoom(0.08)` | `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` |

Chi tiet callback dung chung cho stage 2 va stage 3:

| Callback | Cau hinh |
| --- | --- |
| `EarlyStopping` | monitor `val_loss`, `patience=5`, `restore_best_weights=True` |
| `ReduceLROnPlateau` | monitor `val_loss`, `factor=0.5`, `patience=2`, `min_lr=1e-6` |
| `ModelCheckpoint` | monitor `val_loss`, `save_best_only=True` |

### 2.3. Moi truong thu vien va GPU

| Thanh phan | Gia tri trich tu repo |
| --- | --- |
| TensorFlow | `tensorflow==2.21.0` |
| TensorFlow Datasets | `tensorflow-datasets==4.9.9` |
| Keras | Khong pin package `keras` rieng trong `requirements.txt`; repo dung `tf.keras` |
| GPU / phan cung | `(chua co so lieu trong repo)` |
| Ghi chu runtime hien tai | Log danh gia read-only hien tai bao TensorFlow native Windows khong su dung GPU voi TensorFlow >= 2.11 |

### 2.4. Lenh huan luyen va thu muc lien quan

```powershell
python train_mnist_base.py
python finetune_emnist.py
python finetune_personal_data.py
```

Thu muc / artifact tuong ung:

- `models/stage_01_mnist_base.keras`
- `models/stage_02_emnist_finetuned.keras`
- `models/stage_03_final.keras`
- `my_digits_28/train`
- `my_digits_28/val`

## 3. Ket qua mo hinh cuoi

### 3.1. Metric chinh cua `stage_03_final.keras`

So lieu duoi day duoc lay tu lan danh gia read-only hien tai tren repo voi artifact `models/stage_03_final.keras`.

| Tap danh gia | Accuracy | Loss |
| --- | ---: | ---: |
| Train (`my_digits_28/train`) | 0.9603569508 | 0.1333358139 |
| Validation (`my_digits_28/val`) | 0.9567406774 | 0.1479387879 |
| Test | `(chua co so lieu trong repo)` | `(chua co so lieu trong repo)` |

### 3.2. Confusion matrix, per-class accuracy va artifact phan tich loi

Confusion matrix hien tai tren `my_digits_28/val` (`rows=true`, `cols=pred`):

```text
[[440   0   1   3   1   3   0   0   3   0]
 [  1 433   1   1  11   1   0   4   2   0]
 [  0   0 443   8   0   0   0   1   3   0]
 [  1   0   1 433   0   3   0   1   6   1]
 [  2   1   2   0 424   0   3   2   5   3]
 [  1   0   1  11   0 411   0   0   6   0]
 [  4   0   3   2   4   5 409   0   4   0]
 [  1   1   1   1   1   1   0 420   2   1]
 [  2   0   5  12   2   9   0   0 388   4]
 [  4   0   1  10   3   4   0   4   4 379]]
```

Tong so mau du doan sai: **189 / 4,369**.

Per-class accuracy:

| Lop | Accuracy | Dung / Tong |
| --- | ---: | ---: |
| 0 | 0.9756 | 440 / 451 |
| 1 | 0.9537 | 433 / 454 |
| 2 | 0.9736 | 443 / 455 |
| 3 | 0.9709 | 433 / 446 |
| 4 | 0.9593 | 424 / 442 |
| 5 | 0.9558 | 411 / 430 |
| 6 | 0.9490 | 409 / 431 |
| 7 | 0.9790 | 420 / 429 |
| 8 | 0.9194 | 388 / 422 |
| 9 | 0.9267 | 379 / 409 |

Top confusion pairs:

| Hang | Nham lan | So lan |
| --- | --- | ---: |
| 1 | `8 -> 3` | 12 |
| 2 | `1 -> 4` | 11 |
| 3 | `5 -> 3` | 11 |
| 4 | `9 -> 3` | 10 |
| 5 | `8 -> 5` | 9 |
| 6 | `2 -> 3` | 8 |
| 7 | `3 -> 8` | 6 |
| 8 | `5 -> 8` | 6 |
| 9 | `4 -> 8` | 5 |
| 10 | `6 -> 5` | 5 |

Precision / recall theo lop: `(chua co so lieu trong repo)`. Script [`analyze_confusion_matrix.py`](../analyze_confusion_matrix.py) hien chi in confusion matrix, per-class accuracy va top confusion.

Artifact bo tro de xem loi truc quan:

- CSV cac mau sai: [misclassified_validation.csv](misclassified_validation.csv)
- Anh loi mau: [9_to_3_idx3914.png](confusion_pairs/9_to_3_idx3914.png), [9_to_3_idx4355.png](confusion_pairs/9_to_3_idx4355.png)

Luu y ve do lech artifact:

- `reports/misclassified_validation.csv` hien co 267 dong sai, trong khi danh gia hien tai cua `stage_03_final.keras` tren `my_digits_28/val` cho 189 mau sai.
- Thu muc `reports/confusion_pairs/` hien dang co 36 anh `9 -> 3`, trong khi confusion matrix hien tai cho thay cap `9 -> 3` xuat hien 10 lan.
- Vi vay, cac tep trong `reports/` o tren chi nen dung nhu artifact lich su / minh hoa, **khong** dung lam so lieu chinh cho model hien tai.

### 3.3. Nhan xet

- Hai lop dang yeu nhat tren tap validation hien tai la `8` (0.9194) va `9` (0.9267).
- Cum nham lan noi bat xoay quanh `8/3/5` va `9/3`, cho thay cac mau co net cong, vong kin hoac duoi keo dai co the lam hinh dang giua cac lop nay tro nen gan nhau sau khi dua ve `28x28`.
- Cap `1 -> 4` xuat hien 11 lan, goi y mot so mau `1` co net nghieng hoac net ngang/phu tro khi viet tay khien mo hinh nghieng ve `4`.
- Huong cai thien tiem nang tu chinh so lieu hien co:
  - bo sung them mau ca nhan cho cac lop `8`, `9`, `3`, `5`,
  - uu tien cac bien the viet tay kho phan biet,
  - kiem tra lai threshold / preprocessing cho nhom mau bi mat net sau khi chuyen ve `28x28`,
  - co the xuat them precision/recall theo lop neu muon bao cao chi tiet hon o phan danh gia.

## Tai lieu tham khao

- [`models/stage_01_mnist_base.keras`](../models/stage_01_mnist_base.keras)
- [`models/stage_02_emnist_finetuned.keras`](../models/stage_02_emnist_finetuned.keras)
- [`models/stage_03_final.keras`](../models/stage_03_final.keras)
- [`digit_pipeline/models/cnn.py`](../digit_pipeline/models/cnn.py)
- [`train_mnist_base.py`](../train_mnist_base.py)
- [`finetune_emnist.py`](../finetune_emnist.py)
- [`finetune_personal_data.py`](../finetune_personal_data.py)
- [`digit_pipeline/training/configs.py`](../digit_pipeline/training/configs.py)
- [`digit_pipeline/training/runners.py`](../digit_pipeline/training/runners.py)
- [`digit_pipeline/data_loading/datasets.py`](../digit_pipeline/data_loading/datasets.py)
- [`digit_pipeline/preprocessing/augmentations.py`](../digit_pipeline/preprocessing/augmentations.py)
- [`digit_pipeline/preprocessing/images.py`](../digit_pipeline/preprocessing/images.py)
- [`split_personal_data.py`](../split_personal_data.py)
- [`convert_personal_data_to_28.py`](../convert_personal_data_to_28.py)
- [`analyze_confusion_matrix.py`](../analyze_confusion_matrix.py)
- [`requirements.txt`](../requirements.txt)
- [`misclassified_validation.csv`](misclassified_validation.csv)
- [`confusion_pairs/9_to_3_idx3914.png`](confusion_pairs/9_to_3_idx3914.png)
